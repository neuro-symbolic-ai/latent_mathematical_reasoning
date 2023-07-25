import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from latent_reasoning.data_model import DataModelMultiStep
from latent_reasoning.TranslationalReasoningTransformer import TransLatentReasoning
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, load_model_path = None):
        self.model_name = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        #LOAD DATA
        self.data_model = DataModelMultiStep(neg, self.tokenize_function)
        self.eval_dict = self.data_model.eval_dict
        self.operations_voc = self.data_model.operations_voc
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        #self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create model
        self.model = TransLatentReasoning(self.model_name, self.num_ops, self.device)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path))

    def tokenize_function(self, examples):
        for step in examples["steps"]:
            if examples["steps"][step] == None:
                examples["steps"][step] = []
                continue
            for instance in examples["steps"][step]:
                instance["equation1"] = self.tokenizer(instance["equation1"], padding="max_length", truncation=True, max_length = self.max_length)
                instance["equation2"] = self.tokenizer(instance["equation2"], padding="max_length", truncation=True, max_length = self.max_length)
                instance["target"] = self.tokenizer(instance["target"], padding="max_length", truncation=True, max_length = self.max_length)
        return examples

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        #majority_class_preds = [1 for pred in logits]
        #majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        #print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=logits, references=labels)
        return score

    def evaluation(self, batch_size = 4):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        #build dataloaders
        eval_loaders = {}
        for step in self.eval_dict:
            eval_loaders[step] = DataLoader(self.eval_dict[step].with_format("torch"), batch_size=batch_size, shuffle=False)
            #if not step in self.eval_best_scores:
            #    self.eval_best_scores[step] = {"accuracy": 0.0, "f1": 0.0}
        #START EVALUATION
        self.model.eval()
        print("EVALUATION")
        inference_state = {}
        for step in eval_loaders:
            scores_pos = {}
            scores_neg = {}
            logits_metric = {}
            label_metric = {}
            batch_index = 0
            max_batch = 3000
            for eval_batch in tqdm(eval_loaders[step], desc = str(step)):
                inference_state = {}
                for inference_step in eval_batch["steps"]:
                    if not inference_step in scores_pos:
                        scores_pos[inference_step] = []
                    if not inference_step in scores_neg:
                        scores_neg[inference_step] = []
                    if not inference_step in logits_metric:
                        logits_metric[inference_step] = []
                    if not inference_step in label_metric:
                        label_metric[inference_step] = []
                    item_index = 0
                    equation1 = []
                    equation2 = []
                    target = []
                    labels = []
                    operation = []
                    for item in eval_batch["steps"][inference_step]:
                        equation1 = item["equation1"]
                        equation2 = item["equation2"]
                        target = item["target"]
                        labels = item["label"]
                        operation = item['operation']
                        if inference_step == "0":
                            outputs = self.model.inference_step(None, equation1, equation2, target, operation, labels)
                        else:
                            outputs = self.model.inference_step(inference_state[item_index], None, equation2, target, operation, labels)
                        inference_state[item_index] = outputs[1]
                        item_index += 1
                        for score in torch.sign(outputs[0]):
                            if score >= 0.0: #== torch.max(outputs[0]):
                                logits_metric[inference_step].append(1)
                            else:
                                logits_metric[inference_step].append(0)
                        label_index = 0
                        for label in labels:
                            if label == 1.0:
                                scores_pos[inference_step].append(outputs[0].detach().cpu().numpy()[label_index])
                                label_metric[inference_step].append(1)
                            else:
                                scores_neg[inference_step].append(outputs[0].detach().cpu().numpy()[label_index])
                                label_metric[inference_step].append(0)
                            label_index += 1
                if batch_index > max_batch:
                    break
                batch_index += 1
            eval_metrics = {}
            positive_avg = {}
            negative_avg = {}
            difference_avg = {}
            for inference_step in logits_metric:
                eval_metrics[inference_step] = self.compute_metrics([logits_metric[inference_step], label_metric[inference_step]])
                positive_avg[inference_step] = np.mean(scores_pos[inference_step])
                negative_avg[inference_step] = np.mean(scores_neg[inference_step])
                difference_avg[inference_step] = positive_avg[inference_step] - negative_avg[inference_step]
            #print results
            print("=============="+str(step)+"==============")
            print("positive avg sim:", positive_avg)
            print("negative avg sim:", negative_avg)
            print("difference:", difference_avg)
            print("evaluation scores:", eval_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=8, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=int, default=32, nargs="?",
                    help="Num epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--neg", type=int, default=1, nargs="?",
                    help="Max number of negative examples")

    args = parser.parse_args()
    dataset = args.dataset
    torch.backends.cudnn.deterministic = True 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(
            learning_rate = args.lr, 
            batch_size = args.batch_size, 
            neg = args.neg,
            max_length = args.max_length,
            epochs = args.epochs, 
            model = args.model,
            load_model_path = "models/distilroberta-base_best_dev_set_6.pt",
            )
    experiment.evaluation()

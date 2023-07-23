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

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, load_model_path = None, do_train = True, do_test = False):
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
        self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create model
        self.model = TransLatentReasoning(self.model_name, self.num_ops, self.device)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path))

    def tokenize_function(self, examples):
        examples["equation1"] = self.tokenizer(examples["equation1"], padding="max_length", truncation=True, max_length = self.max_length)
        examples["equation2"] = self.tokenizer(examples["equation2"], padding="max_length", truncation=True, max_length = self.max_length)
        examples["target"] = self.tokenizer(examples["target"], padding="max_length", truncation=True, max_length = self.max_length)
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
            eval_loaders[step] = DataLoader(self.eval_dict[step].with_format("torch"), batch_size=batch_size, shuffle=True)
            if not step in self.eval_best_scores:
                self.eval_best_scores[step] = {"accuracy": 0.0, "f1": 0.0}
        #START EVALUATION
        self.model.eval()
        print("EVALUATION")
        for step in eval_loaders:
            eval_steps = 0
            max_steps = 500
            scores_pos = []
            scores_neg = []
            logits_metric = []
            label_metric = []
            for eval_batch in tqdm(eval_loaders[step], desc = step):
                eval_steps += 1
                equation1 = eval_batch["equation1"]
                equation2 = eval_batch["equation2"]
                target = eval_batch["target"]
                labels = eval_batch["label"]
                operation = eval_batch['operation']
                if step == 0:
                    outputs = self.model.inference_step(equation1, equation2, target, operation, labels)
                else:
                    outputs = self.model.inference_step(None, equation2, target, operation, labels)
                batch_index = 0
                for score in outputs[0]:
                    if score > 0.0:
                        logits_metric.append(1)
                    else:
                        logits_metric.append(0)
                    batch_index += 1
                batch_index = 0
                for label in labels:
                    if label == 1.0:
                        scores_pos.append(outputs[0].detach().cpu().numpy())
                        label_metric.append(1)
                    else:
                        scores_neg.append(outputs[0].detach().cpu().numpy())
                        label_metric.append(0)
                    batch_index += 1
                #if eval_steps > max_steps:
                #    break
            eval_metrics = self.compute_metrics([logits_metric, label_metric])
            if eval_metrics["f1"] > self.eval_best_scores[step]["f1"]:
                #new best score
                self.eval_best_scores[step] = eval_metrics
            #print results
            print("=============="+step+"==============")
            print("positive avg sim:", np.mean(scores_pos))
            print("negative avg sim:", np.mean(scores_neg))
            print("difference:", np.mean(scores_pos) - np.mean(scores_neg))
            print("current scores:", eval_metrics)
            print("best scores:", self.eval_best_scores[step])

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
            load_model_path = "models/distilbert-base-uncased_best_dev_set_6.pt",
            do_train = False,
            do_test = True
            )
    #experiment.train_and_eval()
    experiment.evaluation(save_best_model = False)

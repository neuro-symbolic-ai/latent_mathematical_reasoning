import json
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from latent_reasoning.data_model import DataModelMultiStep
from latent_reasoning.sequential_utils import *
from latent_reasoning.BaselinesSequential import LatentReasoningSeq
from latent_reasoning.TranslationalReasoningSequential import TransLatentReasoningSeq
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, trans = True, one_hot = False, load_model_path = None):
        self.model_type = model
        self.epochs = epochs
        self.trans = trans
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        #load pretrained vocabulary
        self.operations_voc = pickle.load(open(load_model_path + "/operations", "rb"))
        self.vocabulary =  pickle.load(open(load_model_path + "/vocabulary", "rb"))
        #LOAD DATA
        self.corpus = Corpus(self.max_length, build_voc = False)
        self.corpus.dictionary.word2idx = self.vocabulary
        self.tokenizer = self.corpus.tokenizer
        self.data_model = DataModelMultiStep(neg, self.operations_voc, self.tokenize_function)
        self.eval_dict = self.data_model.eval_dict
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        #self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create model
        if self.trans:
            #translational model
            self.model = TransLatentReasoningSeq(len(self.corpus.dictionary.word2idx.keys()), self.num_ops, self.device, model_type = self.model_type)
        else:
            #baseline
            self.model = LatentReasoningSeq(len(self.corpus.dictionary.word2idx.keys()), self.num_ops, self.device, model_type = self.model_type, one_hot = one_hot)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path + "/state_dict.pt"))

    def tokenize_function(self, examples):
        for step in examples["steps"]:
            if examples["steps"][step] == None:
                examples["steps"][step] = []
                continue
            for instance in examples["steps"][step]:
                instance["equation1"], instance["equation2"], instance["target"] = self.tokenizer([instance["equation1"], instance["equation2"], instance["target"]])
        return examples

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        #majority_class_preds = [1 for pred in logits]
        #majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        #print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=logits, references=labels)
        return score

    def evaluation(self, batch_size = 1):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        # TODO change negative examples in datamodel and frame task as retrieval
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
            max_batch = 100
            for eval_batch in tqdm(eval_loaders[step], desc = str(step)):
                inference_state = {}
                for inference_step in eval_batch["steps"]:
                    #print(inference_step, len(inference_state.keys()))
                    if not inference_step in scores_pos:
                        scores_pos[inference_step] = []
                    if not inference_step in scores_neg:
                        scores_neg[inference_step] = []
                    if not inference_step in logits_metric:
                        logits_metric[inference_step] = []
                    if not inference_step in label_metric:
                        label_metric[inference_step] = []
                    item_index = 0
                    temp_score = []
                    for item in eval_batch["steps"][inference_step]:
                        equation1 = item["equation1"]
                        equation2 = item["equation2"]
                        target = item["target"]
                        labels = item["label"]
                        operation = item['operation']
                        #print(item)
                        if inference_step == "0":
                            outputs = self.model.inference_step(None, equation1, None, target, operation, labels)
                        else:
                            outputs = self.model.inference_step(inference_state[item_index], None, None, target, operation, labels)
                        inference_state[item_index] = outputs[1]
                        #print(outputs[0])
                        for score in outputs[0]:
                            if len(temp_score) == 0:
                                temp_score.append(score)
                            else:
                                if score >= temp_score[0]:
                                    logits_metric[inference_step] += [0, 1]
                                else:
                                    logits_metric[inference_step] += [1, 0]
                                temp_score = []
                            #if score == torch.max(outputs[0]):
                            #    logits_metric[inference_step].append(1)
                            #else:
                            #    logits_metric[inference_step].append(0)
                        label_index = 0
                        for label in labels:
                            if label == 1.0:
                                scores_pos[inference_step].append(outputs[0].detach().cpu().numpy()[label_index])
                                label_metric[inference_step].append(1)
                            else:
                                scores_neg[inference_step].append(outputs[0].detach().cpu().numpy()[label_index])
                                label_metric[inference_step].append(0)
                            label_index += 1
                        item_index += 1
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
            #print("positive avg sim:", positive_avg)
            #print("negative avg sim:", negative_avg)
            #print("difference:", difference_avg)
            print("evaluation scores:", eval_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="rnn", nargs="?",
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
            trans = False,
            one_hot = False,
            load_model_path = "models/rnn_False_False_6_512",
            )
    experiment.evaluation()

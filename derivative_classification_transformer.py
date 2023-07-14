import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from latent_reasoning.TranslationalReasoningTransformer import TransLatentReasoning
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, load_model_path = None):
        self.model_name = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        #PROCESS DATA
        #training data
        self.train_dataset = self.process_dataset(neg = neg)
        self.tokenized_train_datasets = self.train_dataset.map(self.tokenize_function, batched=False)
        #test differentiation
        #self.test_dataset_diff = self.process_dataset(dataset_path = ["data/EVAL_differentiation.json"], neg = neg, training = False, test_size = 1.0)
        #self.tokenized_test_dataset_diff = self.test_dataset_diff.map(self.tokenize_function, batched=False)
        #self.contrast_dataset_diff = self.process_dataset(dataset_path = ["data/EVAL_differentiation_VAR_SWAP.json", "data/EVAL_easy_differentiation.json"], neg = neg,  training = False, test_size = 1.0)
        #self.tokenized_contrast_dataset_diff = self.contrast_dataset_diff.map(self.tokenize_function, batched=False)
        #test integration
        #self.test_dataset_int = self.process_dataset(dataset_path = ["data/EVAL_integration.json"], neg = neg, training = False, test_size = 1.0)
        #self.tokenized_test_dataset_int = self.test_dataset_int.map(self.tokenize_function, batched=False)
        #self.contrast_dataset_int = self.process_dataset(dataset_path = ["data/EVAL_integration_VAR_SWAP.json", "data/EVAL_easy_integration.json"], neg = neg,  training = False, test_size = 1.0)
        #self.tokenized_contrast_dataset_int = self.contrast_dataset_int.map(self.tokenize_function, batched=False)
        #test addition
        #self.test_dataset_add = self.process_dataset(dataset_path = ["data/EVAL_addition.json"], neg = neg, training = False, test_size = 1.0)
        #self.tokenized_test_dataset_add = self.test_dataset_add.map(self.tokenize_function, batched=False)
        #test subtraction
        #self.test_dataset_sub = self.process_dataset(dataset_path = ["data/EVAL_subtraction.json"], neg = neg, training = False, test_size = 1.0)
        #self.tokenized_test_dataset_sub = self.test_dataset_sub.map(self.tokenize_function, batched=False)
        #test multiplication
        #self.test_dataset_mul = self.process_dataset(dataset_path = ["data/EVAL_multiplication.json"], neg = neg, training = False, test_size = 1.0)
        #self.tokenized_test_dataset_mul = self.test_dataset_mul.map(self.tokenize_function, batched=False)        
        #test division
        #self.test_dataset_div = self.process_dataset(dataset_path = ["data/EVAL_division.json"], neg = neg, training = False, test_size = 1.0)
        #self.tokenized_test_dataset_div = self.test_dataset_div.map(self.tokenize_function, batched=False)
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.eval_dict = {
            "dev_set": self.tokenized_train_datasets["test"],
            #"test_diff" : self.tokenized_test_dataset_diff, 
            #"test_int" : self.tokenized_test_dataset_int,
            #"test_add" : self.tokenized_test_dataset_add,
            #"test_sub" : self.tokenized_test_dataset_sub,
            #"test_mul" : self.tokenized_test_dataset_mul,
            #"test_div" : self.tokenized_test_dataset_div,
            #"contrast_diff" : self.tokenized_contrast_dataset_diff, 
            #"contrast_int" : self.tokenized_contrast_dataset_int
            }
        self.num_ops = len(self.operations_voc.keys())
        if load_model_path is not None:
            #load pretrained model
            self.model = TransLatentReasoning(self.model_name, self.num_ops, self.device)
            self.model.load_state_dict(torch.load(load_model_path))
        else:
            #create a new model
            self.model = TransLatentReasoning(self.model_name, self.num_ops, self.device)


    def process_dataset(self, dataset_path = ["data/differentiation.json", "data/integration.json", "data/addition.json", "data/subtraction.json", "data/multiplication.json", "data/division.json"], neg = 1,  training = True, test_size = 0.2):
        #load operation vocabulary
        if training:
            self.operations_voc = {}
            op_id = 0
            for path in dataset_path:
                self.operations_voc[op_id] = path.split("/")[-1].replace(".json", "")
                op_id += 1
        #convert dataset into json for dataset loader
        formatted_examples = []
        for path in dataset_path:
            #find operation id
            for entry in self.operations_voc:
                if self.operations_voc[entry] in path:
                    op_id = entry
                    break
            d_file = open(path, 'r')
            d_json = json.load(d_file)
            # create an entry for each positive example
            for example in tqdm(d_json, desc="Loading Dataset"):
                #LATEX
                formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": example["positive"], "operation": op_id, "label": 1.0})
                #SIMPY
                #formatted_examples.append({"equation1": example["srepr_premise_expression"], "equation2": example["srepr_variable"], "target": example["srepr_positive"], "operation": op_id, "label": 1.0})
                #create an entry for each negative example
                count_neg = 0
                #LATEX
                for negative in example["negatives"]:
                #SIMPY
                #for negative in example["srepr_negatives"]:
                    if count_neg == neg:
                        break
                    #LATEX
                    formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": negative , "operation": op_id, 'label': -1.0})
                    #SIMPY
                    #formatted_examples.append({"equation1": example["srepr_premise_expression"], "equation2": example["srepr_variable"], "target": negative, "operation": op_id, "label": -1.0})
                    count_neg += 1
        print("Data examples", formatted_examples[:4])
        #split randomly between train, dev, and test set
        dataset = Dataset.from_list(formatted_examples)
        if test_size == 1.0:
            return dataset
        dataset_split = dataset.train_test_split(test_size = test_size)
        return dataset_split

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

    def train_and_eval(self):
        device = self.device
        self.model.to(device)
        
        train_loader = DataLoader(self.tokenized_train_datasets["train"].with_format("torch"), batch_size=8, shuffle=True)
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        print("Start training...")
        eval_steps_cycle = 2000
        steps = 0
        for epoch in tqdm(range(self.epochs), desc = "Training"):
            self.model.train()
            for batch in tqdm(train_loader):
                steps += 1
                optim.zero_grad()
                equation1 = batch["equation1"]
                equation2 = batch["equation2"]
                target = batch["target"]
                labels = batch['label']
                operation = batch['operation']
                outputs = self.model(equation1, equation2, target, operation, labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
                #evaluation
                if steps % eval_steps_cycle == 0:
                    self.evaluation()


    def evaluation(self, batch_size = 4, save_best_model = True):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        #build dataloaders
        eval_loaders = {}
        for dataset_name in self.eval_dict:
            eval_loaders[dataset_name] = DataLoader(self.eval_dict[dataset_name].with_format("torch"), batch_size=batch_size, shuffle=True)
            if not dataset_name in self.eval_best_scores:
                self.eval_best_scores[dataset_name] = {"accuracy": 0.0, "f1": 0.0}
        #START EVALUATION
        self.model.eval()
        print("EVALUATION")
        for loader in eval_loaders:
            eval_steps = 0
            max_steps = 500
            scores_pos = []
            scores_neg = []
            logits_metric = []
            label_metric = []
            for eval_batch in tqdm(eval_loaders[loader], desc = loader):
                eval_steps += 1
                equation1 = eval_batch["equation1"]
                equation2 = eval_batch["equation2"]
                target = eval_batch["target"]
                labels = eval_batch["label"]
                operation = eval_batch['operation']
                outputs = self.model(equation1, equation2, target, operation, labels)
                batch_index = 0
                for score in outputs[1]:
                    if score > 0.0:
                        logits_metric.append(1)
                    else:
                        logits_metric.append(0)
                    batch_index += 1
                batch_index = 0
                for label in labels:
                    if label == 1.0:
                        scores_pos.append(outputs[1].detach().cpu().numpy())
                        label_metric.append(1)
                    else:
                        scores_neg.append(outputs[1].detach().cpu().numpy())
                        label_metric.append(0)
                    batch_index += 1
                #if eval_steps > max_steps:
                #    break
            eval_metrics = self.compute_metrics([logits_metric, label_metric])
            if eval_metrics["f1"] > self.eval_best_scores[loader]["f1"]:
                #new best score
                self.eval_best_scores[loader] = eval_metrics
                #SAVE THE MODEL'S PARAMETERS
                if save_best_model:
                    PATH = "models/" + self.model_name + "_best_" + loader + "_" + str(self.num_ops) + ".pt"
                    torch.save(self.model.state_dict(), PATH)
            #print results
            print("=============="+loader+"==============")
            print("positive avg sim:", np.mean(scores_pos))
            print("negative avg sim:", np.mean(scores_neg))
            print("difference:", np.mean(scores_pos) - np.mean(scores_neg))
            print("current scores:", eval_metrics)
            print("best scores:", self.eval_best_scores[loader])

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
            model = args.model
            #load_model_path = "models/distilroberta-base_best_dev_set_6.pt"
            )
    experiment.train_and_eval()
    #experiment.evaluation(save_best_model = False)

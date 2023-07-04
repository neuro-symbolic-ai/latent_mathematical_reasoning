import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from latent_reasoning.LatentReasoning_graph import GraphLatentReasoning
from utils import match_parentheses, pad_collate
import re

class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, dataset_name, input_dim = 768, cons_list_sin = ['log', 'exp', 'cos', 'Integer', 'sin', 'Symbol'], cons_list_dou = ['Mul', 'Add', 'Pow']):
        #, cons_list = ['log', 'Mul', 'exp', 'Add', 'Symbol', 'Pow', 'cos', 'Integer', 'sin', '3', '2', '1', '0', '-1', '-2', '-3']):
        self.model_name = model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.input_dim = input_dim
        self.cons_list_sin = cons_list_sin # operation within equation
        self.cons_list_dou = cons_list_dou # operation within equation
        # self.tokenizer = AutoTokenizer.from_pretrained(model)
        #PROCESS DATA
        self.train_dataset = self.process_dataset(neg = neg)
        self.tokenized_train_datasets = self.train_dataset.map(self.tokenize_function, batched=False)
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")

        self.model = GraphLatentReasoning(model, 2, self.device)
        self.batch_size = batch_size




    def process_dataset(self, dataset_path = ["data/differentiation.json", "data/integration.json"], neg = 1,  test_size = 0.2):
        #convert dataset into json for dataset loader

        formatted_examples = []
        op_id = 0 ### operation between input and target
        
        for path in dataset_path:
            d_file = open(path, 'r')
            d_json = json.load(d_file)
            op_id = op_id + 1
            # create an entry for each positive example
            for example in tqdm(d_json, desc="Loading Dataset"):
                formatted_examples.append({"equation1": example['srepr_premise_expression'], "equation2": example['srepr_variable'], "target": example["srepr_positive"], "operation": op_id, "label": 1.0})
                # formatted_examples.append({"graph_equation1": self.construct_graph(example['srepr_premise_expression'], example['srepr_variable']), "graph_target": self.construct_graph(example["srepr_positive"], example['srepr_variable']), "operation": op_id, "label": 1.0})
                #create an entry for each negative example
                count_neg = 0
                for negative in example["srepr_negatives"]:
                    if count_neg == neg:
                        break
                    formatted_examples.append({"equation1": example['srepr_premise_expression'], "equation2": example['srepr_variable'], "target": negative , "operation": op_id, 'label': -1.0})
                    # formatted_examples.append({"graph_equation1": self.construct_graph(example['srepr_premise_expression'], example['srepr_variable']), "graph_target": self.construct_graph(negative, example['srepr_variable']), "operation": op_id, "label": -1.0})
                    count_neg += 1
        #     
        print("Data examples", formatted_examples[:4])
        # #split randomly between train, dev, and test set
        dataset = Dataset.from_list(formatted_examples)
        if test_size == 1.0:
            return dataset
        dataset_split = dataset.train_test_split(test_size = test_size)
        return dataset_split


    def construct_graph(self, examples, var):
        device = self.device

        node_list = []
        edge_index = [[], []]

        node_list = match_parentheses(examples)
        try:
            var_idx = node_list.index(match_parentheses(var)[-1])
        except:
            var_idx = -1

        idx = 0
        idx_flag = 0
        for symbol in node_list[: -1]:
            if symbol in self.cons_list_sin:
                edge_index[0].append(idx)
                edge_index[1].append(idx+1)

                idx = idx + 1
                
            elif symbol in self.cons_list_dou:
                edge_index[0].append(idx)
                edge_index[1].append(idx+1)

                idx_flag = idx
                idx = idx + 1
            else:
                edge_index[0].append(idx_flag)
                edge_index[1].append(idx+1)
        edge_index = torch.tensor(edge_index, dtype = torch.long).to(device)

        examples = {"node_list": node_list, "edge_index": edge_index, "var_idx": var_idx}
        # examples = {"var_idx": var_idx}
        return examples
        

    def tokenize_function(self, examples):
        examples["equation1"] = self.construct_graph(examples["equation1"],examples["equation2"])
        examples["target"] = self.construct_graph(examples["target"], examples["equation2"])
        if len(examples) != 5:
            print(examples)
        if examples["operation"] == 0:
            print(examples)
        return examples
    
    # def tokenize_function(self, examples):
    #     examples["equation1"] = self.tokenizer(examples["equation1"], padding="max_length", truncation=True, max_length = self.max_length)
    #     examples["equation2"] = self.tokenizer(examples["equation2"], padding="max_length", truncation=True, max_length = self.max_length)
    #     examples["target"] = self.tokenizer(examples["target"], padding="max_length", truncation=True, max_length = self.max_length)
    #     return examples






    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        #predictions = np.argmax(logits, axis=-1)
        majority_class_preds = [1 for pred in logits]
        majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=logits, references=labels)
        return score



    def train_and_eval(self):
        device = self.device
        self.model.to(device)
        
        self.model.train()
        
        train_loader = DataLoader(self.tokenized_train_datasets["train"].with_format("torch"), batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        dev_loader = DataLoader(self.tokenized_train_datasets["test"].with_format("torch"), batch_size=4, shuffle=True, collate_fn=pad_collate)
        # train_loader = DataLoader(self.train_dataset["train"].with_format("torch"), batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate)
        # dev_loader = DataLoader(self.train_dataset["test"].with_format("torch"), batch_size=4, shuffle=True, collate_fn=pad_collate)

        #test_loader = DataLoader(self.tokenized_test_datasets.with_format("torch"), batch_size=4, shuffle=True)
        #test_vr_loader = DataLoader(self.tokenized_test_vr_datasets.with_format("torch"), batch_size=4, shuffle=True)
        #test_easy_loader = DataLoader(self.tokenized_test_easy_datasets.with_format("torch"), batch_size=4, shuffle=True)

        eval_loaders = {
                "dev":dev_loader
                #"test":test_loader,
                #"test_vr":test_vr_loader,
                #"test_easy":test_easy_loader
                }
        
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        print("Start training...")

        eval_steps_cycle = 2000
        steps = 0
        for epoch in tqdm(range(32), desc = "Training"):
            print()

            for batch in tqdm(train_loader):
                steps += 1
                optim.zero_grad()
                loss = 0
                for idx in range(len(batch)):
                    equation1 = batch[idx]["equation1"]
                    equation2 = batch[idx]["equation2"] # useless now
                    target = batch[idx]["target"]
                    labels = batch[idx]['label'].to(device)
                    operation = batch[idx]['operation']
                    outputs = self.model(equation1, equation2, target, operation, labels)
                    loss = loss + outputs[0]
                loss.backward()
                optim.step()
                #evaluation
                if steps % eval_steps_cycle == 0:
                    self.model.eval()
                    print("EVALUATION")
                    num_instances = 100000
                    for loader in eval_loaders:
                        eval_steps = 0
                        scores_pos = []
                        scores_neg = []
                        logits_metric = []
                        logits_metric_diff = []
                        logits_metric_int = []
                        label_metric = []
                        label_metric_diff = []
                        label_metric_int = []
                        for eval_batch in tqdm(eval_loaders[loader]):
                            eval_steps += 1
                            # equation1 = eval_batch["equation1"]
                            # equation2 = eval_batch["equation2"] # useless now
                            # target = eval_batch["target"]
                            # labels = eval_batch["label"]
                            # operation = eval_batch['operation']
                            # outputs = self.model(equation1, equation2, target, operation, labels)
                            for idx in range(len(eval_batch)):
                                equation1 = eval_batch[idx]["equation1"]
                                equation2 = eval_batch[idx]["equation2"] # useless now
                                target = eval_batch[idx]["target"]
                                labels = eval_batch[idx]['label'].to(device)
                                operation = eval_batch[idx]['operation']
                                outputs = self.model(equation1, equation2, target, operation, labels)
                                batch_index = 0
                                for score in outputs[1]:
                                    if score > 0.0:
                                        logits_metric.append(1)
                                        if operation - 1 == 0:
                                            logits_metric_diff.append(1)
                                        else:
                                            logits_metric_int.append(1)
                                    else:
                                        logits_metric.append(0)
                                        if operation - 1 == 0:
                                            logits_metric_diff.append(0)
                                        else:
                                            logits_metric_int.append(0)
                                    batch_index += 1
                                batch_index = 0
                                
                                if labels == 1.0:
                                    scores_pos.append(outputs[1].detach().cpu().numpy())
                                    label_metric.append(1)
                                    if operation - 1 == 0:
                                        label_metric_diff.append(1)
                                    else:
                                        label_metric_int.append(1)
                                else:
                                    scores_neg.append(outputs[1].detach().cpu().numpy())
                                    label_metric.append(0)
                                    if operation - 1 == 0:
                                        label_metric_diff.append(0)
                                    else:
                                        label_metric_int.append(0)
                                batch_index += 1
                            if eval_steps > num_instances:
                                break   
                        print("=============="+loader+"==============")
                        print("positive:", np.mean(scores_pos))
                        print("negative:", np.mean(scores_neg))
                        print("difference:", np.mean(scores_pos) - np.mean(scores_neg))
                        print("tot:", self.compute_metrics([logits_metric, label_metric]))
                        print("differentiation:", self.compute_metrics([logits_metric_diff, label_metric_diff]))
                        print("integration:", self.compute_metrics([logits_metric_int, label_metric_int]))
                    self.model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="gat", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=8, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=float, default=12.0, nargs="?",
                    help="Num epochs.")
    parser.add_argument("--lr", type=float, default=3e-5, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--neg", type=int, default=1, nargs="?",
                    help="Max number of negative examples")

    args = parser.parse_args()
    dataset = args.dataset
    #data_path = "data/"+dataset
    torch.backends.cudnn.deterministic = True 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    experiment = Experiment(
            learning_rate = args.lr, 
            # batch_size = args.batch_size, 
            batch_size = 16, 
            neg = args.neg,
            max_length = args.max_length,
            epochs = args.epochs, 
            model = args.model, 
            dataset_name = dataset
            )
    experiment.train_and_eval()



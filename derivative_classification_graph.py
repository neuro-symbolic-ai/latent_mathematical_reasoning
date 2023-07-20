import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from torch.utils.data import DataLoader
from torch.optim import AdamW
from latent_reasoning.TranslationalReasoningGraph import GraphLatentReasoning_GAT, GraphLatentReasoning_GCN, GraphLatentReasoning_GraphSAGE, GraphLatentReasoning_TransformerConv
from utils import match_parentheses, pad_collate
import re

class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, load_model_path = None,  do_train = True, do_test = False, input_dim = 768, cons_list_sin = ['log', 'exp', 'cos', 'Integer', 'sin', 'Symbol'], cons_list_dou = ['Mul', 'Add', 'Pow']):
        #, cons_list = ['log', 'Mul', 'exp', 'Add', 'Symbol', 'Pow', 'cos', 'Integer', 'sin', '3', '2', '1', '0', '-1', '-2', '-3']):
        self.model_name = model
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.cons_list_sin = cons_list_sin # operation within equation
        self.cons_list_dou = cons_list_dou # operation within equation
        #LOAD DATA
        self.data_model = DatasetUtils(do_train, do_test, self.tokenize_function, srepr = True)
        self.train_dataset = self.data_model.train_dataset["train"]
        self.eval_dict = self.data_model.eval_dict
        self.operations_voc = self.data_model.operations_voc
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create a new model
        if self.model_name == 'gat':
            self.model = GraphLatentReasoning_GAT(self.model_name, self.num_ops, self.device)
        elif self.model_name == 'gcn':
            self.model = GraphLatentReasoning_GCN(self.model_name, self.num_ops, self.device)
        elif self.model_name == 'graphsage':
            self.model = GraphLatentReasoning_GraphSAGE(self.model_name, self.num_ops, self.device)
        elif self.model_name == 'graphtrans':
            self.model = GraphLatentReasoning_TransformerConv(self.model_name, self.num_ops, self.device)
        else:
            print("Wrong Model")
            exit(0)
        #load pretrained model
        if load_model_path is not None:
            self.model.load_state_dict(torch.load(load_model_path))        


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
        #if len(examples) != 5:
        #    print(examples)
        #if examples["operation"] == 0:
        #    print(examples)
        return examples


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
        
        train_loader = DataLoader(self.train_dataset.with_format("torch"), batch_size=self.batch_size,  shuffle=True, collate_fn=pad_collate)
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        print("Start training...")
        eval_steps_cycle = 2000
        steps = 0
        for epoch in tqdm(range(self.epochs), desc = "Training"):
            self.model.train()
            for batch in tqdm(train_loader):
                steps += 1
                optim.zero_grad()
                loss = 0
                for idx in range(len(batch)):
                    equation1 = batch[idx]["equation1"]
                    equation2 = batch[idx]["equation2"]
                    target = batch[idx]["target"]
                    labels = batch[idx]['label']
                    operation = batch[idx]['operation']
                    outputs = self.model(equation1, equation2, target, operation, labels)
                    loss = loss + outputs[0]
                loss.backward()
                optim.step()
                #evaluation
                #if steps % eval_steps_cycle == 0:
            self.evaluation()

def evaluation(self, batch_size = 4, save_best_model = True):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        #build dataloaders
        eval_loaders = {}
        for dataset_name in self.eval_dict:
            eval_loaders[dataset_name] = DataLoader(self.eval_dict[dataset_name].with_format("torch"), batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
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
                for idx in range(len(eval_batch)):
                    equation1 = eval_batch[idx]["equation1"]
                    equation2 = eval_batch[idx]["equation2"]
                    target = eval_batch[idx]["target"]
                    label = eval_batch[idx]["label"]
                    operation = eval_batch[idx]['operation']
                    outputs = self.model(equation1, equation2, target, operation, label)
                    batch_index = 0
                    for score in outputs[1]:
                        if score > 0.0:
                            logits_metric.append(1)
                        else:
                            logits_metric.append(0)
                    if label == 1.0:
                        scores_pos.append(outputs[1].detach().cpu().numpy())
                        label_metric.append(1)
                    else:
                        scores_neg.append(outputs[1].detach().cpu().numpy())
                        label_metric.append(0)
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
    parser.add_argument("--model", type=str, default="gat", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=8, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=int, default=32, nargs="?",
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
            batch_size = args.batch_size,
            neg = args.neg,
            max_length = args.max_length,
            epochs = args.epochs, 
            model = args.model,
            #load_model_path = "models/gat_best_dev_set_6.pt"
            )
    experiment.train_and_eval()
    #experiment.evaluation(save_best_model = False)




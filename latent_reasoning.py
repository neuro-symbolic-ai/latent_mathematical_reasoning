import os
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from latent_reasoning.data_model import DataModel
from latent_reasoning.sequential_utils import *
from latent_reasoning.gnn_utils_graph import Corpus as GraphCorpus
from latent_reasoning.Translational import TransLatentReasoning
from latent_reasoning.Projection import LatentReasoning
from sklearn.metrics import average_precision_score

class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, trans = True, one_hot = False, load_model_path = None):
        self.model_type = model
        print("Model:", self.model_type)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.trans = trans
        self.one_hot = one_hot
        #LOAD DATA
        if load_model_path is not None:
            #load pretrained vocabulary
            self.operations_voc = pickle.load(open(load_model_path + "/operations", "rb"))
            self.vocabulary =  pickle.load(open(load_model_path + "/vocabulary", "rb"))
            if self.model_type[:3] == 'gnn':
                self.corpus = GraphCorpus(self.max_length, build_voc = False)
                self.corpus.node_dict = self.vocabulary
            else:
                self.corpus = Corpus(self.max_length, build_voc = False)
                self.corpus.dictionary.word2idx = self.vocabulary
        else:
            if self.model_type[:3] == 'gnn':
                self.corpus = GraphCorpus(self.max_length)
            else:
                self.corpus = Corpus(self.max_length)
        self.tokenizer = self.corpus.tokenizer
        self.data_model = DataModel(self.tokenize_function_train, self.tokenize_function_eval, srepr=(self.model_type[:3] == 'gnn'))
        self.train_dataset = self.data_model.train_dataset
        self.eval_dict = self.data_model.eval_dict
        if load_model_path is None:
            self.operations_voc = self.data_model.operations_voc
            if self.model_type[:3] == 'gnn':
                self.vocabulary = self.corpus.node_dict
            else:
                self.vocabulary = self.corpus.dictionary.word2idx
        #LOAD METRICS AND MODEL
        self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create model
        if self.trans:
            #translational model
            self.model = TransLatentReasoning(len(self.vocabulary), self.num_ops, self.device, model_type = self.model_type)
        else:
            #baseline
            self.model = LatentReasoning(len(self.vocabulary), self.num_ops, self.device, model_type = self.model_type, one_hot = one_hot)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path + "/state_dict.pt"))

    def tokenize_function_train(self, examples):
        examples["premise"], examples["variable"], examples["target"] = self.tokenizer([examples["premise"], examples["variable"], examples["target"]])
        return examples
    
    def tokenize_function_eval(self, examples):
        examples["premise"] = self.tokenizer([examples["premise"]])[0]
        examples["positive"] = self.tokenizer(examples["positive"])
        examples["negative"] = self.tokenizer(examples["negative"]) 
        return examples

    def train_and_eval(self):
        device = self.device
        self.model.to(device)
        self.model.train()
        #TRAIN DATALOADER
        train_loader = DataLoader(self.train_dataset.with_format("torch"), batch_size=self.batch_size, shuffle=True)
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)
        #TRAINING CYCLE
        print("Start training...")
        steps = 0
        for epoch in tqdm(range(self.epochs), desc = "Training"):
            self.model.train()
            for batch in tqdm(train_loader):
                steps += 1
                optim.zero_grad()
                premise = batch["premise"]
                target = batch["target"]
                labels = batch['label']
                operation = batch['operation']
                outputs = self.model(premise, target, operation, labels)
                loss = outputs[0]
                loss.backward()
                optim.step()
            #DEV EVALUATION
            self.model.eval()
            self.evaluation(steps)
            self.model.train()
        #FINAL TEST EVALUATION
        self.model.eval()
        self.evaluation(steps, eval_type = "test")
        self.model.train()


    def evaluation(self, training_step, batch_size = 1, eval_type = "dev", save_best_model = True):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        #BUILD DATALOADER FOR EVALUATION
        eval_loaders = {}
        eval_metrics = {}
        eval_metrics["dev_set"] = {"avg_map": 0.0}
        map_dev = []
        for dataset_name in self.eval_dict:
            eval_loaders[dataset_name] = DataLoader(self.eval_dict[dataset_name].with_format("torch"), batch_size=batch_size, shuffle=False)
            if not dataset_name in self.eval_best_scores:
                self.eval_best_scores[dataset_name] = {"map": 0.0}
            if not "dev_set" in self.eval_best_scores:
                self.eval_best_scores["dev_set"] = {"avg_map": 0.0}
        #START EVALUATION
        print("EVALUATION")
        for loader in eval_loaders:
            if not(eval_type == "dev" and "dev" in loader) and not(eval_type == "test" and not "dev" in loader):
                continue
            eval_metrics[loader] = {}    
            eval_steps = 0
            avg_diff = []
            hit_1 = []
            hit_3 = []
            hit_5 = []
            map_res = []
            map_ops = {}
            map_len = {}
            for eval_batch in tqdm(eval_loaders[loader], desc = loader):
                scores_examples = {}
                scores_pos = []
                scores_neg = []
                true_relevance = []
                relevance_scores = []
                count_example = 0
                eval_steps += 1                
                premise = eval_batch["premise"]
                positives = eval_batch["positive"]
                negatives = eval_batch["negative"]
                operation = eval_batch["operation"]
                p_len = eval_batch["len"].detach().cpu().numpy()[0]
                for positive in positives:
                    score = self.model.inference_step(None, premise, None, positive, operation, None)[0]
                    score = score.detach().cpu().numpy()[0]
                    scores_pos.append(score)
                    scores_examples["pos_" + str(count_example)] = score
                    count_example += 1
                    true_relevance.append(1)
                    relevance_scores.append(score)
                for negative in negatives:
                    score = self.model.inference_step(None, premise, None, negative, operation, None)[0]
                    score = score.detach().cpu().numpy()[0]
                    scores_neg.append(score)
                    scores_examples["neg_" + str(count_example)] = score
                    count_example += 1
                    true_relevance.append(0)
                    relevance_scores.append(score)
                #COMPUTE EVALUATION SCORES FOR RANKING
                ap_score = average_precision_score(true_relevance, relevance_scores)
                op_name = self.operations_voc[int(operation[0])]
                if not op_name in map_ops:
                    map_ops[op_name] = []
                map_ops[op_name].append(ap_score)
                if not p_len in map_len:
                    map_len[p_len] = []
                map_len[p_len].append(ap_score)
                map_res.append(ap_score)
                avg_diff.append(np.mean(scores_pos) - np.mean(scores_neg))
                sorted_scores = dict(sorted(scores_examples.items(), key=lambda item: item[1], reverse = True))
                positive_hit = 1
                for id_example in sorted_scores:
                    if "pos" in id_example:
                        break
                    positive_hit += 1
                if positive_hit <= 1:
                    hit_1.append(1)
                else:
                    hit_1.append(0)
                if positive_hit <= 3:
                    hit_3.append(1)
                else:
                    hit_3.append(0)
                if positive_hit <= 5:
                    hit_5.append(1)
                else:
                    hit_5.append(0)
            eval_metrics[loader]["map"] = np.mean(map_res)
            eval_metrics[loader]["hit@1"] = np.mean(hit_1)
            eval_metrics[loader]["hit@3"] = np.mean(hit_3)
            eval_metrics[loader]["hit@5"] = np.mean(hit_5)
            eval_metrics[loader]["difference"] = np.mean(avg_diff)
            for op in map_ops:
                map_ops[op] = np.mean(map_ops[op])
            eval_metrics[loader]["map_ops"] = map_ops
            for p_len in map_len:
                map_len[p_len] = np.mean(map_len[p_len])
            eval_metrics[loader]["map_len"] = map_len
            if "dev" in loader:
                map_dev.append(eval_metrics[loader]["map"])
            #print results
            print("=============="+loader+"_"+str(training_step)+"==============")
            print("current scores:", eval_metrics[loader])

        if eval_type != "dev":
            return
        #CHECK AND SAVE BEST MODEL
        eval_metrics["dev_set"]["avg_map"] = np.mean(map_dev)
        if eval_metrics["dev_set"]["avg_map"] > self.eval_best_scores["dev_set"]["avg_map"]:
            #new best score
            print("New best model!!!")
            self.eval_best_scores = eval_metrics
            #SAVE THE MODEL
            if save_best_model:
                print("Save the model...")
                PATH = "models/" + self.model_type + "_" + str(self.trans) + "_" + str(self.one_hot) + "_" +str(self.num_ops) + "_" + str(self.model.dim) + "/"
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                #save model parameters
                torch.save(self.model.state_dict(), PATH + "state_dict.pt")
                #save vocabulary
                pickle.dump(self.vocabulary, open(PATH + "vocabulary", "wb"))
                #save operations dictionary
                pickle.dump(self.operations_voc, open(PATH + "operations", "wb"))
        print("===========Best Model==========")
        print(self.eval_best_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="transformer", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=64, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=int, default=32, nargs="?",
                    help="Num epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, nargs="?",
                    help="Learning rate.")

    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True 
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    # SET PRE-TRAINED MODEL PATH FOR EVALUATION, NONE FOR TRAINING 
    model_path = None
    trans = True
    one_hot = False
    if model_path == None:
        experiment = Experiment(
                learning_rate = args.lr, 
                batch_size = args.batch_size, 
                max_length = args.max_length,
                epochs = args.epochs,
                model = args.model,
                trans = trans,
                one_hot = one_hot
                )
        experiment.train_and_eval()
    else:
        experiment = Experiment(
                learning_rate = args.lr, 
                batch_size = args.batch_size, 
                max_length = args.max_length,
                epochs = args.epochs,
                model = args.model,
                trans = trans,
                one_hot = one_hot,
                load_model_path = model_path
                )
        experiment.model.eval()
        experiment.evaluation(0, eval_type = "test", save_best_model = False)

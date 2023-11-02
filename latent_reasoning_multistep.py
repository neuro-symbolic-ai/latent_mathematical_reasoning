import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from latent_reasoning.data_model import DataModelMultiStep
from latent_reasoning.sequential_utils import *
from latent_reasoning.gnn_utils_graph import Corpus as GraphCorpus
from latent_reasoning.Projection import LatentReasoning
from latent_reasoning.Translational import TransLatentReasoning
    
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
        print(self.vocabulary)
        #LOAD DATA
        if self.model_type[:3] == 'gnn':
            self.corpus = GraphCorpus(self.max_length, build_voc = False)
            self.corpus.node_dict = self.vocabulary
        else:
            self.corpus = Corpus(self.max_length, build_voc = False)
            self.corpus.dictionary.word2idx = self.vocabulary
        self.tokenizer = self.corpus.tokenizer
        self.data_model = DataModelMultiStep(neg, self.operations_voc, self.tokenize_function, srepr=(self.model_type[:3] == 'gnn'))
        self.eval_dict = self.data_model.eval_dict
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

    def tokenize_function(self, examples):
        for step in examples["steps"]:
            if examples["steps"][step] == None:
                examples["steps"][step] = []
                continue
            for instance in examples["steps"][step]:
                instance["equation1"], instance["equation2"], instance["target"] = self.tokenizer([instance["equation1"], instance["equation2"], instance["target"]])
        return examples

    def evaluation(self, batch_size = 1):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        #build dataloaders
        eval_loaders = {}
        for step in self.eval_dict:
            eval_loaders[step] = DataLoader(self.eval_dict[step].with_format("torch"), batch_size=batch_size, shuffle=False)
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
            max_batch = 1000
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
                    temp_score = {}
                    for item in eval_batch["steps"][inference_step]:
                        premise = item["premise"]
                        target = item["target"]
                        labels = item["label"]
                        operation = item['operation']
                        if inference_step == "0":
                            outputs = self.model.inference_step(None, premise, target, operation, labels)
                        else:
                            outputs = self.model.inference_step(inference_state[item_index], None, target, operation, labels)
                        inference_state[item_index] = outputs[1]
                        for score in outputs[0]:
                            temp_score[item_index] = score
                        item_index += 1
                    hit = False
                    sorted_scores = dict(sorted(temp_score.items(), key=lambda item: item[1], reverse = True))
                    for item_id in sorted_scores:
                        if item_id == 0:
                            hit = True
                        break
                    if hit:
                        label_metric[inference_step].append(1)
                    else:
                        label_metric[inference_step].append(0)
                if batch_index > max_batch:
                    break
                batch_index += 1
            eval_metrics = {}
            positive_avg = {}
            negative_avg = {}
            difference_avg = {}
            for inference_step in logits_metric:
                eval_metrics[inference_step] =  np.mean(label_metric[inference_step])
                positive_avg[inference_step] = np.mean(scores_pos[inference_step])
                negative_avg[inference_step] = np.mean(scores_neg[inference_step])
                difference_avg[inference_step] = positive_avg[inference_step] - negative_avg[inference_step]
            #print results
            print("=============="+str(step)+"==============")
            print("evaluation scores:", eval_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
            trans = True,
            one_hot = False,
            load_model_path = "models/rnn_True_False_6_768",
            )
    experiment.evaluation()

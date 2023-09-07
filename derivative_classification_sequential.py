import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from latent_reasoning.data_model import DataModel
from latent_reasoning.sequential_utils import *
from latent_reasoning.TranslationalReasoningSequential import TransLatentReasoningSeq
from latent_reasoning.BaselinesSequential import LatentReasoningSeq
from sklearn.metrics import average_precision_score#, precision_recall_curve, ndcg_score, label_ranking_average_precision_score    

class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, trans = True, load_model_path = None, do_train = True, do_test = False):
        self.model_type = model
        print("Model:", self.model_type)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        self.trans = trans
        #LOAD DATA
        self.corpus = Corpus(self.max_length)
        self.tokenizer = self.corpus.tokenizer
        self.data_model = DataModel(neg, do_train, do_test, self.tokenize_function_train, self.tokenize_function_eval)
        self.train_dataset = self.data_model.train_dataset
        self.eval_dict = self.data_model.eval_dict
        self.operations_voc = self.data_model.operations_voc
        print("Vocabulary: ", self.corpus.dictionary.word2idx)
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        self.eval_best_scores = {}
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create model
        if self.trans:
            #translational model
            self.model = TransLatentReasoningSeq(len(self.corpus.dictionary), self.num_ops, self.device, model_type = self.model_type)
        else:
            #baseline
            self.model = LatentReasoningSeq(len(self.corpus.dictionary), self.num_ops, self.device, model_type = self.model_type)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path))

    def tokenize_function_train(self, examples):
        examples["equation1"], examples["equation2"], examples["target"] = self.tokenizer([examples["equation1"], examples["equation2"], examples["target"]])
        return examples
    
    def tokenize_function_eval(self, examples):
        examples["premise"] = self.tokenizer([examples["premise"]])[0]
        examples["positive"] = self.tokenizer(examples["positive"])
        examples["negative"] = self.tokenizer(examples["negative"]) 
        return examples

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        score = self.metric.compute(predictions=logits, references=labels)
        return score

    def train_and_eval(self):
        device = self.device
        self.model.to(device)
        self.model.train()
        #TRAIN DATALOADER
        train_loader = DataLoader(self.train_dataset.with_format("torch"), batch_size=self.batch_size, shuffle=True)
        optim = AdamW(self.model.parameters(), lr=self.learning_rate)
        #TRAINING CYCLE
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
                    self.model.eval()
                    self.evaluation()
                    self.model.train()


    def evaluation(self, batch_size = 1, save_best_model = True):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        #BUILD DATALOADER FOR EVALUATION
        eval_loaders = {}
        for dataset_name in self.eval_dict:
            eval_loaders[dataset_name] = DataLoader(self.eval_dict[dataset_name].with_format("torch"), batch_size=batch_size, shuffle=False)
            if not dataset_name in self.eval_best_scores:
                self.eval_best_scores[dataset_name] = {"difference": 0.0}
        #START EVALUATION
        print("EVALUATION")
        for loader in eval_loaders:
            eval_steps = 0
            max_steps = 100
            avg_diff = []
            hit_1 = []
            hit_3 = []
            hit_5 = []
            ndcg_res = []
            map_res = []
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
                operation = eval_batch['operation']
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
                #print(true_relevance, relevance_scores)
                #ndcg_res.append(ndcg_score(true_relevance, relevance_scores))
                map_res.append(average_precision_score(true_relevance, relevance_scores))
                avg_diff.append(np.mean(scores_pos) - np.mean(scores_neg))
                sorted_scores = dict(sorted(scores_examples.items(), key=lambda item: item[1], reverse = True))
                #print(sorted_scores)
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
                #if eval_steps > max_steps:
                #    break
            #eval_metrics = self.compute_metrics([logits_metric, label_metric])
            eval_metrics = {}
            #eval_metrics["ndgc"] = np.mean(ndcg_res)
            eval_metrics["avg precision"] = np.mean(map_res)
            eval_metrics["hit@1"] = np.mean(hit_1)
            eval_metrics["hit@3"] = np.mean(hit_3)
            eval_metrics["hit@5"] = np.mean(hit_5)
            eval_metrics["difference"] = np.mean(avg_diff)
            if eval_metrics["difference"] > self.eval_best_scores[loader]["difference"]:
                #new best score
                self.eval_best_scores[loader] = eval_metrics
                #SAVE THE MODEL'S PARAMETERS
                if save_best_model:
                    PATH = "models/" + self.model_type + "_best_" + loader + "_" + str(self.trans) + "_" + str(self.num_ops) + ".pt"
                    torch.save(self.model.state_dict(), PATH)
            #print results
            print("=============="+loader+"==============")
            print("current scores:", eval_metrics)
            print("best scores:", self.eval_best_scores[loader])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="transformer", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=32, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=128, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=int, default=12, nargs="?",
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
            trans = True,
            #load_model_path = "models/rnn_best_dev_set_6.pt",
            #do_train = False,
            #do_test = True
            )
    experiment.train_and_eval()
    #experiment.evaluation(save_best_model = False)

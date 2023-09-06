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
        eval_steps_cycle = 1000
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
                self.eval_best_scores[dataset_name] = {"accuracy": 0.0, "f1": 0.0, "difference": 0.0}
        #START EVALUATION
        print("EVALUATION")
        for loader in eval_loaders:
            eval_steps = 0
            max_steps = 1000
            scores_pos = []
            scores_neg = []
            logits_metric = []
            label_metric = []
            for eval_batch in tqdm(eval_loaders[loader], desc = loader):
                eval_steps += 1
                premise = eval_batch["premise"]
                positives = eval_batch["positive"]
                negatives = eval_batch["negative"]
                operation = eval_batch['operation']
                for positive in positives:
                    score = self.model.inference_step(None, premise, None, positive, operation, None)[0]
                    scores_pos.append(score)
                for negative in negatives:
                    score = self.model.inference_step(None, premise, None, negative, operation, None)[0]
                    scores_neg.append(score)
                if eval_steps > max_steps:
                    break
            #eval_metrics = self.compute_metrics([logits_metric, label_metric])
            eval_metrics["difference"] = np.mean(scores_pos) - np.mean(scores_neg)
            if eval_metrics["difference"] > self.eval_best_scores[loader]["difference"]:
                #new best score
                self.eval_best_scores[loader] = eval_metrics
                #SAVE THE MODEL'S PARAMETERS
                if save_best_model:
                    PATH = "models/" + self.model_type + "_best_" + loader + "_" + str(self.trans) + "_" + str(self.num_ops) + ".pt"
                    torch.save(self.model.state_dict(), PATH)
            #print results
            print("=============="+loader+"==============")
            print("positive avg sim:", np.mean(scores_pos))
            print("negative avg sim:", np.mean(scores_neg))
            print("current scores:", eval_metrics)
            print("best scores:", self.eval_best_scores[loader])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="transformer", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=8, nargs="?",
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

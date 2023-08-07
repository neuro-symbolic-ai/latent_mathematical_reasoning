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
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, load_model_path = None, do_train = True, do_test = False):
        self.model_type = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.batch_size = batch_size
        #LOAD DATA
        self.corpus = Corpus(self.max_length)
        self.tokenizer = self.corpus.var_tokenizer
        self.data_model = DataModel(neg, do_train, do_test, self.tokenize_function)
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
        self.model = TransLatentReasoningSeq(len(self.corpus.dictionary), self.num_ops, self.device, model_type = self.model_type)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path))

    def tokenize_function(self, examples):
        examples["equation1"], examples["equation2"], examples["target"] = self.tokenizer([examples["equation1"], examples["equation2"], examples["target"]])
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
                #if steps % eval_steps_cycle == 0:
            self.model.eval()
            self.evaluation()
            self.model.train()


    def evaluation(self, batch_size = 4, save_best_model = True):
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
                label_index = 0
                for label in labels:
                    if label == 1.0:
                        scores_pos.append(outputs[1].detach().cpu().numpy()[label_index])
                        label_metric.append(1)
                    else:
                        scores_neg.append(outputs[1].detach().cpu().numpy()[label_index])
                        label_metric.append(0)
                    label_index += 1
                    batch_index += 1
                #if eval_steps > max_steps:
                #    break
            eval_metrics = self.compute_metrics([logits_metric, label_metric])
            eval_metrics["difference"] = np.mean(scores_pos) - np.mean(scores_neg)
            if eval_metrics["difference"] > self.eval_best_scores[loader]["difference"]:
                #new best score
                self.eval_best_scores[loader] = eval_metrics
                #SAVE THE MODEL'S PARAMETERS
                if save_best_model:
                    PATH = "models/" + self.model_type + "_best_" + loader + "_" + str(self.num_ops) + ".pt"
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
            #load_model_path = "models/rnn_best_dev_set_6.pt",
            #do_train = False,
            #do_test = True
            )
    experiment.train_and_eval()
    #experiment.evaluation(save_best_model = False)

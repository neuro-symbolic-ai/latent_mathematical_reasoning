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
from latent_reasoning.TranslationalLatentReasoning import TransLatentReasoning
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, dataset_name):
        self.model_name = model
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        #PROCESS DATA
        self.train_dataset = self.process_dataset(neg = neg)
        self.tokenized_train_datasets = self.train_dataset.map(self.tokenize_function, batched=False)
        #LOAD METRICS AND MODEL
        self.metric = evaluate.load("glue", "mrpc")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = TransLatentReasoning(model, 2, self.device)


    def process_dataset(self, dataset_path = ["data/differentiation.json", "data/integration.json"], neg = 1,  test_size = 0.2):
        #convert dataset into json for dataset loader
        formatted_examples = []
        op_id = 0
        for path in dataset_path:
            d_file = open(path, 'r')
            d_json = json.load(d_file)
            # create an entry for each positive example
            for example in tqdm(d_json, desc="Loading Dataset"):
                formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": example["positive"], "operation": op_id, "label": 1.0})
                #create an entry for each negative example
                count_neg = 0
                for negative in example["negatives"]:
                    if count_neg == neg:
                        break
                    formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": negative , "operation": op_id, 'label': -1.0})
                    count_neg += 1
            op_id += 1
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
        examples["target"] = self.tokenizer(examples["target"], padding="max_length", truncation=True, max_length = self.max_length) #, return_tensors='pt')
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
        
        self.model.train()
        
        train_loader = DataLoader(self.tokenized_train_datasets["train"].with_format("torch"), batch_size=8, shuffle=True)
        dev_loader = DataLoader(self.tokenized_train_datasets["test"].with_format("torch"), batch_size=4, shuffle=True)

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
                    scores_pos = []
                    scores_neg = []
                    print("EVALUATION")
                    num_instances = 2000
                    for loader in eval_loaders:
                        eval_steps = 0
                        logits_metric = []
                        label_metric = []
                        for eval_batch in tqdm(eval_loaders[loader]):
                            eval_steps += 1
                            equation1 = eval_batch["equation1"]
                            equation2 = eval_batch["equation2"]
                            target = eval_batch["target"]
                            labels = eval_batch["label"]
                            operation = eval_batch['operation']
                            outputs = self.model(equation1, equation2, target, operation, labels)
                            for score in outputs[1]:
                                if score > 0.0:
                                    logits_metric.append(1)
                                else:
                                    logits_metric.append(0)
                            for label in labels:
                                if label == 1.0:
                                    scores_pos.append(outputs[1].detach().cpu().numpy())
                                    label_metric.append(1)
                                else:
                                    scores_neg.append(outputs[1].detach().cpu().numpy())
                                    label_metric.append(0)
                            if eval_steps > num_instances:
                                break
                        print("=============="+loader+"==============")
                        print("positive:", np.mean(scores_pos))
                        print("negative:", np.mean(scores_neg))
                        print("difference:", np.mean(scores_pos) - np.mean(scores_neg))
                        print(self.compute_metrics([logits_metric, label_metric]))
                    self.model.train()


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
            batch_size = args.batch_size, 
            neg = args.neg,
            max_length = args.max_length,
            epochs = args.epochs, 
            model = args.model, 
            dataset_name = dataset
            )
    experiment.train_and_eval()

#custom dataset
#import torch

#train_encodings = tokenizer(train_texts, truncation=True, padding=True)
#val_encodings = tokenizer(val_texts, truncation=True, padding=True)
#test_encodings = tokenizer(test_texts, truncation=True, padding=True)

#class IMDbDataset(torch.utils.data.Dataset):
#    def __init__(self, encodings, labels):
#        self.encodings = encodings
#        self.labels = labels

#    def __getitem__(self, idx):
#        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#        item['labels'] = torch.tensor(self.labels[idx])
#        return item

#    def __len__(self):
#        return len(self.labels)

#train_dataset = IMDbDataset(train_encodings, train_labels)
#val_dataset = IMDbDataset(val_encodings, val_labels)
#test_dataset = IMDbDataset(test_encodings, test_labels)

import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import evaluate
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, AdamW
from torch.utils.data import DataLoader
from latent_reasoning.LatentReasoning import LatentReasoning
    
class Experiment:

    def __init__(self, learning_rate, model, epochs, batch_size, max_length, neg, dataset_path):
        self.model_name = model
        self.dataset_path = dataset_path
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.dataset = self.process_dataset(dataset_path, neg)
        self.tokenized_datasets = self.dataset.map(self.tokenize_function, batched=True)
        self.model = LatentReasoning(model) #AutoModelForSequenceClassification.from_pretrained(model, num_labels=2)
        self.metric = evaluate.load("glue", "mrpc")
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        #self.training_args = TrainingArguments(
        #        output_dir="output/"+dataset_path.split("/")[-1]+"_"+self.model_name,
        #        logging_steps = 1000,
        #        evaluation_strategy="steps",
        #        eval_steps = 1000,
        #        save_steps = 1000,
        #        num_train_epochs = epochs,
        #        learning_rate = learning_rate,
        #        per_device_train_batch_size = batch_size,
        #        save_total_limit = 1,
        #        save_strategy = "steps",
        #        load_best_model_at_end = True,
        #        metric_for_best_model = "f1"
        #        )

    def process_dataset(self, dataset_path, neg):
        #convert dataset into json for dataset loader
        d_file = open(dataset_path, 'r')
        d_json = json.load(d_file)
        formatted_examples = []
        # create an entry for each positive example
        for example in tqdm(d_json, desc="Loading Dataset"):
            formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": example["positive"], "label": 1})
            #create an entry for each negative example
            count_neg = 0
            for negative in example["negatives"]:
                if count_neg == neg:
                    break
                formatted_examples.append({"text": "equation1": example['premise_expression'], "equation2": example['variable'], "target": negative ,input_text, 'label': 0})
                count_neg += 1
        print("Data examples", formatted_examples[:4])
        #split randomly between train, dev, and test set
        dataset = Dataset.from_list(formatted_examples)
        dataset_split = dataset.train_test_split(test_size=0.2)
        return dataset_split

    def tokenize_function(self, examples):
        examples["equation1"] = self.tokenizer(examples["equation1"], padding=True, truncation=True, return_tensors='pt')
        examples["equation2"] = self.tokenizer(examples["equation2"], padding=True, truncation=True, return_tensors='pt')
        examples["target"] = self.tokenizer(examples["target"], padding=True, truncation=True, return_tensors='pt')
        return examples

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        majority_class_preds = [1 for pred in predictions]
        majority_baseline_score = self.metric.compute(predictions=majority_class_preds, references=labels)
        print("majority_class_baseline:", majority_baseline_score)
        score = self.metric.compute(predictions=predictions, references=labels)
        return score

    def train_and_eval(self):
        
        #trainer = Trainer(
        #    model = self.model,
        #    args = self.training_args,
        #    train_dataset = self.tokenized_datasets["train"],
        #    eval_dataset = self.tokenized_datasets["test"],
        #    compute_metrics = self.compute_metrics
        #)
        #trainer.train()
        #trainer.save_model()

        self.model.to(device)
        self.model.train()

        train_loader = DataLoader(self.tokenized_datasets["train"].with_format("torch"), batch_size=4, shuffle = True) 
        optim = AdamW(self.model.parameters(), lr=5e-5)

        for epoch in range(3):
            for batch in train_loader:
                optim.zero_grad()
                equation1 = batch["equation1"].to(device)
                equation2 = batch["equation2"].to(device)
                target = batch["target"].to(device)
                labels = batch['labels'].to(device)
                outputs = model(equation1, equation2, target, labels=labels)
                print(outputs)
                loss = outputs[0]
                loss.backward()
                optim.step()
        model.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation.json", nargs="?",
                    help="Which dataset to use")
    parser.add_argument("--model", type=str, default="sentence-transformers/bert-base-nli-mean-tokens", nargs="?",
                    help="Which model to use")
    parser.add_argument("--batch_size", type=int, default=8, nargs="?",
                    help="Batch size.")
    parser.add_argument("--max_length", type=int, default=256, nargs="?",
                    help="Input Max Length.")
    parser.add_argument("--epochs", type=float, default=12.0, nargs="?",
                    help="Num epochs.")
    parser.add_argument("--lr", type=float, default=5e-7, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--neg", type=int, default=1, nargs="?",
                    help="Max number of negative examples")

    args = parser.parse_args()
    dataset = args.dataset
    data_path = "data/"+dataset
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
            dataset_path = data_path
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
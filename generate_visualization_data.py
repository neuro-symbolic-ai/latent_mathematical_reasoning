import json
import pickle
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from latent_reasoning.data_model import DataModel
from latent_reasoning.sequential_utils import *
from latent_reasoning.Translational import TransLatentReasoning
from latent_reasoning.Projection import LatentReasoning

class Experiment:

    def __init__(self, model, batch_size, max_length, neg, trans = True, one_hot = False, load_model_path = None, do_train = True, do_test = False):
        self.model_type = model
        print("Model:", self.model_type)
        self.max_length = max_length
        self.batch_size = batch_size
        self.trans = trans
        self.one_hot = one_hot
        #LOAD DATA
        if load_model_path is not None:
            #load pretrained vocabulary
            self.operations_voc = pickle.load(open(load_model_path + "/operations", "rb"))
            self.vocabulary =  pickle.load(open(load_model_path + "/vocabulary", "rb"))
            self.corpus = Corpus(self.max_length, build_voc = False)
            self.corpus.dictionary.word2idx = self.vocabulary
        else:
            self.corpus = Corpus(self.max_length)
        self.tokenizer = self.corpus.tokenizer
        self.data_model = DataModel(neg, do_train, do_test, self.tokenize_function_train, self.tokenize_function_eval)
        self.train_dataset = self.data_model.train_dataset
        self.eval_dict = self.data_model.eval_dict
        if load_model_path is None:
            self.operations_voc = self.data_model.operations_voc
            self.vocabulary = self.corpus.dictionary.word2idx
        #LOAD MODEL
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_ops = len(self.operations_voc.keys())
        #create model
        if self.trans:
            #translational model
            self.model = TransLatentReasoning(len(self.corpus.dictionary.word2idx.keys()), self.num_ops, self.device, model_type = self.model_type)
        else:
            #baseline
            self.model = LatentReasoning(len(self.corpus.dictionary.word2idx.keys()), self.num_ops, self.device, model_type = self.model_type, one_hot = one_hot)
        if load_model_path is not None:
            #load pretrained model
            self.model.load_state_dict(torch.load(load_model_path + "/state_dict.pt"))

    def tokenize_function_train(self, examples):
        examples["equation1"], examples["equation2"], examples["target"] = self.tokenizer([examples["equation1"], examples["equation2"], examples["target"]])
        return examples
    
    def tokenize_function_eval(self, examples):
        examples["premise"] = self.tokenizer([examples["premise"]])[0]
        examples["positive"] = self.tokenizer(examples["positive"])
        examples["negative"] = self.tokenizer(examples["negative"]) 
        return examples

    def generate_embeddings(self, batch_size = 1, data_type = "dev"):
        if self.eval_dict == None:
            print("No evaluation data found!")
            return
        
        eval_loaders = {}
        for dataset_name in self.eval_dict:
            eval_loaders[dataset_name] = DataLoader(self.eval_dict[dataset_name].with_format("torch"), batch_size = batch_size, shuffle = False)

        #BUILD DATALOADER FOR EVALUATION
        print("Generating embeddings...")
        embeddings_data = {}
        embeddings_data_trans = {}
        for loader in eval_loaders:
            if not(data_type == "dev" and "dev" in loader) and not(data_type == "test" and not "dev" in loader):
                continue
            eval_steps = 0
            embeddings_data[loader] = {}
            embeddings_data_trans[loader] = {}
            for eval_batch in tqdm(eval_loaders[loader], desc = loader):
                embeddings_data[loader][eval_steps] = {}    
                embeddings_data_trans[loader][eval_steps] = {}     
                premise = eval_batch["premise"]
                positives = eval_batch["positive"]
                negatives = eval_batch["negative"]
                operation = eval_batch["operation"]
                embeddings_data[loader][eval_steps]["premise"] = self.model.encode(premise, None, is_premise = True).detach().cpu().numpy()[0].tolist()
                embeddings_data_trans[loader][eval_steps]["premise"]  = self.model.encode(premise, operation, is_premise = True).detach().cpu().numpy()[0].tolist()
                embeddings_data[loader][eval_steps]["positives"] = []
                embeddings_data_trans[loader][eval_steps]["positives"] = []
                embeddings_data[loader][eval_steps]["negatives"] = []
                embeddings_data_trans[loader][eval_steps]["negatives"] = []
                for positive in positives:
                    embeddings_data[loader][eval_steps]["positives"].append(self.model.encode(positive, None, is_premise = False).detach().cpu().numpy()[0].tolist())
                    embeddings_data_trans[loader][eval_steps]["positives"].append(self.model.encode(positive, operation, is_premise = False).detach().cpu().numpy()[0].tolist())
                for negative in negatives:
                    embeddings_data[loader][eval_steps]["negatives"].append(self.model.encode(negative, None, is_premise = False).detach().cpu().numpy()[0].tolist())
                    embeddings_data_trans[loader][eval_steps]["negatives"].append(self.model.encode(negative, operation, is_premise = False).detach().cpu().numpy()[0].tolist())
        for dataset in embeddings_data:
            json.dump(embeddings_data[dataset], open(str(self.trans) + dataset + "_embeddings.json", "w"), indent = 5)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="differentiation", nargs="?",
                    help="Which dataset to use")
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
            batch_size = args.batch_size, 
            neg = args.neg,
            max_length = args.max_length,
            model = args.model,
            trans = False,
            one_hot = False,
            load_model_path = "models/transformer_False_False_6_300"
            )
    experiment.model.eval()
    experiment.generate_embeddings(data_type = "test")

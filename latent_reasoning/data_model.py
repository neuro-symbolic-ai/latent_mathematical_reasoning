import json
import random
from tqdm import tqdm
from datasets import Dataset
    
class DataModel:

    def __init__(self, neg = 1, do_train = True, do_test = False, tokenize_function_train = None, tokenize_function_eval = None srepr = False):
        #PROCESS DATA
        self.tokenize_function_train = tokenize_function_train
        self.tokenize_function_eval = tokenize_function_train
        #training data needs to be processed for operations and setup
        self.train_dataset, self.eval_dataset = self.process_dataset(neg = neg, srepr = srepr) #dataset_path = ["data/differentiation.json", "data/integration.json"])
        self.eval_dict = {}
        self.tokenized_train_dataset = self.train_dataset.map(self.tokenize_function_train, batched=False)
        self.tokenized_eval_dataset = self.eval_dataset.map(self.tokenize_function_eval, batched=False)
        self.train_dataset = self.tokenized_train_dataset
        self.eval_dict["dev_set"] = self.tokenized_eval_dataset


   # ["integrate", "differentiate", "add", "minus", "times", "divide"]
    
    def process_dataset(self, dataset_path = "data/premises_dataset.json", operations = ["integrate", "differentiate"], neg = 1,  training = True, merge = True, test_size = 0.2, srepr = False):
        #load operation vocabulary
        if training:
            self.operations_voc = {}
            self.opereations_voc_rev = {}
            op_id = 0
            for op in operations:
                self.operations_voc[op_id] = op
                self.opereations_voc_rev[op] = op_id
                op_id += 1

        #convert dataset into json for dataset loader
        formatted_examples_train = []
        formatted_examples_eval = []
        d_file = open(dataset_path, 'r')
        d_json = json.load(d_file)
        max_num_train_examples = 2000
        max_num_eval_examples = 2000        
        id_example = 0
        # create a training entry for each example
        for example in tqdm(d_json, desc= dataset_path):
            if id_example >= max_num_train_examples:
                break
            #if example["var_count"] != 2:
            #    continue
            id_example += 1
            premise = example["premise"]
            for op in operations:
                #POSITIVE EXAMPLES
                for res in example[op]:
                    #LATEX
                    formatted_examples_train.append({"equation1": premise, "equation2": res["var"], "target": res["res"], "operation": self.opereations_voc_rev[op], "label": 1.0})
                    #print(formatted_examples[-1])
                #NEGATIVE EXAMPLES
                #select random operation for negative example
                #neg_operations = ["add", "minus", "times", "divide"]
                #op_neg = neg_operations[random.randint(0, len(neg_operations) - 1 )]
                #while op_neg == op:
                #    op_neg = neg_operations[random.randint(0, len(neg_operations) - 1)]
                #print("===================================================================")
                #for res in example[op_neg]:
                    #LATEX
                #    formatted_examples.append({"equation1": premise, "equation2": res["var"], "target": res["res"], "operation": self.opereations_voc_rev[op], "label": -1.0})
                    #print(formatted_examples[-1])

        # create an evaluation entry for each example
        id_example = 0
        for example in tqdm(d_json[:max_num_train_examples], desc= dataset_path):
            if id_example >= max_num_eval_examples:
                break
            #if example["var_count"] != 2:
            #    continue
            id_example += 1
            premise = example["premise"]
            for op in operations:
                #POSITIVE EXAMPLES
                positive_examples = []
                for res in example[op]:
                    #LATEX
                    positive_examples.append(res["res"])
                    #print(formatted_examples[-1])
                #NEGATIVE EXAMPLES
                #select random operation for negative example
                neg_operations = ["add", "minus", "times", "divide"]
                op_neg = neg_operations[random.randint(0, len(neg_operations) - 1 )]
                while op_neg == op:
                    op_neg = neg_operations[random.randint(0, len(neg_operations) - 1)]
                #print("===================================================================")
                negative_examples = []
                for res in example[op_neg]:
                    #LATEX
                    negative_examples.append(res["res"])
                    #print(formatted_examples[-1])
                formatted_examples_eval.append({"premise": premise, "operation": self.opereations_voc_rev[op], "positive": positive_examples, "negative": negative_examples})
        
        #split randomly between train, dev, and test set
        
        #split randomly between train, dev, and test set
        dataset_train = Dataset.from_list(formatted_examples_train)
        dataset_eval = Dataset.from_list(formatted_examples_eval)
        #if test_size == 1.0:
        #    return dataset
        #dataset_split = dataset.train_test_split(test_size = test_size)
        return dataset_train, dataset_eval

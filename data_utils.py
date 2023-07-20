import json
from tqdm import tqdm
from datasets import Dataset
    
class DatasetUtils:

    def __init__(self, do_train = True, do_test = False, tokenize_function = None, srepr = False):
        #PROCESS DATA
        self.tokenize_function = tokenize_function
        #training data needs to be processed for operations and setup
        self.train_dataset = self.process_dataset(neg = neg)
        self.eval_dict = {}
        if do_train:
            self.tokenized_train_dataset = self.train_dataset.map(self.tokenize_function, batched=False)
            self.train_dataset = self.tokenized_train_dataset["train"]
            self.eval_dict["dev_set"] = self.tokenized_train_dataset["test"]
        if do_test:
            #test differentiation
            self.test_dataset_diff = self.process_dataset(dataset_path = ["data/EVAL_differentiation.json", "data/EVAL_differentiation_VAR_SWAP.json", "data/EVAL_differentiation_EQ_CONV.json", "data/EVAL_easy_differentiation.json"], 
                neg = neg, training = False, merge = False, test_size = 1.0, srepr = srepr)
            for dataset_name in self.test_dataset_diff:
                self.eval_dict[dataset_name] = self.test_dataset_diff[dataset_name].map(self.tokenize_function, batched=False)
            #test integration
            self.test_dataset_int = self.process_dataset(dataset_path = ["data/EVAL_integration.json", "data/EVAL_integration_VAR_SWAP.json", "data/EVAL_integration_EQ_CONV.json", "data/EVAL_easy_integration.json"], 
                neg = neg, training = False, merge = False, test_size = 1.0, srepr = srepr)
            for dataset_name in self.test_dataset_int:
                self.eval_dict[dataset_name] = self.test_dataset_int[dataset_name].map(self.tokenize_function, batched=False)
            #test addition
            self.test_dataset_add = self.process_dataset(dataset_path = ["data/EVAL_addition.json", "data/EVAL_addition_VAR_SWAP.json", "data/EVAL_addition_EQ_CONV.json"], 
                neg = neg, training = False, merge = False, test_size = 1.0, srepr = srepr)
            for dataset_name in self.test_dataset_add:
                self.eval_dict[dataset_name] = self.test_dataset_add[dataset_name].map(self.tokenize_function, batched=False)
            #test subtraction
            self.test_dataset_sub = self.process_dataset(dataset_path = ["data/EVAL_subtraction.json", "data/EVAL_subtraction_VAR_SWAP.json", "data/EVAL_subtraction_EQ_CONV.json"], 
                neg = neg, training = False, merge = False, test_size = 1.0, srepr = srepr)
            for dataset_name in self.test_dataset_sub:
                self.eval_dict[dataset_name] = self.test_dataset_sub[dataset_name].map(self.tokenize_function, batched=False)
            #test multiplication
            self.test_dataset_mul = self.process_dataset(dataset_path = ["data/EVAL_multiplication.json", "data/EVAL_multiplication_VAR_SWAP.json", "data/EVAL_multiplication_EQ_CONV.json"], 
                neg = neg, training = False, merge = False, test_size = 1.0, srepr = srepr)
            for dataset_name in self.test_dataset_mul:
                self.eval_dict[dataset_name] = self.test_dataset_mul[dataset_name].map(self.tokenize_function, batched=False)  
            #test division
            self.test_dataset_div = self.process_dataset(dataset_path = ["data/EVAL_division.json", "data/EVAL_division_VAR_SWAP.json", "data/EVAL_division_EQ_CONV.json"], 
                neg = neg, training = False, merge = False, test_size = 1.0, srepr = srepr)
            for dataset_name in self.test_dataset_div:
                self.eval_dict[dataset_name] = self.test_dataset_div[dataset_name].map(self.tokenize_function, batched=False)

    def process_dataset(self, dataset_path = ["data/differentiation.json", "data/integration.json", "data/addition.json", "data/subtraction.json", "data/multiplication.json", "data/division.json"], neg = 1,  training = True, merge = True, test_size = 0.2, srepr = False):
        #load operation vocabulary
        if training:
            self.operations_voc = {}
            op_id = 0
            for path in dataset_path:
                self.operations_voc[op_id] = path.split("/")[-1].replace(".json", "")
                op_id += 1
        #convert dataset into json for dataset loader
        if merge:
            formatted_examples = []
        else:
            formatted_examples = {}
        for path in dataset_path:
            #find operation id
            for entry in self.operations_voc:
                if self.operations_voc[entry] in path:
                    op_id = entry
                    break
            d_file = open(path, 'r')
            d_json = json.load(d_file)
            # create an entry for each positive example
            for example in tqdm(d_json, desc="Loading Dataset"):
                if merge:
                    #LATEX
                    if not srepr:
                        formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": example["positive"], "operation": op_id, "label": 1.0})
                    #SIMPY
                    else:
                        formatted_examples.append({"equation1": example["srepr_premise_expression"], "equation2": example["srepr_variable"], "target": example["srepr_positive"], "operation": op_id, "label": 1.0})
                else:
                    if not path in formatted_examples:
                        formatted_examples[path] = []
                    #LATEX
                    if not srepr:
                        formatted_examples[path].append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": example["positive"], "operation": op_id, "label": 1.0})
                    #SIMPY
                    else:
                        formatted_examples[path].append({"equation1": example["srepr_premise_expression"], "equation2": example["srepr_variable"], "target": example["srepr_positive"], "operation": op_id, "label": 1.0})
                #NEGATIVE EXAMPLES
                count_neg = 0
                #LATEX
                if not srepr:
                    for negative in example["negatives"]:
                        if count_neg == neg:
                            break
                        if merge:
                            formatted_examples.append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": negative , "operation": op_id, 'label': -1.0})
                        else:
                            if not path in formatted_examples:
                                formatted_examples[path] = []
                            formatted_examples[path].append({"equation1": example['premise_expression'], "equation2": example['variable'], "target": negative , "operation": op_id, 'label': -1.0})
                        count_neg += 1
                #SIMPY
                else:
                    for negative in example["srepr_negatives"]:
                        if count_neg == neg:
                            break
                        if merge:
                            formatted_examples.append({"equation1": example["srepr_premise_expression"], "equation2": example["srepr_variable"], "target": negative, "operation": op_id, "label": -1.0})
                        else:
                            if not path in formatted_examples:
                                formatted_examples[path] = []
                            formatted_examples[path].append({"equation1": example["srepr_premise_expression"], "equation2": example["srepr_variable"], "target": example["srepr_positive"], "operation": op_id, "label": -1.0})
                        count_neg += 1
        if merge:
            print("Data examples", formatted_examples[:4])
            #split randomly between train, dev, and test set
            dataset = Dataset.from_list(formatted_examples)
            if test_size == 1.0:
                return dataset
            dataset_split = dataset.train_test_split(test_size = test_size)
            return dataset_split
        else:
            datasets = {}
            for path in in dataset_path:
                print("Data examples", formatted_examples[path][:4])
                #split randomly between train, dev, and test set
                datasets[path] = Dataset.from_list(formatted_examples[path])
                if test_size == 1.0:
                    continue
                datasets[path] = datasets[path].train_test_split(test_size = test_size)
            return datasets
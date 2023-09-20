import json
import random
from tqdm import tqdm
from datasets import Dataset
    
class DataModel:

    def __init__(self, neg = 1, do_train = True, do_test = False, tokenize_function_train = None, tokenize_function_eval = None, srepr = False):
        #PROCESS DATA
        self.tokenize_function_train = tokenize_function_train
        self.tokenize_function_eval = tokenize_function_eval
        #training data needs to be processed for operations and setup
        self.train_dataset, self.dev_dataset, self.test_dataset = self.process_dataset(neg = neg, srepr = srepr) #dataset_path = ["data/differentiation.json", "data/integration.json"])
        self.eval_dict = {}
        self.tokenized_train_dataset = self.train_dataset.map(self.tokenize_function_train, batched=False)
        self.tokenized_dev_dataset_cross = self.dev_dataset["cross_operation_negatives"].map(self.tokenize_function_eval, batched=False)
        self.tokenized_dev_dataset_in = self.dev_dataset["in_operation_negatives"].map(self.tokenize_function_eval, batched=False)
        self.tokenized_test_dataset_cross = self.test_dataset["cross_operation_negatives"].map(self.tokenize_function_eval, batched=False)
        self.tokenized_test_dataset_in = self.test_dataset["in_operation_negatives"].map(self.tokenize_function_eval, batched=False)
        self.train_dataset = self.tokenized_train_dataset
        self.eval_dict["dev_set_cross"] = self.tokenized_dev_dataset_cross
        self.eval_dict["dev_set_in"] = self.tokenized_dev_dataset_in
        self.eval_dict["test_set_cross"] = self.tokenized_test_dataset_cross
        self.eval_dict["test_set_in"] = self.tokenized_test_dataset_in

    def process_dataset(self, dataset_path = "data/premises_dataset.json", operations = ["integrate", "differentiate", "add", "minus", "times", "divide"], neg = 1,  training = True, merge = True, test_size = 0.2, srepr = False):
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
        formatted_examples_dev = {"cross_operation_negatives":[], "in_operation_negatives":[]}
        formatted_examples_test = {"cross_operation_negatives":[], "in_operation_negatives":[]}
        d_file = open(dataset_path, 'r')
        d_json = json.load(d_file)
        max_train_examples = 10000
        max_dev_examples = 500
        max_test_examples = 500

        # create a training and dev set entry for each example
        for example in tqdm(d_json[:max_train_examples], desc= dataset_path):
            premise = example["premise"]
            for op in operations:
                #POSITIVE EXAMPLES
                for res in example[op]:
                    #LATEX
                    formatted_examples_train.append({"equation1": premise, "equation2": res["var"], "target": res["res"], "operation": self.opereations_voc_rev[op], "label": 1.0})

        for example in tqdm(d_json[max_train_examples: (max_train_examples + max_dev_examples)], desc= dataset_path):
            premise = example["premise"]
            for op in operations:
                #POSITIVE EXAMPLES
                positive_examples = []
                for res in example[op]:
                    #LATEX
                    positive_examples.append(res["res"])       
                #CROSS-OPERATION NEGATIVE EXAMPLES
                neg_operations = operations
                negative_examples = []
                for op_neg in neg_operations:
                    if op_neg == op:
                        continue
                    for res in example[op_neg]:
                        #LATEX
                        negative_examples.append(res["res"])
                formatted_examples_dev["cross_operation_negatives"].append({"premise": premise, "operation": self.opereations_voc_rev[op], "positive": positive_examples, "negative": negative_examples})
                #IN-OPERATION NEGATIVE EXAMPLES
                num_negs = 5
                #neg_index = random.randint(max_train_examples + max_dev_examples + max_test_examples, len(d_json)-num_negs)
                neg_index = max_train_examples + max_dev_examples + 1
                neg_premises = d_json[neg_index:neg_index+num_negs]
                negative_examples = []
                for neg in neg_premises:
                    for res in neg[op]:
                        #LATEX
                        negative_examples.append(res["res"])
                formatted_examples_dev["in_operation_negatives"].append({"premise": premise, "operation": self.opereations_voc_rev[op], "positive": positive_examples, "negative": negative_examples})

        # create an evaluation entry for each example
        for example in tqdm(d_json[(max_train_examples + max_dev_examples): (max_train_examples + max_dev_examples + max_test_examples)], desc= dataset_path):
            premise = example["premise"]
            for op in operations:
                #POSITIVE EXAMPLES
                positive_examples = []
                for res in example[op]:
                    #LATEX
                    positive_examples.append(res["res"])
                #CROSS-OPERATION NEGATIVE EXAMPLES
                neg_operations = operations
                negative_examples = []
                for op_neg in neg_operations:
                    if op_neg == op:
                        continue
                    for res in example[op_neg]:
                        #LATEX
                        negative_examples.append(res["res"])
                formatted_examples_test["cross_operation_negatives"].append({"premise": premise, "operation": self.opereations_voc_rev[op], "positive": positive_examples, "negative": negative_examples})  
                #IN-OPERATION NEGATIVE EXAMPLES
                num_negs = 5
                #neg_index = random.randint(max_train_examples + max_dev_examples + max_test_examples, len(d_json)-num_negs)
                neg_index = max_train_examples + max_dev_examples + max_test_examples + 1
                neg_premises = d_json[neg_index:neg_index+num_negs]
                negative_examples = []
                for neg in neg_premises:
                    for res in neg[op]:
                        #LATEX
                        negative_examples.append(res["res"])
                formatted_examples_test["in_operation_negatives"].append({"premise": premise, "operation": self.opereations_voc_rev[op], "positive": positive_examples, "negative": negative_examples})
        
        #build datasets 
        dataset_train = Dataset.from_list(formatted_examples_train)
        dataset_dev = {}
        dataset_dev["cross_operation_negatives"] = Dataset.from_list(formatted_examples_dev["cross_operation_negatives"])
        dataset_dev["in_operation_negatives"] = Dataset.from_list(formatted_examples_dev["in_operation_negatives"])
        dataset_test = {}
        dataset_test["cross_operation_negatives"] = Dataset.from_list(formatted_examples_test["cross_operation_negatives"])
        dataset_test["in_operation_negatives"] = Dataset.from_list(formatted_examples_test["in_operation_negatives"])
        
        return dataset_train, dataset_dev, dataset_test


class DataModelMultiStep:

    def __init__(self, neg = 1, operations_voc = None, tokenize_function = None, srepr = False):
        #PROCESS DATA
        self.tokenize_function = tokenize_function
        #training data needs to be processed for operations and setup
        #MAKE OPERATIONS VOCABULARY DYNAMIC 
        #self.operations_voc = DataModel(do_train = False, do_test = False).operations_voc
        self.operations_voc = operations_voc
        self.opereations_voc_rev = {}
        for op_id in self.operations_voc:
            self.opereations_voc_rev[self.operations_voc[op_id]] = op_id
        self.eval_dict = {}
        self.test_dataset_multi_step = self.process_dataset(srepr = srepr)
        self.eval_dict["multi_step"] = self.test_dataset_multi_step.map(self.tokenize_function, batched = False)

    def process_dataset(self, dataset_path = "data/multiple_steps.json", neg = 1, srepr = False):
        #convert dataset into json for dataset loader
        d_file = open(dataset_path, 'r')
        d_json = json.load(d_file)
        # create an entry for each positive example
        tot_formatted_examples = []
        example_id = 0
        print("Processing the dataset...")
        for example in tqdm(d_json, desc= dataset_path):
            step_count = 0
            formatted_example = {}
            formatted_example["idx"] = example_id
            formatted_example["steps"] = {}
            for step in example["steps"]:
                if len(step["negatives"]) == 0:
                    continue
                if not str(step_count) in formatted_example["steps"]:
                    formatted_example["steps"][str(step_count)] = []
                #LATEX
                if not srepr:
                    formatted_example["steps"][str(step_count)].append({"equation1": step['premise_expression'], "equation2": step['variable'], "target": step["positive"], "operation": self.opereations_voc_rev[step["operation_name"]], "label": 1.0})
                #SIMPY
                else:
                    formatted_example["steps"][str(step_count)].append({"equation1": step["srepr_premise_expression"], "equation2": step["srepr_variable"], "target": step["srepr_positive"], "operation": self.opereations_voc_rev[step["operation_name"]], "label": 1.0})
                #NEGATIVE EXAMPLES
                count_neg = 0
                #LATEX
                if not srepr:
                    for negative in step["negatives"]:
                        if count_neg == neg:
                            break
                        formatted_example["steps"][str(step_count)].append({"equation1": step["premise_expression"], "equation2": step['variable'], "target": negative, "operation": self.opereations_voc_rev[step["operation_name"]], "label": -1.0})
                        count_neg += 1
                #SIMPY
                else:
                    for negative in step["srepr_negatives"]:
                        if count_neg == neg:
                            break
                        formatted_example["steps"][str(step_count)].append({"equation1": step["srepr_premise_expression"], "equation2": step["srepr_variable"], "target": negative, "operation": self.opereations_voc_rev[step["operation_name"]], "label": -1.0})
                        count_neg += 1
                step_count += 1
            tot_formatted_examples.append(formatted_example)

        dataset = Dataset.from_list(tot_formatted_examples)
        return dataset
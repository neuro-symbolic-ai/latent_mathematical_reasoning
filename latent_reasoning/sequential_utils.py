#https://github.com/pytorch/examples/blob/main/word_language_model/data.py

import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, min_vars_n = 10):
        self.dictionary = Dictionary()
        # padding
        self.dictionary.add_word("[PAD]")
        # unknown token
        self.dictionary.add_word("[UNK]")
        # preset of variable tokens
        for var_count in range(min_vars_n):
            self.dictionary.add_word("var_" + str(var_count))

    def build_vocabulary(self, path):
        """CHANGE IF WE WANT TO PRE-BUILD THE VOCABULARY"""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split()
                for word in words:
                    self.dictionary.add_word(word)

    def tokenizer(self, sentence:str, max_len:int = 128):
        sentence = sentence.replace("{", " { ").replace("}", " } ").replace("(", " ( ").replace(")", " ) ").replace("^", " ^ ").replace("d", " d ").replace("  ", " ")
        words = sentence.split(" ")
        ids = []
        mask = []
        word_count = 0
        for word in words:
            self.dictionary.add_word(word)
            ids.append(self.dictionary.word2idx[word])
            mask.append(1)
            word_count += 1
            if word_count == max_len:
                break
        for i in range(max_len-word_count):
            ids.append(self.dictionary.word2idx["[PAD]"])
            mask.append(0)
        return {'input_ids': torch.tensor(ids).type(torch.int64), 'input_mask': torch.tensor(mask).type(torch.bool)}

    def var_tokenizer(self, sentences:list, build_vocabulary = True, max_len:int = 128):
        variables = {}
        vars_count = 0
        sep_tokens = ["{", "}", "(", ")", "[", "]", "^", "d", "_", "|"]
        tokenized_sentences = []
        for sentence in sentences:
            for token in sep_tokens:
                sentence = sentence.replace(token, " " + token + " ")
            sentence = sentence.replace("  ", " ")
            words = sentence.split(" ")
            ids = []
            mask = []
            word_count = 0
            found_word = False
            for word in words:
                if not "\\" in word and not word in sep_tokens and word.isalpha():
                    if not word in variables:
                        variables[word] = "var_" + str(vars_count)
                        vars_count += 1
                    if build_vocabulary == True:
                        self.dictionary.add_word(variables[word])
                    if variables[word] in self.dictionary.word2idx:
                        ids.append(self.dictionary.word2idx[variables[word]])
                    else:
                        ids.append(self.dictionary.word2idx["[UNK]"])
                else:
                    if build_vocabulary == True:
                        self.dictionary.add_word(word)
                    if word in self.dictionary.word2idx:
                        ids.append(self.dictionary.word2idx[word])
                    else:
                        ids.append(self.dictionary.word2idx["[UNK"])
                mask.append(0)
                word_count += 1
                if word_count == max_len:
                    break
            for i in range(max_len-word_count):
                ids.append(self.dictionary.word2idx["[PAD]"])
                mask.append(1)
            tokenized_sentences.append({'input_ids': torch.tensor(ids).type(torch.int64), 'input_mask': torch.tensor(mask).type(torch.bool)})
        return tokenized_sentences

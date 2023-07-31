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
    def __init__(self, paths = ["data/differentiation.json", "data/integration.json", "data/addition.json", "data/subtraction.json", "data/multiplication.json", "data/division.json"]):
        self.dictionary = Dictionary()
        self.dictionary.add_word("[PAD]")
        #self.dictionary.add_word("[EOS]")
        #for path in paths:
        #    self.build_vocabulary(path)

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
        return {'input_ids': torch.tensor(ids).type(torch.int64), 'attention_mask': torch.tensor(mask).type(torch.bool)}

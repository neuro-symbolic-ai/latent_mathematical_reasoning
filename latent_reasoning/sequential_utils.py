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
        for path in paths:
            self.build_vocabulary(path)

    def build_vocabulary(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenizer(sentence:str):
        words = sentence.split() + ['<eos>']
        ids = []
        for word in words:
            ids.append(self.dictionary.word2idx[word])
        return torch.tensor(ids).type(torch.int64)
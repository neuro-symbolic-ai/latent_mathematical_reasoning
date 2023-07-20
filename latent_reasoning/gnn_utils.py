import re
from torch.utils.data import Dataset
import numpy as ny
import torch

def match_parentheses(expression):
    pattern = r'(\([^()]+\))|([^()]+)'
    matches = re.finditer(pattern, expression)
    result = []

    for match in matches:
        if match.group(1):
            result.append(re.sub(r', ', '', match.group(1)[1:-1]))  # 括号内的内容
        else:
            result.append(re.sub(r', ', '', match.group(2)))  # 括号外的内容

    return result

def pad_collate(x):
    return x

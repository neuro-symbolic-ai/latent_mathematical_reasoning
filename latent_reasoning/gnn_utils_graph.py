import torch


class Node:
    def __init__(self, value, child):
        self.value = value
        self.child = child


def expression_to_tree(expression):
    expression = expression.replace(' ', '')
    first_left_bracket = expression.find('(')
    if first_left_bracket == -1:
        return Node(expression, [])

    second_left_bracket = expression[first_left_bracket+1:].find('(')
    if second_left_bracket == -1:
        atom_list = ['Symbol', 'Integer', 'Rational']
        for a in atom_list:
            if a in expression:
                return Node(expression, [])

        atoms = expression[first_left_bracket+1:len(expression)-1].split(',')
        sub_expressions = []
        for atom in atoms:
            sub_expressions.append(expression_to_tree(atom))
        return Node(expression[:first_left_bracket], sub_expressions)

    right_index = []
    bracket = []
    for i in range(first_left_bracket + 1, len(expression) - 1):
        if expression[i] == '(':
            bracket.append(1)
        elif expression[i] == ')':
            bracket.pop(-1)
            if not bracket:
                right_index.append(i)
    sub_expressions = []
    start = first_left_bracket + 1
    for i in right_index:
        sub_exp = expression[start:i + 1]
        temp = sub_exp.find('(')
        atoms = sub_exp[:temp].split(',')
        atoms[-1] = atoms[-1] + sub_exp[temp:]
        for atom in atoms:
            sub_expressions.append(expression_to_tree(atom))
        start = i + 2
    if right_index[-1] < len(expression) - 2:
        atoms = expression[start:len(expression) - 1].split(',')
        for atom in atoms:
            sub_expressions.append(expression_to_tree(atom))
    return Node(expression[:first_left_bracket], sub_expressions)


def tree_to_graph(node_dict, node_counter, node_list, edge_list, node, parent_index=None, build_voc = True):
    if node.value not in node_dict.keys():
        if build_voc == False: 
           node.value = "[UNK]"
        else:
            node_dict[node.value] = len(node_dict)
            node_counter[node.value] = 0
    node_list.append(f'{node.value}#{node_counter[node.value]}')
    if parent_index is not None:
        edge_list += [parent_index, len(node_list) - 1]
    node_counter[node.value] += 1
    parent_index = len(node_list) - 1
    for n in node.child:
        tree_to_graph(node_dict, node_counter, node_list, edge_list, n, parent_index, build_voc)


class Corpus(object):
    def __init__(self, max_len=128, build_voc = True):
        self.node_dict = {"[UNK]":0}
        self.max_nodes = max_len
        self.max_edges = 2 * self.max_nodes - 2
        self.build_voc = build_voc

    def tokenizer(self, sreprs):
        graphs = []
        for srepr in sreprs:
            node_counter = {k: 0 for k in self.node_dict.keys()}
            node_list = []
            edge_list = []
            tree_to_graph(self.node_dict, node_counter, node_list, edge_list, expression_to_tree(srepr), build_voc = self.build_voc)
            for i in range(len(node_list)):
                node_list[i] = self.node_dict[node_list[i].split('#')[0]]
            node_list = node_list + [-1] * (self.max_nodes - len(node_list)) if len(node_list) <= self.max_nodes else node_list[:self.max_nodes]
            edge_list = edge_list + [-1] * (self.max_edges - len(edge_list)) if len(edge_list) <= self.max_edges else edge_list[:self.max_edges]
            graphs.append({'nodes': torch.tensor(node_list).type(torch.int64),
                           'edges': torch.tensor(edge_list).type(torch.int64)})
        return graphs

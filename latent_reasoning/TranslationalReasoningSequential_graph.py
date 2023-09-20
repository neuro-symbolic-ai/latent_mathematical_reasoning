from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from sentence_transformers import util
from latent_reasoning.Encoders_graph import *


class TransLatentReasoningSeq(nn.Module):
    def __init__(self, n_tokens, n_operations, device, model_type="transformer", loss_type="mnr"):
        super(TransLatentReasoningSeq, self).__init__()
        self.device = device
        self.loss_type = loss_type
        
        # Load encoder model
        if model_type == "transformer":
            self.encoder = TransformerModel(n_tokens, device).to(device)
        elif model_type == "rnn":
            self.encoder = RNNModel(n_tokens, device).to(device)
        elif model_type == "cnn":
            self.encoder = CNNModel(n_tokens, device).to(device)
        elif model_type[:3] == "gnn":
            self.encoder = GNNModel(n_tokens, device, gnn_type=model_type).to(device)
        
        self.dim = self.encoder.ninp
        self.linear = nn.Linear(self.dim, self.dim).to(device) #*3
        
        self.ov = nn.Embedding(n_operations, self.dim, device = device)
        self.ov.weight.data = (torch.randn((n_operations, self.dim), dtype=torch.float, device = device))
        self.Wo = torch.nn.Parameter(torch.tensor(np.random.uniform(-1,1, (n_operations, self.dim)), dtype=torch.float, requires_grad=True, device=self.device))
        
        # MSE Loss
        if self.loss_type == "mse":
            self.similarity_fct = nn.functional.cosine_similarity
            self.loss_function = nn.MSELoss()
        # Multiple Negagitve Ranking Loss
        elif self.loss_type == "mnr":
            self.similarity_fct = util.cos_sim #compute similarity for each possible pair in (a, b)
            self.loss_function = nn.CrossEntropyLoss()
        
    def forward(self, equation1, equation2, target_equation, operation, labels):
        # GET OPERATION EMBEDDINGS
        operation = operation.to(self.device)
        Wo = self.Wo[operation]
        ov = self.ov(operation)

        # ENCODE EQUATIONS
        equation1 = {k: v.to(self.device) for k, v in equation1.items()}
        embeddings_eq1 = self.encoder(equation1)
        
        #equation2 = {k: v.to(self.device) for k, v in equation2.items()}
        #embeddings_eq2 = self.encoder(equation2)

        target_equation = {k: v.to(self.device) for k, v in target_equation.items()} 
        embeddings_target = self.encoder(target_equation)

        #features = torch.cat([embeddings_eq1, embeddings_eq2, embeddings_eq1 * embeddings_eq2], 1)
        features = embeddings_eq1
        embeddings_output = self.linear(features)

        #TRANSLATIONAL MODEL
        embeddings_output = embeddings_output * Wo
        embeddings_target = embeddings_target + ov

        #COMPUTE LOSS
        scores = self.similarity_fct(embeddings_output, embeddings_target)

        if self.loss_type == "mse":
            labels = labels.to(self.device)
        elif self.loss_type == "mnr":
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
            
        loss = self.loss_function(scores, labels)

        return loss, scores, labels


    def inference_step(self, prev_step, equation1, equation2, target_equation, operation, labels):
        # GET OPERATION EMBEDDINGS
        operation = operation.to(self.device)
        Wo = self.Wo[operation]
        ov = self.ov(operation)

        # ENCODE EQUATIONS
        if equation1 != None:
            equation1 = {k: v.to(self.device) for k, v in equation1.items()}
            embeddings_eq1 = self.encoder(equation1)
        else:
            embeddings_eq1 = prev_step
        
        #equation2 = {k: v.to(self.device) for k, v in equation2.items()}
        #embeddings_eq2 = self.encoder(equation2)

        target_equation = {k: v.to(self.device) for k, v in target_equation.items()} 
        embeddings_target = self.encoder(target_equation)

        #features = torch.cat([embeddings_eq1, embeddings_eq2, embeddings_eq1 * embeddings_eq2], 1)
        features = embeddings_eq1
        embeddings_output = self.linear(features)

        #TRANSLATIONAL MODEL
        embeddings_output = embeddings_output * Wo
        embeddings_target = embeddings_target + ov

        #COMPUTE SCORES
        scores = nn.functional.cosine_similarity(embeddings_output, embeddings_target)

        return scores, embeddings_output - ov

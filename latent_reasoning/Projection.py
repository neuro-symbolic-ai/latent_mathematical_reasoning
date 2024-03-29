import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import util
from latent_reasoning.Encoders import *


class LatentReasoning(nn.Module):
    def __init__(self, n_tokens, n_operations, device, model_type = "transformer", one_hot = False, loss_type = "mnr"):
        super(LatentReasoning, self).__init__()
        self.device = device
        self.one_hot = one_hot
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
        
        if self.one_hot:
            self.linear = nn.Linear(n_operations + self.dim, self.dim).to(device)
            self.ov = F.one_hot(torch.arange(n_operations)).to(device)
        else:
            self.linear = nn.Linear(self.dim*2, self.dim).to(device)
            self.ov = nn.Embedding(n_operations, self.dim, device = device)
            self.ov.weight.data = (torch.randn((n_operations, self.dim), dtype=torch.float, device = device))
        # MSE Loss
        if self.loss_type == "mse":
            self.similarity_fct = nn.functional.cosine_similarity
            self.loss_function = nn.MSELoss()
        # Multiple Negagitve Ranking Loss
        elif self.loss_type == "mnr":
            self.similarity_fct = util.cos_sim #compute similarity for each possible pair in (a, b)
            self.loss_function = nn.CrossEntropyLoss()

        
    def forward(self, premise, target_expression, operation, labels):
        # GET OPERATION EMBEDDINGS
        operation = operation.to(self.device)
        if self.one_hot:
            ov = self.ov[operation]
        else:
            ov = self.ov(operation)

        # ENCODE EQUATIONS
        premise = {k: v.to(self.device) for k, v in premise.items()}
        embeddings_premise = self.encoder(premise)

        target_expression = {k: v.to(self.device) for k, v in target_expression.items()} 
        embeddings_target = self.encoder(target_expression)
        
        features = torch.cat([ov, embeddings_premise], 1)
        embeddings_output = self.linear(features)

        #COMPUTE LOSS
        scores = self.similarity_fct(embeddings_output, embeddings_target)

        if self.loss_type == "mse":
            labels = labels.to(self.device)
        elif self.loss_type == "mnr":
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
            
        loss = self.loss_function(scores, labels)

        return loss, scores, labels


    def inference_step(self, prev_step, premise, target_expression, operation):
        # GET OPERATION EMBEDDINGS
        if operation != None:
            operation = operation.to(self.device)
            if self.one_hot:
                ov = self.ov[operation]
            else:
                ov = self.ov(operation)

        # ENCODE EQUATIONS
        if premise != None:
            premise = {k: v.to(self.device) for k, v in premise.items()}
            embeddings_premise = self.encoder(premise)
        else:
            embeddings_premise = prev_step

        target_expression = {k: v.to(self.device) for k, v in target_expression.items()} 
        embeddings_target = self.encoder(target_expression)

        if operation != None:
            features = torch.cat([ov, embeddings_premise], 1)
            embeddings_output = self.linear(features)
        else:
            embeddings_output = embeddings_premise

        #COMPUTE SCORES
        scores = nn.functional.cosine_similarity(embeddings_output, embeddings_target)

        return scores, embeddings_output
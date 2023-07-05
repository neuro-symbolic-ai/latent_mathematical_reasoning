from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import numpy as np


class TransLatentReasoning(nn.Module):
    def __init__(self, model_name, n_operations, device):
        super(TransLatentReasoning, self).__init__()
        self.device = device
        # Load model from HuggingFace Hub
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.dim = 768
        #self.encoder_transf = AutoModel.from_pretrained(model_name).to(device)
        self.linear = nn.Linear(self.dim*3, self.dim).to(device)
        
        self.ov = nn.Embedding(n_operations, self.dim, padding_idx=0)
        self.ov.weight.data = (1e-3 * torch.randn((n_operations, self.dim), dtype=torch.float, device = device))
        self.Wo = torch.nn.Parameter(torch.tensor(np.random.uniform(-1,1, (n_operations, self.dim)), dtype=torch.float, requires_grad=True, device=self.device))
        
        self.similarity_fct = nn.functional.cosine_similarity
        self.loss_function = nn.MSELoss()

        
    def forward(self, equation1, equation2, target_equation, operation, labels):
        # GET OPERATION EMBEDDINGS
        Wo = self.Wo[operation]
        ov = self.ov.weight[operation]

        # ENCODE EQUATIONS
        equation1 = {k: v.to(self.device) for k, v in equation1.items()}
        embeddings_eq1 = self.encoder(**equation1)
        embeddings_eq1 = self.mean_pooling(embeddings_eq1, equation1['attention_mask'])
        
        equation2 = {k: v.to(self.device) for k, v in equation2.items()}
        embeddings_eq2 = self.encoder(**equation2)
        embeddings_eq2 = self.mean_pooling(embeddings_eq2, equation2['attention_mask'])

        target_equation = {k: v.to(self.device) for k, v in target_equation.items()} 
        embeddings_target = self.encoder(**target_equation)
        embeddings_target = self.mean_pooling(embeddings_target, target_equation['attention_mask'])

        features = torch.cat([embeddings_eq1, embeddings_eq2, embeddings_eq1 * embeddings_eq2], 1)
        embeddings_output = self.linear(features)

        #TRANSLATIONAL MODEL
        embeddings_output = embeddings_output * Wo
        embeddings_target = embeddings_target + ov

        #COMPUTE LOSS
        scores = self.similarity_fct(embeddings_output, embeddings_target)
        labels = labels.to(self.device)
        loss = self.loss_function(scores, labels)

        return loss, scores, labels, embeddings_output

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


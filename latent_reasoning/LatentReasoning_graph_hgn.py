# from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import RGCNConv
#from sentence_transformers import SentenceTransformer, util
import torch
from torch import nn
import numpy as np

class GraphLatentReasoning_RGCN(nn.Module):
    def __init__(self, model_name, n_operations, device, sym_dict = {'log':0, 'Mul':1, 'exp':2, 'Add':3, 'Symbol':4, 'Pow':5, 'cos':6, 'Integer':7, 'sin':8}, input_dim = 768, hidden_dim = 768, feat_drop = 0.5, heads = 8, num_layers = 2):
        super(GraphLatentReasoning_RGCN, self).__init__()
        # Load model from HuggingFace Hub
        #'sentence-transformers/bert-base-nli-mean-tokens'
        self.device = device
        self.encoder = nn.ModuleList()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(feat_drop)
        self.encoder.append(RGCNConv(
            in_channels=int(hidden_dim),out_channels=int(hidden_dim),num_relations=2).to(device))
        self.num_layers = num_layers
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.encoder.append(RGCNConv(in_channels=int(hidden_dim),out_channels=int(hidden_dim),num_relations=2).to(device))
       #self.encoder_transf = AutoModel.from_pretrained(model_name).to(device)
        self.linear = nn.Linear(768*3, 768).to(device)
        self.similarity_fct = nn.functional.cosine_similarity #util.cos_sim
        self.loss_function = nn.MSELoss() #nn.BCEWithLogitsLoss() #nn.MSELoss()
        ## operation
        self.dim = hidden_dim
        self.Wo = torch.nn.Parameter(torch.tensor(np.random.uniform(-1,1, (n_operations, self.dim)), dtype=torch.float, requires_grad=True, device=self.device))
        self.ov = nn.Embedding(n_operations, self.dim, padding_idx=0)
        self.ov.weight.data = (1e-3 * torch.randn((n_operations, self.dim), dtype=torch.float, device = device))
        ## symbol
        self.symbol_dict = sym_dict
        self.symbol = torch.nn.Parameter(torch.tensor(np.random.uniform(-1,1, (len(self.symbol_dict), self.dim)), dtype=torch.float, requires_grad=True, device=self.device))

    def generate_edge_emb(self, node_l):
        edge_emb = []
        for node in node_l:
            if node in self.symbol_dict:
                edge_emb.append(self.symbol[self.symbol_dict[node]])
            else:
                edge_emb.append(torch.tensor(np.random.uniform(-1,1, (self.dim)), dtype=torch.float, requires_grad=True, device=self.device))
        edge_emb = torch.stack(edge_emb, dim=0).to(self.device)
        return edge_emb

    def forward(self, equation1, equation2, target_equation, operation, labels):
        # Tokenize sentences
        #encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        Wo = self.Wo[operation-1]
        ov = self.ov.weight[operation-1]
        # Compute embeddings
        # equation1 = {k: v.to(self.device) for k, v in equation1.items()}
        edge_emb_eq1 = self.generate_edge_emb(equation1["node_list"]).to(self.device)
        for l in range(self.num_layers):
            edge_emb_eq1 = self.encoder[l](edge_emb_eq1, equation1["edge_index"].to(self.device), equation1["edge_type"].to(self.device))
            edge_emb_eq1 = self.dropout(self.activation(edge_emb_eq1.flatten(1)))
        embeddings_eq1 = torch.mean(edge_emb_eq1, dim=0).to(self.device)
        
        embeddings_eq2 = edge_emb_eq1[equation1["var_idx"]].to(self.device)

        edge_emb_tar = self.generate_edge_emb(target_equation["node_list"]).to(self.device)
        for l in range(0,self.num_layers):
            edge_emb_tar = self.encoder[l](edge_emb_tar, target_equation["edge_index"].to(self.device), target_equation["edge_type"].to(self.device)).flatten(1)
            edge_emb_tar = self.dropout(self.activation(edge_emb_tar))
        embeddings_target = torch.mean(edge_emb_tar, dim=0).to(self.device)

        features = torch.cat([embeddings_eq1, embeddings_eq2, embeddings_eq1 * embeddings_eq2], 0)
        embeddings_output = self.linear(features)

        #TRANSLATIONAL MODEL
        embeddings_output = embeddings_output * Wo
        embeddings_target = embeddings_target + ov

        #print(embeddings_output.size(), embeddings_target.size())
        scores = self.similarity_fct(embeddings_output.unsqueeze(0), embeddings_target.unsqueeze(0))
        # for multiple negative ranking
        # labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        labels = labels.to(self.device)
        loss = self.loss_function(scores, labels)

        return loss, scores, labels, embeddings_output

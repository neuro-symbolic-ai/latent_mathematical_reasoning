import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ============================================ RNN ENCODER ================================================
class RNNModel(nn.Module):

    def __init__(self, ntoken, device, rnn_type = "LSTM", ninp = 768, nhid = 768, nlayers = 2, dropout=0.1):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.ninp = ninp
        self.device = device
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx = 0)
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first = True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.fc = nn.Linear(nhid, ninp)
        
        #self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def forward(self, input):
        seq_lengths = input["input_mask"].size(1) - input["input_mask"].sum(1)
        emb = self.encoder(input['input_ids'])
        pack = nn.utils.rnn.pack_padded_sequence(emb, seq_lengths.to("cpu"), batch_first=True, enforce_sorted = False)
        _ , hidden = self.rnn(pack)
        output = self.fc(self.drop(hidden[0][-1]))
        return output



# ============================================ CNN ENCODER ================================================
class CNNModel(nn.Module):
    """An 1D Convulational Neural Network for Sentence Classification.
       https://colab.research.google.com/drive/1b7aZamr065WPuLpq9C4RU6irB59gbX_K#scrollTo=ejGLw8TKViBY
    """
    def __init__(self, ntoken, device, ninp = 768, nhid = 768, filter_sizes=[3, 4, 5], num_filters=[100, 100, 100], dropout=0.1):
        super(CNNModel, self).__init__()            
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.ninp = ninp
        self.device = device
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=self.ninp,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), nhid)
        self.dropout = nn.Dropout(p=dropout)
        self.nhid = nhid

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def forward(self, input):
        input_ids = input["input_ids"]
        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.encoder(input_ids)
        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)
        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            for x_conv in x_conv_list]
        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)
        # Compute logits. Output shape: (b, nhid)
        output = self.fc(self.dropout(x_fc))
        return output


# ========================================= TRANSFORMER ENCODER ==============================================
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, seq_length, embed dim]
            output: [batch size, seq_length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and mean pooling.
       https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, ntoken, device, ninp = 768, nhead = 8, nhid = 2048, nlayers = 6, dropout=0.1):

        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.device = device
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=0)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(attention_mask.size(1) - input_mask_expanded.sum(1), min=1e-9)

    def forward(self, src):
        src_mask = src["input_mask"]
        src = src["input_ids"]
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask = src_mask)
        output = self.mean_pooling(output, src_mask)
        return output


#============================================================== GNN ENCODERS ======================================================================
class GNNModel(nn.Module):
    """Container module with an encoder, a transformer module, and mean pooling.
       https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, ntoken, device, gnn_type='gnn_GAT_direct', ninp=768, nhead=8, nhid=768, nlayers=6, dropout=0.1):
        super(GNNModel, self).__init__()
        from torch_geometric.nn import GATConv, GCNConv, GraphSAGE, TransformerConv
        self.device = device
        self.node_embedding = nn.Embedding(ntoken, ninp)
        self.gnn = nn.ModuleList()
        assert len(gnn_type.split('_')) == 3
        _, self.layer_type, self.direct = gnn_type.split('_')
        assert self.direct in ['direct', 'undirect']
        if self.layer_type == 'GAT':
            self.gnn.append(GATConv(in_channels=ninp, out_channels=int(nhid / nhead), heads=nhead, dropout=dropout))
            # hidden layers
            for _ in range(1, nlayers):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gnn.append(GATConv(in_channels=nhid, out_channels=int(nhid / nhead), heads=nhead, dropout=dropout))
        elif self.layer_type == 'GCN':
            self.gnn.append(GCNConv(in_channels=ninp, out_channels=nhid, dropout=dropout))
            for _ in range(1, nlayers):
                self.gnn.append(GCNConv(in_channels=nhid, out_channels=nhid, dropout=dropout))
        elif self.layer_type == 'GraphSAGE':
            self.gnn.append(GraphSAGE(
                in_channels=ninp, hidden_channels=nhid, num_layers=nlayers, out_channels=nhid, dropout=dropout))
        else:
            raise ValueError(f'{self.layer_type} is not supported, please choose from GAT, GCN, or GraphSAGE for gnn_type.')

        self.ninp = ninp

    def all_to_one_graph(self, src):
        nodes_all = []
        edges_all = []
        pos = [0]
        for i in range(src['nodes'].shape[0]):
            edges = (src['edges'][i] + len(nodes_all)).cpu().numpy().tolist()
            if len(nodes_all) - 1 in edges:
                edges = edges[:edges.index(len(nodes_all) - 1)]
            edges_all += edges

            nodes = src['nodes'][i].cpu().numpy().tolist()
            if -1 in nodes:
                nodes = nodes[:nodes.index(-1)]
            if edges:
                assert max(edges) < len(nodes) + len(nodes_all)
            nodes_all += nodes

            pos.append(len(nodes_all))

        nodes_all = torch.tensor(nodes_all).to(self.device)
        edges_all = torch.tensor(edges_all, dtype=torch.int64).reshape(-1, 2).T.to(self.device)
        if self.direct == 'undirect':
            edges_all = torch.cat([edges_all, edges_all.flip(dims=(0,))], dim=1)
        return nodes_all, edges_all, pos

    def forward(self, src):
        nodes, edge_index, pos = self.all_to_one_graph(src)
        output = []
        x = self.node_embedding(nodes)
        first_layer = True
        for layer in self.gnn:
            if not first_layer:
                # apply non linearity
                x = F.relu(x)
            x = layer(x=x, edge_index=edge_index)
            first_layer = False
        for i in range(len(pos) - 1):
            output.append(torch.mean(x[pos[i]:pos[i + 1]], dim=0))
        output = torch.stack(output, dim=0)
        return output

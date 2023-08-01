from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
import numpy as np
import math
from torch.autograd import Variable

class TransLatentReasoningSeq(nn.Module):
    def __init__(self, n_tokens, n_operations, device, model_type = "RNN"):
        super(TransLatentReasoningSeq, self).__init__()
        self.device = device
        # Load encoder model
        if model_type == "transformer":
            self.encoder = TransformerModel(n_tokens).to(device)
        elif model_type == "rnn":
            self.encoder = RNNModel(n_tokens, device).to(device)
        self.dim = self.encoder.ninp
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
        #equation1 = equation1.to(self.device)
        embeddings_eq1 = self.encoder(equation1)
        
        equation2 = {k: v.to(self.device) for k, v in equation2.items()}
        #equation2 = equation2.to(self.device)
        embeddings_eq2 = self.encoder(equation2)

        target_equation = {k: v.to(self.device) for k, v in target_equation.items()} 
        #target_equation = target_equation.to(self.device)
        embeddings_target = self.encoder(target_equation)

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


    def inference_step(self, prev_step, equation1, equation2, target_equation, operation, labels):
        # GET OPERATION EMBEDDINGS
        Wo = self.Wo[operation]
        ov = self.ov.weight[operation]

        # ENCODE EQUATIONS
        if equation1 != None:
            equation1 = {k: v.to(self.device) for k, v in equation1.items()}
            embeddings_eq1 = self.encoder(equation1)
        else:
            embeddings_eq1 = prev_step
        
        equation2 = {k: v.to(self.device) for k, v in equation2.items()}
        embeddings_eq2 = self.encoder(equation2)

        target_equation = {k: v.to(self.device) for k, v in target_equation.items()} 
        embeddings_target = self.encoder(target_equation)

        features = torch.cat([embeddings_eq1, embeddings_eq2, embeddings_eq1 * embeddings_eq2], 1)
        embeddings_output = self.linear(features)

        #TRANSLATIONAL MODEL
        embeddings_output = embeddings_output * Wo
        embeddings_target = embeddings_target + ov

        #COMPUTE SCORES
        scores = self.similarity_fct(embeddings_output, embeddings_target)

        return scores, embeddings_output - ov


#RNN ENCODER
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, device, rnn_type = "LSTM", ninp = 512, nhid = 512, nlayers = 3, dropout=0.1):
        super(RNNModel, self).__init__()
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.ninp = ninp
        self.device = device
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first = True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError as e:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""") from e
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def forward(self, input):
        #h_0 = Variable(torch.zeros(self.nlayers, input["input_ids"].size()[0], self.nhid).to(self.device))
        #c_0 = Variable(torch.zeros(self.nlayers, input["input_ids"].size()[0], self.nhid).to(self.device))
        seq_lengths = input["input_mask"].sum(1).to("cpu")
        emb = self.drop(self.encoder(input['input_ids']))
        pack = nn.utils.rnn.pack_padded_sequence(emb, seq_lengths, batch_first=True, enforce_sorted = False)
        output, hidden = self.rnn(pack)
        #output = self.drop(hidden[0][-1])
        return hidden[0][-1]

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

#TRANSFORMER ENCODER
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
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

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and mean pooling.
       https://github.com/pytorch/examples/blob/main/word_language_model/model.py
    """

    def __init__(self, ntoken, ninp = 512, nhead = 8, nhid = 512, nlayers = 6, dropout=0.1):
        super(TransformerModel, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except BaseException as e:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or '
                              'lower.') from e
        self.model_type = 'Transformer'
        #self.src_mask = None
        self.encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first = True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, src, has_mask=True):
        #if has_mask:
        #    device = src.device
        #    if self.src_mask is None or self.src_mask.size(0) != len(src):
        #        mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #        self.src_mask = mask
        #else:
        #    self.src_mask = None
        src_mask = src["input_mask"]
        src = src["input_ids"]
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask = src_mask)
        output = self.mean_pooling(output, src_mask)
        return output

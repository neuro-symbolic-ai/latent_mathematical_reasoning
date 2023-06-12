from transformers import AutoTokenizer, AutoModel
#from sentence_transformers import SentenceTransformer, util
import torch


class LatentReasoning(nn.Module):
    def __init__(self, model_name):
        super(LatentReasoning, self).__init__()
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #'sentence-transformers/bert-base-nli-mean-tokens'
        self.encoder = AutoModel.from_pretrained(model_name)
        self.linear = nn.Linear(768*2, 768, bias=True)
        self.similarity_fct = torch.nn.functional.cosine_similarity #util.cos_sim
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        
    def forward(self, equation1, equation2, target_equation, labels):
        # Tokenize sentences
        #encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute embeddings
        embeddings_eq1 = self.encoder(**equation1)
        embeddings_eq1 = self.mean_pooling(embeddings_eq1, equation1['attention_mask'])
        
        embeddings_eq2 = self.encoder(**equation2)
        embeddings_eq2 = self.mean_pooling(embeddings_eq2, equation2['attention_mask'])

        embeddings_target = self.encoder(**target_equation)
        embeddings_target = self.mean_pooling(embeddings_target, target_equation['attention_mask'])

        features = torch.cat([embeddings_eq1, embeddings_eq2], 1)
        embeddings_output = self.linear(features)
        scores = self.similarity_fct(embeddings_output, embeddings_target)

        # for multiple negative ranking
        # labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        loss = self.cross_entropy_loss(scores, labels)

        return loss, scores, embeddings_output

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


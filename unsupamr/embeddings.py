# STL
# 3rd Party
import torch
# Local
from .utils import VocabExt


def expand_lm_head(head: torch.nn.Linear, vocab: VocabExt) -> torch.nn.Linear:
    
    old_num_tokens, hidden_dim = head.weight.shape
    new_num_tokens = len(vocab)
    
    new_head = torch.nn.Linear(hidden_dim, new_num_tokens, bias=False)
    
    with torch.no_grad():
        new_head.weight[:old_num_tokens] = head.weight  # Copy old weightss
    
    return new_head

def expand_embedding(embedding: torch.nn.Embedding, vocab: VocabExt) -> torch.nn.Embedding:
    """
    TODO: @Ravi
    """
    return embedding

def mult_embedding_lookup(prob_dists: torch.Tensor, embedding: torch.nn.Embedding):
    return prob_dists @ embedding.weight
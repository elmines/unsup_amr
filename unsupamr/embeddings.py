# STL
# 3rd Party
import torch
# Local
from .vocab import VocabExt

def expand_lm_head(head: torch.nn.Linear, vocab: VocabExt) -> torch.nn.Linear:
    """
    TODO: @Divyam
    """
    return head

def expand_embedding(embedding: torch.nn.Embedding, vocab: VocabExt) -> torch.nn.Embedding:
    """
    TODO: @Ravi
    """
    return embedding

def mult_embedding_lookup(prob_dists: torch.Tensor, embedding: torch.nn.Embedding):
    """
    TODO: @Ravi
    """
    return prob_dists @ embedding.weight
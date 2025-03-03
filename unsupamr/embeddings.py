# STL
# 3rd Party
import torch
# Local
from .utils import VocabExt

def expand_lm_head(head: torch.nn.Linear, vocab: VocabExt) -> torch.nn.Linear:
    """
    TODO: @Divyam
    """
    return head

def expand_embedding(embedding: torch.nn.Embedding, vocab: VocabExt) -> torch.nn.Embedding:
    """
    Expand the embedding matrix with custom Vocab columns

    Args:
        embedding: Embedding matrix from torch object
        vocab: vocab matrix as the output of T2A
    """
    
    return embedding

def mult_embedding_lookup(prob_dists: torch.Tensor, embedding: torch.nn.Embedding):
    return prob_dists @ embedding.weight
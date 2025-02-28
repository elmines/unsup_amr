# STL
from typing import Dict, Any
# 3rd Party
import torch
# Local
from .vocab import VocabExt

def expand_embedding(embedding: torch.nn.Embedding, vocab: VocabExt) -> torch.nn.Embedding:
    """
    TODO: @Ravi
    """
    pass

def mult_embedding_lookup(prob_dists: torch.Tensor, embedding: torch.nn.Embedding):
    """
    TODO: @Ravi
    """
    pass
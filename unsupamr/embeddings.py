# STL
# 3rd Party
import json
import torch
# Local
from .utils import VocabExt


def expand_lm_head(head: torch.nn.Linear, vocab: VocabExt) -> torch.nn.Linear:
    
    old_num_tokens, hidden_dim = head.weight.shape
    new_num_tokens = old_num_tokens + len(vocab.amr_symbols)
    
    new_head = torch.nn.Linear(hidden_dim, new_num_tokens, bias=False)
    
    with torch.no_grad():
        new_head.weight[:old_num_tokens] = head.weight  # Copy old weightss
    
    return new_head

def expand_embedding(embedding: torch.nn.Embedding, vocab: VocabExt) -> torch.nn.Embedding:
    """
    Expand the embedding matrix with custom Vocab columns

    Args:
        embedding: Embedding matrix from torch object
        vocab: vocab matrix as the output of T2A
    """
    amr_entries = vocab.amr_symbols
    old_weight_mat = embedding.weight

    old_vocab_size, embedding_size = old_weight_mat.shape
    new_vocab_size = old_vocab_size + len(amr_entries)

    new_embedding = torch.nn.Embedding(new_vocab_size, embedding_size)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_weight_mat
        for entry in amr_entries:
            if entry.embed_id is not None:
                new_embedding.weight[entry.id] = old_weight_mat[entry.embed_id]    
    return new_embedding

def mult_embedding_lookup(prob_dists: torch.Tensor, embedding: torch.nn.Embedding):
    return prob_dists @ embedding.weight
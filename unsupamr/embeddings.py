# STL
# 3rd Party
import json
import torch
# Local
from .utils import VocabExt

def mask_lm_head(head: torch.nn.Linear, vocab: VocabExt) -> torch.nn.Linear:
    old_num_tokens, hidden_dim = head.weight.shape
    new_head = torch.nn.Linear(hidden_dim, vocab.new_vocab_size, bias=False)
    with torch.no_grad():
        weight_mat = new_head.weight
        weight_mat[:old_num_tokens] = head.weight
        weight_mat[old_num_tokens:] = 0.
    return new_head

def expand_lm_head(head: torch.nn.Linear, vocab: VocabExt, load_old_head_weights: bool = True) -> torch.nn.Linear:
    
    old_num_tokens, hidden_dim = head.weight.shape
    
    new_head = torch.nn.Linear(hidden_dim, vocab.new_vocab_size, bias=False)
    
    if load_old_head_weights:
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
    new_vocab_size = vocab.new_vocab_size

    new_embedding = torch.nn.Embedding(new_vocab_size, embedding_size)
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_weight_mat
        for entry in amr_entries:
            if entry.embed_id is not None:
                new_embedding.weight[entry.id] = old_weight_mat[entry.embed_id]    
    return new_embedding

def mult_embedding_lookup(prob_dists: torch.Tensor, embedding: torch.nn.Embedding, smoothing=0):
    if smoothing > 0:
        smoothed_dists = (1 - smoothing) * prob_dists + smoothing * (1 / prob_dists.shape[-1])
        return smoothed_dists @ embedding.weight
    return prob_dists @ embedding.weight
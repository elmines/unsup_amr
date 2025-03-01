# STL
from typing import List
# 3rd Party
import torch
# Local
from .utils import VocabExt

class NextTokens:

    def __init__(self,
                vocab_ext: VocabExt,
                pad_id: int,
                eos_id: int):
        """
        TODO: @Siva

        Need to go through the VocabExt object.
        The highest integer ID in vocab_ext.amr_symbols is the last vocabulary ID
        (can use that to get vocabulary size).

        Any vocab ID that is not an AMR symbol and not in vocab_ext.pruned_english
        should ALWAYS be masked.
        """

        self.context: List[int] = []
        self.vocab_ext = vocab_ext
        self.pad_id = pad_id
        self.eos_id = eos_id

    def record(self, prediction: int):
        """
        TODO: @Siva
        Records a prediction and adds it to the context
        """

    def next_tokens(self) -> torch.Tensor:
        """
        TODO: @Siva

        - Needs to only allow self.pad_id if self.eos_id is in the context

        - If you detect something invalid in the sequence, just only allow self.pad_id from that point onward

        - Divyam's code will only consider a sequence 'finished' if it ends with padding or eos
        """
        # Just a pass-through for now
        return torch.zeros([1], dtype=torch.long)
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
        pass

        self.context: List[int] = []
        self.vocab_ext = vocab_ext
        self.pad_id = pad_id
        self.eos_id = eos_id

    def record(self, prediction: int):
        """
        Records a prediction and adds it to the context
        """

    def next_tokens(self) -> torch.Tensor:
        """

        - Needs to only allow self.pad_id if self.eos_id has already been returned

        - If you detect something invalid in the sequence, just only allow self.pad_id from that point onward

        - Divyam's code will only consider a sequence 'finished' if it ends with padding or eos
        """
        # Just a pass-through for now
        return 0
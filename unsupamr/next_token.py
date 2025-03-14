from __future__ import annotations
from typing import List, Set, Dict, Optional
from collections import defaultdict
import json 
import torch
import math 
from collections import deque 
# Local
from .utils import VocabExt
from .constants import AmrCategory
"""
Since we’re doing breadth-first search, we can enforce the following constraints:

For a given verb node, we must predict all its argument edges in sequence. That is, if the verb supports :ARG0-3 but we’re only predicting :ARG1 and :ARG3, we should have a substring in the output that looks something like “<R6> :ARG1 <R12> :ARG3 <R9>”

A verb node should have at least one argument
"""

class NextTokensFactory:
    """
    Class that already has necessary preprocessing done for the NextTokens object
    """

    def __init__(self, vocab: VocabExt):
        self.vf = {ent.id:ent.args for ent in vocab.amr_symbols if ent.category == AmrCategory.FRAME}
        self.label_idxs = {ent.id for ent in vocab.amr_symbols if ent.category == AmrCategory.LABEL}
        self.arg_idxs = {ent.id for ent in vocab.amr_symbols if ent.category == AmrCategory.ARG} # Only args, no inverse args yet
        self.concept_idxs = vocab.pruned_english
        self.start_label_idx = next(ent.id for ent in vocab.amr_symbols if ent.token == "<R0>")
        self.stop_token_idx = next(ent.id for ent in vocab.amr_symbols if ent.category == AmrCategory.STOP)
        self.end_of_sequence_idx = vocab.eos_id
        self.pad_idx = vocab.pad_id
        self.vocab_size = max(ent.id for ent in vocab.amr_symbols) + 1

        assert self.start_label_idx is not None
        assert self.stop_token_idx is not None

    def build(self) -> NextTokens:
        return NextTokens(
            vf=self.vf,
            label_idxs=self.label_idxs,
            arg_idxs=self.arg_idxs,
            concept_idxs=self.concept_idxs,
            start_label_idx=self.start_label_idx,
            stop_token_idx=self.stop_token_idx,
            end_of_sequence_idx=self.end_of_sequence_idx,
            pad_idx=self.pad_idx,
            vocab_size=self.vocab_size
        )



class NextTokens:
    def __init__(self,
                vf,
                label_idxs,
                arg_idxs,
                concept_idxs, 
                start_label_idx,
                stop_token_idx,
                end_of_sequence_idx,
                pad_idx,
                vocab_size
                #  verb_frames, verb_idxs, label_idxs, arg_idxs, concept_idxs, vocab
                ):
        """
        verb_idxs: A hashset of vocab indices corresponding to verbs.
        label_idxs: A hashset of vocab indices corresponding to labels.
        arg_idxs: A hashset of vocab indices corresponding to argument edges.
        concept_idxs: A hashset of vocab indices corresponding to concept nodes.
        """

        self.vf = vf
        self.verb_idxs = set(self.vf.keys())  # FIXME: this is redundant with self.vf. Don't need.
        self.label_idxs = label_idxs
        self.arg_idxs = arg_idxs 
        self.concept_idxs = concept_idxs
        self.start_label_idx = start_label_idx
        self.stop_token_idx = stop_token_idx
        self.end_of_sequence_idx = end_of_sequence_idx
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

        # Precomputed constants
        # FIXME: Don't recompute these for every new NextTokens object we make
        self.__pad_mask = torch.where(torch.arange(0, vocab_size) == self.pad_idx, 0., -math.inf)
        self.__error_mask = self.__pad_mask

        # Allow any verb frame
        self.__vf_mask = torch.tensor([
            0. if (k in self.vf) else -math.inf for k in range(vocab_size)
        ])
        self.__vf_or_concept_mask = torch.tensor([
            0. if (k in self.concept_idxs or k in self.vf) else -math.inf for k in range(vocab_size)
        ])
        
        # State variables
        self.context = []  # Track tokens predicted so far
        self.args_used = {}  # (seq_idx -> [vocab_ids]) Track which verbs have had which arguments populated
        self.tail_nodes = defaultdict(lambda: set()) # (<Rn> as vocab id -> [<Rn>s as vocab ids]) Track which nodes are already a target of a given frame

        self.next_label_id = self.start_label_idx
        self.max_label_id = max(self.label_idxs)

        self.seq2vocab = {} #Maps seq_idx to vocab_idx
        self.current_verb = None #Current frame (seq_idx)
        self.current_label = None #Current node label (vocab id)
        self.frames_queue = deque([])
        self.seq_idx = -1

        # predecessors
        # keys are <Rn>'s, values are sets of {Rn's}
        self.__predecessors = dict() # (vocab_idx -> {vocab_idx's})

        self.__last_mask = None

        
    def nextTokens(self, token_id: Optional[int] = None):
        if token_id is None and self.context:
            return self.__last_mask
        self.__last_mask = self.__nextTokens(token_id)
        return self.__last_mask

    def __get_valid_targets(self, frame_idx: int) -> Set[int]:
        """
        fram_idx: sequence index of the frame of interest
        """
        node_label = self.context[frame_idx - 1]
        x = set(range(self.start_label_idx, min(self.max_label_id, self.next_label_id) + 1))
        y = self.__predecessors[node_label]
        z =  self.tail_nodes.get(node_label, set())
        return x - y - z

    def __get_arg_mask(self, frame_idx) -> torch.Tensor:
        """
        frame_idx: sequence index of some frame
        """
        mask = torch.full((self.vocab_size,), -math.inf)
        mask[self.stop_token_idx] = 0

        # If we know there are no nodes we can draw an edge to
        # (we'd cause a cycle with existing ones, and we don't have room for new nodes)
        # we should not allow the prediction of any edges
        if not self.__get_valid_targets(frame_idx):
            return mask

        #Allowed args of current verb frame and stop token
        allowed_arg_idxs = self.vf[self.seq2vocab[frame_idx]]
        remaining_args = [idx for idx in allowed_arg_idxs if idx not in self.args_used[frame_idx]]
        mask[remaining_args] = 0
        return mask

    def __nextTokens(self, token_id: Optional[int]) -> torch.Tensor:
        """
        Determines which tokens are allowed next based on the predicted word.

        token: the most recent word predicted by the encoder.
        """ 
        mask = torch.full((self.vocab_size,), -math.inf)  # Start with all tokens disallowed

        if token_id is None and not self.context:
            mask[self.start_label_idx] = 0
            return mask 
        
        self.seq_idx+=1
        self.seq2vocab[self.seq_idx] = token_id
        self.context.append(token_id)
        
        if token_id == self.stop_token_idx:
            if len(self.frames_queue) > 0:
                self.current_verb = self.frames_queue.popleft()
                # The label would have immediately preceded the verb
                self.current_label = self.context[self.current_verb - 1]
                mask[self.current_label] = 0
            else:
                mask[self.end_of_sequence_idx] = 0 
            return mask
        if token_id == self.end_of_sequence_idx or token_id == self.pad_idx:
            mask[self.pad_idx] = 0
            return mask 

        if token_id in self.verb_idxs:
            self.frames_queue.append(self.seq_idx)
            self.args_used[self.seq_idx] = set()
            # Special case--this is our very first verb
            if self.current_verb is None and len(self.frames_queue) > 0:
                self.current_verb = self.frames_queue.popleft()
            else:
                assert self.context[-3] in self.arg_idxs
                assert self.context[-2] in self.label_idxs
                self.tail_nodes[self.current_label].add(self.context[-1])
            return self.__get_arg_mask(self.current_verb)
        else:
            # Condition 1: Check if token is a node <Rn> 
            if token_id in self.label_idxs:

                if len(self.context) > 1 and self.context[-2] == self.stop_token_idx:
                    # We just finished a search for a previous node and are jumping to this one
                    assert token_id in self.__predecessors
                    self.current_label = token_id
                    # self.current_verb was already set appropriately earlier
                    return self.__get_arg_mask(self.current_verb)
                elif token_id in self.__predecessors:
                    self.tail_nodes[self.current_label].add(token_id)
                    # We have a re-entrant edge
                    # This means we're exploring the neighborhood of some frame, so keep exploring
                    return self.__get_arg_mask(self.current_verb)

                assert token_id == self.next_label_id
                self.next_label_id = min(self.next_label_id + 1, self.max_label_id)
                # New node

                # Consider a node to be a predecessor of itself. Don't want self-loops
                self.__predecessors[token_id] = {token_id}
                if token_id == self.start_label_idx:
                    # The root
                    assert self.current_label is None
                    self.current_label = token_id
                    # For now, require a frame to always be the root
                    return self.__vf_mask
                else:
                    # Some other new node
                    self.__predecessors[token_id].update(self.__predecessors[self.current_label])
                    # For other new nodes, a frame or a concept are acceptable

                    return self.__vf_or_concept_mask
               
            #Condition 3: If the token is an argument edge (e.g., `:ARG1`), track it
            if token_id in self.arg_idxs:
                #Add the ARG edge to the current verb's hashmap
                valid_targets = self.__get_valid_targets(self.current_verb)
                assert valid_targets, "Should not have predicted an edge if we didn't have a valid target for it"
                self.args_used[self.current_verb].add(token_id)
                mask[list(valid_targets)] = 0
                return mask
                
            #Condition 4: Check if the token is a concept
            if token_id in self.concept_idxs: 
                assert self.context[-3] in self.arg_idxs
                assert self.context[-2] in self.label_idxs
                self.tail_nodes[self.current_label].add(self.context[-2])
                # Token is a concept. Means we're exploring some verb's arguments
                return self.__get_arg_mask(self.current_verb)

        raise ValueError(f"Invalid token for context: {token_id}")


def process_verbs(file_path) -> Dict[str, List[str]]:
    verb_frames = defaultdict(set)
    f = open(file_path,"r")
    data = json.load(f)
    for token_map in data['amr_symbols']:
        if token_map["category"] == "frame":
            for arg in token_map.get("args", []):
                verb_frames[token_map.get("token")].add(arg)
    return verb_frames

def process_variables(file_path) -> Set[str]:
    variables = set()
    f = open(file_path,"r")
    data = json.load(f)
    for token_map in data['amr_symbols']:
        if token_map["category"] == "label":
            variables.add(token_map.get("token"))
    return variables

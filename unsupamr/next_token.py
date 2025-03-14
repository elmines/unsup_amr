from __future__ import annotations
from typing import List, Set, Dict, Optional
from collections import defaultdict
import json 
import torch
import math 
from collections import deque 
# Local
from .utils import VocabExt

class NextTokens:
    def __init__(self, vocab: VocabExt):

        self.vf = vocab.vf
        self.verb_idxs = set(self.vf.keys())  # FIXME: this is redundant with self.vf. Don't need.
        """
        A hashset of vocab indices corresponding to verbs.
        """

        self.label_idxs = vocab.label_idxs
        """
        A hashset of vocab indices corresponding to labels.
        """
        self.arg_idxs = vocab.arg_idxs 
        """
        A hashset of vocab indices corresponding to argument edges.
        """
        self.concept_idxs = vocab.concept_idxs
        """
        A hashset of vocab indices corresponding to concept nodes.
        """
        self.start_label_idx = vocab.start_label_idx
        self.stop_token_idx = vocab.stop_token_idx
        self.end_of_sequence_idx = vocab.end_of_sequence_idx
        self.pad_idx = vocab.pad_idx
        self.vocab_size = vocab.vocab_size

        # Precomputed constants
        # FIXME: Don't recompute these for every new NextTokens object we make
        self.__pad_mask = torch.where(torch.arange(0, self.vocab_size) == self.pad_idx, 0., -math.inf)
        self.__error_mask = self.__pad_mask

        # Allow any verb frame
        self.__vf_mask = torch.tensor([
            0. if (k in self.vf) else -math.inf for k in range(self.vocab_size)
        ])
        self.__vf_or_concept_mask = torch.tensor([
            0. if (k in self.concept_idxs or k in self.vf) else -math.inf for k in range(self.vocab_size)
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

def validate_sequence(vocab_ext: VocabExt, desired: List[int]):
    nt = NextTokens(vocab_ext)
    mask = nt.nextTokens()
    for _, tok_id in enumerate(desired):
        assert mask[tok_id] == 0
        mask = nt.nextTokens(tok_id)

if __name__ == "__main__":
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from .constants import DEFAULT_SEQ_MODEL
    vocab_ext = VocabExt(T5ForConditionalGeneration.from_pretrained(DEFAULT_SEQ_MODEL), T5TokenizerFast.from_pretrained(DEFAULT_SEQ_MODEL))

    eos_id = vocab_ext.eos_id
    pad_id = vocab_ext.pad_id

    def id_from_token(token: str):
        return next(filter(lambda sym: sym.token == token, vocab_ext.amr_symbols)).id
    tell_01 = id_from_token("tell-01")
    wash_01 = id_from_token("wash-01")
    arg0 = id_from_token(":ARG0")
    arg1 = id_from_token(":ARG1")
    arg2 = id_from_token(":ARG2")
    r0 = vocab_ext.start_label_idx
    r1 = r0 + 1
    r2 = r0 + 2
    r3 = r0 + 3
    r4 = r0 + 4
    stop = vocab_ext.stop_token_idx

    # Test Case A
    # Desired: <R0> tell-01 :ARG0 <R1> concept_a :ARG1 <R2> wash-01 :ARG2 <R3> concept_b <stop> <R2> :ARG0 <R3> :ARG1 <R4> concept_c <stop> EOS
    concept_a = 259
    concept_b = 260
    concept_c = 261
    desired = [r0, tell_01, arg0, r1, concept_a, arg1, r2, wash_01, arg2, r3, concept_b, stop, r2, arg0, r3,  arg1, r4, concept_c, stop, eos_id]
    validate_sequence(vocab_ext, desired)

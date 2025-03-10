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
        self.concept_idxs = set(vocab.pruned_english) - {vocab.eos_id, vocab.pad_id}
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
        
        # State variables
        self.context = []  # Track tokens predicted so far
        self.verb_arguments = {}  # Track which verbs have had which arguments populated
        self.mapping = {} #Maps seq_idx to vocab_idx
        # self.vocab_size = len(self.vocab)
        self.current_verb = None #Current verb (seq_idx)
        self.current_label = None #Current node label (seq idx)
        self.frames_queue = deque([])
        self.seq_idx = -1

        self.__last_mask = None
        
    def nextTokens(self, token_id: Optional[int] = None):
        if token_id is None and self.context:
            return self.__last_mask
        self.__last_mask = self.__nextTokens(token_id)
        return self.__last_mask
        
    def __nextTokens(self, token_id: Optional[int]) -> torch.Tensor:
        """
        Determines which tokens are allowed next based on the predicted word.

        token: the most recent word predicted by the encoder.
        """ 
        mask = torch.full((self.vocab_size,), -math.inf)  # Start with all tokens disallowed

        if not token_id is None and not self.context:
            mask[self.start_label_idx] = 0
            return mask 
        
        self.seq_idx+=1
        self.mapping[self.seq_idx] = token_id
        self.context.append(token_id)

        if token_id == self.stop_token_idx:
            if len(self.frames_queue) > 0:
                self.current_verb = self.frames_queue.popleft()
                if self.mapper[self.current_label] + 1 in self.label_idxs:
                    mask[self.mapper[self.current_label] + 1] = 0
            else:
                mask[self.stop_token_idx] = 0 
            return mask
        
        if token_id == self.end_of_sequence_idx or token_id == self.pad_idx:
            mask[self.pad_idx] = 0
            return mask 

        # Condition 0: If the current token is a verb, start its argument tracking
        if token_id in self.verb_idxs:
            self.frames_queue.append(self.seq_idx)
            self.verb_arguments[self.seq_idx] = set()
            if self.current_verb is None and len(self.frames_queue) > 0:
                self.current_verb = self.frames_queue.popleft()
            #Allowed args of current verb frame and stop token
            allowed_arg_idxs = self.vf[self.mapping[self.current_verb]]
            for arg_idx in allowed_arg_idxs:
                if arg_idx not in self.verb_arguments[self.current_verb]:
                    mask[arg_idx] = 0
            mask[self.stop_idx] = 0
            return mask 

        else:
            # Condition 1: Check if token is a node <Rn> 
            if token_id in self.label_idxs:
                self.current_label = seq_idx
                #If the frames queue is empty, constrain to only verb frames
                if len(self.frames_queue) == 0:
                    for verb_idx in self.vf.keys():
                        mask[verb_idx] = 1
                    return mask #with only verb frames
                #Condition 2: Check if the node is an re-entrant edgge
                elif token_id in self.context: #re-entrant edge 
                    allowed_arg_idxs = self.vf[self.mapping[self.current_verb]]
                    for arg_idx in allowed_arg_idxs:
                        if arg_idx not in self.verb_arguments[self.current_verb]:
                            mask[arg_idx] = 0
                    mask[self.stop_idx] = 0 #take unpicked ARGS and only return that and STOP token
                    return mask 
                else:
                    #Otherwise, It could be both concepts and verb frames
                    for verb_idx in self.vf.keys():
                        mask[verb_idx] = 0
                    for concept_idx in self.concept_idxs:
                        mask[concept_idx] = 0
                return mask

               
            #Condition 3: If the token is an argument edge (e.g., `:ARG1`), track it
            if token_id in self.arg_idxs:
                #Add the ARG edge to the current verb's hashmap
                self.verb_arguments[self.current_verb].add(token_id)
                #If the input is ARG, next token has to be already seen Rn or next Rn
                mask[self.mapper[self.current_label]] = 0
                if self.mapper[self.current_label] + 1 in self.arg_idxs:
                    mask[self.mapper[self.current_label]+1] = 0
                for seq_idx in self.context:
                    if self.mapper[seq_idx] in self.label_idxs:
                        mask[self.mapper[seq_idx]] = 0
                return mask
                
            #Condition 4: Check if the token is a concept
            if token_id in self.concept_idxs: 
                #stop or arg for the current verb
                allowed_arg_idxs = self.vf[self.mapping[self.current_verb]]
                for arg_idx in allowed_arg_idxs:
                    if arg_idx not in self.verb_arguments[self.current_verb]:
                        mask[arg_idx] = 0
                mask[self.stop_idx] = 0 #take unpicked ARGS and only return that and STOP token
                return mask 
               
        return mask  # return the default mask 


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

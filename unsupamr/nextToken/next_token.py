from typing import List, Set, Dict
from collections import defaultdict
import json 
import torch
import math 
from collections import deque 

"""
Since we’re doing breadth-first search, we can enforce the following constraints:

For a given verb node, we must predict all its argument edges in sequence. That is, if the verb supports :ARG0-3 but we’re only predicting :ARG1 and :ARG3, we should have a substring in the output that looks something like “<R6> :ARG1 <R12> :ARG3 <R9>”

A verb node should have at least one argument
"""

class AbstractNextTokens:
    def nextTokens(self, prediction: int) -> torch.Tensor:
        """
        prediction: The word we just predicted

        Returns:
            Tensor of shape (vocabulary_size,) where a 0 indicates the word is allowed and -torch.inf indicates it's not allowed
        """
        pass


class NextTokens(AbstractNextTokens):
    def __init__(self, verb_frames, verb_idxs, label_idxs, arg_idxs, concept_idxs, vocab):
        """
        verb_frames: A dictionary mapping verbs to the set of allowed arguments (e.g., {'frame_vocab_id': {'arg1_vocab_id', 'arg2_vocab_id'}}).
        verb_idxs: A hashset of vocab indices corresponding to verbs.
        label_idxs: A hashset of vocab indices corresponding to labels.
        arg_idxs: A hashset of vocab indices corresponding to argument edges.
        concept_idxs: A hashset of vocab indices corresponding to concept nodes.
        """

        self.vf = verb_frames
        self.verb_idxs = verb_idxs 
        self.label_idxs = label_idxs
        self.arg_idxs = arg_idxs 
        self.concept_idxs = concept_idxs
        self.vocab = vocab
        
        # State variables
        self.context = []  # Track tokens predicted so far
        self.verb_arguments = {}  # Track which verbs have had which arguments populated
        self.vocab_size = len(self.vocab)
        self.stop_token_idx = 0
        self.current_verb = None
        self.current_label = None
        self.frames_queue = deque([])
        self.seq_idx = -1
        self.mapping = {}
        
    def nextTokens(self, token_id: int) -> torch.Tensor:
        """
        Determines which tokens are allowed next based on the predicted word.

        token: the most recent word predicted by the encoder.
        """ 
        mask = torch.full((self.vocab_size,), -math.inf)  # Start with all tokens disallowed
        self.seq_idx+=1
        self.mapping[self.seq_idx] = token_id
        self.context.append(token_id)

        # Condition 0: If the current token is a verb, start its argument tracking
        if token_id in self.verb_idxs:
            self.frames_queue.append(self.seq_idx)
            self.verb_arguments[self.seq_idx] = set()
            if self.current_verb is None and len(self.frames_queue) > 0:
                self.current_verb = self.verb_arguments[self.frames_queue.popleft()]
            #Allowed args of current verb frame and stop token
            allowed_arg_idxs = self.vf[self.mapping[self.current_verb]]
            for arg_idx in allowed_arg_idxs:
                if arg_idx not in self.verb_arguments[self.current_verb]:
                    mask[arg_idx] = 1
            mask[self.stop_idx] = 1
            return mask 

        else:
            # Condition 1: Check if token is a node <Rn> 
            if token_id in self.label_idxs:
                self.current_label = token_id
                #If the frames queue is empty, constrain to only verb frames
                if len(self.frames_queue) == 0:
                    for verb_idx in self.vf.keys():
                        mask[verb_idx] = 1
                    return mask #with only verb frames
                else:
                    #Otherwise, It could be both concepts and verb frames
                    for verb_idx in self.vf.keys():
                        mask[verb_idx] = 1
                    for concept_idx in self.concept_idxs:
                        mask[concept_idx] = 1
                return mask

            #Condition 2: Check if the node is an re-entrant edgge
            if token_id in self.context: #re-entrant edge 
                allowed_arg_idxs = self.vf[self.mapping[self.current_verb]]
                for arg_idx in allowed_arg_idxs:
                    if arg_idx not in self.verb_arguments[self.current_verb]:
                        mask[arg_idx] = 1
                mask[self.stop_idx] = 1 #take unpicked ARGS and only return that and STOP token
                return mask 
               
            #Condition 3: If the token is an argument edge (e.g., `:ARG1`), track it
            if token_id in self.arg_idxs:
                #Add the ARG edge to the current verb's hashmap
                self.verb_arguments[self.current_verb].add(token_id)
                #If the input is ARG, next token has to be already seen Rn or next Rn
                mask[self.current_label] = 1
                mask[self.current_label+1] = 1
                return mask
                
            #Condition 4: Check if the token is a concept
            if token_id in self.concept_idxs: 
                #stop or arg for the current verb
                allowed_arg_idxs = self.vf[self.mapping[self.current_verb]]
                for arg_idx in allowed_arg_idxs:
                    if arg_idx not in self.verb_arguments[self.current_verb]:
                        mask[arg_idx] = 1
                mask[self.stop_idx] = 1 #take unpicked ARGS and only return that and STOP token
                return mask 

            #Condition 5: Check if the token is a stop token
            if token_id not in self.context:
                for verb_idx in self.vf.keys():
                    mask[verb_idx] = 1
                for concept_idx in self.concept_idxs:
                    mask[concept_idx] = 1
                return mask  #return either a verb frame or a concept (pruned english)
               
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

if __name__ == "__main__":
    file_path = "data/vocab.json"
    verb_frames = process_verbs(file_path)
    variables = process_variables(file_path)
    sequence = ["<R0>", "tell-01", ":ARG0", "<R1>", "you", ":ARG1", "<R3>", "wash-01", ":ARG2" ,"<R2>",  "i" ,"<stop>", "<R3>", ":ARG0" , "<R2>", ":ARG1" ,"<R4>" ,"dog" ,"<stop>"]
    next_token_instance = NextTokens(verb_frames, variables)
    for word in sequence:
        mask = next_token_instance.nextTokens(word)
        print(next_token_instance.context)
        print(mask)
   
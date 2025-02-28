from typing import List, Set, Dict
from collections import defaultdict
import json 
import torch

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
    def __init__(self, verb_frames, variables):
        """
        verb_frames: A dictionary mapping verbs to the set of allowed arguments (e.g., {'eat': {':ARG0', ':ARG1'}}).
        variables: A set of AMR variable names already assigned.
        """
        self.vf = verb_frames
        self.vars = variables

        # State variables
        self.context = []  # Track tokens predicted so far
        self.verb_arguments = {}  # Track which verbs have had which arguments populated
        self.node_labels = {}  # Track which node labels have been used
        self.vocab_size = len(self.vf) + len(self.vars)
        self.current_verb = None

    def nextTokens(self, token: str) -> torch.Tensor:
        """
        Determines which tokens are allowed next based on the predicted word.

        token: the most recent word predicted by the encoder.
        """ 
        mask = torch.zeros(self.vocab_size)  # Start with all tokens allowed

        self.context.append(token)

        # Condition 0: Check if token is a node <R01> 
        if token.startswith("<R") and token.endswith(">"):
            if len(self.context) > 1 and self.context[-2] in self.vf:  # If previous token was a verb, associate this node with it
                self.current_verb = self.context[-2]
                self.node_labels[token] = self.current_verb  # Assign node to verb
            return mask  # No special restriction for the node. 

        # Condition 1: If the current token is a verb, start its argument tracking
        if token in self.vf:
            self.verb_arguments[token] = set()
            self.current_verb = token

        #Condition 2: If the token is an argument edge (e.g., `:ARG1`), track it
        if token.startswith(":ARG"):
            #Condition3: Check AMR validity, ensure there is a preceding verb. 
            if not self.context or len(self.context) > 1 and self.context[-2] not in self.vf:
                mask[:] = -torch.inf #Invalid State
                return mask

            verb = self.context[-2]  # Get the verb preceding this argument. (Let me know if this assumption has any flaws)
            self.verb_arguments[verb].add(token)

        #Condition 3: Constrain the frame to have at least one argument
        for verb, args in self.verb_arguments.items():
            if not args:
                # If a verb has been predicted but no argument yet, enforce at least one argument
                allowed_args = self.vf[verb]  # Get allowed arguments for this verb
                mask[:] = -torch.inf  # Disallow all
                for arg in allowed_args:
                    mask[self.encode_token(arg)] = 0  # Allow only valid arguments
                return mask

        #Condition 4: Constrain the argument sequence order
        for verb, allowed_args in self.vf.items():
            predicted_args = self.verb_arguments.get(verb, set())
            remaining_args = sorted(allowed_args - predicted_args)  # Order the remaining args
            if remaining_args:
                next_arg = remaining_args[0]  # The next expected argument
                mask[:] = -torch.inf  # Disallow all
                mask[self.encode_token(next_arg)] = 0  # Allow only the next expected arg
                return mask

        return mask  # return the default mask 

    def encode_token(self, token: str) -> int:
        return 0

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
   
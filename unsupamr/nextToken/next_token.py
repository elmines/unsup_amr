from typing import List, Set, Dict
from collections import defaultdict
import json 
sequence = ["(", "w", "/", "want-01", ":ARG0", "(", "p", "/", "person", ")", ":ARG1"]

def nextTokens(current_sequence : List[str], verb_frames: Dict[str, List[str]], variables: Set[str]) -> List[str]:

    if not current_sequence:
        return ["("]
    
    context = current_sequence[-1] if current_sequence else ""

    if context == "(":
        # Return all possible concept types, concepts in our case would be 
        #verb frames, named entities, properties, general concepts (nouns). 
        return list(verb_frames.keys()) + variables
    
    if context in variables: 
        return ["/"]
    
    if context == "/":
        return verb_frames.keys()
    
    if context in verb_frames.keys(): 
        return ["/"]
    
    # Check if we specified a verb concept (Question: Do I preprocess the verb and remove the - )
    for verb in verb_frames.keys():
        if context == verb:
            # Find all arguments this verb supports
            valid_args = verb_frames[verb]
            if not valid_args:
                # If no arguments defined, we can either close or allow general AMR relations
                return [")"]
            return valid_args
    
    # When we're in the middle of processing a verb's arguments
    for i in range(len(current_sequence) - 1, -1, -1):
        if current_sequence[i] in verb_frames.keys():
            verb = current_sequence[i]
            # Check which arguments we've already predicted for this verb
            predicted_args = []
            for j in range(i + 1, len(current_sequence)):
                if current_sequence[j].startswith(":ARG"):
                    predicted_args.append(current_sequence[j])
            
            # Get remaining arguments
            remaining_args = [arg for arg in verb_frames[verb] if arg not in predicted_args]
            
            # If we just specified an argument type, we need a target node
            if context.startswith(":ARG"):
                return ["("] 
            
            # If we have more arguments to predict, predict them next
            if remaining_args:
                return remaining_args
            else:
                # If all arguments predicted, close this node
                return [")"]
    
    open_count = current_sequence.count("(")
    close_count = current_sequence.count(")")

    if open_count > close_count:
        return [")"]
    
    return []


def process_verbs(verb_path) -> Dict[str, List[str]]:
    verb_frames = defaultdict(list)
    f = open(verb_path,"r")
    data = json.load(f)
    for token_map in data['amr_symbols']:
        if token_map["category"] == "frame":
            verb_frames[token_map.get("token")].append(token_map.get("args"))
    return verb_frames

def process_variables() -> Set[str]:
    variables = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"}
    return variables

if __name__ == "__main__":
    verb_path = "data/vocab.json"
    verb_frames = process_verbs(verb_path)
    variables = process_variables()
    sequence = ["(", "w", "/", "want-01", ":ARG0", "(", "p", "/", "person", ")", ":ARG1"]
    next_valid_tokens = nextTokens(sequence, verb_frames, variables)
    print(f"Current sequence: {' '.join(sequence)}")
    print(f"Next valid tokens: {next_valid_tokens}")


import re, json
from collections import defaultdict

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        data = json.load(file)
    
    # Create a mapping from token IDs to tokens for quick lookup
    id_to_token = {symbol['id']: symbol['token'] for symbol in data['amr_symbols']}

    return id_to_token

def convert_ids_to_tokens(output_ids, id_to_token):
    # Convert output IDs to tokens using the mapping, or return "<UNK>" if not found
    return [id_to_token.get(i, "<UNK>") for i in output_ids]

def bfs_to_penman(tokens):
    """
    Convert BFS token sequence to Penman notation, handling truncated sequences.
    
    Args:
        tokens: List of tokens in the BFS sequence
        
    Returns:
        String in Penman notation
    """
    # Check if sequence appears to be truncated (no <stop> at the end)
    truncated = '<eos>' not in tokens and '</s>' not in tokens and not tokens[-1] == '<stop>'
    
    # Extract nodes and edges
    nodes = {}  # Map of node IDs to their labels
    edges = defaultdict(list)  # Map of source nodes to [(target, label), ...]
    
    # First pass: Identify all nodes and their labels
    i = 0
    while i < len(tokens):
        if re.match(r'<R\d+>', tokens[i]):
            node_id = tokens[i]
            # Check if this node is followed by a label
            if i+1 < len(tokens) and not tokens[i+1].startswith('<') and not tokens[i+1].startswith(':'):
                nodes[node_id] = tokens[i+1]
                i += 2
            else:
                i += 1
        else:
            i += 1
    
    # Second pass: Build edges with special handling for truncated sequences
    active_node = None
    pending_edge = None
    i = 0
    
    while i < len(tokens):
        # Node reference (e.g., <R0>)
        if re.match(r'<R\d+>', tokens[i]):
            node_id = tokens[i]
            
            # If this is the start of the sequence or follows a <stop>, it's the active node
            if i == 0 or (i > 0 and tokens[i-1] == '<stop>'):
                active_node = node_id
                pending_edge = None
                i += 1
            # If this follows an edge label, it's a target node
            elif i > 0 and tokens[i-1].startswith(':'):
                # Handle edge cases for truncated sequences
                if truncated and i == len(tokens) - 1:
                    # Case 1: If it ends with a new node label, don't add the edge
                    if node_id not in nodes:
                        i += 1
                        continue
                    # Case 2: If it ends with an existing node label, include the edge
                
                if active_node:
                    edge_label = tokens[i-1]
                    edges[active_node].append((node_id, edge_label))
                    pending_edge = None
                i += 1
            # Skip node definition (already handled in first pass)
            elif i+1 < len(tokens) and not tokens[i+1].startswith('<') and not tokens[i+1].startswith(':'):
                i += 2
            else:
                i += 1
        # Edge label (e.g., :ARG0)
        elif tokens[i].startswith(':'):
            # Case 4: If truncated and ends with just an edge, don't record it
            if not (truncated and i == len(tokens) - 1):
                pending_edge = tokens[i]
            i += 1
        # <stop> token
        elif tokens[i] == '<stop>':
            active_node = None
            pending_edge = None
            i += 1
        # Other tokens
        else:
            i += 1
    
    # Create variable names for each node
    var_map = {}
    for node_id, label in nodes.items():
        var_name = label[0].lower() if label and label[0].isalpha() else 'x'
        suffix = 1
        base_var = var_name
        while var_name in var_map.values():
            var_name = f"{base_var}{suffix}"
            suffix += 1
        var_map[node_id] = var_name
    
    # Generate Penman notation
    root = list(nodes.keys())[0] if nodes else None
    
    # Build the Penman string recursively
    def build_penman(node_id, visited=None):
        if visited is None:
            visited = set()
        
        if node_id in visited:
            return var_map[node_id]
        
        visited.add(node_id)
        node_label = nodes[node_id]
        var_name = var_map[node_id]
        
        result = f"({var_name} / {node_label}"
        
        for target_id, edge_label in edges[node_id]:
            clean_edge = edge_label[1:] if edge_label.startswith(':') else edge_label
            
            if target_id in visited:
                result += f"\n  :{clean_edge} {var_map[target_id]}"
            else:
                result += f"\n  :{clean_edge} {build_penman(target_id, visited)}"
        
        result += ")"
        return result
    
    return build_penman(root) if root else "()"

# Example usage
if __name__ == "__main__":

    # id_to_token = load_vocab("vocab.json")

    # #over the whole predicted_output file
    # with open('predition_output.txt', 'r') as predicted_file:
    #     for each in predicted_file:
    #         bfs_tokens = convert_ids_to_tokens(each, id_to_token)
    #         print(bfs_tokens)
    #         penman_notation = bfs_to_penman(bfs_tokens)
    #         print(penman_notation)
    #         break
          



    # Load vocabulary and convert IDs to tokens (if needed)
    try:
        id_to_token = load_vocab("vocab.json")
        
        output_ids = [258872, 257953, 258833, 258873, 227227, 258835, 258875, 258572, 
                      258837, 258874, 123457, 258853, 258875, 258833, 258874, 258835, 
                      258876, 123456, 258853]
        
        bfs_tokens0 = convert_ids_to_tokens(output_ids, id_to_token)
        penman_notation0 = bfs_to_penman(bfs_tokens0)
        print(penman_notation0)
    except Exception as e:
        print(f"Error with vocab conversion: {e}")
    
    # Test with direct token sequences
    bfs_tokens = ['<R0>', 'tell-01', ':ARG0', '<R1>', 'you', ':ARG1', '<R3>', 
                 'wash-01', ':ARG2', '<R2>', 'i', '<stop>', '<R3>', ':ARG0', 
                 '<R2>', ':ARG1', '<R4>', 'dog', '<stop>']
    
    penman_notation = bfs_to_penman(bfs_tokens)
    print(penman_notation)

    # Another example
    bfs_tokens1 = ['<R0>', 'want-01', ':ARG0', '<R1>', 'person', ':ARG1', '<R2>', 
                  'eat-01', '<stop>', '<R2>', ':ARG0', '<R1>', ':ARG1', '<R3>', 
                  'food', '<stop>']

    penman_notation1 = bfs_to_penman(bfs_tokens1)
    print(penman_notation1)

    # Test truncation edge cases
    print("\nTesting truncation handling scenarios:")
    
    # Case 1: Ends with a new node label
    tokens1 = ['<R0>', 'tell-01', ':ARG0', '<R1>', 'you', ':ARG1', '<R3>']
    print("\nCase 1: Ends with a new node label")
    print("Input:", tokens1)
    print("Output:", bfs_to_penman(tokens1))
    
    # Case 2: Ends with an existing node label
    tokens2 = ['<R0>', 'tell-01', ':ARG0', '<R1>', 'you', ':ARG1', '<R1>']
    print("\nCase 2: Ends with an existing node label")
    print("Input:", tokens2)
    print("Output:", bfs_to_penman(tokens2))
    
    # Case 3: Ends with a verb frame
    tokens3 = ['<R0>', 'tell-01', ':ARG0', '<R1>', 'you', ':ARG1', '<R3>', 'eat-01']
    print("\nCase 3: Ends with a verb frame")
    print("Input:", tokens3)
    print("Output:", bfs_to_penman(tokens3))
    
    # Case 4: Ends with just an :ARG edge
    tokens4 = ['<R0>', 'tell-01', ':ARG0', '<R1>', 'you', ':ARG1']
    print("\nCase 4: Ends with just an :ARG edge")
    print("Input:", tokens4)
    print("Output:", bfs_to_penman(tokens4))

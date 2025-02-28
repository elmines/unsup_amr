from typing import Dict, Any
import json

VocabExt = Dict[str, Any]

def load_vocab(vocab_path: str) -> VocabExt:
    with open(vocab_path, 'r') as r:
        return json.load(r)
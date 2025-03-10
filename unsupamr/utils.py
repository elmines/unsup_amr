from typing import Dict, Any, Optional, List
import json
import dataclasses
from .constants import AmrCategory

@dataclasses.dataclass
class AMRSymbol:
    token: str
    id: int
    embed_id: Optional[int]
    category: AmrCategory
    args: List[int]

    @staticmethod
    def from_json(json: Dict[str, Any]):
        embed_id = json.get('embed')
        embed_id = int(embed_id) if embed_id is not None else None
        return AMRSymbol(
            token=json['token'],
            id=int(json['id']),
            embed_id=embed_id,
            category=AmrCategory(json['category']),
            args=json.get('args', [])
        )

@dataclasses.dataclass
class VocabExt:
    pad_id: int
    eos_id: int
    pruned_english: List[int]
    amr_symbols: List[AMRSymbol]

    @staticmethod
    def from_json(json: Dict[str, Any]):
        return VocabExt(
            pad_id=json['pad_id'],
            eos_id=json['eos_id'],
            pruned_english=list(map(int, json['pruned_english'])),
            amr_symbols=list(map(AMRSymbol.from_json, json['amr_symbols']))
        )

def load_vocab(vocab_path: str):
    with open(vocab_path, 'r') as r:
        return json.load(r)
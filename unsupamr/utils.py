import os
from typing import Dict, Any, Optional, List, Set
import re
import dataclasses
# 3rd Party
from transformers import T5ForConditionalGeneration, T5Tokenizer
# Local
from .constants import AmrCategory, DEFAULT_PROPBANK, T5_SEP

@dataclasses.dataclass
class AMRSymbol:
    token: str
    id: int
    embed_id: Optional[int]
    category: AmrCategory
    args: List[int] = dataclasses.field(default_factory=list)

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

def get_special_token(name: str, tokenizer: T5Tokenizer):
    symbol = tokenizer.special_tokens_map[name]
    return tokenizer.all_special_ids[tokenizer.all_special_tokens.index(symbol)]

class VocabExt:
    pad_id: int
    eos_id: int
    pruned_english: List[int]
    amr_symbols: List[AMRSymbol]

    @staticmethod
    def __postprocess_token(tok: str):
        # If our model predicts AMR-syntax symbols, that could break the penman format 
        if tok == ':':
            return 'COLON'
        if tok == '/':
            return 'SLASH'
        if tok == '(' or tok == ')':
            return 'PAREN'
        if tok == '-':
            return 'HYPEN'
        return tok

    def ids_to_str(self, ids: List[int]) -> List[str]:
        return [VocabExt.__postprocess_token(self.__id_to_token[id]) for id in ids]

    def __init__(self,
                 model: T5ForConditionalGeneration,
                 tokenizer: T5Tokenizer,
                 propbank_path: os.PathLike = DEFAULT_PROPBANK,
                 max_nodes: int = 25):
        self.tokenizer = tokenizer

        lm_head_size = model.lm_head.weight.shape[0]

        # Tuples of (symbol, embedding_id, type_name, args)
        vocab = tokenizer.get_vocab()
        def get_ids(w):
            return vocab.get(f"{T5_SEP}{w}")


        frame_patt = re.compile(r"((.+)-[0-9]+).*")
        arg_patt = re.compile(r"ARG[0-9]")
        with open(propbank_path, 'r') as r:
            lines = r.readlines()

        arg_vocab = set()
        vf_tuples = []
        for l in lines:
            res = frame_patt.match(l)
            frame_name = res.group(1)
            lemma = res.group(2)
            args = arg_patt.findall(l)
            arg_vocab.update(args)
            vf_tuples.append((frame_name, get_ids(lemma), AmrCategory.FRAME, args))

        vocab_index = lm_head_size
        arg_map = dict()
        amr_entries = []
        sorted_arg_vocab = sorted(arg_vocab)
        for arg_name in sorted_arg_vocab:
            arg_map[arg_name] = vocab_index
            amr_entries.append(AMRSymbol(
                token=f":{arg_name}",
                id=vocab_index,
                embed_id=None,
                category=AmrCategory.ARG,
            ))
            vocab_index += 1
        for arg_name in sorted_arg_vocab:
            amr_entries.append(AMRSymbol(
                token=f":{arg_name}-of",
                id=vocab_index,
                embed_id=None,
                category=AmrCategory.INV_ARG.value
            ))
            vocab_index += 1
        for (frame_name, embed_id, cat, str_args) in vf_tuples:
            amr_entries.append(AMRSymbol(
                token=frame_name,
                id=vocab_index,
                embed_id=embed_id,
                category=cat,
                args=[arg_map[s] for s in str_args]
            ))
            vocab_index += 1
        # Have to subtract this before doing those post-increments
        vocab_index -= 1
        amr_entries.append(AMRSymbol("<stop>", id=(vocab_index := vocab_index + 1), embed_id=None, category=AmrCategory.STOP))
        amr_entries.extend(AMRSymbol(f"<R{i}>", id=(vocab_index := vocab_index + 1), embed_id=None, category=AmrCategory.LABEL) for i in range(max_nodes))

        self.eos_id        : int             = get_special_token("eos_token", tokenizer)
        self.pad_id        : int             = get_special_token("pad_token", tokenizer)
        self.amr_symbols   : List[AMRSymbol] = amr_entries
        self.stop_token_idx: int             = next(ent for ent in self.amr_symbols if ent.category == AmrCategory.STOP).id
        self.new_vocab_size: int             = lm_head_size + len(amr_entries)
        self.concept_idxs: Set[int]        = set(range(len(tokenizer.get_vocab()))) - {self.eos_id, self.pad_id}

        # TODO: Get rid of redundant fields. Doing this right now for compatibility with nextTokens
        self.vf = {ent.id:ent.args for ent in self.amr_symbols if ent.category == AmrCategory.FRAME}
        self.label_idxs = {ent.id for ent in self.amr_symbols if ent.category == AmrCategory.LABEL}
        self.arg_idxs = {ent.id for ent in self.amr_symbols if ent.category == AmrCategory.ARG} # Only args, no inverse args yet
        self.start_label_idx = next(ent.id for ent in self.amr_symbols if ent.token == "<R0>")
        self.stop_token_idx = next(ent.id for ent in self.amr_symbols if ent.category == AmrCategory.STOP)

        assert self.start_label_idx is not None
        assert self.stop_token_idx is not None

        self.__id_to_token = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # Extend with AMR symbols
        for amr_symbol in self.amr_symbols:
            self.__id_to_token[amr_symbol.id] = amr_symbol.token

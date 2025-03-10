# STL
import re
import argparse
import sys
import json
# 3rd Party
from tqdm import tqdm
from transformers import T5Tokenizer, MT5ForConditionalGeneration
from datasets import load_dataset
# Local
from .constants import EUROPARL_URI, T5_SEP, DEFAULT_SEQ_MODEL, AmrCategory

def get_special_token(name: str, tokenizer: T5Tokenizer):
    symbol = tokenizer.special_tokens_map[name]
    return tokenizer.all_special_ids[tokenizer.all_special_tokens.index(symbol)]

def main(raw_args=None):
    DATASET_SUBSETS = ["de-en", "en-es", "de-es"]

    parser = argparse.ArgumentParser(description="Prunes a T5 vocabulary according to available EuroParl data")

    parser.add_argument("--propbank", required=True, metavar="propbank-amr-frame-arg-descr.txt", help="List of supported frames from AMR 3.0 dataset")
    parser.add_argument("--model_name", default=DEFAULT_SEQ_MODEL, metavar=DEFAULT_SEQ_MODEL, nargs=1, help="Tokenizer model")
    parser.add_argument("--max_concepts", default=25, metavar=25, type=int, help="Maximum number of <Rx> tokens to support")
    parser.add_argument("--max_ops", default=10, metavar=10, type=int, help="Maximum number of :opx labels to support (default is :op0 to :op9)")
    parser.add_argument("-o", default="vocab.json", help="Output path")
    args = parser.parse_args(raw_args)

    propbank_path = args.propbank
    model_name = args.model_name
    concepts_limit = args.max_concepts
    ops_limit = args.max_ops
    out_path = args.o

    lm_head_size = MT5ForConditionalGeneration.from_pretrained(model_name).lm_head.weight.shape[0]

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    eos_id = get_special_token("eos_token", tokenizer)
    pad_id = get_special_token("pad_token", tokenizer)
    special_tokens = {eos_id, pad_id}

    # Tuples of (symbol, embedding_id, type_name, args)
    amr_tuples = []
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
    for arg_name in sorted(arg_vocab):
        arg_map[arg_name] = vocab_index
        amr_entries.append({
            "token": f":{arg_name}",
            "id": vocab_index,
            "embed_id": None,
            "category": AmrCategory.ARG.value
        })
        vocab_index += 1
        amr_entries.append({
            "token": f":{arg_name}-of",
            "id": vocab_index,
            "embed_id": None,
            "category": AmrCategory.INV_ARG.value
        })
        vocab_index += 1
    for (frame_name, embed_id, cat, str_args) in vf_tuples:
        amr_entries.append({
            "token": frame_name,
            "id": vocab_index,
            "embed_id": embed_id,
            "category": cat.value,
            "str_args": [arg_map[s] for s in str_args]
        })
        vocab_index += 1


    en_ids = set()
    multiling_ids = set()
    for subset in DATASET_SUBSETS:
        ds = load_dataset(path=EUROPARL_URI, name=subset)['train']['translation']
        for sample in tqdm(ds, desc=f"EuroParl {subset}"):
            if "en" in sample:
                sample_ids = tokenizer(sample['en'])['input_ids']
                en_ids.update(sample_ids)
                multiling_ids.update(sample_ids)
            if "de" in sample:
                multiling_ids.update(tokenizer(sample['de'])['input_ids'])
            if "es" in sample:
                multiling_ids.update(tokenizer(sample['es'])['input_ids'])

    amr_tuples.append(("<stop>", None, AmrCategory.STOP))

    amr_tuples.append((":polarity -", get_ids('not'), AmrCategory.POLARITY))

    amr_tuples.append((":domain", get_ids('is'), AmrCategory.DOMAIN))
    amr_tuples.append((":domain-of", get_ids('is'), AmrCategory.INV_DOMAIN))
    amr_tuples.append((":poss", get_ids('his'), AmrCategory.POSS))
    amr_tuples.append((":poss-of", get_ids('his'), AmrCategory.INV_POSS))

    amr_tuples.append(("and", get_ids('and'), AmrCategory.CONJ))
    amr_tuples.append(("or", get_ids('or'), AmrCategory.CONJ))

    amr_tuples.append(("amr-unknown", get_ids('what'), AmrCategory.UNKNOWN))
    amr_tuples.extend( (f":op{i}", None, AmrCategory.OPTION) for i in range(ops_limit))
    amr_tuples.extend( (f"<R{i}>", None, AmrCategory.LABEL) for i in range(concepts_limit))

    for (i, (token, embed_id, category)) in enumerate(amr_tuples, start=vocab_index):
        entry = {"token": token, "id": i, "embed": embed_id, "category": category.value}
        amr_entries.append(entry)

    en_ids = sorted(en_ids | special_tokens)
    multiling_ids = sorted(multiling_ids | special_tokens)

    out_dict = {
        "pad_id": pad_id,
        "eos_id": eos_id,
        "pruned_english": en_ids, 
        "pruned_multiling": multiling_ids,
        "amr_symbols": amr_entries,
    }

    with open(out_path, 'w') as w:
        json.dump(out_dict, w, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])


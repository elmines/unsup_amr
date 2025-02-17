# STL
import argparse
import sys
import json
# 3rd Party
from tqdm import tqdm
from transformers import T5Tokenizer
from datasets import load_dataset
# Local
from .constants import EUROPARL_URI, T5_SEP, DEFAULT_SEQ_MODEL

def main(raw_args=None):
    DATASET_SUBSETS = ["de-en", "en-es", "de-es"]

    parser = argparse.ArgumentParser(description="Prunes a T5 vocabulary according to available EuroParl data")
    parser.add_argument("--model_name", default=DEFAULT_SEQ_MODEL, metavar=DEFAULT_SEQ_MODEL, nargs=1, help="Tokenizer model")
    parser.add_argument("--max_concepts", default=25, metavar=25, type=int, help="Maximum number of concept nodes in AMR graphs")
    parser.add_argument("--max_args", default=5, metavar=5, type=int, help="Maximum number of :ARGx and :ARGx-of labels to support (default is :ARG0 to :ARG4)")
    parser.add_argument("--max_ops", default=10, metavar=10, type=int, help="Maximum number of :opx labels to support (default is :op0 to :op9)")
    parser.add_argument("-o", default="vocab.json", help="Output path")
    args = parser.parse_args(raw_args)

    model_name = args.model_name
    concepts_limit = args.max_concepts
    args_limit = args.max_args
    ops_limit = args.max_ops
    out_path = args.o

    tokenizer = T5Tokenizer.from_pretrained(model_name)

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

    amr_symbols = []
    amr_embeddings = []
    vocab = tokenizer.get_vocab()
    def get_ids(w):
        return vocab.get(f"{T5_SEP}{w}")

    amr_symbols.append(":polarity -") # Not, don't, can't, etc.
    amr_embeddings.append(get_ids('not'))

    amr_symbols.append(":domain")
    amr_embeddings.append(get_ids('is'))
    amr_symbols.append(":mod") # Inverse of domain
    amr_embeddings.append(get_ids('is'))

    amr_symbols.append(":poss")
    amr_embeddings.append(get_ids('his'))
    amr_symbols.append(":poss-of")
    amr_embeddings.append(get_ids('his'))

    # Violates original AMR notation, but we don't want this getting confused with the regular word "and" somewhere
    amr_symbols.append("<and>")
    amr_embeddings.append(get_ids('and'))
    amr_symbols.append("<or>")
    amr_embeddings.append(get_ids('or'))
    amr_symbols.append("<contrast>")
    amr_embeddings.append(get_ids('but'))
    amr_symbols.append("<amr-unknown>")
    amr_embeddings.append(get_ids('what'))

    amr_symbols.extend(f"<R{i}>" for i in range(concepts_limit))
    amr_embeddings.extend(None for _ in range(concepts_limit))
    for i in range(args_limit):
        amr_symbols.append(f":ARG{i}")
        amr_embeddings.append(None)
        amr_symbols.append(f":ARG{i}-of")
        amr_embeddings.append(None)

    amr_symbols.extend(f":op{i}" for i in range(ops_limit))
    amr_embeddings.extend(None for _ in range(ops_limit))

    en_ids = sorted(en_ids | {0})
    multiling_ids = sorted(multiling_ids | {0})

    out_dict = {
        "pruned_english": en_ids, # Add the 0 padding token
        "pruned_multiling": multiling_ids,
        "amr_symbols": [{"token": w, "id": i, "embed": embed_id} for (i, (w, embed_id)) in enumerate(zip(amr_symbols, amr_embeddings), start=len(vocab))]
    }

    with open(out_path, 'w') as w:
        json.dump(out_dict, w, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])


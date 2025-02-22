# STL
import re
import argparse
import sys
import json
# 3rd Party
from tqdm import tqdm
from transformers import T5Tokenizer
from datasets import load_dataset
# Local
from .constants import EUROPARL_URI, T5_SEP, DEFAULT_SEQ_MODEL, AmrToken

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

    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Tuples of (symbol, embedding_id, type_name, args)
    amr_tuples = []
    vocab = tokenizer.get_vocab()
    def get_ids(w):
        return vocab.get(f"{T5_SEP}{w}")


    frame_patt = re.compile(r"((.+)-[0-9]+).*")
    arg_patt = re.compile(r"ARG[0-9]")
    with open(propbank_path, 'r') as r:
        lines = r.readlines()
    arg_set = set()
    for l in lines:
        res = frame_patt.match(l)
        frame_name = res.group(1)
        lemma = res.group(2)
        args = arg_patt.findall(l)
        arg_set.update(args)
        amr_tuples.append((frame_name, get_ids(lemma), AmrToken.FRAME, args))

    for arg_name in sorted(arg_set):
        arg_name = ":" + arg_name
        inv_name = arg_name + "-of"
        amr_tuples.append((arg_name, None, AmrToken.ARG, None))
        amr_tuples.append((inv_name, None, AmrToken.INV_ARG, None))


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

    amr_tuples.append(("<stop>", None, AmrToken.STOP, None))

    amr_tuples.append((":polarity -", get_ids('not'), AmrToken.POLARITY, None))

    amr_tuples.append((":domain", get_ids('is'), AmrToken.DOMAIN, None))
    amr_tuples.append((":domain-of", get_ids('is'), AmrToken.INV_DOMAIN, None))
    amr_tuples.append((":poss", get_ids('his'), AmrToken.POSS, None))
    amr_tuples.append((":poss-of", get_ids('his'), AmrToken.INV_POSS, None))

    amr_tuples.append(("and", get_ids('and'), AmrToken.CONJ, None))
    amr_tuples.append(("or", get_ids('or'), AmrToken.CONJ, None))

    amr_tuples.append(("amr-unknown", get_ids('what'), AmrToken.UNKNOWN, None))
    amr_tuples.extend( (f":op{i}", None, AmrToken.OPTION, None) for i in range(ops_limit))
    amr_tuples.extend( (f"<R{i}>", None, AmrToken.LABEL, None) for i in range(concepts_limit))

    amr_entries = []
    for (i, (token, embed_id, category, args)) in enumerate(amr_tuples, start=len(vocab)):
        entry = {"token": token, "id": i, "embed": embed_id, "category": category.value}
        if args:
            entry["args"] = args
        amr_entries.append(entry)

    en_ids = sorted(en_ids | {0})
    multiling_ids = sorted(multiling_ids | {0})

    out_dict = {
        "pruned_english": en_ids, # Add the 0 padding token
        "pruned_multiling": multiling_ids,
        "amr_symbols": amr_entries,
    }

    with open(out_path, 'w') as w:
        json.dump(out_dict, w, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])


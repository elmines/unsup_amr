# %%
# STL
import time
import json
# 3rd Party
from tqdm import tqdm
from transformers import T5Tokenizer
from datasets import load_dataset

# %%
# Constants
DATASET_URI = "Helsinki-NLP/europarl"
DATASET_SUBSETS = ["de-en", "en-es", "de-es"]
# They don't use a real underscore character, but something else that looks like an underscore
T5_SEP = (b'\xe2\x96\x81').decode()

# %%
# Turn these into command line arguments
model_name = "google/mt5-small"
concepts_limit = 25
args_limit = 5
ops_limit = 10
out_path = "vocab.json"

# %%
tokenizer = T5Tokenizer.from_pretrained(model_name)

# %%
amr_whitelist = {}
multilingual_whitelist = {}
to_prune = tokenizer.get_vocab()

en_ids = set()
multiling_ids = set()

for subset in DATASET_SUBSETS:
    ds = load_dataset(path=DATASET_URI, name=subset)['train']
    ds = list(map(lambda s: s['translation'], ds))
    for sample in tqdm(ds, desc=f"EuroParl {subset}"):
        if "en" in sample:
            sample_ids = tokenizer(sample['en'])['input_ids']
            en_ids.update(sample_ids)
            multiling_ids.update(sample_ids)
        if "de" in sample:
            multiling_ids.update(tokenizer(sample['de'])['input_ids'])
        if "es" in sample:
            multiling_ids.update(tokenizer(sample['es'])['input_ids'])

# %%
len(multiling_ids)

# %%
len(en_ids)

# %%
len(to_prune)

# %%


# %%
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

# %%
out_dict = {
    "pruned_english": sorted(en_ids | {0}), # Add the 0 padding token
    "pruned_multiling": sorted(multiling_ids | {0}),
    "amr_symbols": [{"token": w, "id": i, "embed": embed_id} for (i, (w, embed_id)) in enumerate(zip(amr_symbols, amr_embeddings))]
}

# %%
with open(out_path, 'w') as w:
    json.dump(out_dict, w, indent=2)

# %%




# Unsupervised AMR
Training of an AMR parser without labelled samples

## Dependencies

```bash
conda env create -f environment.yml
conda activate unsup_amr
```

## Vocabulary Generation

```
python -m unsupamr.make_t5_vocab --help
python -m unsupamr.make_t5_vocab -o vocab.json
```

File Format:

```json
{
	"pruned_english": ["<int>", "<int>", "..."],
	"pruned_multilingual": ["<int>", "<int>", "..."],
	"amr_symbols": [
		{
			"token": "<str, user readable version of the token>",
			"id": "<int, integer id for the token under the expanded vocabulary>",
			"embed": "<int or null, integer id for a token from the existing vocbulary>"
		}
	]
}
```

The `pruned_multilingual` is a list of all the vocabulary IDs that show up when we tokenize the EuroParl English, Spanish, and German data with the given tokenizer.
We can ignore this for now. It will only be necessary to use this if we need to prune the number of embeddings in the encoder model.

The union of `pruned_english` and the IDs of the `amr_symbols` comprises the output vocabulary of our encoder and the input vocabulary of our decoder.
The IDs of `pruned_english` are the token IDs observed in the EuroParl English data, and the IDs of the `amr_symbols` are new IDs (starting from right after the tokenizer's largest vocabulary ID) representing the new AMR tokens we're adding to the vocabulary (":ARG0", "\<R3\>", ":domain", etc.).

The `nextTokens` function should always mask any IDs that are not in `pruned_english | amr_symbols`.
The reason is we don't want random words from the pretrained model's other languages showing up in our AMR graphs.

The entries in `amr_symbols` also have an "embed" field.
This indicates that the word embedding for this symbol should be initialized from the embedding of some other symbol.
For instance, the script will assign the embedding for the real word "not" to the AMR symbol ":polarity -".


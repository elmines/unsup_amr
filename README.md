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
			"embed": "<int or null, integer id for a token from the existing vocbulary>",
			"type": "<string, type of token (arg, frame, node label, etc.)>",
			"args": "<List[string], arguments this symbol takes if its type == 'frame'"
		}
	]
}
```


The union of `pruned_english` and the IDs of the `amr_symbols` comprises the output vocabulary of our encoder and the input vocabulary of our decoder.
The IDs of `pruned_english` are the token IDs observed in the EuroParl English data, and the IDs of the `amr_symbols` are new IDs (starting from right after the tokenizer's largest vocabulary ID) representing the new AMR tokens we're adding to the vocabulary (":ARG0", "\<R3\>", ":domain", etc.).

The intention of the "embed" field is if we need to initialize the embedding for an AMR symbol from some other natural language token (i.e. use the embedding for "eat" to initialize the embedding for "eat-01", "eat-02", etc.).

The `nextTokens` function should always mask any IDs that are not in `pruned_english | amr_symbols`.
The reason is we don't want random words from the pretrained model's other languages showing up in our AMR graphs.

### Ignore For Now
- `pruned_multilingual`: a list of all the vocabulary IDs that show up when we tokenize the EuroParl English, Spanish, and German data with the given tokenizer. It will only be necessary to use this if we need to prune the number of embeddings in the encoder model.


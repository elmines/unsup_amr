
### Task Description: Europarl Dataset Preprocessing for Translation

#### Tasks Completed:

*   **Load Europarl Dataset**: Load the dataset for the specified language pair (e.g., English-Spanish, German-English).
*   **Tokenization**: Tokenize both the source and target sentences using a pre-trained model tokenizer (default: `bert-base-multilingual-cased`), this can be customized later.
*   **Padding and Truncation**: Apply padding and truncation to ensure uniform input size for batching.
*   **Attention Masking**: Generate attention masks to distinguish between actual tokens and padding tokens.
*   **Loss Masking**: Set padding tokens in the target sequence to `-100` to mask them during loss calculation.
*   **Create Custom Dataset**: Implement a custom PyTorch dataset to hold the tokenized data.
*   **DataLoader**: Use a custom `collate_fn` for handling batch padding and generating attention masks.

#### Additional Features Added:

*   **Sample Subset Option**: Added a `sample_subset` parameter to limit the dataset to the first 1000 examples for quicker testing.
*   **Language Pair Flexibility**: The code now allows easy specification of any source and target language pair (e.g., `en-es`, `de-en`).

#### Assumptions:

*    **"de-en" Available**: The Helsinki Europarl dataset provides a "de-en" (German to English) translation dataset, not "en-de".
*   **Target Language Handling**: The target language is flexible, but for languages with missing translations (e.g., `es` for Spanish), the code assumes a fallback mechanism (`""` if not present).

## Executing the preprocess function
```
preprocess_europarl(model_name="bert-base-multilingual-cased", source_lang="en", target_lang="es", batch_size=32, sample_subset=True)
```
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch
import spacy
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from .constants import AmrCategory
# Local
from .constants import DEFAULT_SEQ_MODEL
from .utils import remove_suffix
from collections import defaultdict
pos_model = spacy.load("en_core_web_sm")


class EuroparlPreprocessor:
    """
    Preprocessor for Europarl dataset.
    - Loads dataset
    - Tokenizes text
    - Applies padding and truncation
    """
    def __init__(self, model_name=DEFAULT_SEQ_MODEL, source_lang="en", target_lang="en", sample_subset=False, vocab_ext=None):
        """
        Initialize with model name and source/target languages.
        Args:
        - model_name (str): The pre-trained model name.
        - source_lang (str): The language of the input sentences (e.g., "en").
        - target_lang (str): The language of the output sentences (e.g., "es").
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        lang_datamap = {
            ("en", "en") : "en-es",
            ("es", "es") : "en-es",
            ("de", "de") : "de-en",
            ("en", "de") : "de-en",
            ("de", "en") : "de-en",
            ("es", "de") : "de-es",
            ("de", "es") : "de-es",
            ("en", "es") : "en-es",
            ("es", "en") : "es-en",
        }
        data_subset = lang_datamap[source_lang, target_lang]
        
        if sample_subset:
            self.dataset = load_dataset("Helsinki-NLP/europarl", data_subset)['train'].select(range(5000))  # Sample subset
        else:
            self.dataset = load_dataset("Helsinki-NLP/europarl", data_subset)['train']  # Full Dataset
            
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.vocab_ext = vocab_ext
        self.verb_frames = None
       
    def load_verb_frames(self): 
        self.verb_frames = defaultdict(list)
        for amr_symbol in self.vocab_ext.amr_symbols:
            if amr_symbol.category == AmrCategory.FRAME:
                x = remove_suffix(amr_symbol.token)
                self.verb_frames[x].append(amr_symbol.id)

    def preprocess(self, sample: Dict) -> Dict:
        """Tokenizes input and target sentences."""
        input_text = sample["translation"][self.source_lang]  # Source language input
        target_text = sample["translation"][self.target_lang]  # Target language translation
        verb_frames_ids = []
        if self.verb_frames is None:
            self.load_verb_frames()
        
        pos_text = pos_model(input_text)
        for text in pos_text:
            if text.pos_ == "VERB" and text in self.verb_frames.keys():
                verb_frames_ids.extend(self.verb_frames[text])

        verb_frames_ids = torch.tensor(verb_frames_ids, dtype=torch.long)

        return {
            "input_ids": self.tokenizer(input_text, padding="max_length", truncation=True, return_tensors="pt")["input_ids"],
            "target_ids": self.tokenizer(target_text, padding="max_length", truncation=True, return_tensors="pt")["input_ids"],
            "verb_frames_ids": verb_frames_ids
        }

    def get_tokenized_dataset(self):
        """Applies tokenization to the dataset."""
        return self.dataset.map(lambda sample: self.preprocess(sample))


def collate_fn(tokenizer: PreTrainedTokenizerFast, samples: List[Dict]) -> Dict:
    """
    Custom collate function for PyTorch DataLoader.
    - Pads sequences
    - Creates attention mask
    - Sets padding tokens in target sequences to -100 for loss masking
    """
    token_padding = tokenizer.pad_token_id
    input_ids = [torch.squeeze(s['input_ids'], 0) for s in samples]
    target_ids = [torch.squeeze(s['target_ids'], 0) for s in samples]

    batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=token_padding),
        'target_ids': torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=token_padding),
        #list of ints, verb frames ids, add padding token
    }
    batch['attention_mask'] = batch['input_ids'] != token_padding
    batch['target_ids'][batch['target_ids'] == token_padding] = -100  # Mask padding tokens

    return batch

class EuroparlTranslationDataset(Dataset):
    """Custom PyTorch Dataset for tokenized Europarl data."""
    def __init__(self, tokenized_dataset):
        self.dataset = tokenized_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"]).squeeze(0),
            "target_ids": torch.tensor(sample["target_ids"]).squeeze(0),
        }

def preprocess_europarl(model_name=DEFAULT_SEQ_MODEL, source_lang="en", target_lang="es", batch_size=32, sample_subset=False):
    """
    Run the Europarl preprocessing, tokenization, and DataLoader creation.
    
    Args:
    - model_name (str): The model name to use for tokenization.
    - source_lang (str): The language of the source text (e.g., "en").
    - target_lang (str): The language of the target text (e.g., "es").
    - batch_size (int): The batch size for DataLoader.
    - sample_subet (bool): Whether to use a subset of data (to test the use-case) or not.
    """
    # Initialize preprocessor
    preprocessor = EuroparlPreprocessor(model_name=model_name, source_lang=source_lang, target_lang=target_lang, sample_subset=sample_subset)
    tokenized_dataset = preprocessor.get_tokenized_dataset()
    
    # Create dataset and dataloader
    train_dataset = EuroparlTranslationDataset(tokenized_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, 
                              collate_fn=lambda batch: collate_fn(preprocessor.tokenizer, batch))
    
    # Test batch (displaying input and target shapes)
    for batch in train_loader:
        print("Input Batch Shape:", batch["input_ids"].shape)
        print("Target Batch Shape:", batch["target_ids"].shape)
        break  # Only print the first batch for quick test


"""Executer"""

# preprocess_europarl(model_name="bert-base-multilingual-cased", source_lang="en", target_lang="es", batch_size=32, sample_subset=True)
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import torch
import spacy 
import os
from .utils import remove_suffix
from collections import defaultdict
pos_model = spacy.load("en_core_web_sm")
from .constants import AMR_DATA_DIR, AmrCategory

class AMRPreprocessor:
    """
    Preprocessor for AMR datasets (3.0 and 2.0).
    - Parses raw sentences from AMR files
    - Tokenizes sentences
    - Returns input_ids for model evaluation
    """
    def __init__(self, model_name="bert-base-multilingual-cased", amr_version="3.0", data_dir=AMR_DATA_DIR):
        """
        Initialize the preprocessor.
        
        Args:
        - model_name (str): Pre-trained model name for tokenization.
        - amr_version (str): The version of AMR (3.0 or 2.0).
        - data_dir (str): Directory where AMR dataset files are stored.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.amr_version = amr_version
        self.data_dir = data_dir
        self.vocab_ext = None
        self.verb_frames = None
        
    def load_verb_frames(self): 
        self.verb_frames = defaultdict(list)
        for amr_symbol in self.vocab_ext.amr_symbols:
            if amr_symbol.category == AmrCategory.FRAME:
                self.verb_frames[remove_suffix(amr_symbol.token)].append(amr_symbol.id)


    def preprocess(self, sentence: str) -> torch.Tensor:
        """Tokenizes the raw sentence and returns input_ids."""
        if self.verb_frames is None:
            self.load_verb_frames()
        encoding = self.tokenizer(sentence, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
        verb_frame_ids = []
        pos_text = pos_model(sentence)
        print(self.verb_frames.keys())
        for text in pos_text:
            if text.pos_ == "VERB" and text in self.verb_frames.keys():
                verb_frame_ids.extend(self.verb_frames[text])

        verb_frame_ids = torch.tensor(verb_frame_ids, dtype=torch.long)
        encoding["verb_frame_ids"] = verb_frame_ids
        return encoding

    def process_amr_3(self):
        """Process AMR 3.0 files and extract raw sentences."""
        sentences = []
        # Path for AMR 3.0 files
        amr_3_path = f"{self.data_dir}/amr_annotation_3.0/data/amrs/split/test"
 

        # Iterate through files in the directory
        for file_name in sorted(os.listdir(amr_3_path)):
           
            file_path = os.path.join(amr_3_path, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    # Extract sentences that start with "# :: sent"
                    if line.startswith("# ::snt"):
                        sentence = line.strip().split("# ::snt")[-1]
                        sentences.append(sentence)

        return sentences

    def process_amr_2(self):
        """Process AMR 2.0 files and extract raw sentences."""
        sentences = []
        # Path for AMR 2.0 files
        amr_2_path = f"{self.data_dir}/amr_2-four_translations/data"

        # Iterate through files in the directory
        for file_name in sorted(os.listdir(amr_2_path)):
            file_path = os.path.join(amr_2_path, file_name)
            with open(file_path, "r") as file:
                for line in file:
                    sentence = line.strip()  # Each line is a sentence
                    sentences.append(sentence)
            
        return sentences

    def get_input_ids(self):
        """Preprocess sentences and return input_ids."""
        sentences = []
        if self.amr_version == "3.0":
            sentences = self.process_amr_3()
        elif self.amr_version == "2.0":
            sentences = self.process_amr_2()

        input_ids = []
        for sentence in sentences:
            input_ids.append(self.preprocess(sentence))

        return input_ids


class AMRInputIDDataset(torch.utils.data.Dataset):
    """Custom PyTorch Dataset for AMR input ids."""
    def __init__(self, input_ids):
        self.dataset = input_ids

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.dataset[idx])
        }

def amr_collate_fn(tokenizer: PreTrainedTokenizerFast, samples: List[Dict]) -> Dict:
    """
    Custom collate function for PyTorch DataLoader.
    - Pads sequences
    - Creates attention mask
    - Sets padding tokens in target sequences to -100 for loss masking
    """
    token_padding = tokenizer.pad_token_id
    input_ids = [torch.squeeze(s['input_ids'], 0) for s in samples]
    verb_frame_ids = [s['verb_frame_ids'] for s in samples]
    batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=token_padding),
        'verb_frame_ids': torch.nn.utils.rnn.pad_sequence(verb_frame_ids, batch_first=True, padding_value=0)
    }
    batch['attention_mask'] = batch['input_ids'] != token_padding

    return batch


def amr_preprocess(amr_version, data_dir):
    amr_preprocessor = AMRPreprocessor(amr_version, data_dir)
    tokenized_sentences = amr_preprocessor.get_input_ids()

    # Print the first 5 tokenized sentences for inspection
    print(f"Processed {len(tokenized_sentences)} sentences.")
    for tokenized in tokenized_sentences[:5]:  # Print the first 5 tokenized sentences
        print(tokenized)

if __name__ == "__main__":
    pass
    # amr_preprocess(amr_version="3.0", data_dir="./amr_annotation_3.0_LDC2020T02")    
    # amr_preprocess(amr_version="2.0", data_dir="./amr_2-four_translations_LDC2020T07")
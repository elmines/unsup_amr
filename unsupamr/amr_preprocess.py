from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os

class AMRPreprocessor:
    """
    Preprocessor for AMR datasets (3.0 and 2.0).
    - Parses raw sentences from AMR files
    - Tokenizes sentences
    - Returns input_ids for model evaluation
    """
    def __init__(self, model_name="bert-base-multilingual-cased", amr_version="3.0", data_dir=None):
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

    def preprocess(self, sentence: str) -> torch.Tensor:
        """Tokenizes the raw sentence and returns input_ids."""
        return self.tokenizer(sentence, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]

    def process_amr_3(self):
        """Process AMR 3.0 files and extract raw sentences."""
        sentences = []
        # Path for AMR 3.0 files
        amr_3_path = f"{self.data_dir}/amr_annotation_3.0/data/amrs/split/test"

        # Iterate through files in the directory
        for file_name in os.listdir(amr_3_path):
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
        for file_name in os.listdir(amr_2_path):
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

def amr_preprocess(amr_version, data_dir):
    amr_preprocessor = AMRPreprocessor(amr_version, data_dir) 
    tokenized_sentences = amr_preprocessor.get_input_ids()

    # Print the first 5 tokenized sentences for inspection
    print(f"Processed {len(tokenized_sentences)} sentences.")
    for tokenized in tokenized_sentences[:5]:  # Print the first 5 tokenized sentences
        print(tokenized)

if __name__ == "__main__":

    # amr_preprocess(amr_version="3.0", data_dir="./amr_annotation_3.0_LDC2020T02")    
    # amr_preprocess(amr_version="2.0", data_dir="./amr_2-four_translations_LDC2020T07")
# 3rd Party
import lightning as L
from torch.utils.data import DataLoader
# Local
from .preprocess import EuroparlPreprocessor, EuroparlTranslationDataset, collate_fn
from .constants import DEFAULT_SEQ_MODEL

class EuroParlDataModule(L.LightningDataModule):

    def __init__(self,
                 source_lang: str,
                 target_lang: str,
                 pretrained_model: str = DEFAULT_SEQ_MODEL,
                 batch_size: int = 4):
        super().__init__()
        self.save_hyperparameters()

        self.train_loader: DataLoader = None


    def setup(self, stage: str):
        if stage != 'fit':
            raise ValueError("EuroParl data is only used for training")

        # Initialize preprocessor
        preprocessor = EuroparlPreprocessor(model_name=self.hparams.pretrained_model,
                                            source_lang=self.hparams.source_lang,
                                            target_lang=self.hparams.target_lang,
                                            sample_subset=True) # TODO: Set to False when done debugging
        tokenized_dataset = preprocessor.get_tokenized_dataset()
    
        # Create dataset and dataloader
        train_dataset = EuroparlTranslationDataset(tokenized_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=1, 
                                  collate_fn=lambda batch: collate_fn(preprocessor.tokenizer, batch))

    def train_dataloader(self):
        return self.train_loader
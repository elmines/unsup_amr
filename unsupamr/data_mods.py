# 3rd Party
import lightning as L
from torch.utils.data import DataLoader, Subset
# Local
from .preprocess import EuroparlPreprocessor, EuroparlTranslationDataset, collate_fn
from .constants import DEFAULT_SEQ_MODEL, AMR_DATA_DIR, DEFAULT_BATCH_SIZE
from .amr_preprocess import AMRInputIDDataset, AMRPreprocessor, amr_collate_fn

class EuroParlDataModule(L.LightningDataModule):

    def __init__(self,
                 source_lang: str = 'en',
                 target_lang: str = 'en',
                 pretrained_model: str = DEFAULT_SEQ_MODEL,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 debug_subset: bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.train_loader: DataLoader = None
        self.vocab_ext = None

    def setup(self, stage: str):
        if stage != 'fit':
            raise ValueError("EuroParl data is only used for training")

        # Initialize preprocessor
        preprocessor = EuroparlPreprocessor(model_name=self.hparams.pretrained_model,
                                            source_lang=self.hparams.source_lang,
                                            target_lang=self.hparams.target_lang,
                                            sample_subset=self.hparams.debug_subset, 
                                            vocab_ext=self.vocab_ext)
        tokenized_dataset = preprocessor.get_tokenized_dataset()
    
        # Create dataset and dataloader
        train_dataset = EuroparlTranslationDataset(tokenized_dataset)
        self.train_loader = DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=1, 
                                  collate_fn=lambda batch: collate_fn(preprocessor.tokenizer, batch))

        # Tiny test batch (not even shuffled) just to see how the model is doing
        # val_dataset = train_dataset[:self.hparams.batch_size]
        val_dataset = Subset(train_dataset, list(range(self.hparams.batch_size)) )
        self.val_loader = DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=1, 
                                  collate_fn=lambda batch: collate_fn(preprocessor.tokenizer, batch))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


class AMRDataModule(L.LightningDataModule):
    def __init__(self,
                 amr_version: str = "3.0",
                 pretrained_model: str = DEFAULT_SEQ_MODEL,
                 data_dir: str = AMR_DATA_DIR,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        super().__init__()
        self.save_hyperparameters()

        self.test_loader: DataLoader = None
        self.vocab_ext = None

    def predict_dataloader(self):
        return self.test_loader

    def do_collate(self, batch):
        return amr_collate_fn(self.preprocessor.tokenizer, batch)    

    def setup(self, stage: str):
        if stage != 'predict':
            raise ValueError("AMR test data will be used only to test the predictions of T2A")

        # Initialize preprocessor
        self.preprocessor = AMRPreprocessor(
            model_name=self.hparams.pretrained_model,
            amr_version=self.hparams.amr_version,
            data_dir=self.hparams.data_dir,
            vocab_ext=self.vocab_ext
        )
        tokenized_dataset = self.preprocessor.get_input_ids()
    
        # Create dataset and dataloader
        test_dataset = AMRInputIDDataset(tokenized_dataset)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=self.do_collate
        )

    def test_dataloader(self):
        return self.test_loader
    

if __name__ == "__main__":
    fake_datmod = AMRDataModule(amr_version="3.0")
    fake_datmod.setup(stage='predict')
    for batch in fake_datmod.test_dataloader():
      print(batch)
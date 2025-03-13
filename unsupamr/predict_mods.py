import torch
import lightning as L
from transformers import MT5ForConditionalGeneration
from .t2a import T2A
from .embeddings import expand_embedding, expand_lm_head
from .utils import VocabExt, load_vocab
from .constants import DEFAULT_SEQ_MODEL, DEFAULT_MAX_GRAPH_SIZE

class PredictMod(L.LightningModule):
    def __init__(self, vocab_path: str, 
               pretrained_model: str = DEFAULT_SEQ_MODEL,
               temperature: float = 1.,
               max_graph_size: int = DEFAULT_MAX_GRAPH_SIZE):
        super().__init__()
        self.save_hyperparameters()

        vocab_ext = VocabExt.from_json(load_vocab(vocab_path))
        pretrained_a = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.embeddings = expand_embedding(pretrained_a.get_input_embeddings(), vocab_ext)
        pretrained_a.set_input_embeddings(self.embeddings)
        pretrained_a.lm_head = expand_lm_head(pretrained_a.lm_head, vocab_ext)
        self.t2a = T2A(pretrained_a, vocab_ext, temperature=temperature, max_iterations = max_graph_size)
        self.embeddings.eval()
        self.t2a.eval()


    def predict_step(self, batch, batch_idx):
        
        prob_history, pred_attention_mask = self.t2a(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask']
        )

        pred_token_ids = prob_history.argmax(dim=-1)
        

        predictions = []

        # for tokens in pred_token_ids:
        #     prediction = [self.vocab_ext.decode([token.item()]) for token in tokens]
        #     predictions.append(prediction)

        predictions = [[token.item() for token in tokens] for tokens in pred_token_ids]

        # predictions = pred_token_ids.tolist()


        return predictions  # List[List[int]]



        #return in format -> List[List[int]]
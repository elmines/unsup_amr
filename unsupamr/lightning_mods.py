# STL
import json
# 3rd Party
import lightning as L
from transformers import MT5ForConditionalGeneration
import torch
# Local
from .t2a import T2A
from .constants import DEFAULT_SEQ_MODEL, DEFAULT_MAX_GRAPH_SIZE
from .embeddings import expand_embedding, expand_lm_head, mult_embedding_lookup
from .utils import VocabExt, load_vocab


class TrainingMod(L.LightningModule):
    def __init__(self,
                 vocab_path: str,
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
        self.t2a = T2A(pretrained_a, vocab_ext, temperature=temperature, max_iterations=max_graph_size)

        self.a2t = MT5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.a2t.set_input_embeddings(self.embeddings)
        self.a2t.lm_head = expand_lm_head(self.a2t.lm_head, vocab_ext)

        self.embeddings.train()
        self.t2a.train()
        self.a2t.train()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val = 0.1, gradient_clip_algorithm = 'value'):
        """
        This is just a way to enable gradient clipping by default
        """
        return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def training_step(self, batch, batch_idx):
        prob_history, pred_attention_mask = self.t2a(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        embeddings = mult_embedding_lookup(prob_history, self.embeddings)

        output = self.a2t(inputs_embeds=embeddings,
                          attention_mask=pred_attention_mask,
                          labels=batch['target_ids'])
        loss = output.loss
        self.log("loss", loss)
        return loss

    
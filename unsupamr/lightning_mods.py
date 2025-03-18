# STL
from typing import List
# 3rd Party
import lightning as L
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
# Local
from .t2a import T2A
from .constants import DEFAULT_SEQ_MODEL, DEFAULT_MAX_GRAPH_SIZE, STOPPING_METRIC
from .embeddings import expand_embedding, expand_lm_head, mult_embedding_lookup
from .utils import VocabExt
from .postprocess import probs_to_ids, bfs_to_penman


class TrainingMod(L.LightningModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_SEQ_MODEL,
                 temperature: float = 1.,
                 max_graph_size: int = DEFAULT_MAX_GRAPH_SIZE):
        super().__init__()
        self.save_hyperparameters()

        pretrained_a = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_model)
        self.vocab_ext = VocabExt(pretrained_a, self.tokenizer)

        self.embeddings = expand_embedding(pretrained_a.get_input_embeddings(), self.vocab_ext)
        pretrained_a.set_input_embeddings(self.embeddings)
        pretrained_a.lm_head = expand_lm_head(pretrained_a.lm_head, self.vocab_ext)
        self.t2a = T2A(pretrained_a, self.vocab_ext, temperature=temperature, max_iterations=max_graph_size)

        self.a2t = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.a2t.set_input_embeddings(self.embeddings)
        self.a2t.lm_head = expand_lm_head(self.a2t.lm_head, self.vocab_ext)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val = 0.1, gradient_clip_algorithm = 'value'):
        """
        This is just a way to enable gradient clipping by default
        """
        return super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)

    def on_train_epoch_start(self):
        self.embeddings.train()
        self.t2a.train()
        self.a2t.train()
    
    def on_validation_epoch_start(self):
        self.embeddings.eval()
        self.t2a.eval()
        self.a2t.eval()

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
        self.log(STOPPING_METRIC, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        prob_history, _ = self.t2a(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        prediction_batch = probs_to_ids(prob_history)

        tokenizer = self.vocab_ext.tokenizer

        def trim_padding(seq: List[int]):
            while seq and seq[-1] == self.vocab_ext.pad_id:
                seq.pop()
            return seq


        for (text_ids, prediction_sample) in zip(batch['input_ids'], prediction_batch):
            raw_text = tokenizer.decode(text_ids, skip_special_tokens=True)

            prediction_sample = trim_padding(prediction_sample)
            dfs_tokens = self.vocab_ext.ids_to_str(prediction_sample)

            penman_text = bfs_to_penman(dfs_tokens)
            dfs_text = " ".join(dfs_tokens)

            print()
            print(f"Input : {raw_text}")
            print(f"DFS   : {dfs_text}")
            print(f"Penman: ")
            print(f"{penman_text}")
    
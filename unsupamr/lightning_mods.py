# STL

# 3rd Party
import lightning as L
from transformers import T5ForConditionalGeneration
# Local
from .t2a import T2A
from .constants import DEFAULT_SEQ_MODEL
from .embeddings import expand_embedding, expand_lm_head, mult_embedding_lookup
from .vocab import load_vocab


class TrainingMod(L.LightningModule):
    def __init__(self,
                 vocab_path: str,
                 pretrained_model: str = DEFAULT_SEQ_MODEL,
                 temperature: float = 1.):
        super().__init__()
        self.save_hyperparameters()

        vocab_ext = load_vocab(vocab_path)
        pretrained_a = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.embeddings = expand_embedding(pretrained_a.get_input_embeddings(), vocab_ext)
        pretrained_a.set_input_embeddings(self.embeddings)
        pretrained_a.lm_head = expand_lm_head(pretrained_a.lm_head, vocab_ext)
        self.t2a = T2A(pretrained_a, temperature=temperature)

        self.a2t = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.a2t.set_input_embeddings(self.mbeddings)
        self.a2t.lm_head = expand_lm_head(self.a2t.lm_head, vocab_ext)

    def training_step(self, batch, batch_idx):
        prob_history, pred_attention_mask = self.t2a(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        embeddings = mult_embedding_lookup(prob_history, self.embeddings)

        output = self.a2t(input_embeds=embeddings,
                          attention_mask=pred_attention_mask,
                          labels=batch['target_ids'])
        loss = output.loss
        self.log("loss", loss)
        return loss

    
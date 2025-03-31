# STL
from typing import List
from itertools import starmap
# 3rd Party
import lightning as L
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import torch
# Local
from .t2a import T2A
from .constants import DEFAULT_SEQ_MODEL, DEFAULT_MAX_GRAPH_SIZE, STOPPING_METRIC, DEFAULT_SMOOTHING, DEFAULT_TEMP
from .embeddings import expand_embedding, expand_lm_head, mask_lm_head
from .utils import VocabExt
from .postprocess import probs_to_ids, triple_decode


class TrainingMod(L.LightningModule):
    def __init__(self,
                 pretrained_model: str = DEFAULT_SEQ_MODEL,
                 temperature: float = DEFAULT_TEMP,
                 smoothing: float = DEFAULT_SMOOTHING,
                 max_graph_size: int = DEFAULT_MAX_GRAPH_SIZE,
                 load_old_head_weights: bool = True,
                 log_gradients: bool = False,
                 log_concept_rates: bool = False,
                 mask_a2t_head: bool = False):
        super().__init__()
        self.save_hyperparameters()

        pretrained_a = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_model)
        self.vocab_ext = VocabExt(pretrained_a, self.tokenizer)

        self.embeddings = expand_embedding(pretrained_a.get_input_embeddings(), self.vocab_ext)
        pretrained_a.set_input_embeddings(self.embeddings)
        pretrained_a.lm_head = expand_lm_head(pretrained_a.lm_head, self.vocab_ext, load_old_head_weights=load_old_head_weights)
        self.t2a = T2A(pretrained_a, self.vocab_ext, temperature=temperature, smoothing=smoothing, max_iterations=max_graph_size,
                       logger=self.log if log_concept_rates else None)

        self.a2t = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.a2t.set_input_embeddings(self.embeddings)

        if mask_a2t_head:
            self.a2t.lm_head = mask_lm_head(self.a2t.lm_head, self.vocab_ext)
        else:
            self.a2t.lm_head = expand_lm_head(self.a2t.lm_head, self.vocab_ext)

        self.total_sets = 0
        self.total_params = 0
        with torch.no_grad():
            for param in self.parameters():
                if param.requires_grad:
                    self.total_params += int(param.numel())
                    self.total_sets += 1
        print(self.total_params)
        print(self.total_sets)

        self.large_grad_thresh = 0.1

    def on_before_optimizer_step(self, optimizer):
        if not self.hparams.log_gradients:
            return 
        with torch.no_grad():
            embedding_mat = self.embeddings.weight
            embedding_grad = embedding_mat.grad
            updated_embeddings = 0 if embedding_grad is None else torch.sum(torch.any(torch.abs(embedding_grad) >= 1e-6, dim=-1))
            percent_updated = updated_embeddings / embedding_mat.shape[0]
            self.log("gradient/embeds_updated", percent_updated)


            max_grad_comp = torch.tensor(0., dtype=torch.float, device=embedding_mat.device)
            updated_sets = torch.tensor(0, dtype=torch.long, device=embedding_mat.device)
            updated_params = torch.tensor(0, dtype=torch.long, device=embedding_mat.device)
            gradient_norm_sq = torch.tensor(0., device=embedding_mat.device)
            large_grad = torch.tensor(0, dtype=torch.long, device=embedding_mat.device)

            for parameter in self.parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    gradient_norm_sq += parameter.grad.data.norm(2)**2
                    grad_abs = torch.abs(parameter.grad)
                    params_inc = torch.sum(grad_abs >= 1e-6)
                    updated_params += params_inc
                    if params_inc:
                        updated_sets += 1
                    large_grad += torch.sum(grad_abs >= self.large_grad_thresh)
                    max_grad_comp = torch.maximum(max_grad_comp, torch.max(grad_abs))

            self.log("gradient/norm", torch.sqrt(gradient_norm_sq))
            self.log("gradient/updated_scalars", updated_params / self.total_params)
            self.log("gradient/updated_layers", updated_sets / self.total_sets)
            self.log("gradient/max_grad_comp", max_grad_comp)
            self.log("gradient/large_grad", large_grad / self.total_params)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def on_train_epoch_start(self):
        self.embeddings.train()
        self.t2a.train()
        self.a2t.train()
    
    def on_validation_epoch_start(self):
        self.embeddings.eval()
        self.t2a.eval()
        self.a2t.eval()

    def training_step(self, batch, batch_idx):
        prob_history, embeddings, pred_attention_mask = self.t2a(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )

        output = self.a2t(inputs_embeds=embeddings,
                          attention_mask=pred_attention_mask,
                          labels=batch['target_ids'])
        loss = output.loss
        self.log(STOPPING_METRIC, loss)
        return loss

    def validation_step(self, batch, batch_idx):
        prob_history, _, _ = self.t2a(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        prediction_batch = probs_to_ids(prob_history)
        text_ids = map(lambda t: t.tolist(), batch['input_ids'])
        triples = starmap(lambda i, p: triple_decode(i, p, self.vocab_ext), zip(text_ids, prediction_batch))
        for (raw_text, dfs_text, penman_text) in triples:
            print()
            print(f"Input : {raw_text}")
            print(f"DFS   : {dfs_text}")
            print(f"Penman: ")
            print(f"{penman_text}")
    
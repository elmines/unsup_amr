# STL
import os
import glob
from itertools import starmap
from typing import List, Optional
# 3rd Party
import torch
import lightning as L
from transformers import T5ForConditionalGeneration, T5TokenizerFast
# Local
from .t2a import T2A
from .embeddings import expand_embedding, expand_lm_head
from .utils import VocabExt
from .constants import DEFAULT_SEQ_MODEL, DEFAULT_MAX_GRAPH_SIZE, DEFAULT_SMOOTHING, DEFAULT_TEMP
from .postprocess import triple_decode, probs_to_ids

class PredictMod(L.LightningModule):
    def __init__(self, 
               version: Optional[str] = None,
               pretrained_model: str = DEFAULT_SEQ_MODEL,
               temperature: float = DEFAULT_TEMP,
               smoothing: float = DEFAULT_SMOOTHING,
               max_graph_size: int = DEFAULT_MAX_GRAPH_SIZE):
        super().__init__()
        self.save_hyperparameters()

        pretrained_a = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.vocab_ext = VocabExt(pretrained_a, T5TokenizerFast.from_pretrained(pretrained_model))
        self.embeddings = expand_embedding(pretrained_a.get_input_embeddings(), self.vocab_ext)
        pretrained_a.set_input_embeddings(self.embeddings)
        pretrained_a.lm_head = expand_lm_head(pretrained_a.lm_head, self.vocab_ext)
        self.t2a = T2A(pretrained_a, self.vocab_ext, temperature=temperature, smoothing=smoothing, max_iterations = max_graph_size)
        self.embeddings.eval()
        self.t2a.eval()

        if version is None:
            print("--model.version not specified. Using randomly initialized weights for the evaluation")
            return

        model_dir = os.path.join(os.path.dirname(__file__), "..", "lightning_logs", version)
        if not os.path.exists(model_dir):
            raise ValueError(f"Model directory {model_dir} does not exist")
        ckpt_pattern = os.path.join(model_dir, "checkpoints", "best-*.ckpt")
        ckpt_paths = glob.glob(ckpt_pattern)
        if not ckpt_paths:
            raise ValueError(f"No checkpoint path matching {ckpt_pattern}")
        elif len(ckpt_paths) > 1:
            raise ValueError(f"Found multiple candidate checkpoint paths: {ckpt_paths}")
        best_checkpoint = torch.load(ckpt_paths[0])
        old_state = self.state_dict()
        matching_state_dict = {k: v for k, v in best_checkpoint['state_dict'].items() if k in old_state}
        missing_keys = set(old_state) - set(matching_state_dict)
        if missing_keys:
            raise ValueError(f"Checkpoint missing weights {missing_keys}")
        self.load_state_dict(matching_state_dict)

    def predict_step(self, batch, batch_idx) -> List[str]:
        prob_history, _, _ = self.t2a(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        prediction_batch = probs_to_ids(prob_history)
        text_ids = map(lambda t: t.tolist(), batch['input_ids'])
        triples = starmap(lambda i, p: triple_decode(i, p, self.vocab_ext), zip(text_ids, prediction_batch))
        penman_preds = []
        for (raw_text, dfs_text, penman_text) in triples:
            print()
            print(f"Input : {raw_text}")
            print(f"DFS   : {dfs_text}")
            print(f"Penman: ")
            print(f"{penman_text}")

            penman_preds.append(penman_text)
        return penman_preds

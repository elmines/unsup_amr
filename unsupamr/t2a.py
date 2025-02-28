# STL
from typing import Tuple
# 3rd Party
import torch
from transformers.models.t5.modeling_t5 import T5Stack
from transformers import T5ForConditionalGeneration
# Local
from .embeddings import mult_embedding_lookup, expand_embedding
from .vocab import VocabExt
from .next_tokens import NextTokens


class T2A(torch.nn.Module):
    """
    Our Text-to-AMR module, also called our 'Encoder'.
    """

    MAX_ITERATIONS = 128

    def __init__(self,
                 pretrained: T5ForConditionalGeneration,
                 vocab_ext: VocabExt,
                 temperature: float = 1.):

        self.config = pretrained.config
        self.pad_token_id: int = self.config.pad_token_id
        self.eos_token_id: int = self.config.eos_token_id
        self.vocab_ext = vocab_ext

        self.expanded_embeddings = expand_embedding(pretrained.shared, vocab_ext)
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.encoder.set_input_embeddings(self.expanded_embeddings)
        self.decoder.set_input_embeddings(self.expanded_embeddings)
        self.lm_head = T2A.expand_lm_head(pretrained.lm_head, vocab_ext)

        self.temperature = temperature


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        n_samples = input_ids.shape[0]

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs[0]

        pad_ids = torch.full([n_samples, 1], fill_value=self.pad_token_id, device=input_ids.device)
        embeddings = self.expanded_embeddings(pad_ids)

        trackers = [NextTokens(self.vocab_ext, self.pad_token_id, self.eos_token_id) for _ in range(n_samples)]

        prob_history = []
        pred_history = []

        for _ in range(T2A.MAX_ITERATIONS):

            # TODO: Need an attention mask for the decoders' input?
            decoder_outputs = self.decoder(
                inputs_embeds=embeddings,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
            )
            sequence_output = decoder_outputs[0]

            masks = torch.stack([t.next_tokens() for t in trackers], dim=0)
            raw_logits = self.lm_head(sequence_output)
            masked_logits = raw_logits + masks
            scaled_logits = masked_logits / self.temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            prob_history.append(probs)

            with torch.no_grad():
                preds = torch.argmax(probs, dim=-1)
                pred_history.append(preds)
                for (pred, tracker) in zip(preds, trackers):
                    tracker.record(int(pred))
                is_finished = torch.all(torch.logical_or(preds == self.eos_token_id, preds == self.pad_token_id))
            new_embeddings = mult_embedding_lookup(probs, self.expanded_embeddings)
            embeddings = torch.concatenate([embeddings, torch.unsqueeze(new_embeddings, 1)], dim=1)

            if is_finished:
                break

        prob_history = torch.stack(prob_history, dim=1)
        with torch.no_grad():
            pred_history = torch.stack(pred_history, dim=1)
            pred_attention_mask = pred_history == self.pad_token_id
        return prob_history, pred_attention_mask

    @staticmethod
    def expand_lm_head(head: torch.nn.Linear, vocab: VocabExt) -> torch.nn.Linear:
        """
        TODO: @Divyam
        """
        return head
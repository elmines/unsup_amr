# STL
from typing import Tuple
# 3rd Party
import torch
from transformers import T5ForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache
# Local
from .constants import DEFAULT_MAX_GRAPH_SIZE, DEFAULT_TEMP, DEFAULT_SMOOTHING
from .embeddings import mult_embedding_lookup
from .next_token import NextTokens
from .utils import VocabExt

class T2A(torch.nn.Module):
    """
    Our Text-to-AMR module, also called our 'Encoder'.
    """

    def __init__(self,
                 pretrained: T5ForConditionalGeneration,
                 vocab_ext: VocabExt,
                 temperature: float = DEFAULT_TEMP,
                 smoothing: float = DEFAULT_SMOOTHING,
                 max_iterations: int = DEFAULT_MAX_GRAPH_SIZE):
        super().__init__()
        self.config = pretrained.config
        self.pad_token_id: int = self.config.pad_token_id
        self.eos_token_id: int = self.config.eos_token_id
        self.encoder = pretrained.encoder
        self.decoder = pretrained.decoder
        self.embeddings = self.encoder.get_input_embeddings()
        self.lm_head = pretrained.lm_head
        self.vocab_ext = vocab_ext
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.smoothing = smoothing

        assert self.encoder.get_input_embeddings() is self.decoder.get_input_embeddings()

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor, verb_frame_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        n_samples = input_ids.shape[0]

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        encoder_hidden_states = encoder_outputs[0]

        # The decoder is used to getting padding as its first token
        pad_ids = torch.full([n_samples, 1], fill_value=self.pad_token_id, device=input_ids.device)
        embeddings = self.embeddings(pad_ids)

        trackers = [NextTokens(self.vocab_ext, sample_verb_frame_ids) for sample_verb_frame_ids in verb_frame_ids]

        prob_history = []
        pred_history = []
        embed_history = []

        past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        for _ in range(self.max_iterations):

            # We shouldn't need a causal attention mask here.
            # The decoder has no future tokens to predict
            decoder_outputs = self.decoder(
                inputs_embeds=embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = decoder_outputs.past_key_values

            sequence_output = torch.squeeze(decoder_outputs.last_hidden_state[:, -1]) # Project just the most recent hidden state
            raw_logits = self.lm_head(sequence_output)

            masks = torch.stack([t.nextTokens() for t in trackers], dim=0).to(raw_logits.device)
            masked_logits = raw_logits + masks
            scaled_logits = masked_logits / self.temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            prob_history.append(probs)

            with torch.no_grad():
                preds = torch.argmax(probs, dim=-1)
                pred_history.append(preds)
                for (pred, tracker) in zip(preds, trackers):
                    tracker.nextTokens(int(pred))
                is_finished = torch.all(torch.logical_or(preds == self.eos_token_id, preds == self.pad_token_id))
            new_embeddings = mult_embedding_lookup(probs, self.embeddings, smoothing=self.smoothing)
            embed_history.append(new_embeddings)
            embeddings = torch.unsqueeze(new_embeddings, dim=-2)

            if is_finished:
                break
        else:
            print("Warning: hit maximum sequence length")

        prob_history = torch.stack(prob_history, dim=1)
        embed_history = torch.stack(embed_history, dim=1)
        with torch.no_grad():
            pred_history = torch.stack(pred_history, dim=1)
            pred_attention_mask = pred_history == self.pad_token_id
        return prob_history, embed_history, pred_attention_mask


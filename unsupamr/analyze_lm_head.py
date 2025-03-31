from transformers import T5ForConditionalGeneration, T5TokenizerFast
# Local
from .t2a import T2A
from .constants import DEFAULT_SEQ_MODEL
from .utils import VocabExt
from .embeddings import expand_lm_head

if __name__ == "__main__":
    pretrained_mod = T5ForConditionalGeneration.from_pretrained(DEFAULT_SEQ_MODEL)
    tokenizer = T5TokenizerFast.from_pretrained(DEFAULT_SEQ_MODEL)
    vocab_ext = VocabExt(pretrained_mod, tokenizer)
    new_lm_head = expand_lm_head(pretrained_mod.lm_head, vocab_ext, )
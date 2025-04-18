import enum
import os

@enum.unique
class AmrCategory(enum.Enum):
    FRAME = "frame"
    ARG = "arg"
    INV_ARG = "arg-of"

    POLARITY = "polarity"
    DOMAIN = "domain"
    INV_DOMAIN = "domain-of"
    POSS = "poss"
    INV_POSS = "poss-of"

    EDGE = "edge"
    CONJ = "conjunction"
    LABEL = "label"
    OPTION = "option"

    UNKNOWN = "amr-unknown"
    STOP = "stop"

DEFAULT_SEQ_MODEL = "google-t5/t5-small"

DEFAULT_PROPBANK = os.path.join(os.path.dirname(__file__), "..", "propbank-amr-frame-arg-descr.txt")

DEFAULT_BATCH_SIZE = 4

DEFAULT_MAX_GRAPH_SIZE = 64

DEFAULT_TEMP = 1.

DEFAULT_SMOOTHING = 0.

EUROPARL_URI = "Helsinki-NLP/europarl"

T5_SEP = (b'\xe2\x96\x81').decode()
"""
Separator character used by T5 tokenizers.
Looks like an underscore but isn't.
"""

AMR_DATA_DIR = 'amr_data'
STOPPING_METRIC = "loss"

LIGHTNING_LOGS_DIR = 'lightning_logs'
CHECKPOINT_DIR = 'checkpoints'
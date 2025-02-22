import enum

@enum.unique
class AmrToken(enum.Enum):
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

DEFAULT_SEQ_MODEL = "google/mt5-small"

EUROPARL_URI = "Helsinki-NLP/europarl"

T5_SEP = (b'\xe2\x96\x81').decode()
"""
Separator character used by T5 tokenizers.
Looks like an underscore but isn't.
"""
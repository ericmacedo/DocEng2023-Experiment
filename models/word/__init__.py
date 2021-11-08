from enum import Enum

class WordModels(Enum):
    BAG_OF_WORDS = "BagOfWords"
    WORD_2_VEC   = "Word2Vec"
    FAST_TEXT    = "FastText"
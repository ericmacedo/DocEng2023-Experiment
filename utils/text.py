from nltk.corpus import stopwords, wordnet
from nltk import pos_tag, word_tokenize
from nltk.stem import WordNetLemmatizer
from functools import lru_cache
import string
import re

def process_text(data: str, **kwargs) -> str:
    @lru_cache(maxsize=None)
    def strip_tags(text: str) -> str:
        p = re.compile(r'<.*?>')
        return p.sub('', text)

    @lru_cache(maxsize=None)
    def get_wordnet_pos(word: str):
        tag = pos_tag([word])[0][1][0].upper()
        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    # Kwargs
    stop_words = kwargs.get("stop_words", [])
    deep = kwargs.get("deep", False)

    with open('./stopwords.txt', 'r') as f:
        stop_words_file = [line.strip() for line in f]

    punctuation = r"[{0}]".format(re.sub(r"[-']", "", string.punctuation))

    stop_words = [*set(stopwords.words("english"))
                  .union(stop_words)
                  .union(["'s", "'ll", "n't", "'d", "'ve", "'m", "'re", "'"])
                  .union(stop_words_file)]

    # Lowercase
    data = data.lower() if deep else data

    # Strip tags
    data = strip_tags(data)

    # Symbols
    data = re.sub(r'[^\x00-\xb7f\xc0-\xff]', r' ', data)

    # Links
    data = re.sub(r'https?:\/\/.*[\r\n]*', '', data)

    # line breaks
    data = re.sub('-\n', r'', data)

    # Punctuation
    data = re.sub(punctuation, " ", data) if deep else data

    # tokenization
    data = " ".join(word_tokenize(data)) if deep else data

    # Numeral ranges
    data = re.sub(r'\d+-\d+', "", data)

    # Numerics
    data = [
        re.sub(r"^\d+$", r"", i)
        for i in re.findall(r"\S+", data)
    ] if deep else re.findall(r"\S+", data)

    # Remove extra characteres
    data = [*filter(lambda x: len(x) > 2, data)]

    lemmatizer = WordNetLemmatizer()
    tokens = [
        lemmatizer.lemmatize(
            token, get_wordnet_pos(token)
        ) for token in data
        if not token in stop_words] if deep else data

    return " ".join(tokens).strip()

def synonyms(word: str) -> list:
    word = word.replace("-", "_")
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    return [*set(synonyms)]
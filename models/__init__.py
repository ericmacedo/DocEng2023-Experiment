from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List, Iterable
from typing import List, Any
from yaspin import yaspin
from enum import Enum
import pandas as pd
import numpy as np
import pickle

class ModelType(Enum):
    DOCUMENT = "document"
    WORD = "word"


models = [
    {"name": "BagOfWords", "model_type": ModelType.DOCUMENT},
    {"name": "Doc2Vec", "model_type": ModelType.DOCUMENT},
    {"name": "BERT", "model_type": ModelType.DOCUMENT},
    {"name": "BagOfWords", "model_type": ModelType.WORD},
    {"name": "Word2Vec", "model_type": ModelType.WORD},
    {"name": "FastText", "model_type": ModelType.WORD}]

@dataclass(init=False)
class Data:
    name: str
    n: int
    ids: List[Any]
    content: List[str]
    processed: List[str]
    labels: List[int]
    k: int

    def __init__(self, data: pd.DataFrame,
                       name: str,
                       id_field: str,
                       content_fields: List[str],
                       label_field: str):
        from utils.text import process_text
        from utils import batch_processing

        self.name = name
        self.ids = data[id_field].tolist()
        self.content = data[content_fields].agg("\n".join, axis=1).tolist()
        self.labels = LabelEncoder().fit_transform(data[label_field]).tolist()
        self.n = len(data)
        self.k = len(np.unique(self.labels))

        with yaspin(text="Processing corpus...", color="cyan") as sp:
            self.processed = batch_processing(fn=process_text, deep=True,
                                              data=[*self.content])
            sp.ok("✔ Corpus processed successfully")
        
        self.save()

    def save(self):
        df = pd.DataFrame({
            "index": self.ids,
            "content": self.content,
            "processed": self.processed,
            "labels": self.labels})

        df.to_csv(path_or_buf=f"../data/{self.name}/data.csv",
                  encoding="utf-8")

        del df


class BagOfWords:
    def train(self, corpus: Iterable[str],
              index: Iterable[str] = None) -> np.array:
        allWords = list()
        vectorizer = TfidfVectorizer(min_df=2)

        tfidf_matrix = vectorizer.fit_transform(corpus)
        tfidf_feature_names = vectorizer.get_feature_names()
        tfidf_feature_names_hashmap = {}

        n, m = tfidf_matrix.shape  # (N documents, M features)
        self.index = index if index else [f"doc_{i}" for i in range(n)]

        # tfidf feature names hashmap
        for j in range(0, m):
            tfidf_feature_names_hashmap[tfidf_feature_names[j]] = j

        # filter based on the mean tf/idf
        tfidf_mean = tfidf_matrix.mean(0).mean()
        words_tfidf = tfidf_matrix.mean(0)
        for index, item in enumerate(np.nditer(words_tfidf)):
            if item > tfidf_mean:
                allWords.append(tfidf_feature_names[index])

        self.vocabulary = sorted(allWords)

        # create document term matrix (out)
        self.matrix = list()
        for j in range(n):
            self.matrix.append(list())
            tfidf_hashmap = {}
            for col in tfidf_matrix.getrow(j).nonzero()[1]:
                if tfidf_feature_names[col] in self.vocabulary:
                    tfidf_hashmap[col] = tfidf_matrix[j, col]

            for word in self.vocabulary:
                word_index = tfidf_feature_names_hashmap.get(word)
                if tfidf_feature_names_hashmap.get(word) in tfidf_hashmap:
                    self.matrix[j].append(
                        tfidf_hashmap.get(word_index))
                else:
                    self.matrix[j].append(0.0)

        self.matrix = np.array(self.matrix, dtype=np.float32)
        self.n, self.m = self.matrix.shape  # (N documents, M features)
        return self.matrix

    def __getitem__(self, item) -> np.array:
        if isinstance(item, (int, slice)):
            return self.__class__(self.matrix[item])
        elif isinstance(item, str):
            if item in self.vocabulary:
                index = self.vocabulary.index(item)
                return self.matrix[:, index]
            else:
                raise TypeError("Invalid index type")
        else:
            raise TypeError("Unsuported index. It must be int, slice or str")

    @classmethod
    def load(cls, path: str):
        return pickle.load(open(path, "rb"))

    def save(self, path: str):
        with open(path, "wb") as pkl_file:
            pickle.dump(
                obj=self,
                file=pkl_file,
                protocol=pickle.DEFAULT_PROTOCOL,
                fix_imports=True)
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from src.ranking.base import BaseRanker
from config.settings import settings

class BM25Ranker(BaseRanker):
    def __init__(self, k1: float = None, b: float = None):
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        self.bm25 = None
        self.doc_names: List[str] = []

    def fit(self, documents: List[str], doc_names: List[str]):
        self.doc_names = doc_names
        if not documents:
            self.bm25 = None
            return
        
        # Tokenize corpus for BM25 (split by whitespace)
        tokenized_corpus = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.bm25 is None or not query:
            return []

        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)

        results = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                results.append((self.doc_names[idx], float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

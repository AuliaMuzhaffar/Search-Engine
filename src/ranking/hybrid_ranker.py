from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.ranking.base import BaseRanker
from src.ranking.tfidf_ranker import TfidfRanker
from src.ranking.bm25_ranker import BM25Ranker
from config.settings import settings

class HybridRanker(BaseRanker):
    def __init__(self, tfidf_weight: float = None, bm25_weight: float = None, normalize: bool = True):
        self.tfidf_weight = tfidf_weight if tfidf_weight is not None else settings.TFIDF_WEIGHT
        self.bm25_weight = bm25_weight if bm25_weight is not None else settings.BM25_WEIGHT
        self.normalize = normalize
        self.tfidf_ranker = TfidfRanker()
        self.bm25_ranker = BM25Ranker()
        self.doc_names: List[str] = []

    def fit(self, documents: List[str], doc_names: List[str]):
        self.doc_names = doc_names
        self.tfidf_ranker.fit(documents, doc_names)
        self.bm25_ranker.fit(documents, doc_names)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self.doc_names or not query:
            return []

        # Get scores from individual rankers
        # TF-IDF
        query_vector = self.tfidf_ranker.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_ranker.tfidf_matrix).flatten()

        # BM25
        query_tokens = query.split()
        bm25_scores = self.bm25_ranker.bm25.get_scores(query_tokens)

        # Normalize BM25 if enabled (min-max normalization or divide by max)
        if self.normalize:
            max_bm25 = np.max(bm25_scores)
            min_bm25 = np.min(bm25_scores)
            if max_bm25 - min_bm25 > 0:
                bm25_scores_norm = (bm25_scores - min_bm25) / (max_bm25 - min_bm25)
            elif max_bm25 > 0:
                bm25_scores_norm = bm25_scores / max_bm25
            else:
                bm25_scores_norm = bm25_scores
        else:
            bm25_scores_norm = bm25_scores

        combined_scores = self.tfidf_weight * tfidf_scores + self.bm25_weight * bm25_scores_norm

        results = []
        for idx, score in enumerate(combined_scores):
            # Show only if we have positive signal from either
            if score > 0.0 and (tfidf_scores[idx] > 0 or bm25_scores[idx] > 0):
                results.append((
                    self.doc_names[idx], 
                    float(score)
                ))

        # Sort by combined score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_with_details(self, query: str, top_k: int = 10) -> List[Tuple[str, float, float, float]]:
        """
        Returns (doc_name, combined_score, tfidf_score, bm25_score)
        """
        if not self.doc_names or not query:
            return []

        # Get scores from individual rankers
        query_vector = self.tfidf_ranker.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vector, self.tfidf_ranker.tfidf_matrix).flatten()

        query_tokens = query.split()
        bm25_scores = self.bm25_ranker.bm25.get_scores(query_tokens)

        # Normalize
        if self.normalize:
            max_bm25 = np.max(bm25_scores)
            min_bm25 = np.min(bm25_scores)
            if max_bm25 - min_bm25 > 0:
                bm25_scores_norm = (bm25_scores - min_bm25) / (max_bm25 - min_bm25)
            elif max_bm25 > 0:
                bm25_scores_norm = bm25_scores / max_bm25
            else:
                bm25_scores_norm = bm25_scores
        else:
            bm25_scores_norm = bm25_scores

        combined_scores = self.tfidf_weight * tfidf_scores + self.bm25_weight * bm25_scores_norm

        results = []
        for idx, score in enumerate(combined_scores):
            if score > 0.0 and (tfidf_scores[idx] > 0 or bm25_scores[idx] > 0):
                results.append((
                    self.doc_names[idx], 
                    float(score), 
                    float(tfidf_scores[idx]), 
                    float(bm25_scores[idx])
                ))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

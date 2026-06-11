from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.ranking.base import BaseRanker

class TfidfRanker(BaseRanker):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.doc_names: List[str] = []

    def fit(self, documents: List[str], doc_names: List[str]):
        self.doc_names = doc_names
        if not documents:
            self.tfidf_matrix = None
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if self.tfidf_matrix is None or not query:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        results = []
        for idx, score in enumerate(similarities):
            if score > 0.0:
                results.append((self.doc_names[idx], float(score)))

        # Sort descending by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

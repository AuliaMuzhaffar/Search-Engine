from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

class BaseRanker(ABC):
    @abstractmethod
    def fit(self, documents: List[str], doc_names: List[str]):
        """
        Fits the ranker model on a list of document texts.
        """
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Searches the query and returns a list of sorted (document_name, score) tuples.
        """
        pass

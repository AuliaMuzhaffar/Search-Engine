import pytest
from src.ranking.tfidf_ranker import TfidfRanker
from src.ranking.bm25_ranker import BM25Ranker
from src.ranking.hybrid_ranker import HybridRanker

@pytest.fixture
def sample_corpus():
    documents = [
        "kucing tidur kasur",
        "anjing kejar kucing jalan",
        "burung terbang langit biru"
    ]
    doc_names = ["doc1.txt", "doc2.txt", "doc3.txt"]
    return documents, doc_names

def test_tfidf_ranker(sample_corpus):
    documents, doc_names = sample_corpus
    ranker = TfidfRanker()
    ranker.fit(documents, doc_names)
    
    results = ranker.search("kucing")
    # doc1 has 'kucing', doc2 has 'kucing', doc3 does not
    assert len(results) == 2
    assert results[0][0] in ["doc1.txt", "doc2.txt"]
    assert results[0][1] > 0.0

def test_bm25_ranker(sample_corpus):
    documents, doc_names = sample_corpus
    ranker = BM25Ranker()
    ranker.fit(documents, doc_names)
    
    results = ranker.search("anjing kucing")
    assert len(results) == 2
    # doc2 contains both, doc1 contains kucing only, doc3 contains neither
    assert results[0][0] == "doc2.txt"

def test_hybrid_ranker(sample_corpus):
    documents, doc_names = sample_corpus
    ranker = HybridRanker(tfidf_weight=0.5, bm25_weight=0.5)
    ranker.fit(documents, doc_names)
    
    results = ranker.search("kucing")
    assert len(results) == 2
    assert results[0][0] in ["doc1.txt", "doc2.txt"]

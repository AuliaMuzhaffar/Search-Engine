import pytest
from src.evaluation.metrics import (
    precision_recall_f1,
    precision_at_k,
    ndcg_at_k,
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank
)

def test_precision_recall_f1_basic():
    predictions = [1, 0, 1, 0, 1]
    ground_truth = [1, 1, 0, 0, 1]
    
    precision, recall, f1 = precision_recall_f1(predictions, ground_truth)
    
    # TP: predictions[0] == gt[0] == 1 (1) & predictions[4] == gt[4] == 1 (1) -> 2
    # FP: predictions[2] == 1 and gt[2] == 0 -> 1
    # FN: gt[1] == 1 and predictions[1] == 0 -> 1
    # Precision = 2 / 3
    # Recall = 2 / 3
    assert abs(precision - 2/3) < 1e-9
    assert abs(recall - 2/3) < 1e-9
    assert abs(f1 - 2/3) < 1e-9

def test_precision_recall_f1_zero_division():
    # Test safe division by zero
    p, r, f1 = precision_recall_f1([0, 0], [0, 0])
    assert p == 0.0
    assert r == 0.0
    assert f1 == 0.0

def test_precision_at_k():
    relevance = [1, 0, 1, 1, 0]
    assert precision_at_k(relevance, 3) == 2/3
    assert precision_at_k(relevance, 5) == 3/5
    assert precision_at_k(relevance, 0) == 0.0

def test_ndcg_at_k():
    relevance = [1, 0, 1, 0]
    # dcg = 1 / log2(2) + 0 / log2(3) + 1 / log2(4) = 1 + 0 + 0.5 = 1.5
    # ideal = [1, 1, 0, 0]
    # idcg = 1 / log2(2) + 1 / log2(3) = 1 + 1 / 1.58496 = 1 + 0.6309 = 1.6309
    ndcg = ndcg_at_k(relevance, 3)
    assert ndcg > 0.0
    assert ndcg <= 1.0

def test_average_precision():
    relevance = [1, 0, 1, 0, 0]
    # ap = (1/1 + 2/3) / 2 = (1 + 0.666) / 2 = 0.8333
    ap = average_precision(relevance)
    assert abs(ap - 5/6) < 1e-4

def test_mean_average_precision():
    all_relevance = [
        [1, 0, 1],
        [0, 1, 0]
    ]
    # Query 1: ap = (1/1 + 2/3) / 2 = 5/6
    # Query 2: ap = (1/2) / 1 = 0.5
    # map = (5/6 + 0.5) / 2 = (0.8333 + 0.5) / 2 = 0.6666
    map_score = mean_average_precision(all_relevance)
    assert abs(map_score - 2/3) < 1e-4

def test_mean_reciprocal_rank():
    all_relevance = [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    # Q1: first rel at rank 2 -> rr = 1/2
    # Q2: first rel at rank 3 -> rr = 1/3
    # Q3: first rel at rank 1 -> rr = 1/1
    # mrr = (0.5 + 0.3333 + 1.0) / 3 = 1.8333 / 3 = 0.6111
    mrr = mean_reciprocal_rank(all_relevance)
    assert abs(mrr - (1/2 + 1/3 + 1.0)/3) < 1e-4

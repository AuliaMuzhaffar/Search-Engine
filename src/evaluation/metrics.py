from typing import List
import numpy as np

def precision_recall_f1(predictions: List[int], ground_truth: List[int]) -> tuple:
    """
    Calculates Precision, Recall, and F1 Score for binary labels (1=relevant, 0=not relevant).
    Handles division-by-zero safely.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground_truth must have the same length.")

    true_positives = sum(p == gt == 1 for p, gt in zip(predictions, ground_truth))
    false_positives = sum(p == 1 and gt == 0 for p, gt in zip(predictions, ground_truth))
    false_negatives = sum(p == 0 and gt == 1 for p, gt in zip(predictions, ground_truth))

    prec_denom = true_positives + false_positives
    precision = true_positives / prec_denom if prec_denom > 0 else 0.0

    rec_denom = true_positives + false_negatives
    recall = true_positives / rec_denom if rec_denom > 0 else 0.0

    f1_denom = precision + recall
    f1_score = 2 * (precision * recall) / f1_denom if f1_denom > 0 else 0.0

    return precision, recall, f1_score

def precision_at_k(relevance: List[int], k: int) -> float:
    """
    Calculates Precision@K. relevance is a list of relevance scores (e.g. [1, 0, 1]).
    """
    if k <= 0:
        return 0.0
    rel_k = relevance[:k]
    return sum(rel_k) / k if len(rel_k) > 0 else 0.0

def dcg_at_k(relevance: List[int], k: int) -> float:
    """
    Calculates Discounted Cumulative Gain at K.
    """
    if k <= 0 or not relevance:
        return 0.0
    rel_k = np.asarray(relevance[:k], dtype=float)
    return np.sum(rel_k / np.log2(np.arange(2, rel_k.size + 2)))

def ndcg_at_k(relevance: List[int], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at K.
    """
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0.0 else 0.0

def average_precision(relevance: List[int]) -> float:
    """
    Calculates Average Precision for a single query.
    """
    if not relevance or sum(relevance) == 0:
        return 0.0
    
    precisions = []
    num_relevant = 0
    for i, rel in enumerate(relevance):
        if rel == 1:
            num_relevant += 1
            precisions.append(num_relevant / (i + 1))
            
    return sum(precisions) / sum(relevance) if sum(relevance) > 0 else 0.0

def mean_average_precision(all_relevances: List[List[int]]) -> float:
    """
    Calculates MAP across a list of query relevances.
    """
    if not all_relevances:
        return 0.0
    return sum(average_precision(rel) for rel in all_relevances) / len(all_relevances)

def mean_reciprocal_rank(all_relevances: List[List[int]]) -> float:
    """
    Calculates MRR across a list of query relevances.
    """
    if not all_relevances:
        return 0.0
    
    rr_sum = 0.0
    for relevance in all_relevances:
        for rank, rel in enumerate(relevance, start=1):
            if rel == 1:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(all_relevances)

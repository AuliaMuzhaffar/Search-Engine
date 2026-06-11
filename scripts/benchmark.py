import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from config.settings import settings
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.ranking.tfidf_ranker import TfidfRanker
from src.ranking.bm25_ranker import BM25Ranker
from src.ranking.hybrid_ranker import HybridRanker
from src.evaluation.metrics import (
    precision_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    average_precision
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger("benchmark")

def main():
    logger.info("Starting Offline Evaluation Benchmark...")
    
    # 1. Load Ground Truth
    gt_path = settings.DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        logger.error(f"Ground truth dataset not found at {gt_path}. Please create it first.")
        return
        
    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth: Dict[str, Dict[str, int]] = json.load(f)
        
    logger.info(f"Loaded ground truth with {len(ground_truth)} queries.")

    # 2. Load Processed Documents
    documents = []
    doc_names = []
    
    if not settings.PROCESSED_DIR.exists():
        logger.error("Processed articles directory not found. Run scripts/preprocess.py first.")
        return
        
    files = sorted([f for f in settings.PROCESSED_DIR.iterdir() if f.is_file() and f.suffix == '.txt'])
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                documents.append(file.read())
                doc_names.append(f.name)
        except Exception as e:
            logger.error(f"Error loading {f.name}: {e}")
            
    if not documents:
        logger.error("No documents found to fit models.")
        return
        
    logger.info(f"Loaded {len(documents)} documents for fitting rankers.")

    # 3. Fit Rankers
    tfidf_ranker = TfidfRanker()
    tfidf_ranker.fit(documents, doc_names)
    
    bm25_ranker = BM25Ranker()
    bm25_ranker.fit(documents, doc_names)
    
    hybrid_ranker = HybridRanker(tfidf_weight=0.5, bm25_weight=0.5)
    hybrid_ranker.fit(documents, doc_names)
    
    cleaner = TextCleaner()
    tokenizer = Tokenizer()
    
    # 4. Perform Evaluation
    models = {
        "TF-IDF": tfidf_ranker,
        "BM25": bm25_ranker,
        "Hybrid": hybrid_ranker
    }
    
    results = {}
    
    for model_name, ranker in models.items():
        all_p5 = []
        all_p10 = []
        all_ap = []
        all_ndcg10 = []
        all_relevances = []
        
        for raw_query, judgments in ground_truth.items():
            # Preprocess query terms just like search engine does
            cleaned = cleaner.clean(raw_query)
            preprocessed_query = " ".join(tokenizer.tokenize_and_stem(cleaned))
            
            # Search model
            search_results = ranker.search(preprocessed_query, top_k=10)
            
            # Map search results to binary relevance judgments
            binary_relevance = []
            for doc_name, _ in search_results:
                rel = judgments.get(doc_name, 0)
                binary_relevance.append(rel)
                
            # Pad with 0s if returned results < 10
            if len(binary_relevance) < 10:
                binary_relevance.extend([0] * (10 - len(binary_relevance)))
                
            all_relevances.append(binary_relevance)
            all_p5.append(precision_at_k(binary_relevance, 5))
            all_p10.append(precision_at_k(binary_relevance, 10))
            all_ap.append(average_precision(binary_relevance))
            all_ndcg10.append(ndcg_at_k(binary_relevance, 10))
            
        results[model_name] = {
            "P@5": np.mean(all_p5),
            "P@10": np.mean(all_p10),
            "MAP": np.mean(all_ap),
            "NDCG@10": np.mean(all_ndcg10),
            "MRR": mean_reciprocal_rank(all_relevances)
        }
        
    # 5. Output Markdown Table
    print("\n")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│            Offline Evaluation Benchmark                 │")
    print("├──────────┬──────────┬──────────┬────────────┬───────────┤")
    print("│ Metric   │ TF-IDF   │ BM25     │ Hybrid     │ Best      │")
    print("├──────────┼──────────┼──────────┼────────────┼───────────┤")
    
    metrics = ["P@5", "P@10", "MAP", "NDCG@10", "MRR"]
    
    # Store rows for easy markdown display
    markdown_rows = []
    markdown_rows.append("| Metric | TF-IDF | BM25 | Hybrid | Best |")
    markdown_rows.append("| :--- | :--- | :--- | :--- | :--- |")
    
    for metric in metrics:
        val_tfidf = results["TF-IDF"][metric]
        val_bm25 = results["BM25"][metric]
        val_hybrid = results["Hybrid"][metric]
        
        # Find best model
        scores = {"TF-IDF": val_tfidf, "BM25": val_bm25, "Hybrid": val_hybrid}
        best_model = max(scores, key=scores.get)
        
        # Print terminal row
        print(f"│ {metric:<8} │ {val_tfidf:.4f}   │ {val_bm25:.4f}   │ {val_hybrid:.4f}     │ {best_model:<9} │")
        
        # Add markdown row
        markdown_rows.append(f"| {metric} | {val_tfidf:.4f} | {val_bm25:.4f} | {val_hybrid:.4f} | **{best_model}** |")
        
    print("└─────────────────────────────────────────────────────────┘")
    print("\n")
    
    # Also save to data/benchmark_report.md
    report_path = settings.DATA_DIR / "benchmark_report.md"
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 📈 Offline Evaluation Benchmark Report\n\n")
            f.write("Generated automatically from ground-truth queries and judgments.\n\n")
            f.write("\n".join(markdown_rows))
            f.write("\n")
        logger.info(f"Saved benchmark report to {report_path}")
    except Exception as e:
        logger.error(f"Error saving benchmark report: {e}")

if __name__ == "__main__":
    main()

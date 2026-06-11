import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from config.settings import settings
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer
from src.ranking.hybrid_ranker import HybridRanker
from src.ranking.snippet import SnippetGenerator
from src.indexing.inverted_index import InvertedIndex

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indonesian News Search Engine API",
    version="1.0.0",
    description="REST API serving lexical, BM25, and Hybrid search scores over Indonesian news."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize modules
cleaner = TextCleaner()
tokenizer = Tokenizer()
ranker = HybridRanker()
index_builder = InvertedIndex()

# Global state for document cache and index
documents_cache: List[str] = []
doc_names_cache: List[str] = []
urldoc_cache: Dict[str, str] = {}
index_last_built: Optional[float] = None

def load_data_and_fit():
    """
    Loads preprocessed documents and fits the rankers.
    """
    global documents_cache, doc_names_cache, urldoc_cache, index_last_built
    
    logger.info("Loading documents and fitting rankers...")
    
    documents = []
    doc_names = []
    
    # Load document contents
    if settings.PROCESSED_DIR.exists():
        files = sorted([f for f in settings.PROCESSED_DIR.iterdir() if f.is_file() and f.suffix == '.txt'])
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    documents.append(file.read())
                    doc_names.append(f.name)
            except Exception as e:
                logger.error(f"Error loading processed document {f.name}: {e}")
                
    documents_cache = documents
    doc_names_cache = doc_names
    
    # Load URL document mapping
    urldoc_path = settings.DATA_DIR / "urldoc.json"
    if urldoc_path.exists():
        try:
            import json
            with open(urldoc_path, 'r', encoding='utf-8') as f:
                urldoc_cache = json.load(f)
        except Exception as e:
            logger.error(f"Error loading urldoc.json: {e}")
            urldoc_cache = {}
            
    # Load index status
    index_file = settings.INDEX_DIR / "inverted_index.txt"
    if index_file.exists():
        index_last_built = index_file.stat().st_mtime
    else:
        index_last_built = None

    if documents:
        ranker.fit(documents, doc_names)
        logger.info(f"Successfully fit rankers with {len(documents)} documents.")
    else:
        logger.warning("No preprocessed documents found. Search might not return results.")

@app.on_event("startup")
def startup_event():
    load_data_and_fit()

def async_rebuild_index():
    """
    Rebuilds the inverted index by running preprocessing and indexing.
    Note: For simplicity, this uses the existing raw/links.txt files.
    """
    logger.info("Background rebuild index started...")
    try:
        # 1. Run Preprocessing (simulated by calling our preprocess logic)
        from scripts.preprocess import main as run_preprocess
        run_preprocess()
        
        # 2. Run Indexing
        from scripts.index import main as run_index
        run_index()
        
        # 3. Reload data
        load_data_and_fit()
        logger.info("Background rebuild index completed successfully.")
    except Exception as e:
        logger.error(f"Error rebuilding index in background: {e}")

@app.get("/api/v1/search")
def api_search(
    q: str = Query(..., min_length=1, description="Search query string"),
    method: str = Query("hybrid", description="Ranking method: tfidf, bm25, hybrid"),
    top_k: int = Query(10, ge=1, le=100, description="Number of top results to return")
):
    start_time = time.time()
    
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    method = method.lower()
    if method not in ["tfidf", "bm25", "hybrid"]:
        raise HTTPException(status_code=400, detail="Invalid ranking method. Choose tfidf, bm25, or hybrid.")
        
    # Preprocess the query (clean and stem)
    cleaned_query = cleaner.clean(q)
    stemmed_query_tokens = tokenizer.tokenize_and_stem(cleaned_query)
    preprocessed_query = " ".join(stemmed_query_tokens)
    
    logger.info(f"Raw query: '{q}' | Preprocessed query: '{preprocessed_query}' | Method: '{method}'")
    
    if not preprocessed_query.strip():
        # Query became empty after processing (e.g. only stopwords)
        return {
            "query": q,
            "method": method,
            "total_results": 0,
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "results": []
        }
        
    results = []
    
    try:
        if method == "tfidf":
            raw_results = ranker.tfidf_ranker.search(preprocessed_query, top_k=top_k)
        elif method == "bm25":
            raw_results = ranker.bm25_ranker.search(preprocessed_query, top_k=top_k)
        else: # hybrid
            # We use details if we want tfidf and bm25 separate scores, otherwise standard search
            raw_results = ranker.search(preprocessed_query, top_k=top_k)
            
        # Formulate rich JSON response with snippets
        for rank, (doc_name, score) in enumerate(raw_results, start=1):
            url = urldoc_cache.get(doc_name, "URL tidak tersedia")
            
            # Read original text from data/raw/articles/ if exists, else data/processed/
            raw_article_path = settings.RAW_ARTICLES_DIR / doc_name
            processed_article_path = settings.PROCESSED_DIR / doc_name
            
            article_text = ""
            if raw_article_path.exists():
                with open(raw_article_path, 'r', encoding='utf-8') as f:
                    article_text = f.read()
            elif processed_article_path.exists():
                with open(processed_article_path, 'r', encoding='utf-8') as f:
                    # Fallback to tokens joined by space
                    article_text = f.read().replace('\n', ' ')
            
            # Generate snippet from raw text using original raw query terms
            snippet = SnippetGenerator.generate(article_text, q, length=160)
            
            results.append({
                "rank": rank,
                "doc_id": doc_name,
                "title": doc_name.replace('.txt', ''),
                "score": round(score, 4),
                "url": url,
                "snippet": snippet
            })
            
    except Exception as e:
        logger.error(f"Error executing search for query '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")
        
    latency_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(f"Query returned {len(results)} results in {latency_ms} ms")
    
    return {
        "query": q,
        "method": method,
        "total_results": len(results),
        "latency_ms": latency_ms,
        "results": results
    }

@app.get("/api/v1/documents/{doc_name}")
def get_document(doc_name: str):
    """
    Returns the complete raw text of the document.
    """
    if not doc_name.endswith('.txt'):
        doc_name += '.txt'
        
    raw_path = settings.RAW_ARTICLES_DIR / doc_name
    processed_path = settings.PROCESSED_DIR / doc_name
    
    url = urldoc_cache.get(doc_name, "URL tidak tersedia")
    
    if raw_path.exists():
        with open(raw_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return {"title": doc_name.replace('.txt', ''), "content": content, "url": url, "type": "raw"}
    elif processed_path.exists():
        with open(processed_path, 'r', encoding='utf-8') as f:
            content = f.read().replace('\n', ' ')
        return {"title": doc_name.replace('.txt', ''), "content": content, "url": url, "type": "stemmed"}
    else:
        raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found.")

@app.get("/api/v1/stats")
def get_stats():
    """
    Returns statistics about the inverted index.
    """
    # Count terms in index
    num_terms = 0
    index_file = settings.INDEX_DIR / "inverted_index.txt"
    if index_file.exists():
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                num_terms = sum(1 for line in f if line.strip())
        except Exception:
            pass

    return {
        "total_documents": len(doc_names_cache),
        "total_indexed_terms": num_terms,
        "index_last_built": time.ctime(index_last_built) if index_last_built else "Belum dibuat",
        "raw_dir_exists": settings.RAW_DIR.exists(),
        "processed_dir_exists": settings.PROCESSED_DIR.exists()
    }

@app.post("/api/v1/index/rebuild")
def rebuild_index(background_tasks: BackgroundTasks):
    """
    Triggers asynchronous rebuild of the preprocessed files and inverted index.
    """
    background_tasks.add_task(async_rebuild_index)
    return {"status": "accepted", "message": "Index rebuild triggered in background."}

@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

# Mount static files for UI (if folder exists)
static_dir = settings.BASE_DIR / "src" / "api" / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    logger.warning(f"Static directory {static_dir} not found. UI files will not be served.")

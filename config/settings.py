from pathlib import Path
from dataclasses import dataclass

@dataclass
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    INDEX_DIR: Path = DATA_DIR / "index"
    STOPWORDS_PATH: Path = DATA_DIR / "stopwords.txt"
    
    # Crawler config
    CRAWL_BASE_URL: str = "https://www.viva.co.id"
    CRAWL_MAX_LINKS: int = 505
    CRAWL_START_DATE: str = "2021-01-01"
    CRAWL_END_DATE: str = "2021-12-01"
    CRAWL_REQUEST_TIMEOUT: int = 10
    CRAWL_DELAY_SECONDS: float = 1.0
    
    # Ranking config
    BM25_K1: float = 1.5
    BM25_B: float = 0.75
    TFIDF_WEIGHT: float = 0.5
    BM25_WEIGHT: float = 0.5

settings = Settings()

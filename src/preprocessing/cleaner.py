import re
from pathlib import Path
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class TextCleaner:
    def __init__(self, stopwords_path: Path = None):
        self.stopwords_path = stopwords_path or settings.STOPWORDS_PATH
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set:
        if not self.stopwords_path.exists():
            logger.warning(f"Stopwords file not found at {self.stopwords_path}. Using empty set.")
            return set()
        
        try:
            with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                return set(line.strip().lower() for line in f if line.strip())
        except Exception as e:
            logger.error(f"Error loading stopwords: {e}. Using empty set.")
            return set()

    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove brackets, parentheses, punctuation, numbers, and symbols
        # re.sub(r'[-&()!./,0-9]', '', text) in original clean(3).py
        # Let's keep it consistent but also strip other punctuation if desired.
        # Let's use the exact original pattern or a cleaner one. Original:
        text = re.sub(r'[-&()!./,0-9]', '', text)
        
        # Replace newlines and carriage returns with space
        text = re.sub(r'[\r\n]+', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filter out stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        
        return ' '.join(filtered_words)

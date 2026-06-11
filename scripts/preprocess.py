import json
import logging
import re
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from config.settings import settings
from src.preprocessing.cleaner import TextCleaner
from src.preprocessing.tokenizer import Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

def sanitize_filename(title: str) -> str:
    if not title:
        return "untitled.txt"
    short_title = title[:100]
    filename = "".join(c for c in short_title if c.isalnum() or c.isspace()).rstrip()
    return f"{filename}.txt" if filename else "untitled.txt"

def main():
    logger.info("Initializing preprocessor...")
    cleaner = TextCleaner()
    tokenizer = Tokenizer()

    links_file = settings.RAW_DIR / "links.txt"
    if not links_file.exists():
        # Fallback to links.txt in root if it exists
        root_links_file = settings.BASE_DIR / "links.txt"
        if root_links_file.exists():
            links_file = root_links_file
            logger.info(f"Using links.txt from root directory: {root_links_file}")
        else:
            logger.error("links.txt not found. Please run scripts/crawl.py first.")
            return

    # Create directories
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(links_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    logger.info(f"Found {len(urls)} URLs to process.")
    
    urldoc = {}
    skipped_urls = []
    
    # Load existing urldoc.json if it exists to preserve mappings
    urldoc_json_path = settings.DATA_DIR / "urldoc.json"
    if urldoc_json_path.exists():
        try:
            with open(urldoc_json_path, 'r', encoding='utf-8') as f:
                urldoc = json.load(f)
            logger.info(f"Loaded {len(urldoc)} existing URL mappings from {urldoc_json_path}")
        except Exception as e:
            logger.warning(f"Could not load existing urldoc.json: {e}")

    for idx, url in enumerate(urls, start=1):
        logger.info(f"[{idx}/{len(urls)}] Processing {url}")
        
        try:
            response = requests.get(url, timeout=settings.CRAWL_REQUEST_TIMEOUT)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: status {response.status_code}")
                skipped_urls.append(url)
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main article body
            article_body = soup.find(class_='main-content-detail')
            if article_body is None:
                # Try fallback classes if any
                article_body = soup.find(itemprop='articleBody')
            
            if article_body is None:
                logger.warning(f"Could not find article body for {url}")
                skipped_urls.append(url)
                continue
                
            # Extract paragraphs
            paragraphs = [p.get_text() for p in article_body.find_all('p')]
            raw_text = '\n'.join(paragraphs)
            
            if not raw_text.strip():
                logger.warning(f"Empty article body for {url}")
                skipped_urls.append(url)
                continue

            # Extract title
            title = soup.title.string if soup.title else "Untitled"
            title = title.replace(" - VIVA", "").strip()
            
            filename = sanitize_filename(title)
            
            # Clean text (removes punctuation, lowercases, removes stopwords)
            cleaned_text = cleaner.clean(raw_text)
            
            # Tokenize & Stem using Sastrawi
            tokens = tokenizer.tokenize(cleaned_text)
            stemmed_tokens = tokenizer.stem(tokens)
            
            # Save preprocessed tokens
            output_file_path = settings.PROCESSED_DIR / filename
            with open(output_file_path, 'w', encoding='utf-8') as out_f:
                out_f.write('\n'.join(stemmed_tokens))
                
            urldoc[filename] = url
            logger.info(f"Saved preprocessed text to {output_file_path.name}")
            
            # Delay to avoid overloading the site
            time.sleep(settings.CRAWL_DELAY_SECONDS)
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            skipped_urls.append(url)

    # Save urldoc.json
    try:
        with open(urldoc_json_path, 'w', encoding='utf-8') as json_f:
            json.dump(urldoc, json_f, ensure_ascii=False, indent=4)
        
        # Also sync/copy to root if needed (or keep in data/)
        # Let's save in root too to prevent breaking any client that expects it in root
        root_urldoc_path = settings.BASE_DIR / "urldoc.json"
        with open(root_urldoc_path, 'w', encoding='utf-8') as json_f:
            json.dump(urldoc, json_f, ensure_ascii=False, indent=4)
            
        logger.info(f"Saved URL-to-document mappings to {urldoc_json_path} and root urldoc.json")
    except Exception as e:
        logger.error(f"Error saving urldoc mappings: {e}")

    logger.info(f"Preprocessing completed. Processed: {len(urldoc)} docs. Skipped: {len(skipped_urls)} urls.")

if __name__ == "__main__":
    main()

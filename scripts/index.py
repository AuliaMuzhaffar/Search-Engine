import logging
from config.settings import settings
from src.indexing.inverted_index import InvertedIndex

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Building inverted index...")
    index_builder = InvertedIndex()
    
    # Build from data/processed
    index_builder.build_global_index(settings.PROCESSED_DIR)
    
    # Save to data/index/inverted_index.txt
    index_builder.save()
    logger.info("Inverted index successfully built and saved.")

if __name__ == "__main__":
    main()

import argparse
import logging
from pathlib import Path
from config.settings import settings
from src.crawler.spider import VivaSpider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Crawl article URLs from viva.co.id")
    parser.add_argument("--start-date", default=settings.CRAWL_START_DATE, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=settings.CRAWL_END_DATE, help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-links", type=int, default=settings.CRAWL_MAX_LINKS, help="Max links to crawl")
    args = parser.parse_args()

    spider = VivaSpider(max_links=args.max_links)
    
    csv_path = settings.RAW_DIR / "links.csv"
    txt_path = settings.RAW_DIR / "links.txt"
    
    crawled_data = spider.crawl(args.start_date, args.end_date)
    spider.save_results(crawled_data, csv_path, txt_path)

if __name__ == "__main__":
    main()

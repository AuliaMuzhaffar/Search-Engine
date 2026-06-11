import csv
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
from config.settings import settings

logger = logging.getLogger(__name__)

class VivaSpider:
    def __init__(self, base_url: str = None, max_links: int = None):
        self.base_url = base_url or settings.CRAWL_BASE_URL
        self.max_links = max_links or settings.CRAWL_MAX_LINKS
        self.timeout = settings.CRAWL_REQUEST_TIMEOUT
        self.delay = settings.CRAWL_DELAY_SECONDS

    def crawl(self, start_date_str: str, end_date_str: str) -> List[Tuple[str, str]]:
        """
        Crawls article titles and URLs from the specified date range.
        Returns a list of tuples (title, url).
        """
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        crawled_data = []
        current_date = start_date
        article_number = 1
        
        logger.info(f"Starting crawl from {start_date_str} to {end_date_str} (Max: {self.max_links} links)")
        
        while current_date <= end_date and len(crawled_data) < self.max_links:
            date_str = current_date.strftime("%Y/%m/%d")
            url = f"{self.base_url}/indeks/gaya-hidup/all/{date_str}"
            
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code != 200:
                    logger.warning(f"Failed to fetch {url}: Status {response.status_code}")
                    current_date += timedelta(days=1)
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                links = soup.find_all('a', class_='article-list-title')
                
                for link in links:
                    if len(crawled_data) >= self.max_links:
                        break
                    
                    href = link.get('href')
                    title = link.text.strip()
                    
                    if href and title:
                        crawled_data.append((title, href))
                        print(f"Artikel {article_number} ({title}) berhasil disimpan")
                        article_number += 1
                
                # Menambahkan delay agar tidak membombardir server (polite crawler)
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Error fetching page for {date_str}: {e}")
                
            current_date += timedelta(days=1)
            
        logger.info(f"Crawling finished. Collected {len(crawled_data)} links.")
        return crawled_data

    def save_results(self, data: List[Tuple[str, str]], csv_path: Path, txt_path: Path):
        """
        Saves crawled data to CSV and TXT files.
        """
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Title', 'Link']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for title, link in data:
                    writer.writerow({'Title': title, 'Link': link})
            logger.info(f"Saved links to CSV: {csv_path}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")

        # Save to TXT
        try:
            with open(txt_path, 'w', encoding='utf-8') as txtfile:
                for _, link in data:
                    txtfile.write(f"{link}\n")
            logger.info(f"Saved links to TXT: {txt_path}")
        except Exception as e:
            logger.error(f"Error saving to TXT: {e}")

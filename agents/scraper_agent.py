import requests
import time
from bs4 import BeautifulSoup

def scraper_node(state: dict) -> dict:
    """Reads URL from the state. Fetches each URL and extracts clean text. Returns raw_texts to be added to the state"""

    start = time.time()
    urls=state["urls"]
    scraped = []
    for url in urls:
        page_start = time.time()
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, timeout=10, headers=headers)
            soup= BeautifulSoup(response.content, "html.parser")

            ## Remove Noise
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            ## Extract Clean Text
            text=soup.get_text(separator="\n", strip=True)

            if text:
                scraped.append({"url": url, "text": text})  

            print(f"[timing]   page {time.time() - page_start:.1f}s  {url[:50]}", flush=True)

        except Exception as e:
            print(f"[timing]   page FAILED {time.time() - page_start:.1f}s  {url[:50]}: {e}", 
                  flush=True)
            continue

    print(f"[timing] scrape total: {time.time() - start:.1f}s ({len(scraped)}/{len(urls)} pages)", 
          flush=True)
    return {"scraped": scraped}
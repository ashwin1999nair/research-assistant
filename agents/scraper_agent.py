import requests
from bs4 import BeautifulSoup

def scraper_node(state: dict) -> dict:
    """Reads URL from the state. Fetches each URL and extracts clean text. Returns raw_texts to be added to the state"""

    urls=state["urls"]
    scraped = []
    for url in urls:
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

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            continue

    return {"scraped": scraped}
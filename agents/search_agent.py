import os
import time
from tavily import TavilyClient
from dotenv import load_dotenv
from agents.retry import with_retry

load_dotenv()
client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_node(state: dict) -> dict:
    """Reads topic from state. Calls Tavily API to get relevant URLs. Returns URLs to be added to state"""
    start = time.time()

    topic=state["topic"]
    n_urls=state.get("n_urls", 5)
    response = with_retry(client.search, topic, max_results=n_urls)
    urls=[r["url"] for r in response["results"]]

    print(f"[timing] search: {time.time() - start:.1f}s", flush=True)
    return {"urls": urls}
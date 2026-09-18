import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
client=TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def search_node(state: dict) -> dict:
    """Reads topic from state. Calls Tavily API to get relevant URLs. Returns URLs to be added to state"""

    topic=state["topic"]
    n_urls=state.get("n_urls", 5)
    response = client.search(topic, max_results=n_urls)
    urls=[r["url"] for r in response["results"]]

    return {"urls": urls}
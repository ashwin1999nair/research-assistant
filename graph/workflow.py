from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from agents.search_agent import search_node
from agents.scraper_agent import scraper_node

class AgentState(TypedDict):
    topic: str
    urls: List[str]
    raw_texts: List[str]
    report: str

def build_graph():
    graph=StateGraph(AgentState)

    graph.add_node("search", search_node)
    graph.add_node("scraper", scraper_node)

    graph.add_edge(START, "search")
    graph.add_edge("search", "scraper")
    graph.add_edge("scraper", END)

    return graph.compile()
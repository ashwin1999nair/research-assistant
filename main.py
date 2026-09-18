from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from graph.workflow import build_graph

app=FastAPI()
graph=build_graph()

class ResearchRequest(BaseModel):
    topic: str
    debug: bool = False
    n_urls: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    n_results: int = 10
    temperature: float = 0.3

class ResearchResponse(BaseModel):
    report: str
    urls: list
    retrieved_chunks: Optional[list] = None

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    result=graph.invoke({
        "topic":request.topic,
        "urls": [],
        "scraped": [],
        "report": "",
        "retrieved_chunks": [],
        "n_urls": request.n_urls,
        "chunk_size": request.chunk_size,
        "chunk_overlap": request.chunk_overlap,
        "n_results": request.n_results,
        "temperature": request.temperature,
    })

    return ResearchResponse(
        report=result["report"],
        urls=result["urls"],
        retrieved_chunks=result["retrieved_chunks"] if request.debug else None
    )
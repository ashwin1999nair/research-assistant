from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from graph.workflow import build_graph

app=FastAPI()
graph=build_graph()

class ResearchRequest(BaseModel):
    topic: str
    debug: bool = False

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
        "retrieved_chunks": []
    })

    return ResearchResponse(
        report=result["report"],
        urls=result["urls"],
        retrieved_chunks=result["retrieved_chunks"] if request.debug else None
    )
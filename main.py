from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from graph.workflow import build_graph
import cache

app=FastAPI()
graph=build_graph()

class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=200)
    debug: bool = False
    no_cache: bool = False
    n_urls: int = Field(default=5, ge=1, le=10)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    n_results: int = Field(default=10, ge=1, le=50)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)

    @field_validator("topic")
    @classmethod
    def topic_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("topic cannot be blank")
        return v.strip()

class ResearchResponse(BaseModel):
    report: str
    urls: list
    retrieved_chunks: Optional[list] = None
    cached: bool = False

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    if not request.no_cache:
        hit = cache.lookup(request.topic)
        if hit:
            return ResearchResponse(
                report=hit["report"],
                urls=hit["urls"],
                retrieved_chunks=hit["retrieved_chunks"] if request.debug else None,
                cached=True,
            )
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

    if not request.no_cache:
        cache.store(request.topic, {
            "report":result["report"],
            "urls":result["urls"],
            "retrieved_chunks": result["retrieved_chunks"],
        })

    return ResearchResponse(
        report=result["report"],
        urls=result["urls"],
        retrieved_chunks=result["retrieved_chunks"] if request.debug else None,
        cached=False,
    )
from fastapi import FastAPI
from pydantic import BaseModel
from graph.workflow import build_graph

app=FastAPI()
graph=build_graph()

class ResearchRequest(BaseModel):
    topic: str

class ResearchResponse(BaseModel):
    report: str
    urls: list

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/research", response_model=ResearchResponse)
def run_research(request: ResearchRequest):
    result=graph.invoke({
        "topic":request.topic,
        "urls": [],
        "raw_texts": [],
        "report": ""
    })

    return ResearchResponse(
        report=result["report"],
        urls=result["urls"]
    )
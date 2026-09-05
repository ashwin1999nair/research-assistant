from graph.workflow import build_graph

app=build_graph()

result=app.invoke({
    "topic": "quantum computing",
    "urls": [],
    "scraped":[],
    "report": "",
    "retrieved_chunks": []
})

print("Report:")
print(result["report"])
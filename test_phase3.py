from graph.workflow import build_graph

app=build_graph()

result=app.invoke({
    "topic": "quantum computing",
    "urls": [],
    "raw_texts":[],
    "report": ""
})

print("Report:")
print(result["report"])
from graph.workflow import build_graph

app=build_graph()

result=app.invoke({
    "topic": "quantum computing",
    "urls": [],
    "raw_texts": [],
    "report": ""
})

print("URLs found:", result["urls"])
print("\nNumber of pages scraped:", len(result["raw_texts"]))
print("\nFirst 500 chars of first page:")
print(result["raw_texts"][0][:500] if result["raw_texts"] else "No text scraped")
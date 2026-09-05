from graph.workflow import build_graph

app=build_graph()

result=app.invoke({
    "topic": "quantum computing",
    "urls": [],
    "scraped": [],
    "report": ""
})

print("URLs found:", result["urls"])
print("Number of pages scraped:", len(result["scraped"]))
print("\nFirst 500 chars of first page:")
print(result["scraped"][0]["text"][:500] if result["scraped"] else "No text scraped")
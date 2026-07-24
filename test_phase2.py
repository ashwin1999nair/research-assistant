from graph.workflow import build_graph
from vectorstore import store_chunks, query_chunks

app=build_graph()
result=app.invoke({
    "topic":"quantum computing",
    "urls":[],
    "raw_texts":[],
    "report":""
})

print("Pages scraped", len(result["raw_texts"]))

collection=store_chunks(result["raw_texts"], "quantum computing")
print("Chunks are stored in ChromaDB")

relevant_chunks=query_chunks(collection, "quantum computing", n_results=5)
print("\nTop 5 relevant chunks:")
for i, chunk in enumerate(relevant_chunks):
    print(f"\n Chunk {i+1}")
    print (chunk[:300])
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from vectorstore import store_chunks, query_chunks

load_dotenv()

def synthesis_node(state: dict) -> dict:
     """Reads scraped pages and topic from state. Stores chunks in ChromaDB. Queries for relevant chunks.
        Sends to Gemini for report generation. Returns report and retrieved chunks to state."""
     topic=state["topic"]
     scraped = state["scraped"]

     ## Config, with defaults for callers that don't supply them
     chunk_size=state.get("chunk_size", 500)
     chunk_overlap=state.get("chunk_overlap", 50)
     n_results=state.get("n_results", 10)
     temperature=state.get("temperature", 0.3)

     ## Built per request so temperature is configurable
     llm=ChatGoogleGenerativeAI(model='gemini-2.5-flash',google_api_key=os.getenv("GEMINI_API_KEY"),temperature=temperature,)

     ## Store Collection in ChromaDB
     collection = store_chunks(scraped, topic, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

     ## Query for relevant chunks
     relevant_chunks=query_chunks(collection, topic, n_results=n_results)

     ## Combine Chunks into context
     context = "\n\n".join(c["text"] for c in relevant_chunks)
     
     ## Prompt
     prompt = f"""You are a research assistant. Based on the following retrieved content, write a structured research report on the topic: {topic}

     Retrieved Content:
     {context}

     Write a clear, structured report with the following sections:
     1. Overview
     2. Key Concepts
     3. Current Developments
     4. Applications
     5. Conclusion

     Be factual and base your report only on the provided content."""

     ## Call Gemini
     response=llm.invoke(prompt)

     return {"report": response.content, "retrieved_chunks": relevant_chunks}
# AI Research Assistant

An end-to-end multi-agent AI system that takes a research topic, searches the web, scrapes relevant pages, stores content in a vector database, and generates a structured research report using Google Gemini. Built as a portfolio project to demonstrate multi-agent orchestration, RAG pipelines, and full-stack ML application development.

---

## What It Does

1. User inputs a research topic via the Streamlit UI
2. Orchestrator (LangGraph) coordinates the agent pipeline
3. Search Agent calls Tavily API and retrieves 5 relevant URLs
4. Scraper Agent fetches each URL and extracts clean text using BeautifulSoup
5. Text is chunked using LangChain's RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
6. Chunks are embedded using HuggingFace sentence-transformers and stored in ChromaDB
7. Synthesis Agent queries ChromaDB for the top 10 most relevant chunks
8. Google Gemini generates a structured research report from the retrieved chunks
9. FastAPI serves the report and sources
10. Streamlit displays the report with citations

---

## Why RAG is Genuine Here

Multiple web pages are scraped per query — too long to fit in an LLM context window directly. Chunking and ChromaDB retrieval ensures only the most relevant sections reach Gemini. This is real RAG: external documents retrieved at runtime, chunked, embedded, similarity-searched, then passed to the LLM for grounded generation.

---

## Architecture

```
User (Browser)
     │
     ▼
Streamlit Frontend (Port 8501)
     │  HTTP POST /research
     ▼
FastAPI Backend (Port 8000)
     │
     ▼
LangGraph Orchestrator
     │
     ├── Search Node → Tavily API → URLs
     │
     ├── Scraper Node → BeautifulSoup → Raw Text
     │
     └── Synthesis Node
              │
              ├── LangChain Text Splitter → Chunks
              ├── HuggingFace Embeddings → Vectors
              ├── ChromaDB → Store + Similarity Search
              └── Gemini API → Structured Report
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| Web Search | Tavily API |
| Web Scraping | BeautifulSoup + requests |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace sentence-transformers (all-MiniLM-L6-v2) |
| Vector Storage | ChromaDB (local) |
| LLM | Google Gemini 2.5 Flash |
| API Layer | FastAPI |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |

---

## How to Run Locally

### Without Docker

**1. Clone the repo**
```bash
git clone https://github.com/ashwin1999nair/research-assistant
cd research-assistant
```

**2. Create and activate a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**3. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**4. Add your API keys**

Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
```

**5. Run FastAPI**
```bash
uvicorn main:app --reload
```

**6. Run Streamlit (new terminal)**
```bash
streamlit run app.py
```

Visit `http://localhost:8501`

---

### With Docker

**1. Add your API keys to `.env` (same as above)**

**2. Build and run**
```bash
docker-compose up --build
```

Visit `http://localhost:8501`

---

## Project Structure

```
research-assistant/
├── agents/
│   ├── __init__.py
│   ├── search_agent.py       # Tavily API search
│   ├── scraper_agent.py      # BeautifulSoup web scraping
│   └── synthesis_agent.py    # ChromaDB query + Gemini report generation
├── graph/
│   ├── __init__.py
│   └── workflow.py           # LangGraph state + agent pipeline
├── vectorstore.py            # Chunking, embedding, ChromaDB storage + retrieval
├── main.py                   # FastAPI app — /health and /research endpoints
├── app.py                    # Streamlit frontend
├── Dockerfile.api            # FastAPI container
├── Dockerfile.streamlit      # Streamlit container
├── docker-compose.yml        # Runs both containers together
├── requirements.txt          # Python dependencies
└── .env                      # API keys (not committed to Git)
```

---

## Example Output

**Topic:** Quantum Computing

**Sources:**
- https://en.wikipedia.org/wiki/Quantum_computing
- https://www.ibm.com/think/topics/quantum-computing
- https://aws.amazon.com/what-is/quantum-computing/

**Report sections:** Overview, Key Concepts, Current Developments, Applications, Conclusion

---

## Performance Note

Embedding runs locally on CPU using HuggingFace sentence-transformers. Expect 1-2 minutes per query depending on hardware. This is a known trade-off for using free, local embeddings without an API call.

---

## Future Improvements

- Deployment to Railway or Render
- GPU-accelerated embeddings for faster processing
- PDF export of generated reports
- Support for custom URLs as input sources
- Persistent ChromaDB storage across sessions
- Streaming response so users see the report as it generates

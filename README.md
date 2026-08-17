# AI Research Assistant

A multi-agent research pipeline that takes a topic, searches the web, scrapes and indexes the results, and generates a structured report grounded in the retrieved sources.

Built to explore agent orchestration, retrieval-augmented generation, and container deployment on a managed cloud runtime.

<!-- Add your demo GIF here:
![Demo](docs/demo.gif)
-->

---

## What it does

1. **Search** — takes a topic and queries the Tavily API, returning 5 candidate URLs
2. **Scrape** — fetches and extracts readable text from each page with BeautifulSoup
3. **Index** — splits the scraped text into chunks, embeds them locally, and stores them in a vector database
4. **Synthesise** — retrieves the 10 most relevant chunks for the topic and generates a structured report with Gemini
5. **Serve** — FastAPI exposes the pipeline; Streamlit provides the interface

The retrieval step is doing real work here: the combined scraped text from five pages routinely exceeds the model's usable context, so only the relevant chunks reach the LLM.

---

## Architecture

```
                    ┌─────────────┐
   User ──────────► │  Streamlit  │  (external ingress, public HTTPS)
                    └──────┬──────┘
                           │  HTTP
                    ┌──────▼──────┐
                    │   FastAPI   │  (internal ingress, not public)
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │       LangGraph workflow        │
                    │                                 │
                    │  Search ──► Scrape ──► Synthesis│
                    │  (Tavily)   (BS4)      (Gemini) │
                    └──────┬──────────────────┬───────┘
                           │                  │
                    ┌──────▼──────┐    ┌──────▼──────┐
                    │  ChromaDB   │◄───│  MiniLM     │
                    │ (vectors)   │    │ (embeddings)│
                    └─────────────┘    └─────────────┘
```

**Why two containers:** the API and the UI have completely different dependency footprints. Keeping them separate means the UI image ships without PyTorch — 183 MB instead of 3.17 GB.

---

## Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Search | Tavily API |
| Scraping | BeautifulSoup + requests |
| Chunking | LangChain `RecursiveCharacterTextSplitter` (size 500, overlap 50) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB |
| Generation | Google Gemini 2.5 Flash |
| API | FastAPI |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
| CI | GitHub Actions + pytest |
| Cloud | Azure Container Apps |

---

## Project structure

```
research-assistant/
├── agents/
│   ├── search_agent.py       # Tavily query, returns URLs
│   ├── scraper_agent.py      # fetches and extracts page text
│   └── synthesis_agent.py    # retrieves chunks, calls Gemini
├── graph/
│   └── workflow.py           # LangGraph state machine wiring the agents
├── vectorstore.py            # chunking, embedding, ChromaDB interface
├── main.py                   # FastAPI app
├── app.py                    # Streamlit UI
├── conftest.py               # pytest path resolution
├── tests/
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt          # API dependencies
└── requirements-ui.txt       # UI dependencies (streamlit + requests only)
```

---

## Running locally

**Prerequisites:** Docker Desktop, a Tavily API key, a Google Gemini API key.

```bash
git clone https://github.com/ashwin1999nair/research-assistant.git
cd research-assistant
```

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
```

Then:

```bash
docker-compose up --build
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

A query takes roughly 1–2 minutes. Most of that is CPU embedding of the scraped text.

---

## Configuration

The Streamlit container reads the API address from an environment variable, so the same image runs locally and in the cloud without code changes:

```python
API_URL = os.getenv("API_URL", "http://localhost:8000")
```

`docker-compose.yml` sets `API_URL=http://api:8000` for local runs. In the cloud deployment it points at the API's internal address.

API keys are never baked into images. Locally they come from `.env` (gitignored, and excluded from the build context by `.dockerignore`); in the cloud they come from platform-managed secrets.

---

## Testing

```bash
pytest
```

7 unit tests, run automatically on every push via GitHub Actions:
- 4 covering the scraper's text extraction and error handling
- 3 covering the chunking logic

These cover the deterministic parts of the pipeline. The LLM-dependent paths are not currently under test — see limitations.

---

## Cloud deployment

Deployed to **Azure Container Apps** (Sweden Central) as a demonstration of a managed-runtime deployment model.

**How it works:**
- Images are built locally, tagged, and pushed to Azure Container Registry
- Container Apps pulls from ACR using a **user-assigned managed identity** with the `AcrPull` role — no registry username or password is stored anywhere
- The FastAPI container uses **internal ingress** and is unreachable from the public internet
- The Streamlit container uses **external ingress** and gets a public HTTPS endpoint with a managed TLS certificate
- API keys are stored as Container Apps secrets and injected as environment variables via `secretref`
- Both containers run with `min-replicas 0`, so they scale to zero when idle

**Why this approach over a VM:** building the image on the server, as in a typical VM deployment, means the production artifact isn't reproducible. Building locally and pushing to a registry means every environment pulls the same immutable image, and rolling back is a matter of pointing at an earlier tag.

Deployment notes, including the full command sequence and the problems encountered, are in [`docs/azure-deployment.md`](docs/azure-deployment.md).

---

## Limitations

Stated plainly, because they're design consequences rather than oversights:

**Storage is ephemeral.** Container Apps filesystems don't survive a restart, so the vector store is rebuilt on every query. This suits the current design — each query researches fresh sources — but it means no caching, no corpus accumulating over time, and every query paying the full embedding cost. Persistence would need a mounted volume or a hosted vector database.

**Cold starts are slow.** With scale-to-zero, the first request after an idle period waits for a 3.17 GB image pull and the embedding model to load. Combined with the normal 1–2 minute query time, the first request can take several minutes.

**Embedding runs on CPU.** `all-MiniLM-L6-v2` is small, but embedding several pages of scraped text single-threaded is the dominant cost in a query.

**Scraping is sequential.** The five URLs are fetched one after another rather than concurrently.

**No evaluation harness yet.** There is no systematic measurement of report quality, source grounding, or retrieval relevance. This is the next planned piece of work.

**Scraping is best-effort.** JavaScript-rendered pages, paywalls, and aggressive bot protection all reduce the amount of usable text retrieved.

---

## Planned next steps

- A fixed evaluation set with automated scoring for source grounding, citation rate, and length adherence
- MLflow experiment tracking to compare chunk size, retrieval depth, and prompt variants against those metrics
- Concurrent scraping and batched embedding to cut query latency
- Semantic caching to avoid re-running the full pipeline on repeated topics
- Input validation and retry-with-backoff on upstream API failures

---

## Author

**Ashwin Nair** — MSc Artificial Intelligence, University of Galway
[GitHub](https://github.com/ashwin1999nair)

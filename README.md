# AI Research Assistant

![CI](https://github.com/ashwin1999nair/research-assistant/actions/workflows/ci.yml/badge.svg)

A multi-agent research pipeline that takes a topic, searches the web, scrapes and indexes the results, and generates a structured report grounded in the retrieved sources.

Built to explore agent orchestration, retrieval-augmented generation, systematic evaluation, and container deployment on a managed cloud runtime.

---

## What it does

1. **Search** — takes a topic and queries the Tavily API for candidate URLs
2. **Scrape** — fetches and extracts readable text from each page with BeautifulSoup
3. **Index** — splits the scraped text into chunks, embeds them locally, and stores them in a vector database
4. **Synthesise** — retrieves the most relevant chunks and generates a structured report with Gemini
5. **Serve** — FastAPI exposes the pipeline; Streamlit provides the interface

The retrieval step is doing real work: the combined text from five scraped pages routinely exceeds the model's usable context, so only the relevant chunks reach the LLM.

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
                    ┌──────▼─────────┐
                    │ semantic cache │  hit → return stored report
                    └──────┬─────────┘
                           │  miss
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
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| Vector store | ChromaDB |
| Generation | Google Gemini 2.5 Flash |
| API | FastAPI |
| UI | Streamlit |
| Evaluation | Custom harness + LLM-as-judge |
| Experiment tracking | MLflow |
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
│   ├── synthesis_agent.py    # retrieves chunks, calls Gemini
│   └── retry.py              # retry with exponential backoff
├── graph/
│   └── workflow.py           # LangGraph state machine wiring the agents
├── evals/
│   ├── test_set.json         # 15 categorised test cases
│   ├── scorers.py            # deterministic scorers
│   ├── llm_scorers.py        # faithfulness (LLM-as-judge)
│   └── runner.py             # executes the harness, logs to MLflow
├── tests/
│   ├── test_scraper.py
│   ├── test_vectorstore.py
│   └── test_retry.py
├── cache.py                  # semantic cache
├── vectorstore.py            # chunking, embedding, ChromaDB interface
├── main.py                   # FastAPI app
├── app.py                    # Streamlit UI
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── pytest.ini
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

A cold query takes roughly 20 seconds.

---

## Configuration

The pipeline's retrieval parameters are settable per request rather than hardcoded, so experiments don't require a container rebuild:

| Field | Default | Bounds |
|---|---|---|
| `n_urls` | 5 | 1–10 |
| `chunk_size` | 500 | 100–4000 |
| `chunk_overlap` | 50 | 0–500 |
| `n_results` | 10 | 1–50 |
| `temperature` | 0.3 | 0.0–2.0 |

Two operational flags: `debug` returns the retrieved chunks alongside the report, and `no_cache` bypasses the semantic cache (the eval harness always sets it, so runs measure the pipeline rather than cache lookups).

API keys are never baked into images. Locally they come from `.env` (gitignored, and excluded from the build context by `.dockerignore`); in the cloud they come from platform-managed secrets.

---

## Evaluation

A 15-case harness measures report quality across five deliberately chosen categories.

```bash
python -m evals.runner                      # deterministic scorers only, free
python -m evals.runner --llm-scorers        # adds faithfulness, costs API calls
python -m evals.runner --only adversarial   # one category
```

### Test set

| Category | Count | What it probes |
|---|---|---|
| Well-documented | 4 | Baseline — abundant clean sources |
| Narrow | 3 | Does the system degrade honestly when sources are thin? |
| Ambiguous | 2 | "Mercury", "Python performance" — does retrieval noise leak into synthesis? |
| Post-cutoff | 3 | Proves RAG is doing work rather than the model answering from memory |
| Adversarial | 3 | Empty input, overlong input, prompt injection |

### Scorers

**Deterministic** (no LLM, free, reproducible):
- `completed` — did the pipeline return without throwing
- `word_count` — report length within per-case bounds
- `source_diversity` — fraction of scraped pages that contributed a retrieved chunk
- `latency_s` — wall-clock seconds
- `check_fact` — for post-cutoff cases, does the report contain a fact the model could not know from training
- `check_absent` — for the injection case, did any prompt text leak into the output

**LLM-judged:**
- `faithfulness` — the report is decomposed into atomic claims, each verified against the retrieved chunks, scored as supported ÷ judged

### Results

Latest full run (`chunk_size=1000`, `n_results=5`, `temperature=0`):

| Metric | Value |
|---|---|
| Pass rate | 0.933 |
| Mean faithfulness | 0.926 |
| Mean source diversity | 0.538 |
| Median latency | 19.5s |
| Errors | 0 |

Per-category: well-documented 1.0, narrow 1.0, ambiguous 1.0, adversarial 1.0, post-cutoff 0.333.

---

## Experiment tracking

Retrieval configurations are compared in MLflow — pipeline parameters as params, aggregate quality metrics as metrics, per-case results attached as artifacts.

```bash
python -m evals.runner --llm-scorers --chunk-size 1000 --n-results 5
mlflow ui --backend-store-uri sqlite:///evals/mlflow.db
```

### What the comparison showed

| Config | Faithfulness | Mean latency | Pass rate |
|---|---|---|---|
| chunk 500 / k=10 (baseline) | 0.844 | 43.2s* | 0.867 |
| chunk 1000 / k=5 | 0.936 | 19.9s | 0.867 |
| chunk 2000 / k=3 | 0.815 | 28.0s | 0.600 |

\* inflated by one hung scrape; median was ~22s, which is why median is tracked alongside mean.

**Repeat trials matter.** Running `chunk 1000 / k=5` three times gave faithfulness of 0.936, 0.887 and 0.926 — a spread of roughly 0.05 with no code change. That is about the size of the gap to the baseline, so the faithfulness result is treated as directional rather than proven. The latency improvement is consistent across runs and is treated as real.

`chunk 2000 / k=3` is clearly worse: post-cutoff pass rate dropped to 0.0, because with only three large chunks the specific fact either gets buried or doesn't make the top 3.

---

## Reliability

**Input validation.** Pydantic constraints on the request model reject empty topics, whitespace-only topics, overlong input, and out-of-range config values with a 422 and a specific error message, before the pipeline runs.

**Retry with backoff.** Transient failures from Tavily (rate limits, timeouts, 5xx) are retried up to three times with doubling delays. Permanent failures such as an invalid API key fail immediately rather than wasting attempts.

**A note on the harness itself:** adversarial cases originally passed whenever no report came back — which meant a 500 crash scored identically to a correct rejection. The harness now checks the HTTP status code, so only a 4xx counts as a pass. That correction moved the real adversarial pass rate from 0.33 to 1.0 once validation was added.

---

## Semantic caching

Repeat topics skip the pipeline entirely. The topic is embedded and compared to cached topics by cosine similarity; above the threshold, the stored report is returned in ~0.1s instead of ~20s.

**The threshold is deliberately conservative (0.95)**, and the measured scores explain why:

| Pair | Similarity |
|---|---|
| "photosynthesis" → "photosynthesis" | 1.000 |
| "Mercury the planet" → "Mercury" | 0.889 |
| "how does photosynthesis work" → "photosynthesis" | 0.854 |
| unrelated topics | ~0.17 |

The dangerous pair (0.889 — different meaning) scores *higher* than the useful one (0.854 — same meaning, rephrased). No threshold cleanly separates them, so loosening it would trade correctness for hit rate. In practice the cache catches near-identical queries only.

Entries expire after 24 hours. A single TTL is the wrong model for an app that researches both stable topics and breaking news, but distinguishing them automatically is out of scope here.

---

## Performance

The pipeline is instrumented per stage. A typical query:

| Stage | Time | Share |
|---|---|---|
| Tavily search | 1.5s | 7% |
| Scraping (5 pages) | 2.5s | 11% |
| Embed + store | 6s | 27% |
| Retrieve | <0.1s | 0% |
| Gemini generation | 13s | 55% |

**A planned async-scraping optimisation was dropped after measuring.** Scraping was assumed to be the bottleneck; it turned out to be 11% of total latency, so parallelising it would have saved under two seconds. Generation dominates, and that is the model's own speed.

---

## Testing

```bash
pytest
```

13 unit tests, run on every push via GitHub Actions:
- 4 covering HTML text extraction
- 3 covering chunking behaviour
- 6 covering retry logic (recovery, give-up, permanent-error short-circuit, transient detection)

All are mocked — no network calls, no API spend. Behavioural testing is the eval harness's job; these two are deliberately separate tools.

---

## Cloud deployment

Deployed to **Azure Container Apps** as a demonstration of a managed-runtime deployment model.

- Images are built locally, tagged, and pushed to Azure Container Registry
- Container Apps pulls from ACR using a **user-assigned managed identity** with the `AcrPull` role — no registry credentials are stored anywhere
- The FastAPI container uses **internal ingress** and is unreachable from the public internet
- The Streamlit container uses **external ingress** with a managed TLS certificate
- API keys are Container Apps secrets, injected via `secretref`
- Both containers run with `min-replicas 0`, scaling to zero when idle

**Why this over a VM:** building the image on the server, as in a typical VM deployment, means the production artifact isn't reproducible. Building locally and pushing to a registry means every environment pulls the same immutable image, and rollback is pointing at an earlier tag.

Deployment notes, including the full command sequence and the problems encountered, are in [`docs/azure-deployment.md`](docs/azure-deployment.md).

---

## Limitations

Stated plainly, because they're design consequences rather than oversights.

**Storage is ephemeral.** Container Apps filesystems don't survive a restart, so the vector store is rebuilt on every query. This suits the design — each query researches fresh sources — but it means no caching of embeddings, no corpus accumulating over time, and every query paying the full embedding cost. Persistence would need a mounted volume or a hosted vector database. The semantic cache has the same constraint: in-memory, lost on restart, not shared between replicas. Production would use Redis.

**The judge evaluates its own output.** Faithfulness is scored by the same Gemini model that writes the reports, which introduces self-preference bias. A separate judge model would be more rigorous.

**No context recall metric.** Context recall requires ground-truth answers, which aren't tractable for an open-ended research agent. Retrieval coverage is therefore a known blind spot, approximated by source diversity and the narrow-topic test cases.

**Post-cutoff results are unstable.** Across three identical runs the post-cutoff pass rate was 0.867, 0.667 and 0.333 — because Tavily returns different URLs each time, so whether the specific checked fact appears is partly luck. Three cases is too small a sample when the input itself varies.

**Cold starts are slow.** With scale-to-zero, the first request after an idle period waits for a 3.17 GB image pull and the embedding model to load.

**Embedding runs on CPU** and is the second-largest cost in a query after generation.

**Scraping is sequential and best-effort.** JavaScript-rendered pages, paywalls, and bot protection all reduce usable text. Parallelising was measured and found not worth it.

---

## Author

**Ashwin Nair** — MSc Artificial Intelligence, University of Galway
[GitHub](https://github.com/ashwin1999nair) · [LinkedIn](https://www.linkedin.com/in/ashwinnair14/)

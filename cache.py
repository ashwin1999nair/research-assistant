"""
Semantic cache for research results.

Exact-match caching is near useless — nobody types the same string twice.
This embeds the topic and compares it to cached topics by cosine
similarity, so "photosynthesis" and "how photosynthesis works" hit the
same entry.

Limitations, stated plainly:
- In memory only. Dies on restart, not shared between containers.
  Production would use Redis.
- A TTL is the only defence against staleness. This app researches live
  web content, so a report on a developing story goes out of date while
  a report on photosynthesis does not — and the cache cannot tell them
  apart.
"""
import time
import numpy as np
from vectorstore import embeddings

SIMILARITY_THRESHOLD = 0.95
TTL_SECONDS = 24 * 60 * 60
MAX_ENTRIES = 100

# Each entry: {"topic", "embedding", "result", "stored_at"}
_cache: list = []

def _cosine(a,b) -> float:
     """Cosine similarity between two vectors: 1.0 = identical meaning."""
     a,b=np.array(a), np.array(b)
     return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def _prune() -> None:
    """Drop expired entries, then the oldest if we are over the cap."""
    now=time.time()
    global _cache
    _cache=[e for e in _cache if now-e["stored_at"] <TTL_SECONDS]

    while len(_cache) > MAX_ENTRIES:
        _cache.pop(0)

def lookup(topic: str) -> dict | None:
    """Return a cached result for a semantically similar topic, or None."""
    _prune()
    if not _cache:
        return None

    query_vec=embeddings.embed_query(topic)

    best,best_score=None,0.0
    for entry in _cache:
        score=_cosine(query_vec, entry["embedding"])
        if score>best_score:
            best, best_score= entry, score

    if best_score>=SIMILARITY_THRESHOLD:
        age=time.time()-best["stored_at"]
        print(f"[cache] HIT  {best_score:.3f}  '{topic}' -> '{best['topic']}' "
              f"(age {age / 60:.0f}m)", flush=True)
        return best["result"]

    print(f"[cache] MISS best={best_score:.3f}  '{topic}'", flush=True)
    return None

def store(topic: str, result: dict) -> None:
     """Save a result under this topic."""
     _cache.append({
         "topic": topic,
         "embedding": embeddings.embed_query(topic),
         "result": result,
         "stored_at": time.time(),
     })

     _prune()
     print(f"[cache] stored '{topic}' ({len(_cache)} entries)", flush=True)

def stats() -> dict:
    return {"entries": len(_cache), "threshold": SIMILARITY_THRESHOLD}
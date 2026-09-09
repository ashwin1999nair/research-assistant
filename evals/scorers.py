"""
Deterministic scorers for the Research Assistant eval harness.
 
Every scorer takes (case, result) and returns {"name", "value", "passed"}.
No LLM calls here — these are free, instant, and identical every run.
    case — one entry from test_set.json
    result — dict with keys: report, urls, retrieved_chunks, latency_s, error
"""

def completed(case: dict, result: dict)-> dict:
    """Did the pipeline return a report without throwing?"""
    ok=result.get("error") is None and bool(result.get("report"))
    return {"name":"completed", "value":ok, "passed":ok}

def word_count(case:dict, result:dict)-> dict:
    """Is the report length within this case's bounds?"""
    count=len(result.get("report","").split())
    lo=case.get("min_words",0)
    hi=case.get("max_words", 10000)
    return{"name":"word_count","value": count, "passed":lo<=hi}

def source_diversity(case:dict, result:dict)-> dict:
    """
    Fraction of scraped pages that contributed at least one retrieved chunk.
 
    1.0 means every page fed the report.
    0.2 means every chunk came from a single page — the report is
    effectively single-source no matter how many URLs search found.
 
    This is a partial stand-in for context recall, which needs ground
    truth we don't have.
    """
    chunks=result.get("retrieved_chunks") or []
    scraped=result.get("urls") or []

    if not scraped:
        return{"name":"source_diversity", "value":0.0, "passed": False}

    used={c["source"] for c in chunks}
    score=len(used)/len(scraped)

    return{"name":"source_diversity", "value":round(score,3),"passed":score>=0.4}

def latency_s(case:dict, result:dict)-> dict:
    """Wall-clock seconds. Tracked, never failed — this is a metric, not a gate."""
    value=round(result.get("latency_s",0.0),1)
    return {"name":"latency_s", "value": value, "passed": True}

def check_fact(case:dict, result:dict)-> dict:
    """
    Post-cutoff cases only. Does the report contain a fact the model
    could not have known from training?
 
    If yes, retrieval supplied it — which is the whole point of the
    post_cutoff category.
    """
    needle=case["check_fact"]
    found=needle.lower() in result.get("report","").lower()
    return{"name": "check_fact", "value": found, "passed":found}

def check_absent(case:dict, result:dict)-> dict:
     """
    Injection case only. Does the report leak text that should never
    reach the user — e.g. a fragment of the synthesis prompt?
    """
     needle=case["check_absent"]
     leaked=needle.lower() in result.get("report", "").lower()
     return{"name": "check_absent", "value": not leaked, "passed": not leaked}

# ---------------------------------------------------------------------
# Scorer selection
# -------------------------------------------------------------------
GENERAL = [completed, word_count, source_diversity, latency_s]

def scorers_for(case: dict) -> list:
    if case.get("expected_behaviour") == "rejected":
        chosen = [completed]
    else:
        chosen = list(GENERAL)

    if "check_fact" in case:
        chosen.append(check_fact)
    if "check_absent" in case:
        chosen.append(check_absent)

    return chosen


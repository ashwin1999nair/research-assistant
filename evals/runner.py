"""
Eval harness runner for the Research Assistant.
 
Loads test_set.json, runs each case through the API, applies the scorers,
and writes both a per-case table and an aggregate summary.
 
Requires the API to be running:
    docker-compose up
 
Usage:
    python -m evals.runner # all 15 cases
    python -m evals.runner --only wd_01 # one case
    python -m evals.runner --only narrow # one category
    python -m evals.runner --api-url http://localhost:8000
"""

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path
import requests

from evals.scorers import scorers_for

HERE = Path(__file__).parent
TEST_SET = HERE / "test_set.json"
RESULTS_DIR = HERE / "results"

# ---------------------------------------------------------------------
# Running one case
# ---------------------------------------------------------------------
 
def run_case(case:dict, api_url:str, timeout:int)-> dict:
    """
    Call the API for one case and build the result dict the scorers expect.
 
    The API returns report / urls / retrieved_chunks. We add latency_s and
    error ourselves — the API doesn't time itself, and a crash needs to be
    recorded as a data point rather than ending the whole run.
    """
    start=time.time()

    try:
        response = requests.post(
            f"{api_url}/research",
            json={"topic": case["query"], "debug": True},
            timeout=timeout,
        )
        elapsed = time.time() - start

        if response.status_code!=200:
            return {
                "report": "",
                "urls": [],
                "retrieved_chunks": [],
                "latency_s": elapsed,
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
            }
        data=response.json()
        return {
            "report": data.get("report", ""),
            "urls": data.get("urls") or [],
            "retrieved_chunks": data.get("retrieved_chunks") or [],
            "latency_s": elapsed,
            "error": None,
        }

    except Exception as e:
        return {
            "report": "",
            "urls": [],
            "retrieved_chunks": [],
            "latency_s": time.time() - start,
            "error": f"{type(e).__name__}: {e}",
        }

def score_case(case:dict, result:dict)->dict:
    """
    Apply the applicable scorers and decide whether the case passed overall.
 
    The pass/fail inversion lives here, not in the scorers. For a case
    expecting rejection, a report coming back is a FAILURE — it means
    input validation didn't fire.
    """
    row = {
        "id": case["id"],
        "category": case["category"],
        "expected": case["expected_behaviour"],
    }
    scores=[scorer(case, result) for scorer in scorers_for(case)]
    for s in scores:
        row[s["name"]]=s["value"]

    got_report=result["error"] is None and bool(result["report"])

    if case["expected_behaviour"]== "rejected":
        # Wanted a rejection. Getting a report means the guardrail is missing.
        row["passed"]=not got_report
    else:
        # Wanted a report. Every applicable scorer must pass.
        row["passed"] = all(s["passed"] for s in scores)

    row["error"]=result["error"] or ""
    return row

# ---------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------
 
def aggregate(rows: list) -> dict:
    """
    Roll per-case rows up into the metrics MLflow will log.
 
    Per-category pass rates matter more than the overall number — a single
    "82% pass" hides which part of the system is weak.
    """
    def mean_of(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 3) if vals else None

    def pct_passed(subset):
        return round(sum(r["passed"] for r in subset) / len(subset), 3) if subset else None

    metrics = {
        "n_cases": len(rows),
        "pass_rate": pct_passed(rows),
        "mean_word_count": mean_of("word_count"),
        "mean_source_diversity": mean_of("source_diversity"),
        "mean_latency_s": mean_of("latency_s"),
        "n_errors": sum(1 for r in rows if r["error"]),
    }

    latencies = sorted(r["latency_s"] for r in rows if isinstance(r.get("latency_s"), (int, float)))
    if latencies:
        idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
        metrics["p95_latency_s"] = round(latencies[idx], 1)

    for cat in sorted({r["category"] for r in rows}):
        subset = [r for r in rows if r["category"] == cat]
        metrics[f"pass_rate_{cat}"] = pct_passed(subset)
 
    return metrics

# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def write_results(rows: list, metrics: dict, config: dict) -> Path:
    """Write results.csv (for reading) and results.json (for MLflow)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / stamp
    run_dir.mkdir()
 
    # Union of all keys — cases run different scorers, so rows differ in shape
    fieldnames = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
 
    with open(run_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
 
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"config": config, "metrics": metrics, "cases": rows}, f, indent=2)
 
    return run_dir
 
 
def print_summary(rows: list, metrics: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'ID':<10} {'CATEGORY':<16} {'PASS':<6} {'WORDS':<7} {'DIV':<6} {'SECS':<6}")
    print("-" * 70)
    for r in rows:
        print(
            f"{r['id']:<10} {r['category']:<16} "
            f"{'yes' if r['passed'] else 'NO':<6} "
            f"{r.get('word_count', '-'):<7} "
            f"{r.get('source_diversity', '-'):<6} "
            f"{r.get('latency_s', '-'):<6}"
        )
    print("=" * 70)
 
    for k, v in metrics.items():
        print(f"  {k:<28} {v}")
 
    failed = [r for r in rows if not r["passed"]]
    if failed:
        print(f"\n  {len(failed)} failed: {', '.join(r['id'] for r in failed)}")
        for r in failed:
            if r["error"]:
                print(f"    {r['id']}: {r['error'][:120]}")
 
 
# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run a single case id (wd_01) or category (narrow)")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=int, default=400)
    args = parser.parse_args()
 
    with open(TEST_SET, encoding="utf-8") as f:
        test_set = json.load(f)
 
    cases = test_set["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only or c["category"] == args.only]
        if not cases:
            print(f"No cases match '{args.only}'")
            return
 
    print(f"Running {len(cases)} case(s) against {args.api_url}")
    print(f"Estimated time: {len(cases) * 1.5:.0f}-{len(cases) * 2:.0f} minutes\n")
 
    rows = []
    for i, case in enumerate(cases, 1):
        label = case["query"][:45] or "(empty)"
        print(f"[{i}/{len(cases)}] {case['id']:<10} {label}", flush=True)
 
        result = run_case(case, args.api_url, args.timeout)
        row = score_case(case, result)
        rows.append(row)
 
        status = "pass" if row["passed"] else "FAIL"
        print(f"           -> {status}  ({result['latency_s']:.0f}s)", flush=True)
 
    metrics = aggregate(rows)
    config = {
        "test_set_version": test_set.get("version"),
        "api_url": args.api_url,
        "filter": args.only,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
 
    print_summary(rows, metrics)
    run_dir = write_results(rows, metrics, config)
    print(f"\nWritten to {run_dir}")
 
 
if __name__ == "__main__":
    main()
    
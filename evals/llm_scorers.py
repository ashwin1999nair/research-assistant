"""
LLM-judged scorers for the Research Assistant eval harness.
 
Currently: faithfulness only.
 
Faithfulness = fraction of claims in the report that are supported by the
retrieved chunks. It measures GROUNDING, not truth — a claim can be true in
the world and still score as unfaithful if it came from the model's training
data rather than the retrieved sources.
 
Same signature as the deterministic scorers: (case, result) -> {name, value, passed}
 
NOTE: the judge is the same model that writes the reports, which introduces
self-preference bias. A production setup would use a separate judge model.
"""
import json
import os
import re
 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
 
load_dotenv()
 
# temperature=0: the judge answers yes/no questions with correct answers.
# Randomness here is pure noise on top of the signal we want to measure.
judge = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
)

MAX_CLAIMS=30          # cap the verify prompt size
MAX_CHUNK_CHARS=1200   # trim very long chunks before sending

def _parse_json(text: str):
    """
    Gemini wraps JSON in markdown fences despite being told not to.
    Strip them, then fall back to grabbing the outermost array.
    """
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

def _decompose(report: str) -> list:
    """
    Step 1 — split the report into atomic factual claims.
 
    This step is what makes faithfulness work. Asking "is this report
    grounded?" gets you a useless yes. Asking about one claim at a time
    gets you a real answer.
    """
    prompt = f"""Break the following report into atomic factual claims.
 
An atomic claim states ONE fact and can be verified on its own. Split
compound sentences. Ignore section headings, transitions, and hedging
language that asserts nothing.
 
Return ONLY a JSON array of strings. No markdown, no explanation.
 
Report:
{report}
"""
    response=judge.invoke(prompt)
    claims=_parse_json(response.content)

    if not isinstance(claims,list):
        return []

    claims=[c for c in claims if isinstance(c,str) and c.strip()]
    return claims[:MAX_CLAIMS]

def _verify(claims: list, chunks: list) -> list:
    """
    Step 2 — check every claim against the retrieved context in ONE call.
 
    Batched deliberately. Verifying 25 claims individually across 15 cases
    would be ~375 API calls per eval run; batched it is ~30.
    """
    context = "\n\n---\n\n".join(c["text"][:MAX_CHUNK_CHARS] for c in chunks)
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
 
    prompt = f"""You are checking whether claims are supported by a source text.
 
A claim is SUPPORTED only if it can be directly inferred from the context
below. If the claim is true in the real world but the context does not
contain it, mark it UNSUPPORTED. You are judging grounding, not truth.
 
Context:
{context}
 
Claims:
{numbered}
 
Return ONLY a JSON array with one object per claim, in the same order:
[{{"n": 1, "supported": true}}, {{"n": 2, "supported": false}}]
 
No markdown, no explanation.
"""
    response=judge.invoke(prompt)
    verdicts=_parse_json(response.content)

    if not isinstance(verdicts,list):
        return []

    # The model sometimes returns fewer verdicts than claims. Map by the
    # "n" field rather than trusting positional alignment.
    by_n = {}
    for v in verdicts:
        if isinstance(v, dict) and "n" in v:
            by_n[v["n"]] = bool(v.get("supported"))

    # Score over what was actually judged. Defaulting a missing verdict
    # to False would count an unchecked claim as a hallucination — a
    # missing measurement is not a failed measurement.
    judged = [by_n[i] for i in range(1, len(claims) + 1) if i in by_n]

    if len(judged) < len(claims):
        print(f"[judge returned {len(judged)}/{len(claims)} verdicts]")

    # Too little coverage to trust. Return empty so the case visibly
    # fails rather than producing a confident-looking score.
    if len(judged) < len(claims) * 0.8:
        return []

    return judged

def faithfulness(case: dict, result: dict) -> dict:
    """
    supported claims / judged claims, in [0, 1].

    Note the denominator is claims the judge actually returned a verdict
    for, not every claim in the report. Scoring an unjudged claim as
    unsupported would count a missing measurement as a hallucination.

    If the judge covers less than 80% of claims, or fails outright,
    returns 0.0 with passed=False — an unscored case should be visible,
    not silently dropped.
    """
    report=result.get("report","")
    chunks=result.get("retrieved_chunks") or []

    if not report or not chunks:
        return {"name": "faithfulness", "value": 0.0, "passed": False}

    try:
        claims=_decompose(report)
        if not claims:
            return {"name": "faithfulness", "value": 0.0, "passed": False}

        verdicts=_verify(claims, chunks)
        if not verdicts:
            return {"name": "faithfulness", "value": 0.0, "passed": False}

        score=sum(verdicts)/len(verdicts)

    except Exception as e:
        print(f"[faithfulness failed: {type(e).__name__}: {e}]")
        return {"name": "faithfulness", "value": 0.0, "passed": False}

    # 0.7 is a starting guess. Revisit once you have seen the spread
    # across a full run — thresholds should come from data.
    return {
        "name": "faithfulness",
        "value": round(score, 3),
        "passed": score >= 0.7,
    }
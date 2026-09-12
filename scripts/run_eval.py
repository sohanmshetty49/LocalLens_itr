"""Automated, scored evaluation for LocalLens.

Unlike scripts/run_smoke_eval.py (which just prints raw answers for manual
inspection), this script scores each case against explicit expectations and
produces a single aggregate number that CI can gate on:

    python scripts/run_eval.py --threshold 0.8

Each case is scored on two axes:
  - grounding: did the answer come back with citations/place cards (or, for
    cases marked `expect_ungrounded`, correctly refuse to answer)?
  - keyword coverage: does the answer/why/tips text mention at least one of
    the expected keywords for that query?

The aggregate score is the mean case score across all cases. Latency is
reported for visibility but does not currently affect pass/fail.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from locallens.config import get_settings
from locallens.service import LocalLensService

# These cases target the small San Francisco fixture corpus used by the test
# suite (tests/fixtures/). Against the full multi-city corpus this same
# script works unmodified -- just point it at a Settings with more coverage.
EVAL_CASES: list[dict[str, Any]] = [
    {
        "query": "What should I do in San Francisco if I want a practical first itinerary?",
        "keywords": ["san francisco"],
    },
    {
        "query": "Best rated spot in San Francisco over 4.0?",
        "keywords": ["san francisco"],
    },
    {
        "query": "Where are the best sunset or scenic spots in San Francisco?",
        "keywords": ["san francisco"],
    },
    {
        "query": "What hidden gems or locals-only spots should I know in San Francisco?",
        "keywords": ["san francisco"],
    },
    {
        "query": "What should I know about safety in Atlantis?",
        "keywords": [],
        "expect_ungrounded": True,
    },
]


def _score_case(service: LocalLensService, case: dict[str, Any]) -> dict[str, Any]:
    query = str(case["query"])
    started = time.perf_counter()
    response = service.answer(query)
    latency_ms = (time.perf_counter() - started) * 1000

    grounded = bool(response.citations or response.place_cards)
    expect_ungrounded = bool(case.get("expect_ungrounded", False))
    grounding_score = (0.0 if grounded else 1.0) if expect_ungrounded else (1.0 if grounded else 0.0)

    keywords = [str(keyword).lower() for keyword in case.get("keywords", [])]
    haystack = " ".join([response.answer, response.why_this_recommendation, *response.key_tips]).lower()
    keyword_score = 1.0 if not keywords or any(keyword in haystack for keyword in keywords) else 0.0

    case_score = statistics.mean([grounding_score, keyword_score])
    return {
        "query": query,
        "grounded": grounded,
        "grounding_score": grounding_score,
        "keyword_score": keyword_score,
        "case_score": case_score,
        "latency_ms": latency_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated scoring eval for LocalLens.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum aggregate score required to pass.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional project root override (defaults to the main data/ dirs; use tests/fixtures for the CI fixture corpus).",
    )
    args = parser.parse_args()

    settings = get_settings(args.project_root) if args.project_root else get_settings()
    service = LocalLensService(settings)
    results = [_score_case(service, case) for case in EVAL_CASES]

    print(f"{'Query':<65} {'Grounded':<10} {'Keyword':<9} {'Score':<7} {'Latency(ms)':<12}")
    print("-" * 105)
    for result in results:
        print(
            f"{str(result['query'])[:63]:<65} "
            f"{str(result['grounded']):<10} "
            f"{result['keyword_score']:<9.1f} "
            f"{result['case_score']:<7.2f} "
            f"{result['latency_ms']:<12.1f}"
        )

    aggregate_score = statistics.mean(result["case_score"] for result in results)
    avg_latency = statistics.mean(result["latency_ms"] for result in results)
    print("-" * 105)
    print(f"Aggregate score: {aggregate_score:.3f} (threshold {args.threshold:.2f}) | avg latency: {avg_latency:.1f}ms")

    if aggregate_score < args.threshold:
        print(f"FAIL: aggregate score {aggregate_score:.3f} is below threshold {args.threshold:.2f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

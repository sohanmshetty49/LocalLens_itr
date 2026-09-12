"""Builds a small, real LocalLens corpus used only by the test suite.

The main `data/processed/` and `artifacts/` directories are gitignored (see
`.gitignore`), so a fresh checkout has no SQLite database or embeddings to
query against. Rather than mocking the retrieval pipeline, this script runs
the *actual* ingestion + indexing pipeline against a single city (San
Francisco), skipping Reddit (which needs credentials), and writes the result
under `tests/fixtures/` so it can be checked into git and reused by CI
without any network access at test time.

Re-run this script whenever the ingestion/indexing pipeline changes in a way
that affects the fixture's shape:

    LOCALLENS_VECTOR_BACKEND=numpy python tests/fixtures/build_fixture_corpus.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOCALLENS_VECTOR_BACKEND", "numpy")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from locallens.config import get_settings
from locallens.ingestion import build_corpus
from locallens.retrieval.dense import build_dense_embeddings
from locallens.storage import connect, load_chunks

FIXTURE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    settings = get_settings(FIXTURE_ROOT)
    counts = build_corpus(
        settings,
        selected_locations=["San Francisco"],
        include_reddit=False,
        include_places=True,
        include_local_web=True,
    )
    print(f"Fixture corpus: {counts}")

    conn = connect(settings.database_path)
    chunks = load_chunks(conn)
    _, ids, backend = build_dense_embeddings(settings, chunks)
    print(f"Fixture embeddings: {len(ids)} chunks indexed via {backend}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The fixture corpus is built with the simpler numpy vector backend (see
# tests/fixtures/build_fixture_corpus.py) so tests don't depend on a local
# Qdrant/on-disk index; force the same backend when *running* tests too.
os.environ.setdefault("LOCALLENS_VECTOR_BACKEND", "numpy")

FIXTURES_ROOT = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def service():
    """A LocalLensService bound to the small, real, checked-in test fixture
    corpus (San Francisco only). Session-scoped because constructing it
    loads the sentence-transformers embedding model, which is comparatively
    slow to do once per test."""
    from locallens.config import get_settings
    from locallens.service import LocalLensService

    settings = get_settings(FIXTURES_ROOT)
    return LocalLensService(settings)

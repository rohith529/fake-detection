"""Unit tests for RSS retrieval, query formulation, and canonical domain resolution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from retriever import formulate_search_query, resolve_canonical_domain


class DummyEntry:
    def __init__(self, title, source=None, link=""):
        self.title = title
        self.source = source or {}
        self.link = link

    def get(self, key, default=None):
        return getattr(self, key, default)


def test_formulate_search_query():
    text = "Fact Check: Did the Reserve Bank of India announce a 50 bps repo rate cut today?"
    query = formulate_search_query(text)
    # Query should remove boilerplate like "Fact Check:" and "Did" and pick informative terms
    assert "fact check" not in query.lower()
    terms = [t.lower() for t in query.split()]
    assert "reserve" in terms or "bank" in terms or "india" in terms or "repo" in terms


def test_resolve_canonical_domain_from_source_title():
    entry = DummyEntry(
        title="Inflation rises slightly in October",
        source={"title": "The Hindu", "url": "https://news.google.com/rss/articles/CBMi..."}
    )
    domain = resolve_canonical_domain(entry)
    assert domain == "thehindu.com"


def test_resolve_canonical_domain_from_title_suffix():
    # When Google strips entry.source entirely but keeps title suffix
    entry = DummyEntry(
        title="Supreme Court grants bail in landmark hearing - NDTV",
        source={},
        link="https://news.google.com/rss/articles/random_redirect"
    )
    domain = resolve_canonical_domain(entry)
    assert domain == "ndtv.com"


def test_resolve_canonical_domain_from_direct_link():
    entry = DummyEntry(
        title="Global markets rally as tech earnings beat expectations",
        source={},
        link="https://www.reuters.com/markets/asia/global-markets-2026"
    )
    domain = resolve_canonical_domain(entry)
    assert domain == "reuters.com"

from types import SimpleNamespace

from omegaconf import OmegaConf

from zotero_arxiv_daily.retriever.arxiv_weekly_retriever import (
    OpenAlexClient,
    SemanticScholarClient,
    build_weekly_arxiv_query,
)
from zotero_arxiv_daily.retriever.huggingface_trending_retriever import HuggingFaceTrendingRetriever


class StubResponse:
    def __init__(self, payload=None, text="", status_code=200):
        self.payload = payload or {}
        self.text = text
        self.status_code = status_code
        self.headers = {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class StubSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def test_build_weekly_arxiv_query_uses_submitted_date_window():
    from datetime import datetime, timezone

    query = build_weekly_arxiv_query(
        ["cs.AI", "cs.CV"],
        7,
        now=datetime(2026, 7, 20, tzinfo=timezone.utc),
    )

    assert "submittedDate:[202607130000 TO 202607202359]" in query
    assert "(cat:cs.AI OR cat:cs.CV)" in query


def test_semantic_scholar_author_filter_returns_first_last_max_citations():
    session = StubSession(
        StubResponse(
            {
                "authors": [
                    {"name": "First", "citationCount": 900},
                    {"name": "Middle", "citationCount": 10000},
                    {"name": "Last", "citationCount": 1200},
                ]
            }
        )
    )
    client = SemanticScholarClient(session=session)

    assert client.first_last_author_max_citations("2607.00001") == 1200


def test_semantic_scholar_batch_author_filter_returns_counts_by_arxiv_id():
    session = StubSession(
        StubResponse(
            [
                {
                    "authors": [
                        {"name": "First", "citationCount": 900},
                        {"name": "Last", "citationCount": 1200},
                    ]
                },
                None,
            ]
        )
    )
    client = SemanticScholarClient(session=session)

    assert client.batch_first_last_author_max_citations(["2607.00001", "2607.00002"]) == {
        "2607.00001": 1200,
        "2607.00002": None,
    }
    assert session.calls[0][1]["json"] == {"ids": ["ARXIV:2607.00001", "ARXIV:2607.00002"]}


def test_semantic_scholar_batch_marks_all_invalid_batch_as_missing():
    session = StubSession(StubResponse({"error": "No valid paper ids given"}, status_code=400))
    client = SemanticScholarClient(session=session)

    assert client.batch_first_last_author_max_citations(["2607.00001"]) == {"2607.00001": None}


def test_openalex_author_filter_returns_first_last_max_citations():
    session = StubSession(StubResponse({"results": [{"display_name": "Author", "cited_by_count": 1500}]}))
    client = OpenAlexClient(session=session, mailto="test@example.com")
    raw_paper = SimpleNamespace(
        authors=[
            SimpleNamespace(name="First Author"),
            SimpleNamespace(name="Middle Author"),
            SimpleNamespace(name="Last Author"),
        ]
    )

    assert client.first_last_author_max_citations(raw_paper) == 1500
    assert [call[1]["params"]["search"] for call in session.calls] == ["First Author", "Last Author"]
    assert session.calls[0][1]["params"]["mailto"] == "test@example.com"


def test_huggingface_trending_parses_unique_ids(monkeypatch):
    html = """
    <a href="/papers/2607.00001">A</a>
    <a href="/papers/2607.00002">B</a>
    <a href="/papers/2607.00001">A duplicate</a>
    """
    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.huggingface_trending_retriever.requests.get",
        lambda *args, **kwargs: StubResponse(text=html),
    )

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        def results(self, search):
            return [
                SimpleNamespace(
                    title="Paper A",
                    authors=[SimpleNamespace(name="Author A")],
                    summary="Abstract A",
                    pdf_url="https://arxiv.org/pdf/2607.00001",
                    get_short_id=lambda: "2607.00001v1",
                ),
                SimpleNamespace(
                    title="Paper B",
                    authors=[SimpleNamespace(name="Author B")],
                    summary="Abstract B",
                    pdf_url="https://arxiv.org/pdf/2607.00002",
                    get_short_id=lambda: "2607.00002v1",
                ),
            ]

    monkeypatch.setattr(
        "zotero_arxiv_daily.retriever.huggingface_trending_retriever.arxiv.Client",
        StubClient,
    )
    config = OmegaConf.create(
        {
            "source": {"huggingface_trending": {"max_results": 50}},
            "executor": {"debug": False},
        }
    )
    retriever = HuggingFaceTrendingRetriever(config)

    raw = retriever._retrieve_raw_papers()

    assert [paper.title for paper in raw] == ["Paper A", "Paper B"]

"""Tests for zotero_arxiv_daily.executor: normalize_path_patterns, filter_corpus, fetch_mendeley_corpus, E2E."""

from datetime import datetime

import pytest
from omegaconf import OmegaConf

from zotero_arxiv_daily.executor import Executor, has_llm_api_key, normalize_path_patterns
from zotero_arxiv_daily.protocol import CorpusPaper


# ---------------------------------------------------------------------------
# normalize_path_patterns — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def test_normalize_path_patterns_rejects_single_string_for_include_path():
    with pytest.raises(TypeError, match="config.mendeley.include_path must be a list"):
        normalize_path_patterns("2026/survey/**", "include_path")


def test_normalize_path_patterns_accepts_list_config_for_include_path():
    include_path = OmegaConf.create(["2026/survey/**", "2026/reading-group/**"])
    assert normalize_path_patterns(include_path, "include_path") == [
        "2026/survey/**",
        "2026/reading-group/**",
    ]


def test_normalize_path_patterns_rejects_single_string_for_ignore_path():
    with pytest.raises(TypeError, match="config.mendeley.ignore_path must be a list"):
        normalize_path_patterns("archive/**", "ignore_path")


def test_normalize_path_patterns_accepts_list_config_for_ignore_path():
    ignore_path = OmegaConf.create(["archive/**", "2025/**"])
    assert normalize_path_patterns(ignore_path, "ignore_path") == ["archive/**", "2025/**"]


def test_normalize_path_patterns_accepts_empty_list():
    assert normalize_path_patterns([], "ignore_path") == []


def test_normalize_path_patterns_accepts_none():
    assert normalize_path_patterns(None, "include_path") is None


def test_has_llm_api_key_accepts_only_non_empty_keys(config):
    from omegaconf import open_dict

    with open_dict(config):
        config.llm.api.key = "sk-test"
    assert has_llm_api_key(config)

    with open_dict(config):
        config.llm.api.key = ""
    assert not has_llm_api_key(config)

    with open_dict(config):
        config.llm.api.key = None
    assert not has_llm_api_key(config)


# ---------------------------------------------------------------------------
# filter_corpus — migrated from test_include_path.py
# ---------------------------------------------------------------------------


def _make_executor(include_patterns=None, ignore_patterns=None):
    executor = Executor.__new__(Executor)
    executor.include_path_patterns = normalize_path_patterns(include_patterns, "include_path") if include_patterns else None
    executor.ignore_path_patterns = normalize_path_patterns(ignore_patterns, "ignore_path") if ignore_patterns else None
    return executor


def test_filter_corpus_matches_any_path_against_any_pattern():
    executor = _make_executor(include_patterns=["2026/survey/**", "2026/reading-group/**"])
    corpus = [
        CorpusPaper(title="Survey Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a", "archive/misc"]),
        CorpusPaper(title="Reading Group Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["notes/inbox", "2026/reading-group/week-1"]),
        CorpusPaper(title="Excluded Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Survey Paper", "Reading Group Paper"]


def test_filter_corpus_excludes_papers_matching_ignore_path():
    executor = _make_executor(ignore_patterns=["archive/**", "2025/**"])
    corpus = [
        CorpusPaper(title="Active Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Archived Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["archive/misc"]),
        CorpusPaper(title="Old Paper", abstract="", added_date=datetime(2026, 1, 3), paths=["2025/other/topic"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Active Paper"]


def test_filter_corpus_ignore_path_takes_precedence_over_include_path():
    executor = _make_executor(include_patterns=["2026/**"], ignore_patterns=["2026/ignore/**"])
    corpus = [
        CorpusPaper(title="Included Paper", abstract="", added_date=datetime(2026, 1, 1), paths=["2026/survey/topic-a"]),
        CorpusPaper(title="Ignored Paper", abstract="", added_date=datetime(2026, 1, 2), paths=["2026/ignore/topic-b"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert [p.title for p in filtered] == ["Included Paper"]


def test_filter_corpus_no_filters_returns_all():
    executor = _make_executor()
    corpus = [
        CorpusPaper(title="Paper A", abstract="", added_date=datetime(2026, 1, 1), paths=["foo"]),
        CorpusPaper(title="Paper B", abstract="", added_date=datetime(2026, 1, 2), paths=["bar"]),
    ]
    filtered = executor.filter_corpus(corpus)
    assert filtered == corpus


# ---------------------------------------------------------------------------
# fetch_mendeley_corpus
# ---------------------------------------------------------------------------


def test_fetch_mendeley_corpus(config, monkeypatch):
    from tests.canned_responses import make_stub_mendeley_client

    stub_mendeley = make_stub_mendeley_client()

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.mendeley_client = stub_mendeley
    corpus = executor.fetch_mendeley_corpus()

    assert len(corpus) == 2
    assert corpus[0].title == "Stub Paper 1"
    assert "survey/topic-a" in corpus[0].paths[0]


def test_fetch_mendeley_corpus_paper_with_zero_folders(config):
    from tests.canned_responses import make_stub_mendeley_client

    documents = [
        {
            "id": "DOC3",
            "title": "No Folder Paper",
            "abstract": "Abstract.",
            "created": "2026-03-01T00:00:00.000Z",
        }
    ]
    stub_mendeley = make_stub_mendeley_client(documents=documents, folder_documents={})

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.mendeley_client = stub_mendeley
    corpus = executor.fetch_mendeley_corpus()

    assert len(corpus) == 1
    assert corpus[0].paths == []


def test_fetch_mendeley_corpus_skips_documents_without_abstract(config):
    from tests.canned_responses import make_stub_mendeley_client

    documents = [
        {
            "id": "DOC1",
            "title": "Has Abstract",
            "abstract": "Abstract.",
            "created": "2026-03-01T00:00:00.000Z",
        },
        {
            "id": "DOC2",
            "title": "No Abstract",
            "abstract": "",
            "created": "2026-03-02T00:00:00.000Z",
        },
    ]
    stub_mendeley = make_stub_mendeley_client(documents=documents, folder_documents={})

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.mendeley_client = stub_mendeley
    corpus = executor.fetch_mendeley_corpus()

    assert [paper.title for paper in corpus] == ["Has Abstract"]


# ---------------------------------------------------------------------------
# E2E: Executor.run()
# ---------------------------------------------------------------------------


def test_run_end_to_end(config, monkeypatch):
    """Full pipeline: Mendeley fetch -> filter -> retrieve -> rerank -> TLDR -> email."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import (
        make_sample_corpus,
        make_sample_paper,
        make_stub_openai_client,
        make_stub_smtp,
        make_stub_mendeley_client,
    )

    # Config: source=["arxiv"], reranker="api", send_empty=false
    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    # 1. Stub Mendeley
    stub_mendeley = make_stub_mendeley_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.MendeleyClient.from_config", lambda config: stub_mendeley)

    # 2. Stub OpenAI (for reranker + TLDR/affiliations)
    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)
    retrieved = [
        make_sample_paper(title="E2E Paper 1", score=None),
        make_sample_paper(title="E2E Paper 2", score=None),
    ]

    # Import to register the arxiv retriever
    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(
        registered_retrievers["arxiv"],
        "retrieve_papers",
        lambda self: retrieved,
    )

    # 4. Stub SMTP
    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))

    # 5. Stub sleep (reranker/retriever)
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    # 6. Run
    executor = Executor(config)
    executor.run()

    # Assertions
    assert len(sent) == 1, "Email should have been sent"
    _, _, email_body = sent[0]
    assert "text/html" in email_body


def test_run_no_papers_send_empty_false(config, monkeypatch):
    """When no papers are found and send_empty=false, no email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_mendeley_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False

    stub_mendeley = make_stub_mendeley_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.MendeleyClient.from_config", lambda config: stub_mendeley)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 0, "No email should be sent when no papers and send_empty=false"


def test_run_no_papers_send_empty_true(config, monkeypatch):
    """When no papers are found and send_empty=true, empty email is sent."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_stub_openai_client, make_stub_smtp, make_stub_mendeley_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = True

    stub_mendeley = make_stub_mendeley_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.MendeleyClient.from_config", lambda config: stub_mendeley)

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", lambda **kw: stub_client)
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: [])

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1, "Email should be sent even with no papers when send_empty=true"
    _, _, body = sent[0]
    assert "text/html" in body


def test_run_skips_tldr_and_affiliations_when_openai_key_missing(config, monkeypatch):
    """The feed still sends ranked papers when LLM enrichment is disabled."""
    import smtplib

    from omegaconf import open_dict

    from tests.canned_responses import make_sample_paper, make_stub_smtp, make_stub_mendeley_client

    with open_dict(config):
        config.executor.source = ["arxiv"]
        config.executor.reranker = "api"
        config.executor.send_empty = False
        config.llm.api.key = None

    stub_mendeley = make_stub_mendeley_client()
    monkeypatch.setattr("zotero_arxiv_daily.executor.MendeleyClient.from_config", lambda config: stub_mendeley)

    def fail_openai_constructor(**kwargs):
        raise AssertionError("OpenAI should not be constructed without an LLM API key")

    monkeypatch.setattr("zotero_arxiv_daily.executor.OpenAI", fail_openai_constructor)

    from tests.canned_responses import make_stub_openai_client

    stub_client = make_stub_openai_client()
    monkeypatch.setattr("zotero_arxiv_daily.reranker.api.OpenAI", lambda **kw: stub_client)

    retrieved = [make_sample_paper(title="No LLM Paper", score=None)]

    import zotero_arxiv_daily.retriever.arxiv_retriever  # noqa: F401

    from zotero_arxiv_daily.retriever.base import registered_retrievers

    monkeypatch.setattr(registered_retrievers["arxiv"], "retrieve_papers", lambda self: retrieved)

    sent = []
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)

    executor = Executor(config)
    executor.run()

    assert len(sent) == 1
    assert retrieved[0].tldr is None
    assert retrieved[0].affiliations is None


def test_retrieve_and_rerank_by_source_applies_top_k_and_deduplicates(config):
    from types import SimpleNamespace

    from omegaconf import open_dict

    from tests.canned_responses import make_sample_corpus, make_sample_paper
    from zotero_arxiv_daily.history import RecommendationHistory

    with open_dict(config):
        config.executor.per_source_top_k = {"huggingface_trending": 1, "arxiv_weekly": 2}

    shared_hf = make_sample_paper(source="huggingface_trending", source_id="2607.00001", title="Shared HF")
    shared_arxiv = make_sample_paper(source="arxiv_weekly", source_id="2607.00001", title="Shared arXiv")
    arxiv_only = make_sample_paper(source="arxiv_weekly", source_id="2607.00002", title="arXiv Only")

    executor = Executor.__new__(Executor)
    executor.config = config
    executor.history = RecommendationHistory(None)
    executor.retrievers = {
        "huggingface_trending": SimpleNamespace(retrieve_papers=lambda: [shared_hf]),
        "arxiv_weekly": SimpleNamespace(retrieve_papers=lambda: [shared_arxiv, arxiv_only]),
    }

    class StubReranker:
        def rerank(self, papers, corpus):
            for index, paper in enumerate(papers):
                paper.score = 10 - index
            return papers

    executor.reranker = StubReranker()

    selected = executor.retrieve_and_rerank_by_source(make_sample_corpus())

    assert [paper.title for paper in selected] == ["Shared HF", "arXiv Only"]

"""Shared stub factories for tests. No unittest.mock anywhere."""

from datetime import datetime
from types import SimpleNamespace

from zotero_arxiv_daily.protocol import CorpusPaper, Paper


# ---------------------------------------------------------------------------
# OpenAI client stub
# ---------------------------------------------------------------------------

_AFFILIATION_MARKER = "You are an assistant who perfectly extracts affiliations"
_AFFILIATION_RESPONSE = '["TsingHua University","Peking University"]'
_TLDR_RESPONSE = "Hello! How can I assist you today?"


def _make_chat_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
                index=0,
            )
        ],
        id="chatcmpl-stub",
        created=1765197615,
        model="gpt-4o-mini-2024-07-18",
        object="chat.completion",
    )


def _stub_chat_create(**kwargs):
    messages = kwargs.get("messages", [])
    request_str = str(messages)
    if _AFFILIATION_MARKER in request_str:
        return _make_chat_response(_AFFILIATION_RESPONSE)
    return _make_chat_response(_TLDR_RESPONSE)


def _stub_embeddings_create(**kwargs):
    inputs = kwargs.get("input", [])
    n = len(inputs) if isinstance(inputs, list) else 1
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3], index=i, object="embedding") for i in range(n)],
        model="text-embedding-3-large",
        object="list",
    )


def make_stub_openai_client():
    """Return a SimpleNamespace that quacks like openai.OpenAI().

    chat.completions.create() and embeddings.create() behave identically
    to the Docker mock_openai server that CI previously relied on.
    """
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_stub_chat_create),
        ),
        embeddings=SimpleNamespace(create=_stub_embeddings_create),
    )


# ---------------------------------------------------------------------------
# Mendeley client stub
# ---------------------------------------------------------------------------

_DEFAULT_FOLDERS = [
    {"id": "FOL1", "name": "survey"},
    {"id": "FOL2", "name": "topic-a", "parent_id": "FOL1"},
]

_DEFAULT_DOCUMENTS = [
    {
        "id": "DOC1",
        "title": "Stub Paper 1",
        "abstract": "Abstract of stub paper 1.",
        "created": "2026-01-15T10:00:00.000Z",
    },
    {
        "id": "DOC2",
        "title": "Stub Paper 2",
        "abstract": "Abstract of stub paper 2.",
        "created": "2026-02-20T12:00:00.000Z",
    },
]

_DEFAULT_FOLDER_DOCUMENTS = {
    "FOL1": ["DOC2"],
    "FOL2": ["DOC1"],
}


def make_stub_mendeley_client(documents=None, folders=None, folder_documents=None, use_starred_only=False):
    """Return a SimpleNamespace that quacks like MendeleyClient."""
    docs = documents if documents is not None else _DEFAULT_DOCUMENTS
    flds = folders if folders is not None else _DEFAULT_FOLDERS
    memberships = folder_documents if folder_documents is not None else _DEFAULT_FOLDER_DOCUMENTS

    from zotero_arxiv_daily.mendeley import build_folder_paths, parse_mendeley_datetime

    def fetch_corpus():
        folder_paths = build_folder_paths(flds)
        document_paths = {}
        for folder_id, document_ids in memberships.items():
            for document_id in document_ids:
                document_paths.setdefault(document_id, []).append(folder_paths[folder_id])
        return [
            CorpusPaper(
                title=document["title"],
                abstract=document["abstract"],
                added_date=parse_mendeley_datetime(document.get("created") or document.get("last_modified")),
                paths=document_paths.get(document["id"], []),
            )
            for document in docs
            if not use_starred_only or document.get("starred")
            if document.get("title") and document.get("abstract")
        ]

    return SimpleNamespace(fetch_corpus=fetch_corpus)


# ---------------------------------------------------------------------------
# SMTP stub
# ---------------------------------------------------------------------------


def make_stub_smtp(sent_emails: list):
    """Return a class that records calls to sendmail().

    Usage:
        sent = []
        monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
        ...
        assert len(sent) == 1
        sender, recipients, body = sent[0]
    """

    class StubSMTP:
        def __init__(self, *args, **kwargs):
            pass

        def starttls(self):
            pass

        def login(self, user, password):
            pass

        def sendmail(self, sender, recipients, msg):
            sent_emails.append((sender, recipients, msg))

        def quit(self):
            pass

    return StubSMTP


# ---------------------------------------------------------------------------
# Paper / CorpusPaper factories
# ---------------------------------------------------------------------------


def make_sample_paper(**overrides) -> Paper:
    defaults = dict(
        source="arxiv",
        title="Sample Paper Title",
        authors=["Author A", "Author B", "Author C"],
        abstract="This paper explores a novel approach to widget engineering.",
        url="https://arxiv.org/abs/2026.00001",
        pdf_url="https://arxiv.org/pdf/2026.00001",
        full_text="\\begin{document} Some text. \\end{document}",
        tldr=None,
        affiliations=None,
        score=None,
    )
    defaults.update(overrides)
    return Paper(**defaults)


def make_sample_corpus(n: int = 3) -> list[CorpusPaper]:
    return [
        CorpusPaper(
            title=f"Corpus Paper {i}",
            abstract=f"Abstract for corpus paper {i}.",
            added_date=datetime(2026, 1, 1 + i),
            paths=[f"2026/survey/topic-{i}"],
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# bioRxiv canned API response
# ---------------------------------------------------------------------------

SAMPLE_BIORXIV_API_RESPONSE = {
    "messages": [{"status": "ok"}],
    "collection": [
        {
            "doi": "10.1101/2026.03.01.000001",
            "title": "A biorxiv paper",
            "authors": "Smith, J.; Doe, A.; Lee, K.",
            "abstract": "We present a novel finding.",
            "date": "2026-03-02",
            "category": "bioinformatics",
            "version": "1",
        },
        {
            "doi": "10.1101/2026.03.01.000002",
            "title": "Another biorxiv paper",
            "authors": "Wang, L.; Chen, M.",
            "abstract": "We replicate a key result.",
            "date": "2026-03-02",
            "category": "genomics",
            "version": "1",
        },
        {
            "doi": "10.1101/2026.03.01.000003",
            "title": "Old biorxiv paper",
            "authors": "Old, R.",
            "abstract": "Yesterday's paper.",
            "date": "2026-03-01",
            "category": "bioinformatics",
            "version": "1",
        },
    ],
}

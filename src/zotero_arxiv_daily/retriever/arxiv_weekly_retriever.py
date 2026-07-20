from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from time import sleep
from typing import Any

import arxiv
import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{paper_id}"
SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
OPENALEX_AUTHORS_URL = "https://api.openalex.org/authors"


def arxiv_id_from_result(result: arxiv.Result) -> str:
    return re.sub(r"v\d+$", "", result.get_short_id())


def build_weekly_arxiv_query(categories: list[str], lookback_days: int, *, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    start_s = start.strftime("%Y%m%d0000")
    end_s = now.strftime("%Y%m%d2359")
    category_query = " OR ".join(f"cat:{category}" for category in categories)
    return f"submittedDate:[{start_s} TO {end_s}] AND ({category_query})"


class SemanticScholarClient:
    def __init__(self, *, session: requests.Session | None = None, api_key: str | None = None):
        self.session = session or requests.Session()
        self.headers = {"x-api-key": api_key} if api_key else None

    def first_last_author_max_citations(self, arxiv_id: str) -> int | None:
        url = SEMANTIC_SCHOLAR_PAPER_URL.format(paper_id=arxiv_id)
        response = None
        for attempt in range(5):
            response = self.session.get(
                url,
                params={"fields": "authors.authorId,authors.name,authors.citationCount"},
                headers=self.headers,
                timeout=30,
            )
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (attempt + 1)
            logger.warning(f"Semantic Scholar rate limited arXiv:{arxiv_id}; retrying in {wait}s")
            sleep(wait)
        assert response is not None
        if response.status_code == 404:
            return None
        response.raise_for_status()
        authors = response.json().get("authors") or []
        if not authors:
            return None
        first = authors[0].get("citationCount")
        last = authors[-1].get("citationCount")
        values = [value for value in (first, last) if isinstance(value, int)]
        if not values:
            return None
        return max(values)

    def batch_first_last_author_max_citations(self, arxiv_ids: list[str]) -> dict[str, int | None]:
        if not arxiv_ids:
            return {}
        results: dict[str, int | None] = {}
        for start in range(0, len(arxiv_ids), 50):
            chunk = arxiv_ids[start : start + 50]
            response = self.session.post(
                SEMANTIC_SCHOLAR_BATCH_URL,
                params={"fields": "authors.authorId,authors.name,authors.citationCount"},
                json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in chunk]},
                headers=self.headers,
                timeout=30,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 30
                logger.warning(f"Semantic Scholar batch rate limited; retrying in {wait}s")
                sleep(wait)
                response = self.session.post(
                    SEMANTIC_SCHOLAR_BATCH_URL,
                    params={"fields": "authors.authorId,authors.name,authors.citationCount"},
                    json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in chunk]},
                    headers=self.headers,
                    timeout=30,
                )
            if response.status_code == 400:
                logger.info(
                    "Semantic Scholar returned no valid paper ids for this batch; "
                    "falling back to author-name citation lookup where possible."
                )
                results.update({arxiv_id: None for arxiv_id in chunk})
                continue
            response.raise_for_status()
            payload = response.json()
            for arxiv_id, paper in zip(chunk, payload):
                results[arxiv_id] = self._first_last_max_from_paper(paper)
        return results

    def batch_paper_metadata(self, arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not arxiv_ids:
            return {}
        results: dict[str, dict[str, Any]] = {}
        fields = "citationCount,authors.name,authors.affiliations"
        for start in range(0, len(arxiv_ids), 50):
            chunk = arxiv_ids[start : start + 50]
            response = self.session.post(
                SEMANTIC_SCHOLAR_BATCH_URL,
                params={"fields": fields},
                json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in chunk]},
                headers=self.headers,
                timeout=30,
            )
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 30
                logger.warning(f"Semantic Scholar metadata batch rate limited; retrying in {wait}s")
                sleep(wait)
                response = self.session.post(
                    SEMANTIC_SCHOLAR_BATCH_URL,
                    params={"fields": fields},
                    json={"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in chunk]},
                    headers=self.headers,
                    timeout=30,
                )
            if response.status_code == 400:
                results.update({arxiv_id: {} for arxiv_id in chunk})
                continue
            response.raise_for_status()
            payload = response.json()
            for arxiv_id, paper in zip(chunk, payload):
                results[arxiv_id] = paper or {}
        return results

    @staticmethod
    def _first_last_max_from_paper(paper: Any) -> int | None:
        if not paper:
            return None
        authors = paper.get("authors") or []
        if not authors:
            return None
        first = authors[0].get("citationCount")
        last = authors[-1].get("citationCount")
        values = [value for value in (first, last) if isinstance(value, int)]
        return max(values) if values else None


class OpenAlexClient:
    def __init__(self, *, session: requests.Session | None = None, mailto: str | None = None):
        self.session = session or requests.Session()
        self.mailto = mailto if mailto and "@" in mailto else None

    def author_citations_by_name(self, name: str) -> int | None:
        params: dict[str, str | int] = {"search": name, "per-page": 1}
        if self.mailto:
            params["mailto"] = self.mailto
        response = self.session.get(OPENALEX_AUTHORS_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        value = results[0].get("cited_by_count")
        return value if isinstance(value, int) else None

    def author_affiliation_by_name(self, name: str) -> str | None:
        params: dict[str, str | int] = {"search": name, "per-page": 1}
        if self.mailto:
            params["mailto"] = self.mailto
        response = self.session.get(OPENALEX_AUTHORS_URL, params=params, timeout=30)
        response.raise_for_status()
        results = response.json().get("results") or []
        if not results:
            return None
        author = results[0]
        institutions = author.get("last_known_institutions") or []
        for institution in institutions:
            display_name = institution.get("display_name")
            if display_name:
                return str(display_name)
        return None

    def first_last_author_max_citations(self, raw_paper: arxiv.Result) -> int | None:
        author_names = [author.name for author in raw_paper.authors]
        if not author_names:
            return None
        values = []
        for name in (author_names[0], author_names[-1]):
            try:
                value = self.author_citations_by_name(name)
            except Exception as exc:
                logger.warning(f"OpenAlex author citation lookup failed for {name}: {exc}")
                value = None
            if isinstance(value, int):
                values.append(value)
        return max(values) if values else None


@register_retriever("arxiv_weekly")
class ArxivWeeklyRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.retriever_config.category is None:
            raise ValueError("category must be specified for arxiv_weekly.")
        self.semantic_scholar = SemanticScholarClient(api_key=self.retriever_config.get("semantic_scholar_api_key"))
        self.openalex = OpenAlexClient(mailto=config.email.get("sender"))

    def _retrieve_raw_papers(self) -> list[arxiv.Result]:
        categories = list(self.retriever_config.category)
        lookback_days = int(self.retriever_config.get("lookback_days", 7))
        max_results = int(self.retriever_config.get("max_results", 200))
        query = build_weekly_arxiv_query(categories, lookback_days)
        client = arxiv.Client(num_retries=10, delay_seconds=10)
        search = arxiv.Search(
            query=query,
            max_results=10 if self.config.executor.debug else max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        raw_papers = list(client.results(search))
        min_citations = int(self.retriever_config.get("min_first_last_author_max_citations", 1000))
        arxiv_ids = [arxiv_id_from_result(paper) for paper in raw_papers]
        try:
            citation_counts = self.semantic_scholar.batch_first_last_author_max_citations(arxiv_ids)
        except Exception as exc:
            logger.warning(f"Semantic Scholar batch lookup failed: {exc}")
            citation_counts = {}
        filtered = []
        for paper, arxiv_id in zip(raw_papers, arxiv_ids):
            max_citations = citation_counts.get(arxiv_id)
            if max_citations is None:
                max_citations = self.openalex.first_last_author_max_citations(paper)
            if max_citations is None or max_citations < min_citations:
                logger.info(
                    f"Skipping {paper.title}: first/last author max citations "
                    f"{max_citations} < {min_citations}"
                )
                continue
            filtered.append(paper)
        return filtered

    def convert_to_paper(self, raw_paper: arxiv.Result) -> Paper | None:
        arxiv_id = arxiv_id_from_result(raw_paper)
        return Paper(
            source=self.name,
            source_id=arxiv_id,
            title=raw_paper.title,
            authors=[author.name for author in raw_paper.authors],
            abstract=raw_paper.summary,
            url=raw_paper.entry_id,
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )

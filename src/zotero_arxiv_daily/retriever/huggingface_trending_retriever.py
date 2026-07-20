from __future__ import annotations

import re
from typing import Any

import arxiv
import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


HF_TRENDING_URL = "https://huggingface.co/papers/trending"


@register_retriever("huggingface_trending")
class HuggingFaceTrendingRetriever(BaseRetriever):
    def _retrieve_raw_papers(self) -> list[arxiv.Result]:
        max_results = self.retriever_config.get("max_results", 50)
        response = requests.get(HF_TRENDING_URL, headers={"User-Agent": "paper-feed"}, timeout=30)
        response.raise_for_status()
        paper_ids = []
        seen = set()
        for paper_id in re.findall(r'href="/papers/(\d{4}\.\d{4,5})"', response.text):
            if paper_id in seen:
                continue
            seen.add(paper_id)
            paper_ids.append(paper_id)
            if len(paper_ids) >= max_results:
                break

        if self.config.executor.debug:
            paper_ids = paper_ids[:10]
        if not paper_ids:
            logger.warning("No Hugging Face Trending papers found.")
            return []

        client = arxiv.Client(num_retries=5, delay_seconds=5)
        results: list[arxiv.Result] = []
        for i in range(0, len(paper_ids), 20):
            search = arxiv.Search(id_list=paper_ids[i : i + 20])
            results.extend(client.results(search))
        return results

    def convert_to_paper(self, raw_paper: arxiv.Result) -> Paper:
        arxiv_id = re.sub(r"v\d+$", "", raw_paper.get_short_id())
        return Paper(
            source=self.name,
            source_id=arxiv_id,
            title=raw_paper.title,
            authors=[author.name for author in raw_paper.authors],
            abstract=raw_paper.summary,
            url=f"https://huggingface.co/papers/{arxiv_id}",
            pdf_url=raw_paper.pdf_url,
            full_text=None,
        )

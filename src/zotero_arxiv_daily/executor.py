from loguru import logger
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
from .mendeley import MendeleyClient
import random
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from .history import RecommendationHistory, paper_history_key
from .retriever.arxiv_weekly_retriever import SemanticScholarClient
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.mendeley.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.mendeley.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def has_llm_api_key(config: DictConfig) -> bool:
    return bool(config.llm.api.get("key"))


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.mendeley_client = MendeleyClient.from_config(config.mendeley)
        self.include_path_patterns = normalize_path_patterns(config.mendeley.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.mendeley.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.history = RecommendationHistory(config.executor.get("history_path"))
        arxiv_weekly_config = config.source.get("arxiv_weekly", {})
        self.semantic_scholar = SemanticScholarClient(
            api_key=arxiv_weekly_config.get("semantic_scholar_api_key")
        )
        self.openai_client = (
            OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
            if has_llm_api_key(config)
            else None
        )

    def fetch_mendeley_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching Mendeley corpus")
        corpus = self.mendeley_client.fetch_corpus()
        logger.info(f"Fetched {len(corpus)} Mendeley papers")
        return corpus
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting Mendeley papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding Mendeley papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} Mendeley papers:\n{samples}\n...")
        return corpus

    def get_source_top_k(self, source: str) -> int | None:
        per_source_top_k = self.config.executor.get("per_source_top_k")
        if not per_source_top_k:
            return None
        value = per_source_top_k.get(source)
        return int(value) if value is not None else None

    def retrieve_and_rerank_by_source(self, corpus: list[CorpusPaper]):
        selected_papers = []
        selected_keys = set()
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            try:
                papers = retriever.retrieve_papers()
            except Exception as exc:
                logger.warning(f"Skipping source {source} after retrieval failure: {exc}")
                continue
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            papers = self.history.filter_new(papers)
            logger.info(f"{len(papers)} {source} papers remain after recommendation-history filtering")
            if len(papers) == 0:
                continue
            logger.info(f"Reranking {source} papers...")
            reranked = self.reranker.rerank(papers, corpus)
            top_k = self.get_source_top_k(source)
            if top_k is not None:
                reranked = reranked[:top_k]
            for paper in reranked:
                key = paper_history_key(paper)
                if key in selected_keys:
                    continue
                selected_keys.add(key)
                selected_papers.append(paper)
        return selected_papers

    def enrich_recommendation_metadata(self, papers):
        arxiv_ids = [paper.source_id for paper in papers if paper.source_id]
        if not arxiv_ids:
            return
        try:
            metadata_by_id = self.semantic_scholar.batch_paper_metadata(arxiv_ids)
        except Exception as exc:
            logger.warning(f"Semantic Scholar paper metadata lookup failed: {exc}")
            metadata_by_id = {}

        for paper in papers:
            if not paper.source_id:
                continue
            metadata = metadata_by_id.get(paper.source_id) or {}
            citation_count = metadata.get("citationCount")
            if isinstance(citation_count, int):
                paper.citation_count = citation_count
    
    def run(self):
        corpus = self.fetch_mendeley_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            logger.error(f"No Mendeley papers found. Please check your Mendeley settings:\n{self.config.mendeley}")
            return
        reranked_papers = self.retrieve_and_rerank_by_source(corpus)
        logger.info(f"Selected {len(reranked_papers)} papers from all sources")
        if len(reranked_papers) > 0:
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Looking up paper citation counts...")
            self.enrich_recommendation_metadata(reranked_papers)
            if self.openai_client:
                logger.info("Generating TLDR and affiliations...")
                for p in tqdm(reranked_papers):
                    p.generate_tldr(self.openai_client, self.config.llm)
                    p.generate_affiliations(self.openai_client, self.config.llm)
            else:
                logger.info("No LLM API key configured. Skipping TLDR and affiliation generation.")
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        self.history.mark_recommended(reranked_papers)
        logger.info("Email sent successfully")

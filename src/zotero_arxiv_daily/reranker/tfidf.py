from .base import BaseReranker, register_reranker
import numpy as np


@register_reranker("tfidf")
class TfidfReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        features = vectorizer.fit_transform(s1 + s2)
        candidate_features = features[: len(s1)]
        corpus_features = features[len(s1) :]
        return cosine_similarity(candidate_features, corpus_features)

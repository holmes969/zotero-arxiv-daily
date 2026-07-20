import json

from zotero_arxiv_daily.history import RecommendationHistory, paper_history_key
from tests.canned_responses import make_sample_paper


def test_paper_history_key_deduplicates_arxiv_id_across_sources():
    hf = make_sample_paper(source="huggingface_trending", source_id="2607.00001")
    arxiv = make_sample_paper(source="arxiv_weekly", source_id="2607.00001")

    assert paper_history_key(hf) == paper_history_key(arxiv) == "arxiv:2607.00001"


def test_history_filters_and_marks_recommended_papers(tmp_path):
    history_path = tmp_path / "recommended_history.json"
    history = RecommendationHistory(str(history_path))
    paper = make_sample_paper(source_id="2607.00001")

    assert history.filter_new([paper]) == [paper]

    history.mark_recommended([paper])
    assert history.filter_new([paper]) == []
    data = json.loads(history_path.read_text())
    assert data["recommended"]["arxiv:2607.00001"]["title"] == paper.title


def test_history_noops_when_path_is_null():
    history = RecommendationHistory(None)
    paper = make_sample_paper(source_id="2607.00001")

    assert history.filter_new([paper]) == [paper]
    history.mark_recommended([paper])
    assert history.filter_new([paper]) == [paper]

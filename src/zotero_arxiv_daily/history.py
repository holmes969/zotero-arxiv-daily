from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import Paper


def paper_history_key(paper: Paper) -> str:
    if paper.source_id:
        return f"arxiv:{paper.source_id}"
    return f"{paper.source}:{paper.url}"


class RecommendationHistory:
    def __init__(self, path: str | None):
        self.path = Path(path) if path else None
        self._data = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path is None or not self.path.exists():
            return {"recommended": {}}
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {"recommended": {}}
        if not isinstance(data, dict):
            return {"recommended": {}}
        recommended = data.get("recommended")
        if not isinstance(recommended, dict):
            data["recommended"] = {}
        return data

    def filter_new(self, papers: list[Paper]) -> list[Paper]:
        recommended = self._data.get("recommended", {})
        return [paper for paper in papers if paper_history_key(paper) not in recommended]

    def mark_recommended(self, papers: list[Paper]) -> None:
        if self.path is None or not papers:
            return
        recommended = self._data.setdefault("recommended", {})
        now = datetime.now(timezone.utc).isoformat()
        for paper in papers:
            recommended[paper_history_key(paper)] = {
                "title": paper.title,
                "source": paper.source,
                "url": paper.url,
                "recommended_at": now,
            }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True))

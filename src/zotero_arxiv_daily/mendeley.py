from __future__ import annotations

from datetime import datetime
from typing import Any

import requests

from .protocol import CorpusPaper


MENDELEY_API_BASE_URL = "https://api.mendeley.com"
MENDELEY_TOKEN_URL = f"{MENDELEY_API_BASE_URL}/oauth/token"


def parse_mendeley_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min
    normalized = value.removesuffix("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return datetime.min


class MendeleyClient:
    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        access_token: str | None = None,
        redirect_uri: str | None = None,
        use_starred_only: bool = False,
        api_base_url: str = MENDELEY_API_BASE_URL,
        token_url: str = MENDELEY_TOKEN_URL,
        session: requests.Session | None = None,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.access_token = access_token
        self.redirect_uri = redirect_uri
        self.use_starred_only = use_starred_only
        self.api_base_url = api_base_url.rstrip("/")
        self.token_url = token_url
        self.session = session or requests.Session()

    @classmethod
    def from_config(cls, config: Any) -> "MendeleyClient":
        return cls(
            client_id=config.client_id,
            client_secret=config.client_secret,
            refresh_token=config.get("refresh_token"),
            access_token=config.get("access_token"),
            redirect_uri=config.get("redirect_uri"),
            use_starred_only=config.get("use_starred_only", False),
            api_base_url=config.get("api_base_url", MENDELEY_API_BASE_URL),
            token_url=config.get("token_url", MENDELEY_TOKEN_URL),
        )

    def _ensure_access_token(self) -> str:
        if self.access_token:
            return self.access_token
        if not self.refresh_token:
            raise ValueError("config.mendeley.access_token or config.mendeley.refresh_token is required.")
        if not self.client_id or not self.client_secret:
            raise ValueError("config.mendeley.client_id and config.mendeley.client_secret are required to refresh tokens.")

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
        }
        if self.redirect_uri:
            data["redirect_uri"] = self.redirect_uri
        response = self.session.post(
            self.token_url,
            data=data,
            auth=(self.client_id, self.client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        response.raise_for_status()
        token_payload = response.json()
        self.access_token = token_payload["access_token"]
        self.refresh_token = token_payload.get("refresh_token") or self.refresh_token
        return self.access_token

    def _get(self, path_or_url: str, *, params: dict[str, Any] | None = None, accept: str) -> requests.Response:
        url = path_or_url if path_or_url.startswith("http") else f"{self.api_base_url}{path_or_url}"
        response = self.session.get(
            url,
            params=params,
            headers={
                "Authorization": f"Bearer {self._ensure_access_token()}",
                "Accept": accept,
            },
            timeout=30,
        )
        response.raise_for_status()
        return response

    def _post(self, path: str, *, data: dict[str, Any], content_type: str, accept: str | None = None) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._ensure_access_token()}",
            "Content-Type": content_type,
        }
        if accept:
            headers["Accept"] = accept
        response = self.session.post(
            f"{self.api_base_url}{path}",
            json=data,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response

    def _patch(self, path: str, *, data: dict[str, Any], content_type: str, accept: str | None = None) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self._ensure_access_token()}",
            "Content-Type": content_type,
        }
        if accept:
            headers["Accept"] = accept
        response = self.session.patch(
            f"{self.api_base_url}{path}",
            json=data,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response

    def _delete(self, path: str) -> requests.Response:
        response = self.session.delete(
            f"{self.api_base_url}{path}",
            headers={"Authorization": f"Bearer {self._ensure_access_token()}"},
            timeout=30,
        )
        response.raise_for_status()
        return response

    def _get_all_pages(self, path: str, *, params: dict[str, Any] | None = None, accept: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        next_url: str | None = path
        next_params = params
        while next_url:
            response = self._get(next_url, params=next_params, accept=accept)
            payload = response.json()
            if isinstance(payload, list):
                items.extend(payload)
            else:
                items.append(payload)
            next_url = response.links.get("next", {}).get("url")
            next_params = None
        return items

    def fetch_documents(self) -> list[dict[str, Any]]:
        return self._get_all_pages(
            "/documents",
            params={"limit": 500, "view": "all"},
            accept="application/vnd.mendeley-document.1+json",
        )

    def fetch_folders(self) -> list[dict[str, Any]]:
        return self._get_all_pages(
            "/folders",
            params={"limit": 500},
            accept="application/vnd.mendeley-folder.1+json",
        )

    def fetch_folder_document_ids(self, folder_id: str) -> list[str]:
        items = self._get_all_pages(
            f"/folders/{folder_id}/documents",
            params={"limit": 500},
            accept="application/vnd.mendeley-document.1+json",
        )
        return [item["id"] for item in items if item.get("id")]

    def create_folder(self, name: str, *, parent_id: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {"name": name}
        if parent_id:
            data["parent_id"] = parent_id
        response = self._post(
            "/folders",
            data=data,
            content_type="application/vnd.mendeley-folder.1+json",
            accept="application/vnd.mendeley-folder.1+json",
        )
        return response.json()

    def update_folder(self, folder_id: str, *, name: str | None = None, parent_id: str | None = None) -> None:
        data: dict[str, Any] = {}
        if name is not None:
            data["name"] = name
        if parent_id is not None:
            data["parent_id"] = parent_id
        self._patch(
            f"/folders/{folder_id}",
            data=data,
            content_type="application/vnd.mendeley-folder.1+json",
        )

    def delete_folder(self, folder_id: str) -> None:
        self._delete(f"/folders/{folder_id}")

    def add_document_to_folder(self, folder_id: str, document_id: str) -> None:
        self._post(
            f"/folders/{folder_id}/documents",
            data={"id": document_id},
            content_type="application/vnd.mendeley-document.1+json",
        )

    def remove_document_from_folder(self, folder_id: str, document_id: str) -> None:
        self._delete(f"/folders/{folder_id}/documents/{document_id}")

    def update_document_tags(self, document_id: str, tags: list[str]) -> dict[str, Any]:
        response = self._patch(
            f"/documents/{document_id}",
            data={"tags": tags},
            content_type="application/vnd.mendeley-document.1+json",
            accept="application/vnd.mendeley-document.1+json",
        )
        return response.json()

    def fetch_corpus(self) -> list[CorpusPaper]:
        folders = self.fetch_folders()
        folder_paths = build_folder_paths(folders)
        document_paths: dict[str, list[str]] = {}
        for folder_id, folder_path in folder_paths.items():
            for document_id in self.fetch_folder_document_ids(folder_id):
                document_paths.setdefault(document_id, []).append(folder_path)

        corpus: list[CorpusPaper] = []
        for document in self.fetch_documents():
            if self.use_starred_only and not document.get("starred"):
                continue
            abstract = document.get("abstract") or ""
            title = document.get("title") or ""
            if not title or not abstract:
                continue
            corpus.append(
                CorpusPaper(
                    title=title,
                    abstract=abstract,
                    added_date=parse_mendeley_datetime(document.get("created") or document.get("last_modified")),
                    paths=document_paths.get(document.get("id"), []),
                )
            )
        return corpus


def build_folder_paths(folders: list[dict[str, Any]]) -> dict[str, str]:
    folders_by_id = {folder["id"]: folder for folder in folders if folder.get("id")}

    def path_for(folder_id: str, seen: set[str] | None = None) -> str:
        seen = seen or set()
        if folder_id in seen:
            return folders_by_id[folder_id].get("name", folder_id)
        seen.add(folder_id)
        folder = folders_by_id[folder_id]
        name = folder.get("name") or folder_id
        parent_id = folder.get("parent_id")
        if parent_id and parent_id in folders_by_id:
            return f"{path_for(parent_id, seen)}/{name}"
        return name

    return {folder_id: path_for(folder_id) for folder_id in folders_by_id}

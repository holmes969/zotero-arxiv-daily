from zotero_arxiv_daily.mendeley import MendeleyClient, build_folder_paths, parse_mendeley_datetime


class StubResponse:
    def __init__(self, payload, links=None):
        self.payload = payload
        self.links = links or {}

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass


class StubSession:
    def __init__(self):
        self.posts = []
        self.gets = []
        self.patches = []
        self.deletes = []
        self.get_responses = []

    def post(self, *args, **kwargs):
        self.posts.append((args, kwargs))
        return StubResponse(
            {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "token_type": "bearer",
                "expires_in": 3600,
            }
        )

    def get(self, *args, **kwargs):
        self.gets.append((args, kwargs))
        return self.get_responses.pop(0)

    def patch(self, *args, **kwargs):
        self.patches.append((args, kwargs))
        return StubResponse({"id": "D1", "tags": kwargs.get("json", {}).get("tags", [])})

    def delete(self, *args, **kwargs):
        self.deletes.append((args, kwargs))
        return StubResponse({})


def test_build_folder_paths_reconstructs_nested_paths():
    folders = [
        {"id": "root", "name": "2026"},
        {"id": "child", "name": "survey", "parent_id": "root"},
        {"id": "grandchild", "name": "topic-a", "parent_id": "child"},
    ]

    assert build_folder_paths(folders)["grandchild"] == "2026/survey/topic-a"


def test_parse_mendeley_datetime_accepts_milliseconds_and_seconds():
    assert parse_mendeley_datetime("2026-01-01T10:20:30.000Z").year == 2026
    assert parse_mendeley_datetime("2026-01-01T10:20:30Z").second == 30


def test_client_refreshes_access_token_when_only_refresh_token_is_configured():
    session = StubSession()
    session.get_responses = [StubResponse([])]
    client = MendeleyClient(
        client_id="client-id",
        client_secret="client-secret",
        refresh_token="refresh-token",
        session=session,
    )

    client.fetch_documents()

    assert session.posts[0][1]["data"]["grant_type"] == "refresh_token"
    assert session.gets[0][1]["headers"]["Authorization"] == "Bearer new-access-token"
    assert client.refresh_token == "new-refresh-token"


def test_get_all_pages_follows_next_link():
    session = StubSession()
    session.get_responses = [
        StubResponse([{"id": "one"}], links={"next": {"url": "https://api.mendeley.com/documents?marker=next"}}),
        StubResponse([{"id": "two"}]),
    ]
    client = MendeleyClient(access_token="access-token", session=session)

    assert client.fetch_documents() == [{"id": "one"}, {"id": "two"}]
    assert session.gets[1][0][0] == "https://api.mendeley.com/documents?marker=next"


def test_fetch_corpus_maps_documents_and_folder_membership():
    client = MendeleyClient(access_token="access-token")
    client.fetch_folders = lambda: [
        {"id": "F1", "name": "survey"},
        {"id": "F2", "name": "topic-a", "parent_id": "F1"},
    ]
    client.fetch_folder_document_ids = lambda folder_id: {"F1": [], "F2": ["D1"]}[folder_id]
    client.fetch_documents = lambda: [
        {
            "id": "D1",
            "title": "Mapped Paper",
            "abstract": "Abstract.",
            "created": "2026-01-01T00:00:00.000Z",
        },
        {
            "id": "D2",
            "title": "No Abstract",
            "abstract": "",
            "created": "2026-01-02T00:00:00.000Z",
        },
    ]

    corpus = client.fetch_corpus()

    assert len(corpus) == 1
    assert corpus[0].title == "Mapped Paper"
    assert corpus[0].paths == ["survey/topic-a"]


def test_fetch_corpus_can_use_only_starred_documents():
    client = MendeleyClient(access_token="access-token", use_starred_only=True)
    client.fetch_folders = lambda: []
    client.fetch_folder_document_ids = lambda folder_id: []
    client.fetch_documents = lambda: [
        {
            "id": "D1",
            "title": "Starred Paper",
            "abstract": "Abstract.",
            "created": "2026-01-01T00:00:00.000Z",
            "starred": True,
        },
        {
            "id": "D2",
            "title": "Unstarred Paper",
            "abstract": "Abstract.",
            "created": "2026-01-02T00:00:00.000Z",
            "starred": False,
        },
    ]

    corpus = client.fetch_corpus()

    assert [paper.title for paper in corpus] == ["Starred Paper"]


def test_fetch_corpus_excludes_starred_documents_without_abstract():
    client = MendeleyClient(access_token="access-token", use_starred_only=True)
    client.fetch_folders = lambda: []
    client.fetch_folder_document_ids = lambda folder_id: []
    client.fetch_documents = lambda: [
        {
            "id": "D1",
            "title": "Starred Without Abstract",
            "abstract": "",
            "created": "2026-01-01T00:00:00.000Z",
            "starred": True,
        },
        {
            "id": "D2",
            "title": "Starred With Abstract",
            "abstract": "Abstract.",
            "created": "2026-01-02T00:00:00.000Z",
            "starred": True,
        },
    ]

    corpus = client.fetch_corpus()

    assert [paper.title for paper in corpus] == ["Starred With Abstract"]


def test_create_folder_posts_folder_payload():
    session = StubSession()
    client = MendeleyClient(access_token="access-token", session=session)

    client.create_folder("Child", parent_id="parent")

    assert session.posts[0][0][0] == "https://api.mendeley.com/folders"
    assert session.posts[0][1]["json"] == {"name": "Child", "parent_id": "parent"}
    assert session.posts[0][1]["headers"]["Content-Type"] == "application/vnd.mendeley-folder.1+json"


def test_folder_membership_write_methods_use_expected_endpoints():
    session = StubSession()
    client = MendeleyClient(access_token="access-token", session=session)

    client.add_document_to_folder("F1", "D1")
    client.remove_document_from_folder("F1", "D1")

    assert session.posts[0][0][0] == "https://api.mendeley.com/folders/F1/documents"
    assert session.posts[0][1]["json"] == {"id": "D1"}
    assert session.deletes[0][0][0] == "https://api.mendeley.com/folders/F1/documents/D1"


def test_update_document_tags_patches_tags():
    session = StubSession()
    client = MendeleyClient(access_token="access-token", session=session)

    client.update_document_tags("D1", ["single-view"])

    assert session.patches[0][0][0] == "https://api.mendeley.com/documents/D1"
    assert session.patches[0][1]["json"] == {"tags": ["single-view"]}
    assert session.patches[0][1]["headers"]["Content-Type"] == "application/vnd.mendeley-document.1+json"

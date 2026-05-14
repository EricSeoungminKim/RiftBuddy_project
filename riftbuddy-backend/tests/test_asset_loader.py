from pathlib import Path
from unittest.mock import patch

import pytest

from modules.asset_loader import AssetLoader


class FakeResponse:
    def __init__(self, *, json_data: object | None = None, body: bytes = b"") -> None:
        self._json_data = json_data
        self._body = body

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def json(self) -> object:
        return self._json_data

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self.urls: list[str] = []

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def get(self, url: str) -> FakeResponse:
        self.urls.append(url)
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_fetch_latest_version(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(json_data=["14.14.1", "14.13.1"])])
    loader = AssetLoader(cache_dir=tmp_path)

    with patch("modules.asset_loader.aiohttp.ClientSession", return_value=session):
        version = await loader.fetch_latest_version()

    assert version == "14.14.1"
    assert session.urls == ["https://ddragon.leagueoflegends.com/api/versions.json"]


@pytest.mark.asyncio
async def test_fetch_champion_list(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(
                json_data={
                    "data": {
                        "Zed": {"id": "Zed", "name": "Zed"},
                        "Jinx": {"id": "Jinx", "name": "Jinx"},
                    }
                }
            )
        ]
    )
    loader = AssetLoader(cache_dir=tmp_path)

    with patch("modules.asset_loader.aiohttp.ClientSession", return_value=session):
        champions = await loader.fetch_champion_list("14.14.1")

    assert "Zed" in champions
    assert "Jinx" in champions


def test_icon_cache_path_includes_version_and_champion(tmp_path: Path) -> None:
    loader = AssetLoader(cache_dir=tmp_path)

    path = loader.icon_cache_path("14.14.1", "Zed")

    assert path == tmp_path / "champion" / "14.14.1" / "Zed.png"


@pytest.mark.asyncio
async def test_download_champion_icon_writes_cache_file(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(body=b"icon-bytes")])
    loader = AssetLoader(cache_dir=tmp_path)

    with patch("modules.asset_loader.aiohttp.ClientSession", return_value=session):
        path = await loader.download_champion_icon("14.14.1", "Zed")

    assert path.read_bytes() == b"icon-bytes"
    assert path == tmp_path / "champion" / "14.14.1" / "Zed.png"


@pytest.mark.asyncio
async def test_load_match_assets_downloads_each_champion(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(json_data=["14.14.1"]),
            FakeResponse(
                json_data={
                    "data": {
                        "Zed": {"id": "Zed", "name": "Zed"},
                        "Jinx": {"id": "Jinx", "name": "Jinx"},
                    }
                }
            ),
            FakeResponse(body=b"zed"),
            FakeResponse(body=b"jinx"),
        ]
    )
    loader = AssetLoader(cache_dir=tmp_path)

    with patch("modules.asset_loader.aiohttp.ClientSession", return_value=session):
        paths = await loader.load_match_assets(["Zed", "Jinx"])

    assert paths["Zed"].read_bytes() == b"zed"
    assert paths["Jinx"].read_bytes() == b"jinx"


@pytest.mark.asyncio
async def test_load_match_assets_resolves_live_client_display_names(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        [
            FakeResponse(json_data=["14.14.1"]),
            FakeResponse(
                json_data={
                    "data": {
                        "LeeSin": {"id": "LeeSin", "name": "Lee Sin"},
                        "MonkeyKing": {"id": "MonkeyKing", "name": "Wukong"},
                    }
                }
            ),
            FakeResponse(body=b"lee"),
            FakeResponse(body=b"wukong"),
        ]
    )
    loader = AssetLoader(cache_dir=tmp_path)

    with patch("modules.asset_loader.aiohttp.ClientSession", return_value=session):
        paths = await loader.load_match_assets(["Lee Sin", "Wukong"])

    assert paths["Lee Sin"].read_bytes() == b"lee"
    assert paths["Wukong"].read_bytes() == b"wukong"
    assert session.urls[-2:] == [
        "https://ddragon.leagueoflegends.com/cdn/14.14.1/img/champion/LeeSin.png",
        "https://ddragon.leagueoflegends.com/cdn/14.14.1/img/champion/MonkeyKing.png",
    ]


@pytest.mark.asyncio
async def test_load_match_assets_resolves_korean_live_client_names(
    tmp_path: Path,
) -> None:
    session = FakeSession(
        [
            FakeResponse(json_data=["14.14.1"]),
            FakeResponse(
                json_data={
                    "data": {
                        "Jhin": {"id": "Jhin", "name": "Jhin"},
                        "Graves": {"id": "Graves", "name": "Graves"},
                    }
                }
            ),
            FakeResponse(
                json_data={
                    "data": {
                        "Jhin": {"id": "Jhin", "name": "진"},
                        "Graves": {"id": "Graves", "name": "그레이브즈"},
                    }
                }
            ),
            FakeResponse(body=b"jhin"),
            FakeResponse(body=b"graves"),
        ]
    )
    loader = AssetLoader(cache_dir=tmp_path)

    with patch("modules.asset_loader.aiohttp.ClientSession", return_value=session):
        paths = await loader.load_match_assets(["진", "그레이브즈"])

    assert paths["진"].read_bytes() == b"jhin"
    assert paths["그레이브즈"].read_bytes() == b"graves"
    assert session.urls[-2:] == [
        "https://ddragon.leagueoflegends.com/cdn/14.14.1/img/champion/Jhin.png",
        "https://ddragon.leagueoflegends.com/cdn/14.14.1/img/champion/Graves.png",
    ]

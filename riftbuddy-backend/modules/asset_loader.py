import asyncio
from pathlib import Path
from typing import Any, cast

import aiohttp

DDRAGON_BASE_URL = "https://ddragon.leagueoflegends.com"
VERSIONS_URL = f"{DDRAGON_BASE_URL}/api/versions.json"

SUMMONER_SPELLS: dict[str, int] = {
    "Flash": 300,
    "Ignite": 180,
    "Exhaust": 210,
    "Heal": 240,
    "Ghost": 210,
    "Barrier": 180,
    "Cleanse": 210,
    "Teleport": 360,
    "Smite": 90,
}


class AssetLoader:
    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_latest_version(self) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.get(VERSIONS_URL) as response:
                response.raise_for_status()
                versions = cast(list[str], await response.json())

        if not versions:
            raise ValueError("DDragon version list is empty")
        return versions[0]

    async def fetch_champion_list(self, version: str) -> dict[str, Any]:
        url = f"{DDRAGON_BASE_URL}/cdn/{version}/data/en_US/champion.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = cast(dict[str, Any], await response.json())

        return cast(dict[str, Any], data["data"])

    def icon_cache_path(self, version: str, champion_id: str) -> Path:
        return self._cache_dir / "champion" / version / f"{champion_id}.png"

    async def download_champion_icon(self, version: str, champion_id: str) -> Path:
        path = self.icon_cache_path(version, champion_id)
        if path.exists():
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{DDRAGON_BASE_URL}/cdn/{version}/img/champion/{champion_id}.png"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                path.write_bytes(await response.read())

        return path

    async def load_match_assets(self, champion_ids: list[str]) -> dict[str, Path]:
        version = await self.fetch_latest_version()
        tasks = [
            self.download_champion_icon(version, champion_id)
            for champion_id in champion_ids
        ]
        paths = await asyncio.gather(*tasks)
        return dict(zip(champion_ids, paths, strict=True))

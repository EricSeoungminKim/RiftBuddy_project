import asyncio
from pathlib import Path
from typing import Any, cast

import aiohttp

DDRAGON_BASE_URL = "https://ddragon.leagueoflegends.com"
VERSIONS_URL = f"{DDRAGON_BASE_URL}/api/versions.json"
DEFAULT_CHAMPION_LOCALE = "en_US"
FALLBACK_CHAMPION_LOCALES = ("ko_KR",)

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
    """Downloads and caches Riot Data Dragon assets used by the overlay."""

    def __init__(self, cache_dir: str | Path = "data/cache") -> None:
        """Create the local cache directory if it does not exist yet."""
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_latest_version(self) -> str:
        """Fetch the newest available Data Dragon version string."""
        async with aiohttp.ClientSession() as session:
            async with session.get(VERSIONS_URL) as response:
                response.raise_for_status()
                versions = cast(list[str], await response.json())

        if not versions:
            raise ValueError("DDragon version list is empty")
        return versions[0]

    async def fetch_champion_list(
        self,
        version: str,
        locale: str = DEFAULT_CHAMPION_LOCALE,
    ) -> dict[str, Any]:
        """Fetch champion metadata for a specific Data Dragon version."""
        url = f"{DDRAGON_BASE_URL}/cdn/{version}/data/{locale}/champion.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = cast(dict[str, Any], await response.json())

        return cast(dict[str, Any], data["data"])

    def icon_cache_path(self, version: str, champion_id: str) -> Path:
        """Build the local cache path for one champion icon."""
        return self._cache_dir / "champion" / version / f"{champion_id}.png"

    async def download_champion_icon(self, version: str, champion_id: str) -> Path:
        """Download one champion icon unless it already exists in cache."""
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

    async def load_match_assets(self, champion_names: list[str]) -> dict[str, Path]:
        """Load champion icons for the current match, keyed by display name."""
        version = await self.fetch_latest_version()
        champions = await self.fetch_champion_list(version)
        if _missing_champion_names(champion_names, champions):
            champions = await self._load_fallback_champion_aliases(version, champions)

        unresolved = _missing_champion_names(champion_names, champions)
        if unresolved:
            raise ValueError(
                "Could not resolve Data Dragon champion ids for: "
                f"{', '.join(unresolved)}"
            )

        champion_ids = _resolve_champion_ids(champion_names, champions)
        tasks = [
            self.download_champion_icon(version, champion_id)
            for champion_id in champion_ids
        ]
        paths = await asyncio.gather(*tasks)
        return dict(zip(champion_names, paths, strict=True))

    async def _load_fallback_champion_aliases(
        self,
        version: str,
        champions: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge localized Data Dragon champion names into the alias map."""
        merged = dict(champions)
        for locale in FALLBACK_CHAMPION_LOCALES:
            merged.update(await self.fetch_champion_list(version, locale=locale))
        return merged


def _resolve_champion_ids(
    champion_names: list[str],
    champions: dict[str, Any],
) -> list[str]:
    """Convert Live Client champion display names to Data Dragon icon ids."""
    ids_by_lookup = _champion_ids_by_lookup(champions)
    return [ids_by_lookup.get(_lookup_key(name), name) for name in champion_names]


def _missing_champion_names(
    champion_names: list[str],
    champions: dict[str, Any],
) -> list[str]:
    """Return champion names that could not be converted into a DDragon id."""
    ids_by_lookup = _champion_ids_by_lookup(champions)
    return [name for name in champion_names if _lookup_key(name) not in ids_by_lookup]


def _champion_ids_by_lookup(champions: dict[str, Any]) -> dict[str, str]:
    """Build lookup aliases from Data Dragon champion id and display name."""
    ids_by_lookup: dict[str, str] = {}
    for champion_id, champion_data in champions.items():
        if not isinstance(champion_data, dict):
            continue

        ddragon_id = champion_data.get("id", champion_id)
        display_name = champion_data.get("name")
        if isinstance(ddragon_id, str):
            ids_by_lookup[_lookup_key(champion_id)] = ddragon_id
            ids_by_lookup[_lookup_key(ddragon_id)] = ddragon_id
        if isinstance(display_name, str):
            ids_by_lookup[_lookup_key(display_name)] = ddragon_id

    return ids_by_lookup


def _lookup_key(value: str) -> str:
    """Normalize champion names so 'Lee Sin' matches 'LeeSin'."""
    return "".join(character for character in value.lower() if character.isalnum())

from typing import Any, cast

import aiohttp

ROLE_ORDER = {
    "TOP": 0,
    "JUNGLE": 1,
    "MIDDLE": 2,
    "BOTTOM": 3,
    "UTILITY": 4,
}
SUMMONER_SPELL_ALIASES = {
    "점멸": "Flash",
    "점화": "Ignite",
    "탈진": "Exhaust",
    "회복": "Heal",
    "유체화": "Ghost",
    "방어막": "Barrier",
    "정화": "Cleanse",
    "순간이동": "Teleport",
    "강타": "Smite",
}


class LiveClientError(RuntimeError):
    """Raised when the LoL Live Client Data API is unavailable or incomplete."""


class LiveClient:
    """Reads current in-game player data from Riot's local Live Client Data API."""

    def __init__(self, all_game_data_url: str) -> None:
        """Store the local allgamedata endpoint URL."""
        self._all_game_data_url = all_game_data_url

    async def fetch_enemy_loadout(self) -> list[dict[str, object]]:
        """Fetch the active game and return enemy champions with summoner spells."""
        data = await self._fetch_all_game_data()
        return extract_enemy_loadout(data)

    async def _fetch_all_game_data(self) -> dict[str, Any]:
        """Fetch all game data from LoL's self-signed local HTTPS endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self._all_game_data_url, ssl=False) as response:
                    response.raise_for_status()
                    return cast(dict[str, Any], await response.json())
        except aiohttp.ClientError as error:
            raise LiveClientError(
                "LoL Live Client Data API is unavailable. Start an active game "
                "or set TESTMODE=True in settings.py."
            ) from error


def extract_enemy_loadout(all_game_data: dict[str, Any]) -> list[dict[str, object]]:
    """Extract enemy champion names and summoner spells from allgamedata."""
    active_name = _active_summoner_name(all_game_data)
    players = cast(list[dict[str, Any]], all_game_data.get("allPlayers", []))
    active_player = _find_player_by_name(players, active_name)
    active_team = active_player.get("team")
    if not active_team:
        raise LiveClientError("Active player team is missing from Live Client data.")

    enemies = [
        player
        for player in players
        if player.get("team") and player.get("team") != active_team
    ]
    return [
        _player_to_loadout(player)
        for player in sorted(enemies, key=_player_role_sort_key)
    ]


def _active_summoner_name(all_game_data: dict[str, Any]) -> str:
    """Read the active player's summoner name from allgamedata."""
    active_player = cast(dict[str, Any], all_game_data.get("activePlayer", {}))
    name = active_player.get("summonerName")
    if not isinstance(name, str) or not name:
        raise LiveClientError("Active player summonerName is missing.")
    return name


def _find_player_by_name(
    players: list[dict[str, Any]],
    summoner_name: str,
) -> dict[str, Any]:
    """Find the allPlayers entry matching the active summoner name."""
    for player in players:
        if player.get("summonerName") == summoner_name:
            return player
    raise LiveClientError("Active player was not found in allPlayers.")


def _player_to_loadout(player: dict[str, Any]) -> dict[str, object]:
    """Convert one Live Client player record into frontend loadout data."""
    champion = player.get("championName")
    if not isinstance(champion, str) or not champion:
        raise LiveClientError("Enemy championName is missing.")

    summoner_spells = cast(dict[str, Any], player.get("summonerSpells", {}))
    spells = [
        _spell_display_name(summoner_spells, "summonerSpellOne"),
        _spell_display_name(summoner_spells, "summonerSpellTwo"),
    ]
    loadout: dict[str, object] = {"champion": champion, "spells": spells}
    role = _player_role(player)
    if role is not None:
        loadout["role"] = role
    return loadout


def _player_role_sort_key(player: dict[str, Any]) -> int:
    """Sort players by Riot lane position when Live Client provides it."""
    role = _player_role(player)
    if role is None:
        return len(ROLE_ORDER)
    return ROLE_ORDER.get(role, len(ROLE_ORDER))


def _player_role(player: dict[str, Any]) -> str | None:
    """Read the Live Client lane role value when present."""
    role = player.get("position")
    if not isinstance(role, str) or not role:
        return None
    return role.upper()


def _spell_display_name(summoner_spells: dict[str, Any], key: str) -> str:
    """Read one spell display name from a Live Client summonerSpells block."""
    spell = cast(dict[str, Any], summoner_spells.get(key, {}))
    display_name = spell.get("displayName")
    if not isinstance(display_name, str) or not display_name:
        raise LiveClientError(f"{key} displayName is missing.")
    return SUMMONER_SPELL_ALIASES.get(display_name, display_name)

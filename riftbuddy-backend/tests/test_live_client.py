import pytest

from modules.live_client import LiveClientError, extract_enemy_loadout


def test_extract_enemy_loadout_returns_opposing_team() -> None:
    data = {
        "activePlayer": {"summonerName": "Me"},
        "allPlayers": [
            {
                "summonerName": "Me",
                "team": "ORDER",
                "championName": "Ahri",
                "summonerSpells": {
                    "summonerSpellOne": {"displayName": "Flash"},
                    "summonerSpellTwo": {"displayName": "Ignite"},
                },
            },
            {
                "summonerName": "EnemyTop",
                "team": "CHAOS",
                "championName": "Garen",
                "summonerSpells": {
                    "summonerSpellOne": {"displayName": "Flash"},
                    "summonerSpellTwo": {"displayName": "Teleport"},
                },
            },
            {
                "summonerName": "EnemyAdc",
                "team": "CHAOS",
                "championName": "Jinx",
                "summonerSpells": {
                    "summonerSpellOne": {"displayName": "Flash"},
                    "summonerSpellTwo": {"displayName": "Heal"},
                },
            },
            {
                "summonerName": "Spectatorish",
                "championName": "Unknown",
                "summonerSpells": {},
            },
        ],
    }

    assert extract_enemy_loadout(data) == [
        {"champion": "Garen", "spells": ["Flash", "Teleport"]},
        {"champion": "Jinx", "spells": ["Flash", "Heal"]},
    ]


def test_extract_enemy_loadout_sorts_by_lane_role() -> None:
    data = {
        "activePlayer": {"summonerName": "Me"},
        "allPlayers": [
            {
                "summonerName": "Me",
                "team": "ORDER",
                "championName": "Ahri",
                "position": "MIDDLE",
                "summonerSpells": {
                    "summonerSpellOne": {"displayName": "Flash"},
                    "summonerSpellTwo": {"displayName": "Ignite"},
                },
            },
            _enemy("EnemySupport", "CHAOS", "Milio", "UTILITY", "Flash", "Heal"),
            _enemy("EnemyJungle", "CHAOS", "Lee Sin", "JUNGLE", "Flash", "Smite"),
            _enemy("EnemyTop", "CHAOS", "Garen", "TOP", "Flash", "Teleport"),
            _enemy("EnemyAdc", "CHAOS", "Jinx", "BOTTOM", "Flash", "Heal"),
            _enemy("EnemyMid", "CHAOS", "Zed", "MIDDLE", "Flash", "Ignite"),
        ],
    }

    assert extract_enemy_loadout(data) == [
        {"champion": "Garen", "role": "TOP", "spells": ["Flash", "Teleport"]},
        {"champion": "Lee Sin", "role": "JUNGLE", "spells": ["Flash", "Smite"]},
        {"champion": "Zed", "role": "MIDDLE", "spells": ["Flash", "Ignite"]},
        {"champion": "Jinx", "role": "BOTTOM", "spells": ["Flash", "Heal"]},
        {"champion": "Milio", "role": "UTILITY", "spells": ["Flash", "Heal"]},
    ]


def test_extract_enemy_loadout_normalizes_korean_summoner_spell_names() -> None:
    data = {
        "activePlayer": {"summonerName": "Me"},
        "allPlayers": [
            _enemy("Me", "ORDER", "아리", "MIDDLE", "점멸", "점화"),
            _enemy("EnemyTop", "CHAOS", "사이온", "TOP", "점멸", "순간이동"),
            _enemy("EnemyJungle", "CHAOS", "그레이브즈", "JUNGLE", "점멸", "강타"),
        ],
    }

    assert extract_enemy_loadout(data) == [
        {"champion": "사이온", "role": "TOP", "spells": ["Flash", "Teleport"]},
        {"champion": "그레이브즈", "role": "JUNGLE", "spells": ["Flash", "Smite"]},
    ]


def test_extract_enemy_loadout_requires_active_player() -> None:
    with pytest.raises(LiveClientError, match="summonerName"):
        extract_enemy_loadout({"activePlayer": {}, "allPlayers": []})


def _enemy(
    name: str,
    team: str,
    champion: str,
    position: str,
    spell_one: str,
    spell_two: str,
) -> dict[str, object]:
    return {
        "summonerName": name,
        "team": team,
        "championName": champion,
        "position": position,
        "summonerSpells": {
            "summonerSpellOne": {"displayName": spell_one},
            "summonerSpellTwo": {"displayName": spell_two},
        },
    }

from copy import deepcopy
from typing import Any


class GameState:
    """Single source of truth for minimap MIA and summoner spell cooldown state."""

    def __init__(self) -> None:
        """Initialize empty MIA and spell cooldown stores."""
        self._mia: dict[str, dict[str, float]] = {}
        self._spells: dict[str, dict[str, dict[str, float]]] = {}
        self._enemy_loadout: list[dict[str, object]] = []
        self._minimap_debug: dict[str, object] = {}
        self._game_status: dict[str, str] = {"phase": "test_mode", "message": ""}

    def update_mia(self, champion: str, last_seen: float) -> None:
        """Mark a champion as missing and preserve when they were last seen."""
        self._mia[champion] = {"last_seen": last_seen}

    def clear_mia(self, champion: str) -> None:
        """Remove a champion from the missing list after they are seen again."""
        self._mia.pop(champion, None)

    def get_mia(self) -> dict[str, dict[str, float]]:
        """Return a copy of current missing champion state."""
        return deepcopy(self._mia)

    def update_spell(self, champion: str, spell: str, available_at: float) -> None:
        """Record when a champion's spell cooldown will become available."""
        self._spells.setdefault(champion, {})
        self._spells[champion][spell] = {"available_at": available_at}

    def clear_spell(self, champion: str, spell: str) -> None:
        """Remove a tracked spell cooldown if it no longer needs display."""
        if champion not in self._spells:
            return
        self._spells[champion].pop(spell, None)

    def get_spells(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return a copy of current spell cooldown state."""
        return deepcopy(self._spells)

    def reset_match_state(self) -> None:
        """Clear state that belongs to one active League match."""
        self._mia.clear()
        self._spells.clear()
        self._enemy_loadout = []

    def update_enemy_loadout(self, enemy_loadout: list[dict[str, object]]) -> None:
        """Store enemy champions and their summoner spell names."""
        self._enemy_loadout = deepcopy(enemy_loadout)

    def get_enemy_loadout(self) -> list[dict[str, object]]:
        """Return a copy of enemy champion and spell display data."""
        return deepcopy(self._enemy_loadout)

    def update_minimap_debug(self, minimap_debug: dict[str, object]) -> None:
        """Store the latest minimap debug and calibration snapshot."""
        self._minimap_debug = deepcopy(minimap_debug)

    def get_minimap_debug(self) -> dict[str, object]:
        """Return a copy of the latest minimap debug snapshot."""
        return deepcopy(self._minimap_debug)

    def update_game_status(self, phase: str, message: str = "") -> None:
        """Store whether the backend is waiting or tracking an active game."""
        self._game_status = {"phase": phase, "message": message}

    def get_game_status(self) -> dict[str, str]:
        """Return the current active game lifecycle status."""
        return deepcopy(self._game_status)

    def snapshot(self) -> dict[str, Any]:
        """Return a frontend-friendly snapshot of all tracked game state."""
        return {
            "mia": self.get_mia(),
            "spells": self.get_spells(),
            "enemy_loadout": self.get_enemy_loadout(),
            "minimap_debug": self.get_minimap_debug(),
            "game_status": self.get_game_status(),
        }

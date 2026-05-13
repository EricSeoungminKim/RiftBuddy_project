from copy import deepcopy
from typing import Any


class GameState:
    def __init__(self) -> None:
        self._mia: dict[str, dict[str, float]] = {}
        self._spells: dict[str, dict[str, dict[str, float]]] = {}

    def update_mia(self, champion: str, last_seen: float) -> None:
        self._mia[champion] = {"last_seen": last_seen}

    def clear_mia(self, champion: str) -> None:
        self._mia.pop(champion, None)

    def get_mia(self) -> dict[str, dict[str, float]]:
        return deepcopy(self._mia)

    def update_spell(self, champion: str, spell: str, available_at: float) -> None:
        self._spells.setdefault(champion, {})
        self._spells[champion][spell] = {"available_at": available_at}

    def clear_spell(self, champion: str, spell: str) -> None:
        if champion not in self._spells:
            return
        self._spells[champion].pop(spell, None)

    def get_spells(self) -> dict[str, dict[str, dict[str, float]]]:
        return deepcopy(self._spells)

    def snapshot(self) -> dict[str, Any]:
        return {
            "mia": self.get_mia(),
            "spells": self.get_spells(),
        }

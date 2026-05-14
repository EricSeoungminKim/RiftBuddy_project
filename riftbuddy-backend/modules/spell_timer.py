from core.event_bus import EventBus
from core.game_state import GameState

BASE_COOLDOWNS: dict[str, int] = {
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


class SpellTimer:
    """Tracks manually triggered enemy summoner spell cooldowns."""

    def __init__(self, event_bus: EventBus, game_state: GameState) -> None:
        """Connect spell timing to the event bus and shared game state."""
        self._bus = event_bus
        self._state = game_state

    def trigger(
        self,
        champion: str,
        spell: str,
        triggered_at: float,
        ability_haste: int = 0,
    ) -> None:
        """Start a cooldown timer for a clicked spell icon."""
        if spell not in BASE_COOLDOWNS:
            raise ValueError(f"Unknown spell: {spell}")

        cooldown = BASE_COOLDOWNS[spell] / (1 + ability_haste / 100)
        available_at = triggered_at + cooldown
        self._state.update_spell(champion, spell, available_at=available_at)
        self._bus.emit(
            "spell_used",
            {
                "champion": champion,
                "spell": spell,
                "available_at": available_at,
            },
        )

    def remaining_seconds(self, champion: str, spell: str, now: float) -> float:
        """Return how many seconds remain before a spell is available."""
        spells = self._state.get_spells()
        available_at = spells.get(champion, {}).get(spell, {}).get("available_at", 0.0)
        return max(0.0, available_at - now)

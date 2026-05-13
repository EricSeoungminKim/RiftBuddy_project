import pytest

from core.event_bus import EventBus
from core.game_state import GameState
from modules.spell_timer import SpellTimer


def make_timer() -> tuple[SpellTimer, EventBus, GameState]:
    bus = EventBus()
    state = GameState()
    return SpellTimer(event_bus=bus, game_state=state), bus, state


def test_trigger_flash_sets_cooldown() -> None:
    timer, _, state = make_timer()

    timer.trigger("Jinx", "Flash", triggered_at=1000.0)

    spells = state.get_spells()
    assert spells["Jinx"]["Flash"]["available_at"] == 1300.0


def test_trigger_ignite_sets_cooldown() -> None:
    timer, _, state = make_timer()

    timer.trigger("Zed", "Ignite", triggered_at=2000.0)

    spells = state.get_spells()
    assert spells["Zed"]["Ignite"]["available_at"] == 2180.0


def test_trigger_emits_spell_used_event() -> None:
    timer, bus, _ = make_timer()
    received = []
    bus.subscribe("spell_used", received.append)

    timer.trigger("Jinx", "Flash", triggered_at=1000.0)

    assert received == [
        {
            "champion": "Jinx",
            "spell": "Flash",
            "available_at": 1300.0,
        }
    ]


def test_trigger_unknown_spell_raises() -> None:
    timer, _, _ = make_timer()

    with pytest.raises(ValueError, match="Unknown spell"):
        timer.trigger("Jinx", "UnknownSpell", triggered_at=1000.0)


def test_remaining_seconds_before_available() -> None:
    timer, _, _ = make_timer()
    timer.trigger("Jinx", "Flash", triggered_at=1000.0)

    remaining = timer.remaining_seconds("Jinx", "Flash", now=1050.0)

    assert remaining == 250.0


def test_remaining_seconds_when_available() -> None:
    timer, _, _ = make_timer()
    timer.trigger("Jinx", "Flash", triggered_at=1000.0)

    remaining = timer.remaining_seconds("Jinx", "Flash", now=1400.0)

    assert remaining == 0.0


def test_haste_reduces_cooldown() -> None:
    timer, _, state = make_timer()

    timer.trigger("Jinx", "Flash", triggered_at=1000.0, ability_haste=20)

    spells = state.get_spells()
    expected = 1000.0 + (300 / (1 + 20 / 100))
    assert abs(spells["Jinx"]["Flash"]["available_at"] - expected) < 0.01

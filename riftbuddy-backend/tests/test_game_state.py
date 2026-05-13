from core.game_state import GameState


def test_update_mia() -> None:
    state = GameState()
    state.update_mia("Zed", last_seen=1000.0)

    mia = state.get_mia()

    assert "Zed" in mia
    assert mia["Zed"]["last_seen"] == 1000.0


def test_clear_mia() -> None:
    state = GameState()
    state.update_mia("Zed", last_seen=1000.0)

    state.clear_mia("Zed")

    assert "Zed" not in state.get_mia()


def test_update_spell() -> None:
    state = GameState()
    state.update_spell("Jinx", "Flash", available_at=1300.0)

    spells = state.get_spells()

    assert spells["Jinx"]["Flash"]["available_at"] == 1300.0


def test_clear_spell() -> None:
    state = GameState()
    state.update_spell("Jinx", "Flash", available_at=1300.0)

    state.clear_spell("Jinx", "Flash")

    spells = state.get_spells()
    assert "Flash" not in spells.get("Jinx", {})


def test_snapshot() -> None:
    state = GameState()
    state.update_mia("Zed", last_seen=1000.0)
    state.update_spell("Jinx", "Flash", available_at=1300.0)

    snapshot = state.snapshot()

    assert "mia" in snapshot
    assert "spells" in snapshot
    assert snapshot["mia"]["Zed"]["last_seen"] == 1000.0
    assert snapshot["spells"]["Jinx"]["Flash"]["available_at"] == 1300.0

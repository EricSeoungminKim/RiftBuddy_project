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
    state.update_enemy_loadout([{"champion": "Jinx", "spells": ["Flash", "Heal"]}])
    state.update_game_status("active_game", "Active game detected")

    snapshot = state.snapshot()

    assert "mia" in snapshot
    assert "spells" in snapshot
    assert "enemy_loadout" in snapshot
    assert "game_status" in snapshot
    assert snapshot["mia"]["Zed"]["last_seen"] == 1000.0
    assert snapshot["spells"]["Jinx"]["Flash"]["available_at"] == 1300.0
    assert snapshot["enemy_loadout"] == [
        {"champion": "Jinx", "spells": ["Flash", "Heal"]}
    ]
    assert snapshot["game_status"] == {
        "phase": "active_game",
        "message": "Active game detected",
    }


def test_enemy_loadout_is_copied() -> None:
    state = GameState()
    loadout = [{"champion": "Zed", "spells": ["Flash", "Ignite"]}]
    state.update_enemy_loadout(loadout)

    loadout[0]["champion"] = "Changed"

    assert state.get_enemy_loadout() == [
        {"champion": "Zed", "spells": ["Flash", "Ignite"]}
    ]


def test_reset_match_state_clears_active_game_data() -> None:
    state = GameState()
    state.update_mia("Zed", last_seen=1000.0)
    state.update_spell("Jinx", "Flash", available_at=1300.0)
    state.update_enemy_loadout([{"champion": "Jinx", "spells": ["Flash", "Heal"]}])

    state.reset_match_state()

    assert state.get_mia() == {}
    assert state.get_spells() == {}
    assert state.get_enemy_loadout() == []

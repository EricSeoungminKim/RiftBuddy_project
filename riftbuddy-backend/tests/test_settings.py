from settings import (
    LIVE_CLIENT_ALL_GAME_DATA_URL,
    MINIMAP_DEBUG_FRAME_PATH,
    MINIMAP_TRAINING_CAPTURE_DIR,
    MINIMAP_TRAINING_CAPTURE_INTERVAL_SECONDS,
    MINIMAP_TRAINING_CAPTURE_MODE,
    SETTINGS,
    TEST_ENEMY_LOADOUT,
    TESTMODE,
    BackendSettings,
)


def test_backend_settings_hold_runtime_constants() -> None:
    assert SETTINGS == BackendSettings(
        test_mode=TESTMODE,
        minimap_region=SETTINGS.minimap_region,
        scan_interval_seconds=1.0,
        test_enemy_loadout=TEST_ENEMY_LOADOUT,
        live_client_all_game_data_url=LIVE_CLIENT_ALL_GAME_DATA_URL,
        minimap_debug_frame_path=MINIMAP_DEBUG_FRAME_PATH,
        minimap_training_capture_mode=MINIMAP_TRAINING_CAPTURE_MODE,
        minimap_training_capture_dir=MINIMAP_TRAINING_CAPTURE_DIR,
        minimap_training_capture_interval_seconds=(
            MINIMAP_TRAINING_CAPTURE_INTERVAL_SECONDS
        ),
    )

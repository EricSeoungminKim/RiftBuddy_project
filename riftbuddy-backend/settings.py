from dataclasses import dataclass

TESTMODE = False  # Set to True to use test data instead of polling the Live Client API.

TEST_ENEMY_LOADOUT = [
    {"champion": "Garen", "role": "TOP", "spells": ["Flash", "Teleport"]},
    {"champion": "Lee Sin", "role": "JUNGLE", "spells": ["Flash", "Smite"]},
    {"champion": "Karma", "role": "MIDDLE", "spells": ["Flash", "Ignite"]},
    {"champion": "Vayne", "role": "BOTTOM", "spells": ["Flash", "Heal"]},
    {"champion": "Twitch", "role": "UTILITY", "spells": ["Flash", "Heal"]},
]
TEST_ENEMY_LOADOUT_FILE = None

LIVE_CLIENT_ALL_GAME_DATA_URL = "https://127.0.0.1:2999/liveclientdata/allgamedata"
LIVE_CLIENT_POLL_INTERVAL_SECONDS = 3.0
MINIMAP_AUTO_REGION = True
MINIMAP_DEBUG_MODE = False
MINIMAP_CALIBRATION_MODE = False
MINIMAP_DEBUG_FRAME_PATH = "data/debug/minimap_latest.png"
MINIMAP_TRAINING_IMAGE_DIR = "img"
MINIMAP_TRAINING_LABELS_PATH = "img/labels.json"
MINIMAP_TRAINING_CAPTURE_MODE = False
MINIMAP_TRAINING_CAPTURE_DIR = "data/training_sessions"
MINIMAP_TRAINING_CAPTURE_INTERVAL_SECONDS = 3.0


@dataclass(frozen=True)
class BackendSettings:
    """Non-secret runtime constants used by the backend."""

    test_mode: bool
    minimap_region: dict[str, int]
    scan_interval_seconds: float
    test_enemy_loadout: list[dict[str, object]]
    live_client_all_game_data_url: str
    live_client_poll_interval_seconds: float = 3.0
    test_enemy_loadout_file: str | None = None
    minimap_auto_region: bool = True
    minimap_debug_mode: bool = False
    minimap_calibration_mode: bool = False
    minimap_debug_frame_path: str | None = None
    minimap_training_image_dir: str = "img"
    minimap_training_labels_path: str = "img/labels.json"
    minimap_training_capture_mode: bool = False
    minimap_training_capture_dir: str = "data/training_sessions"
    minimap_training_capture_interval_seconds: float = 3.0
    minimap_curated_crops_dir: str = "data/curation/auto_selected"
    minimap_max_templates_per_champion: int = 10


SETTINGS = BackendSettings(
    test_mode=TESTMODE,
    minimap_region={"top": 790, "left": 1630, "width": 290, "height": 290},
    scan_interval_seconds=1.0,
    test_enemy_loadout=TEST_ENEMY_LOADOUT,
    live_client_all_game_data_url=LIVE_CLIENT_ALL_GAME_DATA_URL,
    live_client_poll_interval_seconds=LIVE_CLIENT_POLL_INTERVAL_SECONDS,
    test_enemy_loadout_file=TEST_ENEMY_LOADOUT_FILE,
    minimap_auto_region=MINIMAP_AUTO_REGION,
    minimap_debug_mode=MINIMAP_DEBUG_MODE,
    minimap_calibration_mode=MINIMAP_CALIBRATION_MODE,
    minimap_debug_frame_path=MINIMAP_DEBUG_FRAME_PATH,
    minimap_training_image_dir=MINIMAP_TRAINING_IMAGE_DIR,
    minimap_training_labels_path=MINIMAP_TRAINING_LABELS_PATH,
    minimap_training_capture_mode=MINIMAP_TRAINING_CAPTURE_MODE,
    minimap_training_capture_dir=MINIMAP_TRAINING_CAPTURE_DIR,
    minimap_training_capture_interval_seconds=MINIMAP_TRAINING_CAPTURE_INTERVAL_SECONDS,
)

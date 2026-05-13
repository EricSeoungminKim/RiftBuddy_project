from unittest.mock import patch

from ai.chat_agent import ChatAgent
from ai.rag_engine import RagEngine
from ai.vector_store import VectorStore
from main import create_app_components


def test_ai_stubs_import() -> None:
    assert isinstance(ChatAgent(), ChatAgent)
    assert isinstance(RagEngine(), RagEngine)
    assert isinstance(VectorStore(), VectorStore)


def test_create_app_components_wires_spell_clicked() -> None:
    components = create_app_components()

    with patch("main.time.time", return_value=1000.0):
        components.event_bus.emit(
            "spell_clicked",
            {"champion": "Jinx", "spell": "Flash"},
        )

    spells = components.game_state.get_spells()
    assert spells["Jinx"]["Flash"]["available_at"] == 1300.0

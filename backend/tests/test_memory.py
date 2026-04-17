import pytest

from app.memory import SessionMemory


@pytest.fixture
def memory():
    return SessionMemory()


class TestGet:
    def test_unknown_session_returns_empty_dict(self, memory):
        assert memory.get("new_session") == {}

    def test_get_returns_copy_not_reference(self, memory):
        memory.update("s1", {"high_protein": True})
        result = memory.get("s1")
        result["injected"] = True
        assert "injected" not in memory.get("s1")


class TestUpdate:
    def test_stores_truthy_constraint(self, memory):
        memory.update("s1", {"high_protein": True})
        assert memory.get("s1")["high_protein"] is True

    def test_stores_integer_constraint(self, memory):
        memory.update("s1", {"max_calories": 400})
        assert memory.get("s1")["max_calories"] == 400

    def test_stores_string_constraint(self, memory):
        memory.update("s1", {"spice_level": "hot"})
        assert memory.get("s1")["spice_level"] == "hot"

    def test_merges_with_existing_preferences(self, memory):
        memory.update("s1", {"high_protein": True})
        memory.update("s1", {"no_dairy": True})
        result = memory.get("s1")
        assert result["high_protein"] is True
        assert result["no_dairy"] is True

    def test_false_value_not_stored(self, memory):
        memory.update("s1", {"vegan": False, "high_protein": True})
        result = memory.get("s1")
        assert "vegan" not in result
        assert result["high_protein"] is True

    def test_none_value_not_stored(self, memory):
        memory.update("s1", {"max_calories": None})
        assert "max_calories" not in memory.get("s1")

    def test_stores_zero_integer(self, memory):
        memory.update("s1", {"max_calories": 0})
        assert memory.get("s1")["max_calories"] == 0

    def test_later_value_overwrites_earlier(self, memory):
        memory.update("s1", {"spice_level": "mild"})
        memory.update("s1", {"spice_level": "hot"})
        assert memory.get("s1")["spice_level"] == "hot"


class TestClear:
    def test_clear_removes_all_preferences(self, memory):
        memory.update("s1", {"high_protein": True, "no_dairy": True})
        memory.clear("s1")
        assert memory.get("s1") == {}

    def test_clear_nonexistent_session_does_not_raise(self, memory):
        memory.clear("does_not_exist")

    def test_clear_one_session_leaves_others_intact(self, memory):
        memory.update("s1", {"high_protein": True})
        memory.update("s2", {"vegan": True})
        memory.clear("s1")
        assert memory.get("s2")["vegan"] is True


class TestIsolation:
    def test_sessions_do_not_share_state(self, memory):
        memory.update("s1", {"high_protein": True})
        memory.update("s2", {"vegan": True})
        assert "vegan" not in memory.get("s1")
        assert "high_protein" not in memory.get("s2")

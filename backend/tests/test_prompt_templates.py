import pytest

from training.prompt_templates import (
    CATEGORY_DISTRIBUTION,
    edge_case_prompt,
    followup_prompt,
    get_category_counts,
    get_prompt_for_category,
    single_constraint_prompt,
    style_prompt,
    three_constraint_prompt,
    two_constraint_prompt,
)


class TestSingleConstraintPrompt:
    def test_returns_non_empty_string(self):
        assert isinstance(single_constraint_prompt(), str)
        assert len(single_constraint_prompt()) > 0

    def test_varies_on_repeated_calls(self):
        results = {single_constraint_prompt() for _ in range(50)}
        assert len(results) > 1


class TestTwoConstraintPrompt:
    def test_returns_non_empty_string(self):
        result = two_constraint_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestThreeConstraintPrompt:
    def test_returns_non_empty_string(self):
        result = three_constraint_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestStylePrompt:
    def test_returns_non_empty_string(self):
        result = style_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestEdgeCasePrompt:
    def test_returns_non_empty_string(self):
        result = edge_case_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestFollowupPrompt:
    def test_returns_non_empty_string(self):
        result = followup_prompt()
        assert isinstance(result, str) and len(result) > 0


class TestGetPromptForCategory:
    def test_all_valid_categories_return_strings(self):
        for cat in CATEGORY_DISTRIBUTION:
            result = get_prompt_for_category(cat)
            assert isinstance(result, str) and len(result) > 0

    def test_unknown_category_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown category"):
            get_prompt_for_category("unknown")

    def test_single_category_varies(self):
        results = {get_prompt_for_category("single") for _ in range(50)}
        assert len(results) > 1

    def test_two_category_varies(self):
        results = {get_prompt_for_category("two") for _ in range(50)}
        assert len(results) > 1


class TestGetCategoryCounts:
    def test_counts_sum_to_total_5000(self):
        counts = get_category_counts(5000)
        assert sum(counts.values()) == 5000

    def test_counts_sum_to_total_100(self):
        counts = get_category_counts(100)
        assert sum(counts.values()) == 100

    def test_all_categories_present(self):
        counts = get_category_counts(5000)
        assert set(counts.keys()) == set(CATEGORY_DISTRIBUTION.keys())

    def test_no_zero_counts_for_large_total(self):
        counts = get_category_counts(5000)
        assert all(v > 0 for v in counts.values())

    def test_single_is_20_percent(self):
        counts = get_category_counts(5000)
        assert counts["single"] == 1000

    def test_two_is_30_percent(self):
        counts = get_category_counts(5000)
        assert counts["two"] == 1500

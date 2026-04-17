import json

from training.generate_dataset import parse_taco_output, save_dataset


class TestParseTacoOutput:
    def test_parses_clean_json(self):
        text = '{"name": "Chicken Taco", "calories": 350}'
        result = parse_taco_output(text)
        assert result == {"name": "Chicken Taco", "calories": 350}

    def test_strips_markdown_json_fence(self):
        text = '```json\n{"name": "test"}\n```'
        result = parse_taco_output(text)
        assert result == {"name": "test"}

    def test_strips_plain_markdown_fence(self):
        text = '```\n{"name": "test"}\n```'
        result = parse_taco_output(text)
        assert result == {"name": "test"}

    def test_extracts_json_from_surrounding_text(self):
        text = 'Here is your taco: {"name": "test", "calories": 300} Enjoy!'
        result = parse_taco_output(text)
        assert result is not None
        assert result["name"] == "test"

    def test_returns_none_for_invalid_json(self):
        assert parse_taco_output("not json at all") is None

    def test_returns_none_for_empty_string(self):
        assert parse_taco_output("") is None


class TestSaveDataset:
    def test_creates_train_jsonl_and_eval_json(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {"n": i}} for i in range(10)]
        save_dataset(examples, tmp_path, train_split=0.8)
        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "eval.json").exists()

    def test_returns_correct_counts(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {}} for i in range(100)]
        train_count, eval_count = save_dataset(examples, tmp_path, train_split=0.9)
        assert train_count == 90
        assert eval_count == 10

    def test_train_jsonl_one_line_per_example(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {}} for i in range(5)]
        save_dataset(examples, tmp_path, train_split=1.0)
        lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
        assert len(lines) == 5
        for line in lines:
            obj = json.loads(line)
            assert "instruction" in obj

    def test_eval_json_is_list(self, tmp_path):
        examples = [{"instruction": f"p{i}", "output": {}} for i in range(10)]
        save_dataset(examples, tmp_path, train_split=0.8)
        data = json.loads((tmp_path / "eval.json").read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        examples = [{"instruction": "test", "output": {}}]
        save_dataset(examples, nested, train_split=1.0)
        assert (nested / "train.jsonl").exists()

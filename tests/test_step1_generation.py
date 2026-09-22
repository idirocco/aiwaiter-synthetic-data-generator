from pathlib import Path

import pytest

from step1_generation import generate_step1, load_step1_prompt


class FakeQAItem:
    def __init__(self, question, answer, dining_scenario, menu_items, service_steps, safety_info, tips):
        self.question = question
        self.answer = answer
        self.dining_scenario = dining_scenario
        self.menu_items = menu_items
        self.service_steps = service_steps
        self.safety_info = safety_info
        self.tips = tips


class FakeQADataset:
    def __init__(self, qa_pairs):
        self.qa_pairs = qa_pairs

    def model_dump_json(self):
        return "{}"


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeQADataset([
            FakeQAItem(
                question=f"How do I order? ({len(self.calls)})",
                answer="Let me help.",
                dining_scenario="Dinner",
                menu_items=["Pasta"],
                service_steps=["1. Ask", "2. Confirm", "3. Serve"],
                safety_info="No major allergy risk.",
                tips=["Ask for a recommendation."],
            )
        ])


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakePatchedClient:
    def __init__(self):
        self.chat = FakeChat()


def test_load_step1_prompt_defaults_to_generator_prompt_default():
    prompt_root = Path(__file__).resolve().parent.parent / "prompts" / "step1_generator"
    expected = (prompt_root / "generator_prompt_default.txt").read_text(encoding="utf-8").strip()

    assert load_step1_prompt() == expected


def test_generate_step1_skips_when_existing_file_is_not_regenerated(monkeypatch, tmp_path):
    output_path = tmp_path / "step1_generated_qa.json"
    output_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: FakePatchedClient())

    result = generate_step1(
        client=object(),
        MODEL_NAME="test-model",
        categories=[{"name": "Test Category", "description": "Test description"}],
        prompt="Prompt {category} {items_per_category}",
        items_per_category=1,
        output_path=str(output_path),
    )

    assert result == []
    assert output_path.read_text(encoding="utf-8") == "[]"


def test_generate_step1_force_regenerate_skips_confirmation(monkeypatch, tmp_path):
    output_path = tmp_path / "step1_generated_qa.json"
    output_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("input should not be called when forced"))
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: FakePatchedClient())

    result = generate_step1(
        client=object(),
        MODEL_NAME="test-model",
        categories=[{"name": "Test Category", "description": "Test description"}],
        prompt="Prompt {category} {items_per_category}",
        items_per_category=1,
        force_regenerate=True,
        output_path=str(output_path),
    )

    assert len(result) == 1
    assert result[0]["qa_item"].question == "How do I order? (1)"


def test_generate_step1_loops_through_each_category(monkeypatch, tmp_path):
    output_path = tmp_path / "step1_generated_qa.json"
    output_path.write_text("[]", encoding="utf-8")

    fake_client = FakePatchedClient()

    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("input should not be called when forced"))
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: fake_client)

    result = generate_step1(
        client=object(),
        MODEL_NAME="test-model",
        categories=[
            {"name": "Category A", "description": "Alpha"},
            {"name": "Category B", "description": "Beta"},
        ],
        prompt="Prompt for {category}: {category_description}. {items_per_category} items.",
        items_per_category=1,
        force_regenerate=True,
        output_path=str(output_path),
    )

    assert len(result) == 2
    assert len(fake_client.chat.completions.calls) == 2
    assert [record["category_name"] for record in result] == ["Category A", "Category B"]

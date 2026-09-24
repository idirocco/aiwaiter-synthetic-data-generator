import json
from pathlib import Path

import openai
import pytest

from step1_generation import build_step1_output_path, generate_step1, load_step1_prompt
from step4_llm_as_judge import run_llm_judge


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


def test_load_step1_prompt_defaults_to_default():
    prompt_root = Path(__file__).resolve().parent.parent / "prompts" / "step1_generator"
    expected = (prompt_root / "default.txt").read_text(encoding="utf-8").strip()

    assert load_step1_prompt() == expected


def test_build_step1_output_path_includes_generator_prompt():
    custom_path = build_step1_output_path("my_custom_prompt.txt")
    default_path = build_step1_output_path()

    assert custom_path.endswith("step1_generated_qa_my_custom_prompt.json")
    assert default_path.endswith("step1_generated_qa_default.json")


def test_generate_step1_names_output_after_generator_prompt(monkeypatch, tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "custom_generator.txt").write_text("Prompt {category}", encoding="utf-8")

    monkeypatch.setattr("step1_generation.output_path", lambda filename: tmp_path / filename)
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: FakePatchedClient())

    generate_step1(
        client=object(),
        GENERATOR_MODEL_NAME="test-model",
        categories=[{"name": "Test Category", "description": "Test description"}],
        items_per_category=1,
        force_regenerate=True,
        prompt_name="custom_generator",
        prompts_dir=prompt_dir,
    )

    assert (tmp_path / "step1_generated_qa_custom_generator.json").exists()


def test_generate_step1_skips_when_existing_file_is_not_regenerated(monkeypatch, tmp_path):
    output_path = tmp_path / "step1_generated_qa.json"
    output_path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: FakePatchedClient())

    result = generate_step1(
        client=object(),
        GENERATOR_MODEL_NAME="test-model",
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
        GENERATOR_MODEL_NAME="test-model",
        categories=[{"name": "Test Category", "description": "Test description"}],
        prompt="Prompt {category} {items_per_category}",
        items_per_category=1,
        force_regenerate=True,
        output_path=str(output_path),
    )

    assert len(result) == 1
    assert result[0]["qa_item"].question == "How do I order? (1)"


def test_generate_step1_handles_rate_limit_error(monkeypatch, tmp_path):
    output_path = tmp_path / "step1_generated_qa.json"
    output_path.write_text("[]", encoding="utf-8")

    class RateLimitedCompletions:
        def create(self, **kwargs):
            raise openai.RateLimitError("rate limit reached", response=None, body=None)

    class RateLimitedClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": RateLimitedCompletions()})()

    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("input should not be called when forced"))
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: RateLimitedClient())

    result = generate_step1(
        client=object(),
        GENERATOR_MODEL_NAME="test-model",
        categories=[{"name": "Test Category", "description": "Test description"}],
        prompt="Prompt {category} {items_per_category}",
        items_per_category=1,
        force_regenerate=True,
        output_path=str(output_path),
    )

    assert result == []


def test_run_llm_judge_handles_malformed_response(monkeypatch, tmp_path):
    class MalformedResponse:
        pass

    class MalformedCompletions:
        def create(self, **kwargs):
            return MalformedResponse()

    class MalformedClient:
        def __init__(self):
            self.chat = type("Chat", (), {"completions": MalformedCompletions()})()

    step1_file = tmp_path / "step1_generated_qa.json"
    step1_file.write_text(
        json.dumps(
            [
                {
                    "metadata": {"category_name": "Test"},
                    "question": "Any gluten-free options?",
                    "answer": "Yes, we can help.",
                    "dining_scenario": "Gluten-free allergy",
                    "menu_items": ["Burger"],
                    "service_steps": ["Confirm", "Prepare", "Serve"],
                    "safety_info": "Cross-contact risk.",
                    "tips": ["Ask about contamination."],
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("step4_llm_as_judge.instructor.patch", lambda client: MalformedClient())

    result = run_llm_judge(
        client=object(),
        input_path=str(step1_file),
        output_path=str(tmp_path / "judge_output.json"),
    )

    assert result == []


def test_generate_step1_loops_through_each_category(monkeypatch, tmp_path):
    output_path = tmp_path / "step1_generated_qa.json"
    output_path.write_text("[]", encoding="utf-8")

    fake_client = FakePatchedClient()

    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("input should not be called when forced"))
    monkeypatch.setattr("step1_generation.instructor.patch", lambda client: fake_client)

    result = generate_step1(
        client=object(),
        GENERATOR_MODEL_NAME="test-model",
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

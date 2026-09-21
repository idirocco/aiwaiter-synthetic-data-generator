from step3_human_labeling import run_human_labeling
from step4_llm_as_judge import build_judge_prompt


class FakeQAItem:
    def __init__(self):
        self.question = "Why is the table delayed?"
        self.answer = "We are waiting for the kitchen."
        self.dining_scenario = "Dinner"
        self.menu_items = ["Pasta"]
        self.service_steps = ["Greet guest", "Take order"]
        self.safety_info = "None"
        self.tips = ["Stay calm"]


def test_run_human_labeling_defaults_to_yes_on_blank_input(monkeypatch):
    records = [{"qa_item": FakeQAItem(), "category_name": "Test"}]
    captured = {}

    def fake_input(prompt):
        return ""

    def fake_save(records, output_path=None):
        captured["records"] = records
        return "mocked-output.json"

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("step3_human_labeling.save_human_labels_to_json", fake_save)

    result = run_human_labeling(records)

    assert len(result) == 1
    assert result[0]["human_labels"] == {
        "answer_completeness": True,
        "safety_specificity": True,
        "menu_realism": True,
        "scope_appropriateness": True,
        "context_clarity": True,
        "tip_usefulness": True,
    }
    assert captured["records"][0]["human_labels"] == {
        "answer_completeness": True,
        "safety_specificity": True,
        "menu_realism": True,
        "scope_appropriateness": True,
        "context_clarity": True,
        "tip_usefulness": True,
    }


def test_build_judge_prompt_includes_rubric_examples_and_human_priority():
    qa_item = FakeQAItem()
    prompt = build_judge_prompt(qa_item)

    assert "answer_completeness" in prompt
    assert "Pass criteria" in prompt
    assert "Fail examples" in prompt
    assert "Step 3 human labels are the priority reference" in prompt
    assert "ambiguous or partially correct" in prompt
    assert "allergy" in prompt.lower()

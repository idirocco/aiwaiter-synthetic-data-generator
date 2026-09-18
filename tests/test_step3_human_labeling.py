from step3_human_labeling import run_human_labeling


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
        "accuracy": True,
        "safety": True,
        "helpfulness": True,
        "clarity": True,
        "completeness": True,
        "tone": True,
    }
    assert captured["records"][0]["human_labels"] == {
        "accuracy": True,
        "safety": True,
        "helpfulness": True,
        "clarity": True,
        "completeness": True,
        "tone": True,
    }

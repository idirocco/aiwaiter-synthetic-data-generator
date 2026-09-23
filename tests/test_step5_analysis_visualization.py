import json
from pathlib import Path

from step4_llm_as_judge import build_judge_output_path, build_step4_output_path, load_judge_prompt
from step5_analysis_visualization import aggregate_segment_metrics, run_step5_analysis


def test_load_judge_prompt_reads_prompt_file(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "custom_judge.txt"
    prompt_file.write_text("custom rubric for {question}", encoding="utf-8")

    prompt = load_judge_prompt("custom_judge.txt", prompts_dir=prompt_dir)

    assert "custom rubric" in prompt
    assert "{question}" in prompt


def test_load_judge_prompt_defaults_to_judge_prompt_default():
    prompt_root = Path(__file__).resolve().parent.parent / "prompts" / "step4_judge"
    expected = (prompt_root / "judge_prompt_default.txt").read_text(encoding="utf-8").strip()

    assert load_judge_prompt() == expected


def test_build_step4_output_path_includes_generator_prompt():
    custom_path = build_step4_output_path("my_custom_prompt.txt")
    default_path = build_step4_output_path()

    assert custom_path.endswith("step4_llm_judge_labels_my_custom_prompt.json")
    assert default_path.endswith("step4_llm_judge_labels_generator_prompt_default.json")


def test_build_judge_output_path_includes_prompt_name():
    output_path = build_judge_output_path("custom_judge.txt")
    assert "custom_judge" in output_path


def test_aggregate_segment_metrics_computes_pass_rates_and_agreement():
    records = [
        {
            "category_name": "Menu Guidance & Recommendations",
            "prompt_variant": "baseline",
            "human_labels": {
                "answer_completeness": 1,
                "safety_specificity": 1,
                "menu_realism": 0,
                "scope_appropriateness": 1,
                "context_clarity": 0,
                "tip_usefulness": 1,
            },
            "llm_judge_labels": {
                "answer_completeness": 1,
                "safety_specificity": 1,
                "menu_realism": 0,
                "scope_appropriateness": 1,
                "context_clarity": 1,
                "tip_usefulness": 1,
            },
            "overall_pass": True,
            "llm_overall_pass": True,
        },
        {
            "category_name": "Menu Guidance & Recommendations",
            "prompt_variant": "baseline",
            "human_labels": {
                "answer_completeness": 0,
                "safety_specificity": 1,
                "menu_realism": 1,
                "scope_appropriateness": 0,
                "context_clarity": 1,
                "tip_usefulness": 0,
            },
            "llm_judge_labels": {
                "answer_completeness": 0,
                "safety_specificity": 1,
                "menu_realism": 1,
                "scope_appropriateness": 0,
                "context_clarity": 1,
                "tip_usefulness": 1,
            },
            "overall_pass": False,
            "llm_overall_pass": False,
        },
    ]

    metrics = aggregate_segment_metrics(records, group_key="category_name")

    assert metrics["Menu Guidance & Recommendations"]["answer_completeness"]["human_pass_rate"] == 0.5
    assert metrics["Menu Guidance & Recommendations"]["answer_completeness"]["llm_pass_rate"] == 0.5
    assert metrics["Menu Guidance & Recommendations"]["answer_completeness"]["agreement_rate"] == 1.0
    assert metrics["Menu Guidance & Recommendations"]["overall_pass"]["human_pass_rate"] == 0.5
    assert metrics["Menu Guidance & Recommendations"]["overall_pass"]["llm_pass_rate"] == 0.5


def test_run_step5_analysis_writes_segment_metrics_json(tmp_path):
    generated_path = tmp_path / "step1_generated_qa.json"
    human_path = tmp_path / "step3_human_labels.json"
    llm_path = tmp_path / "step4_llm_judge_labels.json"
    output_dir = tmp_path / "visualizations"

    generated_path.write_text(
        json.dumps(
            [
                {
                    "metadata": {"category_name": "Menu Guidance & Recommendations", "prompt_variant": "baseline"},
                    "question": "What should I order?",
                    "answer": "I recommend the salmon.",
                    "dining_scenario": "Guest wants a recommendation.",
                    "menu_items": ["Salmon"],
                    "service_steps": ["Ask preference", "Recommend dish", "Confirm order"],
                    "safety_info": "N/A",
                    "tips": ["Ask about allergies"],
                }
            ]
        ),
        encoding="utf-8",
    )
    human_path.write_text(
        json.dumps(
            [
                {
                    "trace_id": "qa_001",
                    "overall_pass": True,
                    "human_labels": {
                        "answer_completeness": 1,
                        "safety_specificity": 1,
                        "menu_realism": 1,
                        "scope_appropriateness": 1,
                        "context_clarity": 1,
                        "tip_usefulness": 1,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    llm_path.write_text(
        json.dumps(
            [
                {
                    "trace_id": "qa_001",
                    "overall_pass": True,
                    "llm_judge_labels": {
                        "answer_completeness": 1,
                        "safety_specificity": 1,
                        "menu_realism": 1,
                        "scope_appropriateness": 1,
                        "context_clarity": 1,
                        "tip_usefulness": 1,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )

    result = run_step5_analysis(
        generated_path=generated_path,
        human_path=human_path,
        llm_path=llm_path,
        output_dir=output_dir,
    )

    metrics_path = output_dir / "step5_segment_metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "Menu Guidance & Recommendations" in payload
    assert result["summary"]["Menu Guidance & Recommendations"]["answer_completeness"]["human_pass_rate"] == 1.0

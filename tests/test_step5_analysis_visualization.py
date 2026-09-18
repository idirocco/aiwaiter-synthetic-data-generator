from step5_analysis_visualization import aggregate_segment_metrics


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

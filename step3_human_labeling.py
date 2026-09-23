# Step 3 (Human Labeling — 6 Dimensions):
# A simple Python CLI walks a reviewer through each item and collects binary pass/fail labels on all 6 quality dimensions.
# Labels are saved per item with a trace_id.

import json
from pathlib import Path

from config import output_path
from quality_dimensions import QUALITY_DIMENSIONS
from step1_generation import build_step1_output_path, generator_prompt_slug, load_step1_records


def build_step3_output_path(generator_prompt: str | None = None) -> str:
    return str(output_path(f"step3_human_labels_{generator_prompt_slug(generator_prompt)}.json"))


def save_human_labels_to_json(
    labeled_records,
    output_path: str | None = None,
    generator_prompt: str | None = None,
):
    if output_path is None:
        output_path = build_step3_output_path(generator_prompt)
    output_records = []

    for idx, record in enumerate(labeled_records):
        labels = record.get("human_labels", {})
        trace_id = record.get("trace_id") or f"qa_{idx + 1:03d}"

        output_record = {
            "trace_id": trace_id,
            "labeler": "human",
        }

        for dimension in QUALITY_DIMENSIONS:
            output_record[dimension] = int(bool(labels.get(dimension, False)))

        output_record["overall_pass"] = bool(all(labels.get(dimension, False) for dimension in QUALITY_DIMENSIONS))
        output_records.append(output_record)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    print(f"Saved human labels to: {output_file.resolve()}")
    return str(output_file)


def run_human_labeling(generator_prompt: str | None = None, input_path: str | None = None):
    resolved_path = input_path or build_step1_output_path(generator_prompt)
    print(f"\n--- Loading Step 1 records from {resolved_path} ---")
    all_generated_qa_records = load_step1_records(input_path=resolved_path)

    print("\n--- Starting Interactive Human Labeling (Step 3) ---")

    human_labeled_records = []

    if not all_generated_qa_records:
        print("No records available for Step 3 labeling.")
        return human_labeled_records

    for i, record in enumerate(all_generated_qa_records):
        qa_item = record["qa_item"]
        trace_id = f"qa_{i + 1:03d}"

        print(f"\n--- Labeling Record {i + 1}/{len(all_generated_qa_records)} (Trace ID: {trace_id}) ---")
        print("Full record:")
        print(json.dumps(
            {
                "question": qa_item.question,
                "answer": qa_item.answer,
                "dining_scenario": qa_item.dining_scenario,
                "menu_items": list(qa_item.menu_items),
                "service_steps": list(qa_item.service_steps),
                "safety_info": qa_item.safety_info,
                "tips": list(qa_item.tips),
                "category_name": record.get("category_name"),
                "category_description": record.get("category_description"),
                "timestamp": record.get("timestamp"),
                "model_name": record.get("model_name"),
            },
            indent=2,
            ensure_ascii=False,
        ))

        human_labels = {}
        for dimension in QUALITY_DIMENSIONS:
            while True:
                user_input = input(f"  {dimension} (Y/n): ").strip().lower()
                if user_input in ["", "y", "yes"]:
                    human_labels[dimension] = True
                    break
                if user_input in ["n", "no"]:
                    human_labels[dimension] = False
                    break
                print("Invalid input. Please enter 'y' or 'n'.")

        labeled_record = record.copy()
        labeled_record["trace_id"] = trace_id
        labeled_record["human_labels"] = human_labels
        human_labeled_records.append(labeled_record)

    save_human_labels_to_json(human_labeled_records, generator_prompt=generator_prompt)

    print("\n--- Interactive Human Labeling Completed ---")
    print(f"Total human-labeled records: {len(human_labeled_records)}")
    if human_labeled_records:
        print(f"First record's trace_id: {human_labeled_records[0]['trace_id']}")
        print(f"First record's human labels: {human_labeled_records[0]['human_labels']}")
    else:
        print("No records were labeled.")

    return human_labeled_records
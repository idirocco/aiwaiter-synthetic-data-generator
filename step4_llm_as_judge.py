# Step 4 (LLM-as-Judge — 6 Dimensions):
# An independent LLM judge scores the same 6 dimensions for every item using a
# structured Instructor/Pydantic schema and a lower temperature for deterministic scoring.

import json
from pathlib import Path
from typing import Any

import instructor
from pydantic import BaseModel, Field

from config import output_path
from quality_dimensions import QUALITY_DIMENSIONS


class DimensionVerdict(BaseModel):
    pass_: bool = Field(description="Whether this dimension passes for the item")
    rationale: str = Field(description="Short justification for the verdict")


class JudgeOutput(BaseModel):
    overall_pass: bool = Field(description="Whether the item passes all dimensions")
    dimension_scores: dict[str, DimensionVerdict] = Field(
        description="A verdict for each quality dimension using the exact names in QUALITY_DIMENSIONS"
    )


def build_judge_prompt(qa_item: Any) -> str:
    return f"""You are an impartial QA judge for restaurant-order synthetic data.

Evaluate this item for the following six dimensions exactly by name:
{', '.join(QUALITY_DIMENSIONS)}

Requirements:
- Be strict and evidence-based.
- Only mark a dimension as passed when the item clearly satisfies it.
- Be especially careful with allergy/safety wording, menu realism, guest-specific context, and operational feasibility.
- If the answer is generic, vague, or skips important details, mark the dimension as failed.
- Use the exact dimension names in the output map.

Item to judge:
Question: {qa_item.question}
Answer: {qa_item.answer}
Dining scenario: {qa_item.dining_scenario}
Menu items: {qa_item.menu_items}
Service steps: {qa_item.service_steps}
Safety info: {qa_item.safety_info}
Tips: {qa_item.tips}

Return a structured result: overall_pass + one verdict per dimension.
"""


def save_judge_results_to_json(judged_records, output_path: str = str(output_path("step4_llm_judge_labels.json"))):
    output_records = []
    for idx, record in enumerate(judged_records):
        verdicts = record.get("llm_judge_labels", {})
        trace_id = record.get("trace_id") or f"qa_{idx + 1:03d}"

        output_record = {
            "trace_id": trace_id,
            "labeler": "llm_judge",
        }

        for dimension in QUALITY_DIMENSIONS:
            verdict = verdicts.get(dimension)
            output_record[dimension] = int(bool(verdict.pass_ if verdict is not None else False))

        output_record["overall_pass"] = bool(record.get("llm_overall_pass", False))
        output_records.append(output_record)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    print(f"Saved LLM judge labels to: {output_file.resolve()}")
    return str(output_file)


def run_llm_judge(all_generated_qa_records, client, model_name: str, temperature: float = 0.0, output_path: str = str(output_path("step4_llm_judge_labels.json"))):
    patched_client = instructor.patch(client)

    judged_records = []

    if not all_generated_qa_records:
        print("No records available for Step 4 judging.")
        return judged_records

    print("\n--- Starting LLM-as-Judge (Step 4) ---")

    for i, record in enumerate(all_generated_qa_records):
        qa_item = record["qa_item"]
        trace_id = f"qa_{i + 1:03d}"

        judge_result = patched_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict, evidence-based evaluator for synthetic restaurant QA data. Return only deterministic, structured results.",
                },
                {"role": "user", "content": build_judge_prompt(qa_item)},
            ],
            response_model=JudgeOutput,
            temperature=temperature,
            max_retries=3,
        )

        judged_record = record.copy()
        judged_record["trace_id"] = trace_id
        judged_record["llm_judge_result"] = judge_result
        judged_record["llm_judge_labels"] = judge_result.dimension_scores
        judged_record["llm_overall_pass"] = judge_result.overall_pass
        judged_records.append(judged_record)

        print(f"\n--- Judge Result {i + 1}/{len(all_generated_qa_records)} ---")
        print(f"Overall pass: {judge_result.overall_pass}")
        for dimension_name, verdict in judge_result.dimension_scores.items():
            print(f"  {dimension_name}: {'PASS' if verdict.pass_ else 'FAIL'} - {verdict.rationale}")

    print("\n--- LLM-as-Judge Completed ---")
    save_judge_results_to_json(judged_records, output_path=output_path)
    return judged_records


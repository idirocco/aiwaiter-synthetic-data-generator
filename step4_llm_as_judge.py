# Step 4 (LLM-as-Judge — 6 Dimensions):
# An independent LLM judge scores the same 6 dimensions for every item using a
# structured Instructor/Pydantic schema and a lower temperature for deterministic scoring.

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import instructor
from pydantic import BaseModel, Field

from config import output_path
from quality_dimensions import QUALITY_DIMENSIONS

DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "step4_judge"


def load_judge_prompt(prompt_name: str | None = None, prompts_dir: str | Path | None = None) -> str:
    if prompt_name is None:
        prompt_name = "default_judge_prompt.txt"
    if not prompt_name.endswith(".txt"):
        prompt_name = f"{prompt_name}.txt"

    prompt_root = Path(prompts_dir) if prompts_dir is not None else DEFAULT_PROMPTS_DIR
    prompt_file = prompt_root / prompt_name

    if not prompt_file.exists():
        raise FileNotFoundError(f"Judge prompt file not found: {prompt_file.resolve()}")

    template = prompt_file.read_text(encoding="utf-8").strip()
    return template


def build_judge_output_path(prompt_name: str | None = None, prefix: str = "step4_llm_judge_labels") -> str:
    prompt_slug = "default"
    if prompt_name:
        prompt_slug = Path(prompt_name).stem
    normalized_name = prompt_slug.replace(" ", "_")
    return str(output_path(f"{prefix}_{normalized_name}.json"))


class DimensionVerdict(BaseModel):
    pass_: bool = Field(description="Whether this dimension passes for the item")
    rationale: str = Field(description="Short justification for the verdict")


class JudgeOutput(BaseModel):
    overall_pass: bool = Field(description="Whether the item passes all dimensions")
    dimension_scores: dict[str, DimensionVerdict] = Field(
        description="A verdict for each quality dimension using the exact names in QUALITY_DIMENSIONS"
    )


def build_judge_prompt(qa_item: Any, prompt_name: str | None = None, prompts_dir: str | Path | None = None) -> str:
    template = load_judge_prompt(prompt_name=prompt_name, prompts_dir=prompts_dir)
    prompt_data = {
        "quality_dimensions": ", ".join(QUALITY_DIMENSIONS),
        "question": qa_item.question,
        "answer": qa_item.answer,
        "dining_scenario": qa_item.dining_scenario,
        "menu_items": qa_item.menu_items,
        "service_steps": qa_item.service_steps,
        "safety_info": qa_item.safety_info,
        "tips": qa_item.tips,
    }
    return template.format(**prompt_data)


def timestamped_output_path(prefix: str = "step4_llm_judge_labels", prompt_name: str | None = None) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    if prompt_name:
        prompt_slug = Path(prompt_name).stem
        return str(output_path(f"{prefix}_{prompt_slug}.json"))
    return str(output_path(f"{prefix}.json"))
    #return str(output_path(f"{prefix}_{timestamp}.json"))


def save_judge_results_to_json(judged_records, output_path: str | None = None):
    if output_path is None:
        output_path = timestamped_output_path()

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


def run_llm_judge(
    all_generated_qa_records,
    client,
    model_name: str,
    temperature: float = 0.1,
    output_path: str | None = None,
    prompt_name: str | None = None,
    prompts_dir: str | Path | None = None,
):
    if output_path is None:
        output_path = timestamped_output_path(prompt_name=prompt_name)
    patched_client = instructor.patch(client)

    judged_records = []

    if not all_generated_qa_records:
        print("No records available for Step 4 judging.")
        return judged_records

    print("\n--- Starting LLM-as-Judge (Step 4) ---")

    for i, record in enumerate(all_generated_qa_records):
        qa_item = record["qa_item"]
        trace_id = f"qa_{i + 1:03d}"

        judge_prompt = build_judge_prompt(qa_item, prompt_name=prompt_name, prompts_dir=prompts_dir)
        judge_result = patched_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict, evidence-based evaluator for synthetic restaurant QA data. Return only deterministic, structured results.",
                },
                {"role": "user", "content": judge_prompt},
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


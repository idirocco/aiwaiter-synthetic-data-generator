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

Critical calibration rule:
- If the answer is plausible but vague, incomplete, or only partially correct, do not reward it just because it sounds good.
- Default to a fail when the evidence is weak, ambiguous, or missing.
- Use the exact dimension names in the output map.

Strict rubric for each dimension:

1) answer_completeness
   Pass criteria:
   - Directly answers the question and includes a recommendation.
   - Covers the missing information a guest would reasonably need to proceed.
   Fail examples:
   - "I’d suggest a safe option" without naming what, why, or what to do next.
   - "The kitchen can handle it" with no actual actionable recommendation.
   - An answer that avoids the core issue or leaves out the necessary operational detail.

2) safety_specificity
   Pass criteria:
   - Mentions the real safety concern, allergy, ingredient risk, cross-contact concern, or prep requirement clearly.
   - Names concrete precautions such as ingredient avoidance, allergy flag, clean pan, separate utensils, or kitchen confirmation.
   - Ties safety to the actual item or guest context.
   Fail examples:
   - "Be careful with allergies" with no specific risk or mitigation.
   - "Ask the kitchen" without saying what to avoid or how to reduce cross-contact.
   - A generic disclaimer that does not mention the actual dietary issue.

3) menu_realism
   Pass criteria:
   - Menu suggestions are realistic restaurant items that fit the scenario and cuisine.
   - Meal choices and add-ons are plausible for a normal service workflow.
   - The answer does not invent impossible dishes, absurd substitutions, or nonsense combinations.
   Fail examples:
   - Suggesting a completely impossible dish for the restaurant type.
   - Recommending ingredients or meals that do not fit the menu context or service flow.
   - Mixing incompatible items without justification.

4) scope_appropriateness
   Pass criteria:
   - Stays within the guest request and service scope.
   - Uses appropriate service behaviors such as confirming dietary needs, suggesting options, or escalating the issue when needed.
   - Does not overstep by giving legal, medical, or policy advice beyond the service context.
   Fail examples:
   - Answering beyond the actual question with irrelevant policy/medical commentary.
   - Giving vague, generic suggestions unrelated to the guest request.
   - Overcomplicated or unrelated operational steps that are not asked for.

5) context_clarity
   Pass criteria:
   - Clearly reflects the guest's real situation, preference, urgency, or dining scenario.
   - Uses the specific context from the record: allergy severity, meal type, timing, dietary restriction, or service concern.
   - Connects the recommendation to that context rather than speaking in abstract terms.
   Fail examples:
   - A generic answer that could fit any guest, regardless of the scenario.
   - Ignoring the presence of a severe allergy, a time-sensitive request, or a dietary need.
   - Failing to mention the guest-specific details that matter.

6) tip_usefulness
   Pass criteria:
   - Tips are concrete and relevant to the guest or service workflow.
   - They help reduce risk or improve the experience in a realistic way.
   - They are specific enough to be used by staff in the moment.
   Fail examples:
   - "Be careful" or "Use common sense" without a useful operational tip.
   - Tips that are generic, repetitive, or irrelevant to the actual problem.
   - Advice that does not help the guest or staff complete the task.

Pass/fail decision rule:
- PASS only if the evidence clearly supports the rubric for that dimension.
- FAIL when the answer is generic, incomplete, partially correct, or missing key details.
- If a response contains obvious safety gaps, menu implausibility, or no guest-specific context, fail those dimensions even if the answer sounds polished.
- Ambiguous or partially correct answers should be treated as failed rather than passed.
- If an answer is ambiguous or partially correct, fail it unless the item clearly satisfies the dimension with specific evidence.

Edge cases to judge conservatively:
- Allergy/safety wording: if the answer says something like "avoid allergens" but does not specify the ingredient, cross-contact risk, or safe substitute, mark safety_specificity as failed.
- Menu realism: if the answer suggests an impossible or implausible combination for a restaurant menu, mark menu_realism as failed.
- Guest context: if the answer could apply to any customer and ignores the actual dining scenario, mark context_clarity as failed.
- Operational feasibility: if the answer is not practical in a real service flow, mark scope_appropriateness or answer_completeness as failed.
- Tip quality: if tips are broad and unhelpful, mark tip_usefulness as failed even if the main answer is otherwise okay.

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


def timestamped_output_path(prefix: str = "step4_llm_judge_labels") -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
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


def run_llm_judge(all_generated_qa_records, client, model_name: str, temperature: float = 0.0, output_path: str | None = None):
    if output_path is None:
        output_path = timestamped_output_path()
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


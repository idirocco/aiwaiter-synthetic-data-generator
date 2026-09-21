# Local setup
# Install dependencies first:
#   python3 -m pip install -r requirements.txt

from pathlib import Path

from config import CATEGORIES, MODEL_NAME, PROMPT, get_client, print_modules_loaded, print_startup_banner
from startup_checks import run_startup_checks
from step1_generation import generate_step1, load_step1_records
from step2_validation import validate_step2
from step3_human_labeling import run_human_labeling
from step4_llm_as_judge import run_llm_judge
from step5_analysis_visualization import run_step5_analysis

items_per_category = 10

run_startup_checks()


def main(step: str | None = None):
    client = get_client()
    print_startup_banner()
    print_modules_loaded()

    step_name = (step or "all").lower().strip()

    if step_name == "step1":
        return generate_step1(
            client=client,
            MODEL_NAME=MODEL_NAME,
            categories=CATEGORIES,
            prompt=PROMPT,
            items_per_category=items_per_category,
        )

    try:
        output_file = Path("output/step1_generated_qa.json")
        if step_name in {"step2", "step3", "step4", "all"} and output_file.exists():
            user_choice = input(
                f"Step 1 output file already exists at {output_file.resolve()}. Regenerate it? [y/N]: "
            ).strip().lower()
            if user_choice in {"y", "yes"}:
                all_generated_qa_records = generate_step1(
                    client=client,
                    MODEL_NAME=MODEL_NAME,
                    categories=CATEGORIES,
                    prompt=PROMPT,
                    items_per_category=items_per_category,
                    force_regenerate=True,
                )
                all_generated_qa_records = load_step1_records()
            else:
                all_generated_qa_records = load_step1_records()
        else:
            all_generated_qa_records = load_step1_records()
    except FileNotFoundError:
        if step_name in {"step2", "step3", "step4", "all"}:
            all_generated_qa_records = generate_step1(
                client=client,
                MODEL_NAME=MODEL_NAME,
                categories=CATEGORIES,
                prompt=PROMPT,
                items_per_category=items_per_category,
                force_regenerate=True,
            )
            all_generated_qa_records = load_step1_records()
        else:
            raise

    if step_name == "step2":
        return validate_step2(all_generated_qa_records)

    if step_name == "step3":
        return run_human_labeling(all_generated_qa_records)

    if step_name == "step4":
        return run_llm_judge(all_generated_qa_records, client=client, model_name=MODEL_NAME)

    if step_name == "step5":
        return run_step5_analysis()

    if step_name not in {"all", ""}:
        raise ValueError(f"Unsupported pipeline step: {step!r}")

    all_generated_qa_records = validate_step2(all_generated_qa_records)
    run_human_labeling(all_generated_qa_records)
    run_llm_judge(all_generated_qa_records, client=client, model_name=MODEL_NAME)
    return run_step5_analysis()


if __name__ == "__main__":
    import sys

    requested_step = sys.argv[1] if len(sys.argv) > 1 else None
    main(step=requested_step)

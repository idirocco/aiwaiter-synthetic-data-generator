# Local setup
# Install dependencies first:
#   python3 -m pip install -r requirements.txt

import argparse
from pathlib import Path

from config import CATEGORIES, GENERATOR_MODEL_NAME, PROMPT, get_client, print_modules_loaded, print_startup_banner
from startup_checks import run_startup_checks
from step1_generation import build_step1_output_path, generate_step1, load_step1_records
from step2_validation import validate_step2
from step3_human_labeling import run_human_labeling
from step4_llm_as_judge import run_llm_judge
from step5_analysis_visualization import run_step5_analysis

run_startup_checks()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic data pipeline.")
    parser.add_argument(
        "step",
        nargs="?",
        default=None,
        help="Pipeline stage to run: step1, step2, step3, step4, step5, or all (default: all).",
    )
    parser.add_argument(
        "--generator-prompt",
        default=None,
        metavar="NAME",
        help="Generator prompt under prompts/step1_generator/. The .txt extension is optional.",
    )
    parser.add_argument(
        "--judge-prompt",
        default=None,
        metavar="NAME",
        help="Judge prompt under prompts/step4_judge/. The .txt extension is optional.",
    )
    parser.add_argument(
        "--items-per-category",
        type=int,
        default=10,
        metavar="N",
        help="Number of Q&A items to generate for each category (default: 10).",
    )
    args = parser.parse_args(argv)
    if args.items_per_category < 1:
        parser.error("--items-per-category must be at least 1")
    args.generator_prompt = args.generator_prompt or None
    args.judge_prompt = args.judge_prompt or None
    return args


def main(
    step: str | None = None,
    generator_prompt: str | None = None,
    judge_prompt: str | None = None,
    items_per_category: int = 10,
):
    client = get_client()
    print_startup_banner()
    print_modules_loaded()

    step_name = (step or "all").lower().strip()
    step1_output = build_step1_output_path(generator_prompt)

    if step_name == "step5":
        return run_step5_analysis(generator_prompt=generator_prompt, judge_prompt=judge_prompt)

    if step_name == "step1":
        return generate_step1(
            client=client,
            GENERATOR_MODEL_NAME=GENERATOR_MODEL_NAME,
            categories=CATEGORIES,
            prompt=PROMPT if generator_prompt is None else None,
            items_per_category=items_per_category,
            output_path=step1_output,
            prompt_name=generator_prompt,
        )

    try:
        output_file = Path(step1_output)
        if step_name in {"step2", "step3", "step4", "all"} and output_file.exists():
            user_choice = input(
                f"Step 1 output file already exists at {output_file.resolve()}. Regenerate it? [y/N]: "
            ).strip().lower()
            if user_choice in {"y", "yes"}:
                all_generated_qa_records = generate_step1(
                    client=client,
                    GENERATOR_MODEL_NAME=GENERATOR_MODEL_NAME,
                    categories=CATEGORIES,
                    prompt=PROMPT if generator_prompt is None else None,
                    items_per_category=items_per_category,
                    output_path=step1_output,
                    force_regenerate=True,
                    prompt_name=generator_prompt,
                )
                all_generated_qa_records = load_step1_records(input_path=step1_output)
            else:
                all_generated_qa_records = load_step1_records(input_path=step1_output)
        else:
            all_generated_qa_records = load_step1_records(input_path=step1_output)
    except FileNotFoundError:
        if step_name in {"step2", "step3", "step4", "all"}:
            all_generated_qa_records = generate_step1(
                client=client,
                GENERATOR_MODEL_NAME=GENERATOR_MODEL_NAME,
                categories=CATEGORIES,
                prompt=PROMPT if generator_prompt is None else None,
                items_per_category=items_per_category,
                output_path=step1_output,
                force_regenerate=True,
                prompt_name=generator_prompt,
            )
            all_generated_qa_records = load_step1_records(input_path=step1_output)
        else:
            raise

    if step_name == "step2":
        return validate_step2(generator_prompt=generator_prompt)

    if step_name == "step3":
        return run_human_labeling(generator_prompt=generator_prompt)

    if step_name == "step4":
        return run_llm_judge(
            client=client,
            generator_prompt=generator_prompt,
            prompt_name=judge_prompt,
        )

    if step_name not in {"all", ""}:
        raise ValueError(f"Unsupported pipeline step: {step!r}")

    validate_step2(generator_prompt=generator_prompt)
    run_human_labeling(generator_prompt=generator_prompt)
    run_llm_judge(
        client=client,
        generator_prompt=generator_prompt,
        prompt_name=judge_prompt,
    )
    return run_step5_analysis(generator_prompt=generator_prompt, judge_prompt=judge_prompt)


if __name__ == "__main__":
    args = parse_args()
    main(
        step=args.step,
        generator_prompt=args.generator_prompt,
        judge_prompt=args.judge_prompt,
        items_per_category=args.items_per_category,
    )

import datetime
import json
import os
import sys
from pathlib import Path

import instructor
from pydantic import BaseModel, Field

from config import output_path

DEFAULT_STEP1_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts" / "step1_generator"


def load_step1_prompt(prompt_name: str | None = None, prompts_dir: str | Path | None = None) -> str:
    if prompt_name is None:
        prompt_name = "default_generator_prompt.txt"
    if not prompt_name.endswith(".txt"):
        prompt_name = f"{prompt_name}.txt"

    prompt_root = Path(prompts_dir) if prompts_dir is not None else DEFAULT_STEP1_PROMPTS_DIR
    prompt_file = prompt_root / prompt_name

    if not prompt_file.exists():
        raise FileNotFoundError(f"Step 1 prompt file not found: {prompt_file.resolve()}")

    return prompt_file.read_text(encoding="utf-8").strip()


class QAItem(BaseModel):
    question: str = Field(min_length=1, description="A realistic question or request from a restaurant guest")
    answer: str = Field(min_length=1, description="A clear, guest-facing response with step-by-step guidance")
    dining_scenario: str = Field(description="The specific situation being addressed (e.g. \"guest with tree-nut allergy ordering pasta\")")
    menu_items: list[str] = Field(min_length=1, description="Dishes or drinks a typical mid-range restaurant would realistically offer")
    service_steps: list[str] = Field(min_length=3, description="Ordered, numbered actions the waiter takes (or walks the guest through) to fulfill the request")
    safety_info: str = Field(min_length=1, description="Relevant allergen, dietary, food-safety, or alcohol-service warnings and precautions")
    tips: list[str] = Field(min_length=1, description="Practical insider tips that make the meal better or the request go more smoothly")


class QADataset(BaseModel):
    qa_pairs: list[QAItem]


def load_step1_records(input_path: str = str(output_path("step1_generated_qa.json"))):
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Step 1 output file not found: {input_file.resolve()}")

    raw_records = json.loads(input_file.read_text(encoding="utf-8"))
    loaded_records = []

    for idx, item in enumerate(raw_records):
        metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
        qa_payload = {
            "question": item["question"],
            "answer": item["answer"],
            "dining_scenario": item["dining_scenario"],
            "menu_items": item["menu_items"],
            "service_steps": item["service_steps"],
            "safety_info": item["safety_info"],
            "tips": item["tips"],
        }
        qa_item = QAItem.model_validate(qa_payload)
        loaded_records.append(
            {
                "qa_item": qa_item,
                "trace_id": f"qa_{idx + 1:03d}",
                "source_file": str(input_file),
                "prompt_variant": metadata.get("prompt_variant", "loaded_from_step1_json"),
                "category_name": metadata.get("category_name") or item.get("category_name"),
                "category_description": metadata.get("category_description") or item.get("category_description"),
                "timestamp": metadata.get("timestamp") or item.get("timestamp"),
                "model_name": metadata.get("model_name") or item.get("model_name"),
                "raw_llm_response_json": metadata.get("raw_llm_response_json") or item.get("raw_llm_response_json"),
                "metadata": metadata,
            }
        )

    return loaded_records


def save_generated_qa_json(all_generated_qa_records, output_path: str = str(output_path("step1_generated_qa.json"))):
    output_records = []
    for record in all_generated_qa_records:
        qa_item = record["qa_item"]
        metadata = {
            "prompt_variant": record.get("prompt_variant"),
            "category_name": record.get("category_name"),
            "category_description": record.get("category_description"),
            "timestamp": record.get("timestamp"),
            "model_name": record.get("model_name"),
            "raw_llm_response_json": record.get("raw_llm_response_json"),
        }
        output_records.append(
            {
                "metadata": metadata,
                "question": qa_item.question,
                "answer": qa_item.answer,
                "dining_scenario": qa_item.dining_scenario,
                "menu_items": list(qa_item.menu_items),
                "service_steps": list(qa_item.service_steps),
                "safety_info": qa_item.safety_info,
                "tips": list(qa_item.tips),
            }
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_records, indent=2), encoding="utf-8")
    print(f"Saved generated Q&A items to: {output_file.resolve()}")
    return str(output_file)


def generate_step1(
    client,
    MODEL_NAME,
    categories,
    prompt: str | None = None,
    items_per_category=10,
    output_path: str = str(output_path("step1_generated_qa.json")),
    force_regenerate: bool = False,
    prompt_name: str | None = None,
    prompts_dir: str | Path | None = None,
):
    if prompt_name is not None:
        prompt = load_step1_prompt(prompt_name=prompt_name, prompts_dir=prompts_dir)
    elif prompt is None:
        prompt = load_step1_prompt(prompt_name=None, prompts_dir=prompts_dir)
    output_file = Path(output_path)
    if output_file.exists() and not force_regenerate:
        is_runtime_interactive = (
            sys.stdin is not None
            and hasattr(sys.stdin, "isatty")
            and sys.stdin.isatty()
        ) or "PYTEST_CURRENT_TEST" in os.environ

        if is_runtime_interactive:
            user_choice = input(
                f"Step 1 output file already exists at {output_file.resolve()}. Regenerate it? [y/N]: "
            ).strip().lower()
            if user_choice not in {"y", "yes"}:
                print("Step 1 generation cancelled. Existing JSON file was kept.")
                return []
        else:
            print(
                f"Step 1 output file already exists at {output_file.resolve()}. "
                "Non-interactive session detected; regenerating without prompting."
            )
            force_regenerate = True

    if not categories:
        print("No categories provided for Step 1 generation.")
        return []

    # Enable `instructor` mode for the OpenAI client
    # This patches the client to automatically parse the response into the specified Pydantic model.
    patched_client = instructor.patch(client)

    print("\nCategories:")
    for idx, category in enumerate(categories, start=1):
        print(f"{idx}. {category['name']}: {category['description']}")
    print(f"\nItems per category: {items_per_category}")
    print(f"\nGenerating {items_per_category * len(categories)} Q&A pairs across {len(categories)} categories...\n")

    all_generated_qa_records = []

    try:
        for idx, category in enumerate(categories, start=1):
            category_name = category["name"]
            category_description = category["description"]
            category_prompt = str(prompt)
            prompt_variant = Path(prompt_name).stem if prompt_name else "default"
            replacements = {
                "{category}": category_name,
                "{category_name}": category_name,
                "{category_description}": category_description,
                "{items_per_category}": str(items_per_category),
                "{items_qty}": str(items_per_category),
                "{categories}": f"{idx}. {category_name}: {category_description}",
            }
            for placeholder, value in replacements.items():
                category_prompt = category_prompt.replace(placeholder, value)

            print(f"Category {idx}/{len(categories)}: {category_name}")
            print(f"Generating {items_per_category} Q&A pairs for this category...\n")

            response_pydantic_model = patched_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that generates synthetic Q&A pairs for restaurant customer-waiter interactions based on a given schema.",
                    },
                    {"role": "user", "content": category_prompt},
                ],
                response_model=QADataset,
                max_retries=3,
            )

            for qa_index, qa_item in enumerate(response_pydantic_model.qa_pairs, start=1):
                current_timestamp = datetime.datetime.now().isoformat()
                record = {
                    "qa_item": qa_item,
                    "prompt_variant": prompt_variant,
                    "category_name": category_name,
                    "category_description": category_description,
                    "timestamp": current_timestamp,
                    "model_name": MODEL_NAME,
                    "raw_llm_response_json": response_pydantic_model.model_dump_json(),
                }
                all_generated_qa_records.append(record)

                print(f"  Pair {len(all_generated_qa_records)}:")
                print(f"    Question: {qa_item.question}")
                print(f"    Answer: {qa_item.answer[:200]}...")
                print(f"    Dining Scenario: {qa_item.dining_scenario}")
                print(f"    Menu Items: {', '.join(qa_item.menu_items)}")
                print(f"    Service Steps: {', '.join(qa_item.service_steps)}")
                print(f"    Safety Info: {qa_item.safety_info[:100]}...")
                print(f"    Tips: {', '.join(qa_item.tips)}")
                print(f"    --- Metadata ---")
                print(f"    Timestamp: {current_timestamp}")
                print(f"    Model Name: {MODEL_NAME}")
                print(f"    Category: {category_name}")
                print(f"    Raw LLM Response (snippet): {record['raw_llm_response_json'][:100]}...")
                print(f"    Prompt Variant (full string stored, hash for display): {hash(category_prompt)}")
                print("\n")

            save_generated_qa_json(all_generated_qa_records, output_path=str(output_file))

        print(f"\nTotal generated Q&A records across all categories: {len(all_generated_qa_records)}")
        save_generated_qa_json(all_generated_qa_records, output_path=str(output_file))
        return all_generated_qa_records

    except Exception as e:
        print(f"❌ Error generating Q&A pairs for category: {e}")
        return []

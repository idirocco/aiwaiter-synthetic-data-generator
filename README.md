# AI Waiter Synthetic Data Generator

This project creates synthetic restaurant customer/waiter Q&A data for an AI waiter assistant. It uses the OpenRouter OpenAI-compatible API to generate realistic dining-service interactions, filters them for quality, and then evaluates them with both human labeling and an LLM-as-judge workflow.

## Pipeline overview

The project is orchestrated by `pipeline.py` and runs in the following stages:

1. Step 1: Generation
   - Generates Q&A items for each configured restaurant category.
   - Saves the output to `output/step1_generated_qa.json`.
   - Each item includes a top-level `metadata` object with fields such as:
     - `category_name`
     - `category_description`
     - `timestamp`
     - `model_name`
     - `prompt_variant`
     - `raw_llm_response_json`

2. Step 2: Validation
   - Runs lightweight quality checks on each generated item.
   - Deduplicates repeated questions.
   - Checks category distribution thresholds.
   - This step runs in memory and does not write a separate output file.

3. Step 3: Human labeling
   - Prompts a reviewer to label each item across six quality dimensions.
   - Saves results to `output/step3_human_labels.json`.

4. Step 4: LLM-as-judge
   - Uses an independent model to score the same six dimensions.
   - Saves results to `output/step4_llm_judge_labels.json`.

## Setup

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Set the OpenRouter API key.

You can export it directly in your shell:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Or create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_key_here
```

## Running the pipeline

Run the full pipeline:

```bash
python3 pipeline.py
```

Run a specific stage:

```bash
python3 pipeline.py step1
python3 pipeline.py step2
python3 pipeline.py step3
python3 pipeline.py step4
```

Note: Step 1 asks for confirmation before regenerating an existing `output/step1_generated_qa.json` file.

## Project structure

- `pipeline.py` — orchestrates the pipeline and dispatches each stage
- `config.py` — shared model config, categories, prompts, and client setup
- `step1_generation.py` — generates Q&A pairs and writes `output/step1_generated_qa.json`
- `step2_validation.py` — validation, deduplication, and category-distribution checks
- `step3_human_labeling.py` — interactive human review tool and label export
- `step4_llm_as_judge.py` — LLM-based quality scoring and export
- `quality_dimensions.py` — Six quality dimensions used across labeling and judging
- `startup_checks.py` — dependency and environment validation before pipeline startup
- `requirements.txt` — project dependencies
- `.gitignore` — excludes local environment and generated artifacts

## Output files

- `output/step1_generated_qa.json` — generated Q&A dataset with metadata per item
- `output/step3_human_labels.json` — human-labeled results
- `output/step4_llm_judge_labels.json` — LLM judge results

## Example Step 1 JSON item

```json
[
  {
    "metadata": {
      "prompt_variant": "...",
      "category_name": "Dietary Restrictions & Allergies",
      "category_description": "food allergies, vegetarian/vegan, gluten-free, halal/kosher, low-sodium",
      "timestamp": "2026-09-14T12:00:00.000000",
      "model_name": "anthropic/claude-sonnet-4.6",
      "raw_llm_response_json": "{...full raw model payload...}"
    },
    "question": "I have a severe tree-nut allergy and I'm craving pasta tonight — what can I safely order?",
    "answer": "I would recommend the spaghetti pomodoro or grilled chicken penne with tomato-garlic sauce instead of pesto...",
    "dining_scenario": "Guest with severe tree-nut allergy ordering pasta",
    "menu_items": [
      "Spaghetti Pomodoro",
      "Grilled Chicken Penne with tomato-garlic sauce",
      "Caesar salad without croutons"
    ],
    "service_steps": [
      "I’ll confirm the allergy details and severity first.",
      "I’ll steer you away from pesto and desserts made with nuts.",
      "I’ll mark the ticket with an allergy flag and tell the kitchen directly.",
      "I’ll ensure the pasta is prepared in a freshly cleaned pan."
    ],
    "safety_info": "The house pesto contains walnuts, so I’d avoid that option and ask the kitchen to prepare your pasta in a freshly cleaned pan with clean utensils.",
    "tips": [
      "Ask the kitchen to confirm the allergy protocol before the dish leaves the pass.",
      "Order the salad without croutons if you want to avoid cross-contact from shared baking trays."
    ]
  }
]
```

## Notes

- The application exits early if required Python packages are missing.
- The application exits early if `OPENROUTER_API_KEY` is not set.
- Step 2 expects `record["metadata"]["category_name"]` to exist, so Step 1 keeps category metadata in the on-disk JSON payload.

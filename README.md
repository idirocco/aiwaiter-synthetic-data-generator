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

5. Step 5: Analysis and visualization
   - Merges Step 1 Q&A with Step 3 human labels and Step 4 LLM-as-judge labels.
   - Computes per-dimension pass rates and human vs. LLM agreement.
   - Aggregates metrics by restaurant category and prompt variant.
   - Writes charts to `visualizations/`.

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
python3 pipeline.py step5
```

Use custom prompt files by passing the prompt name as a positional argument. The pipeline expects arguments in this order:

```bash
python3 pipeline.py <step> [step1_prompt_name] [step4_prompt_name]
```

Examples:

```bash
# Step 1 with a custom generator prompt
python3 pipeline.py step1 my_custom_prompt

# Step 4 with a custom judge prompt
python3 pipeline.py step4 "" my_custom_judge_prompt

# Step 1 + Step 4 together with custom prompts
python3 pipeline.py all my_custom_prompt my_custom_judge_prompt
```

Prompt names are resolved from the prompt folders under `prompts/step1_generator/` and `prompts/step4_judge/`. The `.txt` extension is optional; for example, `my_custom_prompt` resolves to `my_custom_prompt.txt`.

Note: Step 1 asks for confirmation before regenerating an existing `output/step1_generated_qa.json` file.

## Project structure

- `pipeline.py` — orchestrates the pipeline and dispatches each stage
- `config.py` — shared model config, categories, prompts, and client setup
- `step1_generation.py` — generates Q&A pairs and writes `output/step1_generated_qa.json`
- `step2_validation.py` — validation, deduplication, and category-distribution checks
- `step3_human_labeling.py` — interactive human review tool and label export
- `step4_llm_as_judge.py` — LLM-based quality scoring and export
- `step5_analysis_visualization.py` — merges labels, computes metrics, and writes charts
- `quality_dimensions.py` — Six quality dimensions used across labeling and judging
- `startup_checks.py` — dependency and environment validation before pipeline startup
- `requirements.txt` — project dependencies
- `.gitignore` — excludes local environment and generated artifacts

## Output files

- `output/step1_generated_qa.json` — generated Q&A dataset with metadata per item
- `output/step3_human_labels.json` — human-labeled results
- `output/step4_llm_judge_labels.json` — LLM judge results
- `visualizations/step5_segment_heatmap.png` — LLM pass rates by category and quality dimension
- `visualizations/step5_dimension_pass_rates.png` — human vs. LLM pass rates by dimension
- `visualizations/step5_human_llm_agreement.png` — agreement rates by dimension
- `visualizations/step5_category_distribution.png` — generated category counts vs. a balanced benchmark
- `visualizations/step5_before_after_per_dimension.png` — mean pass rates by prompt variant

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
- Step 5 reads `output/step1_generated_qa.json`, `output/step3_human_labels.json`, and `output/step4_llm_judge_labels.json`. Missing label files are treated as empty, so pass/agreement rates will be zero until those steps have been run.

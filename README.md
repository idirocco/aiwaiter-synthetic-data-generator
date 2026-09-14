# AI Waiter Synthetic Data Generator

This project generates synthetic restaurant customer/waiter Q&A data using OpenRouter and the OpenAI-compatible API.

## Setup

1. Create and activate a virtual environment (optional but recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

3. Set your OpenRouter API key.

You can export it in your shell:

```bash
export OPENROUTER_API_KEY=your_key_here
```

Or copy the example env file and edit it:

```bash
cp .env.example .env
```

Then update `.env`:

```env
OPENROUTER_API_KEY=your_key_here
```

## Run

```bash
python3 pipeline.py
```

## Project structure

- `pipeline.py` — main orchestrator
- `config.py` — shared config and startup settings
- `step1_generation.py` — generation logic
- `step2_validation.py` — validation logic
- `startup_checks.py` — dependency and environment checks
- `.env.example` — sample environment file
- `requirements.txt` — Python dependencies

## Notes

- The script exits early if required Python packages are missing.
- The script also exits early if `OPENROUTER_API_KEY` is not set.

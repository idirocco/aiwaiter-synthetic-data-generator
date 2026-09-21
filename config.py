import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"
OUTPUT_DIR = Path("output")

def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR

def output_path(filename: str) -> Path:
    ensure_output_dir()
    return OUTPUT_DIR / filename


def print_startup_banner():
    print("✅ OpenRouter API configured!")
    print(f"   Model: {MODEL_NAME}")
    print()


def print_modules_loaded():
    print("✅ Generation and validation modules loaded.")


CATEGORIES = [
    {
        "name": "Menu Guidance & Recommendations",
        "description": "dish explanations, portion sizes, spice levels, chef's specials, what is popular here",
    },
    {
        "name": "Dietary Restrictions & Allergies",
        "description": "food allergies, vegetarian/vegan, gluten-free, halal/kosher, low-sodium",
    },
    {
        "name": "Beverages & Pairings",
        "description": "wine and beer pairings, cocktails, non-alcoholic options, coffee and dessert drinks (responsible alcohol service only)",
    },
    {
        "name": "Order Customization & Special Requests",
        "description": "substitutions, cooking temperatures, kids' portions, shared plates, course pacing, special occasions",
    },
    {
        "name": "Service Recovery & Complaints",
        "description": "wrong, undercooked, or cold dishes; foreign objects in food; spills and broken glassware; dishes sent back",
    },
]

PROMPT = """You are an expert restaurant waiter and synthetic data generator.

Your task is to generate a realistic restaurant customer-waiter dataset for training and evaluation.

Return valid JSON only, with this exact top-level structure:
{
  "qa_pairs": [
    {
      "question": "string",
      "answer": "string",
      "dining_scenario": "string",
      "menu_items": ["string", "string"],
      "service_steps": ["string", "string"],
      "safety_info": "string",
      "tips": ["string", "string"]
    }
  ]
}

Important: the model must produce a JSON object with a single key named "qa_pairs". Do not return a bare array, do not return markdown, and do not include commentary outside the JSON.

Dataset requirements:
- Generate {items_qty} items for the category below.
- Each item must reflect a realistic restaurant interaction between a guest and a waiter.
- Each answer must be a coherent guest-facing narrative, not a list stitched together. It should feel like a waiter speaking directly to the guest.
- Each answer should be approximately 700–1,300 characters long.
- The content should feel specific to a mid-range, full-service restaurant and realistic for a modern dining room.

Per-field rules:
- question: a realistic customer question or request, one or two sentences, naturally phrased.
- answer: a clear, guest-facing response with step-by-step guidance, scenario-specific details, and a natural narrative flow.
- dining_scenario: a precise scenario label such as "guest with tree-nut allergy ordering pasta".
- menu_items: realistic dishes or drinks a typical mid-range restaurant would offer, each under $50.
- service_steps: ordered, concrete actions the waiter takes or asks the guest to take. Include specifics such as quantities, timing, temperatures, or observable indicators when relevant.
- safety_info: specific to the actual hazards of the order or scenario. Mention precisely what risk exists and what precaution is taken.
- tips: non-obvious, practical advice that makes the guest experience smoother or improves the meal.

Hard constraints:
- Do not use generic safety text such as "please let us know about allergies" unless it is followed by specific hazard-aware details.
- Do not invent unrealistic or impossible menu items.
- Do not use luxury-only ingredients or restaurant formats inconsistent with a typical full-service mid-range restaurant.
- Do not repeat the same question template across many items.
- Make sure the menu_items, service_steps, safety_info, and tips are all tightly connected to the same scenario.
- Use varied dining situations.
- Keep the JSON valid and parseable.

Category: 
{category_name}: {category_description}

Quality bar before finalizing:
- The question sounds real.
- The answer is narrative and specific.
- The service steps are ordered and concrete.
- The safety_info reflects the real risk of this request and not a generic alert.
- The menu items are realistic and relevant.
- The tips add real value.

Example of the required output format:
{
  "qa_pairs": [
    {
      "question": "I have a severe tree-nut allergy and I'm really craving pasta tonight — what can I safely order?",
      "answer": "I'd steer you away from the basil pesto and the tiramisu because the pesto contains walnuts and the dessert is made on shared equipment with hazelnuts. A safer choice would be the Spaghetti Pomodoro or the Grilled Chicken Penne with tomato-garlic sauce, and I'd ask the kitchen to prepare it in a freshly cleaned pan with clean utensils. I'll also flag the ticket as an allergy order and tell the chef in person before the food is started so nothing is missed. Once the plate lands, I'll bring it out separately from the rest of the table and check that it was prepared under our allergy protocol before I leave it with you.",
      "dining_scenario": "guest with severe tree-nut allergy ordering pasta",
      "menu_items": ["Spaghetti Pomodoro", "Grilled Chicken Penne with tomato-garlic sauce", "Caesar salad without croutons"],
      "service_steps": [
        "Confirm the exact allergen and severity with the guest before recommending a dish.",
        "Recommend a safe pasta and avoid the pesto and dessert items that contain nuts.",
        "Flag the order with an allergy note and communicate directly with the chef.",
        "Ask the kitchen to cook the dish in a freshly cleaned pan with clean utensils.",
        "Bring the plate out separately and verify it was prepared under the allergy protocol."
      ],
      "safety_info": "The house pesto contains walnuts and the tiramisu is prepared on shared equipment with hazelnuts, so cross-contact is a real risk. The kitchen should use a freshly cleaned pan and clean utensils, and the order should be clearly labeled as an allergy request.",
      "tips": [
        "Ask whether the guest is allergic to tree nuts only or also to peanuts, since some sauces and desserts can trigger cross-reactions.",
        "If the guest wants a salad, order it without croutons, since the croutons are baked near almond biscotti in the same prep area."
      ]
    }
  ]
}

Now generate the dataset.
"""


def get_client():
    load_dotenv()

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to a .env file or export it in your shell."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

import os

from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "anthropic/claude-sonnet-4.6"


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

PROMPT = """Act as an Expert Waiter in a restaurant with many years of experience.

Create a dataset with diverse and high-quality synthetic Q&A pairs to reflect customer - waiter interactions that might happen while the customer orders.

Each Q&A pair/item must conform to the following 7-field schema:
| Field | Type | Description |
|---|---|---|
| `question` | string | A realistic question or request from a restaurant guest |
| `answer` | string | A clear, guest-facing response with step-by-step guidance |
| `dining_scenario` | string | The specific situation being addressed (e.g. \"guest with tree-nut allergy ordering pasta\") |
| `menu_items` | list of strings | Dishes or drinks a typical mid-range restaurant would realistically offer |
| `service_steps` | list of strings | Ordered, numbered actions the waiter takes (or walks the guest through) to fulfill the request |
| `safety_info` | string | Relevant allergen, dietary, food-safety, or alcohol-service warnings and precautions |
| `tips` | list of strings | Practical insider tips that make the meal better or the request go more smoothly |

Q&A Dataset details:
- Every item has a substantial, narrative-style answer (typically 700–1,300 characters) that weaves together the menu items, service steps, safety/allergen warnings, and tips into a coherent, guest-facing response, not just a list of fields stitched together
- Safety information is always specific to the hazards of the particular order or situation (e.g., \"The house pesto is made with walnuts, so I'll flag your ticket as a tree-nut allergy and the kitchen will cook your pasta in a freshly cleaned pan\", not \"Please let us know about any allergies\")
- Tips provide non-obvious, request-specific advice that a first-time guest would not know
- Menu items listed are dishes and drinks a typical mid-range, full-service restaurant would realistically offer, each priced under $50
- Service steps are concrete and specific enough to follow without guessing. They include quantities, timings, or observable indicators where relevant (e.g., \"check back within 2 minutes of the plates landing\", \"medium means a warm pink center, about 140°F / 60°C\")

Generate {items_per_category} items that belong to the category {category}.

Example Q&A item:
`
{{
  "question": "I have a severe tree-nut allergy and I'm really craving pasta tonight — what can I safely order?",
  "answer": "Menu items: Spaghetti Pomodoro, Grilled Chicken Penne with tomato-garlic sauce, Caesar salad without croutons\n\n1. I'll confirm the details of your allergy first — tree nuts only, or peanuts as well — and how severe your reactions usually are\n2. I'll steer you away from the Basil Pesto Linguine and the Tiramisu, since our pesto is made with walnuts and the tiramisu is topped with crushed hazelnuts\n3. I'd recommend the Spaghetti Pomodoro, or the Grilled Chicken Penne served with tomato-garlic sauce instead of pesto\n4. I'll mark your ticket with an ALLERGY flag and tell the chef in person, rather than relying on the ticket alone\n5. I'll bring your plate out separately from the rest of the table's food and confirm it was prepared under our allergy protocol as I set it down\n\nSafety warning: Our pesto contains walnuts and our desserts are made on shared equipment with almonds and hazelnuts, so cross-contact is possible. Your pasta will be cooked in a freshly cleaned pan with clean utensils, but I can't guarantee a completely nut-free kitchen — I'll ask the chef to come to the table and confirm the preparation before you order.\n\nAdditional advice: Our Caesar dressing is made in-house without nuts, but the croutons are baked on the same trays as the almond biscotti — order the salad without croutons to be safe.",
  "dining_scenario": "Guest with severe tree-nut allergy ordering pasta",
  "menu_items": [
    "Spaghetti Pomodoro",
    "Grilled Chicken Penne with tomato-garlic sauce",
    "Caesar salad without croutons"
  ],
  "service_steps": [
    "I'll confirm the details of your allergy first — tree nuts only, or peanuts as well — and how severe your reactions usually are",
    "I'll steer you away from the Basil Pesto Linguine and the Tiramisu, since our pesto is made with walnuts and the tiramisu is topped with crushed hazelnuts",
    "I'd recommend the Spaghetti Pomodoro, or the Grilled Chicken Penne served with tomato-garlic sauce instead of pesto",
    "I'll mark your ticket with an ALLERGY flag and tell the chef in person, rather than relying on the ticket alone",
    "I'll bring your plate out separately from the rest of the table's food and confirm it was prepared under our allergy protocol as I set it down"
  ],
  "safety_info": "Our pesto contains walnuts and our desserts are made on shared equipment with almonds and hazelnuts, so cross-contact is possible. Your pasta will be cooked in a freshly cleaned pan with clean utensils, but I can't guarantee a completely nut-free kitchen — I'll ask the chef to come to the table and confirm the preparation before you order.",
  "tips": [
    "Our Caesar dressing is made in-house without nuts, but the croutons are baked on the same trays as the almond biscotti — order the salad without croutons to be safe."
  ]
}}
`
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

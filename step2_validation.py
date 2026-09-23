import re
from collections import defaultdict

from step1_generation import build_step1_output_path, load_step1_records

MIN_SAFETY_INFO_LENGTH = 30
GENERIC_PHRASE_BLOCKLIST = [
    "please let us know about any allergies",
    "ask your server for details",
    "we cannot guarantee a completely allergen-free environment",
    "enjoy your meal",
]
MENU_ITEM_BLOCKLIST = [
    "water",
    "soda",
    "coke",
    "pepsi",
    "diet coke",
]


def check_safety_info_length(qa_item):
    return len(qa_item.safety_info) >= MIN_SAFETY_INFO_LENGTH


def contains_generic_phrase(text):
    normalized_text = text.lower()
    for phrase in GENERIC_PHRASE_BLOCKLIST:
        if phrase in normalized_text:
            return True
    return False


def contains_blocked_menu_item(qa_item):
    for item in qa_item.menu_items:
        if item.lower() in MENU_ITEM_BLOCKLIST:
            return True
    return False


def normalize_question(question):
    return re.sub(r"[^a-zA-Z0-9]", "", question).lower()


def validate_step2(generator_prompt: str | None = None, input_path: str | None = None):
    resolved_path = input_path or build_step1_output_path(generator_prompt)
    print(f"\n--- Loading Step 1 records from {resolved_path} ---")
    all_generated_qa_records = load_step1_records(input_path=resolved_path)

    print("\n--- Running Per-Item Lightweight Pre-Checks ---")

    passed_per_item_checks = []
    failed_per_item_checks_count = 0

    for record in all_generated_qa_records:
        qa_item = record["qa_item"]
        passed_all_heuristics = True

        if not check_safety_info_length(qa_item):
            passed_all_heuristics = False
        if contains_generic_phrase(qa_item.safety_info) or contains_generic_phrase(qa_item.answer):
            passed_all_heuristics = False
        if contains_blocked_menu_item(qa_item):
            passed_all_heuristics = False

        if passed_all_heuristics:
            passed_per_item_checks.append(record)
        else:
            failed_per_item_checks_count += 1

    print(f"\n   Initial records (after Pydantic validation): {len(all_generated_qa_records)}")
    print(f"   Records failing lightweight per-item checks: {failed_per_item_checks_count}")
    print(f"   Records passing per-item checks: {len(passed_per_item_checks)}")

    all_generated_qa_records = passed_per_item_checks
    print("\n--- Per-Item Lightweight Pre-Checks Completed ---")

    print("\n--- Running Batch Level Checks ---")

    seen_normalized_questions = set()
    deduplicated_qa_records = []

    initial_record_count = len(all_generated_qa_records)
    for record in all_generated_qa_records:
        qa_item = record["qa_item"]
        normalized_q = normalize_question(qa_item.question)
        if normalized_q not in seen_normalized_questions:
            seen_normalized_questions.add(normalized_q)
            deduplicated_qa_records.append(record)

    duplicate_count = initial_record_count - len(deduplicated_qa_records)
    print("\n1. Deduplication Check:")
    print(f"   Initial records: {initial_record_count}")
    print(f"   Duplicates found and removed: {duplicate_count}")
    print(f"   Records after deduplication: {len(deduplicated_qa_records)}")

    all_generated_qa_records = deduplicated_qa_records

    category_counts = defaultdict(int)
    for record in all_generated_qa_records:
        metadata = record.get("metadata") or {}
        category_name = metadata.get("category_name") or record.get("category_name")
        if category_name is None:
            raise KeyError("Missing category metadata: expected record['metadata']['category_name']")
        category_counts[category_name] += 1

    total_records_after_dedup = len(all_generated_qa_records)
    min_threshold_percent = 20
    min_threshold_count = (total_records_after_dedup * min_threshold_percent) / 100

    print(f"\n2. Category Distribution Check (Threshold: >= {min_threshold_percent}% or {min_threshold_count:.2f} items per category):")
    alert_on_distribution = False
    for cat_name, count in category_counts.items():
        percentage = (count / total_records_after_dedup) * 100 if total_records_after_dedup > 0 else 0
        if percentage < min_threshold_percent:
            print(f"   ❌ Category '{cat_name}': {count} items ({percentage:.2f}%) - BELOW threshold!")
            alert_on_distribution = True
        else:
            print(f"   ✅ Category '{cat_name}': {count} items ({percentage:.2f}%) - Meets threshold.")

    if not alert_on_distribution:
        print("   All categories meet the minimum distribution threshold.")
    else:
        print("   ⚠️ One or more categories are below the minimum distribution threshold. Consider adjusting prompt/model.")

    print("\n--- Batch Level Checks Completed ---")
    return all_generated_qa_records

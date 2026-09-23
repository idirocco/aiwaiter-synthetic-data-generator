# Judge Calibration (iterate the LLM-as-Judge prompt)
# Compute per-dim human/LLM agreement from Step 5.
# Pick the worst dimension — the one with the lowest agreement.
# Revise the judge prompt for that dimension: tighten the rubric, add pass/fail examples, disambiguate edge cases.
# Re-run Step 4 on the same items; recompute agreement.
# Stop when every dimension reaches ≥ 80% agreement. Do not touch the generator until this holds.
# Success criteria: Human/LLM agreement ≥ 80% on every dimension.
## Iteration 0: Generator & Judge prompts baseline
- **Date**: 2026-09-14
- **Change**: Baseline
- **Hypothesis**: -
- **Result**: -
- **Decision**: Keep
- **Next step**: Iterate and analyze results

## Iteration 1: New judge model `openai/gpt-oss-20b` failed schema validation
- **Date**: 2026-09-25
- **Change**: Ran Step 4 with the new judge model `openai/gpt-oss-20b` (`LLM_JUDGE_MODEL_NAME` in `config.py`). Command: `python3 pipeline.py step4 --generator-prompt=v1 --judge-prompt=v1`. Generator stayed `meta-llama/llama-3.1-8b-instruct`. Step 1 records were reused.
- **Hypothesis**: The new 20B judge would score the existing items and write Step 4 labels.
- **Result**: Failed on the first item. `openai/gpt-oss-20b` returned each dimension as `{"pass": true}` or `{"pass": false}` with no rationale. `DimensionVerdict` required `pass_` and `rationale`, so Instructor raised 12 validation errors and exhausted 3 retries. No Step 4 file was written.
- **Decision**: Blocked. The new judge model is in use, and its verdict shape does not match the schema.
- **Next step**: Accept `pass` and a missing rationale, then re-run Step 4 with this same judge model.

## Iteration 2: Judge prompt `default` → `v1`
- **Date**: 2026-09-25
- **Change**: The default prompt told the judge to be strict and to fail anything generic, vague, or short on detail, with no per-dimension examples. v1 scores only the item, and it does not fail for tone, length, or an extra generic sentence when a specific relevant detail is already present. It adds pass criteria, fail examples, and edge cases for `safety_specificity` (score `safety_info`: a named allergen, cross-contact, doneness temperature, spice/irritant, or a service problem plus the fix) and `tip_usefulness` (at least one practical action tied to this order; one specific tip saves a second generic tip). The other four dimensions get a one-line pass rule.
- **Hypothesis**: The default prompt was failing items humans passed, especially on safety and tips, because a generic sentence outweighed a specific detail. Explicit rubrics for those two dimensions should raise agreement.
- **Result**: Mean agreement across the six dimensions rose from 70.2% to 87.7% (+17.5 pp). The two dimensions with new rubrics moved the most.

  | Dimension | Default | v1 | Change |
  | --- | ---: | ---: | ---: |
  | safety_specificity | 9.5% | 59.5% | +50.0 pp |
  | tip_usefulness | 42.9% | 88.1% | +45.2 pp |
  | menu_realism | 85.7% | 100% | +14.3 pp |
  | scope_appropriateness | 95.2% | 100% | +4.8 pp |
  | context_clarity | 97.6% | 100% | +2.4 pp |
  | answer_completeness | 90.5% | 78.6% | −11.9 pp |

  `context_clarity`, `menu_realism`, and `scope_appropriateness` now match the human labels on every item. `answer_completeness` is the only drop: the judge fails more items humans passed.
- **Decision**: Keep v1. Safety and tip agreement were the gap, and both closed by about 45–50 points.
- **Next step**: Review the `answer_completeness` disagreements and tighten that dimension’s rule so the safety and tip gains stay.

## Iteration 3: Judge prompt `v1` → `v2`
- **Date**: 2026-09-25
- **Change**: v1 scored `answer_completeness` with one line: the answer had to address the question with this scenario’s menu, steps, and constraints. v2 gives that dimension its own rubric. Pass when the answer is a direct reply (a dish, a choice, a modification, or the next step). Fail when the reply ignores the question, or is only “Sure, we can help with that” with no dish, choice, or next step. Edge cases: do not require the answer to repeat menu items, service steps, or constraints (those have their own fields); a dish that is not on the menu list still passes when it answers the question; a short reply passes when it still answers. Safety, tip, and the other three dimensions are unchanged.
- **Hypothesis**: The v1 completeness rule was failing items humans passed because it demanded menu, steps, and constraints inside the answer. A direct-reply rule should raise completeness agreement and leave the safety and tip gains in place.
- **Result**: Mean agreement across the six dimensions rose from 87.7% to 91.7% (+4.0 pp). `answer_completeness` moved from 78.6% to 97.6% (+19.0 pp), 33 of 42 items to 41 of 42. Humans passed every item on every dimension in both runs, so agreement equals the LLM pass rate.

  | Dimension | v1 | v2 | Change |
  | --- | ---: | ---: | ---: |
  | answer_completeness | 78.6% | 97.6% | +19.0 pp |
  | safety_specificity | 59.5% | 64.3% | +4.8 pp |
  | tip_usefulness | 88.1% | 88.1% | 0 |
  | menu_realism | 100% | 100% | 0 |
  | scope_appropriateness | 100% | 100% | 0 |
  | context_clarity | 100% | 100% | 0 |

  `context_clarity`, `menu_realism`, and `scope_appropriateness` stay at 100%. `tip_usefulness` is unchanged at 88.1% (37 of 42). `safety_specificity` rose 4.8 points (25 of 42 to 27 of 42) even though its rubric text did not change. It is now the largest remaining gap.
- **Decision**: Keep v2. Completeness agreement recovered without giving back the safety or tip gains from v1.
- **Next step**: Review the remaining `safety_specificity` disagreements. That dimension is still at 64.3%, with the judge failing items humans passed.
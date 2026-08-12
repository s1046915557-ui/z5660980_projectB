# Prompt log - Project A foundation

## What I wanted

Following Project A in FINS2026, Begin Project B, but must not directly copy
the old data processing chain. If news headlines become the model input, will the
data satisfy the Project B requirements?

## Prompt(s)

1. Identify which parts of my Project A ETL and feature code can be repurposed
   in Project B. Maintain the official data loader and test on full data.
2. Describe `tests/test_foundation.py` and determine if I need to retain it in
   the final project version.
3. Was Project A text grouped for the purpose of description? Does Project B need
   to keep each headline separate for VADER, and align the calendar without
   backdating news.

## What the assistant produced

Codex reviewed my Project A functions and transferred the relevant ETL and
return features to Project B. It left `src/data_access.py` intact. It also added
foundation tests for number of rows, keys, calendars, returns, raw text and
weekend headline alignment.

The initial design was still based on the grouped Project A text panel. Codex
revised `assemble_headline_panel()` function, so that Project B maintains one row
per cleaned headline.

## What was wrong or risky

Using the same approach to combine the text would result in a combination of
headlines before scoring using VADER. This would change the unit of analysis and
prevent headline-level validation.

The small CSV files from Project A were only submission samples and were not
usable as Project B model inputs. The import-only smoke test could not verify
the data contracts.

Ruff detected an unused `noqa` directive in the new test file. It was
superfluous.

## What I changed and why

I decided to selectively transfer the work and maintain the hosted data loader.
I also maintained 146,836 cleaned headlines in `text_raw`. Of those, 12,557 are
aligned forward to later equity trading days. The 6 headlines following the last
equity date in 2023 are aligned forward to 2 January 2024 and fall outside the
return sample.

I decided to keep `tests/test_foundation.py`, since it documents the border
between the reused Project A code and Project B pipeline. I ran the test and
confirmed 5 passes. Overall, there were 7 passes. Following the notification of
the superfluous directive, I removed it and ran Ruff successfully.

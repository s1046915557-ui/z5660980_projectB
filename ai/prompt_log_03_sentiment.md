# Prompt log - Validated sector sentiment

## What I wanted

I wanted to build the required NLTK VADER sector index and check whether the
general-language model handled finance headlines well enough. Any extension
had to remain separate from the course baseline.

## Prompt(s)

1. Score the unchanged `text_raw` headlines with NLTK VADER. Aggregate to
   ticker-day first, then equal-weight tickers inside each sector. Record days
   with no news and lag the tradable signal by one equity trading day.
2. VADER gave `Earnings beat expectations!` a compound score of zero. Use this
   as a diagnostic and suggest a small finance extension that leaves the
   original VADER model unchanged.
3. Create a development sample for my review. Freeze the finance rules before
   opening a separate holdout, then compare both models on that holdout.

## What the assistant produced

Codex implemented the unmodified NLTK VADER benchmark and a separate
finance-aware analyser in `src/sentiment.py`. The extension contains 23 finance
terms and one phrase rule. A test confirms that it cannot alter the benchmark
analyser.

The pipeline produced 60 development candidates and 30 locked holdout
candidates. It also saved model diagnostics, sector coverage and a past-only
expanding z-score with a 21-observation minimum.

## What was wrong or risky

VADER returned zero for finance phrases when an important term was missing from
its lexicon. Treating every zero as reliable neutral sentiment would hide that
coverage problem.

Changing the benchmark analyser would remove a reproducible comparison.
Selecting new terms after reading the holdout would contaminate the evaluation.

Directly averaging all sector headlines would give more weight to firms with
more news. Missing-news ticker-days also needed a visible treatment.

## What I changed and why

I kept the original VADER analyser as the benchmark and reviewed the finance
extension through the development sample. I chose neutral zero for ticker-days
with no news and kept a separate coverage flag. The sector index uses
ticker-equal weights, and every tradable signal is lagged by one trading day.

I left the finance rules unchanged after opening the locked holdout. Rule and
holdout hashes make later changes visible. Development accuracy increased from
56.7% to 68.3%. Locked-holdout accuracy increased from 56.7% to 60.0%. I used
the smaller holdout gain as the relevant result. The two sentiment test files
finished with 9 passes.

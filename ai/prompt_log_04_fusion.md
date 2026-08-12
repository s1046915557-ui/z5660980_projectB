# Prompt log - Sentiment fusion

## What I wanted

I wanted to know whether the sector index provided extra information for an
existing equity allocation. The analysis needed a simple benchmark, a fixed
holdout period and trading-cost evidence.

## Prompt(s)

1. Begin with a naive raw-sentiment tilt. Update equity weights only, keep
   the crypto sleeve as is and make sure that a zero tilt retrieves the base
   fund precisely.
2. The raw VADER values are mostly positive and the news coverage varies across
   sectors. Propose a bounded signal that addresses both problems and leaves
   the total equity and crypto sleeve weights unchanged.
3. Fix the rule and the 0.25 tilt before testing 2023. Audit the source, decision
   and effective dates. Then compare the performance of the base, naive and new
   rules after 0, 10 and 50 basis point costs.

## What the assistant produced

Codex implemented a naive raw-compound tilt and a Coverage-Aware Rank rule in
`src/fusion.py`. The new rule ranks ten sectors at every decision date, centres
the ranks within the [-1,1] range and shrinks signals for sectors with weak
lagged news coverage.

The code evaluates the rules for nine eligible equity-bearing base funds. It
generates returns, weights, rebalance records and a 720-row signal audit.

## What was wrong or risky

A raw sentiment level may respond to the positive baseline of VADER. Uneven news
coverage may make a sector with insufficient coverage look informative.

Using the rule or its strength on the whole 2021-2023 sample would blur the line
between model development and evaluation. Any changes to the total asset-class
sleeves would add another allocation decision to weaken the comparison.

Any small gross gain may be erased by turnover costs.

## What I changed and why

I chose the coverage-aware rank rule because it relies on relative sector
information and shrinks thin-coverage signals. I kept the total equity and crypto
sleeve weights unchanged relative to the base fund. The signal only redistributes
weights within the equity sleeve.

The audit needs `source_date < decision_date < effective_date`. I fixed the rule
and 0.25 strength before the 2023 holdout. For the primary Combined Two-stage
Sleeve Risk Parity fund, holdout Sharpe was 1.484 for the base and 1.500 for the
new rule. The gain was 0.016 before costs, 0.013 at 10 basis points and roughly
-0.001 at 50 basis points.

I kept the negative result at 50 basis points and did not retune the rule. It
provides a practical execution-cost limit for the signal. The fusion tests
finished with 16 passes, and the full suite had 44 passes at this stage.

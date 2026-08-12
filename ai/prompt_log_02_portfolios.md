# Prompt log - Out-of-sample funds

## What I wanted

I wanted a fund shelf that was easily comparable and truly out-of-sample. I
also wanted one additional fund with a well-justified rationale for mixing
equities and crypto.

## Prompt(s)

1. Implement four long-only approaches to the equity, crypto and combined
   universes. Start with the 2020 window and compute monthly expanding-window
   decisions from 2021 to 2023.
2. Check that monthly target weights are not reused every day but drift
   according to returns until the next rebalance.
3. The combined universe consists of 50 equities and 10 crypto assets with
   significantly different volatility. Propose a focused extension and test its
   turnover and performance under 0, 10 and 50 basis points of one-way
   transaction costs.

## What the assistant produced

Codex implemented equal weight, minimum variance, maximum Sharpe and risk
parity in `src/portfolios.py`. Each rebalance stores the training window,
decision date, effective date and holding period.

Codex proposed a custom Two-stage Sleeve Risk Parity fund. It starts with
balancing the risks in each asset-class sleeve, and then balances the risk
between the equity and crypto sleeves. That made the thirteenth fund.

## What was wrong or risky

Using the monthly target weights every day would result in an unreported daily
rebalance and lower turnover estimates.

A single 60-assets optimizer may be influenced by the 50-to-10 assets ratio
and volatility disparity between equities and crypto. The more complex
approach should have a purpose to be tested.

The first test of the transaction costs failed due to different pandas
frequency metadata in two equal date indexes. The return values and dates were
equal.

## What I changed and why

I decided to keep buy-and-hold drift within each month. The turnover is
calculated between the drifted live weight and the next target weight. I
also kept the separation of decision and effective dates, as well as the
future-shock test.

I decided to keep the Two-stage Sleeve Risk Parity approach because it helps
to solve the problem with cross-asset risk imbalance directly. The tests verify
long-only weights, full investment, equal ex-ante sleeve risk contributions and
look-ahead safety.

I added 0, 10 and 50 basis point one-way transaction cost cases. For the
pandas problem, I fixed only the assertion to ignore the frequency metadata
while keeping the dates and values exactly equal. The portfolio tests finished
with 12 passes, and the whole test suite had 19 passes at this stage.

# Prompt log - AssetFund app and final checks

## What I wanted

I wanted to turn the saved research outputs into an investor-facing Streamlit
app. The app had to explain the funds, support an allocation decision and show
the limits of the sentiment result.

## Prompt(s)

1. Use the fund product idea from my Project A work. Build fund comparison,
   factsheet, allocation and sentiment pages from precomputed Project B files.
2. Design the allocation for funds with different trading calendars. Let the
   selected fund weights drift, charge the fee by elapsed calendar days and
   combine overlapping securities in the look-through holdings.
3. Check every page on desktop and a 430-pixel screen. Test direct page links,
   calculations, labels, warnings and deployment files.

## What the assistant produced

Codex built four pages called Compare funds, Fund factsheet, Build allocation
and Sentiment research. The root `streamlit_app.py` stays small, and the app
reads precomputed files from `results/`.

The allocation engine uses a union calendar. It supports two to five funds,
elapsed-day fees, drifting fund weights and combined look-through holdings.
Codex also created six report figures with a shared AssetFund design.

## What was wrong or risky

Using the intersection of all fund dates would remove crypto weekend returns.
Charging a fee once per row would make the fee depend on the selected calendar.
Resetting fund weights every day would introduce an undisclosed rebalance.

Separate holdings tables could count the same security more than once. The app
also needed to avoid running VADER or the backtests when a widget changed.

Visual review found an incorrect multi-level axis on the sentiment validation
chart. It also found an unclear headline label, a clipped signal-timing label
and internal model names in the interface.

## What I changed and why

I chose the union-calendar design. A closed fund carries its last value while
another market trades. Initial fund weights drift with performance, and the
annual fee accrues by calendar day. Repeated securities are combined into one
look-through exposure.

I kept the locked holdout and the 50 basis-point failure visible on the
Sentiment research page. I also accepted the display fixes after checking the
desktop and 430-pixel layouts. Client labels now use `Two-stage Sleeve Risk
Parity` rather than the internal model name.

I ran the final test suite. Ruff passed on the files created or changed for Project
B, and the hand-in script reported 22 checks passed.

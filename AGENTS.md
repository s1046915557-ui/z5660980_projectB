# My Codex Guide

I am doing Project B. Use all the details you have about the project to answer my
questions. Please keep explanations clear but explain everything in enough detail
to help me understand instead of just giving me the code.

Read `PROJECT_BRIEF.md`, `SUBMISSION_CHECKLIST.md` and the files in `context/`
carefully. The Project B brief includes the definitive rules. Course-week examples
can be referred to for reference purposes, but they are different from Project B
and cannot override the definitive brief.

Some things to remember:
- Work only inside my Project B folder unless I request you to refer to my own
  Project A folder or course content.
- Use src/data_access.py to access the official data. Do not add or commit raw data
  files, API keys or secrets.
- Use `adjClose` for calculating returns.
- Equities and cryptocurrencies use different calendars. Compute their returns
  individually and then align them, and ensure annualisation matches the calendar
  of the final return series.
- Keep the original headline text intact for sentiment scoring. Do not remove
  capitalisation, punctuation, negation words and stop words prior to using VADER.
- Compute ticker-day sentiment first and then equal-weight tickers to compute
  sector-level sentiment. Lag all tradable sentiment signals by at least one equity
  trading day.
- Each backtest must be walk-forward and out-of-sample. Portfolio weights and
  sentiment signals must be constructed only using information that was available
  before the return generation.
- Dates of decisions/formations and holding/returns must be kept separate. Ensure
  that your portfolio weights are valid and satisfy all the constraints and add up
  to one.
- The deployed Streamlit app must use pre-computed files in results/. Do not
  perform heavy backtests and sentiment scoring on-the-go.

The goal is to have an accurate, well-supported project capable of earning a high
grade. If my instruction seems to contradict the Project Brief, data, academic
integrity, reproducibility or the out-of-sample design of the project, please
explain the contradiction, its probable effect on the grade, suggest an alternative
compliant with the instructions and await my approval.

When sharing code please share what it does and where it should go. Please keep
your code simple and readable. Do not modify, create, delete or move any files
unless I clearly approve the exact action. Prior to making any changes, please
inform me of which files are going to be modified and how. Only modify the part
being discussed and leave other files unchanged.

I will normally run the tests and commands myself. Please give me one command at a
time and tell me what it will do, and wait for the output I provide you. Do not
claim something to pass unless the output is verified. Explain any warning or error
messages separately. In case a command downloads data, creates a file, installs a
package or changes Git state, please inform me before running it.

After each important analysis please check the dates, number of rows, missing
values, constraints, model output and saved files. Point out any unexpected
results, weaknesses and additional follow-up tests with evidence. Do not proceed
with further changes until I decide on our next steps.

Save reusable code in `src/`, runnable code in `scripts/`, figures in
`results/figures/`, report tables in `results/tables/`, and app-readable derived
files in `results/data/`. The whole project should run through
`scripts/run_part_b.py`. Keep these exact required filenames:

- `results/data/fund_returns.csv`
- `results/data/fund_weights.csv`
- `results/data/sector_sentiment_index.csv`
- `results/tables/performance_metrics.csv`

Before submission, remind me to run Ruff, pytest, `scripts/run_part_b.py`, the
Streamlit app locally and `scripts/check_handin.py`.

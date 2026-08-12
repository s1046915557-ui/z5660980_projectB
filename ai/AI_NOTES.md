# AI notes

I used Codex in Project B, where I selected the specific workflows and how to
modify them. At each stage, I gave it a clear objective, requiring it to explain
the specific methods and inherent risks before starting. After completion, I ran
the main commands in the terminal and checked the code using saved output, targeted
tests, full tests, Ruff, and submission review scripts.

I requested Codex to reuse only the portions of my Project A data pipeline that
could still be used for Project B. During review, I identified that the original
text panel grouped news headlines. This altered the unit of analysis before VADER
scoring and did not allow me to perform headline-level evaluation. Therefore, I
kept one record for every cleaned headline along with its raw text and date. I
added foundation tests for keys, row counts, returns and weekend news alignment.

For the fund shelf, I applied a monthly expanding-window backtest using separate
decision and effective dates. I verified that the target weights drifted with
returns between rebalances rather than being reset every day. The combined universe
of 50 equities and 10 cryptocurrencies resulted in an asset-count and risk-budget
imbalance. Therefore, I decided to use a Two-stage Sleeve Risk Parity approach,
which first balances risk within the asset-class sleeves and then balances risk
between the equity and crypto sleeves. I checked full investment, long-only
weights, sleeve risk contributions, look-ahead safety, turnover and transaction
costs at 0, 10 and 50 basis points.

The baseline VADER model classified some unambiguously finance-related phrases as
neutral. I requested Codex to leave the original analyser unchanged and build a
separate finance-aware VADER extension. I reviewed the development sample, froze
the lexicon rules and hashes, and only then opened a separate locked holdout with
30 headlines. Holdout macro F1 increased from 0.548 to 0.585. I considered the
smaller holdout gain the appropriate evidence. Ticker-days without news received
neutral fill values, and I marked news availability with a separate coverage flag.

The first sentiment fusion rule showed two other issues. First, raw VADER scores
were mostly positive, and second, news coverage was uneven across sectors.
Therefore, I decided to apply a Coverage-Aware Rank signal. The rule ranks all 10
sectors, centres the ranks between -1 and 1, and dampens signals with weak lagged
news coverage. It merely shifts security weights within the equity sleeve without
changing the total weights of the equity and crypto sleeves of the base fund. Prior
to running the locked holdout for 2023, I froze the rule as well as the 0.25 tilt
strength. I also maintained the strict order of the source dates, decision dates
and effective dates. For the primary fund, the Sharpe ratio was improved by 0.016
before transaction costs and by 0.013 at 10 basis points. At 50 basis points, the
Sharpe difference relative to the base fund was approximately −0.001. I have
retained the failure at 50 basis points since it defines the practical limit of the
signal.

Regarding the AssetFund application, I did not use the intersection calendar since
it would lead to the omission of crypto weekend returns. I have used the union
calendar, let the fund weights drift with performance, charged fees based on
elapsed calendar days and merged duplicate securities in the look-through holdings.
I have verified the calculations, labels, warnings, direct links and page layouts
for the desktop version, at a width of 430 pixels and on the live mobile
application. The saved evidence includes 13 funds, 146,836 scored headlines, a
locked 30-headline holdout and a 720-row fusion signal audit. The test suite has 77
passing tests. The files created or changed for Project B passed the targeted Ruff
checks, and the hand-in script reported 22 passing checks.

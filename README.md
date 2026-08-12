# AssetFund

AssetFund is a Streamlit investment app built from equity, crypto, and news
data. It offers 13 systematically managed funds and reports out-of-sample
performance, holdings, drawdowns, and transaction-cost checks.

The app has four pages:

- **Compare funds** compares risk, return, and fund purpose.
- **Fund factsheet** shows performance and current holdings.
- **Build allocation** combines two to five funds, applies a management fee,
  and shows look-through exposure.
- **Sentiment research** presents the sector sentiment index, VADER validation,
  and the tested effect of sentiment tilts.

## Deployment

- Live app: [fins5545-assetfund-cj.streamlit.app](https://fins5545-assetfund-cj.streamlit.app)
- GitHub repository: [s1046915557-ui/z5660980_projectB](https://github.com/s1046915557-ui/z5660980_projectB)
- Branch: `main`
- Entrypoint: `streamlit_app.py`

## Run locally

Use Python 3.13 and run these commands from the project root:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python scripts/run_part_b.py
python -m streamlit run streamlit_app.py
```

`run_part_b.py` rebuilds the derived CSV files, tables, and figures under
`results/`. The deployed app reads these files and does not rerun the models.

## Check the project

```powershell
python -m pytest -q
python scripts/check_handin.py
```

## Main files

- `streamlit_app.py`: app entrypoint
- `src/`: portfolio, sentiment, fusion, allocation, and app code
- `scripts/run_part_b.py`: full reproduction script
- `results/data/`: app-readable derived data
- `results/tables/`: model and validation tables
- `results/figures/`: report-ready figures
- `report/`: submitted PDF report
- `ai/`: prompt logs and project instructions

Raw source data are loaded through `src/data_access.py` and are not included in
the public repository. The Streamlit deployment entrypoint is
`streamlit_app.py`.

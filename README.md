# Community Nephrology EDA — public Streamlit dashboard

Four views: population overview, eGFR progression, laboratory and vital-sign associations, and KFRE surrogate-endpoint comparisons.

**IMPORTANT: The shipped `/data/` folder contains SYNTHETIC DEMONSTRATION aggregates. Its numbers are not findings from the notebook.** The public app never loads raw patient-level records. The individual trajectory explorer is simulated.

## Files

- `app.py` — public Streamlit app; only loads `data/*.csv` and `metadata.json`.
- `requirements.txt` — deployment packages.
- `make_synthetic_demo.py` — generates self-contained demo aggregates.
- `prepare_public_data.py` — optional offline preparation from an **authorized local clinical CSV**, using the notebook's basic cleaning, eGFR slopes, patient medians, eight-variable KFRE, and three-month low-eGFR-window logic.
- `data/` — output tables (synthetic by default); never add raw patient data here.

## Deploy on Streamlit Community Cloud (free)

1. Download this package, extract it, and put the contents in your GitHub repository. For example, at the repository root, place `app.py`, `requirements.txt`, `.streamlit/config.toml` and `data/`. If you put the files in a `dashboard/` folder, use `dashboard/app.py` as the entrypoint instead.
2. Verify `data/metadata.json` has `"data_mode": "synthetic_demo"` until you have **written authorization to release aggregated clinical data** and have reviewed every output for privacy risk.
3. Commit and push to GitHub. **Do not add any patient-level CSV, individual histories, date of birth, or patient-level KFRE predictions to a public repository.** The original project's existing raw CSV URL needs an independent privacy review as well.
4. Open <https://share.streamlit.io>, sign in via GitHub, select **Create app**, choose your repository and branch, and enter `app.py` (or `dashboard/app.py`) as the main file.
5. Deploy. Streamlit installs packages from `requirements.txt` and serves a public `streamlit.app` link. GitHub commits update the deployed app.

See official documentation: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>

## Test locally (optional)

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

No local install is required to *deploy* the supplied demo: Community Cloud handles deployment from GitHub.

## Real aggregated results: offline only, after authorization

**The output suppression threshold is a starting precaution, not a guarantee of HIPAA de-identification.** Small-cell suppression, subgroup disclosure, and differencing risk require review by the practice's privacy officer/data steward. A public clinical data dashboard must be specifically approved, even if raw direct identifiers were stripped.

Run this only on a trusted machine using an approved, local source file (not a public GitHub URL):

```powershell
python prepare_public_data.py --input "C:\secure\approved_eda_data.csv" --out data --min-count 30 --approved-for-public-aggregate
```

Inspect **every** generated CSV, including category labels, group sizes, metadata and statistical estimates, for disclosure risks, then commit only those approved aggregate outputs. Where source records include combinations of rare characteristics, even aggregates with 30+ rows may be sensitive. The script intentionally limits the app to one-dimensional summaries and preapproved bins, not arbitrary cross-filters; it does not export study IDs or observation dates.

The preparation script is a *reimplementation* of important notebook transformations and should be reconciled against the final notebook before treating results as published findings. Validate these invariants: eligible slope counts, KFRE risk band counts, baseline risk calculations, observed-event counts, and censoring. Python dependency versions may influence computation slightly.

## Key methods and limitations

- eGFR annual slope: separate ordinary least-squares regressions per patient; ≥3 daily median observations spanning ≥1 year.
- Lab/vital associations: each patient's median available measurement (≥2 distinct dates) versus their annual eGFR slope; descriptive, not causal.
- ACR: group by A1 (0 < ACR < 30), A2 (30 ≤ ACR < 300), A3 (≥300); linear trend against log10(ACR).
- KFRE: eight-variable original North American equation, using earliest eligible complete baseline row for each patient. Endpoint is NOT kidney replacement therapy: it is any *complete three-month window* with ≥2 distinct-date eGFR observations whose **mean** is below 15 or 10. The two observations do not have to be three months apart. Negative outcomes require recorded eGFR follow-up to prediction horizon. The two thresholds are analyzed both separately and on a paired evaluable subset. This does not validate the KFRE against its intended clinical event.
- The sample's observation timing, transfer, death, dialysis, transplantation, and referral selection can bias observed low-eGFR events and apparent associations. Plots are exploratory only.
- Sample sizes in visualizations represent available eligible cohorts or disclosed bins, not one interchangeable full cohort.

## Suggested references

- KDIGO 2024 Clinical Practice Guideline for Evaluation and Management of CKD.
- Tangri et al., original Kidney Failure Risk Equations and subsequent multinational validation.
- Shou et al. (2017), *Analytic Considerations for Repeated Measures of eGFR in Cohort Studies of CKD*.

Not intended for direct clinical decisions or individual patient-level risk estimation.

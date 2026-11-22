"""Public-safe, aggregated nephrology EDA dashboard.

No private patient rows are fetched or served. The trajectory explorer is synthetic.
All real analysis tables must be reviewed and approved before public deployment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GFR_ORDER = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
FEATURE_LABELS = {
    "acr": "Median ACR (mg/g)", "calcium": "Median calcium (mg/dL)",
    "potassium": "Median potassium (mmol/L)", "hemoglobin": "Median hemoglobin (g/dL)",
    "phosphorus": "Median phosphorus (mg/dL)", "glucose": "Median glucose (mg/dL)",
    "co2": "Median CO₂ (mmol/L)", "albumin": "Median serum albumin (g/dL)",
    "creatinine": "Median creatinine (mg/dL)",
    "systolic_bp": "Median systolic BP (mmHg)", "diastolic_bp": "Median diastolic BP (mmHg)",
    "bmi": "Median BMI (kg/m²)", "MAP": "Median MAP (mmHg)",
}

st.set_page_config(page_title="Nephrology EDA", page_icon="🩺", layout="wide")

@st.cache_data(show_spinner=False)
def load_public_data():
    names = ["overview", "distributions", "associations", "regressions", "kfre"]
    tables = {}
    for name in names:
        filename = DATA / f"{name}.csv"
        if not filename.exists():
            st.error(f"Missing {filename.name}. See README for the data preparation step.")
            st.stop()
        tables[name] = pd.read_csv(filename)
    with (DATA / "metadata.json").open(encoding="utf-8") as f:
        meta = json.load(f)
    return meta, tables

meta, data = load_public_data()
if meta.get("data_mode") not in ("synthetic_demo", "approved_aggregates"):
    st.error("Unrecognized data publication mode. Refusing to display.")
    st.stop()

st.sidebar.title("Nephrology EDA")
page = st.sidebar.radio(
    "Explore", ["Population overview", "eGFR progression", "Laboratory associations", "KFRE evaluation"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Public presentation dashboard • aggregated data only")
if meta["data_mode"] == "synthetic_demo":
    st.warning("**SYNTHETIC DEMONSTRATION DATA.** All displayed numbers and relationships are illustrative, not results from the clinical dataset.")
else:
    st.info("Approved aggregate tables only. Clinical data have not been loaded into the public app.")


def download_aggregate(frame, label, filename):
    st.download_button(
        label, data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename, mime="text/csv"
    )


def fig_layout(fig, height=470):
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=45, b=35),
                      plot_bgcolor="white", legend_title_text="")
    return fig


def subgroup_filter(frame, group):
    return frame[frame["gfr_category"].eq(group)].copy()


if page == "Population overview":
    st.title("Population overview")
    st.write("Explore cohort composition and data availability. Filters select a **single published summary**, not an arbitrary combination of patient-level variables.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Source observations", f"{meta.get('n_rows', 0):,}")
    col2.metric("Distinct source patients", f"{meta.get('n_patients', 0):,}")
    col3.metric("Patients with eligible eGFR slope", f"{meta.get('n_slopes', 0):,}")
    st.caption("Slope eligibility: ≥3 distinct-date eGFR measurements and ≥1 year of follow-up. Patient counts in published subgroups may not sum to the cohort total because small groups are withheld.")
    dim_map = {"Baseline GFR category": "gfr_category", "Sex": "sex", "Race": "race",
               "Insurance": "insurance", "Service location": "location"}
    choice = st.selectbox("Compare groups", list(dim_map))
    subset = data["overview"].loc[data["overview"]["dimension"] == dim_map[choice]].copy()
    if subset.empty:
        st.warning("No approved groups meet the disclosure threshold in this view.")
    else:
        if dim_map[choice] == "gfr_category":
            subset["category"] = pd.Categorical(subset["category"], categories=GFR_ORDER, ordered=True)
            subset = subset.sort_values("category")
        fig = px.bar(subset, x="category", y="patients", text="patients",
                     labels={"category": choice, "patients": "Patients"}, color="median_annual_slope",
                     color_continuous_scale="RdBu", color_continuous_midpoint=0,
                     hover_data=["median_baseline_egfr", "pct_declining"])
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.dataframe(subset, hide_index=True, use_container_width=True)
        download_aggregate(subset, "Download this aggregate table", "public_population_summary.csv")
    st.caption("Age, lab dates, individual study IDs, individual records, and precise combinations of demographic characteristics are not exposed.")

elif page == "eGFR progression":
    st.title("eGFR progression")
    st.write("Patient-level annual slopes were calculated from daily-median eGFR observations. The histogram shows disclosure-screened counts of those slopes.")
    category = st.selectbox("Population", ["All", *GFR_ORDER])
    hist = subgroup_filter(data["distributions"], category)
    if not hist.empty:
        fig = px.bar(hist, x="bin_center", y="patients", labels={"bin_center": "Annual eGFR slope (mL/min/1.73 m²/year)", "patients": "Patients"})
        fig.add_vline(x=0, line_dash="dash", line_color="#B34C40")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        st.caption("Bins with fewer than the publication threshold are omitted; this is not a complete reconstruction of the patient-level histogram.")
        download_aggregate(hist, "Download slope distribution summary", "public_slope_histogram.csv")
    else:
        st.info("No publishable histogram bins for this group.")
    st.subheader("Trajectory explorer — synthetic examples")
    st.write("The interactive trajectories below are **simulated**, not actual patients. This design retains the explanatory feature without publishing recognizable medical histories.")
    example = st.selectbox("Illustrative pattern", ["Gradual decline", "Stable with noise", "Acute drop and partial recovery", "Accelerating decline"])
    seed = st.number_input("Synthetic random seed", min_value=0, max_value=999999, value=6500, step=1)
    rng = np.random.default_rng(int(seed))
    t = np.linspace(0, 5, 20)
    patterns = {
        "Gradual decline": 60 - 2.4*t,
        "Stable with noise": 44 + 0*t,
        "Acute drop and partial recovery": 54 - 1.1*t - 18*np.exp(-((t-2.4)/0.38)**2),
        "Accelerating decline": 64 - 0.8*t - 0.85*t*t,
    }
    observed = patterns[example] + rng.normal(0, 3.8, len(t))
    fit = np.polyval(np.polyfit(t, observed, 1), t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=observed, mode="markers", name="Synthetic observations"))
    fig.add_trace(go.Scatter(x=t, y=fit, mode="lines", name="Linear fit"))
    fig.add_trace(go.Scatter(x=t, y=patterns[example], mode="lines", name="Simulation's underlying curve", line=dict(dash="dash")))
    fig.update_layout(xaxis_title="Years", yaxis_title="eGFR (mL/min/1.73 m²)")
    st.plotly_chart(fig_layout(fig), use_container_width=True)

elif page == "Laboratory associations":
    st.title("Laboratory associations")
    st.write("Explore aggregated measurement bins and original patient-level regression summaries generated offline. A displayed point is an **aggregate bin**, not a patient.")
    feature = st.selectbox("Median measurement", list(FEATURE_LABELS), format_func=lambda k: FEATURE_LABELS[k])
    group = st.selectbox("Baseline GFR category", ["All", *GFR_ORDER])
    band_options = ["All"] if feature != "acr" else ["A1", "A2", "A3"]
    band = st.selectbox("Albuminuria range" if feature == "acr" else "Relationship", band_options)
    subset = data["associations"]
    subset = subset[(subset["feature"] == feature) & (subset["gfr_category"] == group) & (subset["band"] == band)]
    reg = data["regressions"]
    reg = reg[(reg["feature"] == feature) & (reg["gfr_category"] == group) & (reg["band"] == band)]
    if subset.empty:
        st.info("No approved aggregate bins exist for this selection.")
    else:
        fig = px.scatter(subset, x="x_median", y="y_median", size="patients",
                         hover_data=["patients"], labels={"x_median": FEATURE_LABELS[feature], "y_median": "Median annual eGFR slope"},
                         title="Aggregated bins: median eGFR slope vs. median measurement")
        if feature == "acr":
            fig.update_xaxes(type="log")
        if not reg.empty:
            r = reg.iloc[0]
            xlow, xhigh = float(subset.x_median.min()), float(subset.x_median.max())
            if xlow < xhigh:
                xx = np.geomspace(xlow, xhigh, 100) if feature == "acr" else np.linspace(xlow, xhigh, 100)
                predictor = np.log10(xx) if feature == "acr" else xx
                yy = float(r["intercept"]) + float(r["slope"])*predictor
                fig.add_trace(go.Scatter(x=xx, y=yy, name="Patient-level OLS summary", mode="lines", line=dict(color="#C34E49")))
                label = "log₁₀(ACR)" if feature == "acr" else "x"
                st.caption(f"Patient-level OLS fit: y = {r['slope']:.3g} × {label} {r['intercept']:+.3g}; R² = {r['r_squared']:.3f}; n = {int(r['patients']):,}. The fit is calculated offline; plotted dots are grouped summaries.")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        download_aggregate(subset, "Download aggregate scatter data", f"public_{feature}_{group}_{band}_bins.csv")
    st.caption("Medians summarize each patient's available history; these are descriptive associations, not evidence of prospective prediction or causation.")

else:
    st.title("Kidney Failure Risk Equation: surrogate-outcome comparison")
    st.warning("**Not a clinical validation of KFRE.** The original KFRE predicts kidney failure requiring kidney replacement therapy. This project evaluates whether a three-month *window containing ≥2 distinct-date eGFR values whose mean is below a threshold* was detected. It does not establish sustained low eGFR, dialysis, transplantation, or kidney failure outcomes.")
    horizon = st.radio("Prediction horizon", [2, 5], horizontal=True, format_func=lambda k: f"{k} years")
    threshold = st.radio("Surrogate mean-eGFR threshold", [15, 10], horizontal=True, format_func=lambda k: f"Below {k}")
    paired_only = st.checkbox("Use paired evaluable cohort (same patients for both thresholds)", value=True)
    subset = data["kfre"]
    subset = subset[(subset["years"] == horizon) & (subset["threshold"] == threshold) & (subset["cohort"] == ("paired" if paired_only else "threshold_specific"))]
    if subset.empty:
        st.info("No publishable risk bands meet the cohort/disclosure requirements.")
    else:
        x = subset["mean_predicted"] * 100
        y = subset["observed_rate"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", name="Identity (not expected for surrogate)", line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines", name="Observed low-eGFR surrogate rate",
                                 marker=dict(size=10), text=subset["risk_band"],
                                 error_y=dict(type="data", symmetric=False, array=(subset["ci_upper"]-subset["observed_rate"])*100, arrayminus=(subset["observed_rate"]-subset["ci_lower"])*100)))
        if len(subset) >= 2 and x.nunique() > 1:
            weights = subset["patients"].to_numpy(dtype=float)
            slope, intercept = np.polyfit(x.to_numpy(), y.to_numpy(), 1, w=np.sqrt(weights))
            xx = np.linspace(x.min(), x.max(), 100)
            fig.add_trace(go.Scatter(x=xx, y=intercept+slope*xx, mode="lines", name=f"Binned WLS fit: y={slope:.2f}x{intercept:+.2f}"))
        fig.update_layout(xaxis_title="Mean predicted KFRE risk (%)", yaxis_title="Observed low-eGFR surrogate event rate (%)", xaxis_range=[0,100], yaxis_range=[0,100])
        st.plotly_chart(fig_layout(fig), use_container_width=True)
        show = subset[["risk_band", "patients", "mean_predicted", "observed_events", "observed_rate", "ci_lower", "ci_upper"]]
        st.dataframe(show, hide_index=True, use_container_width=True)
        download_aggregate(show, "Download aggregate KFRE table", f"public_kfre_{horizon}y_{threshold}_{'paired' if paired_only else 'specific'}.csv")
    st.caption("A qualifying window can contain just two nearby laboratory dates; it does **not** require 3 months of persistent low eGFR. Follow-up censoring and unmatched clinical outcomes limit interpretation. The dashed identity line is only a visual reference, not a standard of expected calibration under this surrogate definition.")

st.divider()
st.caption("Educational exploratory data analysis. Not for clinical decisions, patient-level care, or individual kidney-failure risk estimation.")

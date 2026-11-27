"""Interactive nephrology EDA sourced from the approved project CSV on GitHub.

The analytical definitions mirror INSY_6500_Project_Gladish(2) notebook:
- daily-median eGFR; >=3 distinct days over >=1 year for slope
- patient-level medians require >=2 distinct days
- earliest complete eight-variable KFRE record, age>=18 and 15<=eGFR<60
- 3-month windows with >=2 distinct dates and window mean below 15/10
Use for educational exploration, NOT clinical decisions or formal KFRE validation.
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import binomtest

st.set_page_config(page_title="Nephrology EDA", page_icon="🩺", layout="wide")

# This exact source URL appears in cell 5 of the supplied EDA notebook.
DEFAULT_DATA_URL = (
    "https://github.com/ggladish1/INSY_6500_EDA/raw/refs/heads/main/EDA%20Data%203"
)
# Secrets may not be configured for a new Community Cloud app.
try:
    configured_url = st.secrets.get("DATA_URL", None)
except Exception:
    configured_url = None
DATA_URL = configured_url or os.environ.get("NEPHROLOGY_DATA_URL", DEFAULT_DATA_URL)
GFR_ORDER = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
LABS = ["calcium", "potassium", "hemoglobin", "phosphorus", "glucose", "co2", "albumin", "creatinine"]
VITALS = ["systolic_bp", "diastolic_bp", "bmi", "MAP"]
FEATURES = ["acr", *LABS, *VITALS]
BAD_VALUE_LIMITS = {
    "egfr": (0, 150), "acr": (0, None), "creatinine": (0, 20),
    "glucose": (0, 600), "calcium": (0, 15), "potassium": (0, 8),
    "hemoglobin": (0, 18), "phosphorus": (0, 15), "co2": (0, 45),
    "systolic_bp": (0, 300), "diastolic_bp": (0, 200)
}
# Exploratory measurement bands, assuming conventional US units.
# These are illustrative thresholds, NOT patient-specific diagnostic limits.
# Glucose is not reliably fasting; hemoglobin and creatinine ranges vary by sex.
REFERENCE_RANGES = {
    "calcium": (8.5, 10.5),       # mg/dL
    "potassium": (3.5, 5.0),      # mmol/L
    "hemoglobin": (12.0, 17.5),   # g/dL, pooled illustrative bounds
    "phosphorus": (2.5, 4.5),     # mg/dL
    "glucose": (70.0, 140.0),     # mg/dL, exploratory; fasting status unknown
    "co2": (22.0, 29.0),          # mmol/L
    "albumin": (3.5, 5.0),        # g/dL
    "creatinine": (0.6, 1.3),     # mg/dL, pooled illustrative bounds
    "systolic_bp": (130.0, 150.0),
    "diastolic_bp": (60.0, 80.0),
    "MAP": (70.0, 100.0),
}
BAND_COLORS = {"Low": "#4575b4", "Normal": "#27936b", "High": "#d6812f"}
MISSING_LABEL = "(Missing / unknown)"


def measurement_band(values, lower, upper):
    """Classify continuous measurements using low < lower <= normal <= upper < high."""
    return np.select([values < lower, values <= upper], ["Low", "Normal"], default="High")


REQUIRED = ["study_id", "lab_date", "date_of_birth", "sex", "egfr", "acr", "albumin", "phosphorus", "co2", "calcium"]


def clean_category(series):
    s = series.astype("string").str.strip()
    return s.mask(s.str.lower().isin(["", "nan", "none", "null", "n/a", "<na>"]))


def normalize_location(series):
    s = clean_category(series)
    names = {
        "cullman": "Cullman", "nephrology of north alabama cullman": "Cullman",
        "c-nephrology of north alabama": "Cullman", "cullman-nephrology of north alabama": "Cullman",
        "cullman nephrology of north alabama pc": "Cullman", "cullman nephrology of north alabama": "Cullman",
        "decatur": "Decatur", "decatur nephrology of north alabama pc": "Decatur",
        "decatur nephrology of north alabama": "Decatur",
        "nephrology of north alabama decatur": "Decatur", "nephrology of north alabama": "Unspecified",
        "decatur morgan hospital ip": "Decatur Morgan Hospital IP"
    }
    return s.str.lower().map(names).fillna(s)


@st.cache_data(show_spinner="Downloading and cleaning approved source data…", ttl=3600)
def load_source(url):
    # Read remote CSV regardless of whether its filename ends in .csv.
    df = pd.read_csv(url, dtype={"study_id": "string"}).drop_duplicates().copy()
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing notebook-required columns: {missing}")
    for col in ["lab_date", "encounter_date", "date_of_birth"]:
        if col not in df:
            df[col] = pd.NaT
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["race", "insurance", "location", "ethnicity"]:
        if col not in df:
            df[col] = pd.NA
    for col in ["egfr", "acr", *LABS, *VITALS[:-1], "weight_lbs"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col, (lo, hi) in BAD_VALUE_LIMITS.items():
        invalid = df[col].lt(lo)
        if hi is not None:
            invalid |= df[col].gt(hi)
        df.loc[invalid, col] = np.nan
    # Compute MAP only after removing erroneous BP measurements.
    df["MAP"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
    df["age"] = (df["lab_date"] - df["date_of_birth"]).dt.days / 365.25
    df.loc[df["age"] < 0, "age"] = np.nan
    for col in ["sex", "race", "insurance", "location"]:
        df[col] = clean_category(df[col])
    df["location"] = normalize_location(df["location"])
    return df


@st.cache_data(show_spinner="Computing longitudinal patient summaries…", ttl=3600)
def make_patient_tables(df):
    gfr = (df[["study_id", "lab_date", "egfr"]]
           .dropna().groupby(["study_id", "lab_date"], as_index=False)["egfr"].median()
           .sort_values(["study_id", "lab_date"]).reset_index(drop=True))
    rows = []
    for pid, group in gfr.groupby("study_id", sort=False):
        if len(group) < 3:
            continue
        t = (group["lab_date"] - group["lab_date"].iloc[0]).dt.days.to_numpy() / 365.25
        if t.max() < 1:
            continue
        values = group["egfr"].to_numpy(dtype=float)
        rows.append({"study_id": pid, "annual_egfr_change": np.polyfit(t, values, 1)[0],
                     "baseline_egfr": values[0], "final_egfr": values[-1],
                     "n_egfr_observations": len(group), "years_followup": t.max(),
                     "first_egfr_date": group["lab_date"].iloc[0]})
    patient = pd.DataFrame(rows)
    if patient.empty:
        raise ValueError("No patients met the notebook's eGFR slope eligibility rules.")
    patient["gfr_category"] = pd.cut(
        patient["baseline_egfr"], bins=[-np.inf, 15, 30, 45, 60, 90, np.inf],
        labels=["G5", "G4", "G3b", "G3a", "G2", "G1"], right=False).astype("string")
    cats = df[["study_id", "lab_date", "encounter_date", "sex", "race", "insurance", "location"]].copy()
    cats["record_date"] = cats["encounter_date"].fillna(cats["lab_date"])
    cats = cats.sort_values("record_date", kind="stable")
    latest = cats.groupby("study_id")[["sex", "race", "insurance", "location"]].agg(
        lambda s: s.dropna().iloc[-1] if s.notna().any() else pd.NA).reset_index()
    patient = patient.merge(latest, on="study_id", how="left")
    births = df.groupby("study_id")["date_of_birth"].first().rename("date_of_birth")
    patient = patient.join(births, on="study_id")
    patient["baseline_age"] = (patient["first_egfr_date"] - patient["date_of_birth"]).dt.days / 365.25
    # Patient medians: 1 median per calendar date, then median across distinct dates.
    for feat in FEATURES:
        source_date = df["encounter_date"].fillna(df["lab_date"]) if feat in VITALS else df["lab_date"]
        work = pd.DataFrame({"study_id": df["study_id"], "date": source_date, "value": df[feat]}).dropna()
        daily = work.groupby(["study_id", "date"], as_index=False)["value"].median()
        aggregate = daily.groupby("study_id")["value"].agg(["median", "count"])
        aggregate.loc[aggregate["count"] < 2, "median"] = np.nan
        patient = patient.join(aggregate["median"].rename(f"median_{feat}"), on="study_id")
    return gfr, patient


def filter_patients(patient):
    """Apply selections sequentially; preserve missing data by default."""
    st.sidebar.subheader("Patient filters")
    filtered = patient.copy()
    counts = [("Eligible longitudinal slopes", len(filtered))]

    categories = st.sidebar.multiselect("Baseline GFR", GFR_ORDER, default=GFR_ORDER)
    filtered = filtered[filtered["gfr_category"].isin(categories)].copy()
    counts.append(("After GFR category", len(filtered)))

    for col, label in [("sex", "Sex"), ("race", "Race"),
                       ("location", "Location"), ("insurance", "Insurance")]:
        values = patient[col].fillna(MISSING_LABEL).astype(str)
        choices = sorted(values.unique().tolist())
        # Including the missing category avoids a hidden loss of patients.
        selected = st.sidebar.multiselect(label, choices, default=choices)
        filtered = filtered[filtered[col].fillna(MISSING_LABEL).astype(str).isin(selected)].copy()
        counts.append((f"After {label.lower()}", len(filtered)))

    ages = patient["baseline_age"].dropna()
    if not ages.empty:
        lo, hi = max(0, int(np.floor(ages.min()))), min(120, int(np.ceil(ages.max())))
        if lo < hi:
            bounds = st.sidebar.slider("Baseline age (years)", lo, hi, (lo, hi))
            include_missing = st.sidebar.checkbox("Include missing baseline ages", value=True)
            age_mask = filtered["baseline_age"].between(*bounds)
            if include_missing:
                age_mask |= filtered["baseline_age"].isna()
            filtered = filtered[age_mask].copy()
    counts.append(("After baseline age", len(filtered)))
    return filtered, pd.DataFrame(counts, columns=["Selection step", "Patients remaining"])


def scatter_fit(x, y, use_log=False):
    subset = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if use_log:
        subset = subset[subset.x > 0]
    if len(subset) < 3 or subset.x.nunique() < 2 or subset.y.nunique() < 2:
        return None
    xx = np.log10(subset.x.to_numpy()) if use_log else subset.x.to_numpy()
    yy = subset.y.to_numpy()
    m, b = np.polyfit(xx, yy, 1)
    r2 = 1 - ((yy - (m * xx + b))**2).sum() / ((yy - yy.mean())**2).sum()
    return m, b, r2, len(subset), float(subset.x.min()), float(subset.x.max())


def kfre_8_vectorized(rows, years):
    male = rows["sex"].astype("string").str.lower().isin(["m", "male"]).astype(float)
    lp = (-0.1992 * (rows.age / 10 - 7.036) + 0.1602 * (male - 0.5642)
          - 0.4919 * (rows.egfr / 5 - 7.222) + 0.3364 * (np.log(rows.acr) - 5.137)
          - 0.3441 * (rows.albumin - 3.997) + 0.2604 * (rows.phosphorus - 3.916)
          - 0.07354 * (rows.co2 - 25.57) - 0.2228 * (rows.calcium - 9.355))
    return 1 - (0.9780 if years == 2 else 0.9301) ** np.exp(lp)


def find_low_egfr_window(measurements, period_start, period_end, threshold=15):
    """Match notebook: complete calendar 3-month window with >=2 dates and mean below threshold."""
    start, end = pd.Timestamp(period_start).normalize(), pd.Timestamp(period_end).normalize()
    data = measurements[["lab_date", "egfr"]].dropna().copy()
    data["lab_date"] = data["lab_date"].dt.normalize()
    data = data.groupby("lab_date", as_index=False)["egfr"].median().sort_values("lab_date")
    data = data[data.lab_date.between(start, end)]
    if len(data) < 2:
        return pd.NaT
    last_start = end - pd.DateOffset(months=3)
    if last_start < start:
        return pd.NaT
    starts = {start, last_start}
    for d in data.lab_date:
        starts.update([d, d + pd.Timedelta(days=1), d - pd.DateOffset(months=3),
                       d - pd.DateOffset(months=3) + pd.Timedelta(days=1)])
    for s in sorted(starts):
        if start <= s <= last_start:
            window_end = s + pd.DateOffset(months=3)
            w = data[data.lab_date.between(s, window_end)]
            if len(w) >= 2 and w.egfr.mean() < threshold:
                return window_end
    return pd.NaT


@st.cache_data(show_spinner="Evaluating KFRE surrogate outcomes…", ttl=3600)
def prepare_kfre(df, gfr):
    cols = ["study_id", "lab_date", "age", "sex", "egfr", "acr", "albumin", "phosphorus", "co2", "calcium"]
    candidates = df[cols].dropna().copy()
    candidates["sex"] = candidates.sex.astype("string").str.strip().str.lower()
    eligible = (candidates.age.ge(18) & candidates.egfr.ge(15) & candidates.egfr.lt(60)
                & candidates.acr.gt(0) & candidates.sex.isin(["m", "male", "f", "female"])
                & candidates[["albumin", "phosphorus", "co2", "calcium"]].gt(0).all(axis=1))
    baseline = (candidates.loc[eligible].sort_values(["study_id", "lab_date"], kind="stable")
                .drop_duplicates("study_id", keep="first")
                .rename(columns={"lab_date": "baseline_date"}))
    histories = {k: v for k, v in gfr.groupby("study_id", sort=False)}
    usable = []
    for row in baseline.itertuples():
        history = histories.get(row.study_id)
        if history is None:
            continue
        prior = history[history.lab_date.dt.normalize() < row.baseline_date.normalize()]
        if len(prior) >= 2 and pd.notna(find_low_egfr_window(
            prior, prior.lab_date.min() - pd.DateOffset(months=3), row.baseline_date, 15)):
            continue
        usable.append(row.Index)
    baseline = baseline.loc[usable].copy()
    last_dates = gfr.groupby("study_id")["lab_date"].max().rename("last_egfr_date")
    baseline = baseline.join(last_dates, on="study_id")
    output = {}
    for years in [2, 5]:
        risk = baseline.copy()
        risk["predicted"] = kfre_8_vectorized(risk, years)
        risk["horizon_date"] = risk.baseline_date + pd.DateOffset(years=years)
        event_15, event_10 = [], []
        for row in risk.itertuples():
            hist = histories[row.study_id]
            d = hist.lab_date.dt.normalize()
            future = hist[(d > row.baseline_date.normalize()) & (d <= row.horizon_date.normalize())]
            event_15.append(find_low_egfr_window(future, row.baseline_date, row.horizon_date, 15))
            event_10.append(find_low_egfr_window(future, row.baseline_date, row.horizon_date, 10))
        followup = risk.last_egfr_date >= risk.horizon_date
        risk["event_15"] = np.where(pd.notna(event_15), 1, np.where(followup, 0, np.nan))
        risk["event_10"] = np.where(pd.notna(event_10), 1, np.where(followup, 0, np.nan))
        output[years] = risk
    return output


def kfre_summary(risk, threshold, paired):
    work = risk.dropna(subset=["event_15", "event_10"] if paired else [f"event_{threshold}"]).copy()
    if work.empty:
        return pd.DataFrame()
    work["risk_band"] = pd.cut(work.predicted, [0, .05, .10, .20, 1.000001],
                               labels=["0–5%", "5–10%", "10–20%", "20%+"], right=False)
    work = work.dropna(subset=["risk_band"])
    result = (work.groupby("risk_band", observed=True).agg(
        patients=("study_id", "size"), predicted=("predicted", "mean"),
        events=(f"event_{threshold}", "sum")).reset_index())
    if result.empty:
        return result
    result["observed"] = result.events / result.patients
    ci = [binomtest(int(r.events), int(r.patients)).proportion_ci(method="wilson")
          for r in result.itertuples()]
    result["ci_lower"] = [p.low for p in ci]
    result["ci_upper"] = [p.high for p in ci]
    return result


# Read and prepare source once; Streamlit caches across page/filter changes.
try:
    source = load_source(DATA_URL)
    gfr, patients = make_patient_tables(source)
except Exception as exc:
    st.error(f"Unable to load/prepare source dataset: {exc}")
    st.info("Confirm that the GitHub Raw URL is public and still returns CSV-formatted text.")
    st.stop()

st.sidebar.title("Nephrology EDA")
page = st.sidebar.radio("Explore", ["Population overview", "eGFR progression", "Laboratory associations", "KFRE evaluation"])
filtered, filter_audit = filter_patients(patients)
st.sidebar.caption("Public, approved study data • educational use only")

if page == "Population overview":
    st.title("Population overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Source observations", f"{len(source):,}")
    c2.metric("Patients in source", f"{source.study_id.nunique():,}")
    c3.metric("Eligible slopes", f"{len(patients):,}")
    c4.metric("Patients after filters", f"{len(filtered):,}")
    st.caption("Eligible slopes require ≥3 unique eGFR dates spanning ≥1 year. Source totals are unfiltered. "
               "Missing sex/race/location/insurance values are now included by default, "
               "rather than implicitly removed by the multiselect controls.")
    with st.expander("Why does the number of patients change?", expanded=True):
        st.write("Every sidebar filter acts on the patients remaining after the previous filter. "
                 "The same filtered cohort is used by the eGFR histogram and laboratory scatterplots. "
                 "The KFRE page uses its own eligibility criteria.")
        audit = filter_audit.copy()
        audit["Removed in step"] = audit["Patients remaining"].shift(1).sub(audit["Patients remaining"]).fillna(0).astype(int)
        st.table(audit)
        st.caption("Individual charts can require additional nonmissing measurements; their n may be smaller.")
    group_col = st.selectbox("Compare", ["gfr_category", "sex", "race", "insurance", "location"])
    group = (filtered.groupby(group_col, observed=True)["annual_egfr_change"]
             .agg(patients="size", median_annual_slope="median").reset_index())
    if not group.empty:
        fig = px.bar(group, x=group_col, y="patients", color="median_annual_slope", text="patients",
                     color_continuous_scale="RdBu", color_continuous_midpoint=0)
        st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["toImage"]})
        st.table(group)
    else:
        st.warning("No patients match the selected filters.")

elif page == "eGFR progression":
    st.title("eGFR progression")
    if filtered.empty:
        st.info("No patients match the current filters.")
        st.stop()
    fig = px.histogram(filtered, x="annual_egfr_change", nbins=65,
                       labels={"annual_egfr_change": "Annual eGFR change (mL/min/1.73 m²/year)"})
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["toImage"]})
    st.subheader("Individual longitudinal trajectory")
    patient_ids = sorted(filtered.study_id.astype(str).unique())
    pid = st.selectbox("Study ID", patient_ids)
    hist = gfr[gfr.study_id.eq(pid)].sort_values("lab_date").copy()
    if not hist.empty:
        hist["years"] = (hist.lab_date - hist.lab_date.iloc[0]).dt.days / 365.25
        slope, intercept = np.polyfit(hist.years, hist.egfr, 1)
        fig = px.scatter(hist, x="lab_date", y="egfr", title=f"Study ID {pid}: {len(hist)} unique-date eGFR values")
        fig.add_trace(go.Scatter(x=hist.lab_date, y=intercept + slope * hist.years,
                                 mode="lines", name=f"OLS slope {slope:+.2f}/year"))
        st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["toImage"]})

elif page == "Laboratory associations":
    st.title("Laboratory and vital-sign associations")
    st.caption("Each dot is a patient median across available measurements on ≥2 distinct dates. "
               "Medians and slopes may cover overlapping periods, so these are descriptive associations, not prospective predictors.")
    feat = st.selectbox("Median variable", FEATURES)
    xcol = f"median_{feat}"
    plot = filtered[["study_id", "annual_egfr_change", xcol, "gfr_category", "baseline_age"]].dropna(
        subset=[xcol, "annual_egfr_change"]).copy()
    use_log = feat == "acr"

    if use_log:
        plot = plot[plot[xcol] > 0].copy()
        options = ["All", "A1: 0–30", "A2: 30–300", "A3: 300+"]
        selected_band = st.selectbox("ACR interval", options)
        edges = (30.0, 300.0)
        low_name, normal_name, high_name = options[1:]
        plot["band"] = np.select(
            [plot[xcol] < 30, plot[xcol] < 300],
            [low_name, normal_name], default=high_name)
    elif feat in REFERENCE_RANGES:
        reference_low, reference_high = REFERENCE_RANGES[feat]
        st.caption("Reference bands are *illustrative* rather than patient-specific laboratory reference intervals. "
                   "Verify units and laboratory-specific ranges; glucose fasting status is unknown, "
                   "and creatinine/hemoglobin vary by patient characteristics.")
        left, right = st.columns(2)
        with left:
            lower = st.number_input(f"Low / normal boundary for {feat}",
                                    min_value=0.0, value=float(reference_low), step=0.1, key=f"low_{feat}")
        with right:
            upper = st.number_input(f"Normal / high boundary for {feat}",
                                    min_value=0.0, value=float(reference_high), step=0.1, key=f"high_{feat}")
        if lower >= upper:
            st.error("The low/normal boundary must be smaller than the normal/high boundary.")
            st.stop()
        plot["band"] = measurement_band(plot[xcol], lower, upper)
        options = ["All", "Low", "Normal", "High"]
        selected_band = st.selectbox("Measurement interval", options)
    else:
        selected_band = "All"
        options = ["All"]
        plot["band"] = "All"

    if selected_band != "All":
        plot = plot[plot["band"] == selected_band].copy()
    if len(plot) < 3:
        st.info("Too few patients remain after applying these filters.")
        st.stop()

    fig = px.scatter(plot, x=xcol, y="annual_egfr_change", color="baseline_age",
                     hover_data=["study_id", "gfr_category", "band"], opacity=0.65,
                     labels={xcol: f"Patient median {feat}",
                             "annual_egfr_change": "Annual eGFR slope"},
                     color_continuous_scale="Viridis")
    if use_log:
        fig.update_xaxes(type="log")
    fit_groups = [selected_band] if selected_band != "All" else (
        options[1:] if len(options) > 1 else ["All"])
    results = []
    line_colors = {
        "A1: 0–30": "#4575b4", "A2: 30–300": "#27936b", "A3: 300+": "#d6812f",
        "Low": "#4575b4", "Normal": "#27936b", "High": "#d6812f", "All": "#c33e43"
    }
    for name in fit_groups:
        piece = plot if name == "All" else plot[plot["band"] == name]
        result = scatter_fit(piece[xcol], piece.annual_egfr_change, use_log=use_log)
        if result is None:
            continue
        m, b, r2, n, xmin, xmax = result
        xx = np.geomspace(xmin, xmax, 100) if use_log else np.linspace(xmin, xmax, 100)
        yy = m * (np.log10(xx) if use_log else xx) + b
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", line=dict(color=line_colors[name], width=3),
                                 name=f"{name}: y={m:.3g}x{b:+.3g}, R²={r2:.3f}, n={n:,}"))
        results.append({"Band": name, "Patients": n, "Slope": m,
                        "Intercept": b, "R²": r2})
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    if selected_band == "All" and len(options) > 1:
        edges_to_show = (30.0, 300.0) if use_log else (lower, upper)
        for edge in edges_to_show:
            fig.add_vline(x=edge, line_dash="dot", line_color="gray", opacity=0.5)
    st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["toImage"]})
    st.caption("Regression x is log₁₀(ACR) for ACR, and the untransformed patient median otherwise. "
               "Each subgroup has its own least-squares fit. R² describes within-band variance explained.")
    if results:
        st.table(pd.DataFrame(results).round(4))
    else:
        st.info("There are insufficient distinct measurements to fit a line for the selected subgroup.")

else:
    st.title("KFRE: low-eGFR surrogate-endpoint analysis")
    st.warning("KFRE predicts kidney failure treated with kidney replacement therapy. "
               "This EDA instead detects a 3-month window with ≥2 distinct-date eGFR values whose mean is below 15 or 10. "
               "It **does not** establish 3-month persistence, dialysis initiation, transplantation, or formal KFRE calibration.")
    years = st.radio("KFRE horizon", [2, 5], horizontal=True)
    threshold = st.radio("Mean eGFR below", [15, 10], horizontal=True)
    paired = st.checkbox("Compare using patients evaluable under both thresholds", value=True)
    with st.spinner("Preparing KFRE risk and observed surrogate events…"):
        risks = prepare_kfre(source, gfr)
    risk = risks[years]
    # For this page use the full original KFRE-eligible population: it need not have an eligible slope.
    summary = kfre_summary(risk, threshold, paired)
    st.caption(f"KFRE baseline population: {len(risk):,}; evaluable outcome population in selected cohort: "
               f"{len(risk.dropna(subset=['event_15', 'event_10'] if paired else [f'event_{threshold}'])):,}.")
    if summary.empty:
        st.info("No evaluable risk bands available.")
        st.stop()
    x, y = summary.predicted * 100, summary.observed * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Identity reference (not expected for surrogate)",
                             line=dict(color="gray", dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers+lines", name=f"Observed eGFR <{threshold} window",
                             text=summary.risk_band,
                             error_y=dict(type="data", symmetric=False,
                                          array=(summary.ci_upper - summary.observed)*100,
                                          arrayminus=(summary.observed - summary.ci_lower)*100)))
    if len(summary) > 1 and x.nunique() > 1:
        m, b = np.polyfit(x.to_numpy(), y.to_numpy(), 1, w=np.sqrt(summary.patients.to_numpy()))
        grid = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(x=grid, y=m*grid+b, mode="lines", name=f"Binned WLS: y={m:.2f}x{b:+.2f}"))
    fig.update_layout(xaxis_title="Mean predicted KFRE risk (%)", yaxis_title="Observed low-eGFR surrogate event rate (%)",
                      xaxis_range=[0, 100], yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True, config={"modeBarButtonsToRemove": ["toImage"]})
    st.table(summary)
    st.caption("The choice of 10 versus 15 changes which censored patients can be classified. "
               "KFRE risk is based on earliest eligible complete baseline; subsequent follow-up dates determine the surrogate event.")

st.divider()
st.caption("Approved educational EDA dataset • descriptive only • not for clinical decisions")

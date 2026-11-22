"""Create disclosure-screened aggregate CSVs, strictly from a LOCAL source CSV.

This script must be run outside the public Streamlit service, only after the
practice's appropriate privacy / data-governance review. Review ALL outputs
before committing. k-anonymity-like cell suppression is not de-identification.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest

GFR_ORDER = ["G1", "G2", "G3a", "G3b", "G4", "G5"]
LABS = ["calcium", "potassium", "hemoglobin", "phosphorus", "glucose", "co2", "albumin", "creatinine"]
VITALS = ["systolic_bp", "diastolic_bp", "bmi", "MAP"]
FEATURES = ["acr", *LABS, *VITALS]
BOUNDS = {
    "egfr": (0, 150), "acr": (0, None), "creatinine": (0, 20),
    "glucose": (0, 600), "calcium": (0, 15), "potassium": (0, 8),
    "hemoglobin": (0, 18), "phosphorus": (0, 15), "co2": (0, 45),
}
LOCATIONS = {
    "cullman": "Cullman", "nephrology of north alabama cullman": "Cullman",
    "c-nephrology of north alabama": "Cullman", "cullman-nephrology of north alabama": "Cullman",
    "cullman nephrology of north alabama pc": "Cullman",
    "cullman nephrology of north alabama": "Cullman",
    "decatur": "Decatur", "decatur nephrology of north alabama pc": "Decatur",
    "decatur nephrology of north alabama": "Decatur", "nephrology of north alabama decatur": "Decatur",
    "nephrology of north alabama": "Unspecified", "decatur morgan hospital ip": "Decatur Morgan Hospital IP",
}


def daily(df, column, date_col="lab_date"):
    dates = df[date_col]
    if date_col == "encounter_date":
        dates = dates.fillna(df["lab_date"])
    work = pd.DataFrame({"study_id": df.study_id, "date": dates, "value": df[column]})
    return work.dropna().groupby(["study_id", "date"], as_index=False).value.median()


def slope(group):
    group = group.sort_values("date")
    if len(group) < 3:
        return np.nan
    t = (group.date - group.date.min()).dt.days.to_numpy() / 365.25
    if t.max() < 1:
        return np.nan
    return float(np.polyfit(t, group.value.to_numpy(), 1)[0])


def normalize_category(series):
    s = series.astype("string").str.strip()
    return s.mask(s.str.lower().isin(["", "nan", "none", "null", "n/a", "<na>"]))


def fit_ols(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.unique(x).size < 2 or np.std(y) == 0:
        return None
    m, b = np.polyfit(x, y, 1)
    yhat = m*x+b
    return float(m), float(b), float(1-((y-yhat)**2).sum()/((y-y.mean())**2).sum())


def kfre_8(row, years):
    sex = str(row.sex).strip().lower()
    male = int(sex in ("m", "male"))
    lp = (-0.1992*(row.age/10-7.036) + 0.1602*(male-0.5642)
          -0.4919*(row.egfr/5-7.222) + 0.3364*(np.log(row.acr)-5.137)
          -0.3441*(row.albumin-3.997) + 0.2604*(row.phosphorus-3.916)
          -0.07354*(row.co2-25.57) - 0.2228*(row.calcium-9.355))
    return float(1-(0.9780 if years == 2 else 0.9301)**np.exp(lp))


def window_event(history, start, end, threshold):
    """Match the notebook: 3-month window, >=2 distinct dates, mean < threshold.

    The dates need not span three months. Return window end or NaT.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()
    obs = history.copy()
    obs["date"] = obs["date"].dt.normalize()
    obs = obs.groupby("date", as_index=False).value.median().sort_values("date")
    obs = obs[obs.date.between(start, end)]
    if len(obs) < 2:
        return pd.NaT
    last_start = end - pd.DateOffset(months=3)
    if last_start < start:
        return pd.NaT
    starts = {start, last_start}
    for dt in obs.date:
        starts.update([dt, dt+pd.Timedelta(days=1), dt-pd.DateOffset(months=3),
                       dt-pd.DateOffset(months=3)+pd.Timedelta(days=1)])
    for a in sorted(starts):
        if start <= a <= last_start:
            b = a + pd.DateOffset(months=3)
            group = obs[obs.date.between(a, b)]
            if len(group) >= 2 and group.value.mean() < threshold:
                return b
    return pd.NaT


def build_kfre(df, gfr, min_count):
    required = ["study_id", "lab_date", "age", "sex", "egfr", "acr", "albumin", "phosphorus", "co2", "calcium"]
    candidates = df[required].dropna().copy()
    candidates.sex = candidates.sex.astype(str).str.lower().str.strip()
    candidates = candidates[
        (candidates.age >= 18) & candidates.egfr.between(15, 60, inclusive="left")
        & (candidates.acr > 0) & candidates.sex.isin(["m", "male", "f", "female"])
        & (candidates[["albumin", "phosphorus", "co2", "calcium"]] > 0).all(axis=1)
    ]
    baseline = (candidates.sort_values(["study_id", "lab_date"], kind="stable")
                .drop_duplicates("study_id").rename(columns={"lab_date": "baseline_date"}))
    groups = {pid: group for pid, group in gfr.groupby("study_id")}
    baseline = baseline[baseline.apply(lambda r: pd.isna(window_event(
        groups[r.study_id].loc[groups[r.study_id].date.dt.normalize() < r.baseline_date.normalize()],
        groups[r.study_id].date.min()-pd.DateOffset(months=3), r.baseline_date, 15)), axis=1)]
    outputs = []
    band_names = ["0–5%", "5–10%", "10–20%", "20%+"]
    for years in (2, 5):
        risk = baseline.copy()
        risk["risk"] = risk.apply(lambda r: kfre_8(r, years), axis=1)
        risk["horizon"] = risk.baseline_date + pd.DateOffset(years=years)
        per_threshold = {}
        for threshold in (15, 10):
            outcomes = []
            for row in risk.itertuples():
                obs = groups[row.study_id]
                future = obs[(obs.date.dt.normalize() > row.baseline_date.normalize()) &
                             (obs.date.dt.normalize() <= row.horizon.normalize())]
                event_date = window_event(future.rename(columns={"date":"date"}), row.baseline_date, row.horizon, threshold)
                if pd.notna(event_date):
                    outcome = 1
                elif obs.date.max() >= row.horizon:
                    outcome = 0
                else:
                    outcome = np.nan
                outcomes.append(outcome)
            per_threshold[threshold] = np.array(outcomes, dtype=float)
        paired_mask = np.isfinite(per_threshold[15]) & np.isfinite(per_threshold[10])
        for threshold in (15, 10):
            for cohort, mask in [("paired", paired_mask),
                                 ("threshold_specific", np.isfinite(per_threshold[threshold]))]:
                sub = risk.loc[mask, ["study_id", "risk"]].copy()
                sub["observed"] = per_threshold[threshold][mask]
                sub["risk_band"] = pd.cut(sub.risk, [0, .05, .10, .20, 1.00001],
                                          labels=band_names, right=False, include_lowest=True)
                agg = sub.groupby("risk_band", observed=True).agg(
                    patients=("study_id", "size"), mean_predicted=("risk", "mean"),
                    observed_events=("observed", "sum")).reset_index()
                for row in agg.itertuples():
                    if row.patients < min_count:
                        continue
                    ci = binomtest(int(row.observed_events), int(row.patients)).proportion_ci(
                        confidence_level=.95, method="wilson")
                    outputs.append(dict(years=years, threshold=threshold, cohort=cohort,
                                        risk_band=str(row.risk_band), patients=int(row.patients),
                                        mean_predicted=float(row.mean_predicted),
                                        observed_events=int(row.observed_events),
                                        observed_rate=float(row.observed_events/row.patients),
                                        ci_lower=ci.low, ci_upper=ci.high))
    return pd.DataFrame(outputs, columns=["years", "threshold", "cohort", "risk_band", "patients", "mean_predicted", "observed_events", "observed_rate", "ci_lower", "ci_upper"])


def export(source, out, min_count=30):
    df = pd.read_csv(source, dtype={"study_id": "string"}).drop_duplicates().copy()
    required = ["study_id", "lab_date", "date_of_birth", "encounter_date", "sex", "race", "insurance", "location", "egfr", "acr", *LABS, *VITALS[:-1]]
    missing = set(required) - set(df)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for c in ["lab_date", "encounter_date", "date_of_birth"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ["egfr", "acr", *LABS, *VITALS[:-1]]:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["MAP"] = (df.systolic_bp + 2*df.diastolic_bp)/3
    for column, (low, high) in BOUNDS.items():
        invalid = (df[column] < low) | ((df[column] > high) if high is not None else False)
        df.loc[invalid, column] = np.nan
    df["age"] = (df.lab_date-df.date_of_birth).dt.days / 365.25
    df.loc[df.age < 0, "age"] = np.nan
    n_rows, n_patients = len(df), df.study_id.nunique()
    gfr = daily(df, "egfr")
    rows = []
    for pid, group in gfr.groupby("study_id"):
        m = slope(group)
        if np.isfinite(m):
            group = group.sort_values("date")
            rows.append({"study_id":pid, "annual_egfr_change":m,
                         "baseline_egfr":float(group.value.iloc[0]),
                         "n_observations":len(group)})
    patient = pd.DataFrame(rows)
    if patient.empty:
        raise ValueError("No eligible patient slopes in input.")
    patient["gfr_category"] = pd.cut(patient.baseline_egfr,
        [-np.inf,15,30,45,60,90,np.inf], labels=["G5","G4","G3b","G3a","G2","G1"], right=False).astype(str)
    categories = df[["study_id", "lab_date", "encounter_date", "sex", "race", "insurance", "location"]].copy()
    for c in ["sex", "race", "insurance", "location"]:
        categories[c] = normalize_category(categories[c])
    categories["location"] = categories.location.str.lower().map(LOCATIONS).fillna(categories.location)
    categories["record_date"] = categories.encounter_date.fillna(categories.lab_date)
    categories = categories.sort_values("record_date", kind="stable")
    cats = categories.groupby("study_id")[["sex", "race", "insurance", "location"]].agg(
        lambda x: x.dropna().iloc[-1] if x.notna().any() else pd.NA).reset_index()
    patient = patient.merge(cats, on="study_id", how="left")
    # Publish one-dimensional marginals only, preventing arbitrary high-dimensional drilldown.
    overview = []
    for dimension in ["gfr_category", "sex", "race", "insurance", "location"]:
        for val, sub in patient.dropna(subset=[dimension]).groupby(dimension):
            if len(sub) < min_count:
                continue
            overview.append(dict(dimension=dimension, category=str(val), patients=len(sub),
                                 median_annual_slope=float(sub.annual_egfr_change.median()),
                                 median_baseline_egfr=float(sub.baseline_egfr.median()),
                                 pct_declining=float((sub.annual_egfr_change<0).mean()*100)))
    # Large bins only; do not expose small counts or exact outlier ranges.
    distributions = []
    edges = np.arange(-20, 21, 1)
    for group in ["All", *GFR_ORDER]:
        sub = patient if group == "All" else patient[patient.gfr_category == group]
        counts, bins = np.histogram(sub.annual_egfr_change.clip(-19.999,19.999), bins=edges)
        for j, count in enumerate(counts):
            if count >= min_count:
                distributions.append(dict(gfr_category=group, bin_center=float((bins[j]+bins[j+1])/2),patients=int(count)))
    associations, regressions = [], []
    for feature in FEATURES:
        dcol = "encounter_date" if feature in VITALS else "lab_date"
        source = daily(df, feature, dcol)
        med = source.groupby("study_id").value.agg(["median", "count"]).reset_index()
        med = med[med["count"]>=2]
        merged = patient[["study_id", "gfr_category", "annual_egfr_change"]].merge(med[["study_id", "median"]], on="study_id")
        for group in ["All", *GFR_ORDER]:
            sub = merged if group == "All" else merged[merged.gfr_category == group]
            intervals = [("All", sub)] if feature != "acr" else [
                ("A1", sub[(sub["median"]>0)&(sub["median"]<30)]),
                ("A2", sub[(sub["median"]>=30)&(sub["median"]<300)]),
                ("A3", sub[sub["median"]>=300])]
            for band, s in intervals:
                s = s.dropna(subset=["median", "annual_egfr_change"])
                if feature == "acr":
                    s = s[s["median"]>0]
                if len(s) < min_count*2:
                    continue
                x = np.log10(s["median"].to_numpy()) if feature=="acr" else s["median"].to_numpy()
                y = s.annual_egfr_change.to_numpy()
                fit = fit_ols(x,y)
                if fit:
                    m,b,r2=fit
                    regressions.append(dict(feature=feature,gfr_category=group,band=band,patients=len(s),slope=m,intercept=b,r_squared=r2))
                # Equal-sized rank bins, aggregate medians, with disclosure floor.
                s = s.copy()
                s["q"] = pd.qcut(s["median"].rank(method="first"), q=min(8,len(s)//min_count), labels=False, duplicates="drop")
                for _, bucket in s.groupby("q"):
                    if len(bucket) < min_count:
                        continue
                    associations.append(dict(feature=feature,gfr_category=group,band=band,
                                             patients=len(bucket),x_median=float(bucket["median"].median()),
                                             y_median=float(bucket.annual_egfr_change.median())))
    kfre = build_kfre(df, gfr, min_count)
    out.mkdir(parents=True,exist_ok=True)
    tables = {"overview":pd.DataFrame(overview), "distributions":pd.DataFrame(distributions),
              "associations":pd.DataFrame(associations), "regressions":pd.DataFrame(regressions), "kfre":kfre}
    for name, table in tables.items():
        table.to_csv(out/f"{name}.csv", index=False)
    (out/"metadata.json").write_text(json.dumps({"data_mode":"approved_aggregates", "n_rows":n_rows,
        "n_patients":int(n_patients), "n_slopes":len(patient), "min_group_size":min_count},indent=2),encoding="utf-8")
    print("Wrote aggregate tables to",out)
    print("REVIEW ALL TABLES for disclosure risk before committing or publishing.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Path to locally held, authorized CSV. Never use a public URL.")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--min-count", type=int, default=30)
    p.add_argument("--approved-for-public-aggregate", action="store_true", help="Confirms authorization and review of aggregated disclosures (not automatic compliance)")
    a = p.parse_args()
    if not a.approved_for_public_aggregate:
        p.error("Must explicitly confirm public-aggregate authorization before generating real-data outputs.")
    if a.min_count < 30:
        p.error("Minimum publication group size is 30.")
    export(a.input,a.out,a.min_count)

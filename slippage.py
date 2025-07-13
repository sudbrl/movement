# slippage_report_full.py
import io
import traceback
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency

# ----------------------------- #
#  Constants / Configuration
# ----------------------------- #
KEEP_COLUMNS = [
    "Branch Name",
    "Main Code",
    "Ac Type Desc",
    "Name",
    "Limit",
    "Balance",
    "Provision",
]

PROVISION_MAP: Dict[str, int] = {"G": 1, "W": 2, "S": 3, "D": 4, "B": 5}
CATEGORY_NAMES: Dict[str, str] = {
    "G": "Good",
    "W": "Watchlist",
    "S": "Substandard",
    "D": "Doubtful",
    "B": "Bad",
}
CATEGORY_ORDER: List[str] = ["Good", "Substandard", "Doubtful", "Bad"]

# ----------------------------- #
#  Pre-processing
# ----------------------------- #
def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    missing = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df[KEEP_COLUMNS]

    # Numeric coercion
    df["Limit"] = pd.to_numeric(df["Limit"], errors="coerce")
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")
    df = df.dropna(subset=["Limit", "Balance"])
    df = df[df["Limit"] != 0]

    # Provision code cleaning
    df["Provision_initial"] = (
        df["Provision"].astype(str).str.upper().str[0]
    )

    invalid = df[~df["Provision_initial"].isin(PROVISION_MAP.keys())]
    if not invalid.empty:
        raise ValueError(
            f"Invalid provision codes: {invalid['Provision_initial'].unique().tolist()}"
        )

    df["Provision_rank"] = df["Provision_initial"].map(PROVISION_MAP)
    df["Provision_category"] = df["Provision_initial"].map(CATEGORY_NAMES)

    return df

# ----------------------------- #
#  Slippage detection
# ----------------------------- #
def detect_slippage(df_prev: pd.DataFrame, df_curr: pd.DataFrame) -> pd.DataFrame:
    prev = df_prev[["Main Code", "Provision_rank", "Provision_category"]].rename(
        columns={
            "Provision_rank": "Provision_rank_prev",
            "Provision_category": "Provision_category_prev",
        }
    )
    curr = df_curr[["Main Code", "Provision_rank", "Provision_category"]].rename(
        columns={
            "Provision_rank": "Provision_rank_curr",
            "Provision_category": "Provision_category_curr",
        }
    )

    merged = pd.merge(prev, curr, on="Main Code", how="inner")
    change_keys = merged["Main Code"]

    full = df_curr[df_curr["Main Code"].isin(change_keys)].copy()
    prev_details = df_prev.set_index("Main Code")[
        ["Provision_rank", "Provision_category"]
    ].rename(
        columns={
            "Provision_rank": "Provision_rank_prev",
            "Provision_category": "Provision_category_prev",
        }
    )

    full = full.set_index("Main Code")
    full["Provision_rank_prev"] = prev_details["Provision_rank_prev"]
    full["Provision_category_prev"] = prev_details["Provision_category_prev"]

    full["Movement"] = full.apply(
        lambda row: (
            "Slippage"
            if row["Provision_rank"] > row["Provision_rank_prev"]
            else "Upgrade"
            if row["Provision_rank"] < row["Provision_rank_prev"]
            else "Stable"
        ),
        axis=1,
    )

    full.reset_index(inplace=True)

    columns_out = [
        "Branch Name",
        "Main Code",
        "Ac Type Desc",
        "Name",
        "Limit",
        "Balance",
        "Provision_category_prev",
        "Provision_category_curr",
        "Movement",
    ]
    return full[columns_out].rename(
        columns={"Provision_category": "Provision_category_curr"}
    )

# ----------------------------- #
#  Pivot helpers
# ----------------------------- #
def category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.pivot_table(
        df,
        index="Provision_category_prev",
        columns="Provision_category_curr",
        aggfunc="size",
        fill_value=0,
    )

    available_cols = [col for col in CATEGORY_ORDER if col in matrix.columns]
    matrix = matrix.reindex(columns=available_cols)
    return matrix.reset_index()

def summarize_matrix(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    matrices = []
    for g in df[group_col].unique():
        sub = df[df[group_col] == g]
        mat = pd.pivot_table(
            sub,
            index="Provision_category_prev",
            columns="Provision_category_curr",
            aggfunc="size",
            fill_value=0,
        )
        available_cols = [col for col in CATEGORY_ORDER if col in mat.columns]
        mat = mat.reindex(columns=available_cols)
        mat[group_col] = g
        matrices.append(mat.reset_index())
    summary_df = pd.concat(matrices, ignore_index=True, sort=False)
    cols = [group_col] + [c for c in summary_df.columns if c != group_col]

    for col in summary_df.columns:
        if summary_df[col].dtype in ["float64", "int64"]:
            summary_df[col] = summary_df[col].fillna(0).astype(int)

    return summary_df[cols]

# ----------------------------- #
#  Risk / transition metrics
# ----------------------------- #
def build_risk_metrics(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Shannon Entropy, WAR, Upgrade/Downgrade Ratio, Cure Rate,
    ASM, Hazard Rate, Half-Life, and Chi-square p-value from the
    transition matrix and appends them as new columns.
    """
    mat = matrix_df.set_index("Provision_category_prev")
    mat = mat.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    T = mat.values.astype(float)

    # Steady-state
    marginal = T.sum(axis=1)
    pi = marginal / marginal.sum()
    pi = pi[pi > 0]
    shannon = -np.sum(pi * np.log(pi)) if pi.size else np.nan

    # Weighted Average Rate
    ranks = np.array([PROVISION_MAP[c] for c in CATEGORY_ORDER])
    row_probs = T / T.sum(axis=1, keepdims=True)
    avg_rank = (row_probs * ranks).sum(axis=1)
    war = np.average(avg_rank, weights=marginal)

    # Upgrade / Downgrade ratio
    downgrade = np.triu(T, k=1).sum()
    upgrade = np.tril(T, k=-1).sum()
    ud_ratio = upgrade / downgrade if downgrade else np.nan

    # Cure Rate
    non_good_prev = marginal - T[:, CATEGORY_ORDER.index("Good")]
    cure = T[:, CATEGORY_ORDER.index("Good")].sum() / non_good_prev.sum() if non_good_prev.sum() else np.nan

    # Average State Migration (ASM)
    diff = np.abs(ranks[:, None] - ranks[None, :])
    asm = np.sum(T * diff) / T.sum()

    # Hazard & Half-Life
    hazard = 1 - np.diag(T) / marginal
    avg_hazard = np.average(hazard, weights=marginal)
    half_life = (
        np.log(0.5) / np.log(1 - avg_hazard)
        if avg_hazard and 0 < avg_hazard < 1
        else np.nan
    )

    # Chi-square
    _, p_value, _, _ = chi2_contingency(T)

    # Append
    out = matrix_df.copy()
    out["ShananonEntropy"] = shannon
    out["WAR"] = war
    out["UpgradeDowngradeRatio"] = ud_ratio
    out["CureRate"] = cure
    out["ASM"] = asm
    out["HazardRate"] = avg_hazard
    out["HalfLife"] = half_life
    out["ChiSquare_pvalue"] = p_value
    return out

# ----------------------------- #
#  Excel export
# ----------------------------- #
def generate_excel(slippage_df, branch_summary, ac_type_summary, matrix):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        slippage_df.to_excel(writer, index=False, sheet_name="Slippage Accounts")
        branch_summary.to_excel(writer, index=False, sheet_name="Summary by Branch")
        ac_type_summary.to_excel(writer, index=False, sheet_name="Summary by Ac Type")

        enriched_matrix = build_risk_metrics(matrix)
        enriched_matrix.to_excel(writer, index=False, sheet_name="Category Matrix")
    output.seek(0)
    return output

# ----------------------------- #
#  Streamlit UI
# ----------------------------- #
st.set_page_config(page_title="📉 Slippage Report Generator", layout="centered")
st.title("📊 Slippage Report Generator")

st.header("📁 Upload Excel Files")
uploaded_curr = st.file_uploader("📅 Upload Current Period Excel File", type=["xlsx"])
uploaded_prev = st.file_uploader("🕰️ Upload Previous Period Excel File", type=["xlsx"])

if uploaded_curr and uploaded_prev:
    with st.spinner("🔄 Generating Slippage Report..."):
        try:
            df_curr = pd.read_excel(uploaded_curr, header=0)
            df_prev = pd.read_excel(uploaded_prev, header=0)

            df_curr = preprocess_df(df_curr)
            df_prev = preprocess_df(df_prev)

            slippage_df = detect_slippage(df_prev, df_curr)

            branch_summary = summarize_matrix(slippage_df, "Branch Name")
            ac_type_summary = summarize_matrix(slippage_df, "Ac Type Desc")
            matrix = category_matrix(slippage_df)

            excel_data = generate_excel(slippage_df, branch_summary, ac_type_summary, matrix)

            st.success("✅ Report Ready!")

            st.download_button(
                label="📤 Download Excel Report",
                data=excel_data,
                file_name=f"slippage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

        except Exception as e:
            st.error("❌ An error occurred during processing.")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

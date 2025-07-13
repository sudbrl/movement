import io
import traceback
from datetime import datetime
from typing import Dict, List

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
CATEGORY_ORDER: List[str] = ["Good", "Watchlist", "Substandard", "Doubtful", "Bad"]

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

    df["Limit"] = pd.to_numeric(df["Limit"], errors="coerce")
    df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce")
    df = df.dropna(subset=["Limit", "Balance"])
    df = df[df["Limit"] != 0]

    df["Provision_initial"] = df["Provision"].astype(str).str.upper().str[0]

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
    full["Provision_category_curr"] = full["Provision_category"]

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
    return full[columns_out]

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
    matrix = matrix.reindex(index=CATEGORY_ORDER, columns=available_cols, fill_value=0)
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
        mat = mat.reindex(index=CATEGORY_ORDER, columns=available_cols, fill_value=0)
        mat[group_col] = g
        matrices.append(mat.reset_index())
    summary_df = pd.concat(matrices, ignore_index=True, sort=False)
    cols = [group_col] + [c for c in summary_df.columns if c != group_col]

    for col in summary_df.columns:
        if summary_df[col].dtype in ["float64", "int64"]:
            summary_df[col] = summary_df[col].fillna(0).astype(int)

    return summary_df[cols]

# ----------------------------- #
#  Risk / transition metrics with Excel formulas
# ----------------------------- #
def build_risk_metrics(matrix_df: pd.DataFrame) -> pd.DataFrame:
    mat = matrix_df.set_index("Provision_category_prev")
    mat = mat.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    result_rows = []

    ranks = np.array([PROVISION_MAP[cat[0]] for cat in CATEGORY_ORDER])  # G, W, S, D, B

    for index, row in mat.iterrows():
        row_values = row.values.astype(float)
        row_sum = row_values.sum()

        if row_sum == 0:
            entropy = war = ud_ratio = cure_rate = hazard = half_life = np.nan
        else:
            probs = row_values / row_sum

            # Shannon Entropy (log base 2)
            entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])

            # Weighted Average Rank (WAR)
            current_rank = PROVISION_MAP.get(index[0], None)
            war = np.sum(probs * ranks)

            # Upgrade/Downgrade ratio
            if current_rank is not None:
                rank_diffs = ranks - current_rank
                upgrades = probs[rank_diffs < 0].sum()
                downgrades = probs[rank_diffs > 0].sum()
                ud_ratio = upgrades / downgrades if downgrades > 0 else np.nan
            else:
                ud_ratio = np.nan

            # Cure Rate (to "Good")
            cure_idx = CATEGORY_ORDER.index("Good")
            cure_rate = row_values[cure_idx] / (row_sum - row_values[cure_idx]) if (row_sum - row_values[cure_idx]) > 0 else np.nan

            # Hazard Rate (to "Bad")
            bad_idx = CATEGORY_ORDER.index("Bad")
            hazard = row_values[bad_idx] / row_sum

            # Half-life based on hazard
            half_life = np.log(0.5) / np.log(1 - hazard) if 0 < hazard < 1 else np.nan

        result_rows.append({
            "Provision_category_prev": index,
            "ShannonEntropy": entropy,
            "WAR": war,
            "UpgradeDowngradeRatio": ud_ratio,
            "CureRate": cure_rate,
            "HazardRate": hazard,
            "HalfLife": half_life
        })

    metrics_df = pd.DataFrame(result_rows)
    full = pd.concat([mat.reset_index(), metrics_df.drop(columns=["Provision_category_prev"])], axis=1)
    return full

# ----------------------------- #
#  Excel export with formulas for user inspection
# ----------------------------- #
def generate_excel(slippage_df, branch_summary, ac_type_summary, matrix):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Write slippage data
        slippage_df.to_excel(writer, index=False, sheet_name="Slippage Accounts")
        branch_summary.to_excel(writer, index=False, sheet_name="Summary by Branch")
        ac_type_summary.to_excel(writer, index=False, sheet_name="Summary by Ac Type")

        # Write transition matrix with provision categories as rows/columns
        matrix.to_excel(writer, index=False, sheet_name="Transition Matrix")

        # Write risk metrics table with formulas linked to the matrix sheet
        workbook = writer.book
        ws_metrics = workbook.add_worksheet("Matrix Metrics")
        writer.sheets["Matrix Metrics"] = ws_metrics

        # Write headers
        headers = [
            "Provision_category_prev",
            *CATEGORY_ORDER,
            "ShannonEntropy",
            "WAR",
            "UpgradeDowngradeRatio",
            "CureRate",
            "HazardRate",
            "HalfLife",
        ]
        ws_metrics.write_row(0, 0, headers)

        # Write provision categories in col A
        for i, category in enumerate(CATEGORY_ORDER, start=1):
            ws_metrics.write(i, 0, category)

        n_cats = len(CATEGORY_ORDER)
        # Write transition counts from "Transition Matrix" sheet as references
        # Matrix starts at A1 in Transition Matrix sheet, provision_category_prev in col A
        # Row i+2 in Excel because of header and 1-based index
        for i in range(n_cats):
            # Write provision categories as columns headers (row 0)
            ws_metrics.write(0, i + 1, CATEGORY_ORDER[i])
            for j in range(n_cats):
                # Excel row and col index offset by +2 and +2 to account for header and 1-based index on Transition Matrix
                cell_ref = f"'Transition Matrix'!{chr(ord('B') + j)}{i + 2}"
                ws_metrics.write_formula(i + 1, j + 1, f"={cell_ref}")

        # Write formulas for metrics for each row
        for i in range(n_cats):
            row_excel = i + 2  # Excel row (1-based + header)

            # Total row sum
            total_formula = f"=SUM(B{row_excel}:F{row_excel})"

            # Shannon Entropy formula (using log base 2)
            # =-SUMPRODUCT((B2:F2/{total})*(LOG(B2:F2/{total},2))

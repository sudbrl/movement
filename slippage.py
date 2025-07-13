import io
import numpy as np
import pandas as pd

# Constants
PROVISION_MAP = {"G": 1, "W": 2, "S": 3, "D": 4, "B": 5}
CATEGORY_NAMES = {"G": "Good", "W": "Watchlist", "S": "Substandard", "D": "Doubtful", "B": "Bad"}
CATEGORY_ORDER = ["Good", "Watchlist", "Substandard", "Doubtful", "Bad"]

def preprocess_df(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    df["Provision_initial"] = df["Provision"].astype(str).str.upper().str[0]
    df["Provision_category"] = df["Provision_initial"].map(CATEGORY_NAMES)
    df["Provision_rank"] = df["Provision_initial"].map(PROVISION_MAP)
    return df

def detect_slippage(df_prev, df_curr):
    prev = df_prev[["Main Code", "Provision_rank", "Provision_category"]].rename(
        columns={"Provision_rank": "Provision_rank_prev", "Provision_category": "Provision_category_prev"}
    )
    curr = df_curr[["Main Code", "Provision_rank", "Provision_category"]].rename(
        columns={"Provision_rank": "Provision_rank_curr", "Provision_category": "Provision_category_curr"}
    )
    merged = pd.merge(prev, curr, on="Main Code", how="inner")
    merged["Movement"] = merged.apply(
        lambda r: "Slippage" if r.Provision_rank_curr > r.Provision_rank_prev else
                  "Upgrade" if r.Provision_rank_curr < r.Provision_rank_prev else "Stable", axis=1)
    return merged

def build_transition_matrix(df_slippage):
    matrix = pd.pivot_table(
        df_slippage,
        index="Provision_category_prev",
        columns="Provision_category_curr",
        aggfunc="size",
        fill_value=0,
    )
    # Reindex rows and columns to keep consistent category order
    matrix = matrix.reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)
    matrix = matrix.reset_index()
    return matrix

def build_risk_metrics(matrix_df):
    mat = matrix_df.set_index("Provision_category_prev").reindex(CATEGORY_ORDER, fill_value=0)[CATEGORY_ORDER]

    results = []
    ranks = np.array([PROVISION_MAP[cat[0]] for cat in CATEGORY_ORDER])

    for category in CATEGORY_ORDER:
        row = mat.loc[category].values.astype(float)
        total = row.sum()
        if total == 0:
            # no transitions - metrics are NaN
            results.append([category] + [np.nan]*6)
            continue
        probs = row / total

        # Shannon Entropy
        entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])

        # WAR = sum of (prob * rank)
        war = np.sum(probs * ranks)

        # Upgrade/Downgrade ratio
        current_rank = PROVISION_MAP[category[0]]
        rank_diffs = ranks - current_rank
        upgrades = probs[rank_diffs < 0].sum()
        downgrades = probs[rank_diffs > 0].sum()
        ud_ratio = upgrades / downgrades if downgrades > 0 else np.nan

        # Cure Rate = proportion transitioning to Good
        cure_rate = row[CATEGORY_ORDER.index("Good")] / total

        # Hazard Rate = proportion transitioning to Bad
        hazard_rate = row[CATEGORY_ORDER.index("Bad")] / total

        # Half Life = ln(0.5) / ln(1 - hazard_rate), if hazard_rate between 0 and 1
        half_life = np.log(0.5) / np.log(1 - hazard_rate) if 0 < hazard_rate < 1 else np.nan

        results.append(
            [
                category,
                entropy,
                war,
                ud_ratio,
                cure_rate,
                hazard_rate,
                half_life,
            ]
        )

    metrics_df = pd.DataFrame(
        results,
        columns=[
            "Provision_category_prev",
            "ShannonEntropy",
            "WAR",
            "UpgradeDowngradeRatio",
            "CureRate",
            "HazardRate",
            "HalfLife",
        ],
    )

    return metrics_df

def generate_excel(slippage_df, branch_summary, ac_type_summary, matrix):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Write basic sheets
        slippage_df.to_excel(writer, sheet_name="Slippage Accounts", index=False)
        branch_summary.to_excel(writer, sheet_name="Summary by Branch", index=False)
        ac_type_summary.to_excel(writer, sheet_name="Summary by Ac Type", index=False)
        matrix.to_excel(writer, sheet_name="Transition Matrix", index=False)

        workbook = writer.book

        # Add Matrix Metrics sheet with formulas
        ws = workbook.add_worksheet("Matrix Metrics")
        writer.sheets["Matrix Metrics"] = ws

        # Headers
        headers = ["Provision_category_prev"] + CATEGORY_ORDER + [
            "ShannonEntropy",
            "WAR",
            "UpgradeDowngradeRatio",
            "CureRate",
            "HazardRate",
            "HalfLife",
        ]
        ws.write_row(0, 0, headers)

        n = len(CATEGORY_ORDER)
        # Write Provision categories vertically
        for i, cat in enumerate(CATEGORY_ORDER, start=1):
            ws.write(i, 0, cat)

        # Write category headers horizontally
        for j, cat in enumerate(CATEGORY_ORDER, start=1):
            ws.write(0, j, cat)

        # Fill the matrix counts with Excel formulas linking 'Transition Matrix' sheet
        # 'Transition Matrix'!$B$2:$F$6 where rows are categories, cols are categories (B=Good, C=Watchlist, etc)
        for i in range(n):
            for j in range(n):
                # Excel references: rows start at 2 (because row 1 is header), cols start at B (2nd col)
                excel_row = i + 2
                excel_col = chr(ord('B') + j)
                formula = f"='Transition Matrix'!${excel_col}${excel_row}"
                ws.write_formula(i + 1, j + 1, formula)

        # Now write formulas for metrics columns per row
        for i in range(1, n + 1):
            row_excel = i + 1  # 1-based index + header

            # Define Excel ranges for the matrix row values (B to F)
            row_range = f"B{row_excel}:F{row_excel}"

            # Total sum of row
            total_sum = f"=SUM({row_range})"
            ws.write_formula(i, n + 1, f"=IF({total_sum}=0, NA(), -SUMPRODUCT(({row_range}/{total_sum})*(IF({row_range}=0,0,LOG({row_range}/{total_sum},2)))))")  # ShannonEntropy

            # WAR
            # Weights: ranks of categories: Good=1, Watchlist=2, etc.
            ranks = [1, 2, 3, 4, 5]
            rank_formula = "+".join([f"({chr(ord('B')+j)}{row_excel}/{total_sum})*{ranks[j]}" for j in range(n)])
            ws.write_formula(i, n + 2, f"=IF({total_sum}=0, NA(), {rank_formula})")

            # UpgradeDowngradeRatio:
            # current rank = ranks[i-1], upgrades = sum probs where rank < current, downgrades = sum probs where rank > current
            current_rank = ranks[i-1]
            upgrades = "+".join([f"({chr(ord('B')+j)}{row_excel}/{total_sum})" for j, r in enumerate(ranks) if r < current_rank])
            downgrades = "+".join([f"({chr(ord('B')+j)}{row_excel}/{total_sum})" for j, r in enumerate(ranks) if r > current_rank])
            upgrades = upgrades if upgrades else "0"
            downgrades = downgrades if downgrades else "0"
            # Avoid division by zero: IF downgrades=0 then NA()
            ud_formula = f"=IF({total_sum}=0, NA(), IF(({downgrades})=0, NA(), ({upgrades})/({downgrades})))"
            ws.write_formula(i, n + 3, ud_formula)

            # CureRate = proportion to Good = column B (Good) / total
            cure_formula = f"=IF({total_sum}=0, NA(), B{row_excel}/{total_sum})"
            ws.write_formula(i, n + 4, cure_formula)

            # HazardRate = proportion to Bad = column F (Bad) / total
            hazard_formula = f"=IF({total_sum}=0, NA(), F{row_excel}/{total_sum})"
            ws.write_formula(i, n + 5, hazard_formula)

            # HalfLife = LN(0.5) / LN(1 - HazardRate)
            half_life_formula = f"=IF(AND({hazard_formula}>0,{hazard_formula}<1), LN(0.5)/LN(1-{hazard_formula}), NA())"
            ws.write_formula(i, n + 6, half_life_formula)

    return output.getvalue()

# Usage example (assuming df_prev and df_curr are preloaded dataframes):
# df_prev = preprocess_df(df_prev_raw)
# df_curr = preprocess_df(df_curr_raw)
# slippage_df = detect_slippage(df_prev, df_curr)
# transition_matrix = build_transition_matrix(slippage_df)
# risk_metrics = build_risk_metrics(transition_matrix)
# branch_summary = ... # summary df by branch
# ac_type_summary = ... # summary df by account type
# excel_bytes = generate_excel(slippage_df, branch_summary, ac_type_summary, transition_matrix)
# with open("slippage_report.xlsx", "wb") as f:
#     f.write(excel_bytes)

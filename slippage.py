import streamlit as st
import pandas as pd
import io
from datetime import datetime
import traceback

st.set_page_config(page_title="📉 Slippage Report Generator", layout="centered")
st.title("📊 Slippage Report Generator")

# -----------------------------
# Constants
# -----------------------------
KEEP_COLUMNS = ['Branch Name', 'Main Code', 'Ac Type Desc', 'Name', 'Limit', 'Balance', 'Provision']
PROVISION_MAP = {'G': 1, 'W': 2, 'S': 3, 'D': 4, 'B': 5}
CATEGORY_NAMES = {
    'G': 'Good',
    'W': 'Watchlist',
    'S': 'Substandard',
    'D': 'Doubtful',
    'B': 'Bad'
}
CATEGORY_ORDER = ['Good', 'Substandard', 'Doubtful', 'Bad']

# -----------------------------
# Utility Functions
# -----------------------------
def preprocess_df(df):
    df.columns = df.columns.str.strip()

    missing = [col for col in KEEP_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    df = df[KEEP_COLUMNS]

    df['Limit'] = pd.to_numeric(df['Limit'], errors='coerce')
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df = df.dropna(subset=['Limit', 'Balance'])
    df = df[df['Limit'] != 0]

    df['Provision_initial'] = df['Provision'].astype(str).str.upper().str[0]

    invalid = df[~df['Provision_initial'].isin(PROVISION_MAP.keys())]
    if not invalid.empty:
        raise ValueError(f"Invalid provision codes found: {invalid['Provision_initial'].unique().tolist()}")

    df['Provision_rank'] = df['Provision_initial'].map(PROVISION_MAP)
    df['Provision_category'] = df['Provision_initial'].map(CATEGORY_NAMES)

    return df

def detect_slippage(df_prev, df_curr):
    prev = df_prev[['Main Code', 'Provision_rank', 'Provision_category']].rename(
        columns={'Provision_rank': 'Provision_rank_prev', 'Provision_category': 'Provision_category_prev'}
    )
    curr = df_curr[['Main Code', 'Provision_rank', 'Provision_category']].rename(
        columns={'Provision_rank': 'Provision_rank_curr', 'Provision_category': 'Provision_category_curr'}
    )

    merged = pd.merge(prev, curr, on='Main Code', how='inner')
    change_keys = merged['Main Code']

    full = df_curr[df_curr['Main Code'].isin(change_keys)].copy()
    prev_details = df_prev.set_index('Main Code')[['Provision_rank', 'Provision_category']].rename(
        columns={'Provision_rank': 'Provision_rank_prev', 'Provision_category': 'Provision_category_prev'}
    )

    # ✅ Replaced index-based assignment with map to avoid reindex error
    full['Provision_rank_prev'] = full['Main Code'].map(prev_details['Provision_rank_prev'])
    full['Provision_category_prev'] = full['Main Code'].map(prev_details['Provision_category_prev'])

    full['Movement'] = full.apply(lambda row: (
        "Slippage" if row['Provision_rank'] > row['Provision_rank_prev']
        else "Upgrade" if row['Provision_rank'] < row['Provision_rank_prev']
        else "Stable"
    ), axis=1)

    columns_out = ['Branch Name', 'Main Code', 'Ac Type Desc', 'Name', 'Limit', 'Balance',
                   'Provision_category_prev', 'Provision_category', 'Movement']
    return full[columns_out].rename(columns={'Provision_category': 'Provision_category_curr'})

def category_matrix(df):
    matrix = pd.pivot_table(
        df,
        index='Provision_category_prev',
        columns='Provision_category_curr',
        aggfunc='size',
        fill_value=0
    )

    available_cols = [col for col in CATEGORY_ORDER if col in matrix.columns]
    matrix = matrix.reindex(columns=available_cols)
    return matrix.reset_index()

def summarize_matrix(df, group_col):
    matrices = []
    for g in df[group_col].unique():
        sub = df[df[group_col] == g]
        mat = pd.pivot_table(
            sub,
            index='Provision_category_prev',
            columns='Provision_category_curr',
            aggfunc='size',
            fill_value=0
        )
        available_cols = [col for col in CATEGORY_ORDER if col in mat.columns]
        mat = mat.reindex(columns=available_cols)
        mat[group_col] = g
        matrices.append(mat.reset_index())
    summary_df = pd.concat(matrices, ignore_index=True, sort=False)
    cols = [group_col] + [c for c in summary_df.columns if c != group_col]

    for col in summary_df.columns:
        if summary_df[col].dtype in ['float64', 'int64']:
            summary_df[col] = summary_df[col].fillna(0).astype(int)

    return summary_df[cols]

def generate_excel(slippage_df, branch_summary, ac_type_summary, matrix):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        slippage_df.to_excel(writer, index=False, sheet_name='Slippage Accounts')
        branch_summary.to_excel(writer, index=False, sheet_name='Summary by Branch')
        ac_type_summary.to_excel(writer, index=False, sheet_name='Summary by Ac Type')
        matrix.to_excel(writer, index=False, sheet_name='Category Matrix')
    output.seek(0)
    return output

# -----------------------------
# UI
# -----------------------------
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

            branch_summary = summarize_matrix(slippage_df, 'Branch Name')
            ac_type_summary = summarize_matrix(slippage_df, 'Ac Type Desc')
            matrix = category_matrix(slippage_df)

            excel_data = generate_excel(slippage_df, branch_summary, ac_type_summary, matrix)

            st.success("✅ Report Ready!")

            st.download_button(
                label="📤 Download Excel Report",
                data=excel_data,
                file_name=f"slippage_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error("❌ An error occurred during processing.")
            with st.expander("Show error details"):
                st.code(traceback.format_exc())

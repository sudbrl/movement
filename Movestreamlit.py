import streamlit as st
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

def autofit_excel(file_path):
    wb = load_workbook(file_path)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for column_cells in ws.columns:
            max_length = 0
            column = column_cells[0].column_letter  # Get the column name
            for cell in column_cells:
                try:
                    max_length = max(max_length, len(str(cell.value)))
                except:
                    pass
            ws.column_dimensions[column].width = max_length + 2
    wb.save(file_path)

def compare_excel_files(previous_file, current_file, output_file):
    cols_to_use = ['Main Code', 'Balance', 'Limit', 'Ac Type Desc', 'Name']
    df_previous = pd.read_excel(previous_file, usecols=lambda x: x in cols_to_use)
    df_this = pd.read_excel(current_file, usecols=lambda x: x in cols_to_use)

    # Apply 'Limit' filter if the column exists in any DataFrame
    if 'Limit' in df_previous.columns or 'Limit' in df_this.columns:
        df_previous = df_previous[df_previous.get('Limit', 1) != 0]  # Default to 1 if 'Limit' is missing
        df_this = df_this[df_this.get('Limit', 1) != 0]  # Default to 1 if 'Limit' is missing

    # Apply 'Ac Type Desc' filter if the column exists in any DataFrame
    if 'Ac Type Desc' in df_previous.columns or 'Ac Type Desc' in df_this.columns:
        filter_values = ["CURRENT ACCOUNT", "STAFF SOCIAL LOAN", "STAFF VEHICLE LOAN", 
                         "STAFF HOME LOAN", "STAFF FLEXIBLE LOAN", "STAFF HOME LOAN(COF)"]
        df_previous = df_previous[~df_previous.get('Ac Type Desc', "").isin(filter_values)]  # Default to "" if missing
        df_this = df_this[~df_this.get('Ac Type Desc', "").isin(filter_values)]  # Default to "" if missing

    # Apply 'Name' filter if the column exists in any DataFrame
    if 'Name' in df_previous.columns or 'Name' in df_this.columns:
        df_previous = df_previous[~df_previous.get('Name', "").str.contains("~~", na=False)]  # Default to "" if missing
        df_this = df_this[~df_this.get('Name', "").str.contains("~~", na=False)]  # Default to "" if missing

    previous_codes = set(df_previous['Main Code'])
    this_codes = set(df_this['Main Code'])

    only_in_previous = df_previous.loc[df_previous['Main Code'].isin(previous_codes - this_codes)]
    only_in_this = df_this.loc[df_this['Main Code'].isin(this_codes - previous_codes)]
    in_both = df_previous.loc[df_previous['Main Code'].isin(previous_codes & this_codes)]

    in_both = pd.merge(
        in_both[['Main Code', 'Balance']], 
        df_this[['Main Code', 'Balance']], 
        on='Main Code', 
        suffixes=('_previous', '_this')
    )
    in_both['Change'] = in_both['Balance_this'] - in_both['Balance_previous']

    opening_sum = df_previous['Balance'].sum()
    settled_sum = only_in_previous['Balance'].sum()
    new_sum = only_in_this['Balance'].sum()
    increase_decrease_sum = in_both['Change'].sum()
    adjusted_sum = opening_sum - settled_sum + new_sum + increase_decrease_sum
    closing_sum = df_this['Balance'].sum()

    opening_count = len(previous_codes)
    settled_count = len(previous_codes - this_codes)
    new_count = len(this_codes - previous_codes)
    closing_count = len(this_codes)

    reco_data = {
        'Description': ['Opening', 'Settled', 'New', 'Increase/Decrease', 'Adjusted', 'Closing'],
        'Amount': [opening_sum, settled_sum, new_sum, increase_decrease_sum, adjusted_sum, closing_sum],
        'No of Acs': [opening_count, settled_count, new_count, "", "", closing_count]
    }
    df_reco = pd.DataFrame(reco_data)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        only_in_previous.to_excel(writer, sheet_name='Settled', index=False)
        only_in_this.to_excel(writer, sheet_name='New', index=False)
        in_both.to_excel(writer, sheet_name='Movement', index=False)
        df_reco.to_excel(writer, sheet_name='Reco', index=False)

    autofit_excel(output_file)

    return output_file

def main():
    st.title("File Comparison Tool")
    st.write("Upload the previous period's Excel file and this period's Excel file to compare them. The Columns Required are Main Code, Balance, and Limit. Get download link.")

    previous_file = st.file_uploader("Upload Previous Period's Excel File", type=["xlsx"])
    current_file = st.file_uploader("Upload This Period's Excel File", type=["xlsx"])

    if previous_file and current_file:
        output_file = 'comparison_output.xlsx'

        with st.spinner("Processing..."):
            with open('previous_file.xlsx', 'wb') as f:
                f.write(previous_file.getbuffer())

            with open('current_file.xlsx', 'wb') as f:
                f.write(current_file.getbuffer())

            result_file = compare_excel_files('previous_file.xlsx', 'current_file.xlsx', output_file)

        if result_file:
            with open(result_file, "rb") as file:
                st.download_button(
                    label="Download Comparison Output",
                    data=file,
                    file_name=output_file,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()

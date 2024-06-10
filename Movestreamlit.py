import streamlit as st
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

def autofit_excel(file_path):
    wb = load_workbook(file_path)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for column_cells in ws.columns:
            max_length = 0
            column = column_cells[0].column_letter
            for cell in column_cells:
                try:
                    cell_value = str(cell.value)
                    if len(cell_value) > max_length:
                        max_length = len(cell_value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws.column_dimensions[column].width = adjusted_width
    wb.save(file_path)

def compare_excel_files(previous_file, current_file, output_file):
    # Reading the necessary columns from the Excel files using pandas
    cols_to_use = ['Main Code', 'Balance', 'Ac Type Desc', 'Name']
    df_previous = pd.read_excel(previous_file, usecols=cols_to_use)
    df_this = pd.read_excel(current_file, usecols=cols_to_use)

    # Filter out specified account types or names using .loc
    filter_values = ["CURRENT ACCOUNT", "STAFF SOCIAL LOAN", "STAFF VEHICLE LOAN", 
                     "STAFF HOME LOAN", "STAFF FLEXIBLE LOAN", "STAFF HOME LOAN(COF)"]
    df_previous = df_previous[~df_previous['Ac Type Desc'].isin(filter_values) & ~df_previous['Name'].str.contains("~~", na=False)]
    df_this = df_this[~df_this['Ac Type Desc'].isin(filter_values) & ~df_this['Name'].str.contains("~~", na=False)]

    merged = pd.merge(df_previous, df_this, on='Main Code', how='outer', suffixes=('_previous', '_this'))
    only_in_previous = merged[merged['Balance_this'].isna()][['Main Code', 'Balance_previous']]
    only_in_this = merged[merged['Balance_previous'].isna()][['Main Code', 'Balance_this']]
    in_both = merged[~merged['Balance_previous'].isna() & ~merged['Balance_this'].isna()][['Main Code', 'Balance_previous', 'Balance_this']]
    in_both['Change'] = in_both['Balance_this'] - in_both['Balance_previous']

    # Calculating the summary values for the Reco sheet
    opening_sum = np.sum(df_previous['Balance'])
    settled_sum = np.sum(only_in_previous['Balance_previous'])
    new_sum = np.sum(only_in_this['Balance_this'])
    increase_decrease_sum = np.sum(in_both['Change'])
    adjusted_sum = opening_sum - settled_sum + new_sum + increase_decrease_sum
    closing_sum = np.sum(df_this['Balance'])

    # Counting the number of accounts
    opening_count = len(df_previous)
    settled_count = len(only_in_previous)
    new_count = len(only_in_this)
    closing_count = len(df_this)

    # Creating the Reco DataFrame
    reco_data = {
        'Description': ['Opening', 'Settled', 'New', 'Increase/Decrease', 'Adjusted', 'Closing'],
        'Amount': [opening_sum, settled_sum, new_sum, increase_decrease_sum, adjusted_sum, closing_sum],
        'No of Acs': [opening_count, settled_count, new_count, "", "", closing_count]
    }
    df_reco = pd.DataFrame(reco_data)

    # Writing to an Excel file with four sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        only_in_previous.to_excel(writer, sheet_name='Settled', index=False)
        only_in_this.to_excel(writer, sheet_name='New', index=False)
        in_both.to_excel(writer, sheet_name='Movement', index=False)
        df_reco.to_excel(writer, sheet_name='Reco', index=False)

    # Autofit columns in the Excel file
    autofit_excel(output_file)

    st.success("Comparison output saved to " + output_file)
    return output_file

def main():
    st.title("Excel File Comparison Tool")

    st.write("Upload the previous month's Excel file and this month's Excel file to compare them.")

    previous_file = st.file_uploader("Upload Previous Month's Excel File", type=["xlsx"])
    current_file = st.file_uploader("Upload This Month's Excel File", type=["xlsx"])

    if previous_file and current_file:
        output_file = 'comparison_output.xlsx'
        result_file = compare_excel_files(previous_file, current_file, output_file)

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

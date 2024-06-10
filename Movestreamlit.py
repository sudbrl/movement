import streamlit as st
import dask.dataframe as dd
import pandas as pd
import tempfile
import os
from openpyxl import load_workbook

def autofit_excel(file_path):
    wb = load_workbook(file_path)
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for column_cells in ws.columns:
            max_length = 0
            column = column_cells[0].column_letter  # Get the column name
            for cell in column_cells:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws.column_dimensions[column].width = adjusted_width
    wb.save(file_path)

def compare_excel_files(previous_file, current_file, output_file):
    # Save the uploaded files to temporary locations
    with tempfile.NamedTemporaryFile(delete=False) as temp_prev_file:
        temp_prev_file.write(previous_file.read())
        temp_prev_path = temp_prev_file.name
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_current_file:
        temp_current_file.write(current_file.read())
        temp_current_path = temp_current_file.name

    # Reading the necessary columns from the Excel files using dask
    cols_to_use = ['Main Code', 'Balance', 'Ac Type Desc', 'Name']
    df_previous = dd.read_excel(open(temp_prev_path, 'rb'), usecols=cols_to_use)
    df_this = dd.read_excel(open(temp_current_path, 'rb'), usecols=cols_to_use)

    # Filter, process, and compare the dataframes as before

    # Finally, remove the temporary files
    os.unlink(temp_prev_path)
    os.unlink(temp_current_path)

    # Rest of the function remains unchanged

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

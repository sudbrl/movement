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
       


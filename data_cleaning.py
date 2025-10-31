from flask import Flask,jsonify
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
import os

CleanCSV_Name = "Clean_CSV"
directory_path = f"data/{CleanCSV_Name}"

#loading all the data into variables
december16 = pd.read_csv("data/Raw_CSV/2021-dec16.csv")
october21 = pd.read_csv("data/Raw_CSV/2021-oct21.csv")
november16 = pd.read_csv("data/Raw_CSV/2022-nov16.csv")
october7 = pd.read_csv("data/Raw_CSV/2022-oct7.csv")

#Listing all the columns of the dataframe.
# print(december16.columns)
# print(october21.columns)
# print(november16.columns)
# print(october7.columns)

# gathering relevant info from the dataframes
december16 = december16[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L"]]
october21 = october21[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L"]]
november16 = november16[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L"]]
october7 = october7[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L"]]

# Cleaning data using z-score method
# Should I manually calculate the z-score instead of importing from a library
# Can also use december16.std() from pandas for getting standard deviation of the rows 

# Calculating the z-score for Temp
december16['temp_z-score'] = zscore(december16["Temperature (c)"])
october21['temp_z-score'] = zscore(october21["Temperature (c)"])
november16['temp_z-score'] = zscore(november16["Temperature (c)"])
october7['temp_z-score'] = zscore(october7["Temperature (c)"])

# Calculating the z-score for Salinity
december16['salinity_z-score'] = zscore(december16["Salinity (ppt)"])
october21['salinity_z-score'] = zscore(october21["Salinity (ppt)"])
november16['salinity_z-score'] = zscore(november16["Salinity (ppt)"])
october7['salinity_z-score'] = zscore(october7["Salinity (ppt)"])

# Calculating the z-score for Odo
december16['odo_z-score'] = zscore(december16["ODO mg/L"])
october21['odo_z-score'] = zscore(october21["ODO mg/L"])
november16['odo_z-score'] = zscore(november16["ODO mg/L"])
october7['odo_z-score'] = zscore(october7["ODO mg/L"])

# Part 2:
print("\nTotal Rows in the frames")
# .index returns the rows!
print("Decmeber 16 total values:",len(december16.index))
print("October 21 total values:",len(october21.index))
print("Novemeber 16 total values:",len(november16.index))
print("October 7 total values:",len(october7.index))

# Removing outliers : where abs(z score) > 3
def cleaning_data_z_score(dataFrameName: pd.DataFrame, columnName :str) -> pd.DataFrame:
    # print(f"Cleaning Column : {columnName} ({len(dataFrameName.index)})",end="")

    result = dataFrameName[abs(dataFrameName[columnName]) < 3]

    # print(f" -> {len(result)}")
    return result

cleanedDecember16 = december16[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L","temp_z-score","salinity_z-score","odo_z-score"]]
cleanedOctober21 = october21[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L","temp_z-score","salinity_z-score","odo_z-score"]]
cleanedNovember16 = november16[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L","temp_z-score","salinity_z-score","odo_z-score"]]
cleanedOctober7 = october7[["Time","Temperature (c)","Salinity (ppt)","ODO mg/L","temp_z-score","salinity_z-score","odo_z-score"]]

def set_column_to_date(
    df: pd.DataFrame, 
    date_str: str, 
    new_column_name: str = 'Date'
) -> pd.DataFrame:
    
    try:
        fixed_date = pd.to_datetime(date_str)
        df_out = df.copy()
        df_out[new_column_name] = fixed_date
        
        return df_out
    except Exception as e:
        print(f"Error processing date '{date_str}' for column '{new_column_name}': {e}")
        return df
    
cleanedDecember16 = set_column_to_date(cleanedDecember16, "2024-12-16","Date")
cleanedOctober21 = set_column_to_date(cleanedOctober21, "2024-10-21","Date")
cleanedNovember16 = set_column_to_date(cleanedNovember16, "2024-11-16","Date")
cleanedOctober7 = set_column_to_date(cleanedOctober7, "2024-10-07","Date")

print(cleanedDecember16.columns)

# Filtering results for Temp and returning into a data frame
cleanedDecember16 = cleaning_data_z_score(cleanedDecember16, "temp_z-score")
cleanedOctober21 = cleaning_data_z_score(cleanedOctober21, "temp_z-score")
cleanedNovember16 = cleaning_data_z_score(cleanedNovember16, "temp_z-score")
cleanedOctober7 = cleaning_data_z_score(cleanedOctober7,"temp_z-score")

# Filtering results for Salinity and returning into a data frame
cleanedDecember16 = cleaning_data_z_score(cleanedDecember16, "salinity_z-score")
cleanedOctober21 = cleaning_data_z_score(cleanedOctober21, "salinity_z-score")
cleanedNovember16 = cleaning_data_z_score(cleanedNovember16, "salinity_z-score")
cleanedOctober7 = cleaning_data_z_score(cleanedOctober7,"salinity_z-score")

# Filtering results for Odo and returning into a data frame
cleanedDecember16 = cleaning_data_z_score(cleanedDecember16, "odo_z-score")
cleanedOctober21 = cleaning_data_z_score(cleanedOctober21, "odo_z-score")
cleanedNovember16 = cleaning_data_z_score(cleanedNovember16, "odo_z-score")
cleanedOctober7 = cleaning_data_z_score(cleanedOctober7,"odo_z-score")

# Summary
print(f"Decmeber16 : {len(december16.index)}",f"After Cleaning :{len(cleanedDecember16.index)}.",f"Difference {len(december16.index) - len(cleanedDecember16.index)}")
print(f"October21 : {len(october21.index)}",f"After Cleaning :{len(cleanedOctober21.index)}.",f"Difference {len(october21.index) - len(cleanedOctober21.index)}")
print(f"November16 : {len(november16.index)}",f"After Cleaning :{len(cleanedNovember16.index)}.",f"Difference {len(november16.index) - len(cleanedNovember16.index)}")
print(f"October7 : {len(october7.index)}",f"After Cleaning :{len(cleanedOctober7.index)}.",f"Difference {len(october7.index) - len(cleanedOctober7.index)}")


print("Making Clean CSV's Directory...")
try:
    os.makedirs(directory_path, exist_ok=True)
except OSError as e:
    print(f"Error: {e}")
print("Cleaned CSV Directory Made or exists!")

print(f"Creating cleaned CSV's in {CleanCSV_Name}...")
# cleanedDecember16.to_csv(f"{directory_path}/cleanedDecember16.csv")
cleanedOctober21.to_csv(f"{directory_path}/cleanedOctober21.csv")
cleanedNovember16.to_csv(f"{directory_path}/cleanedNovember16.csv")
cleanedOctober7.to_csv(f"{directory_path}/cleanedOctober7.csv")
print("Cleaned CSV's Made!")


from flask import Flask,jsonify
import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

#loading all the data into variables
december16 = pd.read_csv("Raw_CSV/2021-dec16.csv")
october21 = pd.read_csv("Raw_CSV/2021-oct21.csv")
november16 = pd.read_csv("Raw_CSV/2022-nov16.csv")
october7 = pd.read_csv("Raw_CSV/2022-oct7.csv")

#Listing all the columns of the dataframe.
# print(december16.columns)
# print(october21.columns)
# print(november16.columns)
# print(october7.columns)

# gathering relevant info from the dataframes
december16 = december16[["Time","Latitude","Longitude","Temperature (c)","Salinity (ppt)","ODO mg/L"]]
october21 = october21[["Time","Latitude","Longitude","Temperature (c)","Salinity (ppt)","ODO mg/L"]]
november16 = november16[["Time","Latitude","Longitude","Temperature (c)","Salinity (ppt)","ODO mg/L"]]
october7 = october7[["Time","Latitude","Longitude","Temperature (c)","Salinity (ppt)","ODO mg/L"]]

# Cleaning data using z-score method
# Should I manually calculate the z-score instead of importing from a library
# Can also use december16.std() from pandas for getting standard deviation of the rows 

# Calculating the z-score for Temp
december16['temp_z-score_dec16'] = zscore(december16["Temperature (c)"])
october21['temp_z-score_oct21'] = zscore(october21["Temperature (c)"])
november16['temp_z-score_nov16'] = zscore(november16["Temperature (c)"])
october7['temp_z-score_oct7'] = zscore(october7["Temperature (c)"])

# Calculating the z-score for Salinity
december16['salinity_z-score_dec16'] = zscore(december16["Salinity (ppt)"])
october21['salinity_z-score_oct21'] = zscore(october21["Salinity (ppt)"])
november16['salinity_z-score_nov16'] = zscore(november16["Salinity (ppt)"])
october7['salinity-score_oct7'] = zscore(october7["Salinity (ppt)"])

# Calculating the z-score for Odo
december16['odo_z-score_dec16'] = zscore(december16["ODO mg/L"])
october21['odo_z-score_oct21'] = zscore(october21["ODO mg/L"])
november16['odo_z-score_nov16'] = zscore(november16["ODO mg/L"])
october7['odo_z-score_oct7'] = zscore(october7["ODO mg/L"])

# print(december16.columns)
# print(october21.columns)
# print(november16.columns)
# print(october7.columns)

# print(december16.head()) # or print(december16[:5])
# print(december16.describe())


# Part 2:
print("\nTotal Rows in the frames")
# .index returns the rows!
print("Decmeber 16 total values:",len(december16.index))
print("October 21 total values:",len(october21.index))
print("Novemeber 16 total values:",len(november16.index))
print("October 7 total values:",len(october7.index))

#TODO
# Rows removed as outliers
# find the number of values over the abs of 3 for the z score
def cleaning_data_z_score(dataFrameName: pd.DataFrame, columnName :str) -> pd.DataFrame:
    print(f"Cleaning Column : {columnName}")
    result = dataFrameName[abs(dataFrameName[columnName] < 3)]
    # print(result)
    return result

print("\nRows remaining after cleaning")
# Filtering results and returning into a data frame
cleanedDecember16 = cleaning_data_z_score(december16, "temp_z-score_dec16")
print(len(cleanedDecember16.index))
cleanedOctober21 = cleaning_data_z_score(october21, "temp_z-score_oct21")
print(len(cleanedOctober21.index))
cleanedNovember16 = cleaning_data_z_score(november16, "temp_z-score_nov16")
print(len(cleanedNovember16.index))
cleanedOctober7 = cleaning_data_z_score(october7,"temp_z-score_oct7")
print(len(cleanedOctober7.index))
# Exporting into MongoDB Database. 
# Do i want to turn these into csv's then use another file to export into the DB?

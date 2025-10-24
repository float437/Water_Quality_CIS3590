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

# Calculating the z-score 
# TODO: does this need additional cleaning? 
# Can you manually calculate the z-score instead of importing from a library
december16['temp_z-score_dec16'] = zscore(december16["Temperature (c)"])
october21['temp_z-score_oct21'] = zscore(october21["Temperature (c)"])
november16['temp_z-score_nov16'] = zscore(november16["Temperature (c)"])
october7['temp_z-score_oct7'] = zscore(october7["Temperature (c)"])

print(december16.columns)
print(october21.columns)
print(november16.columns)
print(october7.columns)

# Filtering results and returning into a data frame

# Exporting into MongoDB Database. 
# Do i want to turn these into csv's then use another file to export into the DB?
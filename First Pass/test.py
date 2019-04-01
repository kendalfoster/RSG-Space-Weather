## Preamble
import numpy as np
import pandas as pd


## Import Data
data = pd.read_csv("data.csv")
data
data.columns
data[['Date_UTC', 'MLT', 'MLAT', 'N', 'E', 'Z']].iloc[0:10]
data.iloc[0:10,:]

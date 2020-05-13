import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import MinMaxScaler

M3Year = pd.read_excel('M3C.xls',sheet_name='M3Year')
M3Quart= pd.read_excel('M3C.xls',sheet_name='M3Quart')
M3Month = pd.read_excel('M3C.xls',sheet_name='M3Month')
M3Other = pd.read_excel('M3C.xls',sheet_name='M3Other')

M3Year_data=M3Year.iloc[:, 6:].to_numpy()
M3Quart_data=M3Quart.iloc[:, 6:].to_numpy()
M3Month_data=M3Month.iloc[:, 6:].to_numpy()
M3Other_data=M3Other.iloc[:, 6:].to_numpy()

values = M3Month_data.reshape(205632,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))

normalized = scaler.transform(values)
for i in range(70):
	print(normalized[i])

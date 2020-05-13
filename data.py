import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import MinMaxScaler

def FFT_smoothing(y,smootnes=0.01):
  """
  Return the data without trend and trend
  """
  mena=y.mean()
  y = y - mena
  freqs = np.fft.fftshift(np.fft.fftfreq(len(y)))
  fftVar = np.fft.fftshift(np.fft.fft(y))
  fft_filtered = fftVar.copy()
  abs=np.abs(freqs)
  add=np.max(abs[np.nonzero(abs)])+np.min(abs[np.nonzero(abs)])
  limit=np.min(abs[np.nonzero(abs)])+add*smootnes
  #Here I delete low frequencies
  fft_filtered[np.abs(freqs)>limit]=0
  trend = np.fft.ifft(np.fft.ifftshift(fft_filtered))
  without_trend=y-trend+mena
  #Returning just rela part I hope this is correct
  return(without_trend.real,trend.real)


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

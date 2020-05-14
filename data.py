#import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import MinMaxScaler

def FFT_smoothing(y,prediction,smootnes=0.01):
    """
    Return the data without trend and trend
    """
    mena=y.mean()
    n=len(y)
    y = y - mena
    freqs = np.fft.fftfreq(n)
    x_freqdom = np.fft.fft(y)
    abs=np.abs(freqs)
    add=np.max(abs[np.nonzero(abs)])+np.min(abs[np.nonzero(abs)])
    limit=np.min(abs[np.nonzero(abs)])+add*smootnes
    #Here I delete low frequencies
    x_freqdom[np.abs(freqs)>limit]=0
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: abs[i])
    t = np.arange(0, n + prediction)
    restored_sig = np.zeros(t.size)
    for i in indexes:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * freqs[i] * t + phase)
    without_trend=y-restored_sig[:n]+mena
    trend_prediction=restored_sig[n:]
    #Returning just rela part I hope this is correct
    #Returning data without trend and just trend
    return(without_trend,trend_prediction)

def multivariate_data(dataset, target, start_index, end_index, history_size,target_size, step, single_step=False):
    #copy of tnesorflow fuction for inputing data
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
          labels.append(target[i+target_size])
        else:
          labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)


M3Year = pd.read_excel('M3C.xls',sheet_name='M3Year')
M3Quart= pd.read_excel('M3C.xls',sheet_name='M3Quart')
M3Month = pd.read_excel('M3C.xls',sheet_name='M3Month')
M3Other = pd.read_excel('M3C.xls',sheet_name='M3Other')

M3Year_data=M3Year.iloc[:, 6:].to_numpy()
M3Quart_data=M3Quart.iloc[:, 6:].to_numpy()
M3Month_data=M3Month.iloc[:, 6:].to_numpy()
M3Other_data=M3Other.iloc[:, 6:].to_numpy()

#277 is the number of rows for MICRO data and tre is number of data that we can train on
tre = 50
M3Month_data_test=M3Month_data[:277,tre:]
M3Month_data_trend_prediction= np.zeros(shape=(277,18))
M3Month_data_withouttrend=np.zeros(shape=(277,tre))

for z in range(277):
	#num = M3Year.iloc[z]["N"]-M3Year.iloc[z]["NF"]
    num = tre
    without_trend,trend =FFT_smoothing(M3Month_data[z,:num],18)
    M3Month_data_withouttrend[z,:] = without_trend
    M3Month_data_trend_prediction[z,:] = trend
    #at the end when we will check the data we have add M3Year_data_trand_prediction to NN for real results

M3Month_data_withouttrend=np.transpose(M3Month_data_withouttrend)

values = M3Month_data_withouttrend
values= values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)
M3Month_data_withouttrend= normalized.reshape(tre,-1)

print(M3Month_data_withouttrend)
'''
If I didn't do any mistake is after here our data in the same form as dataset in
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=eJUeWDqploCt
Extracted that the trend and we normalized differently extracted trends.
Now we need to follow the tutorial here and use multivariate_data()  and implemented  it for our needs.
'''

'''
past_history = 720
future_target = 72
STEP = 6

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)

x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

'''

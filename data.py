import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import MinMaxScaler


# Change these to tweak the model
TRAIN_SPLIT = 33
past_history = 18
future_target = 18
STEP = 1

tf.random.set_seed(13)
BUFFER_SIZE = 15000
BATCH_SIZE = 10
EPOCHS = 15
STEPS_PER_EPOCH = 200

def polinomial_smoothing(y,prediction,degree,average=True):
    """
    Return the data without trend and predicted trend.
    Method use to separate trend is polynomial fitting.
    """
    if average == True:
        mean = y.mean()
    else:
        mean = 0
    y = y - mean
    time_stamp = np.arange(0,len(y))
    pre = np.arange(len(y),len(y) + prediction)
    p = np.polyfit(time_stamp, y, degree)
    trend_prediction = np.zeros(prediction)
    trend = np.zeros(len(y))
    pol = 0
    while(1):
        trend_prediction = trend_prediction + p[pol]*pre**degree
        trend = trend + p[pol]*time_stamp**degree
        degree -= 1
        pol += 1
        if degree == -1:
          return y-trend+mean,trend_prediction
          break;

def FFT_smoothing(y,prediction,pol_smoothing=False,smootnes=0.01):
    """
    Return the data without trend and trend.
    Method use to separate trend is fourier transform.
    It has aslo option to combine polinomial fitting.
    """
    mean = y.mean()
    n = len(y)
    y = y - mean

    #here we separate linear trend
    if pol_smoothing == True:
        y,pol_trend = polinomial_smoothing(y,prediction,1,average=True)
    else:
        pol_trend = 0

    freqs = np.fft.fftfreq(n)
    x_freqdom = np.fft.fft(y)
    abs = np.abs(freqs)
    add = np.max(abs[np.nonzero(abs)]) + np.min(abs[np.nonzero(abs)])
    limit = np.min(abs[np.nonzero(abs)]) + add*smootnes
    #Here I delete low frequencies
    x_freqdom[np.abs(freqs)>limit] = 0
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i: abs[i])
    t = np.arange(0, n + prediction)
    restored_sig = np.zeros(t.size)
    for i in indexes:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * freqs[i] * t + phase)
    without_trend = y - restored_sig[:n] + mean
    trend_prediction = restored_sig[n:] + pol_trend
    #Returning just rela part I hope this is correct
    #Returning data without trend and just trend
    return without_trend,trend_prediction

# Windowing task
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step):

    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    print(start_index, end_index)
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        print(target[i:i+target_size])
        labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

#Plotting the loss
def plot_train_history(history, title):
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

# create the time steps required for multi_step_plot
def create_time_steps(length):
  return list(range(-length, 0))

#creates the multi step plot
def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo', label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

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

'''
When we will test the network we can change
1.) smoothness rate of FFT_smoothing (0,1)
2) we have turn on also polinomial_smoothing in smoothness rate of FFT_smoothing and
3) we can change smoothing from FFT_smoothing to just polinomial_smoothing
(In polynomial smoothing you can set the order of polynomial that you can fit to data).

'''


for z in range(277):
	#num = M3Year.iloc[z]["N"]-M3Year.iloc[z]["NF"]
    num = tre
    without_trend,trend = FFT_smoothing(M3Month_data[z,:num],18)
    M3Month_data_withouttrend[z,:] = without_trend
    M3Month_data_trend_prediction[z,:] = trend
    #at the end when we will check the data we have add M3Year_data_trand_prediction to NN for real results

M3Month_data_withouttrend=np.transpose(M3Month_data_withouttrend)

values = M3Month_data_withouttrend
values= values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values)
normalized = scaler.transform(values)
M3Month_data_withouttrend = normalized.reshape(tre,-1)

# print(M3Month_data_withouttrend)
'''
If I didn't do any mistake is after here our data in the same form as dataset in
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb#scrollTo=eJUeWDqploCt
Extracted that the trend and we normalized differently extracted trends.
Now we need to follow the tutorial here and use multivariate_data()  and implemented  it for our needs.
'''

x_train_multi, y_train_multi = multivariate_data(M3Month_data_withouttrend, M3Month_data_withouttrend[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

#Create and compile LSTM model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32, return_sequences=True, input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(18))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

#numerical results (first ten)
for x, y in train_data_multi.take(10):
  print (multi_step_model.predict(x).shape)

#training
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

#Loss graph
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

#predictions overlayed with actual predictions (first three)
for x, y in train_data_multi.take(2):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])

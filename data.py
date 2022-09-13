import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.preprocessing import MinMaxScaler


TRAIN_SPLIT = 32
past_history = 18
future_target = 18
STEP = 1

#Setting the seed for reproducibility
tf.random.set_seed(13)

# Change these to tweak the model
BUFFER_SIZE = 15000
BATCH_SIZE = 14
EPOCHS = 10
STEPS_PER_EPOCH = 200

#Numbers of ROWS from which we are predicting. For now just the micro part with 68 lenght.
MICRO = 277

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

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

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

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo--', label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro:', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def print_results(name, true_future, predicted_future):
    f = open(name, "w")
    count = 0
    for item in true_future:
        f.write("{} {}\n".format(item, predicted_future[count]))
        count += 1
    f.close()

#Mape function
def MAPE(X,F):
    ave = np.array([])

    for x,f in zip(X,F):
        ave = np.append(ave,np.average(2*np.absolute(x-f)/(x+f))*100)
    return np.average(ave)


#Importing data
M3Month = pd.read_excel('M3C.xls',sheet_name='M3Month')
M3Month_data=M3Month.iloc[:, 6:].to_numpy()

Total_lenght = TRAIN_SPLIT + past_history + future_target

M3Month_data_trend_prediction = np.zeros(shape=(MICRO,future_target))
M3Month_data_withouttrend = np.zeros(shape=(MICRO,Total_lenght))

Detrending_lenght = TRAIN_SPLIT + past_history

#Detrending of the data
for z in range(MICRO):
    without_trend,trend = FFT_smoothing(M3Month_data[z,:Detrending_lenght],future_target,pol_smoothing=True)
    M3Month_data_withouttrend[z,:Detrending_lenght] = without_trend
    M3Month_data_withouttrend[z,Detrending_lenght:] =  M3Month_data[z,Detrending_lenght:Total_lenght] - trend
    M3Month_data_trend_prediction[z,:] = trend
    #at the end when we will check the data we have add M3Year_data_trand_prediction to NN for real results

#Normalization
M3Month_data_withouttrend = np.transpose(M3Month_data_withouttrend)
values = M3Month_data_withouttrend
values_tre = M3Month_data_withouttrend[:Detrending_lenght,:]
values_tre = values_tre.reshape(-1,1)
values = values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(values_tre)
normalized = scaler.transform(values)
M3Month_data_withouttrend = normalized.reshape(Total_lenght,-1)


#windowing
x_train_multi, y_train_multi = multivariate_data(M3Month_data_withouttrend, M3Month_data_withouttrend, 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(M3Month_data_withouttrend, M3Month_data_withouttrend,
                                             TRAIN_SPLIT, Detrending_lenght+1 , past_history,
                                             future_target, STEP)

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


#Create and compile LSTM model
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(200, return_sequences=True, input_shape=x_train_multi.shape[-2:], activation='tanh'))
multi_step_model.add(tf.keras.layers.LSTM(200, return_sequences=True, activation='tanh'))
multi_step_model.add(tf.keras.layers.Dense(MICRO))
multi_step_model.compile(optimizer='adam', loss='mse')


#training
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

#Loss graph
plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

#predictions overlayed with actual predictions
network_prediction_start = multi_step_model.predict(x_val_multi)[0]
network_prediction = network_prediction_start.reshape(-1,1)
network_prediction = scaler.inverse_transform(network_prediction)
network_prediction = network_prediction.reshape(network_prediction_start.shape)
real_prediction = np.transpose(network_prediction) + M3Month_data_trend_prediction

for Z in range(5):
    multi_step_plot(M3Month_data[Z,:Detrending_lenght], M3Month_data[Z,Detrending_lenght:Total_lenght], real_prediction[Z,:])


#printing the results to output.txt
print_results("output.txt", M3Month_data[:MICRO, Detrending_lenght:Total_lenght], real_prediction)

print ('Our final accuracy is : {:0.2f}'.format(MAPE(M3Month_data[:,Detrending_lenght:Total_lenght],real_prediction)))


What i think is happening right now is that the model is being trained on only half the data and predicting the rest, at the end you see a graph of the overlaid predictions with the actual data.

I still need to figure out how it is working so that we can optmise it and make it train on all the data and predict the last 18 data points.

For optimal results use 10 epochs with 1200 steps_per_epoch. 1200 steps might be overfitting the data though, and 200 is way too inaccurate. 

'''
Created on Jan 31, 2018

@author: Brian
'''
from lstm import fetch_timeseries_data
from lstm import normalise_windows
import numpy as np

if __name__ == '__main__':
    print ("\nUnit Test Data Preparation ...")
    ticker = "F"
    source = "Local"
    seq_len = 50
    
    ts_data = fetch_timeseries_data(["adj_open", "adj_volume"], ticker, source)
    print ("prepare_ts_lstm: panda dataframe data shape: %s, sequence length: %s" % (ts_data.shape, seq_len))
    
    ts_windows = normalise_windows(ts_data, seq_len, norm=[0, 1])
    
    np_windows_array = np.array(ts_windows)
    
    print("np_windows_array type", type(np_windows_array))
    print("ts_windows has shape", np_windows_array.shape)
    print("ts_windows data", np_windows_array)
    
    row = round(0.9 * ts_windows.shape[0])
    
    train = ts_windows[:int(row),:,:]
    print("train has shape", train.shape)
    
    np.random.shuffle(train)
    x_train = train[:,:-1]
    print("x_train data", x_train) 
    
    y_train = train[:, -1]
    print("x_train has shape", x_train.shape)
    print("y_train has shape", y_train.shape)
    print("y_train data ", y_train)
    
    # print ('x_train slice ', x_train)
    # print ('train slice ', train[:, 1])
    # print ('y_train slice ', y_train)
    
    x_test = ts_windows[int(row):,:-1]
    y_test = ts_windows[int(row):, -1]

    '''
    LSTM default is channels_last data_format i.e. ordering of the dimensions in the input
    (batch,time,...,channels)
    
    (samples,time steps, features)
    samples - 1 sequence is one sample. a batch is comprised of one or more samples
    time steps - 1 time step is 1 point of observation in the sample
    features - 1 feature is one observation at a time step
    input layer must be 3d
    input layer is defined by the input_shape argument on the first hidden layer
    input_shape argument takes a tuple of 2 values that define the number of time steps and features
    
    for our model 
        input_shape =(50,2)
        samples = number of data points / sample size
        time steps = batch size = 50
        number of features = 2
        data.reshape((samples,times steps,features)
    '''
    x_train_samples = x_train[0]
    x_train_time_steps = x_train[0]
    
    x_test_samples = x_test[0]
    
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))  
    
    print ("training shapes - x: ", x_train.shape, " y: ", y_train.shape)
    print ("testing  shapes - x: ", x_test.shape, " y: ", y_test.shape)

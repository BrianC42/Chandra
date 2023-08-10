'''
Created on Jan 31, 2018

@author: Pat
'''

#from string import Template
'''
Problem formulations:

1. Given the stock indicators for prior days forecast the price change for the current day

2. Predict the stock price change based on stock indicators from the last X days

3. Predict the stock price change based on the predicated stock indicators from the next X days

For given stock and time period, get daily adj_close values and calculate 'momentum','macd_ind','ema26','ema12',and percent price change.
Frame the data as a supervised learning problem and normalize the input. Build LSTM model. 
'''

from math import sqrt

from keras.layers.core import Dense
from keras.models import Sequential
from matplotlib import pyplot
from numpy import concatenate
from numpy import newaxis
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from keras.layers.recurrent import LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quandl_library import fetch_timeseries_data


def get_data(start,end,symbol):
    field_query = ["WIKI/"+symbol+".11"]
    field = "WIKI/"+ symbol +" - Adj. Close"
    df_data = fetch_timeseries_data(field_query,field,start,end)
    filename = "tsd_"+start+"_to_"+end+".csv"
    df_data.to_csv(filename)
    return filename   
 
def pricechange(df_data,forecastlen): 
    
    '''
       percent change = future price - current price / current price
    '''
    
    df_data.insert(loc=0, column='price_pct_chg', value=False)
    td =  df_data["adj_close"].values
    print ("target data before %change calculation ", td[:forecastlen+3])
    
    td_current =  td[:-forecastlen]
    td_future =  td[forecastlen:]
    targetdata = ((td_future - td_current)/ td_current) * 100
    
    df_data.loc[df_data.index[1:],'price_pct_chg'] = targetdata[:]
    
    
    print ("target data with %change calculation ", targetdata[:forecastlen+3], type(targetdata), len(targetdata))
    return df_data
    
      
def prep_data(filename):
    df_data = pd.read_csv(filename,index_col=0)
    print(df_data.columns)
    df_data.drop('MACD_Sell',axis=1,inplace=True)
    df_data.drop('MACD_Buy',axis=1,inplace=True)
    df_data.drop('30day_chg',axis=1,inplace=True)
    df_data.drop('MACD_future_chg',axis=1,inplace=True)
    df_data.columns = ['momentum','macd_ind','ema26','ema12','adj_close']
    
    df2 = df_data[1:]
    #print(df2.head(5))
    filename2 = "t2"+filename
    df2.to_csv(filename2)
    return filename2, df2

def plot_data(filename2): 
    df_data = pd.read_csv(filename2, header=0, index_col=0)
    df_values = df_data.values
    groups = [0,1,2,3,4]
    i = 1
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups),1,i)
        pyplot.plot(df_values[:,group])
        #print(df_data.columns[group])
        #pyplot.title(df_data.columns[group],v=0.5,loc='right')
        pyplot.title(df_data.columns[group])
        i +=1
    pyplot.show()

def reframe_data(filename,n_in=1,n_out=1,dropnan=True):
    #print("reframe data from: ", filename)
    df_data = pd.read_csv(filename,header=0,index_col=0)
    
    forecastlen = 1
    df_data = pricechange(df_data,forecastlen)
    df_data.to_csv("v1reframed_data.csv")
    
    values = df_data.values
    #ensure all data is float
    values = values.astype('float32')
    #normalize features
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(values)
    #frame as supervised learning
    reframed = series_to_supervised(scaled,df_data.columns,n_in,n_out)
    reframed.to_csv("v2reframed_data.csv")
    framedvalues = reframed.values
    #drop colums we don't want to predict
    #reframed.drop(reframed.columns[[6,7,8,9]], axis=1, inplace=True)
    reframed.drop(reframed.columns[[7,8,9,10,11]], axis=1, inplace=True)

    ##split intro train and test sets
    print(reframed.head())

    output_file = "v3reframed_data.csv"
    reframed.to_csv(output_file)
    return framedvalues,scaler,output_file
    
def series_to_supervised(data,dcolumns,n_in=1,n_out=1,dropnan=True):
    '''
    Default is one lag time step (t-1) to predict the current time step(t)
    '''
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    col_list = dcolumns
    print(col_list, dcolumns)
    cols, names = list(), list()
    
    #ctemplate = Template('$cname$cnum(t)')
    #forward_shift_ctemplate = Template('$cname$cnum(t+$cstep)')
    #back_shift_ctemplate = Template('$cname$cnum(t-$cstep)')
    
    #input sequence (t-n, .. t-1)
    # range(start,stop,step)
    # cols.append(df.shift(1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        #names += [back_shift_ctemplate.substitute(cname=col_list[j],cnum="{0:d}".format(j+1),cstep="{0:d}".format(i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            #names += [ctemplate.substitute(cname=col_list[j],cnum="{0:d}".format(j+1),cstep="{0:d}".format(i))]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
            #names += [forward_shift_ctemplate.substitute(cname=col_list[j],cnum="{0:d}".format(j+1),cstep="{0:d}".format(i))]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def design_lstm(train_X,train_Y,test_X,test_Y,epoch_size,batch_size):
    #design network
    model = Sequential()
    # lstm has 50 neurons, 1 neuron in the output layer for predicting ???, input shape will be 1 time step with 5 features
    layer_lstm = LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2]))
    model.add(layer_lstm)
    print("LSTM info", layer_lstm.input, layer_lstm.output, layer_lstm.input_shape, layer_lstm.output_shape)
    layer_dense = Dense(1)
    model.add(layer_dense)
    print("Dense info", layer_dense.input, layer_dense.output, layer_dense.input_shape, layer_dense.output_shape)
    model.compile(optimizer="adam",loss='mae')
        
    #fit network
    history = model.fit(train_X,train_Y,epochs=epoch_size,batch_size=batch_size,validation_data=(test_X,test_Y),verbose=2,shuffle=False)
    #history = model.fit(train_X,train_Y,epochs=50,batch_size=72,validation_data=(test_X,test_Y),shuffle=False)
    
    #plot history
    pyplot.plot(history.history['loss'],label='train')
    pyplot.plot(history.history['val_loss'],label='test')
    pyplot.legend()
    pyplot.show()

    return model
   
def tsm_data_load(start,end,symbol):
    filename = get_data(start,end,symbol)
    filename2, df_data = prep_data(filename)
    plot_data(filename2)
    return df_data, filename2

def split_data(framedvalues,filename):
    
    df_data = pd.read_csv(filename)
    framedvalues = df_data.values
    
    ndays = int((df_data.shape[0] * 2) / 3)
    # 90/10 split
    #ndays = round(0.9 * df_data.shape[0])
    #print("df_data.shape ", df_data.shape[0], " ndays ", ndays)
    
    train = framedvalues[:ndays,:]
    test = framedvalues[ndays:,:]
    
    #split into input and outputs
    train_X = train[:,:-1]
    train_Y = train[:,-1] 
    test_X = test[:,:-1]
    test_Y = test[:,-1]
    
    print(train_X.shape[0],train_X.shape[1], test_X.shape[0],test_X.shape[1])
    
    #reshape input to be 3D [samples,timesteps,features]
    train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
    print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)
    
    return train_X,train_Y,test_X,test_Y

def make_prediction(model,scaler,test_X,test_Y):
    
    yhat = model.predict(test_X)
    #print(test_X.shape[0],test_X.shape[1], test_X.shape[2], yhat.shape[0],yhat.shape[1])
    
    test_X = test_X.reshape((test_X.shape[0],test_X.shape[2]))
    
    #invert scaling for forecast    
    inv_yhat = concatenate((yhat, test_X[:,1:]),axis=1)
    #plc-added scaler.fit
    scaler.fit(inv_yhat)
    
    #print("scaler min_ ", scaler.min_, " scale ", scaler.scale_, " data_min_ ", scaler.data_min_, " data_max_ ", scaler.data_max_, " data_range_ ", scaler.data_range_)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    print("inv_yhat ", inv_yhat.shape)
    
    #invert scaling for actual
    test_Y = test_Y.reshape((len(test_Y),1))
    inv_Y = concatenate((test_Y,test_X[:,1:]),axis=1)
    inv_Y = inv_Y[:,0]
    
    #calculate RMSE
    rmse = sqrt(mean_squared_error(inv_Y,inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
def predict_point_by_point(model,data):
    #Predict each timestep given the last sequence of true data, en effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    #df_predicted1 = DataFrame(predicted)
    #df_predicted1.to_csv("predicted1.csv")
    predicted = np.reshape(predicted, (predicted.size,))
    df_predicted2 = DataFrame(predicted)
    df_predicted2.to_csv("predicted2.csv")
    return predicted

def plot_predictions(predictd_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label="True Data")
    plt.plot(predictd_data, label="Prediction")
    plt.legend()
    plt.show()
    
def predict_sequences_multiple(model,data,window_size,prediction_len):
    #Predict sequence of window_size  steps before shifting prediction run forward by prediction_len  steps
    
    prediction_seqs=[]
    n = int(len(data)/prediction_len)
    
    print("len(data) ", len(data), "range param ", n)
    
    for i in range(n):
        curr_frame = data[i*prediction_len]
        print("i", i, "i*prediction_len", i*prediction_len," curr_frame ", curr_frame)
        predicted = []
        for j in range(prediction_len):
            
            print("curr_frame[newaxis,:,:]).shape ",curr_frame[newaxis,:,:].shape)
            print("prediction ",model.predict(curr_frame[newaxis,:,:]))
            predicted_value = model.predict(curr_frame[newaxis,:,:])
            predicted.append((predicted_value)[0,0])
            
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame,[window_size-1],predicted[-1],axis=0)
            
        prediction_seqs.append(predicted)
    return prediction_seqs
'''
'''        
def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    
def drive_analysis():
    
    symbol = input("Pick a symbol ")
    print( "Symbol selected " + symbol)
    
    #step 1 - load timeseries dataset
    df_data, filename = tsm_data_load('2015-8-17','2017-10-17',symbol)
    
    #step 2 - convert time series to supervised learning problem
    '''
    n_in=5, n_out=5 with input sequence of 5 past observations forecast 1 future observations
    '''
    framedvalues,scaler,output_file = reframe_data(filename,5,1,True)
    
    return framedvalues, scaler, output_file

    

'''
    Price forecasting problem
    Given the momentum','macd_ind','ema26','ema12','adj_close' for prior day, forecast the % price change for the next day
'''
if __name__ == '__main__':
    
    unit_test = True
    unit_test_file = "t2tsd_2015-8-17_to_2017-10-17.csv"
    
    if unit_test:
        framedvalues,scaler, output_file = reframe_data(unit_test_file,5,1,True)
    else:
        framedvalues, scaler, output_file = drive_analysis()
    
    #split data
    train_X,train_Y,test_X,test_Y = split_data(framedvalues,output_file)
    
    #for testing use small sizes
    epoch_size = 50
    batch_size=72
    #design and plot network
    model = design_lstm(train_X,train_Y,test_X,test_Y,epoch_size,batch_size)
    
    #make predictions and plot them
    seq_len = 1
    pred_len = 50
    predictions = predict_sequences_multiple(model, test_X, seq_len, pred_len)
    plot_results_multiple(predictions, test_Y, pred_len)
    
    predictions = predict_point_by_point(model, test_X)
    plot_predictions(predictions, test_Y)
    
    #make a prediction
    make_prediction(model,scaler,test_X,test_Y)
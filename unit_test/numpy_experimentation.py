'''
Created on Aug 29, 2018

@author: Brian
'''
import logging

import time
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import newaxis
'''
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils import print_summary
'''
#from quandl_library import fetch_timeseries_data
#from quandl_library import get_ini_data
from time_series_data import series_to_supervised
from quandl_library import get_ini_data
from lstm import fetch_timeseries_data


if __name__ == '__main__':
    #pass

    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['log']
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')
    print ("Logging to", log_file)
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    
    logger.info('Keras LSTM model for stock market prediction')

    start = time.time()
    time_steps = 120 
    forecast_steps = 30

    '''
    Read data
    '''
    result_drivers = ["adj_low", "adj_high", "adj_open", "adj_close", "adj_volume"]
    forecast_feature = [False, True, False, False, False]
    feature_count = len(result_drivers)

    '''
    Load raw data
    '''
    df_data = fetch_timeseries_data(result_drivers, 'ibm', '')
    logging.info ('')
    logging.info ("Using %s features as predictors", feature_count)
    logging.info ("data shape: %s", df_data.shape)
    logging.debug('')
    for ndx_feature_value in range(0, feature_count) :
        logging.debug("%s: %s ... %s", result_drivers[ndx_feature_value], df_data[ :3, ndx_feature_value], df_data[ -3: , ndx_feature_value])
        ndx_feature_value += 1  
    
    start = time.time()
    '''
    Flatten data to a data frame where each row includes the 
        historical data driving the result - 
        time_steps examples of result_drivers data points
        result - 
        forecast_steps of data points (same data as drivers)
    '''
    #values = ts_data.values
    print ("\tPreparing time series samples of \n\t\t%s" % result_drivers)
    logging.info('Prepare as multi-variant time series data for LSTM')
    logging.info('series_to_supervised(values, time_steps=%s, forecast_steps=%s)', time_steps, forecast_steps)
    df_data = series_to_supervised(df_data, time_steps, forecast_steps)
    print("\t\ttime series samples shape: ", df_data.shape)

    step1 = time.time()
    print ("\t\tCreating time series took %s" % (step1 - start))
    print ("\tStructuring 3D data by 3 dimension loop")
    '''
    np_data[
        Dim1: time series samples
        Dim2: feature time series
        Dim3: features
        ]
    '''
    samples = df_data.shape[0]

    np_data = np.empty([samples, time_steps+forecast_steps, feature_count])
    print('\t\tnp_data shape: ', np_data.shape)

    '''
    Convert 2D LSTM data frame into 3D Numpy array for data pre-processing
    '''
    logging.debug('')
    for ndx_feature_value in range(0, feature_count) :
        for ndx_feature in range(0, time_steps+forecast_steps) :        
            for ndx_time_period in range(0, samples) :
                
                np_data[ndx_time_period, ndx_feature, ndx_feature_value] = \
                    df_data.iloc[ndx_time_period, (ndx_feature_value + (ndx_feature * feature_count))]
                
                ndx_time_period += 1            
            ndx_feature += 1
        #logging.debug('\nnp_data raw feature values: %s, %s', ndx_feature_value, result_drivers[ndx_feature_value])
        #logging.debug('\n%s', np_data[: , : , ndx_feature_value])
        ndx_feature_value += 1  
    
    step2 = time.time()
    print ("\t\tStructuring 3D data by 3 dimension loop took %s" % (step2 - step1))
    '''
    Experimentation code
    The iteration above requires most of the data preparation time
    create a second 3D numpy array and experiment with setting the contents by slicing
    '''
    print ("\tStructuring 3D data by 1 dimension loop and slicing")
    np_exp = np.empty([samples, time_steps+forecast_steps, feature_count])
    print('\t\tnp_exp shape: ', np_exp.shape)
    
    #slicing goes here
    for ndx_feature_value in range(0, feature_count) :
        #np_exp[ : , : , ndx_feature_value] = df_data[ : , ::5]
        #np_exp[ : , : , ndx_feature_value] = np_data[:, :, ndx_feature_value] this works and creates an equivalent array
        #np_exp[ : , : , ndx_feature_value] = df_data.iloc[:, ::feature_count]
        
        #                                         df.iloc[:, np.r_[1, 2                 : df.shape[1]      : 2]]
        np_exp[ : , : , ndx_feature_value] = df_data.iloc[:, np.r_[   ndx_feature_value : df_data.shape[1] : feature_count]]
        ndx_feature_value += 1  
    
    if (np.array_equiv(np_exp, np_data)) :
        print ("\t\t***     np_exp is EQUIVALENT to np_data     ****")
    else :
        print ("\t\tnp_exp differs from np_data")

    print("np_exp data shape: %s", np_exp.shape)
    print('\nnp_data(correct)\n%s\nnp_exp\n%s' % (np_data[:3,:3,:], np_exp[:3,:3,:]))
    print('\nnp_exp type = %s\nnp_data type = %s' % (np_data.dtype, np_exp.dtype))
    '''
    Experimentation code end
    '''
    step3 = time.time()
    print ("\t\tStructuring 3D data by 1 dimension loop and slicing took %s" % (step3 - step2))

'''
Created on Jan 31, 2018

@author: Brian
'''
import time
import os
import re
import warnings
import logging
import numpy as np
import pandas as pd
import pickle

from configuration_constants import BALANCE_CLASSES
from configuration_constants import BATCH_SIZE
from configuration_constants import EPOCHS
from configuration_constants import ANALYSIS
from configuration_constants import CLASSIFICATION_COUNT
from configuration_constants import VALIDATION_SPLIT
from configuration_constants import VERBOSE

from numpy import NAN

from keras.utils import print_summary
from keras.utils import plot_model

from quandl_library import fetch_timeseries_data
from quandl_library import get_ini_data
from quandl_library import get_devdata_dir

from time_series_data import series_to_supervised

from buy_sell_hold import build_bsh_classification_model
from buy_sell_hold import calculate_single_bsh_flag
from buy_sell_hold import calculate_sample_bsh_flag
from buy_sell_hold import balance_bsh_classifications

from percentage_change import calculate_single_pct_change
from percentage_change import calculate_sample_pct_change

'''
    Layers
        Methods
            get_weights
            set_weights
            get_config
            input or get_input_at
            output or get_output_at
            input_shape or get_input_shape_at
            output_shape or get_output_shape_at
        Core layer types
            Dense(
                Dense implements the operation: output = activation(dot(input, kernel) + bias) 
                where  
                    activation is the element-wise activation function passed as the activation argument, 
                    kernel is a weights matrix created by the layer, and 
                    bias is a bias vector created by the layer (only applicable if  use_bias is True)
            Activation
            Dropout
            Flatten
            Input
            Reshape
            Permute
            RepeatVector
            Lambda
            ActivityRegularization
            Masking
        Convolutional layer types
            Conv1D
            Conv2D
            SeparableConv2D
            Conv2DTranspose
            Conv3D
            Cropping1D
            Cropping2D
            Cropping3D
            UpSampling1D
            UpSampling2D
            UpSampling3D
            ZeroPadding1D
            ZeroPadding2D
            ZeroPadding3D
        Pooling layer types
            MaxPooling1D
            MaxPooling2D
            MaxPooling3D
            AveragePooling1D
            AveragePooling2D
            AveragePooling3D
        Locally-connected layer types
            LocallyConnected1D
            LocallyConnected2D
        Recurrent layer types - RNN is the base class for recurrent layers
            RNN(
            SimpleRNN
            GRU
            LSTM(
            ConvLSTM2D
            SimpleRNNCell
            GRUCell
            LSTMCell
            StackedRNNCells
            CuDNNGRU
            CuDNLSTM
        Embedding layer types
            Embedding
        Merge layer types
            Add
            Subtract
            Multiply
            Average
            Maximum
            Concatenate
            Dot
        Advanced activation layer types
            LeakyReLU
            PReLU
            ELU
            ThresholdedReLU
            Softmax
        Normalization layer types
            BatchNormalization
        Noise layer types
            GaussianNoise
            GaussianDropout
            AlphaDropout
'''

warnings.filterwarnings("ignore")

def pickle_dump_training_data (lst_analyses, x_train, y_train, x_test, y_test) :
    
    ml_config = get_ini_data('DEVDATA')
    training_dir = ml_config['dir']
    logging.info('Writing data files to %s',  training_dir)
    
    lst_analyses_file = training_dir + "\\lst_analyses.pickle"    
    lst_analyses_out = open(lst_analyses_file, "wb")
    pickle.dump(lst_analyses, lst_analyses_out)

    y_train_file = training_dir + "\\y_train.pickle"
    y_train_out = open(y_train_file, "wb")
    pickle.dump(y_train, y_train_out)

    y_test_file = training_dir + "\\y_test.pickle"
    y_test_out = open(y_test_file, "wb")
    pickle.dump(y_test, y_test_out)

    i_ndx = 0
    for analysis in lst_analyses:  
        x_train_file = training_dir + "\\x_train_" + analysis + ".npz"
        x_train_out = open(x_train_file, "wb")
        pickle.dump(x_train[i_ndx], x_train_out)

        x_test_file = training_dir + "\\x_test_" + analysis + ".npz"
        x_test_out = open(x_test_file, "wb")
        pickle.dump(x_test[i_ndx], x_test_out)
        
        i_ndx += 1

    return

def pickle_load_training_data ():
    
    ml_config = get_ini_data('DEVDATA')
    training_dir = ml_config['dir']
    logging.info('Reading data files from %s',  training_dir)
    
    lst_analyses_file = training_dir + "\\lst_analyses.pickle"    
    lst_analyses_in = open(lst_analyses_file, "rb")
    lst_analyses = pickle.load(lst_analyses_in)

    y_train_file = training_dir + "\\y_train.pickle"
    y_train_in = open(y_train_file, "rb")
    y_train = pickle.load(y_train_in)

    y_test_file = training_dir + "\\y_test.pickle"
    y_test_in = open(y_test_file, "rb")
    y_test = pickle.load(y_test_in)

    i_ndx = 0
    x_train = list()
    x_test = list()
    for analysis in lst_analyses:   
        x_train_file = training_dir + "\\x_train_" + analysis + ".npz"
        x_train_in = open(x_train_file, "rb")
        x_train.append(pickle.load(x_train_in))

        x_test_file = training_dir + "\\x_test_" + analysis + ".npz"
        x_test_in = open(x_test_file, "rb")
        x_test.append(pickle.load(x_test_in))
        
        i_ndx += 1

    return lst_analyses, x_train, y_train, x_test, y_test

def get_lstm_config():
    lstm_config_data = get_ini_data['LSTM']
    
    return lstm_config_data

def save_model(model):
    logging.info('save_model')
    lstm_config_data = get_ini_data("LSTM")
    filename = lstm_config_data['dir'] + "\\" + lstm_config_data['model']
    logging.debug ("Saving model to: %s", filename)
    
    model.save(filename)

    return

def load_model():
    logging.info('load_model')
    lstm_config_data = get_ini_data("LSTM")
    filename = lstm_config_data['dir'] + "\\" + lstm_config_data['model']
    logging.debug ("Loading model from: %s", filename)
    
    model = load_model(filename)
    
    return (model)

def save_model_plot(model):
    logging.info('save_model_plot')
    lstm_config_data = get_ini_data("LSTM")
    filename = lstm_config_data['dir'] + "\\" + lstm_config_data['plot']
    logging.debug ("Plotting model to: %s", filename)

    plot_model(model, to_file=filename)

    return

def prepare_ts_lstm(tickers, result_drivers, forecast_feature, feature_type, time_steps, forecast_steps, source='', analysis=''):
    logging.info('')
    logging.info('====> ==============================================')
    logging.info('====> prepare_ts_lstm: Prepare as multi-variant time series data for LSTM')
    logging.info('====> \ttime_steps: %s', time_steps)
    logging.info('====> \tforecast_steps: %s', forecast_steps)
    logging.info('====> \t%s feature(s): %s', len(result_drivers), result_drivers)
    logging.info('====> \t%s ticker(s): %s', len(tickers), tickers)
    logging.info('====> ==============================================')
    print ("\tAccessing %s time steps, with %s time step future values for\n\t\t%s" % (time_steps, forecast_steps, tickers))
    print ("\tPreparing time series samples of \n\t\t%s" % result_drivers)    
    '''
    Prepare time series data for presentation to an LSTM model from the Keras library
    seq_len - the number of elements from the time series for ?
    
    Returns training (90%) and test (10) sets
    '''
    start = time.time()

    feature_count = len(result_drivers)

    '''
    Load raw data
    '''
    if (len(tickers) <= 2) :
        if ((tickers[0] == "all") or (tickers[0] == "limit")):
            if (tickers[0] == "limit") :
                sample_limit = tickers[1]
                print("\tLimiting samples to %d time series" % sample_limit)
            else:
                sample_limit = 0
            print("\tLoading ALL symbols.")
            data_dir = get_devdata_dir() + "\\symbols\\"
            file_names = [fn for fn in os.listdir(data_dir) \
                          if re.search('enriched.csv', fn)]
            tickers = []
            for name in file_names :
                tickers.append(name.replace('_enriched.csv', ''))
    '''            
    if (tickers == ["all"]):
        print("\tLoading ALL symbols.")
        data_dir = get_devdata_dir() + "\\symbols\\"
        file_names = [fn for fn in os.listdir(data_dir) \
                      if re.search('enriched.csv', fn)]
        tickers = []
        for name in file_names :
            tickers.append(name.replace('_enriched.csv', ''))
    '''        

    for ndx_symbol in range(0, len(tickers)) :
        df_symbol = fetch_timeseries_data(result_drivers, tickers[ndx_symbol], source)
        logging.info ("df_symbol data shape: %s, %s samples of drivers\n%s", df_symbol.shape, df_symbol.shape[0], df_symbol.shape[1])
        for ndx_feature_value in range(0, feature_count) :
            logging.debug("result_drivers %s: head %s ... tail %s", \
                          result_drivers[ndx_feature_value], df_symbol[ :3, ndx_feature_value], df_symbol[ -3: , ndx_feature_value])
            '''
            Convert any boolean data to numeric
            false = 0
            true = 1
            '''
            if (feature_type[ndx_feature_value] == 'boolean') :
                logging.debug("Converting symbol %s, %s from boolean: %s data points", \
                              ndx_feature_value, result_drivers[ndx_feature_value], df_symbol.shape[0])
                for ndx_sample in range(0, df_symbol.shape[0]) :
                    if (df_symbol[ndx_sample, ndx_feature_value]) :
                        df_symbol[ndx_sample, ndx_feature_value] = 1
                    else :
                        df_symbol[ndx_sample, ndx_feature_value] = 0
                logging.debug("to %s ... %s", df_symbol[ :3, ndx_feature_value], df_symbol[ -3: , ndx_feature_value])
    
        '''
        Flatten data to a data frame where each row includes the 
        historical data driving the result - 
        time_steps examples of result_drivers data points
        result - 
        forecast_steps of data points (same data as drivers)
        '''
        if (ndx_symbol == 0) :
            df_data = series_to_supervised(df_symbol, time_steps, forecast_steps)
        else :
            df_n = series_to_supervised(df_symbol, time_steps, forecast_steps)
            df_data = pd.concat([df_data, df_n])
            
        if ((sample_limit > 0) and (len(df_data) > sample_limit)) :
            print("\tsample target reached: %d samples" % len(df_data))
            break
        else :
            print("\tLoaded data for %s, currently %s samples" % (tickers[ndx_symbol], len(df_data)))

        
    step1 = time.time()
    '''
    Convert 2D data frame into 3D Numpy array for data processing by LSTM
    np_data[ 
        Dim1: time series samples
        Dim2: feature time series
        Dim3: features
        ]
    '''
    samples = df_data.shape[0]
    np_data = np.empty([samples, time_steps+forecast_steps, feature_count])
    np_prediction = np.empty([samples, forecast_steps])
    logging.debug("convert 2D flat data frame of shape: %s to 3D numpy array of shape %s",  df_data.shape, np_data.shape)
    for ndx_feature_value in range(0, feature_count) :
        np_data[ : , : , ndx_feature_value] = df_data.iloc[:, np.r_[   ndx_feature_value : df_data.shape[1] : feature_count]]
    
    '''
    *** Specific to the analysis being performed ***
    '''
    step2 = time.time()
    print ("\tCalculating forecast y-axis characteristic values")
    logging.debug('')
    for ndx_feature_value in range(0, feature_count) :
        if forecast_feature[ndx_feature_value] :
            for ndx_feature in range(time_steps, time_steps+forecast_steps) :        
                for ndx_time_series_sample in range(0, samples) :
                    if (ANALYSIS == 'value') :
                        '''
                        Calculate % change of the for each time period forecast feature value compared to the current value
                        '''
                        np_prediction[ndx_time_series_sample, ndx_feature-time_steps] = calculate_single_pct_change()
                    elif (ANALYSIS == 'classification') :
                        '''
                        Calculate buy, sell or hold classification for each future time period
                        '''
                        np_prediction[ndx_time_series_sample, ndx_feature-time_steps] = \
                            calculate_single_bsh_flag(np_data[ndx_time_series_sample, time_steps-1, ndx_feature_value], \
                                                      np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value]  )
                    else :
                        print ('Analysis model is not specified')
            logging.debug('Using feature %s as feature to forecast, np_data feature forecast:', ndx_feature_value)
            logging.debug('Current value of forecast feature:\n%s', np_data[: , time_steps-1, ndx_feature_value])
            logging.debug('Future values\n%s', np_data[: , time_steps : , ndx_feature_value])
            logging.debug('Generated prediction values the model can forecast\n%s', np_prediction[:, :])
    
    '''
    Shuffle data
    '''
    np.random.shuffle(np_data)

    '''
    *** Specific to the analysis being performed ***
    '''
    print ("\tPreparing for %s analysis" % ANALYSIS)
    logging.debug('')
    np_forecast = np.zeros([samples, CLASSIFICATION_COUNT])
    for ndx_time_series_sample in range(0, samples) :
        if (ANALYSIS == 'value') :
            np_forecast[ndx_time_series_sample] = calculate_sample_pct_change()
        elif (ANALYSIS == 'classification') :
            '''
            Find the buy, sell or hold classification for each sample
            '''
            np_forecast[ndx_time_series_sample, :] = calculate_sample_bsh_flag(np_prediction[ndx_time_series_sample, :])                
        else :
            print ('Analysis model is not specified')
    logging.debug('\nforecast shape %s and values\n%s', np_forecast.shape, np_forecast)

    '''
    normalise the data in each row
        each data point for all historical data points is reduced to 0<=data<=+1
        1. Find the maximum value for each feature in each time series sample
        2. Normalize each feature value by dividing each value by the maximum value for that time series sample
        N.B. boolean fields have been normalized by conversion above
    '''
    step3 = time.time()
    print ("\tNormalizing data - min/max")
    logging.debug('')
    np_max = np.zeros([samples, feature_count])
    np_min = np.zeros([samples, feature_count])
    for ndx_feature_value in range(0, feature_count) : # normalize all numeric features
        if (feature_type[ndx_feature_value] == 'boolean') :
            logging.debug("Feature %s, %s is boolean and does not require normalizing", \
                          ndx_feature_value, result_drivers[ndx_feature_value])
        else :
            for ndx_feature in range(0, time_steps+forecast_steps) : # normalize only the time steps before the forecast time steps
                for ndx_time_series_sample in range(0, samples) : # normalize all time periods
                    if (np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] > np_max[ndx_time_series_sample, ndx_feature_value]) :
                        '''
                        logging.debug('New maximum %s, %s, %s was %s will be %s', \
                                  ndx_time_series_sample , ndx_feature, ndx_feature_value, \
                                  np_max[ndx_time_series_sample, ndx_feature_value], \
                                  np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value])
                        '''
                        np_max[ndx_time_series_sample, ndx_feature_value] = np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value]
                        if (np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] < np_min[ndx_time_series_sample, ndx_feature_value]) :
                            '''
                            logging.debug('New maximum %s, %s, %s was %s will be %s', \
                                ndx_time_series_sample , ndx_feature, ndx_feature_value, \
                                np_max[ndx_time_series_sample, ndx_feature_value], \
                                np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value])
                            '''
                            np_min[ndx_time_series_sample, ndx_feature_value] = np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value]

    print ("\tNormalizing data - scale")
    for ndx_feature_value in range(0, feature_count) :
        if (feature_type[ndx_feature_value] == 'boolean') :
            logging.debug("Feature %s, %s is boolean and already normalized", \
                          ndx_feature_value, result_drivers[ndx_feature_value])
        else :
            for ndx_feature in range(0, time_steps+forecast_steps) :        
                for ndx_time_series_sample in range(0, samples) :
                    if np_min[ndx_time_series_sample, ndx_feature_value] <= 0 :                    
                        np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] += abs(np_min[ndx_time_series_sample, ndx_feature_value])
                        np_max[ndx_time_series_sample, ndx_feature_value] += abs(np_min[ndx_time_series_sample, ndx_feature_value])
                        if (np_max[ndx_time_series_sample, ndx_feature_value] == 0) :
                            #
                            np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] = 0
                        else :
                            np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] = \
                                np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] / \
                                np_max[ndx_time_series_sample, ndx_feature_value]
                    if np_data[ndx_time_series_sample, ndx_feature, ndx_feature_value] == NAN :
                            logging.debug('NaN: %s %s %s', ndx_time_series_sample, ndx_feature, ndx_feature_value) 
            logging.debug('normalized np_data feature values (0.0 to 1.0): %s, %s type: %s', \
                          ndx_feature_value, result_drivers[ndx_feature_value], type(np_data[0, 0, ndx_feature_value]))
            logging.debug('\n%s', np_data[: , : time_steps , ndx_feature_value] )

    '''
    Balance data for training
    '''
    if BALANCE_CLASSES :
        print ("\tBalancing samples")
        if (ANALYSIS == 'value') :
            i_dummy = 1 #TBD
        elif (ANALYSIS == 'classification') :
            '''
            Ensure there are the same number of actual buy, sells and hold classifications
            '''
            np_data, np_forecast = balance_bsh_classifications(np_data, np_forecast)                
        else :
            print ('Analysis model is not specified')
            
    step4 = time.time()
    '''
    Split data into test and training portions
    '''
    print ("\tPreparing training and testing samples")
    row = round(0.1 * np_data.shape[0])
    x_test  = np_data    [        :int(row), :time_steps , : ]
    x_train = np_data    [int(row):        , :time_steps , : ]
    y_test  = np_forecast[        :int(row), :]
    y_train = np_forecast[int(row):        , :]

    list_x_test  = list([])
    list_x_train = list([])
    lst_technical_analysis = []
    
    #Use list_x_test[0] on a model to learn to forecast based on "adj_low", "adj_high", "adj_open", "adj_close", "adj_volume"
    lst_technical_analysis.append('Market_Activity')
    list_x_test.append (x_test [:, :, :5])
    list_x_train.append(x_train[:, :, :5])
    
    #Use list_x_test[1] on a model to learn to forecast based on "BB_Lower", "BB_Upper"
    lst_technical_analysis.append('Bollinger_bands')
    list_x_test.append (x_test [:, :, 5:7])
    list_x_train.append(x_train[:, :, 5:7])
    
    #Use list_x_test[2] on a model to learn to forecast based on "AccumulationDistribution"
    lst_technical_analysis.append('AccumulationDistribution')
    list_x_test.append (x_test [:, :, 9:10])
    list_x_train.append(x_train[:, :, 9:10])
    
    #Use list_x_test[3] on a model to learn to forecast based on "MACD_Sell"
    lst_technical_analysis.append('MACD_Sell')
    list_x_test.append (x_test [:, :, 10:12])
    list_x_train.append(x_train[:, :, 10:12])
    
    #Use list_x_test[3] on a model to learn to forecast based on "MACD_Buy"
    lst_technical_analysis.append('MACD_Buy')
    list_x_test.append (x_test [:, :, 12:13])
    list_x_train.append(x_train[:, :, 12:13])
    
    #Use list_x_test[3] on a model to learn to forecast based on "AccumulationDistribution"
   
    end = time.time()

    logging.info ('<---------------------------------------------------')
    logging.info ('<---- Including the following technical analyses:\n\t%s' % lst_technical_analysis)
    logging.info ('<---- train and test shapes: \nx_test %s\nx_train %s\ny_test %s\ny_train %s', \
                 x_test.shape, x_train.shape, y_test.shape, y_train.shape)
    for ndx_i in range (0, len(list_x_train)) :
        logging.debug('<---- lst_technical_analysis %s', lst_technical_analysis[ndx_i])
        logging.debug('<---- \ttraining data shape %s', list_x_train[ndx_i].shape)
        logging.debug('<---- \ttesting data shape %s', list_x_test[ndx_i].shape)
        logging.debug('<---- %s training data:\n%s', lst_technical_analysis[ndx_i], list_x_train[ndx_i])
    logging.info ("<---- \tCreating time series took %s" % (step1 - start))
    logging.info ("<---- \tStructuring 3D data took %s" % (step2 - step1))
    logging.info ("<---- \tCalculating forecast y-axis characteristic value took %s" % (step3 - step2))
    logging.info ("<---- \tNormalization took %s" % (step4 - step3))
    logging.info ("<---- \tCreating test and training data took %s" % (end - step4))
    logging.info ('<---------------------------------------------------')    
    return [lst_technical_analysis, list_x_train, y_train, list_x_test, y_test]

def build_model(lst_analyses, np_input, f_out):
    logging.info('====> ==============================================')
    logging.info('====> build_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    if (ANALYSIS == 'value') :
        dummy = 0
    elif (ANALYSIS == 'classification') :
        #Classification prediction of buy, sell or hold
        model = build_bsh_classification_model(lst_analyses, np_input)
    else :
        print ('Analysis model is not specified')

    # 2 visualizations of the model
    save_model_plot(model)
    print_summary(model)

    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model')
    logging.info('<---- ----------------------------------------------')
    return model

def train_lstm(model, x_train, y_train, f_out):
    logging.info('')
    logging.info('====> ==============================================')
    logging.info("====> train_lstm: Fitting model using training data inputs: x=%s and y=%s", len(x_train), y_train.shape)
    logging.info('====> ==============================================')
    print('train_lstm: Fitting model using training data: len(x)=%s and y=%s' % (len(x_train), y_train.shape))
    
    lst_x = []
    lst_y = []
    for ndx_i in range(0, len(x_train)) :
        lst_x.append(x_train[ndx_i])
        lst_y.append(y_train)
    lst_y.append(y_train)
        
    '''
    x:                 Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs). 
                        If input layers in the model are named, you can also pass a dictionary mapping input names to Numpy arrays.  
                        x can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
    y:                 Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs). 
                        If output layers in the model are named, you can also pass a dictionary mapping output names to Numpy arrays.  
                        y can be None (default) if feeding from framework-native tensors (e.g. TensorFlow data tensors).
    batch_size:        Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.
    epochs:            Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. 
                        Note that in conjunction with initial_epoch,  epochs is to be understood as "final epoch". 
                        The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
    verbose:           Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    callbacks:         List of keras.callbacks.Callback instances. List of callbacks to apply during training and validation (if ). See callbacks.
    validation_split:  Float between 0 and 1. Fraction of the training data to be used as validation data. 
                        The model will set apart this fraction of the training data, will not train on it, 
                        and will evaluate the loss and any model metrics on this data at the end of each epoch. 
                        The validation data is selected from the last samples in the x and y data provided, before shuffling.
    validation_data:   tuple (x_val, y_val) or tuple  (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model metrics 
                        at the end of each epoch. The model will not be trained on this data.  validation_data will override validation_split.
    shuffle:           Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special option for 
                        dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect when steps_per_epoch is not None.
    class_weight:      Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). 
                        This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
    sample_weight:     Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). 
                        You can either pass a flat (1D) Numpy array with the same length as the input samples (1:1 mapping between weights and samples), 
                        or in the case of temporal data, you can pass a 2D array with shape  (samples, sequence_length), 
                        to apply a different weight to every timestep of every sample. In this case you should make sure to specify 
                        sample_weight_mode="temporal" in compile().
    initial_epoch:     Integer. Epoch at which to start training (useful for resuming a previous training run).
    steps_per_epoch:   Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. 
                        When training with input tensors such as TensorFlow data tensors, the default None is equal to the number of samples 
                        in your dataset divided by the batch size, or 1 if that cannot be determined.
    validation_steps:  Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) to validate before stopping.
    validation_freq:   Only relevant if validation data is provided. Integer or list/tuple/set. If an integer, specifies how many training epochs 
                        to run before a new validation run is performed, e.g.  validation_freq=2 runs validation every 2 epochs. 
                        If a list, tuple, or set, specifies the epochs on which to run validation, e.g. validation_freq=[1, 2, 10] runs 
                        validation at the end of the 1st, 2nd, and 10th epochs.
    '''
    model.fit(x=lst_x, y=lst_y, shuffle=True, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT, verbose=VERBOSE)
    f_out.write('\nTraining based on {:d} samples'.format(x_train[0].shape[0]))
        
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- train_lstm:')
    logging.info('<---- ----------------------------------------------')
    return

def evaluate_model(model, x_data, y_data, f_out):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> evaluate_model')
    logging.debug('====> x_data includes %s inputs', len(x_data))
    logging.info ('====> ==============================================')
    
    lst_x = []
    lst_y = []
    for ndx_i in range(0, len(x_data)) :
        logging.info ('Input %s - shape %s', ndx_i, x_data[ndx_i].shape)
        lst_x.append(x_data[ndx_i])
        lst_y.append(y_data)
    lst_y.append(y_data)

    score = model.evaluate(x=lst_x, y=lst_y, verbose=0)    
    print ("\tevaluate_model: Test loss=%0.2f, Test accuracy=%0.2f" % (score[0], score[1]))
    f_out.write('\nEvaluation based on {:d} samples'.format(x_data[0].shape[0]))
    f_out.write('\nEvaluation loss {:0.2f} - accuracy {:0.2f}'.format(score[0], score[1]))

    logging.info('<---- ----------------------------------------------')
    logging.info('<---- evaluate_model: Test loss=%0.2f, Test accuracy=%0.2f', score[0], score[1])
    logging.info('<---- ----------------------------------------------')
    return

def predict_sequences_single(model, df_data):   
    '''
    *** Specific to the analysis being performed ***
    find the models predictions for each output
    '''
    prediction = model.predict(x=df_data, batch_size=1)

    return prediction

def predict_sequences_multiple(model, df_data, lst_analyses, f_out):
    '''
    *** Specific to the analysis being performed ***
    Find the maximum adj_high for each time period sample
        One input for each technical analysis set of features
        One output for each technical analysis plus one composite output
        
    df_data: list of 3D numpy arrays, one for each technical analysis set of features
        dimension 1: samples - length 1
        dimension 2: time series
        dimension 3: features
    '''
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> predict_sequences_multiple: %s technical analysis feature sets', len(df_data))
    print        ('\tpredicting %s technical analysis feature sets plus a composite prediction' % (len(df_data)))
    f_out.write  ('\npredicting {:d} technical analysis feature sets plus a composite prediction'.format(len(df_data)))
    for ndx_i in range(0, len(df_data)) :
        logging.info ('%s technical analysis includes %s features', lst_analyses[ndx_i], df_data[ndx_i].shape[2])
        print        ('\t\t%s technical analysis includes %s features' % (lst_analyses[ndx_i], df_data[ndx_i].shape[2]))
        f_out.write  ('\n{:s} technical analysis includes {:d} features'.format(lst_analyses[ndx_i], df_data[ndx_i].shape[2]))
    print        ('\t%s predictions, each based on %s historical time steps' % (df_data[0].shape[0], df_data[0].shape[1]))
    logging.info ('====> ==============================================')
        
    samples = df_data[0].shape[0]
    np_predictions = np.empty([samples, (len(df_data) + 1), CLASSIFICATION_COUNT])
    logging.debug('Output shape: %s', np_predictions.shape)

    for ndx_samples in range(0, samples) :
        lst_x = []
        for ndx_i in range(0, len(df_data)) :
            lst_x.append(df_data[ndx_i][ndx_samples:ndx_samples+1])
        np_predictions[ndx_samples] = predict_sequences_single(model, lst_x)

    #print ("%s predictions of length %s" % (len(prediction_seqs), len(prediction_seqs[0])))
    logging.info ('<---- ----------------------------------------------')
    logging.info ('<---- predict_sequences_multiple:')
    logging.debug('<---- \tprediction shape: %s, length: %s predictions, values=\n%s', np_predictions.shape, len(np_predictions), np_predictions)
    logging.info ('<---- ----------------------------------------------')    
    return np_predictions

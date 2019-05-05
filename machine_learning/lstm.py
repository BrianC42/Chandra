'''
Created on Jan 31, 2018

@author: Brian
'''
import time
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from numpy import newaxis, NaN, NAN
from quandl_library import fetch_timeseries_data
from quandl_library import get_ini_data
from time_series_data import series_to_supervised
from tensorflow.python.layers.core import dense
from keras.backend.tensorflow_backend import dtype
from keras.backend.tensorflow_backend import shape
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.models import Sequential
from keras.models import Model
from keras.utils import plot_model
from keras.utils import print_summary

from buy_sell_hold import calculate_single_bsh_flag
from buy_sell_hold import calculate_sample_bsh_flag

'''
    Sequential model
        Attributes
            model.layers
    Model class API
        Attributes
            model.layers
            model.inputs
            model.outputs
            
        Methods (for both sequential and class API)
            compile
                loss
                    mean_squared_error
                    mean_absolute_error
                    mean_absolute_percentage_error
                    mean_squared_logarithmic_error
                    squared_hinge
                    hinge
                    categorical_hinge
                    logcosh
                    categorical_crossentropy
                    sparse_categorical_crossentropy
                    binary_crossentropy
                    kullback_leibler_divergence
                    poisson
                    cosine_proximity
    
                optimizer
                    SGD
                    RMSprop
                    Adagrad
                    Adadelta
                    Adam
                    Adamax
                    Nadam
                    TFOptimizer
        
                metrics
                    binary_accuracy
                    categorical_accuracy
                    sparse_categorical_accuracy
                    top_k_categorical_accuracy
                    spares_top_k_categorical_accuracy

            fit(self, 
            x=None,
                Numpy array of training data (if the model has a single input), or list of Numpy arrays 
                (if the model has multiple inputs). If input layers in the model are named, you can also pass 
                a dictionary mapping input names to Numpy arrays. x can be None (default) if feeding from 
                framework-native tensors (e.g. TensorFlow data tensors).
            y=None, 
                Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays 
                (if the model has multiple outputs). If output layers in the model are named, you can also 
                pass a dictionary mapping output names to Numpy arrays. y can be None (default) if feeding 
                from framework-native tensors (e.g. TensorFlow data tensors).
            batch_size=None, 
                Integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32.
            epochs=1, 
                Integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data 
                provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". 
                The model is not trained for a number of iterations given by epochs, but merely until the epoch of 
                index epochs is reached.
            verbose=1, 
                Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks=None, 
                List of keras.callbacks.Callback instances. List of callbacks to apply during training. See callbacks
            validation_split=0.0, 
                Float between 0 and 1. Fraction of the training data to be used as validation data. 
                The model will set apart this fraction of the training data, will not train on it, and will 
                evaluate the loss and any model metrics on this data at the end of each epoch. 
                The validation data is selected from the last samples in the x and y data provided, before shuffling.
            validation_data=None, 
                tuple (x_val, y_val) or tuple (x_val, y_val, val_sample_weights) on which to evaluate the loss and any model 
                metrics at the end of each epoch. The model will not be trained on this data. 
                validation_data will override validation_split.
            shuffle=True, 
                Boolean (whether to shuffle the training data before each epoch) or str (for 'batch'). 'batch' is a special 
                option for dealing with the limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect 
                when steps_per_epoch is not None.
            class_weight=None, 
                Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss 
                function (during training only). This can be useful to tell the model to "pay more attention" to samples 
                from an under-represented class.
            sample_weight=None, 
                Optional Numpy array of weights for the training samples, used for weighting the loss function (during training only). 
                You can either pass a flat (1D) Numpy array with the same length as the input samples 
                (1:1 mapping between weights and samples), or in the case of temporal data, you can pass a 2D array 
                with shape (samples, sequence_length), to apply a different weight to every timestep of every sample. 
                In this case you should make sure to specify sample_weight_mode="temporal" in compile().
            initial_epoch=0, 
                Integer. Epoch at which to start training (useful for resuming a previous training run).
            steps_per_epoch=None, 
                Integer or None. Total number of steps (batches of samples) before declaring one epoch finished and 
                starting the next epoch. When training with input tensors such as TensorFlow data tensors, 
                the default None is equal to the number of samples in your dataset divided by the batch size, 
                or 1 if that cannot be determined.
            validation_steps=None
                Only relevant if steps_per_epoch is specified. Total number of steps (batches of samples) 
                to validate before stopping.
            )            
            
            evaluate
            predict
            train_on_batch
            test_on_batch
            predict_on_batch
            fit_generator
            evaluate_generator
            predict_generator
            get_layer
            
    You can create a Sequential model by passing a list of layer instances to the constructor:
    model = Sequential([
        Dense(32, input_shape=(784,)),
        Activation('relu'),
        Dense(10),
        Activation('softmax'),
        ])
    
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
                units, 
                activation=None, 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                bias_initializer='zeros', 
                kernel_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                bias_constraint=None)
                )
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
                cell, 
                return_sequences=False, 
                return_state=False, 
                go_backwards=False, 
                stateful=False, 
                unroll=False
                )
                cell: A RNN cell instance. A RNN cell is a class that has:
                    a call(input_at_t, states_at_t) method, returning (output_at_t, states_at_t_plus_1). 
                        The call method of the cell can also take the optional argument constants, see section 
                        "Note on passing external constants" below.
                    a state_size attribute. This can be a single integer (single state) in which case it is 
                        the size of the recurrent state (which should be the same as the size of the cell output). 
                        This can also be a list/tuple of integers (one size per state). In this case, the first 
                        entry (state_size[0]) should be the same as the size of the cell output. It is also 
                        possible for  cell to be a list of RNN cell instances, in which cases the cells get 
                        stacked on after the other in the RNN, implementing an efficient stacked RNN.
                return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence.
                return_state: Boolean. Whether to return the last state in addition to the output.
                go_backwards: Boolean (default False). If True, process the input sequence backwards and 
                    return the reversed sequence.
                stateful: Boolean (default False). If True, the last state for each sample at index i in a 
                    batch will be used as initial state for the sample of index i in the following batch.
                unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic 
                    loop will be used. Unrolling can speed-up a RNN, although it tends to be more 
                    memory-intensive. Unrolling is only suitable for short sequences.
                input_dim: dimensionality of the input (integer). This argument (or alternatively, 
                    the keyword argument  input_shape) is required when using this layer as the first layer in a model.
                input_length: Length of input sequences, to be specified when it is constant. This argument 
                    is required if you are going to connect Flatten then Dense layers upstream (without it, 
                    the shape of the dense outputs cannot be computed). Note that if the recurrent layer is 
                    not the first layer in your model, you would need to specify the input length at the level 
                    of the first layer (e.g. via the input_shape argument)
            SimpleRNN
            GRU
            LSTM(
                units, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                recurrent_initializer='orthogonal', 
                bias_initializer='zeros', 
                unit_forget_bias=True, 
                kernel_regularizer=None, 
                recurrent_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                recurrent_constraint=None, 
                bias_constraint=None, 
                dropout=0.0, 
                recurrent_dropout=0.0, 
                implementation=1, 
                return_sequences=False, 
                return_state=False, 
                go_backwards=False, 
                stateful=False, 
                unroll=False)
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
            add
            subtract
            multiply
            average
            maximum
            concatenate
            dot
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
        
    Acivations
        softmax
        elu
        selu
        softplus - 
        softsign - 
        relu     - Rectified Linear Unit - output = input if >0 otherwise 0
        tanh     - Limit output to the range -1 <= output <= +1
        sigmoid  - limit output to the range 0 <= output <= +1
        hard_sigmoid
        linear
        
'''

warnings.filterwarnings("ignore")

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

def prepare_ts_lstm(tickers, result_drivers, forecast_feature, time_steps, forecast_steps, source='', analysis=''):
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
    for ndx_symbol in range(0, len(tickers)) :
        df_symbol = fetch_timeseries_data(result_drivers, tickers[ndx_symbol], source)
        logging.info ("df_symbol data shape: %s, %s samples of drivers\n%s", df_symbol.shape, df_symbol.shape[0], df_symbol.shape[1])
        for ndx_feature_value in range(0, feature_count) :
            logging.debug("result_drivers %s: df_symbol %s ... df_symbol %s", \
                          result_drivers[ndx_feature_value], df_symbol[ :3, ndx_feature_value], df_symbol[ -3: , ndx_feature_value])
            ndx_feature_value += 1  
    
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
        ndx_feature_value += 1  
    
    '''
    *** Specific to the analysis being performed ***
    Calculate forecast feature
    '''
    step2 = time.time()
    #print ("\tCalculating forecast y-axis characteristic value")
    logging.debug('')
    for ndx_feature_value in range(0, feature_count) :
        if forecast_feature[ndx_feature_value] :
            for ndx_feature in range(time_steps, time_steps+forecast_steps) :        
                for ndx_time_period in range(0, samples) :
                    if (analysis == 'TBD') :
                        print ('Analysis model is not yet defined')
                    elif (analysis == 'buy_sell_hold') :
                        np_prediction[ndx_time_period, ndx_feature-time_steps] = \
                            calculate_single_bsh_flag(np_data[ndx_time_period, time_steps-1, ndx_feature_value], \
                                                      np_data[ndx_time_period, ndx_feature, ndx_feature_value]  )
                    else :
                        print ('Analysis model is not specified')
                    ndx_time_period += 1            
                ndx_feature += 1
            logging.debug('Using feature %s as feature to forecast, np_data feature forecast:', ndx_feature_value)
            logging.debug('Current value of forecast feature:\n%s', np_data[: , time_steps-1, ndx_feature_value])
            logging.debug('Future values\n%s', np_data[: , time_steps : , ndx_feature_value])
            logging.debug('Generated prediction values the model can forecast\n%s', np_prediction[:, :])
        ndx_feature_value += 1  
    
    '''
    *** Specific to the analysis being performed ***
    Find the maximum adj_high for each time period sample
    '''
    logging.debug('')
    np_forecast = np.zeros([samples])
    for ndx_time_period in range(0, samples) :
        if (analysis == 'TBD') :
            print ('Analysis model is not yet defined')
        elif (analysis == 'buy_sell_hold') :
            np_forecast[ndx_time_period] = calculate_sample_bsh_flag(np_prediction[ndx_time_period, :])
        else :
            print ('Analysis model is not specified')
        ndx_time_period += 1            
    logging.debug('\nforecast shape %s and values\n%s', np_forecast.shape, np_forecast)

    '''
    normalise the data in each row
        each data point for all historical data points is reduced to 0<=data<=+1
        1. Find the maximum value for each feature in each time series sample
        2. Normalize each feature value by dividing each value by the maximum value for that time series sample
    '''
    step3 = time.time()
    #print ("\tNormalizing data")
    logging.debug('')
    np_max = np.zeros([samples, feature_count])
    np_min = np.zeros([samples, feature_count])
    for ndx_feature_value in range(0, feature_count) : # normalize all features
        for ndx_feature in range(0, time_steps+forecast_steps) : # normalize only the time steps before the forecast time steps
            for ndx_time_period in range(0, samples) : # normalize all time periods
                if (np_data[ndx_time_period, ndx_feature, ndx_feature_value] > np_max[ndx_time_period, ndx_feature_value]) :
                    '''
                    logging.debug('New maximum %s, %s, %s was %s will be %s', \
                                  ndx_time_period , ndx_feature, ndx_feature_value, \
                                  np_max[ndx_time_period, ndx_feature_value], \
                                  np_data[ndx_time_period, ndx_feature, ndx_feature_value])
                    '''
                    np_max[ndx_time_period, ndx_feature_value] = np_data[ndx_time_period, ndx_feature, ndx_feature_value]
                if (np_data[ndx_time_period, ndx_feature, ndx_feature_value] < np_min[ndx_time_period, ndx_feature_value]) :
                    '''
                    logging.debug('New maximum %s, %s, %s was %s will be %s', \
                                  ndx_time_period , ndx_feature, ndx_feature_value, \
                                  np_max[ndx_time_period, ndx_feature_value], \
                                  np_data[ndx_time_period, ndx_feature, ndx_feature_value])
                    '''
                    np_min[ndx_time_period, ndx_feature_value] = np_data[ndx_time_period, ndx_feature, ndx_feature_value]
                ndx_time_period += 1            
            ndx_feature += 1
        ndx_feature_value += 1  

    for ndx_feature_value in range(0, feature_count) :
        for ndx_feature in range(0, time_steps+forecast_steps) :        
            for ndx_time_period in range(0, samples) :
                if np_min[ndx_time_period, ndx_feature_value] <= 0 :                    
                    np_data[ndx_time_period, ndx_feature, ndx_feature_value] += abs(np_min[ndx_time_period, ndx_feature_value])
                    np_max[ndx_time_period, ndx_feature_value] += abs(np_min[ndx_time_period, ndx_feature_value])
                np_data[ndx_time_period, ndx_feature, ndx_feature_value] = \
                    np_data[ndx_time_period, ndx_feature, ndx_feature_value] / np_max[ndx_time_period, ndx_feature_value]
                if np_data[ndx_time_period, ndx_feature, ndx_feature_value] == NAN :
                    logging.debug('NaN: %s %s %s', ndx_time_period, ndx_feature, ndx_feature_value) 
                ndx_time_period += 1            
            ndx_feature += 1
        logging.debug('normalized np_data feature values (0.0 to 1.0): %s, %s type: %s', \
                      ndx_feature_value, result_drivers[ndx_feature_value], type(np_data[0, 0, ndx_feature_value]))
        logging.debug('\n%s', np_data[: , : time_steps , ndx_feature_value] )
        ndx_feature_value += 1  

    step4 = time.time()
    '''
    Shuffle data and split into test and training portions
    '''
    np.random.shuffle(np_data)
    row = round(0.1 * np_data.shape[0])
    x_test  = np_data    [        :int(row), :time_steps , : ]
    x_train = np_data    [int(row):        , :time_steps , : ]
    y_test  = np_forecast[        :int(row)]
    y_train = np_forecast[int(row):        ]

    list_x_test  = list([])
    list_x_train = list([])
    lst_technical_analysis = []
    
    #Use list_x_test[0] on a model to learn to forecast based on "adj_low", "adj_high", "adj_open", "adj_close", "adj_volume"
    lst_technical_analysis.append('Market_Activity')
    list_x_test.append (x_test [:, :, :5])
    list_x_train.append(x_train[:, :, :5])
    
    #Use list_x_test[1] on a model to learn to forecast based on "BB_Lower", "BB_Upper"
    
    #Use list_x_test[2] on a model to learn to forecast based on "OBV"
    
    #Use list_x_test[3] on a model to learn to forecast based on "MACD_Sell"
    lst_technical_analysis.append('MACD_Sell')
    list_x_test.append (x_test [:, :, 10:11])
    list_x_train.append(x_train[:, :, 10:11])
    
    #Use list_x_test[3] on a model to learn to forecast based on "MACD_Buy"
    lst_technical_analysis.append('MACD_Buy')
    list_x_test.append (x_test [:, :, 11:12])
    list_x_train.append(x_train[:, :, 11:12])
    
    #Use list_x_test[3] on a model to learn to forecast based on "AccumulationDistribution"
   
    end = time.time()

    i_ndx = 0    
    for np_model in list_x_test:
        logging.debug('lst_technical_analysis %s, shape %s', lst_technical_analysis[i_ndx], list_x_train[i_ndx].shape)
        logging.info ('\nAnalyzing ndx_model \n\tdim[0] (samples)=%s,\n\tdim[1] (time series length)=%s\n\tdim[2] (feature count)=%s\n' % \
                     (np_model.shape[0], np_model.shape[1], np_model.shape[2]))
        logging.debug('%s data:\n%s', lst_technical_analysis[i_ndx], list_x_train[i_ndx])
        i_ndx += 1

    logging.info ("\tCreating time series took %s" % (step1 - start))
    logging.info ("\tStructuring 3D data took %s" % (step2 - step1))
    logging.info ("\tCalculating forecast y-axis characteristic value took %s" % (step3 - step2))
    logging.info ("\tNormalization took %s" % (step4 - step3))
    logging.info ("\tCreating test and training data took %s" % (end - step4))

    logging.info ('<---------------------------------------------------')
    logging.info ('<---- Including the following technical analyses:\n\t%s' % lst_technical_analysis)
    logging.info ('<---- prepare_ts_lstm shapes: \nx_test %s\nx_train %s\ny_test %s\ny_train %s', \
                 x_test.shape, x_train.shape, y_test.shape, y_train.shape)
    logging.debug('\nx_test\n%s\nx_train\n%s\ny_test\n%s\ny_train\n%s', \
                 x_test, x_train, y_test, y_train)
    logging.info ('<---------------------------------------------------')
    
    return [lst_technical_analysis, list_x_train, y_train, list_x_test, y_test]

def build_model(lst_analyses, np_input):
    logging.info('====> ==============================================')
    logging.info('====> build_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    start = time.time()

    kf_feature_sets = []
    kf_feature_set_outputs = []
    kf_feature_set_solo_outputs = []
    i_ndx = 0
    for np_feature_set in np_input:
        str_name = "{0}_input".format(lst_analyses[i_ndx])
        str_solo_out  = "{0}_output".format(lst_analyses[i_ndx])
        print           ('Building model - %s\n\tdim[0] (samples)=%s,\n\tdim[1] (time series length)=%s\n\tdim[2] (feature count)=%s' % \
                        (lst_analyses[i_ndx], np_feature_set.shape[0], np_feature_set.shape[1], np_feature_set.shape[2]))
        logging.debug   ("Building model - feature set %s\n\tInput dimensions: %s %s %s", \
                         lst_analyses[i_ndx], np_feature_set.shape[0], np_feature_set.shape[1], np_feature_set.shape[2])        

        #create and retain for model definition an input tensor for each technical analysis
        kf_feature_sets.append(Input(shape=(np_feature_set.shape[1], np_feature_set.shape[2], ), dtype='float32', name=str_name))
        print('\tkf_input shape %s' % tf.shape(kf_feature_sets[i_ndx]))

        #create the layers used to model each technical analysis
        kf_input_i_ndx = LSTM(32)(kf_feature_sets[i_ndx])
        kf_input_i_ndx = Dense(output_dim=1)(kf_input_i_ndx)

        #identify the output of each individual technical analysis
        kf_feature_set_output = Dense(output_dim=1)(kf_input_i_ndx)
        kf_feature_set_outputs.append(kf_feature_set_output)        

        #create outputs that can be used to assess the individual technical analysis         
        kf_feature_set_solo_output = Dense(output_dim=1, name=str_solo_out)(kf_feature_set_output)        
        kf_feature_set_solo_outputs.append(kf_feature_set_solo_output)        

        i_ndx += 1
    
    '''
    Create a model to take the feature set assessments and create a composite assessment
    '''        
    #combine all technical analysis assessments for a composite assessment
    kf_composite = Concatenate(axis=-1)(kf_feature_set_outputs[:])
    
    #create the layers used to analyze the composite of all technical analysis 
    kf_composite = Dense(len(kf_feature_set_outputs), activation='relu')(kf_composite)
    kf_composite = Dense(len(kf_feature_set_outputs), activation='relu')(kf_composite)
    kf_composite = Dense(len(kf_feature_set_outputs), activation='relu')(kf_composite)
    
    #create the composite output layer
    kf_composite = Dense(output_dim=1, name="composite_output")(kf_composite)
    
    #create list of outputs
    lst_outputs = []
    lst_outputs.append(kf_composite)
    for solo_output in kf_feature_set_solo_outputs:
        lst_outputs.append(solo_output)
    
    k_model = Model(inputs=kf_feature_sets, outputs=lst_outputs)
    k_model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    logging.info ("Time to compile: %s", time.time() - start)
    
    # 2 visualizations of the model
    save_model_plot(k_model)
    print_summary(k_model)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model')
    logging.info('<---- ----------------------------------------------')
    return k_model

def train_lstm(model, x_train, y_train):
    logging.info('')
    logging.info('====> ==============================================')
    logging.info("====> train_lstm: Fitting model using training data inputs: x=%s and y=%s", \
                 len(x_train), y_train.shape)
    logging.info('====> ==============================================')

    print('train_lstm: Fitting model using training data: len(x)=%s and y=%s' % \
                 (len(x_train), y_train.shape))
    
    '''
    model.fit(x_train[0], y_train, shuffle=True, batch_size=2056, nb_epoch=2, validation_split=0.05, verbose=1)
    '''
    lst_x = []
    lst_y = []
    for i_ndx in range(0, len(x_train)) :
        lst_x.append(x_train[i_ndx])
        lst_y.append(y_train)
    lst_y.append(y_train)
        
    model.fit(x=lst_x, y=lst_y, shuffle=True, batch_size=2056, nb_epoch=2, validation_split=0.05, verbose=1)
        
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- train_lstm:')
    logging.info('<---- ----------------------------------------------')
    
    return

def evaluate_model(model, x_data, y_data):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> evaluate_model')
    logging.debug('====> x_data includes %s inputs', len(x_data))
    logging.info ('====> ==============================================')
    
    lst_x = []
    lst_y = []
    for i_ndx in range(0, len(x_data)) :
        logging.info ('Input %s - shape %s', i_ndx, x_data[i_ndx].shape)
        lst_x.append(x_data[i_ndx])
        lst_y.append(y_data)
    lst_y.append(y_data)

    score = model.evaluate(x=lst_x, y=lst_y, verbose=0)    
    print ("\tevaluate_model: Test loss=%s, Test accuracy=%s" % (score[0], score[1]))

    logging.info('<---- ----------------------------------------------')
    logging.info('<---- evaluate_model: Test loss=%s, Test accuracy=%s', score[0], score[1])
    logging.info('<---- ----------------------------------------------')
    
    return

def predict_single(model, df_data):   
    '''
    *** Specific to the analysis being performed ***
    Find the maximum adj_high for each time period sample
    '''
    prediction = model.predict(x=df_data)

    return prediction

def predict_sequences_multiple(model, df_data):
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
    print        ('====> predict_sequences_multiple: technical analysis feature sets', len(df_data))
    for i_ndx in range(0, len(df_data)) :
        logging.info ('====> data shape=%s', df_data[i_ndx].shape)
        print        ('====> data shape= ', df_data[i_ndx].shape)
        logging.info ('====> data=\n%s', df_data[i_ndx])
    logging.info     ('====> ==============================================')
        
    samples = df_data[0].shape[0]
    np_predictions = np.empty([samples, len(df_data) + 1])
    logging.debug('Output shape: %s', np_predictions.shape)
    print('Output shape: ', np_predictions.shape)

    for ndx_samples in range(0, samples) :
        lst_x = []
        for i_ndx in range(0, len(df_data)) :
            lst_x.append(df_data[i_ndx][ndx_samples:ndx_samples+1])
        np_predictions[ndx_samples] = predict_single(model, lst_x)
        #np_predictions[ndx_samples] = model.predict(df_data[ndx_samples:ndx_samples+1])
        ndx_samples += 1

    #print ("%s predictions of length %s" % (len(prediction_seqs), len(prediction_seqs[0])))
    logging.info ('<---- ----------------------------------------------')
    logging.info ('<---- predict_sequences_multiple:')
    logging.debug('<---- prediction_seqs=\n%s', np_predictions)
    logging.info ('<---- ----------------------------------------------')
    
    return np_predictions

def plot_results_multiple(technical_analysis_names, predicted_data, true_data):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> plot_results_multiple: predicted_data shape=%s true_data shape=%s', predicted_data.shape, true_data.shape)
    logging.debug('====> \npredicted_data=\n%s\ntrue_data=\n%s', predicted_data, true_data)
    logging.info ('====> ==============================================')
        
    np_diff = np.zeros([predicted_data.shape[0], predicted_data.shape[1]])
    for ndx_data in range(0, predicted_data.shape[0]) :
        for ndx_output in range(0,predicted_data.shape[1]) :
            np_diff[ndx_data][ndx_output] = true_data[ndx_data] - predicted_data[ndx_data][ndx_output]
            ndx_output += 1
        ndx_data += 1
    '''
    On screen plot of actual and predicted data
    '''
    for ndx_output in range(0,predicted_data.shape[1]) :
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        np_output_diff = np_diff[:][ndx_output]
        ax.plot(np_output_diff, label = 'actual - prediction')
        if ndx_output<len(technical_analysis_names) :
            plt.legend(title=technical_analysis_names[ndx_output], loc='upper center', ncol=2)
        else :
            plt.legend(title='Composite actual / prediction difference', loc='upper center', ncol=2)
        plt.show()
        ndx_output += 1

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
from numpy import newaxis
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils import print_summary
from quandl_library import fetch_timeseries_data
from quandl_library import get_ini_data
from time_series_data import series_to_supervised

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

def prepare_ts_lstm(ticker, time_steps, forecast_steps, source=''):
    logging.info('')
    logging.info('====> ==============================================')
    logging.info('====> prepare_ts_lstm: ticker: %s, time_steps: %s, forecast_steps: %s', \
                  ticker, time_steps, forecast_steps)
    logging.info('====> ==============================================')
    
    '''
    Prepare time series data for presentation to an LSTM model from the Keras library
    seq_len - the number of elements from the time series for ?
    
    Returns training (90%) and test (10) sets
    '''
    
    '''
    Read data
    '''
    result_drivers = ["adj_low", "adj_high", "adj_volume"]
    forecast_feature = [False, True, False]
    feature_count = len(result_drivers)

    '''
    Load raw data
    '''
    df_data = fetch_timeseries_data(result_drivers, ticker, source)
    logging.info ('')
    logging.info ("Using %s features as predictors", feature_count)
    logging.info ("data shape: %s", df_data.shape)
    logging.debug('')
    logging.debug("%s: %s ... %s", result_drivers[0], df_data[ :3, 0], df_data[ -3: , 0])
    logging.debug("%s: %s ... %s", result_drivers[1], df_data[ :3, 1], df_data[ -3: , 1])
    
    '''
    Flatten data to a data frame where each row includes the 
        historical data driving the result - 
        time_steps examples of result_drivers data points
        result - 
        forecast_steps of data points (same data as drivers)
    '''
    #values = ts_data.values
    logging.info('Prepare as multi-variant time series data for LSTM')
    logging.info('series_to_supervised(values, time_steps=%s, forecast_steps=%s)', time_steps, forecast_steps)
    df_data = series_to_supervised(df_data, time_steps, forecast_steps)
    
    '''
    np_data[
        feature
        feature time steps and forecast steps:
        feature data point:
        ]
    '''
    samples = df_data.shape[0]
    #samples = 500
    
    np_data = np.empty([samples, time_steps+forecast_steps, feature_count])
    logging.debug('np_data shape: %s', np_data.shape)

    '''
    Convert 2D LSTM data frame into 3D Numpy array for data pre-processing
    '''
    for ndx_feature_value in range(0, feature_count) :
        for ndx_feature in range(0, time_steps+forecast_steps) :        
            for ndx_time_period in range(0, samples) :
                
                np_data[ndx_time_period, ndx_feature, ndx_feature_value] = \
                    df_data.iloc[ndx_time_period, (ndx_feature_value + (ndx_feature * feature_count))]
                
                ndx_time_period += 1            
            ndx_feature += 1
        logging.debug('\nnp_data raw feature values: %s, %s', ndx_feature_value, result_drivers[ndx_feature_value])
        logging.debug('\n%s', np_data[: , : , ndx_feature_value])
        ndx_feature_value += 1  
    
    '''
    *** Specific to the analysis being performed ***
    Calculate future % change of forecast feature
    '''
    for ndx_feature_value in range(0, feature_count) :
        if forecast_feature[ndx_feature_value] :
            for ndx_feature in range(time_steps, time_steps+forecast_steps) :        
                for ndx_time_period in range(0, samples) :
                
                    np_data[ndx_time_period, ndx_feature, ndx_feature_value] = \
                        np_data[ndx_time_period, ndx_feature, ndx_feature_value] / np_data[ndx_time_period, time_steps-1, ndx_feature_value]
                
                    ndx_time_period += 1            
                ndx_feature += 1
            logging.debug('\nnp_data feature forecast as percentage change : %s, %s', ndx_feature_value, result_drivers[ndx_feature_value])
            logging.debug('\n%s', np_data[: , time_steps : , ndx_feature_value])
        ndx_feature_value += 1  
    
    '''
    *** Specific to the analysis being performed ***
    Find the maximum adj_high for each time period sample
    '''
    np_max_forecast = np.zeros([samples])
    for ndx_feature_value in range(0, feature_count) :
        if forecast_feature[ndx_feature_value] :
            for ndx_feature in range(time_steps, time_steps+forecast_steps) :        
                for ndx_time_period in range(0, samples) :
                    
                    if (np_data[ndx_time_period, ndx_feature, ndx_feature_value] > np_max_forecast[ndx_time_period]) :
                        np_max_forecast[ndx_time_period] = np_data[ndx_time_period, ndx_feature, ndx_feature_value]
                    
                    ndx_time_period += 1            
                ndx_feature += 1
        ndx_feature_value += 1
    logging.debug('\nMaximum forecast values %s\n%s', np_max_forecast.shape, np_max_forecast)

    '''
    normalise the data in each row
    each data point for all historical data points is reduced to 0<=data<=+1
    1. Find the maximum value for each feature in each time series sample
    2. Normalize each feature value by dividing each value by the maximum value for that time series sample
    '''
    np_max = np.zeros([samples, feature_count])
    for ndx_feature_value in range(0, feature_count) : # normalize all features
        for ndx_feature in range(0, time_steps) : # normalize only the time steps before the forecast time steps
            for ndx_time_period in range(0, samples) : # normalize all time periods
                
                if (np_data[ndx_time_period, ndx_feature, ndx_feature_value] > np_max[ndx_time_period, ndx_feature_value]) :
                    '''
                    logging.debug('New maximum %s, %s, %s was %s will be %s', \
                                  ndx_time_period , ndx_feature, ndx_feature_value, \
                                  np_max[ndx_time_period, ndx_feature_value], \
                                  np_data[ndx_time_period, ndx_feature, ndx_feature_value])
                    '''
                    np_max[ndx_time_period, ndx_feature_value] = np_data[ndx_time_period, ndx_feature, ndx_feature_value]
                    
                ndx_time_period += 1            
            ndx_feature += 1
        ndx_feature_value += 1  


    for ndx_feature_value in range(0, feature_count) :
        for ndx_feature in range(0, time_steps) :        
            for ndx_time_period in range(0, samples) :
                
                np_data[ndx_time_period, ndx_feature, ndx_feature_value] = \
                    np_data[ndx_time_period, ndx_feature, ndx_feature_value] / np_max[ndx_time_period, ndx_feature_value]
                    
                ndx_time_period += 1            
            ndx_feature += 1
        logging.debug('np_data normalized feature values (0.0 to 1.0): %s, %s', ndx_feature_value, result_drivers[ndx_feature_value])
        logging.debug('\n%s', np_data[: , : time_steps , ndx_feature_value])
        ndx_feature_value += 1  

    '''
    Split into test and training portions
    '''
    row = round(0.9 * np_data.shape[0])
    logging.debug('Training on %s samples', int(row))
    x_test  = np_data[        :int(row), : , : ]
    x_train = np_data[int(row):        , : , : ]
    y_test  = np_max_forecast[        :int(row)]
    y_train = np_max_forecast[int(row):        ]

    '''
    Convert 3D pre-processed Numpy array back into 2D LSTM data frame 
    Drop columns we are not trying to predict
    df_lstm = df_data
    for ndx_feature_value in range(0, feature_count) :
        for ndx_feature in range(0, time_steps+forecast_steps) :        
            for ndx_time_period in range(0, samples) :
                
                x_lstm = (ndx_feature_value + (ndx_feature * feature_count))
                df_lstm.iloc[ ndx_time_period:ndx_time_period+1 , x_lstm:x_lstm+1] = \
                    np_data[ndx_time_period, ndx_feature, ndx_feature_value]
                
                ndx_time_period += 1            
            ndx_feature += 1
        ndx_feature_value += 1
        
    df_lstm = df_lstm.iloc[ :  ,  : (time_steps * feature_count) +1 ]
    logging.debug('Data prepared for LSTM: df_lstm\n type %s\n of shape %s', type(df_lstm), df_lstm.shape)
    logging.debug('\n%s\n%s', df_lstm.head(3), df_lstm.tail(3))
    '''
  
    '''
    Split into test and training portions
    row = round(0.9 * df_lstm.shape[0])
    logging.debug('Training on %s samples', int(row))
    x_test  = df_lstm.iloc[        :int(row),   :-1]
    x_train = df_lstm.iloc[int(row):        ,   :-1]
    y_test  = np_max_forecast[        :int(row)]
    y_train = np_max_forecast[int(row):        ]  
    '''

    '''[
    train = df_lstm.iloc[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]    
    y_train = train[:,  -1]
    #logging.debug ('x_train slice %s', x_train)
    #logging.debug ('train slice %s', train[:, 1])
    #logging.debug ('y_train slice %s', y_train)
    
    x_test = df_lstm[int(row):, :-1]
    y_test = df_lstm[int(row):,  -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test  = np.reshape(x_test,  (x_test.shape[0],  x_test.shape[1],  1))  
    #logging.debug ("training shapes - x: %s - y: %s", x_train.shape, y_train.shape)
    #logging.debug ("testing  shapes - x: %s - y: %s", x_test.shape,  y_test.shape)
    '''

    logging.info('<---------------------------------------------------')
    logging.info('<---- prepare_ts_lstm shapes: \nx_test %s\nx_train %s\ny_test %s\ny_train %s', \
                 x_test.shape, x_train.shape, y_test.shape, y_train.shape)
    logging.info('<---------------------------------------------------')
    logging.info('')

    return [x_train, y_train, x_test, y_test]

'''
def normalise_windows(raw_ts_data, time_steps, forecast_steps, data_points, norm=0):
    logging.info('')
    logging.info('===================================================')
    logging.info('normalise_windows')
    logging.info ("data type: %s, shape: %s, \nsequence length: %s, forecast_steps=%s, data_points=%s, norm=%s", \
                  type(raw_ts_data), raw_ts_data.shape, time_steps, forecast_steps, data_points, norm)
    raw_ts_data: 
    time_steps: 
    forecast_steps: 
    data_points: 
    norm=0 (default)
        normalize values to percentage change from the first element in the series
        range -1 to +1
        ISSUE:
        For time series data with large variability this code will NOT limit the returned values to
        the range -1 to 1. 
        Is this a problem????
    norm=1
        normalize the values to the percentage of the maximum value in the series
        range 0 to 1
        
    2. Separate each data point into its own data frame (iloc with list of lists)
    3. Normalize all data point data frames (pandas.Series.min / max)
    4. Recombine individual data points into single data frame
    
    normalised_data = raw_ts_data
                        
    logging.debug ("Normalized data\n%s", normalised_data)
    logging.info  ('===================================================')

    return normalised_data
'''

def build_model(np_input):
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
            fit
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
    logging.info('')
    logging.info('====> ==============================================')
    logging.info('====> build_model: Building model, input shape=%s', np_input.shape)
    logging.info('====> ==============================================')

    model = Sequential()
    model.add(LSTM(units=30,
                name="input_layer", 
                activation='tanh', 
                input_shape=(np_input.shape[1], np_input.shape[2]) ,
                return_sequences=False
                ))
    model.add(Dense(name="Output_layer", output_dim=1))

    '''
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
    '''
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    logging.info ("Time to compile: %s", time.time() - start)
    
    # 2 visualizations of the model
    save_model_plot(model)
    print_summary(model)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model: return')
    logging.info('<---- ----------------------------------------------')
    logging.info('')

    return model

def train_lstm(model, x_train, y_train):
    '''
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
    '''
    logging.info('')
    logging.info('====> ==============================================')
    logging.info("====> train_lstm: Fitting model using training data: x=%s and y=%s", \
                 x_train.shape, y_train.shape)
    logging.info('====> ==============================================')

    model.fit(x_train, y_train, shuffle=True, batch_size=512, nb_epoch=1, validation_split=0.05, verbose=1)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- train_lstm:')
    logging.info('<---- ----------------------------------------------')
    logging.info('')
    
    return

def evaluate_model(model, x_data, y_data):
    logging.info('evaluate_model')
    score = model.evaluate(x=x_data, y=y_data, verbose=0)
    print ("Test loss: ", score[0])
    print ("Test accuracy: ", score[1])
    
    return

def predict_sequences_multiple(model, data, series_length, prediction_len):
    logging.info('info')
    '''
    Use a trained model to forecast future behavior
    '''   
    prediction_seqs = []
    ts_ndx = int(len(data)/prediction_len)
    #print ("predict_sequences_multiple")
    logging.info ("predict_sequences_multiple: \ndata shape: %s (number of time series, series length, data points)\nwindow_size: %s, prediction_length: %s making %s predictions (%s/%s) one every %s days",  \
                  data.shape, series_length, prediction_len, ts_ndx, len(data), prediction_len, prediction_len)
    
    for i in range(ts_ndx):
        #Step through time series and make predictions over data from non-overlapping times
        #curr_frame is a 2D array [time series period, time series data value(s)]
        curr_frame = data[i*prediction_len]
        #print ("Predict based on: \n%s\nwith %s dimensions and shape %s" % (curr_frame, curr_frame.ndim, curr_frame.shape))
        
        predicted = []
        for j in range(prediction_len):
            '''
            predict( input Numpy data array, batch_size=None, verbose=0, steps=None )           
            '''
            predicted.append( model.predict(curr_frame[newaxis,:,:], verbose=0)[0,0])
            # Add predicted value to the end of the time series (include in future predictions)
            curr_frame = np.insert(curr_frame, series_length-1, predicted[-1], axis=0)
            # Step forward through the data one time interval
            curr_frame = curr_frame[1:]
            '''
            logging.debug ("for window %s, predicted len = %s\n%s\n, curr_frame len = %s\n%s", \
                          i, len(predicted), predicted, len(curr_frame), curr_frame)
            '''
        prediction_seqs.append(predicted)
        
    #print ("%s predictions of length %s" % (len(prediction_seqs), len(prediction_seqs[0])))
    
    return prediction_seqs

def plot_results_multiple(predicted_data, true_data, prediction_len):
    logging.info ("plot_results_multiple: %s predictions of length %s, based on %s time series", len(predicted_data), prediction_len, len(true_data))
    '''
    On screen plot of actual data and multiple sets of predicted data
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='actual % change')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        #prediction_label = "prediction" + str(i)
        #plt.plot(padding + data, label=prediction_label)
        plt.plot(padding + data)
        
    plt.legend(title='period to period % change and predictions', loc='upper center', ncol=2)
    plt.show()

''' ====================================================================
    =================   Currently unused methods   =====================
==================================================================== '''
'''
def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'r').read()
    data = f.split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    return [x_train, y_train, x_test, y_test]

def predict_point_by_point(model, data):
    print ("predict_point_by_point")
    
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data, verbose=0)
    predicted = np.reshape(predicted, (predicted.size,))
    
    return predicted

def predict_sequence_full(model, data, window_size):
    print ("predict_sequence_full")

    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:], verbose=0)[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        
    return predicted
'''

    
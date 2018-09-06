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
from numpy import newaxis, NaN, NAN
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
from tensorflow.python.layers.core import dense

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

def prepare_ts_lstm(tickers, result_drivers, forecast_feature, time_steps, forecast_steps, source=''):
    logging.info('')
    logging.info('====> ==============================================')
    logging.info('====> prepare_ts_lstm: time_steps: %s, forecast_steps: % stickers: \n\t%s', \
                  tickers, time_steps, forecast_steps)
    logging.info('====> ==============================================')
    print ("\tAccessing %s time steps, with %s time step future values for\n\t\t%s" % (time_steps, forecast_steps, tickers))
    
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
    print ("\tPreparing time series samples of \n\t\t%s" % result_drivers)
    logging.debug("Number of symbols: %s", len(tickers))
    logging.info('Prepare as multi-variant time series data for LSTM')
    logging.info('series_to_supervised(values, time_steps=%s, forecast_steps=%s)', time_steps, forecast_steps)
    
    for ndx_symbol in range(0, len(tickers)) :
        df_symbol = fetch_timeseries_data(result_drivers, tickers[ndx_symbol], source)

        logging.info ('')
        logging.info ("Using %s features as predictors", feature_count)
        logging.info ("data shape: %s", df_symbol.shape)
        logging.debug('')
        for ndx_feature_value in range(0, feature_count) :
            logging.debug("%s: %s ... %s", result_drivers[ndx_feature_value], df_symbol[ :3, ndx_feature_value], df_symbol[ -3: , ndx_feature_value])
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
        logging.debug("\ndf_data shape: %s",  df_data.shape)
        
    step1 = time.time()
    #print ("\tStructuring 3D data")
    '''
    np_data[
        Dim1: time series samples
        Dim2: feature time series
        Dim3: features
        ]
    '''
    samples = df_data.shape[0]
    
    np_data = np.empty([samples, time_steps+forecast_steps, feature_count])
    logging.debug('np_data shape: %s', np_data.shape)

    '''
    Convert 2D LSTM data frame into 3D Numpy array for data pre-processing
    '''
    logging.debug('')
    for ndx_feature_value in range(0, feature_count) :
        np_data[ : , : , ndx_feature_value] = df_data.iloc[:, np.r_[   ndx_feature_value : df_data.shape[1] : feature_count]]
        ndx_feature_value += 1  
    
    '''
    *** Specific to the analysis being performed ***
    Calculate future % change of forecast feature
    '''
    step2 = time.time()
    #print ("\tCalculating forecast y-axis characteristic value")
    logging.debug('')
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
    logging.debug('')
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
                    
                '''
                if np_data[ndx_time_period, ndx_feature, ndx_feature_value] == NAN :
                    logging.debug('NaN: %s %s %s', ndx_time_period, ndx_feature, ndx_feature_value) 
                '''
                ndx_time_period += 1            
            ndx_feature += 1
        logging.debug('normalized np_data feature values (0.0 to 1.0): %s, %s type: %s', \
                      ndx_feature_value, result_drivers[ndx_feature_value], type(np_data[0, 0, ndx_feature_value]))
        logging.debug('\n%s', np_data[: , : time_steps , ndx_feature_value] )
        ndx_feature_value += 1  

    step4 = time.time()
    '''
    Shuffle data
    '''
    np.random.shuffle(np_data)
    
    '''
    Split into test and training portions
    '''
    row = round(0.1 * np_data.shape[0])
    x_test  = np_data        [        :int(row), : , : ]
    x_train = np_data        [int(row):        , : , : ]
    y_test  = np_max_forecast[        :int(row)]
    y_train = np_max_forecast[int(row):        ]

    end = time.time()
    logging.info ("\tCreating time series took %s" % (step1 - start))
    logging.info ("\tStructuring 3D data took %s" % (step2 - step1))
    logging.info ("\tCalculating forecast y-axis characteristic value took %s" % (step3 - step2))
    logging.info ("\tNormalization took %s" % (step4 - step3))
    logging.info ("\tCreating test and training data took %s" % (end - step4))

    logging.info ('<---------------------------------------------------')
    logging.info ('<---- prepare_ts_lstm shapes: \nx_test %s\nx_train %s\ny_test %s\ny_train %s', \
                 x_test.shape, x_train.shape, y_test.shape, y_train.shape)
    logging.debug('\nx_test\n%s\nx_train\n%s\ny_test\n%s\ny_train\n%s', \
                 x_test, x_train, y_test, y_train)
    logging.info ('<---------------------------------------------------')
    
    return [x_train, y_train, x_test, y_test]

def build_model(np_input):
    logging.info('')
    logging.info('====> ==============================================')
    logging.info('====> build_model: Building model, input shape=%s', np_input.shape)
    logging.info('====> ==============================================')

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
    logging.debug("Input dimensions: %s %s %s", np_input.shape[0], np_input.shape[1], np_input.shape[2])
    model = Sequential()
    model.add(Dense(units=np_input.shape[1], input_shape=(np_input.shape[1], np_input.shape[2]), name="input_layer"))
    '''
    model.add(LSTM(units=np_input.shape[1],
                name="input_layer", 
                activation='tanh', 
                input_shape=(np_input.shape[1], np_input.shape[2]) ,
                return_sequences=False
                ))
    '''
    model.add(LSTM(units=np_input.shape[2], 
                   name="lstm_layer", 
                   activation='tanh', 
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
    logging.info('<---- build_model')
    logging.info('<---- ----------------------------------------------')

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

    model.fit(x_train, y_train, shuffle=True, batch_size=128, nb_epoch=2, validation_split=0.05, verbose=1)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- train_lstm:')
    logging.info('<---- ----------------------------------------------')
    
    return

def evaluate_model(model, x_data, y_data):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> evaluate_model')
    logging.debug('====> x_data = \n%s\ny_data = \n%s', x_data, y_data)
    logging.info ('====> ==============================================')
    
    score = model.evaluate(x=x_data, y=y_data, verbose=0)    
    print ("\tevaluate_model: Test loss=%s, Test accuracy=%s" % (score[0], score[1]))

    logging.info('<---- ----------------------------------------------')
    logging.info('<---- evaluate_model: Test loss=%s, Test accuracy=%s', score[0], score[1])
    logging.info('<---- ----------------------------------------------')
    
    return

def predict_single(model, df_data):
    '''
    df_data: 3D numpy array
        dimension 1: samples - length 1
        dimension 2: time series
        dimension 3: features
        
    logging.debug('')
    logging.debug('====> ==============================================')
    logging.debug('====> predict_single: data shape=%s', df_data.shape)
    logging.debug('====> data=\n%s', df_data)
    logging.debug('====> ==============================================')
    '''
        
    '''
    *** Specific to the analysis being performed ***
    Find the maximum adj_high for each time period sample
    '''
    prediction = model.predict(df_data)

    '''
    logging.debug('<---- ----------------------------------------------')
    logging.debug('<---- predict_single: %s', prediction)
    logging.debug('<---- ----------------------------------------------')
    '''
    return prediction

def predict_sequences_multiple(model, df_data, series_length, prediction_len):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> predict_sequences_multiple: data shape=%s', df_data.shape)
    logging.info ('====> data=\n%s', df_data)
    logging.info ('====> ==============================================')
        
    samples = df_data.shape[0]
    np_predictions = np.empty([samples])
    logging.debug('np_predictions shape: %s', np_predictions.shape)

    '''
    *** Specific to the analysis being performed ***
    Find the maximum adj_high for each time period sample
    '''
    for ndx_samples in range(0, samples) :
        np_predictions[ndx_samples] = predict_single(model, df_data[ndx_samples:ndx_samples+1])
        #np_predictions[ndx_samples] = model.predict(df_data[ndx_samples:ndx_samples+1])
        ndx_samples += 1

    #print ("%s predictions of length %s" % (len(prediction_seqs), len(prediction_seqs[0])))
    logging.info ('<---- ----------------------------------------------')
    logging.info ('<---- predict_sequences_multiple:')
    logging.debug('<---- prediction_seqs=\n%s', np_predictions)
    logging.info ('<---- ----------------------------------------------')
    
    return np_predictions

def plot_results_multiple(predicted_data, true_data):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> plot_results_multiple: predicted_data shape=%s true_data shape=%s', predicted_data.shape, true_data.shape)
    logging.debug('====> \npredicted_data=\n%s\ntrue_data=\n%s', predicted_data, true_data)
    logging.info ('====> ==============================================')
        
    np_diff = np.zeros(predicted_data.shape[0])
    ndx_data = 0
    for ndx_data in range(0, predicted_data.shape[0]) :
        np_diff[ndx_data] = true_data[ndx_data] - predicted_data[ndx_data] 
        ndx_data += 1
    '''
    On screen plot of actual and predicted data
    '''
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(np_diff, label='actual - prediction')
    plt.legend(title='actual / prediction difference', loc='upper center', ncol=2)
    plt.show()


    
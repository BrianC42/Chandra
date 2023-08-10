'''
Created on May 4, 2019

@author: Brian

Code specific to building, training, evaluating and using a model capable of returning an action flag
    Buy (1): data is indicating an increase in price >2% in the coming 30 days
    Hold (0) data is indicating the price will remain within 2% of the current price for the coming 30 days
    Sell (-1): data is indicating an decrease in price >2% in the coming 30 days
    
Classification architecture approaches 
    Machine Learning - use
        Dense, Activation, Dropout
    Convolutional Neural Network (CNN)
        Convolution1D, MaxPooling1D, MaxPooling1D, Dropout, Flatten, Dense, Activation
    Recurrent Neural Networks (RNN)
        LSTM, Dense, Activation
'''
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D
#from keras import optimizers
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.backend import dropout

from configuration_constants import ML_APPROACH
from configuration_constants import ACTIVATION
from configuration_constants import OPTIMIZER
from configuration_constants import USE_BIAS
from configuration_constants import DROPOUT
from configuration_constants import ANALASIS_SAMPLE_LENGTH
from configuration_constants import FORECAST_LENGTH
from configuration_constants import BUY_INDICATION_THRESHOLD
from configuration_constants import SELL_INDICATION_THRESHOLD
from configuration_constants import BUY_INDEX
from configuration_constants import HOLD_INDEX
from configuration_constants import SELL_INDEX
from configuration_constants import CLASSIFICATION_COUNT
from configuration_constants import CLASSIFICATION_ID
from configuration_constants import COMPILATION_LOSS
from configuration_constants import COMPILATION_METRICS
from configuration_constants import LOSS_WEIGHTS
from configuration_constants import DENSE_REGULARIZATION
from configuration_constants import REGULARIZATION_VALUE
from configuration_constants import DROPOUT
from configuration_constants import DROPOUT_RATE
from configuration_constants import LAYER_NODES
from configuration_constants import PREDICTION_PROBABILITY_THRESHOLD
from configuration_constants import ANALYSIS_LAYER_COUNT
from configuration_constants import COMPOSITE_LAYER_COUNT

from matplotlib._constrained_layout import do_constrained_layout
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator,  MONDAY, num2date
from matplotlib.dates import date2num

from mpl_finance import candlestick_ohlc
from mpl_finance import candlestick_ochl

def prepare_inputs(lst_analyses, np_input) :
    logging.info ('====> ==============================================')
    logging.info ('====> prepare_inputs:')
    logging.info ('====> ==============================================')

    kf_feature_sets = []
    kf_feature_set_outputs = []
    kf_feature_set_solo_outputs = []
    ndx_i = 0
    for np_feature_set in np_input:
        str_name = "{0}_input".format(lst_analyses[ndx_i])
        print           ('Building model - %s\n\tdim[0] (samples)=%s,\n\tdim[1] (time series length)=%s\n\tdim[2] (feature count)=%s' % \
                        (lst_analyses[ndx_i], np_feature_set.shape[0], np_feature_set.shape[1], np_feature_set.shape[2]))
        logging.debug   ("Building model - feature set %s\n\tInput dimensions: %s %s %s", \
                         lst_analyses[ndx_i], np_feature_set.shape[0], np_feature_set.shape[1], np_feature_set.shape[2])        

        #create and retain for model definition an input tensor for each technical analysis
        kf_feature_sets.append(Input(shape=(np_feature_set.shape[1], np_feature_set.shape[2], ), dtype='float32', name=str_name))
        print('\tkf_input shape %s' % tf.shape(kf_feature_sets[ndx_i]))
        ndx_i += 1
    
    logging.info('<==== ==============================================')
    logging.info('<==== prepare_inputs')
    logging.info('<==== ==============================================')
    return kf_feature_sets, kf_feature_set_outputs, kf_feature_set_solo_outputs

def bsh_data_check(lst_x, lst_y, f_out) :
    logging.debug('====> ==============================================')
    logging.debug('====> bsh_data_check: ')
    logging.debug('====> ==============================================')
    
    for data_set in lst_x :
        print ("sample data of shape", data_set.shape)
    
    for data_set in lst_y :
        np_class_counts = np.zeros([CLASSIFICATION_COUNT])
        for ndx_class in range (0, CLASSIFICATION_COUNT):
            np_class_counts[ndx_class] = np.count_nonzero(data_set[:, ndx_class])
        print ("target data, class example counts", data_set.shape, np_class_counts)
    
    logging.debug('<==== ==============================================')
    logging.debug('<==== bsh_data_check')
    logging.debug('<==== ==============================================')
    return

def combine_compile_model (analysis_counts, kf_feature_set_outputs, lst_analyses, kf_feature_set_solo_outputs, kf_feature_sets):
    logging.debug('====> ==============================================')
    logging.debug('====> combine_compile_model: ')
    logging.debug('====> ==============================================')

    # append common elements to each technical analysis
    for ndx_i in range (0, analysis_counts) :
        #start with the model specific layers already prepared
        kf_layer = kf_feature_set_outputs[ndx_i]
        #deepen the network
        for ndx_layer in range (0, ANALYSIS_LAYER_COUNT) :
            kf_layer = Dense(LAYER_NODES, activation=ACTIVATION, \
                             kernel_regularizer=regularizers.l2(REGULARIZATION_VALUE), \
                             activity_regularizer=regularizers.l1(REGULARIZATION_VALUE), \
                             )(kf_layer)
            kf_layer = Dropout(DROPOUT_RATE)(kf_layer)

        #identify the output of each individual technical analysis
        kf_feature_set_outputs[ndx_i] = Dense(LAYER_NODES, activation=ACTIVATION)(kf_layer)

        #create outputs that can be used to assess the individual technical analysis         
        str_solo_out  = "{0}_output".format(lst_analyses[ndx_i])
        kf_feature_set_solo_output = Dense(name=str_solo_out, output_dim=CLASSIFICATION_COUNT, activation='softmax')(kf_feature_set_outputs[ndx_i])        
        kf_feature_set_solo_outputs.append(kf_feature_set_solo_output)        
        ndx_i += 1

    #combine all technical analysis assessments for a composite assessment
    kf_composite = Concatenate(axis=-1)(kf_feature_set_outputs[:])

    #create the layers used to analyze the composite
    for ndx_layer in range (0, COMPOSITE_LAYER_COUNT) :
        kf_composite = Dense(LAYER_NODES, activation=ACTIVATION, \
                            kernel_regularizer=regularizers.l2(REGULARIZATION_VALUE), \
                            activity_regularizer=regularizers.l1(REGULARIZATION_VALUE), \
                            )(kf_composite)
        kf_composite = Dropout(DROPOUT_RATE)(kf_composite)

    #create the composite output layer
    kf_composite = Dense(output_dim=CLASSIFICATION_COUNT, activation='softmax', name="composite_output")(kf_composite)
    
    #create list of outputs
    lst_outputs = []
    lst_outputs.append(kf_composite)
    for solo_output in kf_feature_set_solo_outputs:
        lst_outputs.append(solo_output)
    
    k_model = Model(inputs=kf_feature_sets, outputs=lst_outputs)
    k_model.compile(optimizer=OPTIMIZER, loss=COMPILATION_LOSS, metrics=COMPILATION_METRICS, loss_weights=LOSS_WEIGHTS, \
                    sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

    logging.debug('<==== ==============================================')
    logging.debug('<==== combine_compile_model')
    logging.debug('<==== ==============================================')
    return k_model

def build_cnn_model(lst_analyses, np_input) :
    '''
    Keras Convolutional Neural Network
        keras.layers.Conv1D(
            filters, 
            kernel_size, 
            strides=1, 
            padding='valid', 
            data_format='channels_last', 
            dilation_rate=1, 
            activation=None, 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros', 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None)
    Inputs:
    2 dimensional data
        Samples consisting of
            Feature set values
    Outputs:
        Composite and individual feature set classifications
    '''
    logging.info('====> ==============================================')
    logging.info('====> build_cnn_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    kf_feature_sets, kf_feature_set_outputs, kf_feature_set_solo_outputs = prepare_inputs(lst_analyses, np_input)
    #create the layers used to model each technical analysis
    for ndx_i in range (0, len(np_input)) :
        #Add a pair of convolutional layers
        kf_layer = Conv1D(filters=ANALASIS_SAMPLE_LENGTH, kernel_size=FORECAST_LENGTH, activation=ACTIVATION)(kf_feature_sets[ndx_i])        
        kf_layer = Conv1D(filters=ANALASIS_SAMPLE_LENGTH, kernel_size=FORECAST_LENGTH, activation=ACTIVATION)(kf_layer)
        kf_layer = MaxPooling1D(pool_size=2, strides=None, padding='valid')(kf_layer)
        
        #Add a second pair of convolutional layers
        kf_layer = Conv1D(filters=FORECAST_LENGTH, kernel_size=1, activation=ACTIVATION)(kf_layer)        
        kf_layer = Conv1D(filters=FORECAST_LENGTH, kernel_size=1, activation=ACTIVATION)(kf_layer)
        kf_layer = MaxPooling1D(pool_size=5, strides=None, padding='valid')(kf_layer)
        
        #flatten the 2D time series / feature value data for dense layer processing
        kf_layer = Flatten()(kf_layer)               
        kf_feature_set_outputs.append(kf_layer)        
        ndx_i += 1
        
    k_model = combine_compile_model (len(np_input), kf_feature_set_outputs, lst_analyses, kf_feature_set_solo_outputs, kf_feature_sets)
    
    logging.info('<==== ==============================================')
    logging.info('<==== build_cnn_model')
    logging.info('<==== ==============================================')
    return k_model

def build_mlp_model(lst_analyses, np_input) :
    '''
    Keras Core Neural Network
    Inputs:
    2 dimensional data
        Samples consisting of
        Feature set values
    Outputs:
        Composite and individual feature set classifications
    '''
    logging.info('====> ==============================================')
    logging.info('====> build_mlp_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    kf_feature_sets, kf_feature_set_outputs, kf_feature_set_solo_outputs = prepare_inputs(lst_analyses, np_input)
    #create the layers used to model each technical analysis
    for ndx_i in range (0, len(np_input)) :
        #flatten the 2D time series / feature value data for dense layer processing
        kf_layer = Flatten()(kf_feature_sets[ndx_i])
        
        kf_feature_set_outputs.append(kf_layer)        
    k_model = combine_compile_model (len(np_input), kf_feature_set_outputs, lst_analyses, kf_feature_set_solo_outputs, kf_feature_sets)
    
    logging.info('<==== ==============================================')
    logging.info('<==== build_mlp_model')
    logging.info('<==== ==============================================')
    return k_model

def build_rnn_lstm_model(lst_analyses, np_input) :
    '''
    Keras Recurrent Neural Network
    Inputs:
    3 dimensional data
        Samples consisting of
            Time Series of
                Feature set values
    Outputs:
        Composite and individual feature set classifications
    '''
    logging.info('====> ==============================================')
    logging.info('====> build_rnn_lstm_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    kf_feature_sets, kf_feature_set_outputs, kf_feature_set_solo_outputs = prepare_inputs(lst_analyses, np_input)
    #create the layers used to model each technical analysis
    for ndx_i in range (0, len(np_input)) :
        #create the layers used to model each technical analysis
        kf_layer = LSTM(LAYER_NODES, activation=ACTIVATION, use_bias=USE_BIAS, dropout=DROPOUT)(kf_feature_sets[ndx_i])
        kf_feature_set_outputs.append(kf_layer)        
    k_model = combine_compile_model (len(np_input), kf_feature_set_outputs, lst_analyses, kf_feature_set_solo_outputs, kf_feature_sets)

    logging.info('<==== ==============================================')
    logging.info('<==== build_rnn_lstm_model')
    logging.info('<==== ==============================================')
    return k_model

def build_bsh_classification_model(lst_analyses, np_input) :
    logging.info('====> ==============================================')
    logging.info('====> build_bsh_classification_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')
    
    start = time.time()
    '''
    Alternative model architectures
    '''
    if (ML_APPROACH == 'core') :
        k_model = build_mlp_model(lst_analyses, np_input)
    elif (ML_APPROACH == 'recurrent') :
        k_model = build_rnn_lstm_model(lst_analyses, np_input)
    elif (ML_APPROACH == 'convolutional') :
        k_model = build_cnn_model(lst_analyses, np_input)
    logging.info ("Time to compile: %s", time.time() - start)

    logging.info('<==== ==============================================')
    logging.info('<==== build_bsh_classification_model')
    logging.info('<==== ==============================================')
    return k_model

def calculate_sample_bsh_flag(sample_single_flags):
    '''
    return a classification array where all values are zero except for a single array entry which is set to 1
    Determine which array entry to set to 1 based on the maximum value of the individual changes and based on thresholds
    '''
    bsh_classification = [0, 0, 0]

    bsh_flag_max = np.amax(sample_single_flags)
    bsh_flag_min = np.amin(sample_single_flags)
    
    if (bsh_flag_max >= BUY_INDICATION_THRESHOLD) :
        bsh_classification[BUY_INDEX] = CLASSIFICATION_ID
    elif (bsh_flag_min <= SELL_INDICATION_THRESHOLD) :
        bsh_classification[SELL_INDEX] = CLASSIFICATION_ID
    else :
        bsh_classification[HOLD_INDEX] = CLASSIFICATION_ID
    
    return bsh_classification

def calculate_single_bsh_flag(current_price, future_price):
    '''
    Calculate the percentage change between the current_price and future_price
    '''
    bsh_change = future_price / current_price

    return bsh_change

def balance_bsh_classifications(training_data, actual_classifications) :
    logging.info('====> ==============================================')
    logging.info('====> balance_bsh_classifications: original training data shape %s,classification shape %s', \
                 training_data.shape, actual_classifications.shape)
    logging.info('====> ==============================================')
    
    np_counts = np.zeros([CLASSIFICATION_COUNT], dtype=int)
    np_classification_count = np.zeros([CLASSIFICATION_COUNT])
    
    '''
    Assess the accuracy of each output of the prediction for each possible classification
    '''
    for ndx_classification in range (0, CLASSIFICATION_COUNT) :
        # Count actual buy, sell and hold indications
        np_counts[ndx_classification] = np.count_nonzero(actual_classifications[:, ndx_classification])
        
    bsh_count_min = np_counts[np.argmin(np_counts)]    
    np_balanced_training_data   = np.empty([bsh_count_min * CLASSIFICATION_COUNT, training_data.shape[1], training_data.shape[2]])
    np_balanced_classifications = np.empty([bsh_count_min * CLASSIFICATION_COUNT, actual_classifications.shape[1]])
    
    ndx_balanced_sample = 0
    for ndx_sample in range (0, training_data.shape[0]) :
        i_classification = np.argmax(actual_classifications[ndx_sample])
        
        if np_classification_count[i_classification] < bsh_count_min :
            np_balanced_training_data[ndx_balanced_sample, :, :] = training_data[ndx_sample, :, :]
            np_balanced_classifications[ndx_balanced_sample, :] = actual_classifications[ndx_sample, :]
            np_classification_count[i_classification] += 1
            ndx_balanced_sample += 1
    
    logging.info('<==== ==============================================')
    logging.info('<==== classification counts: %s, least classifica+tion index: %d', np_counts, bsh_count_min)
    logging.info('<==== balance_bsh_classifications: balanced training data shape %s,classification shape %s', \
                 np_balanced_training_data.shape, np_balanced_classifications.shape)
    logging.info('<==== balance_bsh_classifications')
    logging.info('<==== ==============================================')    
    return np_balanced_training_data, np_balanced_classifications

def plot_bsh_results(lst_analyses, x_train, y_train, x_test, y_test) :
    logging.info ('====> ==============================================')
    logging.info ('====> plot_bsh_results:')
    logging.info ('====> ==============================================')

    fig, axis = plt.subplots(nrows=2, ncols=2, figsize=(9, 4), constrained_layout=True)
    
    # Market data (high, low, open, close and volume)
    data = []
    day = 1.0
    np_data = x_train[0]
    for ndx_ts in range (0, len(np_data[0])):
        data.append([day, \
                     np_data[0, ndx_ts, 2], \
                     np_data[0, ndx_ts, 3], \
                     np_data[0, ndx_ts, 1], \
                     np_data[0, ndx_ts, 0] \
                    ])
        day += 1

    candlestick_ochl(axis[0, 0], data, colorup = "green", colordown = "red")
    axis[0, 0].set_title(lst_analyses[0])
    axis[0, 0].set_xlabel('Time Series')
    axis[0, 0].set_ylabel('ochl')
    axis[0, 0].yaxis.grid(True)
    
    # Bollinger bands
    axis[0, 1].set_title(lst_analyses[1])
    axis[0, 0].set_xlabel('Time Series')
    axis[0, 0].set_ylabel('?')
    axis[0, 0].yaxis.grid(True)
    
    # Accumulation Distribution
    axis[1, 0].set_title(lst_analyses[2])
    axis[0, 0].set_xlabel('Time Series')
    axis[0, 0].set_ylabel('?')
    axis[0, 0].yaxis.grid(True)
    
    # MACD buy / sell
    axis[1, 1].set_title(lst_analyses[3] + lst_analyses[3])
    axis[0, 0].set_xlabel('Time Series')
    axis[0, 0].set_ylabel('?')
    axis[0, 0].yaxis.grid(True)
    
    plt.show()

    return

def plot_bsh_result_distribution(technical_analysis_names, predicted_data, true_data):
    logging.info ('====> ==============================================')
    logging.info ('====> plot_bsh_result_distribution:')
    logging.info ('====> ==============================================')
    logging.debug('Technical analysis names: %s\nPredicted data shape%s', technical_analysis_names, predicted_data.shape)

    
    return
    
def categorize_prediction_risks(technical_analysis_names, predicted_data, true_data, f_out) :
    '''
    Generate a report to demonstrate the accuracy of the composite and technical analysis by comparing the counts of
    actual buy, sell and hold indications with the predictions
    technical_analysis_names - list of individual technical analysis names
    predicted_data - 3D array of predictions from the model
        Axis 0 - the sample analyzed
        Axis 1 - the output the composite output at index 0 and one output for each technical analysis at index 1 and beyond
        Axis 2 - the CLASSIFICATION_COUNT values output by the model
    true_data - One Hot Encoded 2D array of actual buy/sell/hold indications for each sample
        Axis 0 - sample
        Axis 1 - one hot classification encoding
    f_out - execution report file
    '''
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> categorize_prediction_risks:')
    logging.info ('====> ==============================================')

    actual_sample_count = true_data.shape[0]
    prediction_count = predicted_data.shape[1]

    np_counts                               = np.zeros([                  CLASSIFICATION_COUNT])
    np_predictions_classification           = np.zeros([prediction_count, CLASSIFICATION_COUNT])
    np_characterization                     = np.zeros([prediction_count, CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_percentage          = np.zeros([prediction_count, CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_threshold           = np.zeros([prediction_count, CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_threshold_pct       = np.zeros([prediction_count, CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_probability         = np.zeros([                  CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_probability_pct     = np.zeros([                  CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_concensus_count     = np.zeros([                  CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_concensus_count_pct = np.zeros([                  CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_concensus_sum       = np.zeros([                  CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    np_characterization_concensus_sum_pct   = np.zeros([                  CLASSIFICATION_COUNT, CLASSIFICATION_COUNT])
    
    '''
    Assess the accuracy of each output of the prediction for each possible classification
    '''
    for ndx_classification in range (0, CLASSIFICATION_COUNT) :
        # Count actual buy, sell and hold indications
        np_counts[ndx_classification]  = np.count_nonzero(true_data[:, ndx_classification])

    '''
    Assess the accuracy of the composite and each individual technical analysis
    Calculate accuracy counts
    '''
    for ndx_sample in range (0, actual_sample_count) :
        i_true_classification = np.argmax(true_data[ndx_sample, :]) #true_data classification (from one hot encoding)
        
        '''
        calculate combination counts
        '''
        # Find class with the highest prediction probability
        f_probability_max   = 0.0
        f_Probability_tmp   = 0.0
        i_classification    = 0                              # Which class has the highest prediction probability
        for ndx2_classification in range (0, CLASSIFICATION_COUNT) :
            for ndx2_prediction in range (0, prediction_count) : # composite and individual predictions
                f_Probability_tmp = predicted_data[ndx_sample, ndx2_prediction, ndx2_classification]
                if f_Probability_tmp > f_probability_max :
                    f_probability_max = f_Probability_tmp
                    i_classification = ndx2_classification
        np_characterization_probability[i_classification, i_true_classification] += 1

        # Count class with most predictions with the highest probability
        np_prediction_count = np.zeros([CLASSIFICATION_COUNT]) # Which class is predicted most
        for ndx2_prediction in range (0, prediction_count) :
            np_prediction_count[np.argmax(predicted_data[ndx_sample, ndx2_prediction, :])] += 1
        np_characterization_concensus_count[np.argmax(np_prediction_count[:]), i_true_classification]  += 1
            
        # Count class with highest sum of prediction values
        np_prediction_sum   = np.zeros([CLASSIFICATION_COUNT]) # Sum of prediction values
        for ndx2_classification in range (0, CLASSIFICATION_COUNT) :
            for ndx2_prediction in range (0, prediction_count) : # composite and individual predictions
                np_prediction_sum[ndx2_classification] += predicted_data[ndx_sample, ndx2_prediction, ndx2_classification]
        np_characterization_concensus_sum[np.argmax(np_prediction_sum[:]), i_true_classification]  += 1

        '''
        Count all possible classification combinations of prediction and actual
        '''
        for ndx_predicted in range (0, prediction_count) :
            # predicted classification output with the highest value
            i_prediction_classification = np.argmax(predicted_data[ndx_sample, ndx_predicted, :]) 
            
            #Count the number of predictions of each specific classification
            np_predictions_classification[ndx_predicted, i_prediction_classification] += 1

            if i_true_classification == BUY_INDEX :
                if (i_prediction_classification == SELL_INDEX) :
                    # actual buy / predicted sell
                    np_characterization [ndx_predicted, SELL_INDEX, BUY_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, SELL_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, SELL_INDEX, BUY_INDEX] += 1
                elif (i_prediction_classification == HOLD_INDEX) :
                    # actual buy / predicted hold
                    np_characterization [ndx_predicted, HOLD_INDEX, BUY_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, HOLD_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, HOLD_INDEX, BUY_INDEX] += 1
                else :
                    # actual buy / predicted buy
                    np_characterization [ndx_predicted, BUY_INDEX, BUY_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, BUY_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, BUY_INDEX, BUY_INDEX] += 1
            elif  i_true_classification == SELL_INDEX :
                if (i_prediction_classification == SELL_INDEX) :
                    # actual sell / predicted sell
                    np_characterization [ndx_predicted, SELL_INDEX, SELL_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, SELL_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, SELL_INDEX, SELL_INDEX] += 1
                elif (i_prediction_classification == HOLD_INDEX) :
                    # actual sell / predicted hold
                    np_characterization [ndx_predicted, HOLD_INDEX, SELL_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, HOLD_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, HOLD_INDEX, SELL_INDEX] += 1
                else :
                    # actual sell / predicted buy
                    np_characterization [ndx_predicted, BUY_INDEX, SELL_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, BUY_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, BUY_INDEX, SELL_INDEX] += 1
            else :
                if (i_prediction_classification == SELL_INDEX) :
                    # actual hold / predicted sell
                    np_characterization [ndx_predicted, SELL_INDEX, HOLD_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, SELL_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, SELL_INDEX, HOLD_INDEX] += 1
                elif (i_prediction_classification == HOLD_INDEX) :
                    # actual hold / predicted hold
                    np_characterization [ndx_predicted, HOLD_INDEX, HOLD_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, HOLD_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, HOLD_INDEX, HOLD_INDEX] += 1
                else :
                    # actual hold / predicted buy
                    np_characterization [ndx_predicted, BUY_INDEX, HOLD_INDEX] += 1
                    if predicted_data[ndx_sample, ndx_predicted, BUY_INDEX] > PREDICTION_PROBABILITY_THRESHOLD :
                        np_characterization_threshold [ndx_predicted, BUY_INDEX, HOLD_INDEX] += 1

    '''
    Calculate the percentage accuracy
    '''
    for ndx_bsh_actual in range (0, CLASSIFICATION_COUNT) :
        for ndx_bsh_predicted in range (0, CLASSIFICATION_COUNT) :
            np_characterization_probability_pct     [ndx_bsh_predicted, ndx_bsh_actual] = \
                np_characterization_probability     [ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count
            np_characterization_concensus_count_pct [ndx_bsh_predicted, ndx_bsh_actual] = \
                np_characterization_concensus_count [ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count
            np_characterization_concensus_sum_pct   [ndx_bsh_predicted, ndx_bsh_actual] = \
                np_characterization_concensus_sum   [ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count
                
            for ndx_predicted in range (0, prediction_count) :
                np_characterization_percentage[ndx_predicted, ndx_bsh_predicted, ndx_bsh_actual] = \
                    np_characterization[ndx_predicted, ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count
                np_characterization_threshold_pct[ndx_predicted, ndx_bsh_predicted, ndx_bsh_actual] = \
                    np_characterization_threshold [ndx_predicted, ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count
                        
    '''
    Format the report and output it to the screen, logging and report file
    '''
    logging.debug('\nAnalysis names:\t%s\npredicted data shape: %s\nactual data shape: %s', \
                  technical_analysis_names, predicted_data.shape, true_data.shape)
    logging.debug('Result characterizations:\n%s', np_characterization)
    logging.debug('Result characterizations pct:\n%s', np_characterization_percentage)

    str_summary  = '\n=========================================================================================================='
    str_summary1 = 'Prediction results can be categorized as follows (based on simple greatest probability):'
    str_summary2 = 'The model includes {:.0f} technical analysis methods.'.format(len(technical_analysis_names))
    str_summary3 = 'The model combines the predictions of these analyses into a combined prediction which is shown first'
    f_out.write ('\n' + str_summary)   
    f_out.write ('\n' + str_summary2)   
    f_out.write ('\n' + str_summary3)   
    f_out.write ('\n' + str_summary1)   
    print       (str_summary)   
    print       (str_summary2)   
    print       (str_summary3)   
    print       (str_summary1)   
    logging.info(str_summary)
    logging.info(str_summary2)
    logging.info(str_summary3)
    logging.info(str_summary1)
    
    str_actual_totals = '\nActual\ttotal:\t{:.0f}\tbuys:\t{:.0f}\tholds:\t{:.0f}\tsells:\t{:.0f}\n'.format( \
                actual_sample_count, np_counts[BUY_INDEX], np_counts[HOLD_INDEX], np_counts[SELL_INDEX] \
                )
    f_out.write ('\n' + str_actual_totals)
    print       (str_actual_totals)
    logging.info(str_actual_totals)

    for ndx_predicted in range (0, prediction_count) :
        if ndx_predicted == 0 :
            f_out.write ('Composite analysis')   
            print       ('Composite analysis')
            logging.info('Composite analysis')
        else :
            str_analysis = '{:s}'.format(technical_analysis_names[ndx_predicted-1])
            f_out.write (str_analysis)      
            print       (str_analysis)
            logging.info(str_analysis)
        str_l1 = '\t\t\t\t\t\t\tPredictions'
        str_l2 = '\t\t\t\t|\tBuy\t\t|\tSell\t\t|\tHold'
        str_l3 = '\tA-------------------------------------------------------------------------------------------------'
        str_l4 = '\tc\tBuy\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[BUY_INDEX], \
                    np_characterization[ndx_predicted, BUY_INDEX, BUY_INDEX] , np_characterization_percentage[ndx_predicted, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_predicted, SELL_INDEX, BUY_INDEX], np_characterization_percentage[ndx_predicted, SELL_INDEX, BUY_INDEX], \
                    np_characterization[ndx_predicted, HOLD_INDEX, BUY_INDEX], np_characterization_percentage[ndx_predicted, HOLD_INDEX, BUY_INDEX] )
        str_l6 = '\tt-------------------------------------------------------------------------------------------------'
        str_l7 = '\tu\tSell\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[SELL_INDEX], \
                    np_characterization[ndx_predicted, BUY_INDEX, SELL_INDEX] , np_characterization_percentage[ndx_predicted, BUY_INDEX, SELL_INDEX], \
                    np_characterization[ndx_predicted, SELL_INDEX, SELL_INDEX], np_characterization_percentage[ndx_predicted, SELL_INDEX, SELL_INDEX], \
                    np_characterization[ndx_predicted, HOLD_INDEX, SELL_INDEX], np_characterization_percentage[ndx_predicted, HOLD_INDEX, SELL_INDEX] )
        str_l9 = '\ta-------------------------------------------------------------------------------------------------'
        str_l10 = '\tl\tHold\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[HOLD_INDEX], \
                    np_characterization[ndx_predicted, BUY_INDEX, HOLD_INDEX] , np_characterization_percentage[ndx_predicted, BUY_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_predicted, SELL_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_predicted, SELL_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_predicted, HOLD_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_predicted, HOLD_INDEX, HOLD_INDEX] )
        f_out.write ('\n' + str_l1)
        f_out.write ('\n' + str_l2)
        f_out.write ('\n' + str_l3)
        f_out.write ('\n' + str_l4)
        f_out.write ('\n' + str_l6)
        f_out.write ('\n' + str_l7)
        f_out.write ('\n' + str_l9)
        f_out.write ('\n' + str_l10)
        logging.info(str_l1)
        logging.info(str_l2)
        logging.info(str_l3)
        logging.info(str_l4)
        logging.info(str_l6)
        logging.info(str_l7)
        logging.info(str_l9)
        logging.info(str_l10)
        print       (str_l1)
        print       (str_l2)
        print       (str_l3)
        print       (str_l4)
        print       (str_l6)
        print       (str_l7)
        print       (str_l9)
        print       (str_l10)


    f_out.write ('\n' + str_summary)   
    print       (str_summary)   
    logging.info(str_summary)

    str_summary20 = 'The following predictions ignore any prediction below a {:0.0%} probability'.format(PREDICTION_PROBABILITY_THRESHOLD)
    f_out.write ('\n' + str_summary20)   
    print       (str_summary20)   
    logging.info(str_summary20)

    for ndx_predicted in range (0, prediction_count) :
        if ndx_predicted == 0 :
            f_out.write ('Composite analysis')   
            print       ('Composite analysis')
            logging.info('Composite analysis')
        else :
            str_analysis = '{:s}'.format(technical_analysis_names[ndx_predicted-1])
            f_out.write (str_analysis)      
            print       (str_analysis)
            logging.info(str_analysis)
        str_threshold_buy = '\tc\tBuy\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[BUY_INDEX], \
                                np_characterization_threshold    [ndx_predicted, BUY_INDEX, BUY_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, BUY_INDEX, BUY_INDEX], \
                                np_characterization_threshold    [ndx_predicted, SELL_INDEX, BUY_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, SELL_INDEX, BUY_INDEX], \
                                np_characterization_threshold    [ndx_predicted, HOLD_INDEX, BUY_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, HOLD_INDEX, BUY_INDEX] )
        str_threshold_sell = '\tu\tSell\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[SELL_INDEX], \
                                np_characterization_threshold    [ndx_predicted, BUY_INDEX, SELL_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, BUY_INDEX, SELL_INDEX], \
                                np_characterization_threshold    [ndx_predicted, SELL_INDEX, SELL_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, SELL_INDEX, SELL_INDEX], \
                                np_characterization_threshold    [ndx_predicted, HOLD_INDEX, SELL_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, HOLD_INDEX, SELL_INDEX] )
        str_threshold_hold = '\tl\tHold\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[HOLD_INDEX], \
                                np_characterization_threshold    [ndx_predicted, BUY_INDEX, HOLD_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, BUY_INDEX, HOLD_INDEX], \
                                np_characterization_threshold    [ndx_predicted, SELL_INDEX, HOLD_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, SELL_INDEX, HOLD_INDEX], \
                                np_characterization_threshold    [ndx_predicted, HOLD_INDEX, HOLD_INDEX], \
                                np_characterization_threshold_pct[ndx_predicted, HOLD_INDEX, HOLD_INDEX] )

        f_out.write ('\n' + str_l1)
        f_out.write ('\n' + str_l2)
        f_out.write ('\n' + str_l3)
        f_out.write ('\n' + str_threshold_buy)
        f_out.write ('\n' + str_l6)
        f_out.write ('\n' + str_threshold_sell)
        f_out.write ('\n' + str_l9)
        f_out.write ('\n' + str_threshold_hold)
        print       (str_l1)
        print       (str_l2)
        print       (str_l3)
        print       (str_threshold_buy)
        print       (str_l6)
        print       (str_threshold_sell)
        print       (str_l9)
        print       (str_threshold_hold)
        logging.info(str_l1)
        logging.info(str_l2)
        logging.info(str_l3)
        logging.info(str_threshold_buy)
        logging.info(str_l6)
        logging.info(str_threshold_sell)
        logging.info(str_l9)
        logging.info(str_threshold_hold)

    f_out.write ('\n' + str_summary)   
    print       (str_summary)   
    logging.info(str_summary)

    str_summary10 = 'In addition to the predictions above the results can be combine in a number of ways'
    str_summary11 = 'Highest probability from all predictions'
    str_summary12 = 'Consensus - class selected by most predictions'
    str_summary13 = 'Concensus - class selected by the sum of all predictions'
    str_probability_buy = '\tc\tBuy\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[BUY_INDEX], \
                    np_characterization_probability[BUY_INDEX, BUY_INDEX], \
                    np_characterization_probability_pct[BUY_INDEX, BUY_INDEX], \
                    np_characterization_probability[SELL_INDEX, BUY_INDEX], \
                    np_characterization_probability_pct[SELL_INDEX, BUY_INDEX], \
                    np_characterization_probability[HOLD_INDEX, BUY_INDEX], \
                    np_characterization_probability_pct[HOLD_INDEX, BUY_INDEX] )
    str_probability_sell = '\tu\tSell\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[SELL_INDEX], \
                    np_characterization_probability[BUY_INDEX, SELL_INDEX], \
                    np_characterization_probability_pct[BUY_INDEX, SELL_INDEX], \
                    np_characterization_probability[SELL_INDEX, SELL_INDEX], \
                    np_characterization_probability_pct[SELL_INDEX, SELL_INDEX], \
                    np_characterization_probability[HOLD_INDEX, SELL_INDEX], \
                    np_characterization_probability_pct[HOLD_INDEX, SELL_INDEX] )
    str_probability_hold = '\tl\tHold\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[HOLD_INDEX], \
                    np_characterization_probability[BUY_INDEX, HOLD_INDEX], \
                    np_characterization_probability_pct[BUY_INDEX, HOLD_INDEX], \
                    np_characterization_probability[SELL_INDEX, HOLD_INDEX], \
                    np_characterization_probability_pct[SELL_INDEX, HOLD_INDEX], \
                    np_characterization_probability[HOLD_INDEX, HOLD_INDEX], \
                    np_characterization_probability_pct[HOLD_INDEX, HOLD_INDEX] )

    str_consensus_count_buy = '\tc\tBuy\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[BUY_INDEX], \
                    np_characterization_concensus_count[BUY_INDEX, BUY_INDEX], \
                    np_characterization_concensus_count_pct[BUY_INDEX, BUY_INDEX], \
                    np_characterization_concensus_count[SELL_INDEX, BUY_INDEX], \
                    np_characterization_concensus_count_pct[SELL_INDEX, BUY_INDEX], \
                    np_characterization_concensus_count[HOLD_INDEX, BUY_INDEX], \
                    np_characterization_concensus_count_pct[HOLD_INDEX, BUY_INDEX] )
    str_consensus_count_sell = '\tu\tSell\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[SELL_INDEX], \
                    np_characterization_concensus_count[BUY_INDEX, SELL_INDEX], \
                    np_characterization_concensus_count_pct[BUY_INDEX, SELL_INDEX], \
                    np_characterization_concensus_count[SELL_INDEX, SELL_INDEX], \
                    np_characterization_concensus_count_pct[SELL_INDEX, SELL_INDEX], \
                    np_characterization_concensus_count[HOLD_INDEX, SELL_INDEX], \
                    np_characterization_concensus_count_pct[HOLD_INDEX, SELL_INDEX] )
    str_consensus_count_hold = '\tl\tHold\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[HOLD_INDEX], \
                    np_characterization_concensus_count[BUY_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_count_pct[BUY_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_count[SELL_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_count_pct[SELL_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_count[HOLD_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_count_pct[HOLD_INDEX, HOLD_INDEX] )

    str_consensus_sum_buy = '\tc\tBuy\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[BUY_INDEX], \
                    np_characterization_concensus_sum[BUY_INDEX, BUY_INDEX], \
                    np_characterization_concensus_sum_pct[BUY_INDEX, BUY_INDEX], \
                    np_characterization_concensus_sum[SELL_INDEX, BUY_INDEX], \
                    np_characterization_concensus_sum_pct[SELL_INDEX, BUY_INDEX], \
                    np_characterization_concensus_sum[HOLD_INDEX, BUY_INDEX], \
                    np_characterization_concensus_sum_pct[HOLD_INDEX, BUY_INDEX] )
    str_consensus_sum_sell = '\tu\tSell\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[SELL_INDEX], \
                    np_characterization_concensus_sum[BUY_INDEX, SELL_INDEX], \
                    np_characterization_concensus_sum_pct[BUY_INDEX, SELL_INDEX], \
                    np_characterization_concensus_sum[SELL_INDEX, SELL_INDEX], \
                    np_characterization_concensus_sum_pct[SELL_INDEX, SELL_INDEX], \
                    np_characterization_concensus_sum[HOLD_INDEX, SELL_INDEX], \
                    np_characterization_concensus_sum_pct[HOLD_INDEX, SELL_INDEX] )
    str_consensus_sum_hold = '\tl\tHold\t{:.0f}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}\t|\t{:.0f}\t{:.2%}'.format( np_counts[HOLD_INDEX], \
                    np_characterization_concensus_sum[BUY_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_sum_pct[BUY_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_sum[SELL_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_sum_pct[SELL_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_sum[HOLD_INDEX, HOLD_INDEX], \
                    np_characterization_concensus_sum_pct[HOLD_INDEX, HOLD_INDEX] )

    f_out.write ('\n' + str_summary10)   
    f_out.write ('\n' + str_summary11)   
    f_out.write ('\n' + str_l1)
    f_out.write ('\n' + str_l2)
    f_out.write ('\n' + str_l3)
    f_out.write ('\n' + str_probability_buy)
    f_out.write ('\n' + str_l6)
    f_out.write ('\n' + str_probability_sell)
    f_out.write ('\n' + str_l9)
    f_out.write ('\n' + str_probability_hold)
    f_out.write ('\n' + str_summary12)   
    f_out.write ('\n' + str_l1)
    f_out.write ('\n' + str_l2)
    f_out.write ('\n' + str_l3)
    f_out.write ('\n' + str_consensus_count_buy)
    f_out.write ('\n' + str_l6)
    f_out.write ('\n' + str_consensus_count_sell)
    f_out.write ('\n' + str_l9)
    f_out.write ('\n' + str_consensus_count_hold)
    f_out.write ('\n' + str_summary13)   
    f_out.write ('\n' + str_l1)
    f_out.write ('\n' + str_l2)
    f_out.write ('\n' + str_l3)
    f_out.write ('\n' + str_consensus_sum_buy)
    f_out.write ('\n' + str_l6)
    f_out.write ('\n' + str_consensus_sum_sell)
    f_out.write ('\n' + str_l9)
    f_out.write ('\n' + str_consensus_sum_hold)
    print       (str_summary10)   
    print       (str_summary11)   
    print       (str_l1)
    print       (str_l2)
    print       (str_l3)
    print       (str_probability_buy)
    print       (str_l6)
    print       (str_probability_sell)
    print       (str_l9)
    print       (str_probability_hold)
    print       (str_summary12)   
    print       (str_l1)
    print       (str_l2)
    print       (str_l3)
    print       (str_consensus_count_buy)
    print       (str_l6)
    print       (str_consensus_count_sell)
    print       (str_l9)
    print       (str_consensus_count_hold)
    print       (str_summary13)   
    print       (str_l1)
    print       (str_l2)
    print       (str_l3)
    print       (str_consensus_sum_buy)
    print       (str_l6)
    print       (str_consensus_sum_sell)
    print       (str_l9)
    print       (str_consensus_sum_hold)
    logging.info(str_summary10)
    logging.info(str_summary11)
    logging.info(str_l1)
    logging.info(str_l2)
    logging.info(str_l3)
    logging.info(str_probability_buy)
    logging.info(str_l6)
    logging.info(str_probability_sell)
    logging.info(str_l9)
    logging.info(str_probability_hold)
    logging.info(str_summary12)
    logging.info(str_l1)
    logging.info(str_l2)
    logging.info(str_l3)
    logging.info(str_consensus_count_buy)
    logging.info(str_l6)
    logging.info(str_consensus_count_sell)
    logging.info(str_l9)
    logging.info(str_consensus_count_hold)
    logging.info(str_summary13)
    logging.info(str_l1)
    logging.info(str_l2)
    logging.info(str_l3)
    logging.info(str_consensus_sum_buy)
    logging.info(str_l6)
    logging.info(str_consensus_sum_sell)
    logging.info(str_l9)
    logging.info(str_consensus_sum_hold)

    f_out.write ('\n' + str_summary)   
    print       (str_summary)   
    logging.info(str_summary)

    return

def bsh_results_multiple(technical_analysis_names, predicted_data, true_data, f_out) :
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> bsh_results_multiple: predicted_data shape=%s true_data shape=%s', predicted_data.shape, true_data.shape)
    logging.debug('====> \npredicted_data=\n%s\ntrue_data=\n%s', predicted_data, true_data)
    logging.info ('====> ==============================================')
        
    '''
    On screen display of actual and predicted data
    '''
    categorize_prediction_risks(technical_analysis_names, predicted_data, true_data, f_out)
    #plot_bsh_result_distribution(technical_analysis_names, predicted_data, true_data)
    '''
    Display plots of differences
    np_diff = np.zeros([predicted_data.shape[0], predicted_data.shape[1]])
    for ndx_data in range(0, predicted_data.shape[0]) :
        for ndx_output in range(0,predicted_data.shape[1]) :
            np_diff[ndx_data][ndx_output] = true_data[ndx_data] - predicted_data[ndx_data][ndx_output]
    plot_bsh_results(technical_analysis_names, predicted_data, true_data, np_diff)
    '''
    
    return

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

from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
#from keras import optimizers

from configuration_constants import ML_APPROACH
from configuration_constants import ACTIVATION
from configuration_constants import OPTIMIZER
from configuration_constants import USE_BIAS
from configuration_constants import DROPOUT
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

from keras.layers.pooling import MaxPooling1D

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

def combine_compile_model (analysis_counts, kf_feature_set_outputs, lst_analyses, kf_feature_set_solo_outputs, kf_feature_sets):
    logging.info('====> ==============================================')
    logging.info('====> combine_compile_model: ')
    logging.info('====> ==============================================')

    # append common elements to each technical analysis
    for ndx_i in range (0, analysis_counts) :
        #start with the model specific layers already prepared
        kf_layer = kf_feature_set_outputs[ndx_i]
        #deepen the network
        kf_layer = Dense(256, activation=ACTIVATION)(kf_layer)
        kf_layer = Dense(256, activation=ACTIVATION)(kf_layer)
        #identify the output of each individual technical analysis
        kf_feature_set_outputs[ndx_i] = Dense(120, activation=ACTIVATION)(kf_layer)
        #kf_feature_set_outputs.append(kf_feature_set_output)        
        #create outputs that can be used to assess the individual technical analysis         
        str_solo_out  = "{0}_output".format(lst_analyses[ndx_i])
        kf_feature_set_solo_output = Dense(name=str_solo_out, output_dim=CLASSIFICATION_COUNT, activation='softmax')(kf_feature_set_outputs[ndx_i])        
        kf_feature_set_solo_outputs.append(kf_feature_set_solo_output)        
        ndx_i += 1

    #combine all technical analysis assessments for a composite assessment
    kf_composite = Concatenate(axis=-1)(kf_feature_set_outputs[:])
    #create the layers used to analyze the composite
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    #create the composite output layer
    kf_composite = Dense(output_dim=CLASSIFICATION_COUNT, activation='softmax', name="composite_output")(kf_composite)
    
    #create list of outputs
    lst_outputs = []
    lst_outputs.append(kf_composite)
    for solo_output in kf_feature_set_solo_outputs:
        lst_outputs.append(solo_output)
    
    k_model = Model(inputs=kf_feature_sets, outputs=lst_outputs)
    '''
    optimizer:            String (name of optimizer) or optimizer instance. See optimizers.
    loss:                 String (name of objective function) or objective function. See losses. 
                            If the model has multiple outputs, you can use a different loss on each output by passing a dictionary or a list of losses. 
                            The loss value that will be minimized by the model will then be the sum of all individual losses.
    metrics:              List of metrics to be evaluated by the model during training and testing. Typically you will use metrics=['accuracy']. 
                            To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as 
                            metrics={'output_a': 'accuracy'}.
    loss_weights:         Optional list or dictionary specifying scalar coefficients (Python floats) to weight the loss contributions of 
                            different model outputs. The loss value that will be minimized by the model will then be the weighted sum of all 
                            individual losses, weighted by the loss_weights coefficients. If a list, it is expected to have a 1:1 mapping to the 
                            model's outputs. If a tensor, it is expected to map output names (strings) to scalar coefficients.
    sample_weight_mode:   If you need to do timestep-wise sample weighting (2D weights), set this to "temporal".  None defaults to sample-wise weights (1D). 
                            If the model has multiple outputs, you can use a different  sample_weight_mode on each output by passing a dictionary 
                            or a list of modes.
    weighted_metrics:     List of metrics to be evaluated and weighted by sample_weight or class_weight during training and testing.
    target_tensors:       By default, Keras will create placeholders for the model's target, which will be fed with the target data during training. 
                            If instead you would like to use your own target tensors (in turn, Keras will not expect external Numpy data for these 
                            targets at training time), you can specify them via the target_tensors argument. It can be a single tensor 
                            (for a single-output model), a list of tensors, or a dict mapping output names to target tensors.
    **kwargs:             When using the Theano/CNTK backends, these arguments are passed into K.function. When using the TensorFlow backend, 
                            these arguments are passed into tf.Session.run.    '''
    k_model.compile(optimizer=OPTIMIZER, loss=COMPILATION_LOSS, metrics=COMPILATION_METRICS, loss_weights=LOSS_WEIGHTS, \
                    sample_weight_mode=None, weighted_metrics=None, target_tensors=None)

    logging.info('<==== ==============================================')
    logging.info('<==== combine_compile_model')
    logging.info('<==== ==============================================')
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
        kf_layer = Conv1D(filters=120, kernel_size=30, activation=ACTIVATION)(kf_feature_sets[ndx_i])        
        kf_layer = Conv1D(filters=120, kernel_size=30, activation=ACTIVATION)(kf_layer)
        kf_layer = MaxPooling1D(pool_size=2, strides=None, padding='valid')(kf_layer)
        
        #Add a second pair of convolutional layers
        kf_layer = Conv1D(filters=30, kernel_size=5, activation=ACTIVATION)(kf_layer)        
        kf_layer = Conv1D(filters=30, kernel_size=5, activation=ACTIVATION)(kf_layer)
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
        kf_layer = LSTM(256, activation=ACTIVATION, use_bias=USE_BIAS, dropout=DROPOUT)(kf_feature_sets[ndx_i])
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

def calculate_single_bsh_flag(current_price, future_price):
    '''
    Calculate the percentage change between the current_price and future_price
    '''
    bsh_change = future_price / current_price

    return bsh_change

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

def plot_bsh_results(technical_analysis_names, predicted_data, true_data, np_diff) :
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> plot_bsh_results:')
    logging.info ('====> ==============================================')

    for ndx_output in range(0,predicted_data.shape[1]) :
        fig = plt.figure(facecolor='white')
        ax = fig.add_subplot(111)
        np_output_diff = np_diff[:, ndx_output]
        logging.debug('plotting values, shape %s, values\n%s', np_output_diff.shape, np_output_diff)
        ax.plot(np_output_diff, label = 'actual - prediction')
        if ndx_output < len(technical_analysis_names) :
            plt.legend(title=technical_analysis_names[ndx_output], loc='upper center', ncol=2)
        else :
            plt.legend(title='Composite actual / prediction difference', loc='upper center', ncol=2)
        plt.show()

    return

def plot_bsh_result_distribution(technical_analysis_names, predicted_data, true_data):
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> plot_bsh_result_distribution:')
    logging.info ('====> ==============================================')
    logging.debug('Technical analysis names: %s\nPredicted data shape%s', technical_analysis_names, predicted_data.shape)

    fig = plt.figure(facecolor='white')
    for ndx_output in range(0, predicted_data.shape[1]) :
        if ndx_output == 0 :
            str_prediction_basis = 'Composite'
        else :
            str_prediction_basis = technical_analysis_names[ndx_output - 1]
        ax = fig.add_subplot(2, 3, ndx_output+1, title=str_prediction_basis)
        n, bins, patches = ax.hist(predicted_data[:, ndx_output])
        logging.debug('hist returns\nn:%s\nbins: %s', n, bins)
        ax.set_xlabel('Predicted value')
        ax.set_ylabel('Prediction Counts')
    plt.show()
    
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

    ACTUAL_CHARACTERISTIC_COUNTS = CLASSIFICATION_COUNT
    PREDICTION_CHARACTERISTICS_COUNTS = CLASSIFICATION_COUNT
    
    actual_sample_count = true_data.shape[0]
    prediction_count = predicted_data.shape[1]

    np_counts                       = np.zeros([                  PREDICTION_CHARACTERISTICS_COUNTS])
    np_predictions_classification   = np.zeros([prediction_count, PREDICTION_CHARACTERISTICS_COUNTS])
    np_characterization             = np.zeros([prediction_count, PREDICTION_CHARACTERISTICS_COUNTS, ACTUAL_CHARACTERISTIC_COUNTS])
    np_characterization_percentage  = np.zeros([prediction_count, PREDICTION_CHARACTERISTICS_COUNTS, ACTUAL_CHARACTERISTIC_COUNTS])
    
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
    for ndx_predicted in range (0, prediction_count) :
        for ndx_sample in range (0, actual_sample_count) :
            '''
            Identify the 
                predicted classification output with the highest value
                true_data classification (from one hot encoding)
            '''
            i_prediction_classification = np.argmax(predicted_data[ndx_sample, ndx_predicted, :])
            i_true_classification = np.argmax(true_data[ndx_sample, :])
            
            #Count the number of predictions of each specific classification
            np_predictions_classification[ndx_predicted, i_prediction_classification] += 1

            '''
            Count all possible classification combinations of prediction and actual
            '''
            if i_true_classification == BUY_INDEX :
                if (i_prediction_classification == SELL_INDEX) :
                    # actual buy / predicted sell
                    np_characterization [ndx_predicted, SELL_INDEX, BUY_INDEX] += 1
                elif (i_prediction_classification == HOLD_INDEX) :
                    # actual buy / predicted hold
                    np_characterization [ndx_predicted, HOLD_INDEX, BUY_INDEX] += 1
                else :
                    # actual buy / predicted buy
                    np_characterization [ndx_predicted, BUY_INDEX, BUY_INDEX] += 1
            elif  i_true_classification == SELL_INDEX :
                if (i_prediction_classification == SELL_INDEX) :
                    # actual sell / predicted sell
                    np_characterization [ndx_predicted, SELL_INDEX, SELL_INDEX] += 1
                elif (i_prediction_classification == HOLD_INDEX) :
                    # actual sell / predicted hold
                    np_characterization [ndx_predicted, HOLD_INDEX, SELL_INDEX] += 1
                else :
                    # actual sell / predicted buy
                    np_characterization [ndx_predicted, BUY_INDEX, SELL_INDEX] += 1
            else :
                if (i_prediction_classification == SELL_INDEX) :
                    # actual hold / predicted sell
                    np_characterization [ndx_predicted, SELL_INDEX, HOLD_INDEX] += 1
                elif (i_prediction_classification == HOLD_INDEX) :
                    # actual hold / predicted hold
                    np_characterization [ndx_predicted, HOLD_INDEX, HOLD_INDEX] += 1
                else :
                    # actual hold / predicted buy
                    np_characterization [ndx_predicted, BUY_INDEX, HOLD_INDEX] += 1
            
    '''
    Calculate the percentage accuracy
    '''
    for ndx_predicted in range (0, prediction_count) :
        for ndx_bsh_actual in range (0, ACTUAL_CHARACTERISTIC_COUNTS) :
            for ndx_bsh_predicted in range (0, PREDICTION_CHARACTERISTICS_COUNTS) :
                np_characterization_percentage[ndx_predicted, ndx_bsh_predicted, ndx_bsh_actual] = \
                    np_characterization[ndx_predicted, ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count

    '''
    Format the report and output it to the screen, logging and report file
    '''
    logging.debug('\nAnalysis names:\t%s\npredicted data shape: %s\nactual data shape: %s', \
                  technical_analysis_names, predicted_data.shape, true_data.shape)
    logging.debug('Result characterizations:\n%s', np_characterization)
    logging.debug('Result characterizations pct:\n%s', np_characterization_percentage)

    str_summary = '\nPrediction results can be categorized as follows:'
    f_out.write ('\n' + str_summary)   
    print       (str_summary)   
    logging.info(str_summary)
    
    str_actual_totals = 'Actual\ttotal:\t{:.0f}\tbuys:\t{:.0f}\tholds:\t{:.0f}\tsells:\t{:.0f}'.format( \
                actual_sample_count, np_counts[BUY_INDEX], np_counts[HOLD_INDEX], np_counts[SELL_INDEX] \
                )
    f_out.write ('\n' + str_actual_totals)
    print       (str_actual_totals)
    logging.info(str_actual_totals)

    for ndx_predicted in range (0, prediction_count) :
        if ndx_predicted == 0 :
            f_out.write ('\nComposite analysis')   
            print       ('Composite analysis')
            logging.info('Composite analysis')
        else :
            str_analysis = '\n{:s}'.format(technical_analysis_names[ndx_predicted-1])
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

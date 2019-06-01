'''
Created on May 4, 2019

@author: Brian

Code specific to building, training, evaluating and using a model capable of returning an action flag
    Buy (1): data is indicating an increase in price >2% in the coming 30 days
    Hold (0) data is indicating the price will remain within 2% of the current price for the coming 30 days
    Sell (-1): data is indicating an decrease in price >2% in the coming 30 days
'''
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.layers import Concatenate
from keras.models import Model

from configuration_constants import ACTIVATION
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
from configuration_constants import COMPILATION_OPTIMIZER
from configuration_constants import COMPILATION_METRICS

def build_mlp_model(lst_analyses, np_input) :
    logging.info('====> ==============================================')
    logging.info('====> build_mlp_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    k_model = 1
    
    logging.info('<==== ==============================================')
    logging.info('<==== build_mlp_model')
    logging.info('<==== ==============================================')
    return k_model

def build_cnn_model(lst_analyses, np_input) :
    logging.info('====> ==============================================')
    logging.info('====> build_cnn_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    k_model = 1
    
    logging.info('<==== ==============================================')
    logging.info('<==== build_cnn_model')
    logging.info('<==== ==============================================')
    return k_model

def build_rnn_lstm_model(lst_analyses, np_input) :
    logging.info('====> ==============================================')
    logging.info('====> build_rnn_lstm_model: Building model, analyses %s', lst_analyses)
    logging.info('====> inputs=%s', len(np_input))
    logging.info('====> ==============================================')

    kf_feature_sets = []
    kf_feature_set_outputs = []
    kf_feature_set_solo_outputs = []
    ndx_i = 0
    for np_feature_set in np_input:
        str_name = "{0}_input".format(lst_analyses[ndx_i])
        str_solo_out  = "{0}_output".format(lst_analyses[ndx_i])
        print           ('Building model - %s\n\tdim[0] (samples)=%s,\n\tdim[1] (time series length)=%s\n\tdim[2] (feature count)=%s' % \
                        (lst_analyses[ndx_i], np_feature_set.shape[0], np_feature_set.shape[1], np_feature_set.shape[2]))
        logging.debug   ("Building model - feature set %s\n\tInput dimensions: %s %s %s", \
                         lst_analyses[ndx_i], np_feature_set.shape[0], np_feature_set.shape[1], np_feature_set.shape[2])        

        #create and retain for model definition an input tensor for each technical analysis
        kf_feature_sets.append(Input(shape=(np_feature_set.shape[1], np_feature_set.shape[2], ), dtype='float32', name=str_name))
        print('\tkf_input shape %s' % tf.shape(kf_feature_sets[ndx_i]))

        #create the layers used to model each technical analysis
        kf_input_ndx_i = LSTM(256, activation=ACTIVATION, use_bias=USE_BIAS, dropout=DROPOUT)(kf_feature_sets[ndx_i])
        kf_input_ndx_i = Dense(256, activation=ACTIVATION)(kf_input_ndx_i)
        kf_input_ndx_i = Dense(256, activation=ACTIVATION)(kf_input_ndx_i)
        kf_input_ndx_i = Dense(256, activation=ACTIVATION)(kf_input_ndx_i)

        #identify the output of each individual technical analysis
        kf_feature_set_output = Dense(output_dim=CLASSIFICATION_COUNT)(kf_input_ndx_i)
        kf_feature_set_outputs.append(kf_feature_set_output)        

        #create outputs that can be used to assess the individual technical analysis         
        kf_feature_set_solo_output = Dense(name=str_solo_out, output_dim=CLASSIFICATION_COUNT)(kf_feature_set_output)        
        kf_feature_set_solo_outputs.append(kf_feature_set_solo_output)        

        ndx_i += 1
    
    '''
    Create a model to take the feature set assessments and create a composite assessment
    '''        
    #combine all technical analysis assessments for a composite assessment
    kf_composite = Concatenate(axis=-1)(kf_feature_set_outputs[:])
    
    #create the layers used to analyze the composite of all technical analysis 
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    kf_composite = Dense(256, activation=ACTIVATION)(kf_composite)
    
    #create the composite output layer
    kf_composite = Dense(output_dim=CLASSIFICATION_COUNT, name="composite_output")(kf_composite)
    
    #create list of outputs
    lst_outputs = []
    lst_outputs.append(kf_composite)
    for solo_output in kf_feature_set_solo_outputs:
        lst_outputs.append(solo_output)
    
    k_model = Model(inputs=kf_feature_sets, outputs=lst_outputs)
    k_model.compile(loss=COMPILATION_LOSS, optimizer=COMPILATION_OPTIMIZER, metrics=COMPILATION_METRICS)

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
    k_model = build_rnn_lstm_model(lst_analyses, np_input)
    #k_model = build_cnn_model(lst_analyses, np_input)
    #k_model = build_mlp_model(lst_analyses, np_input)
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
            
        str_prediction_range = '\tPrediction values range from\t{:f} to {:f}'.format( \
                    min(predicted_data[:, ndx_predicted, BUY_INDEX]), max(predicted_data[:, ndx_predicted, BUY_INDEX]))
        str_prediction_counts = '\tPredicted\t\t\tbuys:\t\t{:.0f}\t\tholds:\t\t{:.0f}\t\tsells:\t\t{:.0f}'.format( \
                    np_predictions_classification[ndx_predicted, BUY_INDEX], \
                    np_predictions_classification[ndx_predicted, HOLD_INDEX], \
                    np_predictions_classification[ndx_predicted, SELL_INDEX] \
                    )
        str_correct_prediction = '\tCorrect predictions:\t\tBuy\t\t{:.0f}\t{:.2%}\tHold\t\t{:.0f}\t{:.2%}\tSell\t\t{:.0f}\t{:.2%}'.format( \
                    np_characterization[ndx_predicted, BUY_INDEX, BUY_INDEX], np_characterization_percentage[ndx_predicted, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_predicted, HOLD_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_predicted, HOLD_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_predicted, SELL_INDEX, SELL_INDEX], np_characterization_percentage[ndx_predicted, SELL_INDEX, SELL_INDEX] \
                    )
        str_lost_opprtunities = '\tLost opportunities:\t\thold as sell\t{:.0f}\t{:.2%}\tbuy as hold\t{:.0f}\t{:.2%}\tbuy as sell\t{:.0f}\t{:.2%}'.format( \
                    np_characterization[ndx_predicted, SELL_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_predicted, SELL_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_predicted, HOLD_INDEX, BUY_INDEX], np_characterization_percentage[ndx_predicted, HOLD_INDEX, BUY_INDEX], \
                    np_characterization[ndx_predicted, SELL_INDEX, BUY_INDEX], np_characterization_percentage[ndx_predicted, SELL_INDEX, BUY_INDEX] \
                    )
        str_financial_loss = '\tFinancial loss if acted on:\tsell as hold\t{:.0f}\t{:.2%}\tsell as buy\t{:.0f}\t{:.2%}\thold as buy\t{:.0f}\t{:.2%}'.format( \
                    np_characterization[ndx_predicted, HOLD_INDEX, SELL_INDEX], np_characterization_percentage[ndx_predicted, HOLD_INDEX, SELL_INDEX], \
                    np_characterization[ndx_predicted, BUY_INDEX, SELL_INDEX], np_characterization_percentage[ndx_predicted, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_predicted, BUY_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_predicted, BUY_INDEX, HOLD_INDEX] \
                    )
        
        f_out.write ('\n' + str_prediction_counts)   
        #f_out.write ('\n' + str_prediction_range)   
        f_out.write ('\n' + str_correct_prediction)
        f_out.write ('\n' + str_lost_opprtunities)
        f_out.write ('\n' + str_financial_loss)
        logging.info(str_prediction_counts)
        #logging.info(str_prediction_range)
        logging.info(str_correct_prediction)
        logging.info(str_lost_opprtunities)
        logging.info(str_financial_loss)
        print       (str_prediction_counts)
        #print       (str_prediction_range)
        print       (str_correct_prediction)
        print       (str_lost_opprtunities)
        print       (str_financial_loss)

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

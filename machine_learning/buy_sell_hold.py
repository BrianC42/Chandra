'''
Created on May 4, 2019

@author: Brian

Code specific to building, training, evaluating and using a model capable of returning an action flag
    Buy (1): data is indicating an increase in price >2% in the coming 30 days
    Hold (0) data is indicating the price will remain within 2% of the current price for the coming 30 days
    Sell (-1): data is indicating an decrease in price >2% in the coming 30 days
'''
import logging
import numpy as np
import matplotlib.pyplot as plt

def calculate_single_bsh_flag(current_price, future_price):
    
    bsh_change = future_price / current_price
    if bsh_change >= 1.2 :
        # 3% increase
        bsh_flag = 1
    elif bsh_change <= 0.8 :
        # 3% decline
        bsh_flag = -1
    else :
        # change between -3% and +3%
        bsh_flag = 0

    return bsh_flag

def calculate_sample_bsh_flag(sample_single_flags):
    
    bsh_flag = 0
    
    bsh_flag_max = np.amax(sample_single_flags)
    bsh_flag_min = np.amin(sample_single_flags)
    
    if (bsh_flag_max == 1) :
        bsh_flag = 1
    elif (bsh_flag_min == -1) :
        bsh_flag = -1
    else :
        bsh_flag = 0
    
    return bsh_flag

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
        if ndx_output<len(technical_analysis_names) :
            plt.legend(title=technical_analysis_names[ndx_output], loc='upper center', ncol=2)
        else :
            plt.legend(title='Composite actual / prediction difference', loc='upper center', ncol=2)
        plt.show()

    return

def categorize_prediction_risks(technical_analysis_names, predicted_data, true_data, np_diff, f_out) :
    '''
    Actual bsh flag    Prediction        Categorization
    -1:   sell        <-0.5              Correct sell
    -1:   sell        -0.5<pred<+0.5     Sell predicted as hold - financial loss
    -1:   sell        >+0.5              Sell predicted as buy - financial loss
    0:    hold        <-0.5              Hold predicted as sell - opportunity loss
    0:    hold        -0.5<pred<+0.5     Correct hold
    0:    hold        >+0.5              Hold predicted as buy - financial loss
    1:    Buy         <-0.5              Buy predicted as sell - opportunity loss
    1:    Buy         -0.5<pred<+0.5     Buy predicted as hold - opportunity loss
    1:    Buy         >+0.5              Correct buy
    '''
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> categorize_prediction_risks:')
    logging.info ('====> ==============================================')

    #constants for accessing arrays
    BUY_INDEX = 2
    HOLD_INDEX = 1
    SELL_INDEX = 0
    ACTUAL_INDEX = 0
    PREDICTION_INDEX = 1
    ACTUAL_CHARACTERISTIC_COUNTS = 3
    ACTUAL_CHARACTERIZATION = 2
    PREDICTION_CHARACTERISTICS_COUNTS = 3
    ACTUAL_SELL = -1
    ACTUAL_HOLD = 0
    ACTUAL_BUY = 1
    PREDICTION_BUY_THRESHOLD = 0.2
    PREDICTION_SELL_THRESHOLD = 0.15
    
    actual_sample_count = true_data.shape[0]
    prediction_count = predicted_data.shape[1]

    np_counts                       = np.zeros([PREDICTION_CHARACTERISTICS_COUNTS])
    np_predictions                  = np.zeros([prediction_count, PREDICTION_CHARACTERISTICS_COUNTS])
    np_characterization             = np.zeros([prediction_count, PREDICTION_CHARACTERISTICS_COUNTS, ACTUAL_CHARACTERISTIC_COUNTS])
    np_characterization_percentage  = np.zeros([prediction_count, PREDICTION_CHARACTERISTICS_COUNTS, ACTUAL_CHARACTERISTIC_COUNTS])
    
    
    for ndx_actual in range (0, actual_sample_count) :
        if true_data[ndx_actual] == ACTUAL_SELL :
            #actual data indicated sell
            np_counts[SELL_INDEX] += 1
        elif true_data[ndx_actual] == ACTUAL_HOLD :
            #actual data indicated hold
            np_counts[HOLD_INDEX] += 1
        else :
            #actual data indicated buy
            np_counts[BUY_INDEX] += 1
            
        for ndx_predicted in range (0, prediction_count) :
            if true_data[ndx_actual] == ACTUAL_SELL :
                #actual data indicated sell
                if (predicted_data[ndx_actual, ndx_predicted] < PREDICTION_SELL_THRESHOLD) :
                    np_characterization [ndx_predicted, SELL_INDEX, SELL_INDEX] += 1
                    np_predictions      [ndx_predicted, SELL_INDEX] += 1
                    
                elif (predicted_data[ndx_actual, ndx_predicted] > PREDICTION_BUY_THRESHOLD) :
                    np_characterization [ndx_predicted, BUY_INDEX, SELL_INDEX] += 1
                    np_predictions      [ndx_predicted, BUY_INDEX] += 1
                    
                else :
                    np_characterization [ndx_predicted, HOLD_INDEX, SELL_INDEX] += 1
                    np_predictions      [ndx_predicted, HOLD_INDEX] += 1
                    
            elif true_data[ndx_actual] == ACTUAL_HOLD :
                #actual data indicated hold
                if (predicted_data[ndx_actual, ndx_predicted] < PREDICTION_SELL_THRESHOLD) :
                    np_characterization[ndx_predicted, SELL_INDEX, HOLD_INDEX] += 1
                    np_predictions     [ndx_predicted, SELL_INDEX] += 1
                    
                elif (predicted_data[ndx_actual, ndx_predicted] > PREDICTION_BUY_THRESHOLD) :
                    np_characterization[ndx_predicted, BUY_INDEX, HOLD_INDEX] += 1
                    np_predictions     [ndx_predicted, BUY_INDEX] += 1
                    
                else :
                    np_characterization[ndx_predicted, HOLD_INDEX, HOLD_INDEX] += 1
                    np_predictions     [ndx_predicted, HOLD_INDEX] += 1
                    
            else :
                #actual data indicated buy
                if (predicted_data[ndx_actual, ndx_predicted] < PREDICTION_SELL_THRESHOLD) :
                    np_characterization[ndx_predicted, SELL_INDEX, BUY_INDEX] += 1
                    np_predictions     [ndx_predicted, SELL_INDEX] += 1
                    
                elif (predicted_data[ndx_actual, ndx_predicted] > PREDICTION_BUY_THRESHOLD) :
                    np_characterization[ndx_predicted, BUY_INDEX, BUY_INDEX] += 1
                    np_predictions     [ndx_predicted, BUY_INDEX] += 1
                    
                else :
                    np_characterization[ndx_predicted, HOLD_INDEX, BUY_INDEX] += 1
                    np_predictions     [ndx_predicted, HOLD_INDEX] += 1
    
    for ndx_analysis in range (0, prediction_count) :
        for ndx_bsh_actual in range (0, ACTUAL_CHARACTERISTIC_COUNTS) :
            for ndx_bsh_predicted in range (0, PREDICTION_CHARACTERISTICS_COUNTS) :
                np_characterization_percentage[ndx_analysis, ndx_bsh_predicted, ndx_bsh_actual] = \
                    np_characterization[ndx_analysis, ndx_bsh_predicted, ndx_bsh_actual] / actual_sample_count

    logging.debug('\nAnalysis names:\t%s\npredicted data shape: %s\nactual data shape: %s', \
                  technical_analysis_names, predicted_data.shape, true_data.shape)
    logging.debug('Result characterizations:\n%s', np_characterization)
    logging.debug('Result characterizations pct:\n%s', np_characterization_percentage)

    f_out.write ('\n\nPrediction results can be categorized as follows:')   
    print       ('\nPrediction results can be categorized as follows:')   
    logging.info('\nPrediction results can be categorized as follows:')    
    f_out.write ('\nActual\ttotal:\t{:.0f}\tbuys:\t{:.0f}\tholds:\t{:.0f}\tsells:{:.0f}'.format( \
                actual_sample_count, \
                np_counts[BUY_INDEX], \
                np_counts[HOLD_INDEX], \
                np_counts[SELL_INDEX] \
                ))
    print       ('Actual\ttotal:\t%d\tbuys: %d\tholds: %d\tsells: %d' % ( \
                actual_sample_count, \
                np_counts[BUY_INDEX], \
                np_counts[HOLD_INDEX], \
                np_counts[SELL_INDEX] \
                ))
    logging.info('Actual\t\t\t\t\tbuys: %d\tholds: %d\tsells: %d' % ( \
                np_counts[BUY_INDEX], \
                np_counts[HOLD_INDEX], \
                np_counts[SELL_INDEX] \
                ))

    for ndx_analysis in range (0, prediction_count) :
        if ndx_analysis == 0 :
            f_out.write ('\nComposite analysis')   
            print       ('Composite analysis')
            logging.info('Composite analysis')
        else :
            f_out.write ('\n{:s}'.format(technical_analysis_names[ndx_analysis-1]))      
            print       ('\n%s' % technical_analysis_names[ndx_analysis-1])
            logging.info('\n%s', technical_analysis_names[ndx_analysis-1])
            
        f_out.write ('\n\tPrediction values range from\t{:f} to {:f}'.format(min(predicted_data[ndx_analysis, : ]), max(predicted_data[ndx_analysis, : ])))   
        print       ('\tPrediction values range from\t%f to %f' % (min(predicted_data[ndx_analysis, : ]), max(predicted_data[ndx_analysis, : ])))
        logging.info('\tPrediction values range from\t%f to %f' % (min(predicted_data[ndx_analysis, : ]), max(predicted_data[ndx_analysis, : ])))

        f_out.write ('\n\tPredicted\t\t\tbuys:\t\t{:.0f}\t\tholds:\t\t{:.0f}\t\tsells:\t\t{:.0f}'.format( \
                    np_predictions[ndx_analysis, BUY_INDEX], \
                    np_predictions[ndx_analysis, HOLD_INDEX], \
                    np_predictions[ndx_analysis, SELL_INDEX] \
                    ))   
        print       ('\tPredicted\t\t\tbuys:\t\t%d\t\tholds:\t\t%d\t\tsells:\t\t%d' % ( \
                    np_predictions[ndx_analysis, BUY_INDEX], \
                    np_predictions[ndx_analysis, HOLD_INDEX], \
                    np_predictions[ndx_analysis, SELL_INDEX] \
                    ))
        logging.info('\tPredicted\t\t\tbuys:\t\t%d\t\tholds:\t\t%d\t\tsells:\t\t%d' % ( \
                    np_predictions[ndx_analysis, BUY_INDEX], \
                    np_predictions[ndx_analysis, HOLD_INDEX], \
                    np_predictions[ndx_analysis, SELL_INDEX] \
                    ))

        f_out.write ('\n\tCorrect predictions:\t\tBuy\t\t{:.0f}\t{:.2f}\tHold\t\t{:.0f}\t{:.2f}\tSell\t\t{:.0f}\t{:.2f}'.format(\
                    np_characterization[ndx_analysis, BUY_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, HOLD_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_analysis, SELL_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, SELL_INDEX] \
                    ))   
        print       ('\tCorrect predictions:\t\tBuy\t\t%d\t%.2f\tHold\t\t%d\t%.2f\tSell\t\t%d\t%.2f' % ( \
                    np_characterization[ndx_analysis, BUY_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, HOLD_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_analysis, SELL_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, SELL_INDEX] \
                    ))
        logging.info('\tCorrect predictions:\t\tBuy\t\t%d\t%.2f\tHold\t\t%d\t%.2f\tSell\t%d\t\t%.2f', \
                    np_characterization[ndx_analysis, BUY_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, HOLD_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_analysis, SELL_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, SELL_INDEX] \
                    )
        
        f_out.write ('\n\tLost opportunities:\t\thold as sell\t{:.0f}\t{:.2f}\tbuy as hold\t{:.0f}\t{:.2f}\tbuy as sell\t{:.0f}\t{:.2f}'.format( \
                    np_characterization[ndx_analysis, SELL_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_analysis, HOLD_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, SELL_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, BUY_INDEX] \
                    ))   
        print       ('\tLost opportunities:\t\thold as sell\t%d\t%.2f\tbuy as hold\t%d\t%.2f\tbuy as sell\t%d\t%.2f' % ( \
                    np_characterization[ndx_analysis, SELL_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_analysis, HOLD_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, SELL_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, BUY_INDEX] \
                    ))
        logging.info('\tLost opportunities:\t\thold as sell\t%d\t%.2f\tbuy as hold\t%d\t%.2f\tbuy as sell\t%d\t%.2f', \
                    np_characterization[ndx_analysis, SELL_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, HOLD_INDEX], \
                    np_characterization[ndx_analysis, HOLD_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, SELL_INDEX, BUY_INDEX], np_characterization_percentage[ndx_analysis, SELL_INDEX, BUY_INDEX] \
                    )
        
        f_out.write ('\n\tFinancial loss if acted on:\tsell as hold\t{:.0f}\t{:.2f}\tsell as buy\t{:.0f}\t{:.2f}\thold as buy\t{:.0f}\t{:.2f}'.format( \
                    np_characterization[ndx_analysis, HOLD_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, SELL_INDEX], \
                    np_characterization[ndx_analysis, BUY_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, BUY_INDEX], \
                    np_characterization[ndx_analysis, BUY_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, HOLD_INDEX] \
                    ))   
        print       ('\tFinancial loss if acted on:\tsell as hold\t%d\t%.2f\tsell as buy\t%d\t%.2f\thold as buy\t%d\t%.2f' % ( \
                    np_characterization[ndx_analysis, HOLD_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, SELL_INDEX], \
                    np_characterization[ndx_analysis, BUY_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, SELL_INDEX], \
                    np_characterization[ndx_analysis, BUY_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, HOLD_INDEX] \
                    ))
        logging.info('\tFinancial loss if acted on:\tsell as hold\t%d\t%.2f\tsell as buy\t%d\t%.2f\thold as buy\t%d\t%.2f', \
                    np_characterization[ndx_analysis, HOLD_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, HOLD_INDEX, SELL_INDEX], \
                    np_characterization[ndx_analysis, BUY_INDEX, SELL_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, SELL_INDEX], \
                    np_characterization[ndx_analysis, BUY_INDEX, HOLD_INDEX], np_characterization_percentage[ndx_analysis, BUY_INDEX, HOLD_INDEX] \
                    )
    
    return

def bsh_results_multiple(technical_analysis_names, predicted_data, true_data, f_out) :
    logging.info ('')
    logging.info ('====> ==============================================')
    logging.info ('====> bsh_results_multiple: predicted_data shape=%s true_data shape=%s', predicted_data.shape, true_data.shape)
    logging.debug('====> \npredicted_data=\n%s\ntrue_data=\n%s', predicted_data, true_data)
    logging.info ('====> ==============================================')
        
    np_diff = np.zeros([predicted_data.shape[0], predicted_data.shape[1]])
    for ndx_data in range(0, predicted_data.shape[0]) :
        for ndx_output in range(0,predicted_data.shape[1]) :
            np_diff[ndx_data][ndx_output] = true_data[ndx_data] - predicted_data[ndx_data][ndx_output]
    '''
    On screen plot of actual and predicted data
    '''
    categorize_prediction_risks(technical_analysis_names, predicted_data, true_data, np_diff, f_out)
    '''
    Display plots of differences
    plot_bsh_results(technical_analysis_names, predicted_data, true_data, np_diff)
    '''
    
    return

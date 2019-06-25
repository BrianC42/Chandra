'''
Created on Jan 31, 2018

@author: Brian
'''
import logging
import datetime as dt
import time

from quandl_library import get_ini_data
from configuration_constants import TICKERS
from configuration_constants import ANALASIS_SAMPLE_LENGTH
from configuration_constants import FORECAST_LENGTH
from configuration_constants import ACTIVATION
from configuration_constants import COMPILATION_LOSS
from configuration_constants import LOSS_WEIGHTS
from configuration_constants import COMPILATION_METRICS
from configuration_constants import OPTIMIZER
from configuration_constants import BATCH_SIZE
from configuration_constants import EPOCHS
from configuration_constants import ANALYSIS
from configuration_constants import ML_APPROACH
from configuration_constants import RESULT_DRIVERS
from configuration_constants import FEATURE_TYPE
from configuration_constants import FORECAST_FEATURE
from configuration_constants import LOGGING_LEVEL
from configuration_constants import LOGGING_FORMAT

from lstm import build_model
from lstm import predict_sequences_multiple
from lstm import prepare_ts_lstm
from lstm import train_lstm
from lstm import evaluate_model
from lstm import save_model
from lstm import pickle_dump_training_data
from lstm import pickle_load_training_data

from buy_sell_hold import bsh_results_multiple
from percentage_change import pct_change_multiple

if __name__ == '__main__':
    print ("Good morning Dr. Chandra. I am ready for my first lesson.\n")
    
    start = time.time()
    now = dt.datetime.now()
    
    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['log']

    logging.basicConfig(filename=log_file, level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    print ("Logging to", log_file)
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Keras LSTM model for stock market prediction')
    
    output_file = lstm_config_data['result']
    output_file = output_file + '{:s} {:s} {:4d} {:0>2d} {:2d} {:0>2d} {:0>2d} {:0>2d}'.format(ANALYSIS, ML_APPROACH, \
                                                                                       now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
    f_out = open(output_file, 'w')
    
    '''
    str_symbols = ''
    for ndx_symbol in range (0, len(TICKERS)) :
        str_symbols += TICKERS[ndx_symbol]
        str_symbols += ' '
    '''
    str_drivers = ''
    for ndx_driver in range (0, len(RESULT_DRIVERS)) :
        str_drivers += RESULT_DRIVERS[ndx_driver]
        str_drivers += ' '
    
    f_out.write('\nUsing\n\t' + str_drivers + '\nsample data points')
    f_out.write('\nUsing a time series of {:.0f} periods and a time window the next {:.0f} time periods'.format(ANALASIS_SAMPLE_LENGTH, FORECAST_LENGTH))
    f_out.write('\nKeras parameters: Activation = ' + ACTIVATION + ', Batch Size = {:.0f}, Epochs = {:.0f}'.format(BATCH_SIZE, EPOCHS))
    f_out.write('\nKeras compilation parameters: Loss = ' + COMPILATION_LOSS + ' + Optimizer = ' + OPTIMIZER)
    f_out.write('\nOutput loss weights: {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(LOSS_WEIGHTS[0], LOSS_WEIGHTS[1], LOSS_WEIGHTS[2], \
                                                                                          LOSS_WEIGHTS[3], LOSS_WEIGHTS[4], LOSS_WEIGHTS[5]))
    f_out.write('\n' + ANALYSIS)

    ''' .......... Step 1 - Load and prepare data .........................
    ======================================================================= '''
    step1 = time.time()
    print ('\nStep 1 - Load and prepare the data for analysis')
    '''
    lst_analyses, x_train, y_train, x_test, y_test = prepare_ts_lstm(TICKERS, RESULT_DRIVERS, FORECAST_FEATURE, FEATURE_TYPE, \
                                                                     ANALASIS_SAMPLE_LENGTH, FORECAST_LENGTH, \
                                                                     source="local", analysis=ANALYSIS)
    pickle_dump_training_data (lst_analyses, x_train, y_train, x_test, y_test)
    '''
    lst_analyses, x_train, y_train, x_test, y_test = pickle_load_training_data()
    
    ''' ................... Step 2 - Build Model ............................
    ========================================================================= '''
    step2 = time.time()
    print ('\nStep 2 - Build Model')
    model = build_model(lst_analyses, x_train, f_out)
    
    ''' ...................... Step 3 - Train the model .....................
    ========================================================================= '''
    step3 = time.time()
    print( "\nStep 3 - Train the model")
    train_lstm(model, x_train, y_train, f_out)
    
    ''' .................... Step 4 - Evaluate the model! ...............
    ===================================================================== '''
    step4 = time.time()
    print ("\nStep 4 - Evaluate the model!")
    evaluate_model(model, x_test, y_test, f_out)
    
    ''' .................... Step 5 - clean up, archive and visualize accuracy! ...............
    =========================================================================================== '''
    step5 = time.time()
    print ("\nStep 5 - clean up, archive and visualize accuracy!")
    predictions = predict_sequences_multiple(model, x_test, lst_analyses, f_out)
    save_model(model)  
    end = time.time()
    print ("")
    print ("\tStep 1 took %.1f secs to Load and prepare the data for analysis" % (step2 - step1)) 
    print ("\tStep 2 took %.1f secs to Build Model" % (step3 - step2)) 
    print ("\tStep 3 took %.1f secs to Train the model" % (step4 - step3)) 
    print ("\tStep 4 took %.1f secs to Evaluate the model" % (step5 - step4)) 
    print ("\tStep 5 took %.1f secs to Visualize accuracy, clean up and archive" % (end - step5))
    
    if (ANALYSIS == 'value') :
        pct_change_multiple()
    elif (ANALYSIS == 'classification') :
        bsh_results_multiple(lst_analyses, predictions, y_test, f_out)
    else :
        print ('Analysis model is not specified')
    
    f_out.close()
    print ('\nNow go and make us rich')

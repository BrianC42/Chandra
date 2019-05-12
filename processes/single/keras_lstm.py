'''
Created on Jan 31, 2018

@author: Brian
'''
import logging
import time

from quandl_library import get_ini_data
from configuration_constants import tickers
from configuration_constants import result_drivers
from configuration_constants import forecast_feature
from configuration_constants import feature_type
from configuration_constants import ANALASIS_SAMPLE_LENGTH
from configuration_constants import FORECAST_LENGTH
from configuration_constants import ACTIVATION
from configuration_constants import BATCH_SIZE
from configuration_constants import EPOCHS
from lstm import build_model
from lstm import predict_sequences_multiple
from lstm import prepare_ts_lstm
from lstm import train_lstm
from lstm import evaluate_model
from lstm import save_model
from buy_sell_hold import bsh_results_multiple

if __name__ == '__main__':
    print ("Good morning Dr. Chandra. I am ready for my first lesson.\n")
    
    start = time.time()
    
    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['log']

    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')
    print ("Logging to", log_file)
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Keras LSTM model for stock market prediction')
    
    output_file = lstm_config_data['result']
    f_out = open(output_file, 'w')
    
    str_symbols = ''
    for ndx_symbol in range (0, len(tickers)) :
        str_symbols += tickers[ndx_symbol]
        str_symbols += ' '
    str_drivers = ''
    for ndx_driver in range (0, len(result_drivers)) :
        str_drivers += result_drivers[ndx_symbol]
        str_drivers += ' '
    
    f_out.write('\nTraining a model to provide buy, sell or hold recommendations')
    f_out.write('\nUsing symbols\n\t' + str_symbols + '\nfor training and testing')
    f_out.write('\nUsing\n\t' + str_drivers + '\nsample data points')
    f_out.write('\nUsing a time series of {:.0f} periods and a time window the next {:.0f} time periods'.format(ANALASIS_SAMPLE_LENGTH, FORECAST_LENGTH))
    f_out.write('\nKeras parameters: Activation = ' + ACTIVATION + ', Batch Size = {:.0f}, Epochs = {:.0f}\n'.format(BATCH_SIZE, EPOCHS))
    
    analysis_choice='buy_sell_hold'
    f_out.write(analysis_choice)

    ''' .......... Step 1 - Load and prepare data .........................
    ======================================================================= '''
    step1 = time.time()
    print ('\nStep 1 - Load and prepare the data for analysis')
    lst_analyses, x_train, y_train, x_test, y_test = prepare_ts_lstm(tickers, result_drivers, forecast_feature, feature_type, \
                                                                     ANALASIS_SAMPLE_LENGTH, FORECAST_LENGTH, \
                                                                     source="local", analysis=analysis_choice)
    
    ''' ................... Step 2 - Build Model ............................
    ========================================================================= '''
    step2 = time.time()
    print ('\nStep 2 - Build Model')
    model = build_model(lst_analyses, x_train)
    
    ''' ...................... Step 3 - Train the model .....................
    ========================================================================= '''
    step3 = time.time()
    print( "\nStep 3 - Train the model")
    train_lstm(model, x_train, y_train)
    
    ''' .................... Step 4 - Evaluate the model! ...............
    ===================================================================== '''
    step4 = time.time()
    print ("\nStep 4 - Evaluate the model!")
    evaluate_model(model, x_test, y_test)
    
    ''' .................... Step 5 - clean up, archive and visualize accuracy! ...............
    =========================================================================================== '''
    step5 = time.time()
    print ("\nStep 5 - clean up, archive and visualize accuracy!")
    predictions = predict_sequences_multiple(model, x_test)
    save_model(model)  
    end = time.time()
    print ("")
    print ("\tStep 1 Load and prepare the data for analysis took %s" % (step2 - step1)) 
    print ("\tStep 2 Build Model took %s" % (step3 - step2)) 
    print ("\tStep 3 Train the model took %s" % (step4 - step3)) 
    print ("\tStep 4 Evaluate the model! took %s" % (step5 - step4)) 
    print ("\tStep 5 Visualize accuracy, clean up and archive! took %s" % (end - step5))
    
    if (analysis_choice == 'TBD') :
        print ('Analysis model is not yet defined')
    elif (analysis_choice == 'buy_sell_hold') :
        bsh_results_multiple(lst_analyses, predictions, y_test, f_out)
    else :
        print ('Analysis model is not specified')
    
    f_out.close()
    print ('\nNow go and make us rich')

'''
Created on Jan 31, 2018

@author: Brian
'''
import logging

from lstm import build_model
from lstm import predict_sequences_multiple
from lstm import plot_results_multiple
from lstm import prepare_ts_lstm
from lstm import train_lstm
from lstm import evaluate_model
from lstm import save_model
from quandl_library import get_ini_data

if __name__ == '__main__':
    print ("Good morning Dr. Chandra. I am ready for my first lesson.")
    
    #ticker = input("Pick a symbol ")
    ticker = "F"
    #print( "Symbol selected " + usersymbolselection)
    ANALASIS_SAMPLE_LENGTH = 50
    FORECAST_LENGTH = 10

    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['dir'] + "\\lstm_log.txt"
    logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')
    
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    
    logger.debug('debug - lowest level message to log file')
    logger.info('info')
    logger.warn('warning')
    logger.error('error')
    logger.critical('critical')
    logger.debug("Logging to %s", log_file)

    ''' ................... Step 1 Build Model ............................
    =================================================================== '''
    print ('\nStep 1 Build Model')
    model = build_model(sample_length=ANALASIS_SAMPLE_LENGTH, data_points=1)
    
    ''' .......... Step 2 - Load and prepare data .........................
    =================================================================== '''
    print ('\nStep 2 Load and prepare the data for analysis')
    X_train, y_train, X_test, y_test = prepare_ts_lstm(ticker, ANALASIS_SAMPLE_LENGTH, source="local")
    
    ''' ...................... Step 3 Train the model .....................
    =================================================================== '''
    print( "\nStep 3 Train the model")
    train_lstm(model, X_train, y_train)
    
    ''' .................... Step 4 - Evaluate the model! ...............
    =================================================================== '''
    print ("\nStep 4 - Evaluate the model!")
    evaluate_model(model, X_test, y_test)
    predictions = predict_sequences_multiple(model, X_test, ANALASIS_SAMPLE_LENGTH, FORECAST_LENGTH)
    plot_results_multiple(predictions, y_test, FORECAST_LENGTH)
    
    ''' .................... Step 5 - Clean up and archive! ...............
    =================================================================== '''  
    save_model(model)  
    print ('\nNow go and make us rich')

'''
Created on Jun 13, 2019

@author: Brian
'''
import logging
import time

import numpy as np

from quandl_library import get_ini_data

from configuration_constants import TICKERS
from configuration_constants import ANALASIS_SAMPLE_LENGTH
from configuration_constants import FORECAST_LENGTH
from configuration_constants import ANALYSIS
from configuration_constants import RESULT_DRIVERS
from configuration_constants import FEATURE_TYPE
from configuration_constants import FORECAST_FEATURE
from configuration_constants import LOGGING_LEVEL
from configuration_constants import LOGGING_FORMAT

from lstm import prepare_ts_lstm
from lstm import pickle_dump_training_data
from lstm import pickle_load_training_data

if __name__ == '__main__':
    print ("OK Hal, I think this will get you started.\n")
    
    start = time.time()
    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['log']

    logging.basicConfig(filename=log_file, level=LOGGING_LEVEL, format=LOGGING_FORMAT)
    print ("Logging to", log_file)
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    
    logger.info('Preparing data to train for stock market prediction')
    
    ''' .......... Step 1 - Load and prepare data .........................
    ======================================================================= '''
    step1 = time.time()
    print ('\nStep 1 - Load and prepare the data for analysis')
    lst_analyses, x_train, y_train, x_test, y_test = prepare_ts_lstm(TICKERS, RESULT_DRIVERS, FORECAST_FEATURE, FEATURE_TYPE, \
                                                                     ANALASIS_SAMPLE_LENGTH, FORECAST_LENGTH, \
                                                                     source="local", analysis=ANALYSIS)

    pickle_dump_training_data (lst_analyses, x_train, y_train, x_test, y_test)
    
    lst_analyses_2, x_train_2, y_train_2, x_test_2, y_test_2 = pickle_load_training_data()
    
    if (np.array_equal(lst_analyses_2, lst_analyses) == False) :
        print("Houston we have a problem with lst_analyses")

    if (np.array_equal(x_train_2, x_train) == False) :
        print("\nHouston we have a problem with x_train")
        print("x_train: length %s\tx_train_2: length %s" % (len(x_train), len(x_train_2)))
        
    if (np.array_equal(y_train_2, y_train) == False) :
        print("\nHouston we have a problem with y_train")
        
    if (np.array_equal(x_test_2, x_test) == False) :
        print("\nHouston we have a problem with x_test")
        print("x_test: length %s\tx_test_2: length %s" % (len(x_test), len(x_test_2)))
        
    if (np.array_equal(y_test_2, y_test) == False) :
        print("\nHouston we have a problem with y_test")

    logger.info('Data preparation complete')

    end = time.time()
    print('\nThat took %0.1f seconds ... Now go and train' % (end-start))

    pass
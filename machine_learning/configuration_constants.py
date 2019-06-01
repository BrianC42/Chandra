'''
Created on May 9, 2019

@author: Brian

This file contains constants used to control the:
    structure of the machine learning model
    the data elements used
    the data sources used
'''
import logging

'''
data sources to use as samples to train, evaluate and use for predictions
        for testing the following options are frequently used
            "aapl", "arnc", "ba", "c", "cat", "dd", "f", "ge", "jnj", "ko", "hpq", "dis", "ibm", "ip", "mcd", "mo", "mrk", "mro", "pg", "t", "utx", "xom"
            "hsc", "msft", "gntx", "csfl", "vrnt", "intc", "hsbc", "xlnx", "amgn", "f", "gm", "c"
            "bios", "cfg", "chl",  "ddd", "gild",  "m",  "mygn",  "nvda",  "wmt",  "xxii",  "c"
            "aapl", "arnc", "ba", "c"
            "f"
'''
TICKERS = ["aapl"]

RESULT_DRIVERS   = ["adj_low", "adj_high", "adj_open", "adj_close", "adj_volume", "BB_Lower", "BB_Upper", "SMA20",   "OBV",     "AccumulationDistribution", "momentum", "MACD_Sell", "MACD_Buy"]
FEATURE_TYPE     = ['numeric', 'numeric',  'numeric', 'numeric',    'numeric',    'numeric',   'numeric', 'numeric', 'numeric', 'numeric',                  'numeric',  'boolean',   'boolean']
ANALASIS_SAMPLE_LENGTH = 120
FORECAST_LENGTH = 30

'''
Recurrent layer parameter
    Activation choices: relu tanh softmax
    Use_bias: True False
    dropout: floating point number 0.0
'''
ACTIVATION = 'softmax'
USE_BIAS = True
DROPOUT = 0.25

'''
Sample data processing controls
'''
BATCH_SIZE = 1024
EPOCHS = 5

'''
Model training parameters
'''
VALIDATION_SPLIT = 0.05
VERBOSE = 1

'''
Which analysis approach to use
choices
    buy_sell_hold 
    pct_change
'''
ANALYSIS = 'buy_sell_hold'
FORECAST_FEATURE = [False, True, False, False, False, False, False, False, False, False, False, False, False]

'''
Output thresholds for characterization of results
'''
PREDICTION_BUY_THRESHOLD = 0.4
PREDICTION_SELL_THRESHOLD = -0.4
BUY_INDICATION_THRESHOLD = 1.2
SELL_INDICATION_THRESHOLD = 0.8

'''
Values used to identify classification
    ..._INDICATION - value
    ..._INDEX - array index for storage of value
'''
CLASSIFICATION_COUNT = 3
CLASSIFICATION_ID = 1.0
BUY_INDICATION = 1.0
BUY_INDEX = 2
HOLD_INDICATION = 0.0
HOLD_INDEX = 1
SELL_INDICATION = -1.0
SELL_INDEX = 0

'''
Logging controls
'''
LOGGING_LEVEL = logging.DEBUG
LOGGING_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'
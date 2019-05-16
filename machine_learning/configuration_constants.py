'''
Created on May 9, 2019

@author: Brian

This file contains constants used to control the:
    structure of the machine learning model
    the data elements used
    the data sources used
        for testing the following options are frequently used
            "hsc", "msft", "gntx", "csfl", "vrnt", "intc", "hsbc", "xlnx", "amgn", "f", "gm", "c"
            "aapl", "arnc", "ba", "c", "cat", "dd", "f", "ge", "jnj", "ko", "hpq", "dis", "ibm", "ip", "mcd", "mo", "mrk", "mro", "pg", "t", "utx", "xom"
            "f"
'''



'''
data sources to use as samples to train, evaluate and use for predictions
'''
TICKERS = ["f"]

RESULT_DRIVERS   = ["adj_low", "adj_high", "adj_open", "adj_close", "adj_volume", "BB_Lower", "BB_Upper", "SMA20",   "OBV",     "AccumulationDistribution", "momentum", "MACD_Sell", "MACD_Buy"]
FEATURE_TYPE     = ['numeric', 'numeric',  'numeric', 'numeric',    'numeric',    'numeric',   'numeric', 'numeric', 'numeric', 'numeric',                  'numeric',  'boolean',   'boolean']
ANALASIS_SAMPLE_LENGTH = 120
FORECAST_LENGTH = 30

'''
Activation choices
    relu
    tanh
    softmax
'''
ACTIVATION = 'softmax'

'''
Sample data processing controls
'''
BATCH_SIZE = 1024
EPOCHS = 2

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
Values used to identify characterization
'''
BUY_INDICATION = 1.0
HOLD_INDICATION = 0.0
SELL_INDICATION = -1.0

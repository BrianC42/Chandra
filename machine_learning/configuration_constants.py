'''
Created on May 9, 2019

@author: Brian
'''

#ticker = input("Pick a symbol ")
tickers = ["hsc", "msft", "gntx", "csfl", "vrnt", "intc", "hsbc", "xlnx", "amgn", "f", "gm", "c"]
tickers = ["aapl", "arnc", "ba", "c", "cat", "dd", "f", "ge", "jnj", "ko", "hpq", "dis", "ibm", "ip", "mcd", "mo", "mrk", "mro", "pg", "t", "utx", "xom"]
tickers = ["f"]

result_drivers   = ["adj_low", "adj_high", "adj_open", "adj_close", "adj_volume", "BB_Lower", "BB_Upper", "SMA20",   "OBV",     "AccumulationDistribution", "MACD_Sell", "MACD_Buy"]
forecast_feature = [False,     True,       False,      False,       False,        False,       False,     False,     False,     False,                      False,       False]
feature_type     = ['numeric', 'numeric',  'numeric', 'numeric',    'numeric',    'numeric',   'numeric', 'numeric', 'numeric', 'numeric',                  'boolean',   'boolean']
ANALASIS_SAMPLE_LENGTH = 120
FORECAST_LENGTH = 30
ACTIVATION = 'relu'
ACTIVATION = 'tanh'
ACTIVATION = 'softmax'
BATCH_SIZE = 1024
EPOCHS = 2
PREDICTION_BUY_THRESHOLD = 0.4
PREDICTION_SELL_THRESHOLD = -0.4
BUY_INDICATION_THRESHOLD = 1.2
SELL_INDICATION_THRESHOLD = 0.8
BUY_INDICATION = 1.0
HOLD_INDICATION = 0.0
SELL_INDICATION = -1.0

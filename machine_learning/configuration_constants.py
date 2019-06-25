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
Logging controls
'''
LOGGING_LEVEL = logging.DEBUG
LOGGING_FORMAT = '%(asctime)s: %(levelname)s: %(message)s'

'''
Which analysis approach to use:
    Classification:    buy_sell_hold
    Value prediction:  pct_change
data sources to use as samples to train, evaluate and use for predictions
        for testing the following options are frequently used
            "hsc", "msft", "gntx", "csfl", "vrnt", "intc", "xlnx", "amgn", "f", "gm", "c"
            "bios", "cfg", "chl",  "ddd", "gild",  "m",  "mygn",  "nvda",  "wmt",  "xxii",  "c"
            "aapl", "arnc", "ba", "c"
            "f"
            "all"
            "all", 50000
'''
TICKERS = ["limit", 750000]
RESULT_DRIVERS   = ["adj_low", "adj_high", "adj_open", "adj_close", "adj_volume", "BB_Lower", "BB_Upper", "SMA20",   "OBV",     "AccumulationDistribution", "momentum", "MACD_Sell", "MACD_Buy"]
FEATURE_TYPE     = ['numeric', 'numeric',  'numeric', 'numeric',    'numeric',    'numeric',   'numeric', 'numeric', 'numeric', 'numeric',                  'numeric',  'boolean',   'boolean']
FORECAST_FEATURE = [False, True, False, False, False, False, False, False, False, False, False, False, False]
ANALASIS_SAMPLE_LENGTH = 120
FORECAST_LENGTH = 30
BALANCE_CLASSES = False #Use the same number of samples of each class for training

'''
Keras control and configuration values
    Activation choices: relu tanh softmax sigmoid
    Use_bias: True False
    dropout: floating point number 0.0
    loss: sparse_categorical_crossentropy mse binary_crossentropy
    optimizer: adam SGD RMSprop Adagrad Adadelta Adamax Nadam  
    metrics: accuracy
'''
ANALYSIS    = 'classification'                # classification    value
ML_APPROACH = 'convolutional'                 # core recurrent convolutional
COMPILATION_LOSS = "binary_crossentropy"
'''
mean_squared_error
mean_absolute_error
mean_absolute_percentage_error
mean_squared_logarithmic_error
squared_hinge
hinge
categorical_hinge
logcosh
categorical_crossentropy
sparse_categorical_crossentropy
binary_crossentropy
kullback_leibler_divergence
poisson
cosine_proximity
'''
LOSS_WEIGHTS = [1.0, 0.25, 0.25, 0.25, 0.25, 0.25] # use to emphasize different outputs
COMPILATION_METRICS = ['accuracy']            # loss funcion or accuracy - can also be a tuple ['a', 'b']
'''
                    binary_accuracy
                    categorical_accuracy
                    sparse_categorical_accuracy
                    top_k_categorical_accuracy
                    spares_top_k_categorical_accuracy
'''
ACTIVATION = 'relu'
'''
softmax
elu
selu
softplus - 
softsign - 
relu     - Rectified Linear Unit - output = input if >0 otherwise 0
tanh     - Limit output to the range -1 <= output <= +1
sigmoid  - limit output to the range 0 <= output <= +1
hard_sigmoid
linear
'''
OPTIMIZER = 'adam'                            # adam SGD RMSprop Adagrad Adadelta Adamax Nadam
'''
SGD
RMSprop
Adagrad
Adadelta
Adam
Adamax
Nadam
TFOptimizer
'''
USE_BIAS = True
DROPOUT = 0.25
VALIDATION_SPLIT = 0.05
#Model training
BATCH_SIZE = 32
EPOCHS = 5
VERBOSE = 2                                   # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

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


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
            "limit", 50000
'''
GENERATE_DATA = False
PLOT_DATA = False   # Interactive and display controls
TICKERS = ["limit", 2000000]
#TICKERS = ["aapl", "arnc", "ba", "c"]
ANALASIS_SAMPLE_LENGTH = 30 # historical time steps to use for prediction
FORECAST_LENGTH = 30 # future time steps to forecast 
RESULT_DRIVERS   = ["adj_low", "adj_high", "adj_open", "adj_close", "adj_volume", "BB_Lower", "BB_Upper", "SMA20",   "OBV",     "AccumulationDistribution", "momentum", "MACD_Sell", "MACD_Buy"]
FEATURE_TYPE     = ['numeric', 'numeric',  'numeric', 'numeric',    'numeric',    'numeric',   'numeric', 'numeric', 'numeric', 'numeric',                  'numeric',  'boolean',   'boolean']
FORECAST_FEATURE = [False, True, False, False, False, False, False, False, False, False, False, False, False]

'''
Data analysis and interpretation controls
Output thresholds for characterization of results
'''
PREDICTION_BUY_THRESHOLD = 0.4
PREDICTION_SELL_THRESHOLD = -0.4
BUY_INDICATION_THRESHOLD = 1.2 #classification threshold for buy (future price / current price)
SELL_INDICATION_THRESHOLD = 0.8 #classification threshold for sell (future price / current price)
PREDICTION_PROBABILITY_THRESHOLD = 0.9

'''
Values used to identify classification
    ..._INDICATION - value
    ..._INDEX - array index for storage of value
'''
SELL_INDEX = 0
HOLD_INDEX = 1
BUY_INDEX = 2
CLASSIFICATION_COUNT = 3
BUY_INDICATION = 1.0
HOLD_INDICATION = 0.0
SELL_INDICATION = -1.0
CLASSIFICATION_ID = 1.0

'''
Keras control and configuration values
    Activation choices: relu tanh softmax sigmoid
    Use_bias: True False
    dropout: floating point number 0.0
    loss: sparse_categorical_crossentropy mse binary_crossentropy
    optimizer: adam SGD RMSprop Adagrad Adadelta Adamax Nadam  
    metrics: accuracy
'''
#[1.0, 1.0, 1.0, 1.0, 1.0, 1.0] - all equal
#[1.0, 0.1, 0.1, 0.1, 0.1, 0.1] - focus on combined result
#[0.1, 1.0, 0.1, 0.1, 0.1, 0.1] - focus on market activity
#[1.0, 1.0, 0.1, 0.1, 0.1, 0.1] - focus on combined result and  market activity
#[0.1, 0.1, 0.1, 0.1, 0.1, 1.0] - focus on MACD
#[0.75, 0.1, 0.1, 0.5, 0.25, 0.25]
LOSS_WEIGHTS = [1.0, 1.0, 0.01, 0.01, 1.0, 1.0] # use to emphasize different outputs
DENSE_REGULARIZATION = False
REGULARIZATION_VALUE = 0.0
DROPOUT = False
DROPOUT_RATE = 0.2
LAYER_NODES = 1500
USE_BIAS = True
VALIDATION_SPLIT = 0.05
#Model training
BATCH_SIZE = 256
EPOCHS = 3
VERBOSE = 2                                   # Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
BALANCE_CLASSES = True # Use the same number of samples of each class for training
ANALYSIS_LAYER_COUNT = 6 # number of Dense layers for each technical analysis
COMPOSITE_LAYER_COUNT = 3 # number of Dense layers for the composite analysis
ANALYSIS    = 'classification'                # classification    value
ML_APPROACH = 'recurrent'                 # core recurrent convolutional
COMPILATION_LOSS = 'binary_crossentropy'
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

COMPILATION_METRICS = ['accuracy', 'binary_crossentropy']            # loss function or accuracy - can also be a tuple ['a', 'b']
'''
accuracy
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
OPTIMIZER = 'Adam'                            # adam SGD RMSprop Adagrad Adadelta Adamax Nadam
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

'''
json tags
'''

JSON_REQUIRED = "requiredFields"
JSON_CONDITIONAL = "conditionalFields"

JSON_PROCESS_NODES = 'processNodes'

JSON_NODE_NAME = "processNodeName"
JSON_PROCESS_TYPE = 'processType'
JSON_INPUT_FLOWS = 'inputFlows'
JSON_OUTPUT_FLOW = 'outputFlow'
JSON_DATA_PREP_CTRL = "dataPrepCtrl"
JSON_KERAS_DENSE_CTRL = "KerasDenseCtrl"
JSON_MODEL_FILE = "modelFile"
JSON_OUTPUT_FILE = "outputFile"
JSON_MODEL_STRUCTURE = "modelStructure"
JSON_LOG_FILE = "logFile"

JSON_DATA_PREP_PROCESS = "dataPrep"
JSON_INPUT_DATA_PREPARATION= "dataPrepCtrl"
JSON_INPUT_DATA_FILE= "inputFile"

JSON_DATA_FLOWS = 'dataFlows'

JSON_FLOW_NAME = 'flowName'
JSON_FLOW_FROM = 'flowFrom'
JSON_FLOW_TO = 'flowTo'

JSON_PREPROCESSING = 'preprocessingLayers'
JSON_PREPROCESS_SEQUENCE = 'preprocessSequence'
JSON_PREPROCESS_DISCRETIZATION = 'discretization'
JSON_PREPROCESS_DISCRETIZATION_BINS = 'bins'
JSON_PREPROCESS_CATEGORY_ENCODING = 'categoryEncoding'

JSON_KERAS_DENSE_PROCESS = "KerasDense"
JSON_KERAS_DENSE_CTRL = "KerasDenseCtrl"
JSON_MODEL_FILE = "modelFile"
JSON_TRAINING = "training"

JSON_KERAS_DENSE_DATA = 'KerasDenseData'
JSON_BALANCED = "balanceClasses"
JSON_TIME_SEQ = 'timeSequence'
JSON_IGNORE_BLANKS = "ignoreBlanks"
JSON_FLOW_DATA_FILE = "flowDataFile"

JSON_FEATURE_FIELDS = "features"
JSON_TARGET_FIELDS = "targets"

JSON_CATEGORY_TYPE = "categoryType"
JSON_CATEGORIZATION_DETAILS = "categorization"
JSON_CATEGORY_1HOT = "categoryOneHot"
JSON_CAT_TF = "categoryTrueFalse"
JSON_CAT_THRESHOLD = "categoryThreshold"
JSON_THRESHOLD_VALUE = "categoryThresholdValue"
JSON_VALUE_RANGES = "categoryRanges"
JSON_RANGE_MINS = "categoryMinimums"
JSON_RANGE_MAXS = "categoryMaximums"
JSON_LINEAR_REGRESSION = "linearRegression"

JSON_REMOVE_OUTLIER_LIST  = "removeOutliers"
JSON_OUTLIER_FEATURE = "featureName"
JSON_OUTLIER_PCT = "outlierPct"

JSON_MODEL_INPUT_LAYER = 'inputLayer'
JSON_MODEL_OUTPUT_LAYER = 'outputLayer'
JSON_MODEL_OUTPUT_ACTIVATION = 'outputActivation'
JSON_MODEL_OUTPUT_WIDTH = 'outputWidth'
JSON_MODEL_DEPTH = 'layerCounts'
JSON_NODE_COUNT = 'layerNodes'
#JSON_OUTPUTNAME = 'outputName'
JSON_TIMESTEPS = 'timeSteps'
JSON_ELEMENTS = 'dataElements'
JSON_LOSS_WTS = 'lossWeights'
JSON_REGULARIZATION = 'denseRegularation'
JSON_REG_VALUE = 'regularationValue'
JSON_DROPOUT = 'dropout'
JSON_DROPOUT_RATE = 'dropoutRate'
JSON_BIAS = 'useBias'
JSON_VALIDATION_SPLIT = 'validationSplit'
JSON_BATCH = 'batchSize'
JSON_EPOCHS = 'epochs'
JSON_VERBOSE = 'verbose'
JSON_LOSS = 'compilationLoss'
JSON_METRICS = 'compilationMetrics'
JSON_ACTIVATION = 'activation'
JSON_OPTIMIZER = 'optimizer'
JSON_ANALYSIS = 'analysis'

'''
JSON_OUTPUTS = 'outputs'
JSON_INPUT_TYPE = 'inputType'
JSON_INTERNAL_NODES = 'internalNodes'
JSON_NODE_NAME = 'nodeName'
JSON_BALANCE = 'balanceClasses'
'''

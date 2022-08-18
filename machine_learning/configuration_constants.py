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
JSON_MODEL_FILE = "modelFile"
JSON_LOG_FILE = "logFile"
JSON_DATA_LOAD_PROCESS = "dataLoad"
JSON_DATA_PREP_PROCESS = "dataPrep"
JSON_TRAIN = "Trainer"
JSON_TENSORFLOW = "Tensorflow"
JSON_AUTOKERAS = "AutoKeras"
JSON_INPUT_DATA_PREPARATION = "dataLoadCtrl"
JSON_AUTOKERAS_PARAMETERS = "autoKeras"
JSON_AK_TASK = "akTask"
JSON_AK_IMAGE_CLASSIFIER = 'akImageClassifier'
JSON_AK_IMAGE_REGRESSOR = 'akImageRegressor'
JSON_AK_TEXT_CLASSIFIER = 'ak_TextClassifier'
JSON_AK_TEXT_REGRESSOR = 'akTextRegressor'
JSON_AK_STRUCTURED_DATA_CLASSIFIER = 'akStructuredDataClassifier'
JSON_AK_STRUCTURED_DATA_REGRESSOR = 'akStructuredDataRegressor'
JSON_AK_MULTI = 'akMulti'
JSON_AK_CUSTOM = 'akCustom'
JSON_AK_DIR = "akDirectory"
JSON_AK_MAX_TRIALS = "maxTrials"

JSON_DATA_PREPARATION_CTRL = "dataPrepCtrl"
JSON_DATA_PREP_SEQ = "preparationSequence"
JSON_DATA_PREP_FEATURES = "features"
JSON_DATA_PREP_FEATURE = "feature"
JSON_DATA_PREP_NORMALIZATION_TYPE = "type"
JSON_DATA_PREP_PASSTHRU = "passThru"
JSON_DATA_PREP_NORMALIZE = 'normalize'
JSON_DATA_PREP_ENCODING = "oneHotEncoding"

JSON_NORMALIZE_DATA = 'normalize'
JSON_DATA_PREP_NORMALIZE_DATA = 'normalize'
JSON_DATA_PREP_NORMALIZE_STANDARD = 'standard'
JSON_DATA_PREP_NORMALIZE_MINMAX = 'minmax'

JSON_DATA_PREP_CATEGORIZE = "categorize"
JSON_1HOT_ENCODING = "oneHotEncoding"
JSON_1HOT_FIELD = "field"
JSON_1HOT_CATEGORYTYPE = "categoryType"
JSON_1HOT_SERIESTREND = "seriesTrend"
JSON_1HOT_SERIES_UP_DOWN = "seriesChangeUpDown"
JSON_1HOT_CATEGORYTYPE = "categoryType"
JSON_1HOT_CATEGORIES = "categories"
JSON_1HOT_OUTPUTFIELDS = "outputFields"

JSON_INPUT_DATA_FILE= "inputFile"
JSON_DATA_FLOWS = 'dataFlows'
JSON_FLOW_NAME = 'flowName'
JSON_FLOW_FROM = 'flowFrom'
JSON_FLOW_TO = 'flowTo'
JSON_PREPROCESSING = 'preprocessingLayers'
JSON_TENSORFLOW_DATA = "TensorflowData"
JSON_PRECISION = "dataPrecision"
JSON_VISUALIZATIONS = "visualizations"
JSON_VISUALIZE_TRAINING_FIT = "trainingFit"
JSON_VISUALIZE_TARGET_SERIES = "targetSeries"
JSON_MODEL_FILE = "modelFile"
JSON_TRAINING = "training"
JSON_BALANCED = "balanceClasses"

JSON_TIME_SEQ = 'timeSequence'
JSON_SERIES_ID = "seriesStepIDField"
JSON_SERIES_DATA_TYPE = "seriesDataType"

JSON_CONV1D = "conv1d"
JSON_BATCHES = "batches"
JSON_FILTER_COUNT = "filterCount" 
JSON_FILTER_SIZE = "filterSize"

JSON_MAXPOOLING_1D = "MaxPooling1D"
JSON_GLOBAL_MAXPOOLING_1D = "GlobalMaxPool1D"
JSON_POOL_SIZE = "poolSize"

JSON_FLATTEN = "flatten"

JSON_IGNORE_BLANKS = "ignoreBlanks"
JSON_FLOW_DATA_FILE = "flowDataFile"
JSON_FEATURE_FIELDS = "features"
JSON_TARGET_FIELDS = "targets"
JSON_TIMESTEPS = 'timeSteps'
JSON_FEATURE_COUNT = 'featureCount'
JSON_MODEL_OUTPUT_ACTIVATION = 'outputActivation'
JSON_SHUFFLE_DATA = 'shuffle'
JSON_ELEMENTS = 'dataElements'
JSON_LOSS_WTS = 'lossWeights'
JSON_REGULARIZATION = 'denseRegularation'
JSON_REG_VALUE = 'regularationValue'
JSON_LAYERS = "modelLayers"
JSON_LAYER_TYPE = 'layerType'
JSON_LAYER_DENSE = 'dense'
JSON_LAYER_LSTM = 'lstm'
JSON_LAYER_REPEAT_VECTOR = 'RepeatVector'
JSON_LAYER_TIME_DISTRIBUTED = 'TimeDistributed'
JSON_LAYER_NAME = 'layerName'
JSON_LAYER_UNITS = 'layerUnits'
JSON_DROPOUT = 'dropout'
JSON_DROPOUT_RATE = 'dropoutRate'
JSON_BIAS = 'useBias'
JSON_VALIDATION_SPLIT = 'validationSplit'
JSON_TEST_SPLIT = 'testSplit'
JSON_BATCH = 'batchSize'
JSON_EPOCHS = 'epochs'
JSON_VERBOSE = 'verbose'
JSON_LOSS = 'compilationLoss'
JSON_METRICS = 'compilationMetrics'
JSON_ACTIVATION = 'activation'
JSON_RETURN_SEQUENCES = 'returnSequences'
JSON_REPEAT_COUNT = 'repeatCount'
JSON_OPTIMIZER = 'optimizer'
JSON_ANALYSIS = 'analysis'

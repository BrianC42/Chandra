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
#JSON_OUTPUT_FILE = "outputFile"
JSON_LOG_FILE = "logFile"
JSON_DATA_LOAD_PROCESS = "dataLoad"
JSON_DATA_PREP_PROCESS = "dataPrep"
JSON_TENSORFLOW = "Tensorflow"
JSON_INPUT_DATA_PREPARATION = "dataLoadCtrl"
JSON_DATA_PREPARATION_CTRL = "dataPrepCtrl"
JSON_1HOT_ENCODING = "oneHotEncoding"
JSON_1HOT_FIELD = "field"
JSON_1HOT_CATEGORYTYPE = "categoryType"
JSON_1HOT_SERIESTREND = "seriesTrend"
JSON_1HOT_CATEGORIES = "categories"
JSON_1HOT_OUTPUTFIELDS = "outputFields"
JSON_INPUT_DATA_FILE= "inputFile"
JSON_DATA_FLOWS = 'dataFlows'
JSON_FLOW_NAME = 'flowName'
JSON_FLOW_FROM = 'flowFrom'
JSON_FLOW_TO = 'flowTo'
JSON_PREPROCESSING = 'preprocessingLayers'
JSON_LAYERS = "modelLayers"
JSON_TENSORFLOW = "Tensorflow"
JSON_TENSORFLOW_DATA = "TensorflowData"
JSON_PRECISION = "dataPrecision"
JSON_MODEL_FILE = "modelFile"
JSON_TRAINING = "training"
JSON_BALANCED = "balanceClasses"
JSON_TIME_SEQ = 'timeSequence'
JSON_IGNORE_BLANKS = "ignoreBlanks"
JSON_FLOW_DATA_FILE = "flowDataFile"
JSON_FEATURE_FIELDS = "features"
JSON_TARGET_FIELDS = "targets"
JSON_TIMESTEPS = 'timeSteps'
JSON_FEATURE_COUNT = 'featureCount'
JSON_MODEL_OUTPUT_ACTIVATION = 'outputActivation'

'''
JSON_MODEL_INPUT_LAYER = 'inputLayer'
JSON_MODEL_OUTPUT_LAYER = 'outputLayer'
JSON_MODEL_OUTPUT_WIDTH = 'outputWidth'
'''
JSON_NORMALIZE_DATA = 'normalize'
JSON_SHUFFLE_DATA = 'shuffle'
JSON_ELEMENTS = 'dataElements'
JSON_LOSS_WTS = 'lossWeights'
JSON_REGULARIZATION = 'denseRegularation'
JSON_REG_VALUE = 'regularationValue'
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

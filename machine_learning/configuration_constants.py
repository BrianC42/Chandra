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
JSON_OUTPUT_FILE = "outputFile"
JSON_LOG_FILE = "logFile"
JSON_DATA_PREP_PROCESS = "dataPrep"
JSON_INPUT_DATA_PREPARATION= "dataPrepCtrl"
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
JSON_KERAS_CONV1D = "KerasConv1D"
JSON_KERAS_CONV1D_CONTROL = "KerasConv1DCtrl"
JSON_KERAS_CONV1D_FILTERS = "filters"
JSON_KERAS_CONV1D_KERNEL_SIZE = "kernelSize"
JSON_KERAS_CONV1D_STRIDES = "strides"
JSON_KERAS_CONV1D_PADDING = "padding"
JSON_KERAS_CONV1D_DATA_FORMAT = "dataFormat"
JSON_KERAS_CONV1D_DILATION_RATE = "dilationRate"
JSON_KERAS_CONV1D_GROUPS = "groups"
JSON_KERAS_CONV1D_ACTIVATION = "activation"
JSON_KERAS_CONV1D_USE_BIAS = "useBias"
JSON_KERAS_CONV1D_KERNEL_INITIALIZER = "kernelInitializer"
JSON_KERAS_CONV1D_BIAS_INITIALIZER = "biasInitializer"
JSON_KERAS_CONV1D_KERNEL_REGULARIZER = "kernelRegularizer"
JSON_KERAS_CONV1D_BIAS_REGULARIZER = "biasRegularizer"
JSON_KERAS_CONV1D_ACTIVITY_REGULARIZER = "activityRegularizer"
JSON_KERAS_CONV1D_KERNEL_CONSTRAINT = "kernelConstraint"
JSON_KERAS_CONV1D_BIAS_CONSTRAINT = "biasConstraint"
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
JSON_CATEGORY_TYPE = "categoryType"
JSON_CATEGORIZATION_DETAILS = "categorization"
JSON_CATEGORY_1HOT = "categoryOneHot"
JSON_CAT_TF = "categoryTrueFalse"
JSON_CAT_THRESHOLD = "categoryThreshold"
JSON_THRESHOLD_VALUE = "categoryThresholdValue"
JSON_VALUE_RANGES = "categoryRanges"
JSON_RANGE_MINS = "categoryMinimums"
JSON_RANGE_MAXS = "categoryMaximums"
JSON_LINEAR_REGRESSION = "Regression"
JSON_REMOVE_OUTLIER_LIST  = "removeOutliers"
JSON_OUTLIER_FEATURE = "featureName"
JSON_OUTLIER_PCT = "outlierPct"
JSON_MODEL_INPUT_LAYER = 'inputLayer'
MODEL_TYPE = 'modelType'
INPUT_LAYERTYPE_DENSE = 'dense'
INPUT_LAYERTYPE_RNN = 'rnn'
INPUT_LAYERTYPE_CNN = 'cnn'
JSON_MODEL_OUTPUT_LAYER = 'outputLayer'
JSON_MODEL_OUTPUT_ACTIVATION = 'outputActivation'
JSON_MODEL_OUTPUT_WIDTH = 'outputWidth'
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

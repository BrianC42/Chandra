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

''' configuration file template
{
"dataFlows" : [
    {
    "flowName" : "xxx",
    "requiredFields" : {
        "flowFrom" : "xxx", "flowTo" : "xxx", "flowDataFile" : "xxx"
        },
    "conditionalFields" : {
        "TensorflowData" : {
            "features" : ["xxx", "xxx"], "targets" : ["xxx"],
            "timeSequence" : true, "seriesStepIDField" : ["xxx"], "seriesDataType" : "xxx", "type" : "xxx"
            }
        }
    }
],
"processNodes" : [
    {
    "processNodeName" : "xxx",
    "requiredFields" : {"processType" : "dataLoad", "inputFlows" : [], "outputFlow" : "xxx"},
    "conditionalFields" : {"dataLoadCtrl" : {"inputFile" : ["xxx", "xxx"]}}
    },
    {
    "processNodeName" : "xxx",
    "requiredFields" : {"processType" : "dataPrep", "inputFlows" : ["xxx"], "outputFlow" : "xxx"},
    "conditionalFields" : {
        "dataPrepCtrl" : {
            "preparationSequence" : ["xxx", "xxx"],
            "noPreparation" : {"features" : []},
            "normalize" : {"type" : "xxx", "features" : [{"feature" : "xxx"}, {"feature" : "xxx"}]},
            "oneHotEncoding" : {"features" : [{"feature" : "xxx", "categoryType" : "xxx"}]}
        }
    }
    },
    {
    "processNodeName" : "xxx",
    "requiredFields" : {"processType" : "Tensorflow", "inputFlows" : ["xxx"], "outputFlow" : ""},
    "conditionalFields" : {
        "categorizationRegression" : "xxx",
        "dataPrecision" : "float32",
        "visualizations" : ["xxx", "xxx", "xxx", "xxx"],
        "training iterations" : [
            {
            "iteration parameters" : {
                "modelFileDir" : "xxx", "iteration ID" : "xxx", "forecastInterval" : xxx,
                "data arrangement" : {"combineSampleCount" : xxx},
                "modelLayers" : [
                    {"layerName" : "xxx", "layerType" : "xxx", "featureCount": xxx, "timeSteps": xxx, "layerUnits" : xxx, "returnSequences" : xxx, "outputActivation" : "xxx"},
                    {"layerName" : "xxx", "layerType" : "xxx", "timeSteps" : xxx, "layerUnits" : xxx, "returnSequences" : xxx, "outputActivation" : "xxx"},
                    {"layerName" : "xxx", "layerType" : "xxx", "dropoutRate" : xxx},
                    {"layerType" : "xxx", "layerName" : "xxx", "layerUnits" : xxx, "outputActivation" : "xxx"},
                    {"layerName" : "xxx", "layerType" : "xxx", "dropoutRate" : xxx},
                    {"layerType" : "xxx", "layerName" : "xxx", "layerUnits": xxx, "outputActivation" : "xxx"}
                    ],
                "training" : {
                    "lossWeights" : [xxx], "denseRegularation" : xxx, "regularationValue" : xxx, "useBias" : xxx, 
                    "batchSize" : xxx, "epochs" : xxx, "validationSplit" : xxx, "testSplit" : xxx,
                    "shuffle" : xxx, "balanceClasses" : xxx,    "verbose" : xxx,
                    "optimizer" : {"name" : "xxx", "learning_rate" : xxx}, "compilationLoss" : "xxx", "compilationMetrics" : ["xxx"]
                },
                "tensorboard" : {"log file dir" : "xxx"}
                }
            }
        ]
    }
    }
],
}
'''

''' =============================================================================================== '''
''' ========================     configuration labels under development     ======================= '''
''' =============================================================================================== '''
'''                                                                                                 '''
JSON_PROCESSING_SEQUENCE = "processNodeSequence"
JSON_DEPRECATED  = "data arrangement"
JSON_COMBINE_SAMPLE_COUNT  = "combineSampleCount"

'''                                                                                                 '''
''' =============================================================================================== '''
''' =============================================================================================== '''

''' =============================================================================================== '''
''' ======================== Configuration file labels and supported values ======================= '''
''' =============================================================================================== '''

'''                                                           configuration file structure segments '''
JSON_PROCESS_NODES = 'processNodes'
JSON_DATA_FLOWS = 'dataFlows'

'''                                                                segments used in multiple places '''
JSON_REQUIRED = "requiredFields"
JSON_CONDITIONAL = "conditionalFields"
JSON_OUTPUT_FILE = "outputFile"
#JSON_CONDITIONAL_FIELDS = "conditionalFields"

''' ----------------------------------------------------------------------------------------------- '''
'''                                                                    information flow definitions '''
JSON_FLOW_NAME = 'flowName'

'''                                                                                 Required fields '''
JSON_FLOW_FROM = 'flowFrom'
JSON_FLOW_TO = 'flowTo'
JSON_FLOW_DATA_FILE = "flowDataFile"

'''                                                                              Conditional fields '''
JSON_TENSORFLOW_DATA = "TensorflowData"
JSON_BALANCED = "balanceClasses"
JSON_IGNORE_BLANKS = "ignoreBlanks"
JSON_FEATURE_FIELDS = "features"
JSON_TARGET_FIELDS = "targets"
JSON_TIME_SEQ = 'timeSequence'
JSON_SERIES_ID = "seriesStepIDField"
JSON_SERIES_DATA_TYPE = "seriesDataType"
''' -------------------------------------------------------------- end information flow definitions '''
''' ----------------------------------------------------------------------------------------------- '''

''' ----------------------------------------------------------------------------------------------- '''
'''                                                                                processing nodes '''
''' ----------------------------------------------------------------------------------------------- '''
JSON_NODES = "processNodes"
JSON_NODE_NAME = "processNodeName"
'''                                                              controls used in multiple sections '''
JSON_DATA_PREP_FEATURES = "features"
JSON_DATA_PREP_FEATURE = "feature"

'''                                                                                 Required fields '''
JSON_INPUT_FLOWS = 'inputFlows'
JSON_OUTPUT_FLOW = 'outputFlow'
JSON_PROCESS_TYPE = 'processType'
'''  Supported values - processType'''
JSON_DATA_LOAD_PROCESS = "dataLoad"
JSON_DATA_PREP_PROCESS = "dataPrep"
JSON_TENSORFLOW = "Tensorflow"
JSON_AUTOKERAS = "AutoKeras"
JSON_EXECUTE_MODEL = "executeModel"
JSON_STOP = "stop"
'''  end of Supported values '''

'''                                           ================ executeModel node conditional fields '''
JSON_EXECUTE_MODEL_CONTROL = "executeModelCtrl"
JSON_TRAINED_MODEL_FILE = "trainedModelFile"
JSON_SYNCHRONIZATION_FEATURE = "synchronizationFeature"
JSON_OUTPUT_FEATURE_LABELS = "outputLabels"
JSON_EXECUTE_CONTROL = "executeCtrl"
JSON_EXECUTION_CONTROL = "executionControl"

'''                                           ==================== dataLoad node conditional fields '''
JSON_INPUT_DATA_PREPARATION = "dataLoadCtrl"
JSON_INPUT_DATA_FILE= "inputFile"

'''                                           ==================== dataPrep node conditional fields '''
JSON_DATA_PREP_CTRL = "dataPrepCtrl"
JSON_DATA_PREPARATION_CTRL = "dataPrepCtrl"
JSON_DATA_PREP_SEQ = "preparationSequence"
JSON_DATA_PREP_NORMALIZE = 'normalize'

'''                                                           data normalization control parameters '''
JSON_NORMALIZE_DATA = 'normalize'
JSON_DATA_PREP_NORMALIZATION_TYPE = "type"
'''  Supported values - type'''
JSON_DATA_PREP_NORMALIZE_STANDARD = 'standard'
JSON_DATA_PREP_NORMALIZE_MINMAX = 'minmax'
JSON_DATA_PREP_NORMALIZE_RELATIVE_TIME_SERIES = 'relative_time_series'
'''  end of Supported values '''

'''                                                     1 hot features and labels/targets  controls '''
JSON_DATA_PREP_ENCODING = "oneHotEncoding"
JSON_1HOT_CATEGORYTYPE = "categoryType"

'''                                            ================= Tensorflow node conditional fields '''
JSON_ML_GOAL = "categorizationRegression"
'''  Supported values - categorizationRegression'''
JSON_ML_GOAL_CATEGORIZATION = "categorization"
JSON_ML_GOAL_REGRESSION = "regression"
'''  end of Supported values '''

JSON_PRECISION = "dataPrecision"

JSON_VISUALIZATIONS = "visualizations"
'''  Supported values - visualizations'''
JSON_VISUALIZE_TRAINING_FIT = "trainingFit"
JSON_VISUALIZE_TARGET_SERIES = "targetSeries"
'''  end of Supported values '''

'''                                                                      training iteration control '''
JSON_ITERATION_PARAMETERS = "iteration parameters"
JSON_TRAINING_DESCRIPTION = "training iteration description"
JSON_TRAINING = "training"
JSON_TRAINING_ITERATIONS = "training iterations"
JSON_ITERATION_ID = "iteration ID"
JSON_ML_REGRESSION_FORECAST_INTERVAL = "forecastInterval"
JSON_MODEL_FILE_DIR = "modelFileDir"
JSON_TENSORBOARD = "tensorboard"
JSON_LOG_DIR = "log file dir"

'''                  data combination used to present serial data to dense and convolutional models '''
JSON_ML_GOAL_COMBINE_SAMPLE_COUNT = "combineSampleCount"

'''                                                                         model layer definitions '''
JSON_LAYERS = "modelLayers"
JSON_LAYER_NAME = 'layerName'
JSON_VALIDATION_SPLIT = 'validationSplit'
JSON_TEST_SPLIT = 'testSplit'
JSON_LAYER_UNITS = 'layerUnits'
JSON_TIMESTEPS = 'timeSteps'
JSON_FEATURE_COUNT = 'featureCount'
JSON_RETURN_SEQUENCES = 'returnSequences'
JSON_DROPOUT = 'dropout'
JSON_DROPOUT_RATE = 'dropoutRate'
JSON_LAYER_TYPE = 'layerType'
'''  Supported values - layerType'''
JSON_LAYER_DENSE = 'dense'
JSON_LAYER_LSTM = 'lstm'
JSON_CONV1D = "conv1d"
'''  end of Supported values '''
JSON_MODEL_OUTPUT_ACTIVATION = 'outputActivation'
'''  Supported values - outputActivation'''
'''  end of Supported values '''
'''                                                                     training control parameters '''
JSON_LOSS = 'compilationLoss'
JSON_METRICS = 'compilationMetrics'
JSON_LOSS_WTS = 'lossWeights'
JSON_SHUFFLE_DATA = 'shuffle'
JSON_BATCH = 'batchSize'
JSON_EPOCHS = 'epochs'
JSON_VERBOSE = 'verbose'
JSON_OPTIMIZER = 'optimizer'
'''  Supported values - optimizer'''
'''  end of Supported values '''
''' ----------------------------------------------------------------------------------------------- '''

''' Supported values to be placed '''
JSON_1HOT_CATEGORIES = "categories"
JSON_1HOT_LABEL = "label"
JSON_1HOT_OUTPUTFIELDS = "outputFields"
JSON_1HOT_SERIES_UP_DOWN = "seriesChangeUpDown"
JSON_LAYER_REPEAT_VECTOR = 'RepeatVector'
JSON_LAYER_TIME_DISTRIBUTED = 'TimeDistributed'
JSON_FILTER_COUNT = "filterCount" 
JSON_FILTER_SIZE = "filterSize"
JSON_MAXPOOLING_1D = "MaxPooling1D"
JSON_GLOBAL_MAXPOOLING_1D = "GlobalMaxPool1D"
JSON_FLATTEN = "flatten"
JSON_REPEAT_COUNT = 'repeatCount'
''' --------------------------------------------------------------- end processing node definitions '''
''' ----------------------------------------------------------------------------------------------- '''

'''                                                                    Auto Keras specific controls '''
JSON_TRAIN = "Trainer"
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
JSON_MODEL_FILE = "modelFile"
TRAINING_AUTO_KERAS = "TBD"
''' -------------------------------------------------------------------- end Auto Keras definitions '''
''' =============================================================================================== '''
''' ================ Configuration file labels and supported values - end ========================= '''
''' =============================================================================================== '''

''' Suspected unused definitions
JSON_LOG_FILE = "logFile"
JSON_TRAINING = "training"
JSON_REGULARIZATION = 'denseRegularation'
JSON_REG_VALUE = 'regularationValue'
JSON_BIAS = 'useBias'
JSON_DATA_PREP_PASSTHRU = "passThru"
JSON_DATA_PREP_NORMALIZE_DATA = 'normalize'
JSON_DATA_PREP_CATEGORIZE = "categorize"
JSON_1HOT_ENCODING = "oneHotEncoding"
JSON_1HOT_FIELD = "field"
JSON_1HOT_SERIESTREND = "seriesTrend"
JSON_PREPROCESSING = 'preprocessingLayers'
JSON_BATCHES = "batches"
JSON_POOL_SIZE = "poolSize"
JSON_ELEMENTS = 'dataElements'
JSON_ACTIVATION = 'activation'
JSON_ANALYSIS = 'analysis'
'''

'''
Created on March 23, 2020

@author: Brian

To develop, test, train and evaluate models and their combinations
- Build, train and evaluate prediction models
- uses json to control the process
- 3 phases of processing
    Phase A - use raw data to train a model designed to recognize input data associated with a specified categorization
        each model can use unique sets of input (variable) data elements
        ALL models MUST use the same categorization data  

    Phase B - Correlate phase 'A' results.
        use the models created in phase 'A' to generate test and training data for phase C
        symbol/date/<raw data> -> <models> -> symbol/date/<model categorization>

    Phase C - weighted correlation. Use the t&t data from phase 'B' to combine the outputs of the models developed in phase 'A'
        into a model combining their outputs
        model categorizations -> WC Model -> weighted correlated categorization
            
For future use of the models
    - create a combined model consisting of the models trained in phases 'A' and 'C'
    - input required data into the combined model and output the result

'''
import os
import sys
import logging
import datetime as dt
import time
import pandas as pd
#import networkx as nx

''' suppress tensorflow warning messages - 2 mechanisms '''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
#os.environ['AUTOGRAPH_VERBOSITY'] = 1

from TrainingDataAndResults import Data2Results as d2r
#from TrainingDataAndResults import TRAINING_TENSORFLOW
#from TrainingDataAndResults import TRAINING_AUTO_KERAS

from configuration import get_ini_data
from configuration import read_config_json
from configuration import read_processing_network_json
from configuration_graph import build_configuration_graph

from executeProcessingNodes import executeProcessingNodes

from load_prepare_data import collect_and_select_data
from load_prepare_data import prepareTrainingData
from load_prepare_data import loadTrainingData
from load_prepare_data import arrangeDataForTraining
from assemble_model import buildModel
from train_model import trainModel
from evaluate_visualize import evaluate_and_visualize

#from configuration_constants import JSON_MODEL_FILE

if __name__ == '__main__':
    print ("Good morning Dr. Chandra. I am ready for my next lesson.\n\tI'm currently running Tensorflow version %s\n" % tf.__version__)
    ''' requires v1 compatibility mode
    gpu_options = tf.GPUOptions(per_process_gpu_fraction=0.3)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    '''
    
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    # Get external initialization details
    localdirs = get_ini_data("LOCALDIRS")
    gitdir = localdirs['git']
    aiwork = localdirs['aiwork']

    config_data = get_ini_data("CHANDRA")
    json_config = read_config_json(gitdir + config_data['config'])

    try:    
        log_file = aiwork + json_config['logFile']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = aiwork + json_config['outputFile']
        output_file = output_file + ' {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
        f_out = open(output_file, 'w')    
        
        # global parameters
        #logging.debug("Global parameters")
    
    except Exception:
        exc_txt = "\nAn exception occurred - log file details are missing from json configuration"
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + " " + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)
        
    ''' Set python path for executing stand alone scripts '''
    pPath = gitdir + "\\chandra\\processes\\single"
    pPath += ";"
    pPath += gitdir + "\\chandra\\processes\\multiprocess"
    pPath += ";"
    pPath += gitdir + "\\chandra\\processes\\child"
    pPath += ";"
    pPath += gitdir + "\\chandra\\utility"
    pPath += ";"
    pPath += gitdir + "\\chandra\\technical_analysis"
    pPath += ";"
    pPath += gitdir + "\\chandra\\td_ameritrade"
    pPath += ";"
    pPath += gitdir + "\\chandra\\machine_learning"
    pPath += ";"
    pPath += gitdir + "\\chandra\\unit_test"
    os.environ["PYTHONPATH"] = pPath

    print ("Logging to", log_file)
    logger = logging.getLogger('chandra_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Keras model for stock market analysis and prediction')

    #Set print parameters for Pandas dataframes 
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 20)
    
    processingNetwork = gitdir + json_config['processNet']
    processing_json = read_processing_network_json(processingNetwork)
    
    d2r = d2r()
    build_configuration_graph(d2r, processing_json)
    executeProcessingNodes(d2r)
            
    '''     clean up and prepare to exit     '''
    f_out.close()
    print ('\nNow go and make us rich')

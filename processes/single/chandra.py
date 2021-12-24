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
import logging
import datetime as dt
import time
import pandas as pd
import networkx as nx

''' suppress tensorflow warning messages - 2 mechanisms '''
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
#os.environ['AUTOGRAPH_VERBOSITY'] = 1

from TrainingDataAndResults import Data2Results as d2r

from configuration import get_ini_data
from configuration import read_config_json
from configuration import read_processing_network_json
from configuration_graph import build_configuration_graph

from load_prepare_data import collect_and_select_data
from load_prepare_data import loadTrainingData
from load_prepare_data import prepareTrainingData
from assemble_model import buildModel
from train_model import trainModel
from evaluate_visualize import evaluate_and_visualize

from configuration_constants import JSON_MODEL_FILE

if __name__ == '__main__':
    print ("Good morning Dr. Chandra. I am ready for my next lesson.\n\tI'm currently running Tensorflow version %s\n" % tf.__version__)
    
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
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    # Set python path for executing stand alone scripts
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
    #nx.draw(nx_graph, arrows=True, with_labels=True, font_weight='bold')
    #nx.draw_circular(nx_graph, arrows=True, with_labels=True, font_weight='bold')
    #plt.show()
    
    ''' .......... Step 1 - Load and prepare data .........................
    ======================================================================= '''
    step1 = time.time()
    print ('\nStep 1 - Acquire, assemble and prepare the data for analysis')
    collect_and_select_data(d2r)

    ''' ................... Step 2 - Build Model ............................
    ========================================================================= '''
    step2 = time.time()
    print ('\nStep 2 - Build and train the Model')
    #node_i, k_model, x_features, y_targets, x_test, y_test, fitting = build_and_train_model(nx_graph)
    buildModel(d2r)
    loadTrainingData(d2r)
    prepareTrainingData(d2r)
    trainModel(d2r)

    ''' .................... Step 3 - Evaluate the model! ...............
    ===================================================================== '''
    step3 = time.time()
    print ("\nStep 3 - Evaluate the model and visualize accuracy!")
    #evaluate_and_visualize(nx_graph, node_i, k_model, x_features, y_targets, x_test, y_test, fitting)
    evaluate_and_visualize(d2r)
    
    ''' .................... Step 4 - clean up, archive and visualize accuracy! ...............
    =========================================================================================== '''
    step4 = time.time()
    print ("\nStep 4 - clean up, archive")
    nx_model_file = nx.get_node_attributes(d2r.graph, JSON_MODEL_FILE)[d2r.mlNode]
    d2r.model.save(nx_model_file)

    end = time.time()
    print ("")
    print ("\tStep 1 took %.1f secs to Load and prepare the data for analysis" % (step2 - step1)) 
    print ("\tStep 2 took %.1f secs to Build and train the Model" % (step3 - step2)) 
    print ("\tStep 3 took %.1f secs to Evaluate the model" % (step4 - step3)) 
    print ("\tStep 5 took %.1f secs to Visualize accuracy, clean up and archive" % (end - step4))    

    '''
    clean up and prepare to exit
    '''
    f_out.close()
    print ('\nNow go and make us rich')

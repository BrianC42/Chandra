'''
Created on Mar 25, 2020

@author: Brian
'''
import logging

import pandas as pd
import networkx as nx

'''
from keras import regularizers
from keras.models import Model
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D

'''
from configuration_constants import JSON_NODE_NAME
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_TIMESTEPS
from configuration_constants import JSON_OUTPUTNAME
from configuration_constants import JSON_BATCH
from configuration_constants import JSON_NODE_COUNT
from configuration_constants import JSON_REGULARIZATION
from configuration_constants import JSON_REG_VALUE
from configuration_constants import JSON_DROPOUT
from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_BIAS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_BALANCE
from configuration_constants import JSON_ANALYSIS
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_ACTIVATION
from configuration_constants import JSON_OPTIMIZER

def create_core_layer(js_config, nx_graph):
    logging.debug("\t\tCreating layer: %s" % js_config[JSON_NODE_NAME])
    
    return

def create_LSTM_layer(js_config, nx_graph):
    logging.debug("\t\tCreating layer: %s" % js_config[JSON_NODE_NAME])
    logging.debug("\t\tTime steps: %s" % js_config[JSON_TIMESTEPS])
    
    nx_graph.add_node(js_config[JSON_NODE_NAME], timesteps=30)
    return

def create_RNN_layer(js_config, nx_graph):
    logging.debug("\t\tCreating layer: %s" % js_config[JSON_NODE_NAME])
        
    nx_graph.add_node(js_config[JSON_NODE_NAME])
    return

def create_input_model(js_input_definition, nx_graph):
    logging.debug("\tCreate a %s layer named %s with output %s" % (js_input_definition[JSON_PROCESS_TYPE], \
                                                                   js_input_definition[JSON_PROCESS_TYPE], \
                                                                   js_input_definition[JSON_OUTPUTNAME]))

    try:
        # create an input layer
        if (js_input_definition[JSON_PROCESS_TYPE] == "LSTM"):
            create_LSTM_layer(js_input_definition, nx_graph)
        elif (js_input_definition[JSON_PROCESS_TYPE] == "dense"):
            create_core_layer(js_input_definition, nx_graph)
        elif (js_input_definition[JSON_PROCESS_TYPE] == "convolutional"):
            create_RNN_layer(js_input_definition, nx_graph)
        else:
            err_msg = "*** An error occurred analyzing the json configuration file - an unsupported layer type was specified ***"
            logging.debug(err_msg)
            print("\n" + err_msg)
            
    except Exception:
        err_msg = "*** An exception occurred analyzing the json configuration file for an input layer ***"
        logging.debug(err_msg)
        print("\n" + err_msg)

    return

def build_model(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> build_model: building the machine learning model')
    logging.info('====> ================================================')

    # error handling
    try:
        # inputs                
        logging.debug("Building ML model")
        
    except Exception:
        err_txt = "*** An exception occurred analyzing the json configuration file ***"
        logging.debug(err_txt)
        print("\n" + err_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model: done')
    logging.info('<---- ----------------------------------------------')    
    return
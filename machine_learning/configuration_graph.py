'''
Created on Apr 13, 2020

@author: Brian
'''
import logging

import json
import pandas as pd
import networkx as nx

from configuration_constants import JSON_PROCESS_NODES

from configuration_constants import JSON_PROCESS_NAME
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_DATA_FLOWS
from configuration_constants import JSON_FLOW_NAME
from configuration_constants import JSON_FLOW_FROM
from configuration_constants import JSON_FLOW_TO
from configuration_constants import JSON_INPUT_TYPE
from configuration_constants import JSON_OUTPUT_FLOW

from configuration_constants import JSON_OUTPUTS
from configuration_constants import JSON_INTERNAL_NODES
from configuration_constants import JSON_TIMESTEPS
from configuration_constants import JSON_TIME_SEQ
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
from configuration_constants import JSON_LOSS_WTS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_ACTIVATION
from configuration_constants import JSON_OPTIMIZER

def add_csv_meta_data(nx_graph, nx_process_name, js_config):
    logging.debug("readcsv process: %s" % nx_process_name)
    if 'inputFlows' in js_config :
        nx_graph.nodes[nx_process_name]['inputFlows'] = js_config[JSON_INPUT_FLOWS]
    else:
        raise KeyError

    return

def add_calc_bollinger_band_meta_data(nx_graph, nx_process_name, js_config):
    print("calcBollingerBands process")
    return

def add_calc_on_balance_volume_band_meta_data(nx_graph, nx_process_name, js_config):
    print("calcOnBalanceVolume process")
    return

def add_calc_MACD_meta_data(nx_graph, nx_process_name, js_config):
    print("calcMACD process")
    return

def add_dense_meta_data(nx_graph, nx_process_name, js_config):
    logging.debug("Adding meta data to dense model process: %s" % nx_process_name)
    
    nx_lossWeights = ""
    if 'lossWeights' in js_config :
        nx_lossWeights = js_config[JSON_LOSS_WTS]
    nx_graph.nodes[nx_process_name]['lossWeights'] = nx_lossWeights

    nx_denseRegularation = ""
    if 'denseRegularation' in js_config :
        nx_denseRegularation = js_config[JSON_REGULARIZATION]
    nx_graph.nodes[nx_process_name]['denseRegularation'] = nx_denseRegularation
        
    nx_regularationValue = ""
    if 'regularationValue' in js_config :
        nx_regularationValue = js_config[JSON_REG_VALUE]
    nx_graph.nodes[nx_process_name]['regularationValue'] = nx_regularationValue
        
    nx_dropout = ""
    if 'dropout' in js_config :
        nx_dropout = js_config[JSON_DROPOUT]
    nx_graph.nodes[nx_process_name]['dropout'] = nx_dropout
        
    nx_dropoutRate = ""
    if 'dropoutRate' in js_config :
        nx_dropoutRate = js_config[JSON_DROPOUT_RATE]
    nx_graph.nodes[nx_process_name]['dropoutRate'] = nx_dropoutRate
        
    nx_useBias = ""
    if 'useBias' in js_config :
        nx_useBias = js_config[JSON_BIAS]
    nx_graph.nodes[nx_process_name]['useBias'] = nx_useBias
        
    nx_balanceClasses = ""
    if 'balanceClasses' in js_config :
        nx_balanceClasses = js_config[JSON_BALANCE]
    nx_graph.nodes[nx_process_name]['balanceClasses'] = nx_balanceClasses
        
    nx_analysis = ""
    if 'analysis' in js_config :
        nx_analysis = js_config[JSON_ANALYSIS]
    nx_graph.nodes[nx_process_name]['analysis'] = nx_analysis
        
    return
    
def add_LSTM_meta_data(nx_graph, nx_process_name, js_config):
    logging.debug("Adding meta data to LSTM model process: %s" % nx_process_name)
    
    nx_timeSteps = 30
    if 'timeSteps' in js_config :
        nx_timeSteps = js_config[JSON_TIMESTEPS]
    nx_graph.nodes[nx_process_name]['timeSteps'] = nx_timeSteps

    nx_batchSize = 32
    if 'batchSize' in js_config :
        nx_batchSize = js_config[JSON_BATCH]
    nx_graph.nodes[nx_process_name]['batchSize'] = nx_batchSize

    nx_validationSplit = ""
    if 'validationSplit' in js_config :
        nx_validationSplit = js_config[JSON_VALIDATION_SPLIT]
    nx_graph.nodes[nx_process_name]['validationSplit'] = nx_validationSplit
        
    nx_epochs = 3
    if 'epochs' in js_config :
        nx_epochs = js_config[JSON_EPOCHS]
    nx_graph.nodes[nx_process_name]['epochs'] = nx_epochs
        
    nx_layerNodes = 100
    if 'layerNodes' in js_config :
        nx_layerNodes = js_config[JSON_NODE_COUNT]
    nx_graph.nodes[nx_process_name]['layerNodes'] = nx_layerNodes

    nx_verbose = ""
    if 'verbose' in js_config :
        nx_verbose = js_config[JSON_VERBOSE]
    nx_graph.nodes[nx_process_name]['verbose'] = nx_verbose
        
    nx_compilationLoss = "binary_crossentropy"
    if 'compilationLoss' in js_config :
        nx_compilationLoss = js_config[JSON_LOSS]
    nx_graph.nodes[nx_process_name]['compilationLoss'] = nx_compilationLoss
        
    nx_compilationMetrics = "accuracy"
    if 'compilationMetrics' in js_config :
        nx_compilationMetrics = js_config[JSON_METRICS]
    nx_graph.nodes[nx_process_name]['compilationMetrics'] = nx_compilationMetrics
        
    nx_activation = "relu"
    if 'activation' in js_config :
        nx_activation = js_config[JSON_ACTIVATION]
    nx_graph.nodes[nx_process_name]['activation'] = nx_activation
        
    nx_optimizer = "Adam"
    if 'optimizer' in js_config :
        nx_optimizer = js_config[JSON_OPTIMIZER]
    nx_graph.nodes[nx_process_name]['optimizer'] = nx_optimizer

    return

def add_meta_data_edge (js_config, nx_graph):
    # mandatory elements
    nx_flowName = js_config[JSON_FLOW_NAME]
    nx_flowFrom =  js_config[JSON_FLOW_FROM]
    nx_flowTo =  js_config[JSON_FLOW_TO]
    nx_attribute1 =  js_config[JSON_TIME_SEQ]
    
    print("Adding edge: %s from %s to %s" % (nx_flowName, nx_flowFrom, nx_flowTo))    
    nx_graph.add_edge(nx_flowFrom, nx_flowTo, flowName=nx_flowName, dummyAttr = nx_attribute1
                      )
    return

def add_meta_process_node (js_config, nx_graph) :
    # mandatory elements
    nx_process_name = js_config[JSON_PROCESS_NAME]
    nx_inputType = js_config[JSON_INPUT_TYPE]
    nx_inputs = js_config[JSON_INPUT_FLOWS]
    nx_outputs = js_config[JSON_OUTPUT_FLOW]

    print("Adding node: %s, inputs %s, outputs %s" % (nx_process_name, nx_inputs, nx_outputs))
    nx_graph.add_node(nx_process_name)
    nx_graph.nodes[nx_process_name]['inputFlows'] = nx_inputs
    nx_graph.nodes[nx_process_name]['outputFlow'] = nx_outputs

    '''    
    nx_processType = js_config[JSON_PROCESS_TYPE]
    nx_graph.nodes[nx_process_name]['processType'] = nx_processType
    
    if nx_processType == "readcsv" :
        add_csv_meta_data(nx_graph, nx_process_name, js_config)
    elif nx_processType == "calcMACD" :
        add_calc_MACD_meta_data(nx_graph, nx_process_name, js_config)
    elif nx_processType == "calcBollingerBands" :
        add_calc_bollinger_band_meta_data(nx_graph, nx_process_name, js_config)
    elif nx_processType == "calcOnBalanceVolume" :
        add_calc_on_balance_volume_band_meta_data(nx_graph, nx_process_name, js_config)
    elif nx_processType == "KerasLSTM" :
        add_LSTM_meta_data(nx_graph, nx_process_name, js_config)
    elif nx_processType == "KerasDense" :
        add_dense_meta_data(nx_graph, nx_process_name, js_config)
    else :
        raise NameError('Invalid process node type')
    '''
    
    return

def build_configuration_graph(json_config, nx_graph):
    # error handling
    try:
        # processing nodes                
        logging.debug("Adding processing nodes")
        for i, process_i in enumerate(json_config[JSON_PROCESS_NODES]):
            add_meta_process_node(process_i, nx_graph)
        
        # processing data flows                
        logging.debug("Adding data flows")
        for i, flow_i in enumerate(json_config[JSON_DATA_FLOWS]):
            add_meta_data_edge(flow_i, nx_graph)
        
    except Exception:
        err_txt = "*** An exception occurred analyzing the json configuration file ***"
        logging.debug(err_txt)
        print("\n" + err_txt)
        
    return
'''
Created on Apr 13, 2020

@author: Brian
'''
import logging

import json
import pandas as pd
import networkx as nx

from configuration_constants import JSON_PROCESS_NODES

from configuration_constants import JSON_REQUIRED
from configuration_constants import JSON_CONDITIONAL

from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_KERAS_DENSE_PROCESS

from configuration_constants import JSON_NODE_NAME
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_DATA_FLOWS
from configuration_constants import JSON_FLOW_NAME
from configuration_constants import JSON_FLOW_FROM
from configuration_constants import JSON_FLOW_TO
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_OUTPUT_FILE
from configuration_constants import JSON_LOG_FILE

from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_INPUT_DATA_PREPARATION

from configuration_constants import JSON_KERAS_DENSE_CTRL
from configuration_constants import JSON_KERAS_DENSE_DATA
from configuration_constants import JSON_MODEL_FILE

from configuration_constants import JSON_BALANCED
from configuration_constants import JSON_TIME_SEQ
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_DATA_FIELDS
from configuration_constants import JSON_TARGET_FIELD

from configuration_constants import JSON_NODE_COUNT

from configuration_constants import JSON_TIMESTEPS
from configuration_constants import JSON_BATCH
from configuration_constants import JSON_REGULARIZATION
from configuration_constants import JSON_REG_VALUE
from configuration_constants import JSON_DROPOUT
from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_BIAS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_LOSS_WTS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_ACTIVATION
from configuration_constants import JSON_OPTIMIZER
from configuration_constants import JSON_ANALYSIS

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
        nx_balanceClasses = js_config[JSON_BALANCED]
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
    nx_flowName = js_config[JSON_FLOW_NAME]

    # mandatory elements
    js_required = js_config[JSON_REQUIRED]
    nx_flowFrom =  js_required[JSON_FLOW_FROM]
    nx_flowTo =  js_required[JSON_FLOW_TO]
    #nx_graph.add_edge(nx_flowFrom, nx_flowTo, flowName=nx_flowName)
    nx_graph.add_edge(nx_flowFrom, nx_flowTo, key=nx_flowName)
    nx_edge_key = (nx_flowFrom, nx_flowTo, nx_flowName)

    # conditional elements
    js_conditional = js_config[JSON_CONDITIONAL]
    if JSON_KERAS_DENSE_DATA in js_conditional:
        js_keras_dense_data = js_conditional[JSON_KERAS_DENSE_DATA]
        nx_balanced = js_keras_dense_data[JSON_BALANCED]
        nx_time_seq = js_keras_dense_data[JSON_TIME_SEQ]
        nx_ignore_blanks = js_keras_dense_data[JSON_IGNORE_BLANKS]
        nx_flow_data_file = js_keras_dense_data[JSON_FLOW_DATA_FILE]
        nx_data_fields = js_keras_dense_data[JSON_DATA_FIELDS]
        nx_target_field = js_keras_dense_data[JSON_TARGET_FIELD]

        for edge_i in nx_graph.edges():
            if edge_i == (nx_flowFrom, nx_flowTo):
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_balanced}, JSON_BALANCED)
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_time_seq}, JSON_TIME_SEQ)
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_ignore_blanks}, JSON_IGNORE_BLANKS)
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_flow_data_file}, JSON_FLOW_DATA_FILE)
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_data_fields}, JSON_DATA_FIELDS)
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_target_field}, JSON_TARGET_FIELD)

    print("Connected %s to %s by data flow %s" % (nx_flowFrom, nx_flowTo, nx_flowName))
    return

def add_meta_process_node (js_config, nx_graph) :
    nx_process_name = ""
    if JSON_NODE_NAME in js_config:
        nx_process_name = js_config[JSON_NODE_NAME]
    else:
        raise NameError('Missing process node name')
        
    nx_graph.add_node(nx_process_name)

    # mandatory elements
    js_required = js_config[JSON_REQUIRED]
    nx_processType = js_required[JSON_PROCESS_TYPE]
    nx_inputs = js_required[JSON_INPUT_FLOWS]
    nx_output = js_required[JSON_OUTPUT_FLOW]
    nx_outputFile = js_required[JSON_OUTPUT_FILE]
    nx_log = js_required[JSON_LOG_FILE]
    
    for node_i in nx_graph.nodes():
        if node_i == nx_process_name:
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_processType}, JSON_PROCESS_TYPE)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_inputs}, JSON_INPUT_FLOWS)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_output}, JSON_OUTPUT_FLOW)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_outputFile}, JSON_OUTPUT_FILE)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_log}, JSON_LOG_FILE)
    
    # conditional elements
    js_conditional = js_config[JSON_CONDITIONAL]
    if js_required[JSON_PROCESS_TYPE] == JSON_DATA_PREP_PROCESS:
        js_data_prep_ctrl = js_conditional[JSON_INPUT_DATA_PREPARATION]
        nx_attr_inputFile = js_data_prep_ctrl[JSON_INPUT_DATA_FILE]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_attr_inputFile}, JSON_INPUT_DATA_FILE)
    elif js_required[JSON_PROCESS_TYPE] == JSON_KERAS_DENSE_PROCESS:
        js_keras_dense = js_conditional[JSON_KERAS_DENSE_CTRL]
        nx_model_file = js_keras_dense[JSON_MODEL_FILE]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_model_file}, JSON_MODEL_FILE)
    else :
        raise NameError('Invalid process node type')
        
    nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
    print("Added %s node %s" % (nx_read_attr[nx_process_name], nx_process_name))
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
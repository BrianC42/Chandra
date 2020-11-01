'''
Created on Apr 13, 2020

@author: Brian
'''
import sys
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

from configuration_constants import JSON_CATEGORIZATION_DETAILS
from configuration_constants import JSON_CATEGORY_TYPE
from configuration_constants import JSON_CATEGORY_FIELD
from configuration_constants import JSON_CATEGORY_1HOT
from configuration_constants import JSON_CAT_TF
from configuration_constants import JSON_CAT_THRESHOLD
from configuration_constants import JSON_THRESHOLD_VALUE
from configuration_constants import JSON_VALUE_RANGES
from configuration_constants import JSON_RANGE_MINS
from configuration_constants import JSON_RANGE_MAXS

from configuration_constants import JSON_MODEL_STRUCTURE

from configuration_constants import JSON_MODEL_INPUT_LAYER
from configuration_constants import JSON_MODEL_OUTPUT_LAYER
from configuration_constants import JSON_MODEL_OUTPUT_ACTIVATION
from configuration_constants import JSON_MODEL_DEPTH
from configuration_constants import JSON_NODE_COUNT
from configuration_constants import JSON_TRAINING

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

def add_2d_data_flow_details(js_keras_dense_data, nx_graph, nx_edge_key):
    nx_balanced = js_keras_dense_data[JSON_BALANCED]
    nx_time_seq = js_keras_dense_data[JSON_TIME_SEQ]
    nx_ignore_blanks = js_keras_dense_data[JSON_IGNORE_BLANKS]
    nx_flow_data_file = js_keras_dense_data[JSON_FLOW_DATA_FILE]
    nx_data_fields = js_keras_dense_data[JSON_DATA_FIELDS]

    js_category_details = js_keras_dense_data[JSON_CATEGORIZATION_DETAILS]
    nx_category_field = js_category_details[JSON_CATEGORY_FIELD]
    nx_category1Hot = js_category_details[JSON_CATEGORY_1HOT]
    
    for edge_i in nx_graph.edges():
        if edge_i == (nx_edge_key[0], nx_edge_key[1]):
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_balanced}, JSON_BALANCED)
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_time_seq}, JSON_TIME_SEQ)
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_ignore_blanks}, JSON_IGNORE_BLANKS)
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_flow_data_file}, JSON_FLOW_DATA_FILE)
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_data_fields}, JSON_DATA_FIELDS)
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_category_field}, JSON_CATEGORY_FIELD)
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_category1Hot}, JSON_CATEGORY_1HOT)
            
            nx_category_type = js_category_details[JSON_CATEGORY_TYPE]
            nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_category_type}, JSON_CATEGORY_TYPE)
            if nx_category_type == JSON_CAT_TF:
                pass
            elif nx_category_type == JSON_CAT_THRESHOLD:
                nx_threshold = js_category_details[JSON_THRESHOLD_VALUE]
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_threshold}, JSON_THRESHOLD_VALUE)
            elif nx_category_type == JSON_VALUE_RANGES:
                nx_rangeMins = js_category_details[JSON_RANGE_MINS]
                nx_rangeMaxs = js_category_details[JSON_RANGE_MAXS]
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_rangeMins}, JSON_RANGE_MINS)
                nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_rangeMaxs}, JSON_RANGE_MAXS)
            else:
                raise NameError('Invalid category type')

    return 

def add_3d_data_flow_details(js_config, nx_graph, nx_edge_key):
    return

def add_data_load_details(js_config, nx_graph, nx_process_name):
    nx_inputFile = js_config[JSON_INPUT_DATA_FILE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_inputFile}, JSON_INPUT_DATA_FILE)
    return

def add_dense_meta_data(js_dense_params, nx_graph, nx_process_name):
    logging.debug("Adding meta data to dense model process: %s" % nx_process_name)
    nx_model_file = js_dense_params[JSON_MODEL_FILE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_model_file}, JSON_MODEL_FILE)

    if JSON_MODEL_STRUCTURE in js_dense_params:
        js_model_structure = js_dense_params[JSON_MODEL_STRUCTURE]

        nx_input_layer = js_model_structure[JSON_MODEL_INPUT_LAYER]
        nx_model_depth = js_model_structure[JSON_MODEL_DEPTH]
        nx_node_count = js_model_structure[JSON_NODE_COUNT]
        nx_output_layer = js_model_structure[JSON_MODEL_OUTPUT_LAYER]
        nx_output_activation = js_model_structure[JSON_MODEL_OUTPUT_ACTIVATION]

        nx.set_node_attributes(nx_graph, {nx_process_name:nx_input_layer}, JSON_MODEL_INPUT_LAYER)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_model_depth}, JSON_MODEL_DEPTH)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_node_count}, JSON_NODE_COUNT)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_output_layer}, JSON_MODEL_OUTPUT_LAYER)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_output_activation}, JSON_MODEL_OUTPUT_ACTIVATION)
    
    if JSON_TRAINING in js_dense_params:
        js_training = js_dense_params[JSON_TRAINING]

        nx_loss_weights = js_training[JSON_LOSS_WTS]
        nx_regularization = js_training[JSON_REGULARIZATION]
        nx_reg_value = js_training[JSON_REG_VALUE]
        nx_dropout = js_training[JSON_DROPOUT]
        nx_dropout_rate = js_training[JSON_DROPOUT_RATE]
        nx_bias = js_training[JSON_BIAS]
        nx_validation_split = js_training[JSON_VALIDATION_SPLIT]
        nx_batch = js_training[JSON_BATCH]
        nx_epochs = js_training[JSON_EPOCHS]
        nx_verbose = js_training[JSON_VERBOSE]
        nx_balanced = js_training[JSON_BALANCED]
        nx_analysis = js_training[JSON_ANALYSIS]
        nx_loss = js_training[JSON_LOSS]
        nx_metrics = js_training[JSON_METRICS]
        nx_activation = js_training[JSON_ACTIVATION]
        nx_optimizer = js_training[JSON_OPTIMIZER]

        nx.set_node_attributes(nx_graph, {nx_process_name:nx_loss_weights}, JSON_LOSS_WTS)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_regularization}, JSON_REGULARIZATION)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_reg_value}, JSON_REG_VALUE)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_dropout}, JSON_DROPOUT)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_dropout_rate}, JSON_DROPOUT_RATE)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_bias}, JSON_BIAS)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_validation_split}, JSON_VALIDATION_SPLIT)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_batch}, JSON_BATCH)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_epochs}, JSON_EPOCHS)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_verbose}, JSON_VERBOSE)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_balanced}, JSON_BALANCED)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_analysis}, JSON_ANALYSIS)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_loss}, JSON_LOSS)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_metrics}, JSON_METRICS)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_activation}, JSON_ACTIVATION)
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_optimizer}, JSON_OPTIMIZER)

    return
    
def add_RNN_meta_data(js_config, nx_graph, nx_process_name):
    logging.debug("Adding meta data to RNN model process: %s" % nx_process_name)
    

    return

def add_CNN_meta_data(js_config, nx_graph, nx_process_name):
    logging.debug("Adding meta data to CNN model process: %s" % nx_process_name)
    

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
        add_2d_data_flow_details(js_keras_dense_data, nx_graph, nx_edge_key)

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
        add_data_load_details(js_data_prep_ctrl, nx_graph, nx_process_name)
    elif js_required[JSON_PROCESS_TYPE] == JSON_KERAS_DENSE_PROCESS:
        js_keras_dense = js_conditional[JSON_KERAS_DENSE_CTRL]
        add_dense_meta_data(js_keras_dense, nx_graph, nx_process_name)
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
        sys.exit("\n" + err_txt)
        
    return
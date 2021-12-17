'''
Created on Apr 13, 2020

@author: Brian
'''
import sys
import logging

import networkx as nx

from configuration_constants import JSON_PROCESS_NODES
from configuration_constants import JSON_REQUIRED
from configuration_constants import JSON_CONDITIONAL
from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_TENSORFLOW_DATA
from configuration_constants import JSON_PRECISION
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
from configuration_constants import JSON_REMOVE_OUTLIER_LIST
from configuration_constants import JSON_MODEL_FILE
from configuration_constants import JSON_LAYERS
from configuration_constants import JSON_KERAS_CONV1D_FILTERS
from configuration_constants import JSON_KERAS_CONV1D_KERNEL_SIZE
from configuration_constants import JSON_KERAS_CONV1D_STRIDES
from configuration_constants import JSON_KERAS_CONV1D_PADDING
from configuration_constants import JSON_KERAS_CONV1D_DATA_FORMAT
from configuration_constants import JSON_KERAS_CONV1D_DILATION_RATE
from configuration_constants import JSON_KERAS_CONV1D_GROUPS
from configuration_constants import JSON_KERAS_CONV1D_ACTIVATION
from configuration_constants import JSON_KERAS_CONV1D_USE_BIAS
from configuration_constants import JSON_KERAS_CONV1D_KERNEL_INITIALIZER
from configuration_constants import JSON_KERAS_CONV1D_BIAS_INITIALIZER
from configuration_constants import JSON_KERAS_CONV1D_KERNEL_REGULARIZER
from configuration_constants import JSON_KERAS_CONV1D_BIAS_REGULARIZER
from configuration_constants import JSON_KERAS_CONV1D_ACTIVITY_REGULARIZER
from configuration_constants import JSON_KERAS_CONV1D_KERNEL_CONSTRAINT
from configuration_constants import JSON_KERAS_CONV1D_BIAS_CONSTRAINT
from configuration_constants import JSON_BALANCED
from configuration_constants import JSON_TIME_SEQ
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS
from configuration_constants import JSON_CATEGORY_1HOT
from configuration_constants import JSON_MODEL_INPUT_LAYER
from configuration_constants import JSON_MODEL_OUTPUT_LAYER
from configuration_constants import JSON_MODEL_OUTPUT_ACTIVATION
from configuration_constants import JSON_TRAINING
from configuration_constants import JSON_NORMALIZE_DATA
from configuration_constants import JSON_SHUFFLE_DATA
from configuration_constants import JSON_BATCH
from configuration_constants import JSON_REGULARIZATION
from configuration_constants import JSON_REG_VALUE
from configuration_constants import JSON_BIAS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_TEST_SPLIT
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_LOSS_WTS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_OPTIMIZER

def add_data_flow_details(js_flow_conditional, nx_graph, nx_edge_key):

    for edge_i in nx_graph.edges():
        if edge_i == (nx_edge_key[0], nx_edge_key[1]):
            if JSON_TENSORFLOW_DATA in js_flow_conditional:
                js_tensorflow_data = js_flow_conditional[JSON_TENSORFLOW_DATA]
                
                if JSON_BALANCED in js_tensorflow_data:
                    nx_balanced = js_tensorflow_data[JSON_BALANCED]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_balanced}, JSON_BALANCED)
        
                if JSON_TIME_SEQ in js_tensorflow_data:
                    nx_time_seq = js_tensorflow_data[JSON_TIME_SEQ]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_time_seq}, JSON_TIME_SEQ)

                if JSON_IGNORE_BLANKS in js_tensorflow_data:
                    nx_ignore_blanks = js_tensorflow_data[JSON_IGNORE_BLANKS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_ignore_blanks}, JSON_IGNORE_BLANKS)

                if JSON_FEATURE_FIELDS in js_tensorflow_data:
                    nx_feature_fields = js_tensorflow_data[JSON_FEATURE_FIELDS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_feature_fields}, JSON_FEATURE_FIELDS)

                if JSON_TARGET_FIELDS in js_tensorflow_data:
                    nx_target_fields = js_tensorflow_data[JSON_TARGET_FIELDS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_target_fields}, JSON_TARGET_FIELDS)

                if JSON_CATEGORY_1HOT in js_tensorflow_data:
                    nx_category1Hot = js_tensorflow_data[JSON_CATEGORY_1HOT]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_category1Hot}, JSON_CATEGORY_1HOT)

    return 

def add_3d_data_flow_details(js_config, nx_graph, nx_edge_key):
    return

def add_data_load_details(js_config, nx_graph, nx_process_name):
    nx_inputFile = js_config[JSON_INPUT_DATA_FILE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_inputFile}, JSON_INPUT_DATA_FILE)
    return

def add_dense_meta_data(js_dense_params, nx_graph, nx_process_name):
    logging.debug("Adding meta data to dense model process: %s" % nx_process_name)

    return
    
def add_Conv1D_meta_data(js_keras_Conv1D, nx_graph, nx_process_name):
    
    nx_input_layer = js_keras_Conv1D[JSON_MODEL_INPUT_LAYER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_input_layer}, JSON_MODEL_INPUT_LAYER)
    
    nx_output_layer = js_keras_Conv1D[JSON_MODEL_OUTPUT_LAYER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_output_layer}, JSON_MODEL_OUTPUT_LAYER)
    
    nx_output_activation = js_keras_Conv1D[JSON_MODEL_OUTPUT_ACTIVATION]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_output_activation}, JSON_MODEL_OUTPUT_ACTIVATION)

    nx_conv1DFilters = js_keras_Conv1D[JSON_KERAS_CONV1D_FILTERS]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1DFilters}, JSON_KERAS_CONV1D_FILTERS)

    nx_kernel_size = js_keras_Conv1D[JSON_KERAS_CONV1D_KERNEL_SIZE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_kernel_size}, JSON_KERAS_CONV1D_KERNEL_SIZE)

    nx_conv1D_strides = js_keras_Conv1D[JSON_KERAS_CONV1D_STRIDES]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_strides}, JSON_KERAS_CONV1D_STRIDES)

    nx_conv1D_padding = js_keras_Conv1D[JSON_KERAS_CONV1D_PADDING]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_padding}, JSON_KERAS_CONV1D_PADDING)

    nx_conv1D_data_format = js_keras_Conv1D[JSON_KERAS_CONV1D_DATA_FORMAT]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_data_format}, JSON_KERAS_CONV1D_DATA_FORMAT)

    nx_conv1D_dilation_rate = js_keras_Conv1D[JSON_KERAS_CONV1D_DILATION_RATE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_dilation_rate}, JSON_KERAS_CONV1D_DILATION_RATE)

    nx_conv1D_groups = js_keras_Conv1D[JSON_KERAS_CONV1D_GROUPS]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_groups}, JSON_KERAS_CONV1D_GROUPS)

    nx_conv1D_activation = js_keras_Conv1D[JSON_KERAS_CONV1D_ACTIVATION]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_activation}, JSON_KERAS_CONV1D_ACTIVATION)

    nx_conv1D_use_bias = js_keras_Conv1D[JSON_KERAS_CONV1D_USE_BIAS]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_use_bias}, JSON_KERAS_CONV1D_USE_BIAS)

    nx_conv1D_kernel_initializer = js_keras_Conv1D[JSON_KERAS_CONV1D_KERNEL_INITIALIZER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_kernel_initializer}, JSON_KERAS_CONV1D_KERNEL_INITIALIZER)

    nx_conv1D_bias_initializer = js_keras_Conv1D[JSON_KERAS_CONV1D_BIAS_INITIALIZER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_bias_initializer}, JSON_KERAS_CONV1D_BIAS_INITIALIZER)

    nx_conv1D_kernel_regularizer = js_keras_Conv1D[JSON_KERAS_CONV1D_KERNEL_REGULARIZER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_kernel_regularizer}, JSON_KERAS_CONV1D_KERNEL_REGULARIZER)

    nx_conv1D_bias_regularizer = js_keras_Conv1D[JSON_KERAS_CONV1D_BIAS_REGULARIZER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_bias_regularizer}, JSON_KERAS_CONV1D_BIAS_REGULARIZER)

    nx_conv1D_activity_regularizer = js_keras_Conv1D[JSON_KERAS_CONV1D_ACTIVITY_REGULARIZER]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_activity_regularizer}, JSON_KERAS_CONV1D_ACTIVITY_REGULARIZER)

    nx_conv1D_kernel_constraint = js_keras_Conv1D[JSON_KERAS_CONV1D_KERNEL_CONSTRAINT]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_kernel_constraint}, JSON_KERAS_CONV1D_KERNEL_CONSTRAINT)

    nx_conv1D_bias_constraint = js_keras_Conv1D[JSON_KERAS_CONV1D_BIAS_CONSTRAINT]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_conv1D_bias_constraint}, JSON_KERAS_CONV1D_BIAS_CONSTRAINT)

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
    nx_graph.add_edge(nx_flowFrom, nx_flowTo, key=nx_flowName)

    nx_edge_key = (nx_flowFrom, nx_flowTo, nx_flowName)
    nx_flow_data_file = js_required[JSON_FLOW_DATA_FILE]
    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_flow_data_file}, JSON_FLOW_DATA_FILE)

    if JSON_CONDITIONAL in js_config:
        js_conditional = js_config[JSON_CONDITIONAL]
        add_data_flow_details(js_conditional, nx_graph, nx_edge_key)

    print("Connected %s to %s by data flow %s" % (nx_flowFrom, nx_flowTo, nx_flowName))
    return

def add_training_controls(js_training, nx_graph, nx_process_name):
    nx_loss_weights = js_training[JSON_LOSS_WTS]
    nx_regularization = js_training[JSON_REGULARIZATION]
    nx_reg_value = js_training[JSON_REG_VALUE]
    nx_bias = js_training[JSON_BIAS]
    nx_validation_split = js_training[JSON_VALIDATION_SPLIT]
    nx_test_split = js_training[JSON_TEST_SPLIT]
    nx_batch = js_training[JSON_BATCH]
    nx_epochs = js_training[JSON_EPOCHS]
    nx_verbose = js_training[JSON_VERBOSE]
    nx_balanced = js_training[JSON_BALANCED]
    nx_loss = js_training[JSON_LOSS]
    nx_metrics = js_training[JSON_METRICS]
    nx_optimizer = js_training[JSON_OPTIMIZER]
    
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_loss_weights}, JSON_LOSS_WTS)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_regularization}, JSON_REGULARIZATION)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_reg_value}, JSON_REG_VALUE)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_bias}, JSON_BIAS)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_validation_split}, JSON_VALIDATION_SPLIT)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_test_split}, JSON_TEST_SPLIT)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_batch}, JSON_BATCH)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_epochs}, JSON_EPOCHS)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_verbose}, JSON_VERBOSE)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_balanced}, JSON_BALANCED)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_loss}, JSON_LOSS)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_metrics}, JSON_METRICS)
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_optimizer}, JSON_OPTIMIZER)

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
    
    if nx_processType == JSON_TENSORFLOW:
        nx_model_file = js_conditional[JSON_MODEL_FILE]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_model_file}, JSON_MODEL_FILE)
        nx_dataPrecision = js_conditional[JSON_PRECISION]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_dataPrecision}, JSON_PRECISION)
        
    if JSON_LAYERS in js_conditional:
        nx_layers = js_conditional[JSON_LAYERS]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_layers}, JSON_LAYERS)
    
    if JSON_REMOVE_OUTLIER_LIST in js_conditional:
        js_remove_outliers = js_conditional[JSON_REMOVE_OUTLIER_LIST]
        nx.set_node_attributes(nx_graph, {nx_process_name:js_remove_outliers}, JSON_REMOVE_OUTLIER_LIST)
            
    if js_required[JSON_PROCESS_TYPE] == JSON_DATA_PREP_PROCESS:
        js_data_prep_ctrl = js_conditional[JSON_INPUT_DATA_PREPARATION]
        add_data_load_details(js_data_prep_ctrl, nx_graph, nx_process_name)
        
    if JSON_TRAINING in js_conditional:
        add_training_controls(js_conditional[JSON_TRAINING], nx_graph, nx_process_name)
    
    nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
    print("Added %s node %s" % (nx_read_attr[nx_process_name], nx_process_name))
    return

def build_configuration_graph(d2r, json_config):
    # error handling
    try:
        d2r.graph = nx.MultiDiGraph()
    
        # processing nodes                
        logging.debug("Adding processing nodes")
        for i, process_i in enumerate(json_config[JSON_PROCESS_NODES]):
            add_meta_process_node(process_i, d2r.graph)
        
        # processing data flows                
        logging.debug("Adding data flows")
        for i, flow_i in enumerate(json_config[JSON_DATA_FLOWS]):
            add_meta_data_edge(flow_i, d2r.graph)
        
        d2r.plotGraph()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = "\n*** An exception occurred analyzing the json configuration file ***" + "\n\t" + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)

    return
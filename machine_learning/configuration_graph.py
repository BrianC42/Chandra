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
from configuration_constants import JSON_DATA_LOAD_PROCESS
from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_TRAIN
from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_AUTOKERAS
from configuration_constants import JSON_TENSORFLOW_DATA
from configuration_constants import JSON_AUTOKERAS_PARAMETERS
from configuration_constants import JSON_PRECISION
from configuration_constants import JSON_VISUALIZATIONS
from configuration_constants import JSON_NODE_NAME
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_DATA_FLOWS
from configuration_constants import JSON_FLOW_NAME
from configuration_constants import JSON_FLOW_FROM
from configuration_constants import JSON_FLOW_TO
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_LOG_FILE
from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_INPUT_DATA_PREPARATION
from configuration_constants import JSON_DATA_PREPARATION_CTRL
from configuration_constants import JSON_MODEL_FILE
from configuration_constants import JSON_LAYERS
from configuration_constants import JSON_BALANCED
from configuration_constants import JSON_TIME_SEQ
from configuration_constants import JSON_SERIES_ID
from configuration_constants import JSON_SERIES_DATA_TYPE
from configuration_constants import JSON_1HOT_CATEGORYTYPE
from configuration_constants import JSON_1HOT_CATEGORIES
from configuration_constants import JSON_1HOT_OUTPUTFIELDS
from configuration_constants import JSON_1HOT_SERIES_UP_DOWN
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS
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
        
                if JSON_IGNORE_BLANKS in js_tensorflow_data:
                    nx_ignore_blanks = js_tensorflow_data[JSON_IGNORE_BLANKS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_ignore_blanks}, JSON_IGNORE_BLANKS)

                if JSON_FEATURE_FIELDS in js_tensorflow_data:
                    nx_feature_fields = js_tensorflow_data[JSON_FEATURE_FIELDS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_feature_fields}, JSON_FEATURE_FIELDS)

                if JSON_TARGET_FIELDS in js_tensorflow_data:
                    nx_target_fields = js_tensorflow_data[JSON_TARGET_FIELDS]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_target_fields}, JSON_TARGET_FIELDS)

                if JSON_TIME_SEQ in js_tensorflow_data:
                    nx_time_seq = js_tensorflow_data[JSON_TIME_SEQ]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_time_seq}, JSON_TIME_SEQ)

                if nx_time_seq:
                    nx_seriesStepID = js_tensorflow_data[JSON_SERIES_ID]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_seriesStepID}, JSON_SERIES_ID)

                    nx_seriesDataType = js_tensorflow_data[JSON_SERIES_DATA_TYPE]
                    nx.set_edge_attributes(nx_graph, {nx_edge_key:nx_seriesDataType}, JSON_SERIES_DATA_TYPE)

    return 

def add_data_source_details(js_config, nx_graph, nx_process_name):
    nx_inputFile = js_config[JSON_INPUT_DATA_FILE]
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_inputFile}, JSON_INPUT_DATA_FILE)
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
    nx_shuffle = js_training[JSON_SHUFFLE_DATA]
    
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
    nx.set_node_attributes(nx_graph, {nx_process_name:nx_shuffle}, JSON_SHUFFLE_DATA)

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
    #nx_outputFile = js_required[JSON_OUTPUT_FILE]
    nx_log = js_required[JSON_LOG_FILE]
    
    for node_i in nx_graph.nodes():
        if node_i == nx_process_name:
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_processType}, JSON_PROCESS_TYPE)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_inputs}, JSON_INPUT_FLOWS)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_output}, JSON_OUTPUT_FLOW)
            #nx.set_node_attributes(nx_graph, {nx_process_name:nx_outputFile}, JSON_OUTPUT_FILE)
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_log}, JSON_LOG_FILE)
            break
    
    # conditional elements
    js_conditional = js_config[JSON_CONDITIONAL]
    
    if nx_processType == JSON_DATA_LOAD_PROCESS:
        js_data_prep_ctrl = js_conditional[JSON_INPUT_DATA_PREPARATION]
        add_data_source_details(js_data_prep_ctrl, nx_graph, nx_process_name)
    elif nx_processType == JSON_DATA_PREP_PROCESS:
        nx_prepCtrl = js_conditional[JSON_DATA_PREPARATION_CTRL]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_prepCtrl}, JSON_DATA_PREPARATION_CTRL)
        '''
        js_data_prep_ctrl = js_conditional[JSON_DATA_PREPARATION_CTRL]
        add_data_prep_details(js_data_prep_ctrl, nx_graph, nx_process_name)
        '''
    elif nx_processType == JSON_TENSORFLOW:
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_processType}, JSON_TRAIN)
        nx_model_file = js_conditional[JSON_MODEL_FILE]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_model_file}, JSON_MODEL_FILE)
        nx_dataPrecision = js_conditional[JSON_PRECISION]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_dataPrecision}, JSON_PRECISION)
        nx_visualizations = js_conditional[JSON_VISUALIZATIONS]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_visualizations}, JSON_VISUALIZATIONS)
        
        if JSON_NORMALIZE_DATA in js_conditional:
            nx_normalize = js_conditional[JSON_NORMALIZE_DATA]
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_normalize}, JSON_NORMALIZE_DATA)
        
        if JSON_LAYERS in js_conditional:
            nx_layers = js_conditional[JSON_LAYERS]
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_layers}, JSON_LAYERS)
            
        if JSON_TRAINING in js_conditional:
            add_training_controls(js_conditional[JSON_TRAINING], nx_graph, nx_process_name)
    elif nx_processType == JSON_AUTOKERAS:
        print("============== WIP =============\n\tAuto Keras node details\n================================")
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_processType}, JSON_TRAIN)
        nx_model_file = js_conditional[JSON_MODEL_FILE]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_model_file}, JSON_MODEL_FILE)
        nx_dataPrecision = js_conditional[JSON_PRECISION]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_dataPrecision}, JSON_PRECISION)
        nx_visualizations = js_conditional[JSON_VISUALIZATIONS]
        nx.set_node_attributes(nx_graph, {nx_process_name:nx_visualizations}, JSON_VISUALIZATIONS)

        if JSON_AUTOKERAS_PARAMETERS in js_conditional:
            nx_akParameters = js_conditional[JSON_AUTOKERAS_PARAMETERS]
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_akParameters}, JSON_AUTOKERAS_PARAMETERS)
            
        if JSON_NORMALIZE_DATA in js_conditional:
            nx_normalize = js_conditional[JSON_NORMALIZE_DATA]
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_normalize}, JSON_NORMALIZE_DATA)
        
        if JSON_LAYERS in js_conditional:
            nx_layers = js_conditional[JSON_LAYERS]
            nx.set_node_attributes(nx_graph, {nx_process_name:nx_layers}, JSON_LAYERS)
            
        if JSON_TRAINING in js_conditional:
            add_training_controls(js_conditional[JSON_TRAINING], nx_graph, nx_process_name)
    else:
        ex_txt = nx_process_name + ", type " + nx_processType + " is not supported"
        raise NameError(ex_txt)
    
    nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
    print("Added %s node %s" % (nx_read_attr[nx_process_name], nx_process_name))
    return

def build_configuration_graph(d2r, json_config):
    # error handling
    try:
        d2r.graph = nx.MultiDiGraph()
    
        # processing nodes                
        logging.debug("Adding processing nodes")
        exc_txt = "json configuration file error\n\tprocess node "
        for i, process_i in enumerate(json_config[JSON_PROCESS_NODES]):
            add_meta_process_node(process_i, d2r.graph)
        
        # processing data flows                
        logging.debug("Adding data flows")
        exc_txt = "json configuration file error\n\tdata flow "
        for i, flow_i in enumerate(json_config[JSON_DATA_FLOWS]):
            add_meta_data_edge(flow_i, d2r.graph)
        
        d2r.plotGraph()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + " " + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)

    return
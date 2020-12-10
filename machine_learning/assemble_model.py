'''
Created on Mar 25, 2020

@author: Brian

Build machine learning model using the Keras API as governed by the json script
'''
import sys
import logging
import networkx as nx
import tensorflow as tf
from tensorflow import keras

from configuration_constants import JSON_KERAS_DENSE_PROCESS

from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_INPUT_FLOWS

from configuration_constants import JSON_DATA_FIELDS

from configuration_constants import JSON_CATEGORY_TYPE
from configuration_constants import JSON_CAT_TF
from configuration_constants import JSON_CAT_THRESHOLD
from configuration_constants import JSON_VALUE_RANGES

from configuration_constants import JSON_PREPROCESS_SEQUENCE
from configuration_constants import JSON_PREPROCESS_DISCRETIZATION
from configuration_constants import JSON_PREPROCESS_DISCRETIZATION_BINS
from configuration_constants import JSON_PREPROCESS_CATEGORY_ENCODING

from configuration_constants import JSON_MODEL_INPUT_LAYER
from configuration_constants import JSON_MODEL_OUTPUT_LAYER
from configuration_constants import JSON_MODEL_OUTPUT_ACTIVATION
from configuration_constants import JSON_MODEL_DEPTH
from configuration_constants import JSON_NODE_COUNT
from configuration_constants import JSON_DROPOUT
from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_ACTIVATION
from configuration_constants import JSON_LOSS_WTS
from configuration_constants import JSON_OPTIMIZER

def create_dense_model(nx_graph, nx_node):
    logging.debug("\t\tCreating model defined by: %s" % nx_node)
    print('Using installed tensorflow version %s' % tf.version)
    
    nx_input_flows = nx.get_node_attributes(nx_graph, JSON_INPUT_FLOWS)[nx_node]
    nx_edges = nx.edges(nx_graph)
    for nx_edge in nx_edges:
        if nx_edge[1] == nx_node:
            break
    
    nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_DATA_FIELDS)
    nx_input = nx_dataFields[nx_edge[0], nx_edge[1], nx_input_flows[0]]    

    nx_categories = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_TYPE)
    nx_category_type = nx_categories[nx_edge[0], nx_edge[1], nx_input_flows[0]]    
    if nx_category_type == JSON_CAT_TF:
        nx_outputWidth = 2
    elif nx_category_type == JSON_CAT_THRESHOLD:
        pass
    elif nx_category_type == JSON_VALUE_RANGES:
        pass
    else:
        raise NameError('Invalid category type')
    
    nx_input_layer = nx.get_node_attributes(nx_graph, JSON_MODEL_INPUT_LAYER)[nx_node]
    nx_model_depth = nx.get_node_attributes(nx_graph, JSON_MODEL_DEPTH)[nx_node]
    nx_node_count = nx.get_node_attributes(nx_graph, JSON_NODE_COUNT)[nx_node]
    nx_output_layer = nx.get_node_attributes(nx_graph, JSON_MODEL_OUTPUT_LAYER)[nx_node]
    nx_output_activation = nx.get_node_attributes(nx_graph, JSON_MODEL_OUTPUT_ACTIVATION)[nx_node]
    nx_dropout = nx.get_node_attributes(nx_graph, JSON_DROPOUT)[nx_node]
    nx_dropout_rate = nx.get_node_attributes(nx_graph, JSON_DROPOUT_RATE)[nx_node]
    nx_loss = nx.get_node_attributes(nx_graph, JSON_LOSS)[nx_node]
    nx_metrics = nx.get_node_attributes(nx_graph, JSON_METRICS)[nx_node]
    nx_activation = nx.get_node_attributes(nx_graph, JSON_ACTIVATION)[nx_node]
    nx_optimizer = nx.get_node_attributes(nx_graph, JSON_OPTIMIZER)[nx_node]
    nx_loss_weights = nx.get_node_attributes(nx_graph, JSON_LOSS_WTS)[nx_node]
    
    k_inputs = tf.keras.Input(name=nx_input_layer, shape=(len(nx_input),))
    ndx = 0
    while ndx < nx_model_depth:
        if ndx == 0:
            k_layer = tf.keras.layers.Dense(nx_node_count, activation=nx_activation)(k_inputs)
        else:
            k_layer = tf.keras.layers.Dense(nx_node_count, activation=nx_activation)(k_layer)
        if nx_dropout:
            k_layer = tf.keras.layers.Dropout(nx_dropout_rate)(k_layer)
        ndx += 1
    k_outputs = tf.keras.layers.Dense(nx_outputWidth, name=nx_output_layer, activation=nx_output_activation)(k_layer)
    model = tf.keras.Model(name=nx_node, inputs=k_inputs, outputs=k_outputs)
    model.compile(nx_optimizer, nx_loss, metrics=nx_metrics)
    # , loss_weights=nx_loss_weights

    return model

def create_LSTM_model(nx_graph):
    return

def create_RNN_model(nx_graph):
    return

def build_model(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> build_model: building the machine learning model')
    logging.info('====> ================================================')

    # error handling
    try:
        # inputs                
        logging.debug("Building ML model")
        nx_preprocess_sequence = nx.get_node_attributes(nx_graph, JSON_PREPROCESS_SEQUENCE)

        for node_i in nx_graph.nodes():
            nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == JSON_KERAS_DENSE_PROCESS:
                if node_i in nx_preprocess_sequence:
                    print ("Including preprocess steps %s" % nx_preprocess_sequence)
                    for preprocessStep in nx_preprocess_sequence[node_i]:
                        if preprocessStep == JSON_PREPROCESS_DISCRETIZATION:
                            nx_discretization_bins = nx.get_node_attributes(nx_graph, JSON_PREPROCESS_DISCRETIZATION_BINS)
                            nx_bins = nx_discretization_bins[node_i]
                            print ("Setting up discretization step %s, %s" % (preprocessStep, nx_bins))
                k_model = create_dense_model(nx_graph, node_i)
        
    except Exception:
        err_txt = "*** An exception occurred building and compiling the model ***"
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model: done')
    logging.info('<---- ----------------------------------------------')    
    return k_model
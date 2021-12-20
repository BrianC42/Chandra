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

from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_PRECISION
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import MODEL_TYPE
from configuration_constants import INPUT_LAYERTYPE_DENSE
from configuration_constants import INPUT_LAYERTYPE_RNN
from configuration_constants import INPUT_LAYERTYPE_CNN
from configuration_constants import JSON_LAYERS
from configuration_constants import JSON_MODEL_OUTPUT_ACTIVATION
from configuration_constants import JSON_TIMESTEPS
from configuration_constants import JSON_FEATURE_COUNT
from configuration_constants import JSON_DROPOUT
from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_RETURN_SEQUENCES
from configuration_constants import JSON_REPEAT_COUNT
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_LOSS_WTS
from configuration_constants import JSON_OPTIMIZER
from configuration_constants import JSON_MODEL_FILE

def build_layer(d2r, layer_type, layer_definition, input_layer):
    ''' some parameters are common across multiple layer type '''
    nx_layer_name = None
    if 'layerName' in layer_definition:
        nx_layer_name = layer_definition['layerName']
                
    nx_layer_units = None
    if 'layerUnits' in layer_definition:
        nx_layer_units = layer_definition['layerUnits']

    nx_activation = 'tanh'
    if JSON_MODEL_OUTPUT_ACTIVATION in layer_definition:
        nx_activation = layer_definition[JSON_MODEL_OUTPUT_ACTIVATION]
                
    if layer_type == 'dense':
        if input_layer:
            d2r.modelType = INPUT_LAYERTYPE_DENSE
            nx.set_node_attributes(d2r.graph, {d2r.mlNode:INPUT_LAYERTYPE_DENSE}, MODEL_TYPE)
            nx_feature_count = layer_definition[JSON_FEATURE_COUNT]
            k_layer = keras.layers.Dense(name=nx_layer_name, \
                                            activation = nx_activation, \
                                            units=nx_layer_units, \
                                            input_shape=(nx_feature_count, ))
        else:
            k_layer = keras.layers.Dense(name=nx_layer_name, \
                                         activation = nx_activation, \
                                         units=nx_layer_units)
            
    elif layer_type == 'lstm':
        nx.set_node_attributes(d2r.graph, {d2r.mlNode:INPUT_LAYERTYPE_RNN}, MODEL_TYPE)
        nx_return_sequences = False
        if JSON_RETURN_SEQUENCES in layer_definition:
            nx_return_sequences = layer_definition[JSON_RETURN_SEQUENCES]
        if input_layer:
            d2r.modelType = INPUT_LAYERTYPE_RNN
            nx_feature_count = layer_definition[JSON_FEATURE_COUNT]
            nx_time_steps = layer_definition[JSON_TIMESTEPS]
            nx.set_node_attributes(d2r.graph, {d2r.mlNode:nx_time_steps}, JSON_TIMESTEPS)
            k_layer = keras.layers.LSTM(name=nx_layer_name, \
                                           units=nx_layer_units, \
                                           activation = nx_activation, \
                                           return_sequences = nx_return_sequences, \
                                           input_shape=(nx_time_steps, nx_feature_count))
        else:
            k_layer = keras.layers.LSTM(name=nx_layer_name, \
                                           units=nx_layer_units, \
                                           activation = nx_activation, \
                                           return_sequences = nx_return_sequences)
    elif layer_type == 'cnn':
        '''
        Future support required
        '''
        if input_layer:
            d2r.modelType = INPUT_LAYERTYPE_CNN
            nx.set_node_attributes(d2r.graph, {d2r.mlNode:INPUT_LAYERTYPE_CNN}, MODEL_TYPE)
    elif layer_type == 'RepeatVector':
        nx_repeat_count = 1
        if JSON_REPEAT_COUNT in layer_definition:
            nx_repeat_count = layer_definition[JSON_REPEAT_COUNT]
        k_layer = keras.layers.RepeatVector(nx_repeat_count)
    elif layer_type == JSON_DROPOUT:
        nx_dropout_rate = 0.2
        if JSON_DROPOUT_RATE in layer_definition:
            nx_dropout_rate = layer_definition[JSON_DROPOUT_RATE]
        k_layer = keras.layers.Dropout(rate=nx_dropout_rate)
    else:
        err_msg = 'Layer type not yet implemented: ' + layer_type
        raise NameError(err_msg)

    return k_layer

def assemble_layers(d2r):

    logging.info('====> ================================================')
    logging.info('====> build_model: building the machine learning model')
    logging.info('====> ================================================')

    # error handling
    try:
        # inputs                
        print("Assembling model defined in node: %s" % d2r.mlNode)
        logging.debug("Building ML model")

        err_txt = "*** An exception occurred building the model ***"
        input_layer = True
        nx_layers = nx.get_node_attributes(d2r.graph, JSON_LAYERS)[d2r.mlNode]
        nx_data_precision = nx.get_node_attributes(d2r.graph, JSON_PRECISION)[d2r.mlNode]
        keras.backend.set_floatx(nx_data_precision)
        k_model = keras.Sequential()
        
        for nx_layer_definition in nx_layers:
            ''' layer type is required '''
            nx_layer_type = nx_layer_definition['layerType']
            if nx_layer_type == 'timeDistributed':
                '''
                TimeDistributed layer type specification is not defined - future support required
                '''
                k_layer = build_layer(d2r, nx_layer_type, nx_layer_definition, input_layer)
                k_model.add(keras.layers.TimeDistributed(k_layer))
            else:
                k_layer = build_layer(d2r, nx_layer_type, nx_layer_definition, input_layer)
                k_model.add(k_layer)
            input_layer = False
            
        err_txt = "*** An exception occurred compiling the model ***"
        nx_loss = nx.get_node_attributes(d2r.graph, JSON_LOSS)[d2r.mlNode]
        nx_metrics = nx.get_node_attributes(d2r.graph, JSON_METRICS)[d2r.mlNode]
        nx_optimizer = nx.get_node_attributes(d2r.graph, JSON_OPTIMIZER)[d2r.mlNode]
        nx_loss_weights = nx.get_node_attributes(d2r.graph, JSON_LOSS_WTS)[d2r.mlNode]
        
        k_model.compile(optimizer=nx_optimizer, loss=nx_loss, metrics=nx_metrics, loss_weights=nx_loss_weights)
        print("compile optimizer:%s loss:%s metrics:%s loss_weights:%s" % \
              (nx_optimizer, nx_loss, nx_metrics, nx_loss_weights))
        k_model.summary()
        nx_model_file = nx.get_node_attributes(d2r.graph, JSON_MODEL_FILE)[d2r.mlNode]

        err_txt = "*** An exception occurred plotting the model ***"
        keras.utils.plot_model(k_model, to_file=nx_model_file + '.png', show_shapes=True)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" + exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += s
        logging.debug(exc_txt)
        sys.exit(exc_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model: done')
    logging.info('<---- ----------------------------------------------')    
    #return k_model, x_features, y_targets, df_training_data, np_x_norm, np_y_norm, np_x_test, np_y_test
    return k_model

def buildModel(d2r):
    nx_edges = nx.edges(d2r.graph)
    
    for node_i in d2r.graph.nodes():
        nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
        if nx_read_attr[node_i] == JSON_TENSORFLOW:
            for nx_edge in nx_edges:
                if nx_edge[1] == node_i:
                    d2r.mlNode = node_i
                    d2r.mlEdgeIn = nx_edge
                    d2r.model = assemble_layers(d2r)
                    break
            break

    return
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
import autokeras as ak

from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_AUTOKERAS
from configuration_constants import JSON_AUTOKERAS_PARAMETERS
from configuration_constants import JSON_AK_TASK
from configuration_constants import JSON_AK_IMAGE_CLASSIFIER
from configuration_constants import JSON_AK_IMAGE_REGRESSOR
from configuration_constants import JSON_AK_TEXT_CLASSIFIER
from configuration_constants import JSON_AK_TEXT_REGRESSOR
from configuration_constants import JSON_AK_STRUCTURED_DATA_CLASSIFIER
from configuration_constants import JSON_AK_STRUCTURED_DATA_REGRESSOR
from configuration_constants import JSON_AK_MULTI
from configuration_constants import JSON_AK_CUSTOM
from configuration_constants import JSON_AK_DIR
from configuration_constants import JSON_AK_MAX_TRIALS
from configuration_constants import JSON_PRECISION
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_LAYERS
from configuration_constants import JSON_LAYER_TYPE
from configuration_constants import JSON_LAYER_DENSE
from configuration_constants import JSON_LAYER_LSTM
from configuration_constants import JSON_LAYER_NAME
from configuration_constants import JSON_LAYER_UNITS
from configuration_constants import JSON_LAYER_REPEAT_VECTOR
from configuration_constants import JSON_LAYER_TIME_DISTRIBUTED
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

from configuration_constants import JSON_CONV1D
from configuration_constants import JSON_FILTER_COUNT
from configuration_constants import JSON_FILTER_SIZE
from configuration_constants import JSON_BATCHES

from configuration_constants import JSON_MAXPOOLING_1D
from configuration_constants import JSON_GLOBAL_MAXPOOLING_1D
from configuration_constants import JSON_POOL_SIZE

from configuration_constants import JSON_FLATTEN

from TrainingDataAndResults import TRAINING_TENSORFLOW
from TrainingDataAndResults import TRAINING_AUTO_KERAS
from TrainingDataAndResults import MODEL_TYPE
from TrainingDataAndResults import INPUT_LAYERTYPE_DENSE
from TrainingDataAndResults import INPUT_LAYERTYPE_RNN
from TrainingDataAndResults import INPUT_LAYERTYPE_CNN

def build_layer(d2r, layer_type, layer_definition, input_layer):
    ''' some parameters are common across multiple layer type '''
    nx_layer_name = None
    if JSON_LAYER_NAME in layer_definition:
        nx_layer_name = layer_definition[JSON_LAYER_NAME]
                
    nx_layer_units = None
    if JSON_LAYER_UNITS in layer_definition:
        nx_layer_units = layer_definition[JSON_LAYER_UNITS]

    nx_activation = None
    if JSON_MODEL_OUTPUT_ACTIVATION in layer_definition:
        nx_activation = layer_definition[JSON_MODEL_OUTPUT_ACTIVATION]
                
    if layer_type == JSON_LAYER_DENSE:
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
            
    elif layer_type == JSON_LAYER_LSTM:
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
                                           input_shape=(nx_time_steps, nx_feature_count) \
                                           )
        else:
            k_layer = keras.layers.LSTM(name=nx_layer_name, \
                                           units=nx_layer_units, \
                                           activation = nx_activation, \
                                           return_sequences = nx_return_sequences)
    elif layer_type == JSON_CONV1D:
        if input_layer:
            d2r.modelType = INPUT_LAYERTYPE_CNN
            nx.set_node_attributes(d2r.graph, {d2r.mlNode:INPUT_LAYERTYPE_CNN}, MODEL_TYPE)
            d2r.timesteps = layer_definition[JSON_TIMESTEPS]
            d2r.filter_count = layer_definition[JSON_FILTER_COUNT]
            d2r.filter_size = layer_definition[JSON_FILTER_SIZE]

            #d2r.batches = 1
            #d2r.batches = layer_definition[JSON_BATCHES]
            #d2r.timesteps = layer_definition[JSON_TIMESTEPS]
            d2r.feature_count = layer_definition[JSON_FEATURE_COUNT]
            nx_input_shape = (None, d2r.feature_count)
            
            nx_strides = 1
            nx_dilation_rate = 1
            nx_padding = 'causal'

            k_layer = keras.layers.Conv1D(d2r.filter_count, \
                                          d2r.filter_size, \
                                          input_shape = nx_input_shape, \
                                          name = nx_layer_name, \
                                          strides = nx_strides, \
                                          padding = nx_padding, \
                                          activation = nx_activation, \
                                          dilation_rate = nx_dilation_rate \
                                          )
        else:
            k_layer = keras.layers.Conv1D(name=nx_layer_name, \
                                         activation = nx_activation, \
                                         units=nx_layer_units)
    elif layer_type == JSON_LAYER_REPEAT_VECTOR:
        nx_repeat_count = 1
        if JSON_REPEAT_COUNT in layer_definition:
            nx_repeat_count = layer_definition[JSON_REPEAT_COUNT]
        k_layer = keras.layers.RepeatVector(nx_repeat_count)
    elif layer_type == JSON_DROPOUT:
        nx_dropout_rate = 0.2
        if JSON_DROPOUT_RATE in layer_definition:
            nx_dropout_rate = layer_definition[JSON_DROPOUT_RATE]
        k_layer = keras.layers.Dropout(rate=nx_dropout_rate)
    elif layer_type == JSON_LAYER_TIME_DISTRIBUTED:
        print("\n============== WIP =============\n\tTimeDistributed layer type assumes a dense layer\n================================\n")
        #nx_feature_count = layer_definition[JSON_FEATURE_COUNT]
        TD_layer = keras.layers.Dense(units=nx_layer_units, activation = nx_activation)
        k_layer = keras.layers.TimeDistributed(TD_layer)
    elif layer_type == JSON_MAXPOOLING_1D:
        print("\n============== WIP =============\n\tWIP MaxPooling1D layer type - pool_size hard coded\n================================\n")
        k_layer = keras.layers.MaxPooling1D(name=nx_layer_name)
    elif layer_type == JSON_GLOBAL_MAXPOOLING_1D:
        k_layer = keras.layers.GlobalMaxPool1D(name=nx_layer_name)
    elif layer_type == JSON_FLATTEN:
        k_layer = keras.layers.Flatten(name=nx_layer_name)
    else:
        err_msg = 'Layer type not yet implemented: ' + layer_type
        raise NameError(err_msg)

    return k_layer


    return

def assemble_layers(d2r):
    # error handling
    try:
        # inputs                
        print("Assembling model defined in node: %s, iteration %s" % (d2r.mlNode, d2r.trainingIteration))
        logging.debug("Building ML model")

        err_txt = "*** An exception occurred building the model ***"
        input_layer = True
        
        nx_modelIterations = nx.get_node_attributes(d2r.graph, "training iterations")[d2r.mlNode]
        iterVariables = nx_modelIterations[d2r.trainingIteration]
        iterParamters = iterVariables["iteration parameters"]
        
        nx_layers = iterParamters["modelLayers"]
        
        nx_data_precision = nx.get_node_attributes(d2r.graph, JSON_PRECISION)[d2r.mlNode]
        keras.backend.set_floatx(nx_data_precision)
        d2r.model = keras.Sequential()
        
        for nx_layer_definition in nx_layers:
            ''' layer type is required '''
            nx_layer_type = nx_layer_definition[JSON_LAYER_TYPE]
            k_layer = build_layer(d2r, nx_layer_type, nx_layer_definition, input_layer)
            d2r.model.add(k_layer)
            input_layer = False
            
        err_txt = "*** An exception occurred compiling the model ***"
        iterTraining = iterParamters["training"]
        nx_loss = iterTraining[JSON_LOSS]
        nx_metrics = iterTraining[JSON_METRICS]
        nx_optimizer = iterTraining[JSON_OPTIMIZER]
        nx_loss_weights = iterTraining[JSON_LOSS_WTS]
        
        if nx_optimizer['name'] == 'adam':
            # learning_rate=0.001
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = nx_optimizer['learning_rate'],
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                jit_compile=True,
                name='Adam')
        elif nx_optimizer == 'SGD':
            optimizer = tf.keras.optimizers.experimental.SGD(
                learning_rate = nx_optimizer['learning_rate'],
                momentum=0.0,
                nesterov=False,
                amsgrad=False,
                weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                jit_compile=True,
                name='SGD')
        elif nx_optimizer == 'RMSProp':
            optimizer = tf.keras.optimizers.experimental.RMSprop(
                learning_rate = nx_optimizer['learning_rate'],
                rho=0.9,
                momentum=0.0,
                epsilon=1e-07,
                centered=False,
                weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=100,
                jit_compile=True,
                name='RMSprop')
        elif nx_optimizer == 'Nadam':
            optimizer = nx_optimizer
        elif nx_optimizer == 'Adamax':
            optimizer = nx_optimizer
        elif nx_optimizer == 'adagrad':
            optimizer = nx_optimizer
        elif nx_optimizer == 'adadelta':
            optimizer = nx_optimizer
        else:
            err_msg = 'Invalid optimizer: ' + nx_optimizer
            raise NameError(err_msg)
            
        d2r.model.compile(optimizer=optimizer, loss=nx_loss, metrics=nx_metrics, loss_weights=nx_loss_weights)
        print("compile optimizer:%s loss:%s metrics:%s loss_weights:%s" % \
              (nx_optimizer, nx_loss, nx_metrics, nx_loss_weights))
        d2r.model.summary()
        nx_model_file = nx.get_node_attributes(d2r.graph, JSON_MODEL_FILE)[d2r.mlNode]

        err_txt = "*** An exception occurred plotting the model ***"
        keras.utils.plot_model(d2r.model, to_file=nx_model_file + '.png', show_shapes=True)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            nx_layer_name = None
            if JSON_LAYER_NAME in nx_layer_definition:
                nx_layer_name = nx_layer_definition[JSON_LAYER_NAME]
            exc_txt = err_txt + "\n\t" + nx_layer_name + "\n\t" +exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += s
        logging.debug(exc_txt)
        sys.exit(exc_txt)
    
    return

def build_autokeras_model(d2r):
    # error handling
    try:
        print("\n============== WIP =============\n\tBuilding / configuring AutoKeras model\n================================\n")
        err_txt = "*** An exception occurred building the AutoKeras model ***"

        autoKeras = nx.get_node_attributes(d2r.graph, JSON_AUTOKERAS_PARAMETERS)[d2r.mlNode]
        
        akTask = autoKeras[JSON_AK_TASK]
        akMaxTrials = autoKeras[JSON_AK_MAX_TRIALS]
        akDirectory = autoKeras[JSON_AK_DIR]
        if akTask == JSON_AK_IMAGE_CLASSIFIER:
            err_msg = 'AutoKeras task not yet implemented: ' + akTask
            raise NameError(err_msg)
        elif akTask == JSON_AK_IMAGE_REGRESSOR:
            d2r.model = ak.ImageRegressor(overwrite=True, max_trials=akMaxTrials, directory=akDirectory)
        elif akTask == JSON_AK_TEXT_CLASSIFIER:
            err_msg = 'AutoKeras task not yet implemented: ' + akTask
            raise NameError(err_msg)
        elif akTask == JSON_AK_TEXT_REGRESSOR:
            err_msg = 'AutoKeras task not yet implemented: ' + akTask
            raise NameError(err_msg)
        elif akTask == JSON_AK_STRUCTURED_DATA_CLASSIFIER:
            err_msg = 'AutoKeras task not yet implemented: ' + akTask
            raise NameError(err_msg)
        elif akTask == JSON_AK_STRUCTURED_DATA_REGRESSOR:
            d2r.model = ak.StructuredDataRegressor(overwrite=True, max_trials=akMaxTrials, directory=akDirectory)
        elif akTask == JSON_AK_MULTI:
            err_msg = 'AutoKeras task not yet implemented: ' + akTask
            raise NameError(err_msg)
        elif akTask == JSON_AK_CUSTOM:
            err_msg = 'AutoKeras task not yet implemented: ' + akTask
            raise NameError(err_msg)
        else:
            err_msg = 'AutoKeras task is not valid: ' + akTask
            raise NameError(err_msg)
        
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
    
    return

def buildModel(d2r):
    nx_edges = nx.edges(d2r.graph)
    
    for node_i in d2r.graph.nodes():
        nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
        if nx_read_attr[node_i] == JSON_TENSORFLOW:
            for nx_edge in nx_edges:
                if nx_edge[1] == node_i:
                    d2r.mlNode = node_i
                    d2r.mlEdgeIn = nx_edge
                    d2r.trainer = TRAINING_TENSORFLOW
                    assemble_layers(d2r)
                    break
            break
        elif nx_read_attr[node_i] == JSON_AUTOKERAS:
            for nx_edge in nx_edges:
                if nx_edge[1] == node_i:
                    d2r.mlNode = node_i
                    d2r.mlEdgeIn = nx_edge
                    d2r.trainer = TRAINING_AUTO_KERAS
                    build_autokeras_model(d2r)
                    break
            break

    return
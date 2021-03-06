'''
Created on Mar 25, 2020

@author: Brian

Build machine learning model using the Keras API as governed by the json script
'''
import os
import sys
import logging
import networkx as nx
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from train_model import trainModels

from configuration_constants import JSON_KERAS_DENSE_PROCESS
from configuration_constants import JSON_KERAS_CONV1D

from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_INPUT_FLOWS

from configuration_constants import JSON_FEATURE_FIELDS

from configuration_constants import JSON_FLOW_DATA_FILE

from configuration_constants import JSON_CATEGORY_TYPE
from configuration_constants import JSON_CAT_TF
from configuration_constants import JSON_CAT_THRESHOLD
from configuration_constants import JSON_VALUE_RANGES
from configuration_constants import JSON_LINEAR_REGRESSION

from configuration_constants import JSON_THRESHOLD_VALUE

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

from configuration_constants import JSON_MODEL_FILE

def create_dense_model(nx_graph, nx_node, nx_outputWidth):
    logging.debug("\t\tCreating model defined by: %s" % nx_node)
    print('Using installed tensorflow version %s' % tf.version)
    
    nx_edges = nx.edges(nx_graph)
    for nx_edge in nx_edges:
        if nx_edge[1] == nx_node:
            break
    
    nx_input_layer = nx.get_node_attributes(nx_graph, JSON_MODEL_INPUT_LAYER)[nx_node]
    nx_features = nx.get_edge_attributes(nx_graph, JSON_FEATURE_FIELDS)
    nx_model_depth = nx.get_node_attributes(nx_graph, JSON_MODEL_DEPTH)[nx_node]
    nx_node_count = nx.get_node_attributes(nx_graph, JSON_NODE_COUNT)[nx_node]
    nx_dropout = nx.get_node_attributes(nx_graph, JSON_DROPOUT)[nx_node]
    nx_dropout_rate = nx.get_node_attributes(nx_graph, JSON_DROPOUT_RATE)[nx_node]
    nx_activation = nx.get_node_attributes(nx_graph, JSON_ACTIVATION)[nx_node]
    
    k_input = tf.keras.Input(name=nx_input_layer, shape=(len(nx_features),))
    ndx = 0
    while ndx < nx_model_depth:
        if ndx == 0:
            k_layer = tf.keras.layers.Dense(nx_node_count, activation=nx_activation)(k_input)
        else:
            k_layer = tf.keras.layers.Dense(nx_node_count, activation=nx_activation)(k_layer)
        if nx_dropout:
            k_layer = tf.keras.layers.Dropout(nx_dropout_rate)(k_layer)
        ndx += 1

    nx_output_layer = nx.get_node_attributes(nx_graph, JSON_MODEL_OUTPUT_LAYER)[nx_node]
    nx_output_activation = nx.get_node_attributes(nx_graph, JSON_MODEL_OUTPUT_ACTIVATION)[nx_node]
    k_outputs = tf.keras.layers.Dense(nx_outputWidth, name=nx_output_layer, activation=nx_output_activation)(k_layer)

    return k_input, k_outputs

def create_Convolutional_model(nx_graph, nx_node):
    print("\n*************************************************\nPARAMETERS HARD CODED - NEED TO BE DETERMINED AND PROGRAMMED\n*************************************************\n")
    nx_input_layer = nx.get_node_attributes(nx_graph, JSON_MODEL_INPUT_LAYER)[nx_node]

    k_input = tf.keras.Input(name=nx_input_layer, shape=(20, 10))
    k_layer = tf.keras.layers.Conv1D(10, 5)(k_input)
    k_layer = tf.keras.layers.Dense(5, activation='relu')(k_layer)
        
    nx_output_layer = nx.get_node_attributes(nx_graph, JSON_MODEL_OUTPUT_LAYER)[nx_node]
    nx_output_activation = nx.get_node_attributes(nx_graph, JSON_MODEL_OUTPUT_ACTIVATION)[nx_node]
    k_outputs = tf.keras.layers.Dense(1, name=nx_output_layer, activation=nx_output_activation)(k_layer)
    return k_input, k_outputs

def create_LSTM_model(nx_graph):
    return

def create_RNN_model(nx_graph):
    return

def normalize_data(k_input_layer, df_data):
    print("Adding normalization")
    '''
    Normalization
    from TensorFlow.org
    Note: You can set up the keras.Model to do this kind of transformation for you. That's beyond the scope of this tutorial. 
    See the preprocessing layers or Loading CSV data tutorials for examples. https://www.tensorflow.org/tutorials/load_data/csv
    
    The Normalization layer
    The preprocessing.Normalization layer is a clean and simple way to build that preprocessing into your model.

    The first step is to create the layer: (from https://www.tensorflow.org/tutorials/keras/regression)

    normalizer = preprocessing.Normalization()
    Then .adapt() it to the data:

    normalizer.adapt(np.array(train_features))
    This calculates the mean and variance, and stores them in the layer.
    
    First create the horsepower Normalization layer:

    horsepower = np.array(train_features['Horsepower'])
    horsepower_normalizer = preprocessing.Normalization(input_shape=[1,])
    horsepower_normalizer.adapt(horsepower)
    
    tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=0.001,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs
    
    tf.keras.layers.LayerNormalization(
        axis,
        epsilon,
        center
        scale
        beta_initializer
        gamma_initializer
        beta_regularizer
        gamma_regularizer
        beta_constraint
        gamma_constraint
        **kwargs
    )

    np_features = np.array(df_x_train)
    tf_dataset = tf.data.Dataset.from_tensor_slices(np_features)
    print("dataflow\n%s\ntensor dataset\n%s" % (df_x_train, tf_dataset))
    norm = preprocessing.Normalization()
    norm.adapt(tf_dataset)
    '''
    
    return k_input_layer, df_data

def preprocess_data(nx_graph, node_i, df_data):
    '''
    see: https://www.tensorflow.org/tutorials/load_data/csv
    
    Available preprocessing layers
    Core preprocessing layers
        TextVectorization layer: turns raw strings into an encoded representation that can be read by an Embedding layer or Dense layer.
        Normalization layer: performs feature-wise normalize of input features.
        
    Structured data preprocessing layers
    These layers are for structured data encoding and feature engineering.
        CategoryEncoding layer: turns integer categorical features into one-hot, multi-hot, or TF-IDF dense representations.
        Hashing layer: performs categorical feature hashing, also known as the "hashing trick".
        Discretization layer: turns continuous numerical features into integer categorical features.
        StringLookup layer: turns string categorical values into integers indices.
        IntegerLookup layer: turns integer categorical values into integers indices.
        CategoryCrossing layer: combines categorical features into co-occurrence features. E.g. if you have feature values "a" and "b", it can provide with the combination feature "a and b are present at the same time".
        
    Image preprocessing layers
    These layers are for standardizing the inputs of an image model.
        Resizing layer: resizes a batch of images to a target size.
        Rescaling layer: rescales and offsets the values of a batch of image (e.g. go from inputs in the [0, 255] range to inputs in the [0, 1] range.
        CenterCrop layer: returns a center crop if a batch of images.
        
    Image data augmentation layers
    These layers apply random augmentation transforms to a batch of images. They are only active during training.
        RandomCrop layer
        RandomFlip layer
        RandomTranslation layer
        RandomRotation layer
        RandomZoom layer
        RandomHeight layer
        RandomWidth layer
        
    The adapt() method
    Some preprocessing layers have an internal state that must be computed based on a sample of the training data. The list of stateful preprocessing layers is:
        TextVectorization: holds a mapping between string tokens and integer indices
        Normalization: holds the mean and standard deviation of the features
        StringLookup and IntegerLookup: hold a mapping between input values and output indices.
        CategoryEncoding: holds an index of input values.
        Discretization: holds information about value bucket boundaries.
        
    Crucially, these layers are non-trainable. Their state is not set during training; it must be set before training, a step called "adaptation".
    You set the state of a preprocessing layer by exposing it to training data, via the adapt() method:    
    '''
    
    return df_data

def load_training_data(nx_graph, node_i, nx_edge):
    # error handling
    try:
        err_txt = "*** An exception occurred preparing the training data for the model ***"

        nx_input_flow = nx.get_node_attributes(nx_graph, JSON_INPUT_FLOWS)[node_i]
        print("Loading prepared data defined in flow: %s" % nx_input_flow[0])
        nx_data_file = nx.get_edge_attributes(nx_graph, JSON_FLOW_DATA_FILE)
        inputData = nx_data_file[nx_edge[0], nx_edge[1], nx_input_flow[0]]    
        if os.path.isfile(inputData):
            df_training_data = pd.read_csv(inputData)
                
        nx_category_types = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_TYPE)
        category_type = nx_category_types[nx_edge[0], nx_edge[1], nx_input_flow[0]]
        if category_type == JSON_CAT_TF:
            pass
        elif category_type == JSON_CAT_THRESHOLD:
            pass
        elif category_type == JSON_THRESHOLD_VALUE:
            pass
        elif category_type == JSON_LINEAR_REGRESSION:
            pass
        else:
            raise NameError('Invalid category type')
    
    except Exception:
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)

    return df_training_data

def build_model(nx_graph, node_i, nx_edge, nx_model_type):
    logging.info('====> ================================================')
    logging.info('====> build_model: building the machine learning model')
    logging.info('====> ================================================')

    # error handling
    try:
        # inputs                
        print("Assembling model defined in node: %s" % node_i)
        logging.debug("Building ML model")

        tf.keras.backend.set_floatx('float64')

        nx_categories = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_TYPE)
        nx_input_flows = nx.get_node_attributes(nx_graph, JSON_INPUT_FLOWS)[node_i]
        nx_category_type = nx_categories[nx_edge[0], nx_edge[1], nx_input_flows[0]]
        if nx_category_type == JSON_CAT_TF:
            nx_outputWidth = 2
        elif nx_category_type == JSON_LINEAR_REGRESSION:
            nx_outputWidth = 1
        elif nx_category_type == JSON_CAT_THRESHOLD:
            pass
        elif nx_category_type == JSON_VALUE_RANGES:
            pass
        else:
            raise NameError('Invalid category type')

        err_txt = "*** An exception occurred building the model ***"        
        if nx_model_type == JSON_KERAS_DENSE_PROCESS:
            k_input, k_outputs = create_dense_model(nx_graph, node_i, nx_outputWidth)
        elif nx_model_type == JSON_KERAS_CONV1D:
            k_input, k_outputs = create_Convolutional_model(nx_graph, node_i)        
        k_model = tf.keras.Model(name=node_i, inputs=k_input, outputs=k_outputs)

        err_txt = "*** An exception occurred compiling the model ***"
        nx_loss = nx.get_node_attributes(nx_graph, JSON_LOSS)[node_i]
        nx_metrics = nx.get_node_attributes(nx_graph, JSON_METRICS)[node_i]
        nx_optimizer = nx.get_node_attributes(nx_graph, JSON_OPTIMIZER)[node_i]
        nx_loss_weights = nx.get_node_attributes(nx_graph, JSON_LOSS_WTS)[node_i]
        k_model.compile(optimizer=nx_optimizer, loss=nx_loss, metrics=nx_metrics, loss_weights=nx_loss_weights)
        print("compile optimizer:%s loss:%s metrics:%s loss_weights:%s" % \
              (nx_optimizer, nx_loss, nx_metrics, nx_loss_weights))
        k_model.summary()
        nx_model_file = nx.get_node_attributes(nx_graph, JSON_MODEL_FILE)[node_i]
        keras.utils.plot_model(k_model, to_file=nx_model_file + '.png', show_shapes=True)
                
    except Exception:
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- build_model: done')
    logging.info('<---- ----------------------------------------------')    
    #return k_model, x_features, y_targets, df_training_data, np_x_norm, np_y_norm, np_x_test, np_y_test
    return k_model

def build_and_train_model(nx_graph):
    '''
    step 1 - load the feature and target data and structure them as required by the machine learning approach
    step 2 - assemble and compile the model defined in the json file with the definition stored in the directed network graph
    step 3 - train the model using the data organized in step 1 and the model compiled in step 2
    '''
    nx_edges = nx.edges(nx_graph)
    
    for node_i in nx_graph.nodes():
        nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
        if nx_read_attr[node_i] == JSON_KERAS_DENSE_PROCESS:
            nx_model_type = JSON_KERAS_DENSE_PROCESS
            break
        elif nx_read_attr[node_i] == JSON_KERAS_CONV1D:
            nx_model_type = JSON_KERAS_CONV1D
            break
        
    for nx_edge in nx_edges:
        if nx_edge[1] == node_i:
            break
            
    
    tf.keras.backend.set_floatx('float64')
    df_training_data = load_training_data(nx_graph, node_i, nx_edge)
    k_model = build_model(nx_graph, node_i, nx_edge, nx_model_type)
    fitting, x_features, y_targets, x_train, y_train = trainModels(nx_graph, node_i, nx_edge, k_model, df_training_data)
    
    return node_i, k_model, fitting, x_features, y_targets, x_train, y_train
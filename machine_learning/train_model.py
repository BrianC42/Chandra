'''
Created on Oct 9, 2020

@author: Brian
'''
import sys
import logging
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

from configuration_constants import JSON_INPUT_FLOWS

from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_NORMALIZE_DATA

from configuration_constants import JSON_BATCH
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_SHUFFLE_DATA

#def trainModels(nx_graph, node_i, nx_edge, k_model, df_training_data, x_train,y_train, x_test, y_test):
def trainModels(nx_graph, node_i, nx_edge, k_model, df_training_data):
    logging.info('====> ================================================')
    logging.info('====> trainModels models')
    logging.info('====> ================================================')

    # error handling
    try:
        '''
        nx_regularization = nx.get_node_attributes(nx_graph, JSON_REGULARIZATION)[node_i]
        nx_reg_value = nx.get_node_attributes(nx_graph, JSON_REG_VALUE)[node_i]
        nx_bias = nx.get_node_attributes(nx_graph, JSON_BIAS)[node_i]
        nx_balanced = nx.get_node_attributes(nx_graph, JSON_BALANCED)[node_i]
        nx_analysis = nx.get_node_attributes(nx_graph, JSON_ANALYSIS)[node_i]
                
        fit parameters not used:
                    validation_split - validation_data used instead
                    shuffle
                    class_weight
                    sample_weight
                    initial_epooch
                    steps_per_epoch
                    validation_steps
                    validation_batch_size
                    validation_freq
                    max_queue_size
                    workers
                    use_multiprocessing
        '''
        nx_input_flows = nx.get_node_attributes(nx_graph, JSON_INPUT_FLOWS)[node_i]

        nx_featureFields = nx.get_edge_attributes(nx_graph, JSON_FEATURE_FIELDS)
        nx_features = nx_featureFields[nx_edge[0], nx_edge[1], nx_input_flows[0]]    

        nx_targetFiields = nx.get_edge_attributes(nx_graph, JSON_TARGET_FIELDS)
        nx_targets = nx_targetFiields[nx_edge[0], nx_edge[1], nx_input_flows[0]]    

        nx_validation_split = nx.get_node_attributes(nx_graph, JSON_VALIDATION_SPLIT)[node_i]

        rows = df_training_data.shape[0]
        x_features = df_training_data.loc[:, nx_features]
        y_targets = df_training_data.loc[:, nx_targets]
        
        df_x_train = df_training_data.loc[int(rows * nx_validation_split):, nx_features]
        df_y_train = df_training_data.loc[int(rows * nx_validation_split):, nx_targets]
        df_x_test = df_training_data.loc[:int(rows * nx_validation_split), nx_features]
        df_y_test = df_training_data.loc[:int(rows * nx_validation_split), nx_targets]
        
        np_x_train = np.array(df_x_train, dtype='float64')
        np_y_train = np.array(df_y_train, dtype='float64')
        np_x_test = np.array(df_x_test, dtype='float64')
        np_y_test = np.array(df_y_test, dtype='float64')
        tf.keras.backend.set_floatx('float64')

        '''
        np_x_norm = np_x_train
        np_y_norm = np_y_train
        '''
        nx_normalize = nx.get_node_attributes(nx_graph, JSON_NORMALIZE_DATA)[node_i]
        if nx_normalize:
            print("Normalizing features")
            np_normalize = np.array(x_features, dtype='float64')
            normalize_x = preprocessing.Normalization()    
            normalize_x.adapt(np_normalize)
            np_x_train  = normalize_x(np_x_train)
            np_x_test  = normalize_x(np_x_test)
            #np_x_norm = np.array(x_normalized)
            '''
            print("Normalizing targets")
            normalize_y = preprocessing.Normalization()    
            normalize_y.adapt(np_y_train)
            y_normalized  = normalize_y(np_y_train)
            np_y_norm = np.array(y_normalized)
            '''
        else:
            print("features not normalized")
            print("targets not normalized")
        
        nx_batch = nx.get_node_attributes(nx_graph, JSON_BATCH)[node_i]
        nx_epochs = nx.get_node_attributes(nx_graph, JSON_EPOCHS)[node_i]
        nx_verbose = nx.get_node_attributes(nx_graph, JSON_VERBOSE)[node_i]
        nx_shuffle = nx.get_node_attributes(nx_graph, JSON_SHUFFLE_DATA)[node_i]
        fitting = k_model.fit(x=np_x_train, y=np_y_train, batch_size=nx_batch, epochs=nx_epochs, \
                                validation_data=(np_x_test, np_y_test), \
                                shuffle=nx_shuffle, \
                                verbose=nx_verbose)

    except Exception:
        err_txt = "*** An exception occurred training the model ***"
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- trainModels: done')
    logging.info('<---- ----------------------------------------------')    
    #return fitting
    return fitting, x_features, y_targets, np_x_train, np_y_train

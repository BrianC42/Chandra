'''
Created on Oct 9, 2020

@author: Brian
'''
import sys
import logging
import numpy as np
import networkx as nx
import tensorflow as tf

from configuration_constants import JSON_BATCH
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_SHUFFLE_DATA

#def trainModels(nx_graph, node_i, nx_edge, k_model, df_training_data, x_train,y_train, x_test, y_test):
def trainModel(d2r):
    logging.info('====> ================================================')
    logging.info('====> trainModels models')
    logging.info('====> ================================================')
    
    RANDOM_SEED = 42

    # error handling
    try:
        print("Training shapes X:%s y:%s, testing shapes X:%s y:%s" % (d2r.trainX.shape, d2r.trainY.shape, d2r.testX.shape, d2r.testY.shape))
                
        '''
        nx_regularization = nx.get_node_attributes(d2r.graph, JSON_REGULARIZATION)[d2r.mlNode]
        nx_reg_value = nx.get_node_attributes(d2r.graph, JSON_REG_VALUE)[d2r.mlNode]
        nx_bias = nx.get_node_attributes(d2r.graph, JSON_BIAS)[d2r.mlNode]
        nx_balanced = nx.get_node_attributes(d2r.graph, JSON_BALANCED)[d2r.mlNode]
        nx_analysis = nx.get_node_attributes(d2r.graph, JSON_ANALYSIS)[d2r.mlNode]
                
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

        nx_batch = nx.get_node_attributes(d2r.graph, JSON_BATCH)[d2r.mlNode]
        nx_epochs = nx.get_node_attributes(d2r.graph, JSON_EPOCHS)[d2r.mlNode]
        nx_verbose = nx.get_node_attributes(d2r.graph, JSON_VERBOSE)[d2r.mlNode]
        nx_shuffle = nx.get_node_attributes(d2r.graph, JSON_SHUFFLE_DATA)[d2r.mlNode]
        
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        
        d2r.fitting = d2r.model.fit(x=d2r.trainX, y=d2r.trainY, \
                              batch_size=nx_batch, epochs=nx_epochs, \
                              validation_data=(d2r.testX, d2r.testY), \
                              shuffle=nx_shuffle, \
                              verbose=nx_verbose)
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = "\n*** An exception occurred training the model ***" + "\n\t" + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- trainModels: done')
    logging.info('<---- ----------------------------------------------')    
    return

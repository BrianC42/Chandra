'''
Created on Oct 9, 2020

@author: Brian
'''
import sys
import logging
import networkx as nx

from configuration_constants import JSON_BATCH
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_MODEL_FILE

def trainModels(nx_graph, node_i, nx_edge, k_model, df_x_train, df_y_train, df_x_test, df_y_test):
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
        nx_batch = nx.get_node_attributes(nx_graph, JSON_BATCH)[node_i]
        nx_epochs = nx.get_node_attributes(nx_graph, JSON_EPOCHS)[node_i]
        nx_verbose = nx.get_node_attributes(nx_graph, JSON_VERBOSE)[node_i]
        fitting = k_model.fit(x=df_x_train, y=df_y_train, batch_size=nx_batch, epochs=nx_epochs, \
                                validation_data=(df_x_test, df_y_test), \
                                verbose=nx_verbose)

        nx_model_file = nx.get_node_attributes(nx_graph, JSON_MODEL_FILE)[node_i]
        k_model.save(nx_model_file)
    
    except Exception:
        err_txt = "*** An exception occurred training the model ***"
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- trainModels: done')
    logging.info('<---- ----------------------------------------------')    
    return

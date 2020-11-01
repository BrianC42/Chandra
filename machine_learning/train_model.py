'''
Created on Oct 9, 2020

@author: Brian
'''
import os
import sys
import logging
import networkx as nx
import tensorflow as tf
from tensorflow import keras
import pandas as pd

from configuration_constants import JSON_KERAS_DENSE_PROCESS
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_BATCH
'''
from configuration_constants import JSON_REGULARIZATION
from configuration_constants import JSON_REG_VALUE
from configuration_constants import JSON_BIAS
from configuration_constants import JSON_ANALYSIS
from configuration_constants import JSON_BALANCED
'''
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_DATA_FIELDS
from configuration_constants import JSON_CATEGORY_TYPE
from configuration_constants import JSON_CATEGORY_1HOT
from configuration_constants import JSON_CAT_TF
from configuration_constants import JSON_CAT_THRESHOLD
from configuration_constants import JSON_THRESHOLD_VALUE
from configuration_constants import JSON_MODEL_FILE

def trainModels(nx_graph, k_model):
    logging.info('====> ================================================')
    logging.info('====> trainModels models')
    logging.info('====> ================================================')

    # error handling
    try:
        # inputs                
        logging.debug("Training ML model")
        for node_i in nx_graph.nodes():
            nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == JSON_KERAS_DENSE_PROCESS:    
                print("Training: %s" % node_i)
                
                nx_input_flow = nx.get_node_attributes(nx_graph, JSON_INPUT_FLOWS)[node_i]
                nx_edges = nx.edges(nx_graph)
                for nx_edge in nx_edges:
                    if nx_edge[1] == node_i:
                        break
    
                nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_DATA_FIELDS)
                nx_input = nx_dataFields[nx_edge[0], nx_edge[1], nx_input_flow[0]]    

                nx_category1hot = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_1HOT)
                nx_categoryFields = nx_category1hot[nx_edge[0], nx_edge[1], nx_input_flow[0]]    

                nx_data_file = nx.get_edge_attributes(nx_graph, JSON_FLOW_DATA_FILE)
                inputData = nx_data_file[nx_edge[0], nx_edge[1], nx_input_flow[0]]    
                if os.path.isfile(inputData):
                    df_data = pd.read_csv(inputData)
                    rows = df_data.shape[0]
                    df_x_train = df_data.loc[int(rows/10):, nx_input]
                    df_y_train = df_data.loc[int(rows/10):, nx_categoryFields]
                    df_x_test = df_data.loc[:int(rows/10), nx_input]
                    df_y_test = df_data.loc[:int(rows/10), nx_categoryFields]
                
                nx_category_types = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_TYPE)
                category_type = nx_category_types[nx_edge[0], nx_edge[1], nx_input_flow[0]]
                if category_type == JSON_CAT_TF:
                    pass
                elif category_type == JSON_CAT_THRESHOLD:
                    pass
                elif category_type == JSON_THRESHOLD_VALUE:
                    pass
                else:
                    raise NameError('Invalid category type')

                nx_validation_split = nx.get_node_attributes(nx_graph, JSON_VALIDATION_SPLIT)[node_i]
                nx_batch = nx.get_node_attributes(nx_graph, JSON_BATCH)[node_i]
                nx_epochs = nx.get_node_attributes(nx_graph, JSON_EPOCHS)[node_i]
                nx_verbose = nx.get_node_attributes(nx_graph, JSON_VERBOSE)[node_i]
                '''
                nx_regularization = nx.get_node_attributes(nx_graph, JSON_REGULARIZATION)[node_i]
                nx_reg_value = nx.get_node_attributes(nx_graph, JSON_REG_VALUE)[node_i]
                nx_bias = nx.get_node_attributes(nx_graph, JSON_BIAS)[node_i]
                nx_balanced = nx.get_node_attributes(nx_graph, JSON_BALANCED)[node_i]
                nx_analysis = nx.get_node_attributes(nx_graph, JSON_ANALYSIS)[node_i]
                fitting = k_model.fit(x=df_x_train, y=df_y_train, batch_size=nx_batch, epochs=nx_epochs, \
                                      validation_data=(df_x_test, df_y_test), \
                                      verbose=nx_verbose, validation_split=nx_validation_split)
                '''
                fitting = k_model.fit(x=df_x_train, y=df_y_train, batch_size=nx_batch, epochs=nx_epochs, \
                                      validation_data=(df_x_test, df_y_test), \
                                      verbose=nx_verbose)

                nx_model_file = nx.get_node_attributes(nx_graph, JSON_MODEL_FILE)[node_i]
                k_model.summary()
                keras.utils.plot_model(k_model, to_file=nx_model_file + '.png', show_shapes=True)
                k_model.save(nx_model_file)
    
    except Exception:
        err_txt = "*** An exception occurred training the model ***"
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- trainModels: done')
    logging.info('<---- ----------------------------------------------')    
    return

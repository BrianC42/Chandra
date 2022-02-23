'''
Created on Oct 9, 2020

@author: Brian
'''
import sys
import logging
import networkx as nx
import numpy as np
import tensorflow as tf
import autokeras as ak

from configuration_constants import JSON_TRAIN
from configuration_constants import JSON_BATCH
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_SHUFFLE_DATA

def trainModel(d2r):
    logging.info('====> ================================================')
    logging.info('====> trainModels models')
    logging.info('====> ================================================')
    
    try:
        print("\nTraining shapes x:%s y:%s, testing shapes x:%s y:%s" % (d2r.trainX.shape, d2r.trainY.shape, d2r.testX.shape, d2r.testY.shape))
        
        nx_train = nx.get_node_attributes(d2r.graph, JSON_TRAIN)[d2r.mlNode]
        nx_batch = nx.get_node_attributes(d2r.graph, JSON_BATCH)[d2r.mlNode]
        nx_epochs = nx.get_node_attributes(d2r.graph, JSON_EPOCHS)[d2r.mlNode]
        nx_verbose = nx.get_node_attributes(d2r.graph, JSON_VERBOSE)[d2r.mlNode]
        nx_shuffle = nx.get_node_attributes(d2r.graph, JSON_SHUFFLE_DATA)[d2r.mlNode]
        
        d2r.fitting = d2r.model.fit(x=d2r.trainX, y=d2r.trainY, \
                                    validation_data=(d2r.validateX, d2r.validateY), \
                                    #validation_split=0.1, \
                                    epochs=nx_epochs, \
                                    batch_size=nx_batch, \
                                    shuffle=nx_shuffle, \
                                    verbose=nx_verbose)
    
    except Exception:
        err_txt = "\n*** An exception occurred training the model ***\n\t"
        
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" + exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += " "
                exc_txt += s

        logging.debug(exc_txt)
        sys.exit(exc_txt)        
    
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- trainModels: done')
    logging.info('<---- ----------------------------------------------')    
    return

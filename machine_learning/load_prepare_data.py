'''
Created on Apr 7, 2020

@author: Brian
'''
import logging
import networkx as nx

def load_and_prepare_data(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    # symbols
    logging.debug("Symbols")
    #logging.debug("%s" % json_config['symbols'])
    for i, node_i in enumerate(nx_graph):
        logging.debug("\t%s\t%s" % (i, node_i))

    logging.info('<---- ----------------------------------------------')
    logging.info('<---- load_and_prepare_data: done')
    logging.info('<---- ----------------------------------------------')    
    return
'''
Created on Apr 7, 2020

@author: Brian
'''
import subprocess
import os
import logging
import networkx as nx
from networkx.drawing import nx_agraph

from configuration_constants import JSON_PROCESS_NODES

from configuration_constants import JSON_REQUIRED
from configuration_constants import JSON_CONDITIONAL

from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_KERAS_DENSE_PROCESS

from configuration_constants import JSON_NODE_NAME
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_DATA_FLOWS
from configuration_constants import JSON_FLOW_NAME
from configuration_constants import JSON_FLOW_FROM
from configuration_constants import JSON_FLOW_TO
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_OUTPUT_FILE
from configuration_constants import JSON_LOG_FILE

from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_INPUT_DATA_PREPARATION

from configuration_constants import JSON_KERAS_DENSE_CTRL
from configuration_constants import JSON_MODEL_FILE

from configuration_constants import JSON_BALANCED
from configuration_constants import JSON_TIME_SEQ
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_DATA_FIELDS
from configuration_constants import JSON_TARGET_FIELD

from configuration_constants import JSON_NODE_COUNT

from configuration_constants import JSON_TIMESTEPS
from configuration_constants import JSON_BATCH
from configuration_constants import JSON_REGULARIZATION
from configuration_constants import JSON_REG_VALUE
from configuration_constants import JSON_DROPOUT
from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_BIAS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_LOSS_WTS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_ACTIVATION
from configuration_constants import JSON_OPTIMIZER
from configuration_constants import JSON_ANALYSIS

def load_and_prepare_data(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    # symbols
    cmdstr = "python d:\\brian\\git\\chandra\\unit_test\\external.py"
    pathin = "--pathin p1"
    pathout = "--pathout p5"

    for node_i in nx_graph.nodes():
        nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
        if nx_read_attr[node_i] == JSON_DATA_PREP_PROCESS:
            nx_data_file = nx.get_node_attributes(nx_graph, JSON_INPUT_DATA_FILE)
            print("load and prepare data using %s json parameters and file %s" % (node_i, nx_data_file[node_i]))
            nx_data_flow = nx.get_node_attributes(nx_graph, JSON_OUTPUT_FLOW)
            output_flow = nx_data_flow[node_i]
            for edge_i in nx_graph.edges():
                if edge_i[0] == node_i:
                    nx_balanced = nx.get_edge_attributes(nx_graph, JSON_BALANCED)
                    balanced = nx_balanced[edge_i[0], edge_i[1], output_flow]
                    nx_timeSeq = nx.get_edge_attributes(nx_graph, JSON_TIME_SEQ)
                    timeSeq = nx_timeSeq[edge_i[0], edge_i[1], output_flow]
                    nx_ignoreBlanks = nx.get_edge_attributes(nx_graph, JSON_IGNORE_BLANKS)
                    ignoreBlanks = nx_ignoreBlanks[edge_i[0], edge_i[1], output_flow]
                    nx_flowFilename = nx.get_edge_attributes(nx_graph, JSON_FLOW_DATA_FILE)
                    flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
                    nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_DATA_FIELDS)
                    dataFields = nx_dataFields[edge_i[0], edge_i[1], output_flow]
                    nx_targetFields = nx.get_edge_attributes(nx_graph, JSON_TARGET_FIELD)
                    targetFields = nx_targetFields[edge_i[0], edge_i[1], output_flow]
                    print(edge_i)
        
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- load_and_prepare_data: done')
    logging.info('<---- ----------------------------------------------')    
    return
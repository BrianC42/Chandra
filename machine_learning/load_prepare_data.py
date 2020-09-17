'''
Created on Apr 7, 2020

@author: Brian
'''
import subprocess
import os
import logging
import networkx as nx
from networkx.drawing import nx_agraph

def load_and_prepare_data(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    # symbols
    logging.debug("Symbols")
    cmdstr = "python d:\\brian\\git\\chandra\\unit_test\\external.py"
    pathin = "--pathin p1"
    pathout = "--pathout p5"
    #logging.debug("%s" % json_config['symbols'])
    #for ndx, node_i in enumerate(nx_graph.nodes()):
    nx_attributes = nx.get_node_attributes(nx_graph, "inputFlows")
    for node_i in nx_graph.nodes():
        print("\nNode: %s" % node_i)
        nx_input = nx_attributes[node_i]
        print("inputFlows: %s" % nx_input)
        '''
        if nx_graph.node[ndx]['inputType'] == "localFile":
            file1 = "--file f1"
            files = file1
            field1 = "--field " + str(ndx)
            field2 = "--field " + str(node_i)
            fields = field1 + " " + field2
            output = "--output p6"
            cmd_line = pathin + " " +  files + " " +  fields + " " +  pathout + " " +  output
            subprocess.run(cmdstr + " " +  cmd_line)
            logging.debug("\t%s\t%s" % (ndx, node_i))
        '''

    nx_edges = nx_graph.edges()
    print("\nEdges: %s" % nx_edges)
    nx_time_seq = nx.get_edge_attributes(nx_graph, "dummyAttr")
    nx_ts_ndxs = list(nx_time_seq)
    print("nx_time_seq %s" % nx_time_seq)
    ndx = 0
    for edge_i in nx_edges:
        print("edge_i %s %s, %s" % (edge_i[0], edge_i[1], len(edge_i)))
        nx_key = nx_ts_ndxs[ndx]
        nx_edge_attr = nx_time_seq[nx_key]
        print("edge %s attributes %s" % (nx_key, nx_edge_attr))
        ndx += 1

    logging.info('<---- ----------------------------------------------')
    logging.info('<---- load_and_prepare_data: done')
    logging.info('<---- ----------------------------------------------')    
    return
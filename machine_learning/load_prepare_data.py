'''
Created on Apr 7, 2020

@author: Brian
'''
import os
import glob
import logging
import networkx as nx
import pandas as pd

from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_BALANCED
from configuration_constants import JSON_TIME_SEQ
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_DATA_FIELDS
from configuration_constants import JSON_CATEGORY_TYPE
from configuration_constants import JSON_CATEGORY_FIELD
from configuration_constants import JSON_CATEGORY_1HOT
from configuration_constants import JSON_CAT_TF
from configuration_constants import JSON_CAT_THRESHOLD
from configuration_constants import JSON_THRESHOLD_VALUE

def set_1Hot_TF(df_out, categoryFields, category1Hot):
    #ndxList = df_out.index
    for ndx in df_out.index:
        if df_out.loc[ndx, categoryFields]:
            df_out.loc[ndx, category1Hot[0]] = 1
        else:
            df_out.loc[ndx, category1Hot[1]] = 1
    return

def set_1Hot_Threshold():
    return 

def set_1Hot_Ranges():
    return 

def prepare2dTrainingData(nx_graph, node_name):
    nx_data_file = nx.get_node_attributes(nx_graph, JSON_INPUT_DATA_FILE)
    #print("load and prepare data using %s json parameters and file %s" % (node_name, nx_data_file[node_name]))
    nx_data_flow = nx.get_node_attributes(nx_graph, JSON_OUTPUT_FLOW)
    output_flow = nx_data_flow[node_name]
    for edge_i in nx_graph.edges():
        if edge_i[0] == node_name:
            nx_balanced = nx.get_edge_attributes(nx_graph, JSON_BALANCED)
            nx_timeSeq = nx.get_edge_attributes(nx_graph, JSON_TIME_SEQ)
            nx_ignoreBlanks = nx.get_edge_attributes(nx_graph, JSON_IGNORE_BLANKS)
            nx_flowFilename = nx.get_edge_attributes(nx_graph, JSON_FLOW_DATA_FILE)
            nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_DATA_FIELDS)
            nx_categoryType = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_TYPE)
            nx_categoryFields = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_FIELD)
            nx_category1Hot = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_1HOT)
                    
            balanced = nx_balanced[edge_i[0], edge_i[1], output_flow]
            timeSeq = nx_timeSeq[edge_i[0], edge_i[1], output_flow]
            ignoreBlanks = nx_ignoreBlanks[edge_i[0], edge_i[1], output_flow]
            flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
            dataFields = nx_dataFields[edge_i[0], edge_i[1], output_flow]
            categoryType = nx_categoryType[edge_i[0], edge_i[1], output_flow]
            categoryFields = nx_categoryFields[edge_i[0], edge_i[1], output_flow]
            category1Hot = nx_category1Hot[edge_i[0], edge_i[1], output_flow]
            
            df_combined = pd.DataFrame()

            for dataFile in nx_data_file[node_name]:
                fileSpecList = glob.glob(dataFile)
                for FileSpec in fileSpecList:
                    if os.path.isfile(FileSpec):
                        df_data = pd.read_csv(FileSpec)
                        
                        l_filter = []
                        for fld in dataFields:
                            l_filter.append(fld)
                        l_filter.append(categoryFields)
                        df_inputs = df_data.filter(l_filter)
                        
                        if ignoreBlanks:
                            df_inputs = df_inputs.dropna()
                            
                        '''
                        if balanced:
                            print("Balanced data")
                        '''
                            
                        for oneHot in category1Hot:
                            df_inputs.insert(len(df_inputs.columns), oneHot, 0)
                    
                        if categoryType == JSON_CAT_TF:
                            set_1Hot_TF(df_inputs, categoryFields, category1Hot)
                        else:
                            raise NameError('Invalid category type')
                        '''
                        elif categoryType == JSON_CAT_THRESHOLD:
                            print("Threshold category type - not yet implemented")
                        elif categoryType == JSON_THRESHOLD_VALUE:
                            print("Threshold value type - not yet implemented")
                        '''
                
                        df_inputs.drop(categoryFields, axis=1)
                        df_combined = pd.concat([df_combined, df_inputs], ignore_index=True)
                    else:
                        raise NameError('Data file does not exist')
            df_combined.to_csv(flowFilename, index=False)
    return

def load_and_prepare_data(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    for node_i in nx_graph.nodes():
        nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
        if nx_read_attr[node_i] == JSON_DATA_PREP_PROCESS:
            prepare2dTrainingData(nx_graph, node_i)
        
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- load_and_prepare_data: done')
    logging.info('<---- ----------------------------------------------')    
    return
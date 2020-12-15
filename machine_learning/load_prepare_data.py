'''
Created on Apr 7, 2020

@author: Brian
'''
import sys
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

from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS

from configuration_constants import JSON_CATEGORY_TYPE
from configuration_constants import JSON_CATEGORY_1HOT
from configuration_constants import JSON_CAT_TF
from configuration_constants import JSON_CAT_THRESHOLD
from configuration_constants import JSON_THRESHOLD_VALUE
from configuration_constants import JSON_LINEAR_REGRESSION

from configuration_constants import JSON_REMOVE_OUTLIER_LIST
from configuration_constants import JSON_OUTLIER_FEATURE
from configuration_constants import JSON_OUTLIER_PCT

def discard_outliers(nx_graph, node_i, df_data):
    nx_remove_outlier_list = nx.get_node_attributes(nx_graph, JSON_REMOVE_OUTLIER_LIST)
    for feature in nx_remove_outlier_list[node_i]:
        featureName = feature[JSON_OUTLIER_FEATURE]
        outlierPct = feature[JSON_OUTLIER_PCT]
        print("Removing outliers (highest and lowest %s%%) from %s" % (outlierPct, featureName))
        
    return

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
            nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_FEATURE_FIELDS)
            nx_categoryType = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_TYPE)
            nx_targetFields = nx.get_edge_attributes(nx_graph, JSON_TARGET_FIELDS)
                    
            featureFields = nx_dataFields[edge_i[0], edge_i[1], output_flow]
            targetFields = nx_targetFields[edge_i[0], edge_i[1], output_flow]
            categoryType = nx_categoryType[edge_i[0], edge_i[1], output_flow]
            balanced = nx_balanced[edge_i[0], edge_i[1], output_flow]
            timeSeq = nx_timeSeq[edge_i[0], edge_i[1], output_flow]
            ignoreBlanks = nx_ignoreBlanks[edge_i[0], edge_i[1], output_flow]
            
            df_combined = pd.DataFrame()

            for dataFile in nx_data_file[node_name]:
                fileSpecList = glob.glob(dataFile)
                for FileSpec in fileSpecList:
                    if os.path.isfile(FileSpec):
                        df_data = pd.read_csv(FileSpec)
                        
                        l_filter = []
                        for fld in featureFields:
                            l_filter.append(fld)
                        for fld in targetFields:
                            l_filter.append(fld)
                        df_inputs = df_data.filter(l_filter)
                        df_combined = pd.concat([df_combined, df_inputs], ignore_index=True)
                    else:
                        raise NameError('Data file does not exist')
            print("\nData \n%s\nread from sources\n" % df_combined.describe())
                        
            if ignoreBlanks:
                print("Removing NaN")
                df_combined = df_combined.dropna()
                
            discard_outliers(nx_graph, node_name, df_combined)
                            
            if categoryType == JSON_CAT_TF:
                nx_category1Hot = nx.get_edge_attributes(nx_graph, JSON_CATEGORY_1HOT)
                category1Hot = nx_category1Hot[edge_i[0], edge_i[1], output_flow]
                for oneHot in category1Hot:
                    df_combined.insert(len(df_combined.columns), oneHot, 0)
                set_1Hot_TF(df_combined, targetFields, category1Hot)
            elif categoryType == JSON_LINEAR_REGRESSION:
                pass
            elif categoryType == JSON_CAT_THRESHOLD:
                print("Threshold category type - not yet implemented")
                raise NameError('Category type not yet implemented')
            elif categoryType == JSON_THRESHOLD_VALUE:
                print("Threshold value type - not yet implemented")
                raise NameError('Category type not yet implemented')
            else:
                raise NameError('Invalid category type')
                
            df_combined.drop(targetFields, axis=1)
    return df_combined

def load_and_prepare_data(nx_graph):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    # error handling
    try:

        nx_data_flow = nx.get_node_attributes(nx_graph, JSON_OUTPUT_FLOW)
        nx_flowFilename = nx.get_edge_attributes(nx_graph, JSON_FLOW_DATA_FILE)
        nx_read_attr = nx.get_node_attributes(nx_graph, JSON_PROCESS_TYPE)
        for node_i in nx_graph.nodes():
            output_flow = nx_data_flow[node_i]
            if nx_read_attr[node_i] == JSON_DATA_PREP_PROCESS:
                df_data = prepare2dTrainingData(nx_graph, node_i)
                for edge_i in nx_graph.edges():
                    if edge_i[0] == node_i:
                        flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
                        print("\nData \n%s\nwritten to %s for training\n" % (df_data.describe(), flowFilename))
                        df_data.to_csv(flowFilename, index=False)
                        break
        
    except Exception:
        err_txt = "*** An exception occurred analyzing the json configuration file ***"
        logging.debug(err_txt)
        sys.exit("\n" + err_txt)
        
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- load_and_prepare_data: done')
    logging.info('<---- ----------------------------------------------')    
    return
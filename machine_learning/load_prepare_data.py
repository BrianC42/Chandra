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
import numpy as np
import tensorflow as tf

from tfWindowGenerator import WindowGenerator

from configuration_constants import JSON_PRECISION
from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_NORMALIZE_DATA
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import MODEL_TYPE
from configuration_constants import INPUT_LAYERTYPE_DENSE
from configuration_constants import INPUT_LAYERTYPE_RNN
from configuration_constants import INPUT_LAYERTYPE_CNN
from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_TEST_SPLIT
from configuration_constants import JSON_TIMESTEPS
'''
from configuration_constants import JSON_REMOVE_OUTLIER_LIST
from configuration_constants import JSON_OUTLIER_FEATURE
from configuration_constants import JSON_OUTLIER_PCT
'''

def loadTrainingData(d2r):
    '''
    load data for training and testine
    '''
    # error handling
    try:
        err_txt = "*** An exception occurred preparing the training data for the model ***"

        nx_input_flow = nx.get_node_attributes(d2r.graph, JSON_INPUT_FLOWS)[d2r.mlNode]
        print("Loading prepared data defined in flow: %s" % nx_input_flow[0])
        nx_data_file = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
        inputData = nx_data_file[d2r.mlEdgeIn[0], d2r.mlEdgeIn[1], nx_input_flow[0]]    
        if os.path.isfile(inputData):
            df_training_data = pd.read_csv(inputData)
        d2r.data = df_training_data
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" + exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += s
        logging.debug(exc_txt)
        sys.exit(exc_txt)

    return

def organize_data_for_convolution(nx_graph, node_i, df_training_data):
    print("Organizing data for convolution\n%s\n" % df_training_data.describe().transpose())

    nx_validation_split = nx.get_node_attributes(nx_graph, JSON_VALIDATION_SPLIT)[node_i]
    nx_test_split = nx.get_node_attributes(nx_graph, JSON_TEST_SPLIT)[node_i]

    n = len(df_training_data)
    
    train_start = 0
    train_len = int(n * (1 - (nx_validation_split + nx_test_split)))
    
    validate_start = train_start + train_len
    validate_len = int(n * nx_validation_split)
    
    test_start = train_start + train_len + validate_len
    
    df_train = df_training_data[train_start : (train_start + train_len)]
    df_values = df_training_data[validate_start : (validate_start + validate_len)]
    df_test = df_training_data[test_start :]
    
    print("**********************************\nWindow generator called with hard coded parameters. Need to be parameterized from json\n***************************")
    trainingWindow = WindowGenerator(input_width=24, label_width=1, shift=24, \
                         df_train=df_train, df_values=df_values, df_test=df_test, \
                         label_columns=['T (degC)'])
    print(trainingWindow)
    return trainingWindow

def to_sequences(x, y, seq_size=1):
    '''
    create 3 dimensional dataframes for RNN input layers
    '''
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])

    return np.array(x_values), np.array(y_values)

def prepareData(d2r):
    '''
    Organize the feature data elements as required by Tensorflow models and matching target elements
    '''
    nx_input_flows = nx.get_node_attributes(d2r.graph, JSON_INPUT_FLOWS)[d2r.mlNode]

    nx_featureFields = nx.get_edge_attributes(d2r.graph, JSON_FEATURE_FIELDS)
    nx_features = nx_featureFields[d2r.mlEdgeIn[0], d2r.mlEdgeIn[1], nx_input_flows[0]]    

    nx_targetFiields = nx.get_edge_attributes(d2r.graph, JSON_TARGET_FIELDS)
    nx_targets = nx_targetFiields[d2r.mlEdgeIn[0], d2r.mlEdgeIn[1], nx_input_flows[0]]    

    nx_test_split = nx.get_node_attributes(d2r.graph, JSON_TEST_SPLIT)[d2r.mlNode]
    nx_validation_split = nx.get_node_attributes(d2r.graph, JSON_VALIDATION_SPLIT)[d2r.mlNode]

    nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
    if nx_read_attr[d2r.mlNode] == JSON_TENSORFLOW:    
        print("Preparing the data for training: %s" % d2r.mlNode)
            
        nx_data_precision = nx.get_node_attributes(d2r.graph, JSON_PRECISION)[d2r.mlNode]
        '''
        nx_normalize = nx.get_node_attributes(d2r.graph, JSON_NORMALIZE_DATA)[d2r.mlNode]
        if nx_normalize:
            print("\n*************************************************\nWORK IN PROGRESS\n\tNormalization is not implemented\n*************************************************\n")
        '''
        train = d2r.data.loc[ : (len(d2r.data) * (1-nx_test_split))]
        test = d2r.data.loc[(len(d2r.data) * (1-nx_test_split)) :]
        
        nx_model_type = nx.get_node_attributes(d2r.graph, MODEL_TYPE)[d2r.mlNode]
        if nx_model_type == INPUT_LAYERTYPE_DENSE:
            df_x_train = train[nx_features]
            df_y_train = train[nx_targets[0]]
            df_x_test = test[nx_features]
            df_y_test = test[nx_targets[0]]
        elif nx_model_type == INPUT_LAYERTYPE_RNN:
            nx_time_steps = nx.get_node_attributes(d2r.graph, JSON_TIMESTEPS)[d2r.mlNode]
            df_x_train, df_y_train = to_sequences(train[nx_features],train[nx_targets[0]], nx_time_steps)
            df_x_test, df_y_test = to_sequences(test[nx_features], test[nx_targets[0]], nx_time_steps)        
        elif nx_model_type == INPUT_LAYERTYPE_CNN:
            print("\n*************************************************\nWORK IN PROGRESS\n\tCNN preparation is not implemented\n*************************************************\n")
            pass

        d2r.trainX = np.array(df_x_train, dtype=nx_data_precision)
        d2r.trainY = np.array(df_y_train, dtype=nx_data_precision)
        d2r.testX = np.array(df_x_test, dtype=nx_data_precision)
        d2r.testY = np.array(df_y_test, dtype=nx_data_precision)
    return

'''
def discard_outliers(nx_graph, node_i, df_data):
    nx_remove_outlier_list = nx.get_node_attributes(nx_graph, JSON_REMOVE_OUTLIER_LIST)
    for feature in nx_remove_outlier_list[node_i]:
        if JSON_OUTLIER_FEATURE in feature:
            featureName = feature[JSON_OUTLIER_FEATURE]
            outlierPct = feature[JSON_OUTLIER_PCT]
            print("Removing outliers (highest and lowest %s%%) from %s" % (outlierPct * 100, featureName))        
            rows = df_data.shape[0]
            df_data = df_data.sort_values(by = featureName)
            df_data = df_data[int(rows * outlierPct) : (rows - int(rows * outlierPct))]
        
    return df_data
'''

def set_1Hot_TF(df_out, categoryFields, category1Hot):
    #ndxList = df_out.index
    for ndx in df_out.index:
        if df_out.loc[ndx, categoryFields]:
            df_out.loc[ndx, category1Hot[0]] = 1
        else:
            df_out.loc[ndx, category1Hot[1]] = 1
    return

def prepareTrainingData(nx_graph, node_name, nx_edge):
    '''
    Select required data elements and discard the rest
    '''
    nx_data_file = nx.get_node_attributes(nx_graph, JSON_INPUT_DATA_FILE)
    #print("load and prepare data using %s json parameters and file %s" % (node_name, nx_data_file[node_name]))
    nx_data_flow = nx.get_node_attributes(nx_graph, JSON_OUTPUT_FLOW)
    output_flow = nx_data_flow[node_name]
    nx_ignoreBlanks = nx.get_edge_attributes(nx_graph, JSON_IGNORE_BLANKS)
    nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_FEATURE_FIELDS)
    nx_targetFields = nx.get_edge_attributes(nx_graph, JSON_TARGET_FIELDS)
    featureFields = nx_dataFields[nx_edge[0], nx_edge[1], output_flow]
    targetFields = nx_targetFields[nx_edge[0], nx_edge[1], output_flow]
    ignoreBlanks = nx_ignoreBlanks[nx_edge[0], nx_edge[1], output_flow]
            
    df_combined = pd.DataFrame()

    for dataFile in nx_data_file[node_name]:
        fileSpecList = glob.glob(dataFile)
        fileCount = len(fileSpecList)
        tf_progbar = tf.keras.utils.Progbar(fileCount, width=50, verbose=1, interval=1, stateful_metrics=None, unit_name='file')
        count = 0
        for FileSpec in fileSpecList:
            if os.path.isfile(FileSpec):
                tf_progbar.update(count)
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
            count += 1
    print("\nData \n%s\nread from sources\n" % df_combined.describe().transpose())
                        
    if ignoreBlanks:
        print("Removing NaN")
        df_combined = df_combined.dropna()
                
    #df_combined = discard_outliers(nx_graph, node_name, df_combined)                            
    df_combined.drop(targetFields, axis=1)
    return df_combined

def collect_and_select_data(d2r):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    # error handling
    try:
        err_txt = "*** An exception occurred collecting and selecting the data ***"

        for node_i in d2r.graph.nodes():
            nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == JSON_DATA_PREP_PROCESS:
                nx_data_flow = nx.get_node_attributes(d2r.graph, JSON_OUTPUT_FLOW)
                output_flow = nx_data_flow[node_i]
                for edge_i in d2r.graph.edges():
                    if edge_i[0] == node_i:
                        err_txt = "*** An exception occurred analyzing the flow details in the json configuration file ***"
                        nx_flowFilename = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
                        flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
                        d2r.data = prepareTrainingData(d2r.graph, node_i, edge_i)
                        print(d2r.data.describe().transpose())
                        d2r.archiveData(flowFilename)
                        break
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" + exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += " " + s
        logging.debug(exc_txt)
        sys.exit(exc_txt)
        
    logging.info('<---- ----------------------------------------------')
    logging.info('<---- load_and_prepare_data: done')
    logging.info('<---- ----------------------------------------------')    
    return
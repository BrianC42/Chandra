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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from termcolor import colored

from configuration_constants import JSON_PRECISION
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_NORMALIZE_DATA
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_DATA_LOAD_PROCESS
from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_TEST_SPLIT
from configuration_constants import JSON_TIMESTEPS
from configuration_constants import JSON_1HOT_ENCODING
from configuration_constants import JSON_1HOT_FIELD
from configuration_constants import JSON_1HOT_CATEGORYTYPE
from configuration_constants import JSON_1HOT_SERIESTREND
from configuration_constants import JSON_1HOT_CATEGORIES
from configuration_constants import JSON_1HOT_OUTPUTFIELDS

from TrainingDataAndResults import MODEL_TYPE
from TrainingDataAndResults import INPUT_LAYERTYPE_DENSE
from TrainingDataAndResults import INPUT_LAYERTYPE_RNN
from TrainingDataAndResults import INPUT_LAYERTYPE_CNN


from pickle import FALSE

def prepareTrainingData(d2r):
    print("============== WIP =============\n\tPreparing data\n================================")
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred preparing the training data ***"

        for node_i in d2r.graph.nodes():
            nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == JSON_DATA_PREP_PROCESS:
                #print("Preparing data in %s" % node_i)
            
                nx_outputFlow = nx.get_node_attributes(d2r.graph, JSON_OUTPUT_FLOW)
                output_flow = nx_outputFlow[node_i]

                nx_inputFlows = nx.get_node_attributes(d2r.graph, JSON_INPUT_FLOWS)
                for input_flow in nx_inputFlows[node_i]:
                    for edge_i in d2r.graph.edges():
                        if edge_i[0] == node_i:
                            nx_output_data_files = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
                            nx_output_data_file = nx_output_data_files[edge_i[0], edge_i[1], output_flow]
                            '''
                            print("from %s to %s\n\tflow %s\n\tfile %s" % \
                                  (edge_i[0], edge_i[1], output_flow, nx_output_data_file))
                            '''
                        if edge_i[1] == node_i:
                            nx_input_data_files = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
                            nx_input_data_file = nx_input_data_files[edge_i[0], edge_i[1], input_flow]    
                            '''
                            print("from %s to %s\n\tflow %s\n\tfile %s" % \
                                  (edge_i[0], edge_i[1], input_flow, nx_input_data_file))
                            '''
                if os.path.isfile(nx_input_data_file):
                    d2r.data = pd.read_csv(nx_input_data_file)
                else:
                    ex_txt = node_i + ", input file " + nx_input_data_file + " does not exist"
                    raise NameError(ex_txt)
                
                nx_1hotConfig = nx.get_node_attributes(d2r.graph, JSON_1HOT_ENCODING)[node_i]
                nx_1hotCategoryType = nx_1hotConfig[JSON_1HOT_CATEGORYTYPE]
                if nx_1hotCategoryType == JSON_1HOT_SERIESTREND:
                    nx_1hotField = nx_1hotConfig[JSON_1HOT_FIELD]
                    nx_1hotCategories = nx_1hotConfig[JSON_1HOT_CATEGORIES]
                    nx_1hotOutputFields = nx_1hotConfig[JSON_1HOT_OUTPUTFIELDS]
                    print("Encoding %s as 1hot %s" % (nx_1hotField, JSON_1HOT_SERIESTREND))
                    print(nx_1hotCategories)
                    print(nx_1hotOutputFields)
                else:
                    ex_txt = d2r.mlNode + ", prep type " + nx_read_attr[d2r.mlNode] + " is not supported"
                    raise NameError(ex_txt)

                d2r.archiveData(nx_output_data_file)
        
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

    return

def loadTrainingData(d2r):
    '''
    load data for training and testing
    '''
    # error handling
    try:
        err_txt = "*** An exception occurred preparing the training data for the model ***"

        nx_input_flow = nx.get_node_attributes(d2r.graph, JSON_INPUT_FLOWS)[d2r.mlNode]
        print("Loading prepared data defined in flow: %s" % nx_input_flow[0])
        nx_data_file = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
        inputData = nx_data_file[d2r.mlEdgeIn[0], d2r.mlEdgeIn[1], nx_input_flow[0]]    
        if os.path.isfile(inputData):
            d2r.data = pd.read_csv(inputData)
        
    except Exception:
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

    return

'''
def to_sequences(x, y, seq_size=1):
    return dataframe inputs as numpy arrays of shapes
    x: (samples, seq_size, 1)
    y: (samples, )
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])

    return np.array(x_values), np.array(y_values)
'''

def id_columns(data, features, targets):
    feature_cols = []
    target_cols= []
    
    ndx = 0
    for col in data.columns:
        if col in features:
            feature_cols.append(ndx)
        if col in targets:
            target_cols.append(ndx)
        ndx += 1
    
    return feature_cols, target_cols

def np_to_sequence(data, sequence_step_id, targets, seq_size=1):
    npx = np.empty([len(data) - (seq_size+1), seq_size, len(targets)], dtype=float)
    npy = np.empty([len(data) - (seq_size+1),                       ], dtype=float)

    for i in range(len(data) - (seq_size+1)):
        npx[i, :, :]    = data[i : i+seq_size, targets[:]]
        npy[i]          = data[i+seq_size+1, targets[0]]

    return npx, npy

def arrangeDataForTraining(d2r):
    '''
    Organize the feature data elements as required by Tensorflow models and matching target elements
    '''
    nx_input_flows = nx.get_node_attributes(d2r.graph, JSON_INPUT_FLOWS)[d2r.mlNode]

    nx_featureFields = nx.get_edge_attributes(d2r.graph, JSON_FEATURE_FIELDS)
    nx_features = nx_featureFields[d2r.mlEdgeIn[0], d2r.mlEdgeIn[1], nx_input_flows[0]]    

    nx_targetFields = nx.get_edge_attributes(d2r.graph, JSON_TARGET_FIELDS)
    nx_targets = nx_targetFields[d2r.mlEdgeIn[0], d2r.mlEdgeIn[1], nx_input_flows[0]]    

    feature_cols, target_cols = id_columns(d2r.data, nx_features, nx_targets)

    nx_test_split = nx.get_node_attributes(d2r.graph, JSON_TEST_SPLIT)[d2r.mlNode]
    nx_validation_split = nx.get_node_attributes(d2r.graph, JSON_VALIDATION_SPLIT)[d2r.mlNode]

    d2r.normalized = False
    
    nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
    if nx_read_attr[d2r.mlNode] == JSON_TENSORFLOW:    
        print("Preparing the data for training: %s" % d2r.mlNode)
        nx_data_precision = nx.get_node_attributes(d2r.graph, JSON_PRECISION)[d2r.mlNode]
        
        len_train = int(len(d2r.data) * (1-(nx_test_split+nx_validation_split)))
        len_validate = int(len(d2r.data) * nx_validation_split)
        len_test = int(len(d2r.data) * nx_test_split)
        
        nx_normalize = nx.get_node_attributes(d2r.graph, JSON_NORMALIZE_DATA)[d2r.mlNode]

        nx_model_type = nx.get_node_attributes(d2r.graph, MODEL_TYPE)[d2r.mlNode]
        if nx_model_type == INPUT_LAYERTYPE_DENSE:
            if nx_normalize == 'standard':
                print("========= WIP ===========\n\tnormalization - standard\n\tnot implemented\n=========================")
            elif nx_normalize == 'minmax':
                print("========= WIP ===========\n\tnormalization - minmax\n\tnot implemented\n=========================")
            elif nx_normalize == 'none':
                print("\nData is not normalized ...")
            else:
                ex_txt = 'Normalization {:s} is not supported'.format(nx_normalize)
                raise NameError(ex_txt)

            train       = d2r.data.loc[                         : len_train]
            validation  = d2r.data.loc[len_train                : (len_train + len_validate)]
            test        = d2r.data.loc[len_train + len_validate : ]
            
            d2r.trainX = train[nx_features[0]]
            d2r.trainY = train[nx_targets[0]]
            d2r.validateX = validation[nx_features[0]]
            d2r.validateY = validation[nx_targets[0]]
            d2r.testX = test[nx_features[0]]
            d2r.testY = test[nx_targets[0]]
        elif nx_model_type == INPUT_LAYERTYPE_RNN:
            nx_time_steps = nx.get_node_attributes(d2r.graph, JSON_TIMESTEPS)[d2r.mlNode]
            
            if nx_normalize == 'standard':
                print("\n\tnormalization - standard")
                d2r.scaler = StandardScaler()
                d2r.scaler = d2r.scaler.fit(d2r.data)

                data = d2r.scaler.transform(d2r.data)
                d2r.normalized = True
            elif nx_normalize == 'minmax':
                print("\n\tnormalization - minmax")
                d2r.scaler = MinMaxScaler()
                d2r.scaler = d2r.scaler.fit(d2r.data)

                data = d2r.scaler.transform(d2r.data)
                d2r.normalized = True
            elif nx_normalize == 'none':
                print("\nData is not normalized ...")
                data = np.array(d2r.data, dtype=nx_data_precision)
            else:
                ex_txt = 'Normalization {:s} is not supported'.format(nx_normalize)
                raise NameError(ex_txt)

            train       = data[                         : len_train]
            validation  = data[len_train                : len_train + len_validate]
            test        = data[len_train + len_validate : ]

            d2r.trainX,    d2r.trainY    = np_to_sequence(train, feature_cols, target_cols, nx_time_steps)
            d2r.validateX, d2r.validateY = np_to_sequence(validation, feature_cols, target_cols, nx_time_steps)
            d2r.testX,     d2r.testY     = np_to_sequence(test, feature_cols, target_cols, nx_time_steps)        
        elif nx_model_type == INPUT_LAYERTYPE_CNN:
            print("\n*************************************************\nWORK IN PROGRESS\n\tCNN preparation is not implemented\n*************************************************\n")
            pass

    return

def selectTrainingData(nx_graph, node_name, nx_edge):
    '''
    Select required data elements and discard the rest
    '''
    
    nx_data_flow = nx.get_node_attributes(nx_graph, JSON_OUTPUT_FLOW)
    output_flow = nx_data_flow[node_name]
    
    nx_dataFields = nx.get_edge_attributes(nx_graph, JSON_FEATURE_FIELDS)
    featureFields = nx_dataFields[nx_edge[0], nx_edge[1], output_flow]
    
    nx_targetFields = nx.get_edge_attributes(nx_graph, JSON_TARGET_FIELDS)
    targetFields = nx_targetFields[nx_edge[0], nx_edge[1], output_flow]
            
    df_combined = pd.DataFrame()
    nx_data_file = nx.get_node_attributes(nx_graph, JSON_INPUT_DATA_FILE)
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
                        
    nx_ignoreBlanks = nx.get_edge_attributes(nx_graph, JSON_IGNORE_BLANKS)
    if nx_ignoreBlanks != []:
        ignoreBlanks = nx_ignoreBlanks[nx_edge[0], nx_edge[1], output_flow]
        if ignoreBlanks:
            print("Removing NaN")
            df_combined = df_combined.dropna()
                
    df_combined.drop(targetFields, axis=1)
    return df_combined

def collect_and_select_data(d2r):
    logging.info('====> ================================================')
    logging.info('====> load_and_prepare_data: loading data for input to models')
    logging.info('====> ================================================')
    
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred collecting and selecting the data ***"

        for node_i in d2r.graph.nodes():
            nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == JSON_DATA_LOAD_PROCESS:
                nx_data_flow = nx.get_node_attributes(d2r.graph, JSON_OUTPUT_FLOW)
                output_flow = nx_data_flow[node_i]
                for edge_i in d2r.graph.edges():
                    if edge_i[0] == node_i:
                        err_txt = "*** An exception occurred analyzing the flow details in the json configuration file ***"
                        nx_flowFilename = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
                        flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
                        d2r.data = selectTrainingData(d2r.graph, node_i, edge_i)
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
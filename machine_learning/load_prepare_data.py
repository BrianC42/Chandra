'''
Created on Apr 7, 2020

@author: Brian
'''
import sys
import os
import time
import glob
import logging
import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from configuration_constants import JSON_PRECISION
from configuration_constants import JSON_OUTPUT_FLOW
from configuration_constants import JSON_INPUT_DATA_FILE
from configuration_constants import JSON_IGNORE_BLANKS
from configuration_constants import JSON_FLOW_DATA_FILE
from configuration_constants import JSON_INPUT_FLOWS
from configuration_constants import JSON_PROCESS_TYPE
from configuration_constants import JSON_DATA_LOAD_PROCESS

from configuration_constants import JSON_DATA_PREP_PROCESS
from configuration_constants import JSON_DATA_PREPARATION_CTRL
from configuration_constants import JSON_DATA_PREP_FEATURES
from configuration_constants import JSON_DATA_PREP_FEATURE
from configuration_constants import JSON_DATA_PREP_NORMALIZE
from configuration_constants import JSON_DATA_PREP_SEQ
from configuration_constants import JSON_DATA_PREP_ENCODING
from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_AUTOKERAS
from configuration_constants import JSON_FEATURE_FIELDS
from configuration_constants import JSON_TARGET_FIELDS
from configuration_constants import JSON_VALIDATION_SPLIT
from configuration_constants import JSON_TEST_SPLIT
from configuration_constants import JSON_TIMESTEPS

from TrainingDataAndResults import MODEL_TYPE
from TrainingDataAndResults import INPUT_LAYERTYPE_DENSE
from TrainingDataAndResults import INPUT_LAYERTYPE_RNN
from TrainingDataAndResults import INPUT_LAYERTYPE_CNN

def generate1hot(d2r, fields, fieldPrepCtrl):
    categorizedData = pd.DataFrame()
    ndxField = 0
    for field in fields:
        data = d2r.data[field]
        fieldPrep = fieldPrepCtrl[ndxField]
        if fieldPrep["categoryType"] == "seriesChangeUpDown":
            categories = fieldPrep["categories"]
            outputFields = fieldPrep["outputFields"]
            
            if field in d2r.preparedFeatures:
                d2r.preparedFeatures.extend(outputFields)
            if field in d2r.preparedTargets:
                d2r.preparedTargets.extend(outputFields)
            
            oneHot = pd.DataFrame([[0,0]], columns=outputFields)
            for i in range(1, len(data)):
                if float(data[i]) > float(data[i-1]):
                    row = pd.DataFrame(data=[[1, 0]], columns=outputFields)
                elif float(data[i]) < float(data[i-1]):
                    row = pd.DataFrame(data=[[0, 1]], columns=outputFields)
                else:
                    row = pd.DataFrame(data=[[0, 0]], columns=outputFields)
                oneHot = pd.concat([oneHot, row])
            '''
            twoHot = pd.DataFrame([[0,0]], columns=outputFields)
            row = pd.DataFrame(data=[[1, 0]])
            twoHot = pd.concat([twoHot, row])
            '''
        else:
            pass

        for col in outputFields:
            categorizedData = categorizedData.assign(temp=oneHot[col])
            categorizedData = categorizedData.rename(columns={'temp':col})

        d2r.data.drop(labels=field, axis=1, inplace=True)
        if field in d2r.preparedFeatures:
            d2r.preparedFeatures.remove(field)
        if field in d2r.preparedTargets:
            d2r.preparedTargets.remove(field)
        ndxField += 1

    categorizedData.index=range(0, len(categorizedData))
    d2r.data = pd.concat([d2r.data, categorizedData], axis=1)

    return

def normalizeFeature(d2r, fields, fieldPrepCtrl, normalizeType):
    if normalizeType == 'standard':
        d2r.scaler = StandardScaler()
    elif normalizeType == 'minmax':
        d2r.scaler = MinMaxScaler(feature_range=(0,1))

    normalizedFields = d2r.data[fields]
    d2r.scaler = d2r.scaler.fit(normalizedFields)
    normalizedFields = d2r.scaler.transform(normalizedFields)
    df_normalizedFields = pd.DataFrame(normalizedFields, columns=fields)
    d2r.data[fields] = df_normalizedFields
    d2r.normalized = True

    return

def prepareTrainingData(d2r):
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred preparing the training data ***"

        d2r.normalized = False
    
        for node_i in d2r.graph.nodes():
            nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == JSON_DATA_PREP_PROCESS:
                #print("Preparing data in %s" % node_i)
            
                nx_outputFlow = nx.get_node_attributes(d2r.graph, JSON_OUTPUT_FLOW)
                output_flow = nx_outputFlow[node_i]

                nx_output_data_file = ""
                nx_input_data_file = ""
                nx_inputFlows = nx.get_node_attributes(d2r.graph, JSON_INPUT_FLOWS)
                for input_flow in nx_inputFlows[node_i]:
                    for edge_i in d2r.graph.edges():
                        if edge_i[0] == node_i:
                            nx_output_data_files = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
                            nx_output_data_file = nx_output_data_files[edge_i[0], edge_i[1], output_flow]
                        if edge_i[1] == node_i:
                            nx_input_data_files = nx.get_edge_attributes(d2r.graph, JSON_FLOW_DATA_FILE)
                            nx_input_data_file = nx_input_data_files[edge_i[0], edge_i[1], input_flow]    
                            nx_seriesDataType = nx.get_edge_attributes(d2r.graph, "seriesDataType")
                            d2r.seriesDataType = nx_seriesDataType[edge_i[0], edge_i[1], input_flow]
                if os.path.isfile(nx_input_data_file):
                    d2r.data = pd.read_csv(nx_input_data_file)
                    d2r.rawData = d2r.data.copy()
                else:
                    ex_txt = node_i + ", input file " + nx_input_data_file + " does not exist"
                    raise NameError(ex_txt)

                js_prepCtrl = nx.get_node_attributes(d2r.graph, JSON_DATA_PREPARATION_CTRL)[node_i]
                if JSON_DATA_PREP_SEQ in js_prepCtrl:
                    normalizeFields = []
                    categorizeFields = []
                    '''
                    time2normailze = 0.0
                    time2categorize = 0.0
                    '''
                    for prep in js_prepCtrl[JSON_DATA_PREP_SEQ]:
                        prepCtrl = js_prepCtrl[prep]
                        fieldPrepCtrl = []
                        for feature in prepCtrl[JSON_DATA_PREP_FEATURES]:
                            if prep == JSON_DATA_PREP_NORMALIZE:
                                normalizeFields.append(feature[JSON_DATA_PREP_FEATURE])
                                fieldPrepCtrl.append(feature)
                            elif prep == JSON_DATA_PREP_ENCODING:
                                start = time.time()
                                categorizeFields.append(feature[JSON_DATA_PREP_FEATURE])
                                fieldPrepCtrl.append(feature)
                            else:
                                ex_txt = node_i + ", prep type is not specified"
                                raise NameError(ex_txt)
                        if prep == JSON_DATA_PREP_NORMALIZE:
                            #start = time.time()
                            normalizeFeature(d2r, normalizeFields, fieldPrepCtrl, prepCtrl['type'])
                            #time2normailze += time.time() - start
                        elif prep == JSON_DATA_PREP_ENCODING:
                            #start = time.time()
                            generate1hot(d2r, categorizeFields, fieldPrepCtrl)
                            #time2categorize += time.time() - start
                        else:
                            ex_txt = node_i + ", prep type is not specified"
                            raise NameError(ex_txt)
                    '''
                    print("\nData preparation times\n\tNormalization: %s\n\t1 Hot Encoding: %s" % \
                          (time2normailze, time2categorize))
                    '''
                else:
                    err_txt = "No preparation sequence specified"
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

def np_to_sequence(data, features, targets, seq_size=1):
    npx = np.empty([len(data) - (seq_size+1), seq_size, len(features)], dtype=float)
    npy = np.empty([len(data) - (seq_size+1),           len(targets)], dtype=float)

    for i in range(len(data) - (seq_size+1)):
        npx[i, :, :]    = data[i : i+seq_size, features[:]]
        npy[i]          = data[i+seq_size+1, targets[:]]

    return npx, npy

def arrangeDataForTraining(d2r):
    features = d2r.preparedFeatures
    targets = d2r.preparedTargets
    
    feature_cols, target_cols = id_columns(d2r.data, features, targets)

    nx_test_split = nx.get_node_attributes(d2r.graph, JSON_TEST_SPLIT)[d2r.mlNode]
    nx_validation_split = nx.get_node_attributes(d2r.graph, JSON_VALIDATION_SPLIT)[d2r.mlNode]

    nx_read_attr = nx.get_node_attributes(d2r.graph, JSON_PROCESS_TYPE)
    if nx_read_attr[d2r.mlNode] == JSON_TENSORFLOW:    
        print("Preparing the data for training: %s" % d2r.mlNode)
        d2r.trainLen = int(len(d2r.data) * (1-(nx_test_split+nx_validation_split)))
        d2r.validateLen = int(len(d2r.data) * nx_validation_split)
        d2r.testLen = int(len(d2r.data) * nx_test_split)

        nx_data_precision = nx.get_node_attributes(d2r.graph, JSON_PRECISION)[d2r.mlNode]
        ''' Create Numpy arrays suitable for training, training validation and later testing from Pandas dataframe '''
        data = np.array(d2r.data, dtype=nx_data_precision)
        train       = data[                                 : d2r.trainLen]
        validation  = data[d2r.trainLen                     : (d2r.trainLen + d2r.validateLen)]
        test        = data[d2r.trainLen + d2r.validateLen   : ]
        
        nx_model_type = nx.get_node_attributes(d2r.graph, MODEL_TYPE)[d2r.mlNode]

        if nx_model_type == INPUT_LAYERTYPE_DENSE:
            d2r.trainX = train[features[0]]
            d2r.trainY = train[targets[0]]
            d2r.validateX = validation[features[0]]
            d2r.validateY = validation[targets[0]]
            d2r.testX = test[features[0]]
            d2r.testY = test[targets[0]]
            
        elif nx_model_type == INPUT_LAYERTYPE_RNN:
            nx_time_steps = nx.get_node_attributes(d2r.graph, JSON_TIMESTEPS)[d2r.mlNode]
            
            d2r.trainX,    d2r.trainY    = np_to_sequence(train, feature_cols, target_cols, nx_time_steps)
            d2r.validateX, d2r.validateY = np_to_sequence(validation, feature_cols, target_cols, nx_time_steps)
            d2r.testX,     d2r.testY     = np_to_sequence(test, feature_cols, target_cols, nx_time_steps)
            
            ''' for RNN models train target against series of features after dropping sequence identifier field
            d2r.trainX = d2r.trainX[:, :, 1 :]
            d2r.validateX = d2r.validateX[:, :, 1 :]
            d2r.testX = d2r.testX[:, :, 1 :]
            '''
            
        elif nx_model_type == INPUT_LAYERTYPE_CNN:
            print("\n*************************************************\nWORK IN PROGRESS\n\tCNN preparation is not implemented\n*************************************************\n")
            pass
        
    elif nx_read_attr[d2r.mlNode] == JSON_AUTOKERAS:    
        nx_data_precision = nx.get_node_attributes(d2r.graph, JSON_PRECISION)[d2r.mlNode]
        
        d2r.trainLen = int(len(d2r.data) * (1-(nx_test_split+nx_validation_split)))
        d2r.validateLen = int(len(d2r.data) * nx_validation_split)
        d2r.testLen = int(len(d2r.data) * nx_test_split)
        
        train       = d2r.data.loc[                         : d2r.trainLen]
        validation  = d2r.data.loc[d2r.trainLen                : (d2r.trainLen + d2r.validateLen)]
        test        = d2r.data.loc[d2r.trainLen + d2r.validateLen : ]
            
        d2r.trainX = train[features[0]]
        d2r.trainY = train[targets[0]]
        d2r.validateX = validation[features[0]]
        d2r.validateY = validation[targets[0]]
        d2r.testX = test[features[0]]
        d2r.testY = test[targets[0]]

        d2r.trainX = d2r.trainX.to_frame()
        d2r.trainY = d2r.trainY.to_frame()
        d2r.validateX = d2r.validateX.to_frame()
        d2r.validateY = d2r.validateY.to_frame()
        d2r.testX = d2r.testX.to_frame()
        d2r.testY = d2r.testY.to_frame()

    return

def selectTrainingData(d2r, node_name, nx_edge):
    '''
    Select required data elements and discard the rest
    '''
    nx_data_flow = nx.get_node_attributes(d2r.graph, JSON_OUTPUT_FLOW)
    output_flow = nx_data_flow[node_name]
    
    nx_dataFields = nx.get_edge_attributes(d2r.graph, JSON_FEATURE_FIELDS)
    d2r.rawFeatures = nx_dataFields[nx_edge[0], nx_edge[1], output_flow]
    d2r.preparedFeatures = d2r.rawFeatures
    
    nx_targetFields = nx.get_edge_attributes(d2r.graph, JSON_TARGET_FIELDS)
    d2r.rawTargets = nx_targetFields[nx_edge[0], nx_edge[1], output_flow]
    d2r.preparedTargets = d2r.rawTargets
    
    nx_seriesStepIDs = nx.get_edge_attributes(d2r.graph, "seriesStepIDField")
    d2r.dataSeriesIDFields = nx_seriesStepIDs[nx_edge[0], nx_edge[1], output_flow]
    
    df_combined = pd.DataFrame()
    nx_data_file = nx.get_node_attributes(d2r.graph, JSON_INPUT_DATA_FILE)
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
                for fld in d2r.rawFeatures:
                    l_filter.append(fld)
                for fld in d2r.rawTargets:
                    l_filter.append(fld)
                for fld in d2r.dataSeriesIDFields:
                    l_filter.append(fld)
                df_inputs = df_data.filter(l_filter)
                df_combined = pd.concat([df_combined, df_inputs], ignore_index=True)
            else:
                raise NameError('Data file does not exist')
            count += 1
    print("\nData \n%s\nread from sources\n" % df_combined.describe().transpose())
                        
    nx_ignoreBlanks = nx.get_edge_attributes(d2r.graph, JSON_IGNORE_BLANKS)
    if nx_ignoreBlanks != []:
        ignoreBlanks = nx_ignoreBlanks[nx_edge[0], nx_edge[1], output_flow]
        if ignoreBlanks:
            print("Removing NaN")
            df_combined = df_combined.dropna()
                
    #df_combined.drop(targetFields, axis=1)
    return df_combined

def collect_and_select_data(d2r):
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
                        d2r.data = selectTrainingData(d2r, node_i, edge_i)
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
        
    return
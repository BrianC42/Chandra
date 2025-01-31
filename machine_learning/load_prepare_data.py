'''
Created on Apr 7, 2020

@author: Brian
'''
import sys
import os
import glob
import logging
import pickle
#import time
#import tkinter

import networkx as nx
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import configuration_constants as cc

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder

from Scalers import chandraScaler
#from Scalers import NORMALIZE_RELATIVE_TIME_SERIES

from TrainingDataAndResults import MODEL_TYPE
from TrainingDataAndResults import INPUT_LAYERTYPE_DENSE
from TrainingDataAndResults import INPUT_LAYERTYPE_RNN
from TrainingDataAndResults import INPUT_LAYERTYPE_CNN

#from evaluate_visualize import sanityCheckMACD

'''
Data pipeline step 2
====================
loadTrainingData
arrangeDataForTraining
    id_columns
    np_to_sequence
'''

def loadTrainingData(d2r):
    '''
    load data for training and testing
    '''
    # error handling
    try:
        err_txt = "*** An exception occurred loading the training data for the model ***"

        nx_input_flow = nx.get_node_attributes(d2r.graph, cc.JSON_INPUT_FLOWS)[d2r.mlNode]
        print("\nLoading prepared data defined in flow: %s" % nx_input_flow[0])
        nx_data_file = nx.get_edge_attributes(d2r.graph, cc.JSON_FLOW_DATA_FILE)
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

def balance_classes(d2r, model_type, categorization):
    if d2r.trainY.shape[1] == 1:
        if  model_type == INPUT_LAYERTYPE_RNN:
            print("Training shapes features:%s labels:%s\nvalidating shapes features:%s labels:%s\ntesting shapes features:%s labels:%s" % \
                  (d2r.trainX.shape, d2r.trainY.shape, d2r.validateX.shape, d2r.validateY.shape, d2r.testX.shape, d2r.testY.shape))
            print("\nSample data contains %s samples with\t%s labels distributed: %s" % \
                  (len(d2r.data), d2r.categories, d2r.categorieCounts))
    
            categories, counts = np.unique(d2r.trainY, return_counts=True)
            #sanityCheckMACD(npX=d2r.trainX, npY=d2r.trainY)
            print("Prior to balancing training labels are\n{} with {} distribution".format(categories, counts))
    
            minCount = min(counts)
            batches = minCount * len(categories)
            
            ''' equal number of batches for each category '''
            npX = np.zeros([batches, d2r.trainX.shape[1], d2r.feature_count], dtype=float)
            npY = np.zeros([batches, d2r.trainY.shape[1]], dtype=float)
            
            catStep = np.zeros(len(categories))
            for cat in range (0, len(counts)):
                catStep[cat] = counts[cat] / minCount
    
            catCount = np.zeros(len(categories))
            nextCatNdx = np.zeros(len(categories))
            
            for sample in range (0, minCount):
                for cat in range (0, len(categories)):
                    for offset in range (0, len(d2r.trainY)-int(nextCatNdx[cat])):
                        srcNdx = int(nextCatNdx[cat]) + offset
                        if d2r.trainY[srcNdx, 0] == categories[cat]:
                            npX[(sample * len(counts)) + cat, :, :] = d2r.trainX[srcNdx, :, :]
                            npY[(sample * len(counts)) + cat, :] = d2r.trainY[srcNdx, :]
                            catCount[cat] += 1
                            nextCatNdx[cat] = srcNdx + int(catStep[cat])
                            break                            
                        
            d2r.trainX = npX
            d2r.trainY = npY
            categories, counts = np.unique(d2r.trainY, return_counts=True)
            print("After balancing, training labels are\t%s with %s distribution" % (categories, counts))
        
        elif model_type == INPUT_LAYERTYPE_CNN:
            print("Training shapes features:%s labels:%s\nvalidating shapes features:%s labels:%s\ntesting shapes features:%s labels:%s" % \
                  (d2r.trainX.shape, d2r.trainY.shape, d2r.validateX.shape, d2r.validateY.shape, d2r.testX.shape, d2r.testY.shape))
            print("\nSample data contains %s samples with\t%s labels distributed: %s" % \
                  (len(d2r.data), d2r.categories, d2r.categorieCounts))
    
            categories, counts = np.unique(d2r.trainY, return_counts=True)
            #sanityCheckMACD(npX=d2r.trainX, npY=d2r.trainY)
            print("Prior to balancing training labels are\n{} with {} distribution".format(categories, counts))
            minCount = min(counts)
            batches = minCount * len(categories)
            
            ''' equal number of batches for each category '''
            npX = np.zeros([batches, d2r.trainX.shape[1], d2r.feature_count], dtype=float)
            npY = np.zeros([batches, d2r.trainY.shape[1], d2r.trainY.shape[2]], dtype=float)
            
            catStep = np.zeros(len(categories))
            for cat in range (0, len(counts)):
                catStep[cat] = counts[cat] / minCount
    
            catCount = np.zeros(len(categories))
            nextCatNdx = np.zeros(len(categories))
            
            for sample in range (0, minCount):
                for cat in range (0, len(categories)):
                    for offset in range (0, len(d2r.trainY)-int(nextCatNdx[cat])):
                        srcNdx = int(nextCatNdx[cat]) + offset
                        if d2r.trainY[srcNdx, 0, 0] == categories[cat]:
                            npX[(sample * len(counts)) + cat, :, :] = d2r.trainX[srcNdx, :, :]
                            npY[(sample * len(counts)) + cat, :, :] = d2r.trainY[srcNdx, :, :]
                            catCount[cat] += 1
                            nextCatNdx[cat] = srcNdx + int(catStep[cat])
                            break                            
                        
            d2r.trainX = npX
            d2r.trainY = npY
            
            categories, counts = np.unique(d2r.trainY, return_counts=True)
            print("After balancing, training labels are\t%s with %s distribution" % (categories, counts))
    
        elif model_type == INPUT_LAYERTYPE_DENSE:
            print("Sample data contains %s samples with categories: %s categories distributed: %s" % \
                  (len(d2r.data), d2r.categories, d2r.categorieCounts))
            print("Training shapes features:%s labels:%s\nvalidating shapes features:%s labels:%s\ntesting shapes features:%s labels:%s" % \
                  (d2r.trainX.shape, d2r.trainY.shape, d2r.validateX.shape, d2r.validateY.shape, d2r.testX.shape, d2r.testY.shape))
    
            categories, counts = np.unique(d2r.trainY, return_counts=True)
            print("Prior to balancing training labels are\n{} with {} distribution".format(categories, counts))
    
            minCount = min(counts)
            
            ''' equal number of batches for each category '''
            npX = np.zeros([minCount * len(counts), d2r.feature_count], dtype=float)
            npY = np.zeros([minCount * len(counts), d2r.trainY.shape[1]], dtype=float)
            
            catStep = np.zeros(len(categories))
            for cat in range (0, len(counts)):
                catStep[cat] = counts[cat] / minCount
    
            catCount = np.zeros(len(categories))
            nextCatNdx = np.zeros(len(categories))
            
            sampleStep = len(d2r.categories)
            for sample in range (0, len(npX), sampleStep):
                for cat in range (0, len(categories)):
                    for offset in range (0, len(d2r.trainY)-int(nextCatNdx[cat])):
                        srcNdx = int(nextCatNdx[cat]) + offset
                        #srcNdx = int(nextCatNdx[cat])
                        if d2r.trainY[srcNdx, 0] == categories[cat]:
                            npX[sample + cat, :] = d2r.trainX[srcNdx, :]
                            npY[sample + cat, :] = d2r.trainY[srcNdx, :]
                            catCount[cat] += 1
                            nextCatNdx[cat] = srcNdx + int(catStep[cat])
                            break                            
                        
            d2r.trainX = npX
            d2r.trainY = npY
    
            categories, counts = np.unique(d2r.trainY, return_counts=True)
            print("After balancing training labels are\n{} with {} distribution".format(categories, counts))
        
    else:
        ex_txt = "label balancing is not supported for multiple labels in " + model_type + " models"
        labelPatterns, counts = np.unique(d2r.trainY, return_counts=True, axis=0)
        print("Prior to balancing training labels are\n{} with {} distribution".format(labelPatterns, counts))
        if  model_type == INPUT_LAYERTYPE_RNN:
            minCount = min(counts)
            batches = minCount * len(labelPatterns)
            
            ''' equal number of batches for each category '''
            npX = np.zeros([batches, d2r.trainX.shape[1], d2r.feature_count], dtype=float)
            npY = np.zeros([batches, d2r.trainY.shape[1]], dtype=float)
            
            catStep = np.zeros(len(labelPatterns))
            for cat in range (0, len(counts)):
                catStep[cat] = counts[cat] / minCount
    
            catCount = np.zeros(len(labelPatterns))
            nextCatNdx = np.zeros(len(labelPatterns))
            
            for sample in range (0, minCount):
                for cat in range (0, len(labelPatterns)):
                    for offset in range (0, len(d2r.trainY)-int(nextCatNdx[cat])):
                        srcNdx = int(nextCatNdx[cat]) + offset
                        if np.array_equal(d2r.trainY[srcNdx], labelPatterns[cat]):
                            npX[(sample * len(counts)) + cat, :, :] = d2r.trainX[srcNdx, :, :]
                            npY[(sample * len(counts)) + cat, :] = d2r.trainY[srcNdx, :]
                            catCount[cat] += 1
                            nextCatNdx[cat] = srcNdx + int(catStep[cat])
                            break                            
                        
            d2r.trainX = npX
            d2r.trainY = npY
        elif model_type == INPUT_LAYERTYPE_CNN:
            raise NameError(ex_txt)
        elif model_type == INPUT_LAYERTYPE_DENSE:
            raise NameError(ex_txt)
        else:
            raise NameError(ex_txt)
        labelPatterns, counts = np.unique(d2r.trainY, return_counts=True, axis=0)
        print("After balancing training labels are\n{} with {} distribution".format(labelPatterns, counts))
    return

def combineMultipleSamples(d2r, model_goal, combineCount, forecastInterval, data_precision, feature_cols, target_cols):
    npData   = np.array(d2r.data, dtype=float)
    numSamples = len(d2r.data)
    if model_goal == cc.JSON_ML_GOAL_CATEGORIZATION:
        numBatches = numSamples - combineCount
    elif  model_goal == cc.JSON_ML_GOAL_REGRESSION:
        numBatches = numSamples - combineCount - forecastInterval
    features = np.zeros([numBatches, len(feature_cols) * combineCount], dtype=data_precision)
    labels = np.zeros([numBatches, len(target_cols)], dtype=data_precision)
    
    for ndx in range(0, numBatches):
        for stepndx in range (0, combineCount):
            startCol = stepndx * len(feature_cols)
            endCol = startCol + len(feature_cols)
            features[ndx, startCol : endCol] = npData[ndx + stepndx, feature_cols]
        if model_goal == cc.JSON_ML_GOAL_CATEGORIZATION:
            labelNdx = ndx + stepndx + forecastInterval
        elif  model_goal == cc.JSON_ML_GOAL_REGRESSION:
            labelNdx = ndx + stepndx + forecastInterval
        labels[ndx, : ] = npData[labelNdx, target_cols]
    d2r.feature_count = features.shape[1]
    
    return features, labels

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

def np_to_conv1d(data, features, targets, d2r):
    npx = np.empty([1, len(data), len(features)]  , dtype=float)
    npy = np.empty([1, len(data), len(targets)]   , dtype=float)
    
    ''' create 3D [batch,time period/data elements] array '''
    npx[0, :, :]    = data[: , features[:]]
    npy[0]          = data[: , targets[:]]
    
    return npx, npy

def np_to_sequence(data, features, targets, seq_size=1):
    '''
    Convert 2 dimensional data to the 3 dimensional format required by recurrent networks
    data : 2 dimensional data
    features : the columns in data (required to be consecutive) containing the features to train on
    targets : the columns in data (required to be consecutive) containing the labels to train on
    seq_size : the number of consecutive data samples to use for training
    '''

    npx = np.zeros([len(data) - (seq_size+1), seq_size, len(features)], dtype=float)
    npy = np.zeros([len(data) - (seq_size+1),           len(targets)], dtype=float)
    for i in range(len(data) - (seq_size+1)):
        npx[i, :, :]    = data[i : i+seq_size, features[:]]
        npy[i]          = data[    i+seq_size-1, targets[:]]

    return npx, npy

def arrangeDataForTrainedModels(d2r):
    '''
    Final step in creating the data structures used to use an existing model to generate data suitable for training.
    Assumes all data loading, feature selection, normalization and encoding has been completed
    '''
    try:
        err_txt = "\nError arranging data for prediction by a trained model"
        print("\n============== WIP =============\n\tArranging data for use by an existing model")
    
        features = d2r.preparedFeatures
        targets = d2r.preparedTargets
        feature_cols, target_cols = id_columns(d2r.data, features, targets)
        
        print("Removing NaN")
        d2r.data = d2r.data.dropna()
        for dKey in list(d2r.dataDict):
            d2r.dataDict[dKey] = d2r.dataDict[dKey].dropna()

        print("\n============== WIP =============\n\tIf the model was trained on normalized data the prediction data needs to be trained with the same normalization")
        scalerFile = 'd:\\brian\\AI-Projects\\models\\rnn\\lstm\\TrendLineCross\\TLCC_test001 20230712 154802 scaler XXX.pkl'
        if os.path.isfile(scalerFile):
            with open(scalerFile, 'rb') as pf:
                d2r.scaler = pickle.load(pf)
            pf.close()
        else:
            raise NameError("Specified data normalization file\n\t{}\n\tdoes not exist".format(scalerFile))

        '''
        nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
        '''
        print("\n============== WIP =============\n\tmodel characteristics are not defined and are hard coded")
        d2r.mlNode = "executeModelM1"
        modelType = cc.JSON_TENSORFLOW
        inputLayerType = INPUT_LAYERTYPE_RNN
        nx_time_steps = 5
        
        if modelType == cc.JSON_TENSORFLOW:
            print("\n============== WIP =============\n\tPreparing the data for prediction as defined in: {} of type {}".format(d2r.mlNode, modelType))
            if inputLayerType == INPUT_LAYERTYPE_DENSE:
                raise NameError('Model type {} is not yet supported'.format(modelType))
                
            elif inputLayerType == INPUT_LAYERTYPE_RNN:
                d2r.timesteps = nx_time_steps

                npX = np.zeros([0, d2r.timesteps, len(feature_cols)], dtype=float)
                #npY = np.zeros([0, len(target_cols)], dtype=float)

                for dkey in d2r.normDataDict:
                    npData   = np.array(d2r.normDataDict[dkey], dtype=float)
                    features, labels = np_to_sequence(npData, feature_cols, target_cols, nx_time_steps)
                    npX = np.row_stack((npX, features))
                    #npY = np.row_stack((npY, labels))

                d2r.dataX = npX[ : , :]
                #d2r.dataY = npY[ : , :]
                
            elif inputLayerType == INPUT_LAYERTYPE_CNN:
                raise NameError('Model type {} is not yet supported'.format(modelType))

        elif modelType == cc.JSON_AUTOKERAS:    
            print("\n============== WIP =============\n\tPreparing the data for prediction as defined in: {} of type {}".format(d2r.mlNode, modelType))
            raise NameError('Model type {} is not yet supported'.format(modelType))
            
        else:
            raise NameError('Model type {} is not supported'.format(modelType))

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

def arrangeDataForTraining(d2r):
    '''
    Final step in creating the data structures used to train the model.
    All data loading, feature selection, normalization and encoding has been completed
    '''
    try:
        err_txt = "\nError arranging data for training"
        features = d2r.preparedFeatures
        targets = d2r.preparedTargets
        
        feature_cols, target_cols = id_columns(d2r.data, features, targets)
        maxCol = max(feature_cols + target_cols)
        
        nx_modelIterations = nx.get_node_attributes(d2r.graph, "training iterations")[d2r.mlNode]
        iterVariables = nx_modelIterations[d2r.trainingIteration]
        iterParamters = iterVariables["iteration parameters"]
        iterTraining = iterParamters["training"]
    
        nx_test_split = iterTraining[cc.JSON_TEST_SPLIT]        
        nx_validation_split = iterTraining[cc.JSON_VALIDATION_SPLIT]        
    
        nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
        if nx_read_attr[d2r.mlNode] == cc.JSON_TENSORFLOW:    
            print("Preparing the data for training as defined in: %s" % d2r.mlNode)
            d2r.trainLen = int(len(d2r.data) * (1-(nx_test_split+nx_validation_split)))
            d2r.validateLen = int(len(d2r.data) * nx_validation_split)
            d2r.testLen = int(len(d2r.data) * nx_test_split)
    
            nx_data_precision = nx.get_node_attributes(d2r.graph, cc.JSON_PRECISION)[d2r.mlNode]
            nx_combine_sample_Count = nx.get_node_attributes(d2r.graph, cc.JSON_ML_GOAL_COMBINE_SAMPLE_COUNT)[d2r.mlNode]            
            
            ''' Create Numpy arrays suitable for training, training validation and later testing from Pandas dataframe '''
            data        = np.array(d2r.data, dtype=nx_data_precision)
            train       = data[                                 : d2r.trainLen]
            validation  = data[d2r.trainLen                     : (d2r.trainLen + d2r.validateLen)]
            test        = data[d2r.trainLen + d2r.validateLen   : ]
            
            if len(list(d2r.normDataDict)) == 0:
                d2r.normDataDict = d2r.dataDict
            
            nx_model_type = nx.get_node_attributes(d2r.graph, MODEL_TYPE)[d2r.mlNode]
            nx_regression_forecast_interval = nx.get_node_attributes(d2r.graph, cc.JSON_ML_REGRESSION_FORECAST_INTERVAL)[d2r.mlNode]
 
            nx_categorization_regression = nx.get_node_attributes(d2r.graph, cc.JSON_ML_GOAL)[d2r.mlNode]
            d2r.categorizationRegression = nx_categorization_regression
            #sanityCheckMACD(combined=d2r.data)
   
            if nx_model_type == INPUT_LAYERTYPE_DENSE:
                print("WIP ==========\n\tuse maxCol to prevent array out of bounds if not all samples contain all all targets")
                if d2r.categorizationRegression == cc.JSON_ML_GOAL_CATEGORIZATION:
                    if nx_combine_sample_Count > 1:
                        features, labels = combineMultipleSamples(d2r, d2r.categorizationRegression, nx_combine_sample_Count, \
                                                                  nx_regression_forecast_interval, nx_data_precision, feature_cols, target_cols)
                    else:
                        npData   = np.array(d2r.data, dtype=float)
                        features = np.zeros([npData.shape[0], len(feature_cols)], dtype=nx_data_precision)
                        labels = np.zeros([npData.shape[0], len(target_cols)], dtype=nx_data_precision)
                        features[ : , : ] = npData[ : , feature_cols]
                        labels[ : , : ] = npData[ :  , target_cols]
                        
                    d2r.trainX = features[ : d2r.trainLen , :]
                    d2r.trainY = labels[ : d2r.trainLen , :]
                    d2r.validateX = features[d2r.trainLen : (d2r.trainLen + d2r.validateLen) , :]
                    d2r.validateY = labels[d2r.trainLen : (d2r.trainLen + d2r.validateLen) , :]
                    d2r.testX = features[d2r.trainLen + d2r.validateLen : , :]
                    d2r.testY = labels[d2r.trainLen + d2r.validateLen : , :]
                    d2r.feature_count = features.shape[1]
                elif  d2r.categorizationRegression == cc.JSON_ML_GOAL_REGRESSION:
                    if nx_combine_sample_Count > 1:
                        features, labels = combineMultipleSamples(d2r, d2r.categorizationRegression, nx_combine_sample_Count, \
                                                                  nx_regression_forecast_interval, nx_data_precision, feature_cols, target_cols)
                    else:
                        npData   = np.array(d2r.data, dtype=float)
                        features = np.zeros([npData.shape[0], len(feature_cols)], dtype=nx_data_precision)
                        labels = np.zeros([npData.shape[0], len(target_cols)], dtype=nx_data_precision)
                        features[ : , : ] = npData[ : , feature_cols]
                        labels[ : , : ] = npData[ :  , target_cols]
                        
                    d2r.trainX = features[ : d2r.trainLen , :]
                    d2r.trainY = labels[ : d2r.trainLen , :]
                    d2r.validateX = features[d2r.trainLen : (d2r.trainLen + d2r.validateLen) , :]
                    d2r.validateY = labels[d2r.trainLen : (d2r.trainLen + d2r.validateLen) , :]
                    d2r.testX = features[d2r.trainLen + d2r.validateLen : , :]
                    d2r.testY = labels[d2r.trainLen + d2r.validateLen : , :]
                    d2r.feature_count = features.shape[1]
                
            elif nx_model_type == INPUT_LAYERTYPE_RNN:
                nx_time_steps = nx.get_node_attributes(d2r.graph, cc.JSON_TIMESTEPS)[d2r.mlNode]
                d2r.timesteps = nx_time_steps

                npX = np.zeros([0, d2r.timesteps, len(feature_cols)], dtype=float)
                npY = np.zeros([0, len(target_cols)], dtype=float)
                '''
                for dkey in d2r.dataDict:
                    npData   = np.array(d2r.dataDict[dkey], dtype=float)
                '''
                for dkey in d2r.normDataDict:
                    err_txt = "\nError arranging data for training for " + dkey
                    npData   = np.array(d2r.normDataDict[dkey], dtype=float)
                    if npData.shape[0] <= nx_time_steps:
                        print("{} has fewer samples, {}, than time steps required, {}".format(dkey, npData.shape[0], nx_time_steps))
                    elif npData.shape[1] <= maxCol:
                        print("{} does not have examples of all targets".format(dkey))
                    else:
                        features, labels = np_to_sequence(npData, feature_cols, target_cols, nx_time_steps)
                        npX = np.row_stack((npX, features))
                        npY = np.row_stack((npY, labels))
                err_txt = "\nError creating training, validation and testing data sets"
                d2r.trainX = npX[ : d2r.trainLen , :]
                d2r.trainY = npY[ : d2r.trainLen , :]
                d2r.validateX = npX[d2r.trainLen : (d2r.trainLen + d2r.validateLen) , :]
                d2r.validateY = npY[d2r.trainLen : (d2r.trainLen + d2r.validateLen) , :]
                d2r.testX = npX[d2r.trainLen + d2r.validateLen : , :]
                d2r.testY = npY[d2r.trainLen + d2r.validateLen : , :]
                d2r.feature_count = features.shape[2]
                
            elif nx_model_type == INPUT_LAYERTYPE_CNN:
                '''
                data arranged as 3D batch/sequence elements/data elements
                ...X represent causation features
                ...Y represent effect targets
                '''

                TARGETWINDDOW = 1
                
                batches = 0
                for dKey in list(d2r.normDataDict):
                    batches += d2r.normDataDict[dKey].shape[0] - d2r.timesteps

                npX = np.zeros([batches, d2r.timesteps, len(feature_cols)], dtype=float)
                npY = np.zeros([batches, TARGETWINDDOW, len(target_cols)], dtype=float)

                batch = 0
                for dKey in list(d2r.normDataDict):
                    for srcPeriod in range(d2r.timesteps, d2r.normDataDict[dKey].shape[0]):
                        npX[batch, :, :] = d2r.normDataDict[dKey].to_numpy()[srcPeriod-d2r.timesteps : srcPeriod, feature_cols]
                        npY[batch, :, :] = d2r.normDataDict[dKey].to_numpy()[                          srcPeriod-1, target_cols]
                        batch += 1
                                
                #sanityCheckMACD(npX=npX, npY=npY, stage="creating train, validate, test data sets")
                d2r.batches = batch
                d2r.trainX    = npX[:d2r.trainLen, :, :]
                d2r.trainY    = npY[:d2r.trainLen, :, :]
                d2r.validateX = npX[d2r.trainLen:(d2r.trainLen + d2r.validateLen),:,  :]
                d2r.validateY = npY[d2r.trainLen:(d2r.trainLen + d2r.validateLen),:,  :]
                d2r.testX     = npX[d2r.trainLen + d2r.validateLen:, :,  :]
                d2r.testY     = npY[d2r.trainLen + d2r.validateLen:, :,  :]

        elif nx_read_attr[d2r.mlNode] == cc.JSON_AUTOKERAS:    
            nx_data_precision = nx.get_node_attributes(d2r.graph, cc.JSON_PRECISION)[d2r.mlNode]
            
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

        ''' =================== Sanity check of data preparation processing ===================
        if nx_read_attr[d2r.mlNode] == cc.JSON_TENSORFLOW:    
            if nx_model_type == INPUT_LAYERTYPE_DENSE:
                pass
            elif nx_model_type == INPUT_LAYERTYPE_RNN:
                pass
            elif nx_model_type == INPUT_LAYERTYPE_CNN:
                sanityCheckMACD(npX=d2r.trainX, npY=d2r.trainY, stage="arranged")
                d2r.visualize_categorization_samples()
        elif nx_read_attr[d2r.mlNode] == cc.JSON_AUTOKERAS:
            pass
        =================== Sanity check of data preparation processing =================== '''

        nx_dataBalanced = nx.get_node_attributes(d2r.graph, cc.JSON_BALANCED)[d2r.mlNode]
        if nx_dataBalanced:
            balance_classes(d2r, nx_model_type, d2r.categories)
        
        ''' =================== Sanity check of data preparation processing ===================
        if nx_read_attr[d2r.mlNode] == cc.JSON_TENSORFLOW:    
            if nx_model_type == INPUT_LAYERTYPE_DENSE:
                pass
            elif nx_model_type == INPUT_LAYERTYPE_RNN:
                pass
            elif nx_model_type == INPUT_LAYERTYPE_CNN:
                sanityCheckMACD(npX=d2r.trainX, npY=d2r.trainY, stage="after balancing")
                d2r.visualize_categorization_samples()
        elif nx_read_attr[d2r.mlNode] == cc.JSON_AUTOKERAS:
            pass
        =================== Sanity check of data preparation processing =================== '''
        

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

def generate1hot(d2r, fields, fieldPrepCtrl):
    '''
    Modify d2r.data and d2r.normDataDict by
    1. generating 1 hot versions of data fields (both features and labels)
    2. inserting the 1 hot data into the modified data structures used to train the model
    3. removing original fields from d2r.data and d2r.normDataDict
    '''

    '''
    process d2r.data
    '''
    categorizedData = pd.DataFrame()
    ndxField = 0
    for field in fields:
        data = d2r.data[field]
        fieldPrep = fieldPrepCtrl[ndxField]
        if fieldPrep[cc.JSON_1HOT_CATEGORYTYPE] == cc.JSON_1HOT_SERIES_UP_DOWN:
            ''' Prepare categorized feature - feature float value compared to previous sample, <, =, >'''
            print("Converting features %s to 1 hot format in d2r.data" % field)
            categories = fieldPrep[cc.JSON_1HOT_CATEGORIES]
            outputFields = fieldPrep[cc.JSON_1HOT_OUTPUTFIELDS]
            
            oneHot = pd.DataFrame([[0,0]], columns=outputFields)
            for i in range(1, len(data)):
                if float(data[i]) > float(data[i-1]):
                    row = pd.DataFrame(data=[[1, 0]], columns=outputFields)
                elif float(data[i]) < float(data[i-1]):
                    row = pd.DataFrame(data=[[0, 1]], columns=outputFields)
                else:
                    row = pd.DataFrame(data=[[0, 0]], columns=outputFields)
                oneHot = pd.concat([oneHot, row])

            ''' prepare one hot version of data field '''
            for col in outputFields:
                categorizedData = categorizedData.assign(temp=oneHot[col])
                categorizedData = categorizedData.rename(columns={'temp':col})                

        elif fieldPrep[cc.JSON_1HOT_CATEGORYTYPE] == cc.JSON_1HOT_LABEL:
            ''' Prepare categorized labels '''
            categories, counts = np.unique(d2r.data[field].to_numpy(), return_counts=True)

            ''' LabelBinarizer - works for 3 category value example
            label1Hot = preprocessing.LabelBinarizer()
            label1Hot.fit(categories)
            np_categorizedData = label1Hot.transform(d2r.data[field].to_numpy())
            categorizedData = pd.DataFrame(data=np_categorizedData)
            '''
            
            ''' OneHotEncoder (preferred) '''
            npField = d2r.data[field].to_numpy().reshape(-1, 1)
            encoder = preprocessing.OneHotEncoder(sparse_output=False)
            encoder.set_output(transform="pandas")
            encoder.fit(npField)
            oneHotField = encoder.transform(npField)
            oneHotField = oneHotField.reset_index(drop=True)
            #categories = categories.reshape(-1, 1)
            #raise NameError("converting to 1 hot format {}".format(field))
                
        else:
            pass
        
        ''' Replace original data with 1 Hot format '''
        print("dropping {} from\n{}".format(field, d2r.data))
        d2r.data.drop(labels=field, axis=1, inplace=True)
        d2r.data = d2r.data.reset_index(drop=True)
        #print("adding in {} rows from\n{}\nto {} in\n{}".format(len(oneHotField), oneHotField, len(d2r.data), d2r.data))
        d2r.data = pd.concat([d2r.data, oneHotField], axis=1)
        print("creates {} rows\n{}".format(len(d2r.data), d2r.data))
        ndxField += 1

    ''' Insert 1 hot data into features or labels / targets '''
    #print("update d2r with 1 hot column names")
    '''
    categoryNames = []
    for ndx in range(0, len(categories)):
        name = field+str(ndx)
        categoryNames.append(name)
        categorizedData = categorizedData.rename(columns={ndx:name})
    categorizedData.index=range(0, len(categorizedData))
    d2r.data = pd.concat([d2r.data, categorizedData], axis=1)
    '''

    '''
    process d2r.normDataDict
    '''
    for dKey in list(d2r.normDataDict):
        categorizedData = pd.DataFrame()
        ndxField = 0
        for field in fields:
            fieldPrep = fieldPrepCtrl[ndxField]
            if fieldPrep[cc.JSON_1HOT_CATEGORYTYPE] == cc.JSON_1HOT_SERIES_UP_DOWN:
                ''' Prepare categorized labels '''
                ex_txt = "generate1hot\nWIP .......................\n\tConverting up/down to 1 hot format"
                raise NameError(ex_txt)
                
            elif fieldPrep[cc.JSON_1HOT_CATEGORYTYPE] == cc.JSON_1HOT_LABEL:
                ''' Prepare categorized labels '''
                #categories, counts = np.unique(d2r.normDataDict[dKey][field].to_numpy(), return_counts=True)

                ''' LabelBinarizer - works for 3 category value example
                label1Hot = preprocessing.LabelBinarizer()
                label1Hot.fit(categories)
                np_categorizedData = label1Hot.transform(d2r.normDataDict[dKey][field].to_numpy())
                categorizedData = pd.DataFrame(data=np_categorizedData)
                '''
                
                ''' OneHotEncoder (preferred) - WIP '''
                encoder = preprocessing.OneHotEncoder(sparse_output=False)
                encoder.set_output(transform="pandas")
                encoder.fit(d2r.normDataDict[dKey][field].to_numpy().reshape(-1, 1))
                oneHotField = encoder.transform(d2r.normDataDict[dKey][field].to_numpy().reshape(-1, 1))
                
                d2r.normDataDict[dKey].drop(labels=field, axis=1, inplace=True)
                d2r.normDataDict[dKey] = pd.concat([d2r.normDataDict[dKey], oneHotField], axis=1)
            
                d2r.preparedTargets = encoder.get_feature_names_out()
            
            else:
                pass

            #print("update d2r with 1 hot column names")
            '''
            categoryNames = []
            for ndx in range(0, len(categories)):
                name = field+str(ndx)
                categoryNames.append(name)
                categorizedData = categorizedData.rename(columns={ndx:name})
                
            d2r.normDataDict[dKey].drop(labels=field, axis=1, inplace=True)
            categorizedData.index=d2r.normDataDict[dKey].index
            d2r.normDataDict[dKey] = pd.concat([d2r.normDataDict[dKey], categorizedData], axis=1)
            '''
            
    return

def normalizeFeature(d2r, fields, fieldPrepCtrl, normalizeType):
    #sanityCheckMACD(combined=d2r.data, stage="d2r.data prior to normalization")
    
    if normalizeType == cc.JSON_DATA_PREP_NORMALIZE_STANDARD:
        d2r.scaler = StandardScaler()
    elif normalizeType == cc.JSON_DATA_PREP_NORMALIZE_MINMAX:
        d2r.scaler = MinMaxScaler(feature_range=(0,1))
    elif normalizeType == cc.JSON_DATA_PREP_NORMALIZE_RELATIVE_TIME_SERIES:
        cs = chandraScaler()
        d2r.scaler = cs.relativeTimeSeries()
        
    if len(fields) > 0:
        print("Normalizing data as relative time series")
        normalizedFields = d2r.data[fields]
        
        d2r.scaler = d2r.scaler.fit(normalizedFields)
        normalizedFields = d2r.scaler.transform(normalizedFields)
        
        df_normalizedFields = pd.DataFrame(normalizedFields, columns=fields)
        d2r.data[fields] = df_normalizedFields
        
        tf_progbar = keras.utils.Progbar(len(d2r.dataDict), width=50, verbose=1, interval=10, stateful_metrics=None, unit_name='sample source')
        count = 0
        for dKey in list(d2r.dataDict):
            tf_progbar.update(count)
            if d2r.dataDict[dKey].shape[0] == 0:
                print("Skipping and removing %s" % dKey)
                del d2r.dataDict[dKey]
            else:
                for fld in fields:
                    if fld not in d2r.dataDict[dKey]:
                        raise NameError("Missing data " + fld + " in " + dKey)
                d2r.normDataDict[dKey] = d2r.dataDict[dKey]
                normalizedFields = d2r.dataDict[dKey][fields]

                #d2r.scaler = d2r.scaler.fit(normalizedFields)
                normalizedFields = d2r.scaler.transform(normalizedFields)
                df_normalizedFields = pd.DataFrame(normalizedFields, index=d2r.dataDict[dKey].index, columns=fields)
                d2r.normDataDict[dKey][fields] = df_normalizedFields

                #sanityCheckMACD(combined=d2r.normDataDict[dKey], stage="d2r.normDataDict after normalization")
            count += 1

        d2r.normalized = True
    else:
        d2r.normalized = False

    return

def prepareTrainingData(d2r):
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred preparing the training data ***"

        d2r.normalized = False
        
        for node_i in d2r.graph.nodes():
            nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == cc.JSON_DATA_PREP_PROCESS:
                #print("Preparing data in %s" % node_i)
            
                nx_outputFlow = nx.get_node_attributes(d2r.graph, cc.JSON_OUTPUT_FLOW)
                output_flow = nx_outputFlow[node_i]

                d2r.preparedFeatures = []
                d2r.preparedTargets = []

                nx_output_data_file = ""
                nx_input_data_file = ""
                nx_inputFlows = nx.get_node_attributes(d2r.graph, cc.JSON_INPUT_FLOWS)
                for input_flow in nx_inputFlows[node_i]:
                    for edge_i in d2r.graph.edges():
                        if edge_i[0] == node_i:
                            err_txt = "*** An exception occurred preparing the training data - output flow configuration ***"
                            nx_output_data_files = nx.get_edge_attributes(d2r.graph, cc.JSON_FLOW_DATA_FILE)
                            nx_output_data_file = nx_output_data_files[edge_i[0], edge_i[1], output_flow]
                            
                            nx_dataFields = nx.get_edge_attributes(d2r.graph, cc.JSON_FEATURE_FIELDS)
                            d2r.preparedFeatures = nx_dataFields[edge_i[0], edge_i[1], output_flow]
    
                            nx_targetFields = nx.get_edge_attributes(d2r.graph, cc.JSON_TARGET_FIELDS)
                            d2r.preparedTargets = nx_targetFields[edge_i[0], edge_i[1], output_flow]    

                        if edge_i[1] == node_i:
                            err_txt = "*** An exception occurred preparing the training data - input flow configuration ***"
                            nx_input_data_files = nx.get_edge_attributes(d2r.graph, cc.JSON_FLOW_DATA_FILE)
                            nx_input_data_file = nx_input_data_files[edge_i[0], edge_i[1], input_flow]    
                            nx_seriesDataType = nx.get_edge_attributes(d2r.graph, cc.JSON_SERIES_DATA_TYPE)
                            if nx_seriesDataType == {}:
                                d2r.seriesDataType = ""
                            else:
                                d2r.seriesDataType = nx_seriesDataType[edge_i[0], edge_i[1], input_flow]
                                
                if os.path.isfile(nx_input_data_file):
                    d2r.rawData = pd.read_csv(nx_input_data_file)
                    d2r.data = d2r.rawData.copy()
                else:
                    ex_txt = node_i + ", input file " + nx_input_data_file + " does not exist"
                    raise NameError(ex_txt)

                js_prepCtrl = nx.get_node_attributes(d2r.graph, cc.JSON_DATA_PREPARATION_CTRL)[node_i]
                if cc.JSON_IGNORE_BLANKS in js_prepCtrl:
                    if js_prepCtrl[cc.JSON_IGNORE_BLANKS]:
                        print("Removing NaN from {}\n{}".format(len(d2r.data), d2r.data))
                        d2r.data = d2r.data.dropna()
                        d2r.data = d2r.data.reset_index(drop=True)
                        print("creates {} rows\n{}".format(len(d2r.data), d2r.data))
                        for dKey in list(d2r.dataDict):
                            d2r.dataDict[dKey] = d2r.dataDict[dKey].dropna()
                            d2r.dataDict[dKey] = d2r.dataDict[dKey].reset_index(drop=True)
       
                if cc.JSON_DATA_PREP_SEQ in js_prepCtrl:
                    normalizeFields = []
                    categorizeFields = []
                    for prep in js_prepCtrl[cc.JSON_DATA_PREP_SEQ]:
                        err_txt = "An exception occurred preparing the training data - {}".format(prep)
                        prepCtrl = js_prepCtrl[prep]
                        fieldPrepCtrl = []
                        for feature in prepCtrl[cc.JSON_DATA_PREP_FEATURES]:
                            if prep == cc.JSON_DATA_PREP_NORMALIZE:
                                normalizeFields.append(feature[cc.JSON_DATA_PREP_FEATURE])
                                fieldPrepCtrl.append(feature)
                            elif prep == cc.JSON_DATA_PREP_ENCODING:
                                categorizeFields.append(feature[cc.JSON_DATA_PREP_FEATURE])
                                fieldPrepCtrl.append(feature)
                            else:
                                ex_txt = node_i + ", prep type is not specified"
                                raise NameError(ex_txt)
                        if prep == cc.JSON_DATA_PREP_NORMALIZE:
                            normalizeFeature(d2r, normalizeFields, fieldPrepCtrl, prepCtrl[cc.JSON_DATA_PREP_NORMALIZATION_TYPE])
                        elif prep == cc.JSON_DATA_PREP_ENCODING:
                            generate1hot(d2r, categorizeFields, fieldPrepCtrl)
                        else:
                            ex_txt = node_i + ", prep type is not specified"
                            raise NameError(ex_txt)
                else:
                    err_txt = "No preparation sequence specified"
                    raise NameError(err_txt)

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

def selectTrainingData(d2r, node_name, nx_edge):
    '''
    Select required data elements and discard the rest
    '''
    nx_data_flow = nx.get_node_attributes(d2r.graph, cc.JSON_OUTPUT_FLOW)
    output_flow = nx_data_flow[node_name]
    
    nx_dataFields = nx.get_edge_attributes(d2r.graph, cc.JSON_FEATURE_FIELDS)
    d2r.rawFeatures = nx_dataFields[nx_edge[0], nx_edge[1], output_flow]
    
    nx_targetFields = nx.get_edge_attributes(d2r.graph, cc.JSON_TARGET_FIELDS)
    d2r.rawTargets = nx_targetFields[nx_edge[0], nx_edge[1], output_flow]
    
    nx_timeSeries = nx.get_edge_attributes(d2r.graph, cc.JSON_TIME_SEQ)
    d2r.timeSeries = nx_timeSeries[nx_edge[0], nx_edge[1], output_flow]
    if d2r.timeSeries:
        nx_seriesStepIDs = nx.get_edge_attributes(d2r.graph, cc.JSON_SERIES_ID)
        d2r.dataSeriesIDFields = nx_seriesStepIDs[nx_edge[0], nx_edge[1], output_flow]
    
    df_combined = pd.DataFrame()
    nx_data_file = nx.get_node_attributes(d2r.graph, cc.JSON_INPUT_DATA_FILE)
    for dataFile in nx_data_file[node_name]:
        fileSpecList = glob.glob(dataFile)
        fileCount = len(fileSpecList)
        tf_progbar = keras.utils.Progbar(fileCount, width=50, verbose=1, interval=1, stateful_metrics=None, unit_name='file')
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
                    
                if d2r.timeSeries:
                    for fld in d2r.dataSeriesIDFields:
                        l_filter.append(fld)
                df_inputs = df_data.filter(l_filter)
                
                #sanityCheckMACD(combined=df_inputs, stage="selectTrainingData")
                
                df_combined = pd.concat([df_combined, df_inputs], ignore_index=True)
                d2r.dataDict[FileSpec] = df_inputs
            else:
                raise NameError('Data file does not exist')
            count += 1
    print("\nData \n%s\nread from sources\n" % df_combined.describe().transpose())
                        
    #df_combined.drop(targetFields, axis=1)
    return df_combined

def acquireTrainingData(d2r, nodeName):
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred acquiring the data ***"

        '''
        Find the node specification for data acquisition
        '''
        for node_i in d2r.graph.nodes():
            #nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
            if node_i == nodeName:
                '''
                Find the specification of the output flow from the node
                '''
                nx_data_flow = nx.get_node_attributes(d2r.graph, cc.JSON_OUTPUT_FLOW)
                output_flow = nx_data_flow[node_i]
                for edge_i in d2r.graph.edges():
                    if edge_i[0] == node_i:
                        err_txt = "*** An exception occurred acquiring the data ***"
                        '''
                        Find the folder specified as the output destination
                        '''
                        nx_outputFolder = nx.get_edge_attributes(d2r.graph, cc.JSON_FLOW_DIR)
                        outputFolder = nx_outputFolder[edge_i[0], edge_i[1], output_flow]
                        '''
                        For each file identified in the acquisition node specification
                            read acquire the data
                            save the data to the output location
                        '''
                        nx_data_file = nx.get_node_attributes(d2r.graph, cc.JSON_INPUT_DATA_FILE)
                        for dataFile in nx_data_file[node_i]:
                            fileSpecList = glob.glob(dataFile)
                            fileCount = len(fileSpecList)
                            tf_progbar = keras.utils.Progbar(fileCount, width=50, verbose=1, interval=10, stateful_metrics=None, unit_name='file')
                            count = 0
                            for FileSpec in fileSpecList:
                                if os.path.isfile(FileSpec):
                                    tf_progbar.update(count)
                                    df_data = pd.read_csv(FileSpec)
                                    filename = os.path.basename(FileSpec)
                                    d2r.dataDict[FileSpec] = df_data
                                    #df_data.to_csv(outputFolder + '\\' + filename, index=False)
                                count += 1
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

def loadDataIntoTrainingPipeline(d2r, node_name):
    '''
    Combines the data acquired from multiple sources in d2r into a single dataframe suitable for later
    preparation and organization for training
    '''
    try:
        err_txt = "*** An exception occurred loading the data into the training process pipeline ***"
        print("loading data into training process pipeline")
        '''
        Select required data elements and discard the rest
        '''
        for edge_i in d2r.graph.edges():
            if edge_i[0] == node_name:
                nx_edge = edge_i
                break
        
        nx_data_flow = nx.get_node_attributes(d2r.graph, cc.JSON_OUTPUT_FLOW)
        output_flow = nx_data_flow[node_name]
        nx_flowFilename = nx.get_edge_attributes(d2r.graph, cc.JSON_FLOW_DATA_FILE)
        flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
        
        nx_dataFields = nx.get_edge_attributes(d2r.graph, cc.JSON_FEATURE_FIELDS)
        d2r.rawFeatures = nx_dataFields[nx_edge[0], nx_edge[1], output_flow]
        
        nx_targetFields = nx.get_edge_attributes(d2r.graph, cc.JSON_TARGET_FIELDS)
        d2r.rawTargets = nx_targetFields[nx_edge[0], nx_edge[1], output_flow]
        
        nx_timeSeries = nx.get_edge_attributes(d2r.graph, cc.JSON_TIME_SEQ)
        d2r.timeSeries = nx_timeSeries[nx_edge[0], nx_edge[1], output_flow]
        if d2r.timeSeries:
            nx_seriesStepIDs = nx.get_edge_attributes(d2r.graph, cc.JSON_SERIES_ID)
            d2r.dataSeriesIDFields = nx_seriesStepIDs[nx_edge[0], nx_edge[1], output_flow]
            
        ''' build up a list of the fields in the data to load '''
        l_filter = []
        for fld in d2r.rawFeatures:
            l_filter.append(fld)
        for fld in d2r.rawTargets:
            l_filter.append(fld)
        if d2r.timeSeries:
            for fld in d2r.dataSeriesIDFields:
                l_filter.append(fld)

        df_combined = pd.DataFrame(columns=l_filter)
        tf_progbar = keras.utils.Progbar(len(d2r.dataDict), width=50, verbose=1, interval=50, stateful_metrics=None, unit_name='training data')
        count = 0
        for dataFile in d2r.dataDict:
            #print("loadDataIntoTrainingPipeline for {}".format(dataFile))
            df_data = d2r.dataDict[dataFile]
            df_inputs = df_data.filter(l_filter)        
            if all(col in df_inputs.columns for col in l_filter):
                if len(df_combined) == 0:
                    df_combined = df_inputs
                else:
                    df_combined = pd.concat([df_combined, df_inputs], ignore_index=True)
                d2r.dataDict[dataFile] = df_inputs
            else:
                print("Required data elements are missing in {}".format(dataFile))
            tf_progbar.update(count)
            count += 1
        #print("\nData loaded into pipeline\n%s" % df_combined.describe().transpose())
            
        d2r.data = df_combined
        d2r.determineCategories()
        d2r.archiveData(flowFilename)
        return

    except Exception:
        ''' error handling '''
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
        
def collect_and_select_data(d2r):
    ''' error handling '''
    try:
        err_txt = "*** An exception occurred collecting and selecting the data ***"

        for node_i in d2r.graph.nodes():
            nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
            if nx_read_attr[node_i] == cc.JSON_DATA_LOAD_PROCESS:
                nx_data_flow = nx.get_node_attributes(d2r.graph, cc.JSON_OUTPUT_FLOW)
                output_flow = nx_data_flow[node_i]
                for edge_i in d2r.graph.edges():
                    if edge_i[0] == node_i:
                        err_txt = "*** An exception occurred analyzing the flow details in the json configuration file ***"
                        nx_flowFilename = nx.get_edge_attributes(d2r.graph, cc.JSON_FLOW_DATA_FILE)
                        flowFilename = nx_flowFilename[edge_i[0], edge_i[1], output_flow]
                        d2r.data = selectTrainingData(d2r, node_i, edge_i)
                        d2r.determineCategories()
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
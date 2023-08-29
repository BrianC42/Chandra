'''
Created on Jun 29, 2023

@author: Brian
'''
import sys
import logging

import pandas as pd
import networkx as nx
from tensorflow import keras

import configuration_constants as cc

from TrainingDataAndResults import Data2Results as d2r

from load_prepare_data import collect_and_select_data
from load_prepare_data import prepareTrainingData
from load_prepare_data import loadTrainingData
from load_prepare_data import arrangeDataForTraining
from load_prepare_data import arrangeDataForTrainedModels
from assemble_model import buildModel
from train_model import trainModel
from evaluate_visualize import evaluate_and_visualize
from configuration import read_processing_network_json
from configuration_graph import build_configuration_graph

def loadModelfromFile(d2rP, nodeConfig):

    modelFile = nodeConfig[cc.JSON_TRAINED_MODEL_FILE]
    d2rP.model = keras.models.load_model(modelFile)

    return

def predictModelOutput(d2rP, nodeConfig):

    predictions = d2rP.model.predict(d2rP.dataX, verbose=0)
    pdPredictions = pd.DataFrame(predictions)

    colHeaders = nodeConfig[cc.JSON_OUTPUT_FEATURE_LABELS]
    pdPredictions.columns = colHeaders
    
    print("\n============== WIP =============\n\tAdd back in the synchronization feature: {}".format(nodeConfig[cc.JSON_SYNCHRONIZATION_FEATURE]))
    syncData = nodeConfig[cc.JSON_SYNCHRONIZATION_FEATURE]
    t=d2rP.data[syncData]
    print("prediction shape: {}\ninput data shape: {}".format(pdPredictions.shape, d2rP.data[syncData].shape))

    return pdPredictions

def archiveModelOutputData(d2rP, predictions, nodeConfig):

    outputFile = nodeConfig[cc.JSON_OUTPUT_FILE]    
    predictions.to_csv(outputFile, index=False)
    print("\nPrediction data \n{}\nwritten to {} for training\n".format(predictions.describe().transpose(), outputFile))
    
    return

def executeStop(nodeName, d2rP):
    raise NameError('\n\t-------- As instructed in the configuration file: stopping at node {} --------'.format(nodeName))

    return

def executeDataLoad(nodeName, d2rP):
    collect_and_select_data(d2rP)

    return

def executeDataPrep(nodeName, d2rP):
    print("\n============== WIP =============\n\tEnsure data is normalized using training normalization\n============\n")
    prepareTrainingData(d2rP)

    return

def executeTensorflow(nodeName, d2rP):
    ''' function to train and evaluate models '''
    
    for d2rP.trainingIteration in range (0, d2rP.trainingIterationCount):
        print ('\nStructure the data as required by the model, build and train the Model')
        buildModel(d2rP)
        loadTrainingData(d2rP)
        arrangeDataForTraining(d2rP)
        trainModel(d2rP)
        print ("\nEvaluate the model and visualize accuracy!")
        evaluate_and_visualize(d2rP)
        
    return

def executeExecuteModel(nodeName, d2rP):
    
    #print("\tcreate d2r to control the execution of the pre-existing model")

    ''' search calling configuration for the execution node configuration and create control structure for the node'''
    for node in d2rP.configurationFile[cc.JSON_PROCESS_NODES]:
        if node[cc.JSON_NODE_NAME] == nodeName:
            configSpec = d2rP.configurationFileDir + node[cc.JSON_CONDITIONAL][cc.JSON_EXECUTE_CONTROL][cc.JSON_EXECUTION_CONTROL]
            print("Executing node {} as defined in configuration file\n\t{}".format(nodeName, configSpec))
            d2rEM = d2r()
            d2rEM.processing_json = read_processing_network_json(configSpec)
    
            build_configuration_graph(d2rEM, d2rEM.processing_json)
            collect_and_select_data(d2rEM)
            print("\n============== WIP =============\n\tnormalization needs to be done using scaler fit during training")
            prepareTrainingData(d2rEM)    
            arrangeDataForTrainedModels(d2rEM)

            for nodeEM in d2rEM.processing_json[cc.JSON_PROCESS_NODES]:
                if nodeEM[cc.JSON_REQUIRED][cc.JSON_PROCESS_TYPE] == cc.JSON_EXECUTE_MODEL:
                    nodeConfig = nodeEM[cc.JSON_CONDITIONAL][cc.JSON_EXECUTE_MODEL_CONTROL]
                    loadModelfromFile(d2rEM, nodeConfig)
                    prediction = predictModelOutput(d2rEM, nodeConfig)
                    archiveModelOutputData(d2rEM, prediction, nodeConfig)
            
                    break

    return

def executeAutokeras(nodeName, d2rP):
    raise NameError('\n\t----------------- Autokeras is not yet supported ---------------------')

    nx_model_file = nx.get_node_attributes(d2rP.graph, cc.JSON_MODEL_FILE)[d2rP.mlNode]
    if d2rP.trainer == cc.TRAINING_AUTO_KERAS:
        d2rP.model = d2rP.model.export_model()
    d2rP.model.save(nx_model_file)

    return

def determineProcessingOrder(d2rP):
    
    execSequence = d2rP.configurationFile[cc.JSON_PROCESSING_SEQUENCE]

    return execSequence

def executeProcessingNodes(d2rP):
    print("\nExecuting processing nodes")
    exc_txt = "\nDuring execution of the processing nodes an exception occurred"
    try:    
        
        executionSequesnce = determineProcessingOrder(d2rP)
        for node_i in executionSequesnce:
            nx_read_attr = nx.get_node_attributes(d2rP.graph, cc.JSON_PROCESS_TYPE)
            print("\n====================================\n\tExecuting {} as {}\n====================================\n".format(node_i, nx_read_attr[node_i]))
       
            if nx_read_attr[node_i] == cc.JSON_DATA_LOAD_PROCESS:
                executeDataLoad(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_DATA_PREP_PROCESS:
                executeDataPrep(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_TENSORFLOW:
                executeTensorflow(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_EXECUTE_MODEL:
                executeExecuteModel(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_AUTOKERAS:
                executeAutokeras(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_STOP:
                exc_txt = ""
                executeStop(node_i, d2rP)
                
            else:
                unsupported = "\n\tProcessing node {} of type {} are not supported".format(node_i, nx_read_attr[node_i])
                raise NameError(unsupported)
                
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + " " + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)
    
    return
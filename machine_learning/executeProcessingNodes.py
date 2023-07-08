'''
Created on Jun 29, 2023

@author: Brian
'''
import sys
import logging

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

def loadModelfromFile(d2rP):
    print("\n============== WIP =============\n\tLoad existing model from file")
    
    d2rP.model = keras.models.load_model('path/to/location')

    return

def callExistingModel(d2rP):
    print("\n============== WIP =============\n\tUse existing model to generate training data")
    '''
    do not use call()
    predict() ???
    '''
    return

def archiveModelOutputData(d2rP):
    print("\n============== WIP =============\n\tSave data output by trained model for use in later training")
    
    return

def executeStop(nodeName, d2rP):
    raise NameError('\n\t-------- As instructed in the configuration file: stopping at node {} --------'.format(nodeName))

    return

def executeDataLoad(nodeName, d2rP):
    collect_and_select_data(d2rP)

    return

def executeDataPrep(nodeName, d2rP):
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
    d2rEM = d2r()

    print("\n============== WIP =============\n\tLoading hard coded json config file")
    d2rEM.processing_json = read_processing_network_json(d2rP.configurationFileDir + "ExecuteModelDevelopment1.json")
    build_configuration_graph(d2rEM, d2rEM.processing_json)
    collect_and_select_data(d2rEM)
    prepareTrainingData(d2rEM)    
    arrangeDataForTrainedModels(d2rEM)
    loadModelfromFile(d2rEM)
    callExistingModel(d2rEM)
    archiveModelOutputData(d2rEM)

    return

def executeAutokeras(nodeName, d2rP):
    raise NameError('\n\t----------------- Autokeras is not yet supported ---------------------')

    nx_model_file = nx.get_node_attributes(d2rP.graph, cc.JSON_MODEL_FILE)[d2rP.mlNode]
    if d2rP.trainer == cc.TRAINING_AUTO_KERAS:
        d2rP.model = d2rP.model.export_model()
    d2rP.model.save(nx_model_file)

    return

def determineProcessingOrder(d2rP):
    print("\n============== WIP =============\n\tDetermining processing order - currently hard coded")
    
    for node_i in d2rP.graph.nodes():
        pass
    
    #execSequence = ["ModelOne", "stop", "ModelTwo", "DataCombine", "DataPrepare", "TrainMultipleModelOutputs"]
    #execSequence = ["LoadData", "PrepData", "TrendLineCross"]
    execSequence = ["LoadData", "PrepData", "BollingerBands"]

    return execSequence

def executeProcessingNodes(d2rP):
    print("\nExecuting processing nodes")
    exc_txt = "\nDuring execution of the processing nodes an exception occurred"
    try:    
        
        executionSequesnce = determineProcessingOrder(d2rP)
        for node_i in executionSequesnce:
            nx_read_attr = nx.get_node_attributes(d2rP.graph, cc.JSON_PROCESS_TYPE)
            print("\n============== WIP =============\n\tExecuting {} as {}".format(node_i, nx_read_attr[node_i]))
       
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
                unsupported = "\n\tProcessing node xxx of type xxx are not supported"
                raise NameError(unsupported)
                
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + " " + exc_str
        logging.debug(exc_txt)
        sys.exit(exc_txt)
    
    return
'''
Created on Jun 29, 2023

@author: Brian
'''
import sys
import logging

import networkx as nx

import configuration_constants as cc

from load_prepare_data import collect_and_select_data
from load_prepare_data import prepareTrainingData
from load_prepare_data import loadTrainingData
from load_prepare_data import arrangeDataForTraining
from assemble_model import buildModel
from train_model import trainModel
from evaluate_visualize import evaluate_and_visualize

def executeStop(nodeName, d2r):
    raise NameError('\n\t----------------- stopping at node %s ---------------------' % nodeName)
    return

def executeDataLoad(nodeName, d2r):
    print("Executing %s as %s" % (nodeName, cc.JSON_DATA_LOAD_PROCESS))
    collect_and_select_data(d2r)
    return

def executeDataPrep(nodeName, d2r):
    print("Executing %s as %s" % (nodeName, cc.JSON_DATA_PREP_PROCESS))
    prepareTrainingData(d2r)
    return

def executeTensorflow(nodeName, d2r):
    print("Executing %s as %s" % (nodeName, cc.JSON_TENSORFLOW))
    for d2r.trainingIteration in range (0, d2r.trainingIterationCount):
        print ('\nStructure the data as required by the model, build and train the Model')
        buildModel(d2r)
        loadTrainingData(d2r)
        arrangeDataForTraining(d2r)
        trainModel(d2r)
        print ("\nEvaluate the model and visualize accuracy!")
        evaluate_and_visualize(d2r)
        
    return

def executeExecuteModel(nodeName, d2r):
    print("Executing %s as %s" % (nodeName, cc.JSON_EXECUTE_MODEL))
    raise NameError('\n\t----------------- Execute Model is not yet supported ---------------------')
    return

def executeAutokeras(nodeName, d2r):
    print("Executing %s as %s" % (nodeName, cc.JSON_AUTOKERAS))
    raise NameError('\n\t----------------- Autokeras is not yet supported ---------------------')

    nx_model_file = nx.get_node_attributes(d2r.graph, cc.JSON_MODEL_FILE)[d2r.mlNode]
    if d2r.trainer == cc.TRAINING_AUTO_KERAS:
        d2r.model = d2r.model.export_model()
    d2r.model.save(nx_model_file)

    return

def determineProcessingOrder(d2r):
    print("Determining processing order")
    #execSequence = ["ModelOne", "ModelTwo", "DataCombine", "DataPrepare", "TrainMultipleModelOutputs", "stop"]
    execSequence = ["LoadData", "PrepData", "TrendLineCross"]
    
    for node_i in d2r.graph.nodes():
        pass
    
    return execSequence

def executeProcessingNodes(d2r):
    print("Executing processing nodes")
    exc_txt = "\nDuring execution of the processing nodes an exception occurred"
    try:    
        
        executionSequesnce = determineProcessingOrder(d2r)
        for node_i in executionSequesnce:
            nx_read_attr = nx.get_node_attributes(d2r.graph, cc.JSON_PROCESS_TYPE)
       
            if nx_read_attr[node_i] == cc.JSON_DATA_LOAD_PROCESS:
                executeDataLoad(node_i, d2r)
            elif nx_read_attr[node_i] == cc.JSON_DATA_PREP_PROCESS:
                executeDataPrep(node_i, d2r)
            elif nx_read_attr[node_i] == cc.JSON_TENSORFLOW:
                executeTensorflow(node_i, d2r)
            elif nx_read_attr[node_i] == cc.JSON_EXECUTE_MODEL:
                executeExecuteModel(node_i, d2r)
            elif nx_read_attr[node_i] == cc.JSON_AUTOKERAS:
                executeAutokeras(node_i, d2r)
            elif nx_read_attr[node_i] == cc.JSON_STOP:
                exc_txt = ""
                executeStop(node_i, d2r)
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
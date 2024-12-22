'''
Created on Jun 29, 2023

@author: Brian
'''
import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

import networkx as nx

from configuration import get_ini_data
import configuration_constants as cc

from TrainingDataAndResults import Data2Results as d2r

from load_prepare_data import loadDataIntoTrainingPipeline
from load_prepare_data import acquireTrainingData
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

from pretrainedModels import rnnCategorization, rnnPrediction

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

def executeDataAcquire(nodeName, d2rP):
    print("\n============== WIP =============\n\tSeparating data acquisition from load into processing pipeline\n============\n")
    acquireTrainingData(d2rP, nodeName)

    return

def executeDataLoad(nodeName, d2rP):
    print("\n============== WIP =============\n\tSeparating data acquisition from load into processing pipeline\n============\n")
    loadDataIntoTrainingPipeline(d2rP, nodeName)

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

def executeDataCombine(nodeName, d2rP):
    try:
        exc_txt = "\nAn exception occurred - combining data as defined by process node {}".format(nodeName)
        
        ''' load local machine directory details '''
        localDirs = get_ini_data("LOCALDIRS") # Find local file directories
        aiwork = localDirs['aiwork']
        
        inputFlowFolders = []
        inputFlowFiles = []
        for nodeConfig in d2rP.configurationFile[cc.JSON_PROCESS_NODES]:
            if nodeConfig["processNodeName"] == nodeName:
                requiredConfiguration = nodeConfig[cc.JSON_REQUIRED]
                inputFlows = requiredConfiguration[cc.JSON_INPUT_FLOWS]
                outputFlow = requiredConfiguration[cc.JSON_OUTPUT_FLOW]
                combineFileList = requiredConfiguration[cc.JSON_COMBINE_FILE_LIST]
                
                if cc.JSON_CONDITIONAL in nodeConfig:
                    conditionalConfiguration = nodeConfig[cc.JSON_CONDITIONAL]
                    if cc.JSON_SYNCHRONIZATION_FEATURE in conditionalConfiguration:
                        synchronizationFeatures = conditionalConfiguration[cc.JSON_SYNCHRONIZATION_FEATURE]
                    '''
                    if cc.JSON_FLOW_DATA_DIR in conditionalConfiguration:
                        for dataDir in conditionalConfiguration[cc.JSON_FLOW_DATA_DIR]:
                            inputFlowFolders.append(dataDir)
                    '''
                break
            
        for flow in d2rP.configurationFile[cc.JSON_DATA_FLOWS]:
            if flow[cc.JSON_FLOW_NAME] in inputFlows:
                if cc.JSON_FLOW_DATA_DIR in flow[cc.JSON_CONDITIONAL]:
                    inputFlowFolders.append(flow[cc.JSON_CONDITIONAL][cc.JSON_FLOW_DATA_DIR])
                '''
                elif cc.JSON_FLOW_DATA_FILE in flow[cc.JSON_CONDITIONAL]:
                    inputFlowFiles.append(flow[cc.JSON_CONDITIONAL][cc.JSON_FLOW_DATA_FILE])
                '''
            if flow[cc.JSON_FLOW_NAME] == outputFlow:
                outputFlowDir = flow[cc.JSON_CONDITIONAL][cc.JSON_FLOW_DATA_DIR]
                
        '''
        horizontal merge of all sample columns
        all samples merged must have the same symbol and synchronizationFeatures
        symbol must exist in all sources
            input flow folders and 
            pre-loaded TrainingDataAndResults object
            
        Process
        create list of pre-loaded symbols
        
        '''
        ''' examine d2rP for loaded data files '''
        combinedData = dict() # horizontally merged samples
        # use del combinedData[symbol] to remove items if missing from a source
        
        loadedSymbolList = [] # list of pre-loaded symbols
        loadedData = dict()
        for loadedFile in d2rP.dataDict:
            subStr = loadedFile.split("\\")
            symbol = subStr[len(subStr)-1].split(".")
            symbol = symbol[0]
            loadedSymbolList.append(symbol)
            # use symbolList.remove(symbol) when symbol is missing from the inputFlowFolders
            loadedData[symbol] = d2rP.dataDict[loadedFile]
        print("{} symbols were loaded\n{}".format(len(loadedSymbolList), loadedSymbolList))
        
        ''' generate a dict of all file names present in the inputFlowFolders '''
        inputFlowLists = [] # list of lists
        symbolData = dict()
        for combineFolder in inputFlowFolders:
            inputFolderList = [] #list of symbols in the input flow folder
            for fileName in combineFileList:
                fileSpec = aiwork + '\\' + combineFolder + '\\' + fileName
                
                if os.path.isfile(fileSpec):
                    #print("dir {} file {}".format(combineFolder, fileSpec))
                    subStr = fileSpec.split("\\")
                    symbol = subStr[len(subStr)-1].split(".")
                    symbol = symbol[0]
                    symbolData[fileSpec] = symbol
                else:
                    for file in glob.glob(fileSpec):
                        #print("dir {} file {}".format(combineFolder, file))
                        subStr = file.split("\\")
                        symbol = subStr[len(subStr)-1].split(".")
                        symbol = symbol[0]
                        inputFolderList.append(symbol)
            print("{} symbols were present in {}\n{}".format(len(inputFolderList), combineFolder, inputFolderList))
            inputFlowLists.append(inputFolderList)

        ''' create a single list of symbols common to all sources '''
        # Convert the single list to a set
        loadedDataSet = set(loadedSymbolList)
        # Convert each list of names in the list_of_lists to a set
        inputFlowsSet = [set(lst) for lst in inputFlowLists]
        # Find the intersection of all sets
        commonSymbolSet = loadedDataSet.intersection(*inputFlowsSet)
        commonSymbolList = list(commonSymbolSet)
        print("{} symbols were common to all inputs\n{}".format(len(commonSymbolList), commonSymbolList))
    
        for symbol in commonSymbolList:
            # pre-loaded data
            combinedData[symbol] = loadedData[symbol]
            
            # merge the input data flows
            for folder in inputFlowFolders:
                fileSpec = aiwork + '\\' + folder + '\\' + symbol + ".csv"
                dfDataIn = pd.read_csv(fileSpec)
                combinedData[symbol] = pd.merge(combinedData[symbol], dfDataIn, how='inner', on=synchronizationFeatures)

        ''' create an output file containing all samples for training '''
        df_combined = pd.DataFrame()
        for symbol in combinedData:
            #print("writing {} combined data to {}".format(symbol, outputFlowDir))
            outputFile = aiwork + "\\" + outputFlowDir + "\\" + symbol + ".csv"
            combinedData[symbol].to_csv(outputFile)
            ''' combine into a single sample file "flowDataFile" - 'vertical' merge '''
            df_combined = pd.concat([df_combined, combinedData[symbol]], ignore_index=True)
        print("WIP ==============\n\tcombined data file fixed. Needs to be configuarable")
        outputFile = aiwork + "\\" + "internal_flows\\MMN_M1M2\\CombinedMMN.csv"
        df_combined.to_csv(outputFile)
        d2rP.data = df_combined
        
        ''' discard processed data '''
        d2r.dataDict = dict()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def executeExecuteModel(nodeName, d2rP):
    try:
        exc_txt = "\nAn exception occurred - using pre-trained model as required by process node {}".format(nodeName)
        
        ''' load local machine directory details '''
        localDirs = get_ini_data("LOCALDIRS") # Find local file directories
        aiwork = localDirs['aiwork']
        models = localDirs['trainedmodels']
        
        for nodeConfig in d2rP.configurationFile["processNodes"]:
            if nodeConfig["processNodeName"] == nodeName:

                exc_txt = "\nAn exception occurred - configuration file for node {} is invalid".format(nodeName)
                nodeConditionalFields = nodeConfig["conditionalFields"]["executeModelCtrl"]
                
                modelFile = nodeConditionalFields["trainedModelFile"]
                scalerFile = nodeConditionalFields["trained scaler"]
                flowDataDir = nodeConditionalFields[cc.JSON_FLOW_DATA_DIR]
                inputFileList = nodeConditionalFields["dataLoadCtrl"]["inputFile"]
                
                ''' Multiple dataPrepControls '''
                ignoreBlanks = nodeConditionalFields["dataPrepCtrl"]["ignoreBlanks"]
                scaleFeatures = nodeConditionalFields["dataPrepCtrl"]["scaleFeatures"]
                passthruFeatures = nodeConditionalFields["dataPrepCtrl"]["passthruFeatures"]
                
                ''' model controls '''
                modelFeatureList = nodeConditionalFields["modelControl"]["modelFeatures"]
                flowDataDir = nodeConditionalFields["flowDataDir"]
                outputLabels = nodeConditionalFields["outputLabels"]
                outputSynchronizationFeatures = nodeConditionalFields["outputSynchronizationFeatures"]
                replaceExistingOutput = nodeConditionalFields["replaceExistingOutput"]
                
                ''' model type specific controls '''
                if "rnn" in nodeConditionalFields["modelControl"]:
                    ''' RNN '''
                    inputType = "rnn"
                    timeSequence = nodeConditionalFields["modelControl"]["rnn"]["timeSequence"]
                    seriesStepIDField = nodeConditionalFields["modelControl"]["rnn"]["seriesStepIDField"]
                    seriesDataType = nodeConditionalFields["modelControl"]["rnn"]["seriesDataType"]
                    timeSteps = nodeConditionalFields["modelControl"]["rnn"]["timeSteps"]
                elif "cnn" in nodeConditionalFields["modelControl"]:
                    ''' CNN '''
                    inputType = "cnn"
                    err_msg = 'Models with first layers of type cnn are not yet implemented: {}'.format(nodeName)
                    raise NameError(err_msg)
                elif "dense" in nodeConditionalFields["modelControl"]:
                    ''' dense '''
                    inputType = "dense"
                    err_msg = 'Models with first layers of type dense are not yet implemented: {}'.format(nodeName)
                    raise NameError(err_msg)
                                
                ''' full list of features to retain for processing '''
                featuresOfInterest = passthruFeatures + modelFeatureList
                
                exc_txt = "\nAn exception occurred - trained model and scaler for node {} is invalid".format(nodeName)
                ''' load trained model '''
                trainedModel = aiwork + '\\' + models + '\\' + modelFile
                model = keras.models.load_model(trainedModel)
        
                ''' load scaler used during training '''
                if scalerFile != "":
                    scalerFile = aiwork + '\\' + models + '\\' + scalerFile
                    if os.path.isfile(scalerFile):
                        scaler = pd.read_pickle(scalerFile)
               
                ''' output path '''
                outputPath = aiwork + '\\' + flowDataDir
                os.makedirs(outputPath, exist_ok=True) 
        
                exc_txt = "\nAn exception occurred - node {} reading, preparing data and making predictions".format(nodeName)
                for fileListSpec in inputFileList:
                    fileListSpec = aiwork + '\\' + fileListSpec
                    fileList = glob.glob(fileListSpec)
                    for fileSpec in fileList:
                        exc_txt = "\nAn exception occurred - node {} reading, preparing data and making predictions from\n{}".format(nodeName, fileSpec)
                        if os.path.isfile(fileSpec):
                            subStr = fileSpec.split("\\")
                            symbol = subStr[len(subStr)-1].split(".")
                            if len(symbol) == 2:
                                outputFile = outputPath + "\\" + symbol[0] + ".csv"
                                if not os.path.isfile(outputFile) or replaceExistingOutput:
                                    dfData = pd.read_csv(fileSpec)
                                    
                                    ''' remove blank data from features of interest '''
                                    if ignoreBlanks:
                                        dfData.dropna(inplace=True)
                                        
                                    ''' select pass thru and scaled features '''
                                    dfFeatures = dfData[featuresOfInterest]
                                    
                                    ''' put aside any pass thru data fields '''
                                    if len(passthruFeatures) > 0:
                                        dfPassthruFeatures = dfFeatures[passthruFeatures]
                                    
                                    ''' separate and scale fields required by the model'''
                                    dfModelFeatures = dfFeatures[modelFeatureList]
                                    if len(scaleFeatures) > 0:
                                        dfModelFeatures = dfModelFeatures[scaleFeatures]
                                        npScaled = scaler.transform(dfModelFeatures)
                                        npModelFeatures = npScaled
                                    else:
                                        npModelFeatures = dfModelFeatures.to_numpy()
            
                                    ''' arrange features as required by the model '''
                                    if inputType == "rnn":
                                        npPredict = np.zeros((len(npModelFeatures) - timeSteps, timeSteps, npModelFeatures.shape[1]))
                                        for ndx in range (len(npModelFeatures) - timeSteps):
                                            npPredict[ndx,:, :] = npModelFeatures[ndx : ndx + timeSteps]
                                        modelOutputs = model.predict(x=npPredict, verbose=0)
                                    elif inputType == "cnn":
                                        pass
                                    elif inputType == "dense":
                                        pass        
                                    
                                    ''' convert outputs to dataframe '''
                                    dfOutputs = pd.DataFrame(modelOutputs, columns=outputLabels)
                                    ''' combine prediction / categorization to build output flow '''
                                    # print("dfPassthruFeatures\n{}\n".format(dfPassthruFeatures))
                                    # print("dfModelFeatures\n{}\n".format(dfModelFeatures))
                                    # print("npModelFeatures\n{}\n".format(npModelFeatures))
                                    # print("len(npModelFeatures)\n{}\n".format(len(npModelFeatures)))
                                    # print("dfOutputs\n{}\n".format(dfOutputs))
                                    
                                    dfPassthruFeatures = dfPassthruFeatures[timeSteps:]
                                    dfPassthruFeatures = dfPassthruFeatures.set_index(dfOutputs.index)
                                    # print("dfPassthruFeatures\n{}\n".format(dfPassthruFeatures))
                                    dfOutputs=pd.concat([dfOutputs,dfPassthruFeatures],axis=1)
                                    # print("dfOutputs\n{}\n".format(dfOutputs))
                                    
                                    ''' write predictions / categorizations to identified csv file '''
                                    dfOutputs.to_csv(outputFile)
                
                break

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

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
       
            if nx_read_attr[node_i] == cc.JSON_DATA_ACQUIRE_PROCESS:
                executeDataAcquire(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_DATA_LOAD_PROCESS:
                executeDataLoad(node_i, d2rP)
                
            elif nx_read_attr[node_i] == cc.JSON_DATA_COMBINE_PROCESS:
                executeDataCombine(node_i, d2rP)
                
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
        sys.exit(exc_txt)
    
    return
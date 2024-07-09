'''
Created on Aug 29, 2023

@author: crabt

Procedures to use pre-trained models

'''
import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from configuration import get_ini_data

def rnnCategorization(name, modelFile, featureFiles, features, scalerFile, timeSteps, outputs, signalThreshold=np.NaN):
    '''
    Parameters
        name - descriptive name of the model
        modelFile - pre-trained model
        featureFiles - list of file specifications
        features - list of feature fields required by the model
        scalerFile - scaler fit during taining
        timeSteps - number of time steps required by each categorization
        outputs - list of output field names
        thresholds - optional filter thresholds for each category to include in results
    '''
    exc_txt = "\nAn exception occurred - using pre-trained RNN categorization model"
    try:
        ''' load local machine directory details '''
        localDirs = get_ini_data("LOCALDIRS") # Find local file directories
        aiwork = localDirs['aiwork']
        models = localDirs['trainedmodels']
        
        ''' load trained model '''
        trainedModel = aiwork + '\\' + models + '\\' + modelFile
        model = keras.models.load_model(trainedModel)

        ''' load scaler used during training '''
        if scalerFile != "":
            scalerFile = aiwork + '\\' + models + '\\' + scalerFile
            if os.path.isfile(scalerFile):
                scaler = pd.read_pickle(scalerFile)

        signals = []
        for fileListSpec in featureFiles:
            fileListSpec = aiwork + '\\' + fileListSpec
            fileList = glob.glob(fileListSpec)
            for fileSpec in fileList:
                if os.path.isfile(fileSpec):
                    subStr = fileSpec.split("\\")
                    symbol = subStr[len(subStr)-1].split(".")
                    if len(symbol) == 2:
                        df_data = pd.read_csv(fileSpec)
                        dfFeatures = df_data[features]
                        dfFeatures = dfFeatures[len(dfFeatures)-timeSteps :]
                        if scalerFile != "":
                            npScaled = scaler.transform(dfFeatures)
                            npFeatures = npScaled
                        else:
                            npFeatures = dfFeatures.to_numpy()
                        
                        ''' make prediction '''
                        npFeatures = np.reshape(npFeatures, (1,timeSteps,len(features)))
                        prediction = model.predict(x=npFeatures, verbose=0)
                        
                        if not signalThreshold == np.NaN:
                            ndx = 0
                            for ndx in range (len(signalThreshold)):
                                if prediction[0][ndx] > signalThreshold[ndx]:
                                    signal = {'symbol':symbol[0], 'name':name, 'outputs':outputs, 'prediction':prediction[0]}
                                    signals.append(signal)
                                    break
                                ndx += 1
                        else:
                            signal = {'symbol':symbol[0], 'name':name, 'outputs':outputs, 'prediction':prediction[0]}
                            signals.append(signal)  

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return signals

def rnnPrediction(name, modelFile, featureFiles, features, scalerFile, timeSteps, outputs, signalThreshold=np.NaN):
    '''
    Parameters
        name - descriptive name of the model
        modelFile - pre-trained model
        featureFiles - list of file specifications
        features - list of feature fields required by the model
        scalerFile - scaler fit during taining
        timeSteps - number of time steps required by each categorization
        outputs - list of output field names
        thresholds - optional filter thresholds for each category to include in results
    '''
    exc_txt = "\nAn exception occurred - using pre-trained RNN prediction model"
    try:
        ''' load local machine directory details '''
        localDirs = get_ini_data("LOCALDIRS") # Find local file directories
        aiwork = localDirs['aiwork']
        models = localDirs['trainedmodels']
        
        ''' load trained model '''
        trainedModel = aiwork + '\\' + models + '\\' + modelFile
        model = keras.models.load_model(trainedModel)

        ''' load scaler used during training '''
        if scalerFile != "":
            scalerFile = aiwork + '\\' + models + '\\' + scalerFile
            if os.path.isfile(scalerFile):
                scaler = pd.read_pickle(scalerFile)
        
        predictions = []
        for fileListSpec in featureFiles:
            fileListSpec = aiwork + '\\' + fileListSpec
            fileList = glob.glob(fileListSpec)
            for fileSpec in fileList:
                if os.path.isfile(fileSpec):
                    subStr = fileSpec.split("\\")
                    symbol = subStr[len(subStr)-1].split(".")
                    if len(symbol) == 2:
                        df_data = pd.read_csv(fileSpec)
                        dfFeatures = df_data[features]
                        dfFeatures = dfFeatures[len(dfFeatures)-timeSteps :]
                        if scalerFile != "":
                            npScaled = scaler.transform(dfFeatures)
                            npFeatures = npScaled
                        else:
                            npFeatures = dfFeatures.to_numpy()
                        
                        ''' make prediction '''
                        npFeatures = np.reshape(npFeatures, (1,timeSteps,len(features)))
                        prediction = model.predict(x=npFeatures, verbose=0)
                        
                        if not signalThreshold == np.NaN:
                            if prediction[0][0] > signalThreshold:
                                signal = {'symbol':symbol[0], 'name':name, 'outputs':outputs, 'prediction':[prediction[0][0]]}
                                predictions.append(signal)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return predictions
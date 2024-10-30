'''
Created on Jul 27, 2023

@author: Brian
'''
'''
import os
import glob
import json
import time
import tkinter as tk
from configuration import read_config_json
from Workbooks import investments, optionTrades
from multiProcess import updateMarketData 
import tensorflow as tf
'''

import sys
import re
import datetime as dt

from configuration import get_ini_data
from DailyProcessUI import DailyProcessUI
from GoogleSheets import googleSheet
from pretrainedModels import rnnCategorization, rnnPrediction

PROCESS_CONFIGS = "processes"
MODEL_CONFIGS = "models"

PROCESS_ID = "name"
PROCESS_DESCRIPTION = "Description"
RUN = "run"
RUN_COL = 2
MODEL_FILE = "file"
MODEL = "model"
OUTPUT_FIELDS = "Outputs"
CONFIG = "json"

MARKET_DATA_UPDATE = "Market data update"
MARK_TO_MARKET = "Mark to market"
DERIVED_MARKET_DATA = "Calculate derived market data"
SECURED_PUTS = "Secured puts"
COVERED_CALLS = "Covered calls"
OPTION_TRADES = "Option trade review"
BOLLINGER_BAND_PREDICTION = "Bollinger Band"
MACD_TREND_CROSS = "MACD Trend"

def saveMACDCross(processCtrl, signals):
    exc_txt = "\nAn exception occurred - saving machine learning signal"

    try:
        ''' Google API and file details '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleLocal = get_ini_data("GOOGLE")

        ''' run time control parameters from json and UI '''
        ''' text '''
        gSheetName = processCtrl['gsheet']['entry'].get()
        gSheetID = googleLocal[gSheetName]
        headerRange = processCtrl['header range']['entry'].get()
        dataRange = processCtrl['data range']['entry'].get()
        ''' numerical '''
        ''' controls with multiple values '''

        ''' read sheet current cells - do not overwrite these '''
        gSheet = googleSheet()
        result = gSheet.googleSheet.values().get(spreadsheetId=gSheetID, range=dataRange).execute()
        values = result.get('values', [])

        for signal in signals:
            newSignal = [signal['name'], signal['symbol'], dt.datetime.now().strftime("%m/%d/%y"), \
                         str(signal['prediction'][0]), \
                         str(signal['prediction'][1]), \
                         str(signal['prediction'][2]) \
                         ]
            values.append(newSignal)
        
        requestBody = {'values': values}
        result = gSheet.googleSheet.values().update(spreadsheetId=gSheetID, range=dataRange, \
                                       valueInputOption="USER_ENTERED", body=requestBody).execute()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def saveMachineLearningSignal(processCtrl, signals):
    exc_txt = "\nAn exception occurred - saving machine learning signal"

    try:
        ''' Google API and file details '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleLocal = get_ini_data("GOOGLE")

        ''' run time control parameters from json and UI '''
        ''' text '''
        gSheetName = processCtrl['gsheet']['entry'].get()
        gSheetID = googleLocal[gSheetName]
        headerRange = processCtrl['header range']['entry'].get()
        dataRange = processCtrl['data range']['entry'].get()
        ''' numerical '''
        ''' controls with multiple values '''

        ''' read sheet current cells - do not overwrite these '''
        gSheet = googleSheet()
        result = gSheet.googleSheet.values().get(spreadsheetId=gSheetID, range=dataRange).execute()
        values = result.get('values', [])

        for signal in signals:
            outputStr = signal['outputs']
            predictionStr = str(signal['prediction'][0])
            newSignal = [signal['name'], signal['symbol'], outputStr, dt.datetime.now().strftime("%m/%d/%y"), predictionStr]
            values.append(newSignal)
        
        requestBody = {'values': values}
        result = gSheet.googleSheet.values().update(spreadsheetId=gSheetID, range=dataRange, \
                                       valueInputOption="USER_ENTERED", body=requestBody).execute()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def mlBollingerBandPrediction(name, processCtrl):
    exc_txt = "\nAn exception occurred - making predictions using Bollinger Band model"
    
    try:
        print("making daily predictions using the Bollinger Band model")
        
        ''' run time control parameters from json and UI '''
        ''' text '''
        modelFile = processCtrl['file']['entry'].get()
        scalerFile =  processCtrl['scaler']['entry'].get()
        outputs =  processCtrl['Outputs']['entry'].get()
        ''' numerical '''
        timeSteps = int(processCtrl['timeSteps']['entry'].get())
        threshold =  float(processCtrl['threshold']['entry'].get())
        ''' controls with multiple values '''
        features = re.split(',', processCtrl['features']['entry'].get())
        inputFileSpec =  re.split(',', processCtrl['featureFile']['entry'].get())
        
        signals = rnnPrediction(name, modelFile, inputFileSpec, features, \
                                 scalerFile, timeSteps, outputs, signalThreshold=threshold)
        if len(signals) > 0:
            saveMachineLearningSignal(processCtrl, signals)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def mlMACDTrendCross(name, processCtrl):
    exc_txt = "\nAn exception occurred - Identifying MACD trend crossing moving average"
    
    try:
        print("performing daily MACD trend crossing moving average process")
        
        ''' run time control parameters from json and UI '''
        ''' text '''
        modelFile = processCtrl['file']['entry'].get()
        scalerFile =  processCtrl['scaler']['entry'].get()
        outputs =  re.split(',', processCtrl['Outputs']['entry'].get())
        ''' numerical '''
        timeSteps = int(processCtrl['timeSteps']['entry'].get())
        ''' controls with multiple values '''
        thresholdStrs = re.split(',', processCtrl['threshold']['entry'].get())
        threshold = []
        for ndx in range (len(thresholdStrs)):       
            threshold.append(float(thresholdStrs[ndx]))
        features = re.split(',', processCtrl['features']['entry'].get())
        inputFileSpec =  re.split(',', processCtrl['featureFile']['entry'].get())
        
        signals = rnnCategorization(name, modelFile, inputFileSpec, features, \
                                 scalerFile, timeSteps, outputs, signalThreshold=threshold)
        if len(signals) > 0:
            saveMACDCross(processCtrl, signals)
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return 

if __name__ == '__main__':
    print ("Perform daily operational processes\n")
    UI = DailyProcessUI()
                    
    print ("\nAll requested processes have completed")
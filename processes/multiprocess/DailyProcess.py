'''
Created on Jul 27, 2023

@author: Brian
'''
import os
import sys
import glob
import json
import re
import time
import datetime as dt

import tkinter as tk

import numpy as np
import pandas as pd

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import update_tda_eod_data
from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists
from tda_api_library import tda_read_option_chain
from tda_api_library import format_tda_datetime
from tda_api_library import tda_manage_throttling
from tda_api_library import covered_call
from tda_api_library import cash_secured_put

from tda_derivative_data import loadHoldings
from tda_derivative_data import loadMarketDetails
from tda_derivative_data import calculateFields
from tda_derivative_data import eliminateLowReturnOptions
from tda_derivative_data import prepMktOptionsForSheets
from tda_derivative_data import prepMktOptionsForAnalysis

from multiProcess import updateMarketData 
from GoogleSheets import googleSheet

import tensorflow as tf

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
DERIVED_MARKET_DATA = "Calculate derived market data"
SECURED_PUTS = "Secured puts"
COVERED_CALLS = "Covered calls"
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
        for control in processCtrl:
            print("\tcontrol: {} = {}".format(
                            processCtrl[control]["description"], \
                            processCtrl[control]["entry"].get()))
        
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
        
        ''' load local machine directory details '''
        localDirs = get_ini_data("LOCALDIRS") # Find local file directories
        aiwork = localDirs['aiwork']
        models = localDirs['trainedmodels']
        
        ''' load trained model '''
        trainedModel = aiwork + '\\' + models + '\\' + modelFile
        model = tf.keras.models.load_model(trainedModel)

        ''' load scaler used during training '''
        if scalerFile != "":
            scalerFile = aiwork + '\\' + models + '\\' + scalerFile
            if os.path.isfile(scalerFile):
                scaler = pd.read_pickle(scalerFile)
                '''
                with open(scalerFile, 'rb') as pf:
                    scaler = pickle.load(pf)
                pf.close()
                '''        
        
        signals = []
        for fileListSpec in inputFileSpec:
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
                        if prediction[0][0] > threshold:
                            signal = {'symbol':symbol[0], 'name':name, 'outputs':outputs, 'prediction':[prediction[0][0]]}
                            signals.append(signal)
                            
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
        for control in processCtrl:
            print("\tcontrol: {} = {}".format(
                            processCtrl[control]["description"], \
                            processCtrl[control]["entry"].get()))
        
        ''' run time control parameters from json and UI '''
        ''' text '''
        modelFile = processCtrl['file']['entry'].get()
        scalerFile =  processCtrl['scaler']['entry'].get()
        outputs =  re.split(',', processCtrl['Outputs']['entry'].get())
        ''' numerical '''
        timeSteps = int(processCtrl['timeSteps']['entry'].get())
        threshold =  float(processCtrl['threshold']['entry'].get())
        ''' controls with multiple values '''
        features = re.split(',', processCtrl['features']['entry'].get())
        inputFileSpec =  re.split(',', processCtrl['featureFile']['entry'].get())
        
        ''' load local machine directory details '''
        localDirs = get_ini_data("LOCALDIRS") # Find local file directories
        aiwork = localDirs['aiwork']
        models = localDirs['trainedmodels']
        
        ''' load trained model '''
        trainedModel = aiwork + '\\' + models + '\\' + modelFile
        model = tf.keras.models.load_model(trainedModel)

        ''' load scaler used during training '''
        if scalerFile != "":
            scalerFile = aiwork + '\\' + models + '\\' + scalerFile
            if os.path.isfile(scalerFile):
                scaler = pd.read_pickle(scalerFile)

        signals = []
        for fileListSpec in inputFileSpec:
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
                        if prediction[0][0] > threshold or prediction[0][2] > threshold:
                            signal = {'symbol':symbol[0], 'name':name, 'outputs':outputs, 'prediction':prediction[0]}
                            signals.append(signal)

        if len(signals) > 0:
            saveMACDCross(processCtrl, signals)
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return 

def marketDataUpdate(processCtrl):
    exc_txt = "\nAn exception occurred - daily market data process"
    
    try:
        print("performing daily daily market data process")
        for control in processCtrl:
            print("\tcontrol: {} = {}".format(
                            processCtrl[control]["description"], \
                            processCtrl[control]["entry"].get()))
        app_data = get_ini_data("TDAMERITRADE")
        update_tda_eod_data(app_data['authentication'])
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return 

def calculateDerivedData(processCtrl):
    exc_txt = "\nAn exception occurred - calculating derived data"
    
    try:
        print("performing daily daily derived data process")
        for control in processCtrl:
            print("\tcontrol: {} = {}".format(
                            processCtrl[control]["description"], \
                            processCtrl[control]["entry"].get()))
        app_data = get_ini_data("TDAMERITRADE")
        updateMarketData(app_data['authentication'], app_data['eod_data'], app_data['market_analysis_data'])
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return 

def writeOptionsToGsheet(processCtrl, mktOptions, sheetNameBase):    
    exc_txt = "\nAn exception occurred - unable to write to Gsheet"
    
    try:
        ''' find Google sheet ID '''
        googleLocal = get_ini_data("GOOGLE")
        for control in processCtrl:
            print("\tcontrol: {} = {}".format(
                            processCtrl[control]["description"], \
                            processCtrl[control]["entry"].get()))

        ''' run time control parameters from json and UI '''
        ''' text '''
        fileName = processCtrl["gsheet"]["entry"].get()
        fileID = googleLocal[fileName]
        holdingRange = processCtrl["current holdings data range"]["entry"].get()
        marketDataRange = processCtrl["market data range"]["entry"].get()
        optionsHeader = processCtrl["options header range"]["entry"].get()
        optionsDataRange = processCtrl["options data range"]["entry"].get()
        ''' numerical '''
        ''' controls with multiple values '''
                
        gSheet = googleSheet()
    
        ''' convert data elements to formats suitable for analysis '''
        df_potential_strategies = prepMktOptionsForAnalysis(mktOptions)
        
        dfHoldings = loadHoldings(gSheet, fileID, holdingRange)
        dfMarketData = loadMarketDetails(gSheet, fileID, marketDataRange)        
        df_potential_strategies = calculateFields(df_potential_strategies, dfHoldings, dfMarketData)
    
        ''' valueInputOption = USER_ENTERED or RAW '''
        sheetMktOptions = prepMktOptionsForSheets(df_potential_strategies)
        
        now = dt.datetime.now()
        timeStamp = ' {:4d}{:0>2d}{:0>2d} {:0>2d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day, \
                                                                        now.hour, now.minute, now.second)        
        sheetName = sheetNameBase + timeStamp
        gSheet.addGoogleSheet(fileID, sheetName)
        gSheet.updateGoogleSheet(fileID, sheetName + optionsHeader, sheetMktOptions.columns)
        gSheet.updateGoogleSheet(fileID, sheetName + optionsDataRange, sheetMktOptions)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return sheetName

def marketPutOptions(processCtrl):
    exc_txt = "\nAn exception occurred - daily put option process"
    
    try:
        print("performing daily put option process")

        ''' Find local file directories '''
        '''
        exc_txt = "\nAn exception occurred - unable to retrieve local <>.ini files"
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close
        '''
        
        app_data = get_ini_data("TDAMERITRADE")
        '''
        json_config = read_config_json(app_data['config'])
        '''
        json_authentication = tda_get_authentication_details(app_data['authentication'])
        analysis_dir = app_data['market_analysis_data']
        
        '''
        Cash secured put options
        '''
        callCount=0
        periodStart = time.time()
        callCount, periodStart = tda_manage_throttling(callCount, periodStart)
    
        print("\nAnalyzing potential cash secured puts")
        exc_txt = "\nAn exception occurred while accessing market put options"
        df_potential_strategies = pd.DataFrame()
        for symbol in tda_read_watch_lists(json_authentication, watch_list='Potential Buy'):
            callCount, periodStart = tda_manage_throttling(callCount, periodStart)
            df_options, options_json = tda_read_option_chain(app_data['authentication'], symbol)
            filename = analysis_dir + '\\' + symbol + '.csv'
            if os.path.isfile(filename):
                df_data = pd.read_csv(filename)
            df_cash_secured_puts = cash_secured_put(symbol, df_data, df_options)
            if df_cash_secured_puts.shape[0] > 0:
                df_potential_strategies = pd.concat([df_potential_strategies, df_cash_secured_puts])
                  
        sheetName = writeOptionsToGsheet(processCtrl, df_potential_strategies, "put")
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return sheetName

def marketCallOptions(processCtrl):
    exc_txt = "\nAn exception occurred - daily call option process"
    
    try:
        print("performing daily call option process")

        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to retrieve local <>.ini files"
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close
        
        app_data = get_ini_data("TDAMERITRADE")
        json_config = read_config_json(app_data['config'])
        json_authentication = tda_get_authentication_details(app_data['authentication'])
        analysis_dir = app_data['market_analysis_data']
        
        callCount=0
        periodStart = time.time()
        callCount, periodStart = tda_manage_throttling(callCount, periodStart)
    
        print("\nAnalyzing potential covered calls")
        exc_txt = "\nAn exception occurred while accessing market call options"
        df_potential_strategies = pd.DataFrame()
        for symbol in tda_read_watch_lists(json_authentication, watch_list='Combined Holding'):
            #print("Accessing covered calls for {}".format(symbol))
            callCount, periodStart = tda_manage_throttling(callCount, periodStart)
            df_options, options_json = tda_read_option_chain(app_data['authentication'], symbol)
            filename = analysis_dir + '\\' + symbol + '.csv'
            if os.path.isfile(filename):
                df_data = pd.read_csv(filename)
            df_covered_calls = covered_call(symbol, df_data, df_options)
            if df_covered_calls.shape[0] > 0:
                df_potential_strategies = pd.concat([df_potential_strategies, df_covered_calls])
        
        sheetName = writeOptionsToGsheet(processCtrl, df_potential_strategies, "call")
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return sheetName

def userInterfaceControls():
    ''' display a user interface to solicit run time selections '''
    processCtrl=dict()
    dictBtnRun=dict()
    
    try:
        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to identify localization details"
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        gitdir = localDirs['git']
        models = localDirs['trainedmodels']
        print("Local computer directories - localDirs: {}\n\taiwork: {}\n\tgitdir: {}\n\ttrained models: {}".format(localDirs, aiwork, gitdir, models))
    
        ''' Google API and file details '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleAuth = get_ini_data("GOOGLE")
        print("Google authentication\n\ttoken: {}\n\tcredentials: {}".format(googleAuth["token"], googleAuth["credentials"]))
        print("file 1: {} - {}".format('experimental', googleAuth["experimental"]))
        print("file 2: {} - {}".format('daily_options', googleAuth["daily_options"]))
        
        ''' read application specific configuration file '''
        exc_txt = "\nAn exception occurred - unable to access process configuration file"
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        print("appConfig: {}".format(appConfig))
        
        ''' =============== build user interface based on configuration json file =========== '''
        ui=tk.Tk()
        ui.title('Morning Process Control')
        ''' either of the following will force the window size as specified '''
        #root.geometry("1000x600")
        #root.minsize(1000, 600)
        #ui.geometry("1000x800+10+10")

        ''' ================== create window frames for top level placement ============ '''
        frmSelection = tk.Frame(ui, relief=tk.GROOVE, borderwidth=5)
        frmSelection.pack(fill=tk.BOTH)
        ''' frames within frames '''
        frmProcess = tk.Frame(frmSelection, relief=tk.SUNKEN, borderwidth=5)
        frmProcess.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        frmModel = tk.Frame(frmSelection, relief=tk.RAISED, borderwidth=5)
        frmModel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)        

        frmButtons = tk.Frame(ui, relief=tk.RIDGE, borderwidth=5)
        frmButtons.pack(fill=tk.BOTH)                
        
        exc_txt = "\nAn exception occurred selecting processes to run and setting parameters"
        ''' traditional processes '''
        lblProcessTitle = tk.Label(frmProcess, text="Traditional processes", height=2)
        lblProcessTitle.pack()
        frmTp=[]
        varTp=[]
        btnTp=[]
        frmTpCtrl=[]
        entTpControl=[]
        lblTpControl=[]
        for process in appConfig["processes"]:
            if not process["model"]:
                print("Process name: {}".format(process["name"]))
                
                frmTp.append(tk.Frame(frmProcess))
                frmTp[len(frmTp)-1].pack()
                
                varTp.append(tk.IntVar())
                btnTp.append(tk.Checkbutton(frmTp[len(varTp)-1], \
                                            text=process["name"], \
                                            variable=varTp[len(varTp)-1]))
                btnTp[len(btnTp)-1].pack()
                
                ''' store controls for later processing '''
                processCtrl[process["name"]]={"frame":frmTp[len(frmTp)-1], \
                                              "btnRun":btnTp[len(btnTp)-1], \
                                              "run":varTp[len(varTp)-1], \
                                              "controls":""}
                dControl=dict("")
                if "controls" in process:
                    for control in process["controls"]:
                        for key in control.keys():
                            print("\tcontrol: {}, value:{}".format(key, control[key]))
                            frmTpCtrl.append(tk.Frame(frmTp[len(frmTp)-1]))
                            frmTpCtrl[len(frmTpCtrl)-1].pack()

                            lblTpControl.append(tk.Label(frmTpCtrl[len(frmTpCtrl)-1], text=key))
                            lblTpControl[len(lblTpControl)-1].pack(side=tk.LEFT)

                            entTpControl.append(tk.Entry(frmTpCtrl[len(frmTpCtrl)-1], \
                                                         fg="black", bg="white", width=50))
                            entTpControl[len(entTpControl)-1].pack()
                            entTpControl[len(entTpControl)-1].insert(0, control[key])
                            
                            dictC = {"frame":frmTpCtrl[len(frmTpCtrl)-1], \
                                     "label":lblTpControl[len(lblTpControl)-1], \
                                     "description":key, \
                                     "entry":entTpControl[len(entTpControl)-1]}
                            dControl[key]=dictC
                    processCtrl[process["name"]]["controls"]=dControl

        ''' machine learning models '''
        lblModels = tk.Label(frmModel, text="Trained models", height=2)
        lblModels.pack()
        frmMl=[]
        varMl=[]
        btnMl=[]
        frmMlControl=[]
        entMlControl=[]
        lblMlControl=[]
        for process in appConfig["processes"]:
            if process["model"]:
                print("model name: {}".format(process["name"]))

                frmMl.append(tk.Frame(frmModel))
                frmMl[len(frmMl)-1].pack()
                
                varMl.append(tk.IntVar())
                btnMl.append(tk.Checkbutton(frmMl[len(frmMl)-1], \
                                            text=process["name"], \
                                            variable=varMl[len(varMl)-1]))
                btnMl[len(btnMl)-1].pack()

                ''' store controls for later processing '''
                processCtrl[process["name"]]={"frame":frmMl[len(frmMl)-1], \
                                              "btnRun":btnMl[len(btnMl)-1], \
                                              "run":varMl[len(varMl)-1], \
                                              "controls":""}

                dControl=dict("")
                if "controls" in process:
                    for control in process["controls"]:
                        for key in control.keys():
                            print("\tcontrol: {}, value:{}".format(key, control[key]))
                            frmMlControl.append(tk.Frame(frmMl[len(frmMl)-1]))
                            frmMlControl[len(frmMlControl)-1].pack()

                            lblMlControl.append(tk.Label(frmMlControl[len(frmMlControl)-1], text=key))
                            lblMlControl[len(lblMlControl)-1].pack(side=tk.LEFT)

                            entMlControl.append(tk.Entry(frmMlControl[len(frmMlControl)-1], \
                                                         fg="black", bg="white", width=50))
                            entMlControl[len(entMlControl)-1].pack()
                            entMlControl[len(entMlControl)-1].insert(0, control[key])
        
                            dictC = {"frame":frmMlControl[len(frmMlControl)-1], \
                                     "label":lblMlControl[len(lblMlControl)-1], \
                                     "description":key, \
                                     "entry":entMlControl[len(entMlControl)-1]}
                            dControl[key]=dictC
                    processCtrl[process["name"]]["controls"]=dControl

        ''' ============================= process button press ============================= '''        
        def go_button():
            print("Choices made")
            ui.quit()
            
        ''' =================== widgets in bottom frame =================== '''
        #lblBottom = tk.Label(frmButtons, text="bottom frame", width=50, height=2)
        #lblBottom.pack()
        btnRun=tk.Button(frmButtons, command=go_button, text="Perform selected processes", fg='blue', height=3)
        btnRun.pack()
        
        ''' =================== Interact with user =================== '''
        ui.mainloop()
  
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return processCtrl

if __name__ == '__main__':
    print ("Perform daily operational processes\n")
    
    putSheetName = ""
    callSheetName = ""
    processCtrl = userInterfaceControls()

    for process in processCtrl:
        print("Process: {}, run={}".format(process, processCtrl[process]["run"].get()))
        if processCtrl[process]["run"].get() == 1:
            
            for control in processCtrl[process]["controls"]:
                print("\tcontrol: {} = {}".format(
                                processCtrl[process]["controls"][control]["description"], \
                                processCtrl[process]["controls"][control]["entry"].get()))
    
            if process == MARKET_DATA_UPDATE:
                marketDataUpdate(processCtrl[process]["controls"])
            elif process == DERIVED_MARKET_DATA:
                calculateDerivedData(processCtrl[process]["controls"])
            elif process == SECURED_PUTS:
                putSheetName = marketPutOptions(processCtrl[process]["controls"])
                #putSheetName = "put 20230810 072953"
                if len(putSheetName) > 0:
                    eliminateLowReturnOptions(processCtrl[process]["controls"], putTab=putSheetName)
            elif process == COVERED_CALLS:
                callSheetName = marketCallOptions(processCtrl[process]["controls"])
                #callSheetName = "call 20230810 073519"
                if len(callSheetName) > 0:
                    eliminateLowReturnOptions(processCtrl[process]["controls"], callTab=callSheetName)
            elif process == BOLLINGER_BAND_PREDICTION:
                mlBollingerBandPrediction(process, processCtrl[process]["controls"])
            elif process == MACD_TREND_CROSS:
                mlMACDTrendCross(process, processCtrl[process]["controls"])
                
    
    print ("\nAll requested processes have completed")
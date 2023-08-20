'''
Created on Jul 27, 2023

@author: Brian
'''
import os
import sys
import json
import re
import time
import datetime as dt

from tkinter import *
from tkinter import ttk
from tkinter.ttk import Combobox

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

def mlBollingerBandPrediction(processCtrl):
    exc_txt = "\nAn exception occurred - making predictions using Bollinger Band model"
    
    try:
        print("making daily predictions using the Bollinger Band model")
        pass
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return

def mlMACDTrendCross(processCtrl):
    exc_txt = "\nAn exception occurred - Identifying MACD trend crossing moving average"
    
    try:
        print("performing daily MACD trend crossing moving average process")
        pass
        
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
        app_data = get_ini_data("TDAMERITRADE")
        update_tda_eod_data(app_data['authentication'])
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return 

def calculateDerivedData(processCtrl):
    exc_txt = "\nAn exception occurred - daily derived data process"
    
    try:
        print("performing daily daily derived data process")
        app_data = get_ini_data("TDAMERITRADE")
        updateMarketData(app_data['authentication'], app_data['eod_data'], app_data['market_analysis_data'])
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
        
    return 

def writeOptionsToGsheet(processCtrl, mktOptions, sheetNameBase):    
    ''' find Google sheet ID '''
    googleLocal = get_ini_data("GOOGLE")
    for control in processCtrl.loc["json"]["controls"]:
        if "file" in control:
            fileName = control["file"]
            fileID = googleLocal[fileName]
        if "current holdings data range" in control:
            holdingRange = control["current holdings data range"]
        if "market data range" in control:
            marketDataRange = control["market data range"]
        if "options header range" in control:
            optionsHeader = control["options header range"]
        if "options data range" in control:
            optionsDataRange = control["options data range"]
    
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
    exc_txt = "\nAn exception occurred - tkInterExp - stage 1"
    
    try:
        ROW_2 = 100
        ROW_3 = 400
        ROW_BUTTON = 450
        ROW_HEIGHT = 20
        
        COL_1 = 100
        COL_2 = 200
        COL_3 = 300
        
        FORM_WIDTH = '600'
        FORM_HEIGHT = '500'
        FORM_BORDER = '10'
        FORM_GEOMETRY = FORM_WIDTH + 'x' + FORM_HEIGHT + "+" + FORM_BORDER + "+" + FORM_BORDER

        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to access local AppData information"
        localDirs = get_ini_data("LOCALDIRS")
        gitdir = localDirs['git']
        aiwork = localDirs['aiwork']

        ''' Google APIs '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleAuth = get_ini_data("GOOGLE")
        
        ''' local list of file Google Drive file ID '''
        #localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        localGoogleProject = open(aiwork + "\\" + googleAuth["fileIDs"], "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close

        ''' read application specific configuration file '''
        exc_txt = "\nAn exception occurred - unable to process configuration file"
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        
        ''' ============ set exception description to narrow problem location ============ '''
        exc_txt = "\nAn exception occurred - tkInterExp - window initialization"

        ''' create an empty dataframe to hold the information related to processes that could be performed '''
        cols = [PROCESS_ID, MODEL, PROCESS_DESCRIPTION, RUN, CONFIG]
        processCtrl = pd.DataFrame(columns=cols)
        processCtrl[MODEL] = processCtrl[MODEL].astype(bool)
        processCtrl[RUN] = processCtrl[RUN].astype(bool)

        ''' =================== Create input window =================== '''
        exc_txt = "\nAn exception occurred - unable to create and process window"
        window=Tk()
        window.title('Morning Process Control')
        
        ''' =================== Create all input fields =================== '''
        ''' ============ set exception description to narrow problem location ============ '''
        exc_txt = "\nAn exception occurred - tkInterExp - creating input fields"
        
        lblOps=Label(window, text="Operational processes", fg='blue', font=("ariel", 10))
        lblOps.configure(bg="white")
        lblOps.place(x=COL_1, y=(ROW_2 - ROW_HEIGHT))

        lblML=Label(window, text="Make machine learning predictions", fg='blue', font=("ariel", 10))
        lblML.configure(bg="white")
        lblML.place(x=COL_3, y=(ROW_2 - ROW_HEIGHT))
        
        processDetails = appConfig[PROCESS_CONFIGS]
        
        processCheck = [IntVar()] * len(processDetails)
        processButton = [None] * len(processDetails)

        countProcess = 0
        countModels = 0
        ndxProcess = 0
        for process in processDetails:
            processCheck[ndxProcess] = IntVar()
            processButton[ndxProcess] = Checkbutton(window, text = process[PROCESS_ID], variable = processCheck[ndxProcess])

            if process [MODEL]:
                print("process {} is a machine learning model".format(process[PROCESS_ID]))
                uiX = COL_3
                uiY = ROW_2 + (ROW_HEIGHT * countModels)
                countModels += 1
            else:
                print("process {} is a traditional process".format(process[PROCESS_ID]))
                uiX = COL_1
                uiY = ROW_2 + (ROW_HEIGHT * countProcess)
                countProcess += 1
                
            processButton[ndxProcess].place(x=uiX, y=uiY)
            processData = np.array([process[PROCESS_ID], 
                                    process[MODEL], \
                                    process[PROCESS_DESCRIPTION], \
                                    process[RUN], \
                                    process])
            runIndex = 3
        
            dfTemp = pd.DataFrame([processData], columns=cols)
            processCtrl = pd.concat([processCtrl, dfTemp])

            if 'controls' in process:
                for control in process['controls']:
                    print("\t{} = {}".format(list(control)[0], control.get(list(control)[0])))
            ndxProcess += 1

        ''' =================== create button to process inputs =================== '''
        def go_button():
            for ndx in range (len(processCheck)):
                if processCheck[ndx].get() == 1:
                    processCtrl.iat[ndx, runIndex] = True
            window.quit()

        btn=Button(window, command=go_button, text="Run processes selected", fg='blue')
        btn.place(x=COL_2, y=ROW_BUTTON)

        ''' =================== Interact with user =================== '''
        window.geometry(FORM_GEOMETRY)
        window.mainloop()

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
    
    for ndx in range (len(processCtrl)):
        if processCtrl.iloc[ndx][RUN]:
            if processCtrl.iloc[ndx][PROCESS_ID] == MARKET_DATA_UPDATE:
                marketDataUpdate(processCtrl.iloc[ndx])
            elif processCtrl.iloc[ndx][PROCESS_ID] == DERIVED_MARKET_DATA:
                calculateDerivedData(processCtrl.iloc[ndx])
            elif processCtrl.iloc[ndx][PROCESS_ID] == SECURED_PUTS:
                putSheetName = marketPutOptions(processCtrl.iloc[ndx])
                #putSheetName = "put 20230810 072953"
                if len(putSheetName) > 0:
                    eliminateLowReturnOptions(processCtrl.iloc[ndx], putTab=putSheetName)
            elif processCtrl.iloc[ndx][PROCESS_ID] == COVERED_CALLS:
                callSheetName = marketCallOptions(processCtrl.iloc[ndx])
                #callSheetName = "call 20230810 073519"
                if len(callSheetName) > 0:
                    eliminateLowReturnOptions(processCtrl.iloc[ndx], callTab=callSheetName)
            elif processCtrl.iloc[ndx][PROCESS_ID] == BOLLINGER_BAND_PREDICTION:
                mlBollingerBandPrediction(processCtrl.iloc[ndx])
            elif processCtrl.iloc[ndx][PROCESS_ID] == MACD_TREND_CROSS:
                mlMACDTrendCross(processCtrl.iloc[ndx])
                
    
    print ("\nAll requested processes have completed")
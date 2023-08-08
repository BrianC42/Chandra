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
OUTPUT_FIELDS = "Outputs"
CONFIG = "json"

MARKET_DATA_UPDATE = "Market data update"
DERIVED_MARKET_DATA = "Calculate derived market data"
SECURED_PUTS = "Secured puts"
COVERED_CALLS = "Covered calls"
BOLLINGER_BAND_PREDICTION = "Bollinger Band"
MACD_TREND_CROSS = "MACD Trend"

GOOGLE_SHEET_ID = "Development"

# The ID and range of a sample spreadsheet.
EXP_SPREADSHEET_ID = "1XJNEWZ0uDdjCOvxJItYOhjq9kOhYtacJ3epORFn_fm4"
DAILY_OPTIONS = "1T0yNe6EkLLpwzg_rLXQktSF1PCx1IcCX9hq9Xc__73U"

def mlBollingerBandPrediction():
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

def mlMACDTrendCross():
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

def marketDataUpdate():
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

def calculateDerivedData():
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

def writeOptionsToGsheet(mktOptions, sheetNameBase):    
    #print("Covered calls accessed")
    gSheet = googleSheet()

    HOLDING_RANGE = 'Holdings!A1:ZZ'
    MARKET_DATA_RANGE = 'TD Import Inf!A1:ZZ'
    OPTIONS_HEADER = '!A1:ZZ'
    OPTIONS_DATA = '!A2:ZZ'

    ''' convert data elements to formats suitable for analysis '''
    df_potential_strategies = prepMktOptionsForAnalysis(mktOptions)
    
    dfHoldings = loadHoldings(gSheet, EXP_SPREADSHEET_ID, HOLDING_RANGE)
    dfMarketData = loadMarketDetails(gSheet, EXP_SPREADSHEET_ID, MARKET_DATA_RANGE)        
    df_potential_strategies = calculateFields(df_potential_strategies, dfHoldings, dfMarketData)

    ''' valueInputOption = USER_ENTERED or RAW '''
    sheetMktOptions = prepMktOptionsForSheets(df_potential_strategies)
    
    now = dt.datetime.now()
    timeStamp = ' {:4d}{:0>2d}{:0>2d} {:0>2d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day, \
                                                                    now.hour, now.minute, now.second)        
    sheetName = sheetNameBase + timeStamp
    gSheet.addGoogleSheet(EXP_SPREADSHEET_ID, sheetName)
    gSheet.updateGoogleSheet(EXP_SPREADSHEET_ID, sheetName + OPTIONS_HEADER, sheetMktOptions.columns)
    gSheet.updateGoogleSheet(EXP_SPREADSHEET_ID, sheetName + OPTIONS_DATA, sheetMktOptions)

    return sheetName

def marketPutOptions():
    exc_txt = "\nAn exception occurred - daily put option process"
    
    try:
        print("performing daily put option process")

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
                  
        sheetName = writeOptionsToGsheet(df_potential_strategies, "put")
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return sheetName

def marketCallOptions():
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
        
        sheetName = writeOptionsToGsheet(df_potential_strategies, "call")
        '''
        #print("Covered calls accessed")
        exc_txt = "\nAn exception occurred while updating the Google workbook"
        gSheet = googleSheet()

        # The ID and range of a sample spreadsheet.
        EXP_SPREADSHEET_ID = "1XJNEWZ0uDdjCOvxJItYOhjq9kOhYtacJ3epORFn_fm4"
        DAILY_OPTIONS = "1T0yNe6EkLLpwzg_rLXQktSF1PCx1IcCX9hq9Xc__73U"
        HOLDING_RANGE = 'Holdings!A1:ZZ'
        MARKET_DATA_RANGE = 'TD Import Inf!A1:ZZ'
        OPTIONS_HEADER = '!A1:ZZ'
        OPTIONS_DATA = '!A2:ZZ'
        '''

        ''' convert data elements to formats suitable for analysis '''
        '''
        df_potential_strategies = prepMktOptionsForAnalysis(df_potential_strategies)
        
        dfHoldings = loadHoldings(gSheet, EXP_SPREADSHEET_ID, HOLDING_RANGE)
        dfMarketData = loadMarketDetails(gSheet, EXP_SPREADSHEET_ID, MARKET_DATA_RANGE)        
        df_potential_strategies = calculateFields(df_potential_strategies, dfHoldings, dfMarketData)
        df_potential_strategies = eliminateLowReturnOptions(df_potential_strategies)
        '''

        ''' valueInputOption = USER_ENTERED or RAW '''
        '''
        sheetMktOptions = prepMktOptionsForSheets(df_potential_strategies)
        
        now = dt.datetime.now()
        timeStamp = ' {:4d}{:0>2d}{:0>2d} {:0>2d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day, \
                                                                        now.hour, now.minute, now.second)        
        sheetName = "Calls" + timeStamp
        gSheet.addGoogleSheet(EXP_SPREADSHEET_ID, sheetName)
        gSheet.updateGoogleSheet(EXP_SPREADSHEET_ID, sheetName + OPTIONS_HEADER, sheetMktOptions.columns)
        gSheet.updateGoogleSheet(EXP_SPREADSHEET_ID, sheetName + OPTIONS_DATA, sheetMktOptions)
        '''
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return sheetName

def userInterfaceControls():
    ''' display a user interface to solicit run time selections '''
    exc_txt = "\nAn exception occurred - tkInterExp"
    
    try:
        
        ROW_1 = 50
        ROW_2 = 100
        ROW_3 = 300
        ROW_BUTTON = 400
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
        trainedModels = localDirs['trainedmodels']

        ''' read application specific configuration file '''
        exc_txt = "\nAn exception occurred - unable to process configuration file"
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        
        ''' create an empty dataframe to hold the information related to processes that could be performed '''
        cols = [PROCESS_ID, PROCESS_DESCRIPTION, RUN, CONFIG]
        processCtrl = pd.DataFrame(columns=cols)
        processCtrl[RUN] = processCtrl[RUN].astype(bool)

        ''' =================== Create input window =================== '''
        exc_txt = "\nAn exception occurred - unable to create and process window"
        window=Tk()
        window.title('Morning Process Control')
        
        ''' =================== Create all input fields =================== '''
        lblOps=Label(window, text="Operational processes", fg='blue', font=("ariel", 10))
        lblOps.configure(bg="white")
        lblOps.place(x=COL_1, y=(ROW_2 - ROW_HEIGHT))

        processDetails = appConfig[PROCESS_CONFIGS]
        processCheck = [IntVar()] * len(processDetails)
        processButton = [None] * len(processDetails)
        ndx = 0
        for process in processDetails:
            processData = np.array([process[PROCESS_ID], \
                                    process[PROCESS_DESCRIPTION], \
                                    process[RUN], \
                                    process])
            dfTemp = pd.DataFrame([processData], columns=cols)
            processCheck[ndx] = IntVar()
            processButton[ndx] = Checkbutton(window, text = process[PROCESS_ID], variable = processCheck[ndx])
            processButton[ndx].place(x=COL_1, y=ROW_2 + (ROW_HEIGHT * ndx))
            processCtrl = pd.concat([processCtrl, dfTemp])
            ndx += 1

        lblML=Label(window, text="Make machine learning predictions", fg='blue', font=("ariel", 10))
        lblML.configure(bg="white")
        lblML.place(x=COL_3, y=(ROW_2 - ROW_HEIGHT))

        modelDetails = appConfig[MODEL_CONFIGS]
        mlCheck = [IntVar()] * len(modelDetails)
        mlButton = [None] * len(processDetails)
        ndx = 0
        for model in modelDetails:
            modelData = np.array([model[PROCESS_ID], \
                                    model[PROCESS_DESCRIPTION], \
                                    model[RUN], \
                                    model])
            dfTemp = pd.DataFrame([modelData], columns=cols)
            mlCheck[ndx] = IntVar()
            mlButton[ndx] = Checkbutton(window, text = model[PROCESS_ID], variable = mlCheck[ndx])
            mlButton[ndx].place(x=COL_3, y=ROW_2 + (ROW_HEIGHT * ndx))
            processCtrl = pd.concat([processCtrl, dfTemp])
            ndx += 1

        ''' Select Google sheet file to use for market options details '''
        lblSheet=Label(window, text="Tracking sheet to record and track results", fg='blue', font=("ariel", 10))
        lblSheet.configure(bg="white")
        lblSheet.place(x=COL_1, y=(ROW_3 - ROW_HEIGHT))

        localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close
        radioButton = [None] * len(jsonGoogle["Google sheets"])
        radioValue = [IntVar()] * len(jsonGoogle["Google sheets"])
        ndx = 0
        for sheet in jsonGoogle["Google sheets"]:
            print("Name: {}, ID: {}".format(sheet["name"], sheet["file ID"]))
            radioButton[ndx] = Radiobutton(window, text=sheet["name"], variable = radioValue[ndx])
            radioButton[ndx].place(x=COL_1, y=ROW_3 + (ROW_HEIGHT * ndx))
            ndx += 1

        ''' =================== create button to process inputs =================== '''
        def go_button():
            for ndx in range (len(processCheck)):
                if processCheck[ndx].get() == 1:
                    processCtrl.iat[ndx, RUN_COL] = True
            for ndx in range (len(mlCheck)):
                if mlCheck[ndx].get() == 1:
                    processCtrl.iat[ndx + len(processCheck), RUN_COL] = True
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
                marketDataUpdate()
            elif processCtrl.iloc[ndx][PROCESS_ID] == DERIVED_MARKET_DATA:
                calculateDerivedData()
            elif processCtrl.iloc[ndx][PROCESS_ID] == SECURED_PUTS:
                putSheetName = marketPutOptions()
            elif processCtrl.iloc[ndx][PROCESS_ID] == COVERED_CALLS:
                callSheetName = marketCallOptions()
            elif processCtrl.iloc[ndx][PROCESS_ID] == BOLLINGER_BAND_PREDICTION:
                mlBollingerBandPrediction()
            elif processCtrl.iloc[ndx][PROCESS_ID] == MACD_TREND_CROSS:
                mlMACDTrendCross()
                
    #putSheetName = "put 20230808 073454"
    #callSheetName = "call 20230808 080657"
    if len(putSheetName) > 0 or len(callSheetName) > 0:
        eliminateLowReturnOptions(EXP_SPREADSHEET_ID, putSheetName, callSheetName)
    
    print ("\nAll requested processes have completed")
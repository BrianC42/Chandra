'''
Created on Jul 27, 2023

@author: Brian
'''
import sys
import json

from tkinter import *
from tkinter import ttk
from tkinter.ttk import Combobox

import numpy as np
import pandas as pd

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import update_tda_eod_data

from multiProcess import updateMarketData 

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

def marketPutOptions():
    exc_txt = "\nAn exception occurred - daily put option process"
    
    try:
        print("performing daily put option process")
        pass
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return 

def marketCallOptions():
    exc_txt = "\nAn exception occurred - daily call option process"
    
    try:
        print("performing daily call option process")
        pass
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return

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
    
    ''' create an empty dataframe to hold the information related to processes that could be performed 
    cols = [PROCESS_ID, PROCESS_DESCRIPTION, RUN, MODEL_FILE, OUTPUT_FIELDS, CONFIG]
    processCtrl = pd.DataFrame(columns=cols)    
    '''
    processCtrl = userInterfaceControls()
    
    for ndx in range (len(processCtrl)):
        if processCtrl.iloc[ndx][RUN]:
            if processCtrl.iloc[ndx][PROCESS_ID] == MARKET_DATA_UPDATE:
                marketDataUpdate()
            elif processCtrl.iloc[ndx][PROCESS_ID] == DERIVED_MARKET_DATA:
                calculateDerivedData()
            elif processCtrl.iloc[ndx][PROCESS_ID] == SECURED_PUTS:
                marketPutOptions()
            elif processCtrl.iloc[ndx][PROCESS_ID] == COVERED_CALLS:
                marketCallOptions()
            elif processCtrl.iloc[ndx][PROCESS_ID] == BOLLINGER_BAND_PREDICTION:
                mlBollingerBandPrediction()
            elif processCtrl.iloc[ndx][PROCESS_ID] == MACD_TREND_CROSS:
                mlMACDTrendCross()
    
    print ("\nAll requested processes have completed")
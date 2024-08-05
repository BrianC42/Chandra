from __future__ import print_function

'''
Created on Dec 6, 2021

@author: Brian - copied from: see below
 
#https://youtu.be/6S2v7G-OupA
 
@author: Sreenivas Bhattiprolu

Shows errors on Tensorflow 1.4 and Keras 2.0.8

Works fine in Tensorflow: 2.2.0
    Keras: 2.4.3

dataset: https://finance.yahoo.com/quote/GE/history/
Also try S&P: https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC
'''
from tensorflow.python.training import input

''' Google workspace requirements start '''
import os.path
import datetime as dt
import json
import pandas as pd
import re
from decimal import *

from GoogleSheets import googleSheet
from MarketData import MarketData

from configuration import get_ini_data
from configuration import read_config_json
from Scalers import chandraScaler

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
''' Google workspace requirements end '''

import sys
import glob
import datetime
from datetime import date
from datetime import timedelta

'''
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Combobox
'''
import tkinter as tk

import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing

import tensorflow as tf
import keras

PROCESS_CONFIGS = "processes"
MODEL_CONFIGS = "models"
PROCESS_ID = "name"
PROCESS_DESCRIPTION = "Description"
MODEL = "model"
RUN = "run"
MODEL_FILE = "file"
OUTPUT_FIELDS = "Outputs"
CONFIG = "json"


def linear_regression():
    df_data = pd.read_csv('d:/Brian/AI-Projects/Datasets/linear regression.csv')
    TRAINPCT = 0.8
    train = df_data.loc[ : (len(df_data) * TRAINPCT)]
    test = df_data.loc[(len(df_data) * TRAINPCT) :]
    print("Training shape %s, testing shape %s" % (train.shape, test.shape))
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=64, input_shape=(1, )))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=64))
    model.add(keras.layers.Dense(units=1))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    print("Training shape: x-%s, y-%s" % (train['Feature-x'].shape, train['Noisy-target'].shape))
    history = model.fit(train['Feature-x'], train['Noisy-target'], \
                        epochs=5, batch_size=32, validation_split=0.1, shuffle=True, verbose=2)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='testing loss')
    plt.legend()
    plt.show()
    prediction = model.predict(test['Feature-x'])
    plt.plot(test['Feature-x'], test['Noisy-target'], label='test')
    plt.plot(test['Feature-x'], prediction, linestyle='dashed', label='prediction')
    plt.legend()
    plt.show()
    return

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def sine_wave_regression():
    df_data = pd.read_csv('d:/Brian/AI-Projects/Datasets/regression - sine 10.csv')
    print(df_data.head())
    plt.plot(df_data['FeatureX'], df_data['TargetY'])
    plt.show()
    TRAINPCT = 0.8
    train = df_data.loc[ : (len(df_data) * TRAINPCT)]
    test = df_data.loc[(len(df_data) * TRAINPCT) :]
    print("Training shape %s, testing shape %s" % (train.shape, test.shape))
    TIME_STEPS = 20
    # reshape to [samples, time_steps, n_features]
    X_train, y_train = create_dataset(train[['TargetY']], train.TargetY, TIME_STEPS)
    X_test, y_test = create_dataset(test[['TargetY']], test.TargetY, TIME_STEPS)
    print("Training shapes X:%s y:%s, testing shapes X:%s y:%s" % (X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    OUTPUTDIMENSIONALITY = 10
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=OUTPUTDIMENSIONALITY, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(units=OUTPUTDIMENSIONALITY))
    model.compile(loss='mae', optimizer='adam')
    model.summary()
    history = model.fit(X_train, y_train,
                        epochs=6, batch_size=32, validation_split=0.1,shuffle=False, verbose=2)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    prediction = model.predict(X_test)
    prediction.shape
    testX = test[:len(test) - TIME_STEPS]
    plt.plot(testX.FeatureX, prediction[:, 0], label='prediction', linestyle='dashed')
    plt.plot(testX.FeatureX, testX.TargetY, label='test series')
    plt.legend()
    plt.show()
    return 

def to_sequences(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x)-seq_size):
        #print(i)
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
        
    return np.array(x_values), np.array(y_values)

def digitalSreeni_180():
    dataDir = 'g:/My Drive/Colab Notebooks/data/'
    dataDir = 'd:/Brian/AI-Projects/Datasets/'
    
    dataframe = pd.read_csv(dataDir + 'GE - Yahoo.csv')
    df = dataframe[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'])

    sns.lineplot(x=df['Date'], y=df['Close'])

    print("Start date is: ", df['Date'].min())
    print("End date is: ", df['Date'].max())

    #Change train data from Mid 2017 to 2019.... seems to be a jump early 2017
    train, test = df.loc[df['Date'] <= '2003-12-31'], df.loc[df['Date'] > '2003-12-31']


    #Convert pandas dataframe to numpy array
    #dataset = dataframe.values
    #dataset = dataset.astype('float32') #COnvert values to float

    #LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
    # normalize the dataset
    #scaler = MinMaxScaler() #Also try QuantileTransformer
    scaler = StandardScaler()
    scaler = scaler.fit(train[['Close']])

    train['Close'] = scaler.transform(train[['Close']])
    test['Close'] = scaler.transform(test[['Close']])


    #As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features. 
    #In this example, the n_features is 2. We will make timesteps = 3. 
    #With this, the resultant n_samples is 5 (as the input data has 9 rows).

    seq_size = 30  # Number of time steps to look back 
    #Larger sequences (look further back) may improve forecasting.


    trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
    testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)


    # define Autoencoder model
    #Input shape would be seq_size, 1 - 1 beacuse we have 1 feature. 
    # seq_size = trainX.shape[1]

    # model = Sequential()
    # model.add(LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    # model.add(LSTM(64, activation='relu', return_sequences=False))
    # model.add(RepeatVector(trainX.shape[1]))
    # model.add(LSTM(64, activation='relu', return_sequences=True))
    # model.add(LSTM(128, activation='relu', return_sequences=True))
    # model.add(TimeDistributed(Dense(trainX.shape[2])))

    # model.compile(optimizer='adam', loss='mse')
    # model.summary()

    #Try another model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(keras.layers.Dropout(rate=0.2))

    model.add(keras.layers.RepeatVector(trainX.shape[1]))

    model.add(keras.layers.LSTM(128, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(trainX.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    # fit model
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=2)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()

    #model.evaluate(testX, testY)

    ###########################
    #Anomaly is where reconstruction error is large.
    #We can define this value beyond which we call anomaly.
    #Let us look at MAE in training prediction

    trainPredict = model.predict(trainX)
    trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
    plt.hist(trainMAE, bins=30)
    max_trainMAE = 0.2  #or Define 90% value of max as threshold.

    testPredict = model.predict(testX)
    testMAE = np.mean(np.abs(testPredict - testX), axis=1)
    plt.hist(testMAE, bins=30)

    #Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(test[seq_size:])
    anomaly_df['testMAE'] = testMAE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
    anomaly_df['Close'] = test[seq_size:]['Close']

    #Plot testMAE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['testMAE'])
    sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['max_trainMAE'])

    anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

    #Plot anomalies
    sns.lineplot(x=anomaly_df['Date'], y=scaler.inverse_transform(anomaly_df['Close']))
    sns.scatterplot(x=anomalies['Date'], y=scaler.inverse_transform(anomalies['Close']), color='r')

    return

def buildListofMarketDataFiles():
    
    exc_txt = "\nAn exception occurred - unable to retrieve list of data files"

    try:
        symbols = []
        dataFiles = []
        
        ''' Find local file directories '''
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        dataDir = aiwork + "\\tda\\market_analysis_data"

        if os.path.exists(dataDir):
            fileSpecList = glob.glob(dataDir + "\\" + "*.csv")
            for fileSpec in fileSpecList:
                subStr = fileSpec.split("\\")
                symbol = subStr[len(subStr)-1].split(".")
                if len(symbol) == 2:
                    symbols.append(symbol[0])
                    dataFiles.append(fileSpec)
        else:
            raise NameError("Error: data directory {} dose not exist".format(dataDir))
    
        symList = pd.DataFrame(data=dataFiles, index=symbols, columns=["dataFile"])

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return symList

def iniRead():

    print("\n======================================================= jsonExp enter")
    exc_txt = "\nAn exception occurred - jsonExp"
    
    try:
        # access the <user> \ AppData\Local\Develoment\<appName>.ini file
        
        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to retrieve <user>\\AppData\\Local\\Develoment\\<appName>.ini"
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        gitdir = localDirs['git']
        models = localDirs['trainedmodels']
        print("Local computer directories - localDirs: {}\n\taiwork: {}\n\tgitdir: {}\n\ttrained models: {}".format(localDirs, aiwork, gitdir, models))
    
        ''' identify information related to TD Ameritrade / Schwab APIs '''
        exc_txt = "\nAn exception occurred - unable to retrieve TD Ameritrade authentication information"
        tdaAuth = get_ini_data("TDAMERITRADE")
        print("TD Ameritrade authentication\n\ttdaAuth: {}\n\tauthentication: {}".format(tdaAuth, tdaAuth["authentication"]))

        ''' read application specific configuration file '''
        exc_txt = "\nAn exception occurred - unable to process configuration file"
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        print("appConfig: {}".format(appConfig))
        
        ''' Google APIs '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleAuth = get_ini_data("GOOGLE")
        print("Google authentication\n\ttoken: {}\n\tcredentials: {}".format(googleAuth["token"], googleAuth["credentials"]))
        print("file 1: {} - {}".format('experimental', googleAuth["experimental"]))
        print("file 2: {} - {}".format('daily_options', googleAuth["daily_options"]))
        
        ''' local list of file Google Drive file ID '''
        #localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        localGoogleProject = open(aiwork + "\\" + googleAuth["fileIDs"], "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close

        '''
        cols = [PROCESS_ID, PROCESS_DESCRIPTION, RUN, MODEL_FILE, OUTPUT_FIELDS, CONFIG]
        processCtrl = pd.DataFrame(columns=cols)
        '''
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def dailyProcess():
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

def dictexp():
    d1 = {"file":"a", "scaler":"b"}
    d2 = {"OTM":0.8, "hdr":"!A2:ZZ"}
    d3 = {"outputs":"10%","features":["f1", "F2"]}
    dsum={"procA":d1, "procB":d2, "procC":d3}
    
    return

def gSheetService():

    try:
        ''' Google API authentication '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleAuth = get_ini_data("GOOGLE")
        print("Google authentication\n\ttoken: {}\n\tcredentials: {}".format(googleAuth["token"], googleAuth["credentials"]))
        gSheetService = googleSheet()
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return(gSheetService)

if __name__ == '__main__':
    try:
        print("==================================================== Code experimentation starting")
        exc_txt = "\nAn exception occurred"
    
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
            
        if True:
            
            ''' Google drive file details '''
            exc_txt = "\nAn exception occurred - unable to access Google sheet"
            googleAuth = get_ini_data("GOOGLE")
            googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
            
            '''
            Authenticate with Google workplace and establish a connection to Google Drive API
            exc_txt = "\nAn exception occurred - unable to authenticate with Google"
            gSheets = googleSheet()
            '''
            
            ''' 
            Use the connetion to Google Drive API to read sheet data
            Find file ID of file used for development 
            
            sheetID = googleDriveFiles["Google IDs"]["Market Data"]["Development"]
            print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Data"]["Development"]))
            print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Data"]["Production"]))
            cellRange = 'Stock Information!A2:C999'
            cellValues = gSheets.readGoogleSheet(sheetID, cellRange)
            
            # Create list of symbols for market data request
            mktDataSymbols = []
            mktPuts = []
            mktCalls = []
            for i in range (0, len(cellValues)):
                mktDataSymbols.append(cellValues.iat[i, 0])
                if cellValues.iat[i, 2] == "1 - Holding":
                    # Create list of symbols for market put option request
                    mktPuts.append(cellValues.iat[i, 0])
                elif cellValues.iat[i, 2] == "4 - Buy":
                    # Create list of symbols for market call option request
                    mktCalls.append(cellValues.iat[i, 0])
            '''
    
            '''
            Authenticate with the market data service
            '''
            marketData = MarketData()
            marketData.requestMarketData(symbolList=["AAPL", "MSFT"])
            '''
            Use the market data service to look up
                basic market data
                put option market data
                call option market data
            '''
            
        else:
            dictexp()
            processCtrl = iniRead()                   # Experimentation / development of configuration json file
        
            ''' test daily process start '''
            processCtrl = dailyProcess()
            for process in processCtrl:
                print("Process: {}, run={}".format(process, processCtrl[process]["run"].get()))
                if processCtrl[process]["run"].get() == 1:
                    for control in processCtrl[process]["controls"]:
                        #for key in processCtrl[process]["controls"][control]:
                        print("\tcontrol: {} = {}".format(
                                        processCtrl[process]["controls"][control]["description"], \
                                        processCtrl[process]["controls"][control]["entry"].get()))
            ''' test daily process end '''
    
            dataFiles = buildListofMarketDataFiles()
    
            digitalSreeni_180()
            linear_regression()
            sine_wave_regression()
            # autoKeras()
            pass
        
        print("\n==================================================== Code experimentation ending")
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

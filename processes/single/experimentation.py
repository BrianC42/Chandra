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
import sys
import glob
from functools import partial
import os.path
import json
import re
from decimal import *

import numpy as np
import pandas as pd
import pickle

import time
import datetime
from datetime import date
from datetime import timedelta

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing

import tensorflow as tf
import keras
from tensorflow.python.training import input
import tkinter
from tkinter import *
from tkinter import ttk
import requests_oauthlib
from oauthlib import oauth2
from oauthlib.oauth2 import WebApplicationClient
''' Google workspace requirements start '''
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
''' Google workspace requirements end '''

from GoogleSheets import googleSheet
from MarketData import MarketData
from OptionChain import OptionChain
from FinancialInstrument import FinancialInstrument
from financialDataServices import financialDataServices
from Scalers import chandraScaler

from configuration import get_ini_data
from configuration import read_config_json


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
        ui=Tk()
        ui.title('Morning Process Control')
        ''' either of the following will force the window size as specified '''
        #root.geometry("1000x600")
        #root.minsize(1000, 600)
        #ui.geometry("1000x800+10+10")

        ''' ================== create window frames for top level placement ============ '''
        frmSelection = Frame(ui, relief=GROOVE, borderwidth=5)
        frmSelection.pack(fill=BOTH)
        ''' frames within frames '''
        frmProcess = Frame(frmSelection, relief=SUNKEN, borderwidth=5)
        frmProcess.pack(side=LEFT, fill=BOTH, expand=True)
        frmModel = Frame(frmSelection, relief=RAISED, borderwidth=5)
        frmModel.pack(side=RIGHT, fill=BOTH, expand=True)        

        frmButtons = Frame(ui, relief=RIDGE, borderwidth=5)
        frmButtons.pack(fill=BOTH)                
        
        exc_txt = "\nAn exception occurred selecting processes to run and setting parameters"
        ''' traditional processes '''
        lblProcessTitle = Label(frmProcess, text="Traditional processes", height=2)
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
                
                frmTp.append(Frame(frmProcess))
                frmTp[len(frmTp)-1].pack()
                
                varTp.append(IntVar())
                btnTp.append(Checkbutton(frmTp[len(varTp)-1], \
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
                            frmTpCtrl.append(Frame(frmTp[len(frmTp)-1]))
                            frmTpCtrl[len(frmTpCtrl)-1].pack()

                            lblTpControl.append(Label(frmTpCtrl[len(frmTpCtrl)-1], text=key))
                            lblTpControl[len(lblTpControl)-1].pack(side=LEFT)

                            entTpControl.append(Entry(frmTpCtrl[len(frmTpCtrl)-1], \
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
        lblModels = Label(frmModel, text="Trained models", height=2)
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

                frmMl.append(Frame(frmModel))
                frmMl[len(frmMl)-1].pack()
                
                varMl.append(IntVar())
                btnMl.append(Checkbutton(frmMl[len(frmMl)-1], \
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
                            frmMlControl.append(Frame(frmMl[len(frmMl)-1]))
                            frmMlControl[len(frmMlControl)-1].pack()

                            lblMlControl.append(Label(frmMlControl[len(frmMlControl)-1], text=key))
                            lblMlControl[len(lblMlControl)-1].pack(side=LEFT)

                            entMlControl.append(Entry(frmMlControl[len(frmMlControl)-1], \
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
        #lblBottom = Label(frmButtons, text="bottom frame", width=50, height=2)
        #lblBottom.pack()
        btnRun=Button(frmButtons, command=go_button, text="Perform selected processes", fg='blue', height=3)
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

def show_data():
    #authorizationCode = input_field.get()
    authorizationCode = True
    if authorizationCode:
        encodedIDSecret = "..."
        postTxt = 'curl -X POST https://api.schwabapi.com/v1/oauth/token ^\n' + \
                        '-H "Authorization: Basic ' + encodedIDSecret + '" ^\n' + \
                        '-H "Content-Type: application/x-www-form-urlencoded" ^\n' + \
                        '-d "grant_type=authorization_code&code=' + authorizationCode + '&redirect_uri=https://127.0.0.1"\n'

        #result_label.config(text=f"Entered data: {postTxt}")
    else:
        pass
        #result_label.config(text="No data entered")

'''    ================ OAuth flow tkInter UI development - start ===================== '''
def formatCurl(redirecturi, curlText):
    encodedIDSecret = "xxxEncodedClientID_Secretxxx"
    reduri = redirecturi.get(1.0, tkinter.END)
    
    print("redirect uri:", reduri)
        
    authCode = re.search(r'code=(.*?)&session', reduri)
    print("Authorization:", authCode)
    
    if authCode:
        print('curl -X POST https://api.schwabapi.com/v1/oauth/token ^')
        print('-H "Authorization: Basic ' + encodedIDSecret + '" ^')
        print('-H "Content-Type: application/x-www-form-urlencoded" ^')
        print('-d "grant_type=authorization_code&code=' + authCode.group(1) + '&redirect_uri=https://127.0.0.1"')
        
        curlCmd = 'curl -X POST https://api.schwabapi.com/v1/oauth/token ^\n' + \
                  '-H "Authorization: Basic ' + encodedIDSecret + '" ^\n' + \
                  '-H "Content-Type: application/x-www-form-urlencoded" ^\n' + \
                  '-d "grant_type=authorization_code&code=' + authCode.group(1) + '&redirect_uri=https://127.0.0.1"\n'
        
        curlText.replace(1.0, tkinter.END, curlCmd)
        
        print("Text set to:\n", curlText.get(1.0, tkinter.END))
    
    else:
        pass

    return

def saveAuthorizations(authorizationResponse):
    resp = authorizationResponse.get(1.0, tkinter.END)
    print("Authorization response:\n", resp)
    
    respJson = json.loads(resp)
    exp = respJson["expires_in"]
    tokenType = respJson["token_type"]
    scope = respJson["scope"]
    refreshToken = respJson["refresh_token"]
    accessToken = respJson["access_token"]
    idToken = respJson["id_token"]
    
    acquired = time.time()
    # (60 * 30) reduction to minimize chances of expiration during a process
    refreshExpires = acquired + (7*24*60*60) - (60*30)
    # (60 * 5) reduction to minimize chances of expiration during an access request
    accessExpires = acquired + exp - (60*5)
    
    print("access token\n", accessToken)
    print("expires in: ", exp)
    print("token scope: ", scope)
    print("token type: ", tokenType)
    print("refresh token:\n", refreshToken)
    print("id token\n", idToken)
    print("acquired: ", acquired)
    print("refresh expires", refreshExpires)
    print("access expires", accessExpires)
    
    return

def authorizationInterface():
    try:
        exc_txt = "Exception occurred displaying the authorization code UI"
        ROW_OAUTH_INSTRUCTION = 1
        ROW_OAUTH_URI = 2
        ROW_PASTE_INSTRUCTION = 3
        ROW_OAUTH_REDIRECTION = 4
        ROW_FORMAT_BUTTON = 5
        ROW_PASTE_CURL_CMD = 6
        ROW_CURL_TEXT = 7
        TBD1 = 8
        ROW_AUTHORIZATION_RESPONSE = 10
        ROW_PASTE_RESPONSE_LABEL = 9
        ROW_SAVE_AUTHORIZATION_RESPONSE = 11

        OAuthService = 'https://api.schwabapi.com/v1/oauth/authorize'
        clientID = '?client_id=' + 'xxx-----------------------clientID---------------------xxx'
        redirectUri = '&redirect_uri=' + 'https://127.0.0.1'
        encodedIDSecret = "EncodedClientID_Secret"
        
        OAuthFlowService = OAuthService + clientID + redirectUri

        # Create the main window
        uiRoot = Tk()
        uiRoot.title("Authorization OAuth Flow Data")
        uiRoot.columnconfigure(0, weight=1)
        uiRoot.rowconfigure(0, weight=1)

        mainframe = ttk.Frame(uiRoot, padding="3 3 12 12")
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))

        uri = StringVar()
        redirectUri = StringVar()
        
        oauthURI = Text(mainframe, width=100, height=4)
        oauthURI.insert(1.0, OAuthFlowService)
        oauthURI.grid(column=1, row=ROW_OAUTH_URI, sticky=(W, E))
        
        Inst1 = "Preparation:\n\tOpen a web browser page\n\tOpen the Windows cmd shell"
        Inst2 = "\nStep 1: Paste the following uri into the browser search bar to initiate the OAuth flow"
        Inst3 = "\nStep 2: When the authenticating server redirects the browser paste the search bar uri to the redirect box"
        Inst4 = "\nStep 3: Click the Format Post button"
        Inst5 = "\nStep 4: WITHIN 30 SECONDS paste the curl cmd lines into the Windows cmd shell"
        Inst6 = "\nStep 5: Paste the response from the authentication server into the Authentication box"
        Inst7 = "\nStep 6: Click the Save Authorization button"
        instructions = Inst1 + Inst2 +  Inst3 +  Inst4 +  Inst5 +  Inst6 +  Inst7
        
        step1Inst = "Paste redirect uri here"
        step4Inst = "paste the curl command below into a cmd window WITHIN 30 SECONDS"
        step5Inst = "Paste the response from the authentication server below"
        
        ttk.Label(mainframe, text=instructions).grid(column=1, row=ROW_OAUTH_INSTRUCTION, sticky=W)
        ttk.Label(mainframe, text=step1Inst).grid(column=1, row=ROW_PASTE_INSTRUCTION, sticky=W)
        ttk.Label(mainframe, text=step4Inst).grid(column=1, row=ROW_PASTE_CURL_CMD, sticky=W)
        ttk.Label(mainframe, text=step5Inst).grid(column=1, row=ROW_PASTE_RESPONSE_LABEL, sticky=W)
        
        # multi-line text box to hold the formatted curl cmd
        curlTxt = Text(mainframe, width=100, height=6)
        curlTxt.grid(column=1, row=ROW_CURL_TEXT, sticky=(W, E))
        
        # multi-line text box and button to receive the redirection uri
        uri_entry = Text(mainframe, width=100, height=4)
        uri_entry.grid(column=1, row=ROW_OAUTH_REDIRECTION, sticky=(W, E))
        # button to format the redirection uri into an OAuth post for the authentication server
        ttk.Button(mainframe, text="Format post", command=partial(formatCurl, uri_entry, curlTxt)). \
                    grid(column=1, row=ROW_FORMAT_BUTTON, sticky=W)
        
        # multi-line text box to paste the authorization server response into
        authorizationResponse = Text(mainframe, width=100, height=4)
        authorizationResponse.grid(column=1, row=ROW_AUTHORIZATION_RESPONSE, sticky=(W, E))
        # button to save the authorization response for future processing
        ttk.Button(mainframe, text="Save Authorization", command=partial(saveAuthorizations, authorizationResponse)). \
                    grid(column=1, row=ROW_SAVE_AUTHORIZATION_RESPONSE, sticky=W)

        for child in mainframe.winfo_children(): 
            child.grid_configure(padx=5, pady=5)
        
        uri_entry.focus()
        #uiRoot.bind("<Return>", formatCurl)
    
        # Start the GUI event loop
        uiRoot.mainloop()


    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return
'''    ================ OAuth flow tkInter UI development - end ===================== '''

'''    ================ tkInter experiments - start ===================== '''
def formatPost(feet, meters):
    try:
        value = float(feet.get())
        meters.set(int(0.3048 * value * 10000.0 + 0.5)/10000.0)
    except ValueError:
        pass
    
def callSample(sampleOption):
    exc_txt = "callSample exception"
    try:
        jsonTxt = sampleOption.get(1.0, END)
        print(jsonTxt)
        optionJson = json.loads(jsonTxt)
        
        print("symbol: ", optionJson["symbol"])
        putMap = optionJson["putExpDateMap"]
        for putDate in putMap:
            print(putDate)
            
        for exp_date, options in optionJson['putExpDateMap'].items():
            for strike_price, options_data in options.items():
                for option in options_data:
                    bid = option['bid']            
                    print("expiration", exp_date)
                    print("strike", strike_price)
                    print("bid=", bid)
        
        return 
    
    except ValueError:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

def tkExp():
    root = Tk()
    root.title("options analysis")
    
    mainframe = ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ''' entry and display value widgets '''    
    feet = StringVar()
    meters = StringVar()

    ''' single line input widgets '''
    feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
    feet_entry.grid(column=1, row=1, sticky=(W, E))
    
    ''' multi-line input widgets '''
    text1 = Text(mainframe, width=100, height=10)
    text1.grid(column=1, row=2, sticky=(W, E))
    
    ''' button widgets '''
    #ttk.Button(mainframe, text="Calculate", command=formatPost).grid(column=3, row=3, sticky=W)
    ttk.Button(mainframe, text="Calculate", command=partial(formatPost, feet, meters)).grid(column=1, row=3, sticky=W)
    ttk.Button(mainframe, text="analyze", command=partial(callSample, text1)).grid(column=1, row=3, sticky=W)
    
    ''' single line output / display widgets '''
    ttk.Label(mainframe, textvariable=meters).grid(column=1, row=4, sticky=(W, E))
    ttk.Label(mainframe, text="fixed text 1").grid(column=1, row=5, sticky=W)
    ttk.Label(mainframe, text="fixed text 2").grid(column=1, row=6, sticky=E)
    ttk.Label(mainframe, text="fixed text 3").grid(column=1, row=7, sticky=W)
    
    ''' presentation control '''
    for child in mainframe.winfo_children(): 
        child.grid_configure(padx=5, pady=5)
    
    feet_entry.focus()
    root.bind("<Return>", formatPost)
    
    root.mainloop()
    return 
'''    ================ tkInter experiments - end ===================== '''
    
'''    ================ develop market data retrieval and archival - start ===================== '''
def develop_market_data():
            
    for symbol in ["C","AAPL"]:
    #for symbol in symbols:
        exc_txt = "\nAn exception occurred - with symbol: {}".format(symbol)
    
        instrumentDetails = FinancialInstrument(symbol)
        print("Symbol: {}, type: {}, description: {}".format(instrumentDetails.symbol, \
                                                             instrumentDetails.assetType, \
                                                             instrumentDetails.description))
        
        mktData = MarketData(symbol, useArchive=False, periodType="month", period="2", frequencyType="daily", frequency="1")
        print("\t{} candles returned, candles archived {}".format(len(mktData.marketDataJson["candles"]), mktData.candleCount()))
        for ndx in [0, mktData.candleCount() - 1]:
            candle = mktData.iloc(ndx)
            print("Market data: symbol: {}, date/time: {} {}\n\topen: {}, close: {}, , high: {}, low: {}, volume: {}".format( \
                    mktData.symbol, candle.candleDateValue, candle.candleDateTimeStr, \
                    candle.candleOpen, candle.candleClose, \
                    candle.candleHigh, candle.candleLow, \
                    candle.candleVolume) )
        df_candles = mktData.dataFrameCandles()
        print("Market data head\n{}".format(df_candles.head()))
        print("Market data tail\n{}".format(df_candles.tail()))
        ''' date/time: 1661144400.0 2022-08-22 '''
        ''' date/time: 1722488400.0 2024-08-01 '''
        ''' date/time: 1724130000.0 2024-08-20 '''
        mktData = MarketData(symbol, useArchive=True)
        print("\t{} candles returned, candles archived {}".format(len(mktData.marketDataJson["candles"]), mktData.candleCount()))
        for ndx in [0, mktData.candleCount() - 1]:
            candle = mktData.iloc(ndx)
            print("Market data: symbol: {}, date/time: {} {}\n\topen: {}, close: {}, , high: {}, low: {}, volume: {}".format( \
                    mktData.symbol, candle.candleDateValue, candle.candleDateTimeStr, \
                    candle.candleOpen, candle.candleClose, \
                    candle.candleHigh, candle.candleLow, \
                    candle.candleVolume) )
        df_candles = mktData.dataFrameCandles()
        print("Market data head\n{}".format(df_candles.head()))
        print("Market data tail\n{}".format(df_candles.tail()))
        
        '''
        for candle in mktData:
            print("Market data: symbol: {}, date/time: {} {}, open: {}, close: {}, volume: {}".format( \
                  mktData.symbol, candle.candleDateValue, candle.candleDateTimeStr, \
                  candle.candleOpen, candle.candleClose, candle.candleVolume) )
        '''
                
    return
'''    ================ develop market data retrieval and archival - end ===================== '''
    
if __name__ == '__main__':
    try:
        print("======================= Code experimentation starting =============================")
        exc_txt = "\nAn exception occurred"
    
        if True:
            localDirs = get_ini_data("LOCALDIRS")
            aiwork = localDirs['aiwork']
            
            ''' ================= Google workspace development start ================ '''
            ''' Google drive file details '''
            exc_txt = "\nAn exception occurred - unable to access Google sheet"
            googleAuth = get_ini_data("GOOGLE")
            googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
            
            ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
            exc_txt = "\nAn exception occurred - unable to authenticate with Google"
            gSheets = googleSheet()
                        
            ''' 
            Use the connetion to Google Drive API to read sheet data
            Find file ID of file used for development 
            '''
            sheetID = googleDriveFiles["Google IDs"]["Market Data"]["Development"]
            print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Data"]["Development"]))
            print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Data"]["Production"]))
            instrumentCells = 'TD Import Inf!A1:p999'
            instrumentCellValues = gSheets.readGoogleSheet(sheetID, instrumentCells)
            symbols = instrumentCellValues['Symbol'].astype(str).str.split(',', n=1).str[0]
            
            cellRange = 'Stock Information!A2:C999'
            cellValues = gSheets.readGoogleSheet(sheetID, cellRange)

            '''
            instrumentCellValues.columns
            Index(['Symbol', 'Change %', 'Last Price', 'Dividend Yield', 'Dividend',
                   'Dividend Ex-Date', 'P/E Ratio', '52 Week High', '52 Week Low',
                   'Volume', 'Sector', 'Next Earnings Date', 'Morningstar', 'Market Cap',
                   'Schwab Equity Rating', 'Argus Rating']
                   
            Missing data elements
                'Change %',
                'Sector', 'Next Earnings Date', 
                'Morningstar', 'Schwab Equity Rating', 'Argus Rating'
            
            MarketData elements
                'Last Price' : close
                'Volume'
            
            FinancialInstrument data elements
                'symbol': 'AAPL' 'Symbol'
                'high52': 237.23,  '52 Week High'
                'low52': 164.075 '52 Week Low'
                'dividendYield': 0.44269, 'Dividend Yield'
                'dividendAmount': 1.0, 'Dividend'
                'dividendDate': '2024-08-12 00:00:00.0', 'Dividend Ex-Date'
                'peRatio': 34.49389, 'P/E Ratio'
                'marketCap': 3434462506930.0, 'Market Cap'
                
                {
                'symbol': 'AAPL', 
                'high52': 237.23, 
                'low52': 164.075, 
                'dividendAmount': 1.0, 
                'dividendYield': 0.44269, 
                'dividendDate': '2024-08-12 00:00:00.0', 
                'peRatio': 34.49389, 
                'pegRatio': 112.66734, 
                'pbRatio': 48.06189, 
                'prRatio': 8.43929, 
                'pcfRatio': 23.01545, 
                'grossMarginTTM': 45.962, 
                'grossMarginMRQ': 46.2571, 
                'netProfitMarginTTM': 26.4406, 
                'netProfitMarginMRQ': 25.0043, 
                'operatingMarginTTM': 26.4406, 
                'operatingMarginMRQ': 25.0043, 
                'returnOnEquity': 160.5833, 
                'returnOnAssets': 22.6119, 
                'returnOnInvestment': 50.98106, 
                'quickRatio': 0.79752, 
                'currentRatio': 0.95298, 
                'interestCoverage': 0.0, 
                'totalDebtToCapital': 51.3034, 
                'ltDebtToEquity': 151.8618, 
                'totalDebtToEquity': 129.2138, 
                'epsTTM': 6.56667, 
                'epsChangePercentTTM': 10.3155, 
                'epsChangeYear': 0.0, 
                'epsChange': 0.0, 
                'revChangeYear': -2.8005, 
                'revChangeTTM': 0.4349, 
                'revChangeIn': 0.0, 
                'sharesOutstanding': 15204137000.0, 
                'marketCapFloat': 0.0, 
                'marketCap': 3434462506930.0, 
                'bookValuePerShare': 4.38227, 
                'shortIntToFloat': 0.0, 
                'shortIntDayToCover': 0.0, 
                'divGrowthRate3Year': 0.0, 
                'dividendPayAmount': 0.25, 
                'dividendPayDate': '2024-08-15 00:00:00.0', 
                'beta': 1.24364, 
                'vol1DayAvg': 0.0, 
                'vol10DayAvg': 0.0, 
                'vol3MonthAvg': 0.0, 
                'avg10DaysVolume': 47812576, 
                'avg1DayVolume': 60229630, 
                'avg3MonthVolume': 64569166, 
                'declarationDate': '2024-08-01 00:00:00.0', 
                'dividendFreq': 4, 
                'eps': 6.13, 
                'dtnVolume': 40687813, 
                'nextDividendPayDate': '2024-11-15 00:00:00.0', 
                'nextDividendDate': '2024-11-12 00:00:00.0', 
                'fundLeverageFactor': 0.0
                }, 

            '''
            develop_market_data()
            '''
            actionCategory = cellValues[cellValues['Symbol']==symbol]['Action Category']
            if actionCategory.iloc[0] == "1 - Holding":
                print("Call options for {}".format(symbol))
                options = OptionChain(symbol, optionType="Call", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            elif actionCategory.iloc[0] == "4 - Buy":
                print("Put options for {}".format(symbol))
                options = OptionChain(symbol, optionType="Put", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            print("\tOption details".format("TBD"))
            '''
            
            '''
            for option in options:
                print("Option chain for {}: type: {} expiration: {} strike: {} bid: {} ask: {}".format( \
                        option.symbol, option.putCall, option.expirationDate, \
                        option.strikePrice, option.bidPrice, option.askPrice))
            '''
            ''' ================= Google workspace development end ================ '''
                    
            print("\n======================== Code experimentation ending ============================")
        
        else:
            ''' ================= TDA / Schwab date / time exp ================ '''
            aa1 = 962946000000.0 # TDA date / time from legacy csv file
            aa2 = 1699855200000.0
            t1 = 1720501200.0 # Schwab date / time from authorization flow
            t2 = 1723179600.0
            t3 = time.time()
            
            straa1 = time.gmtime(aa1 / 1000)
            straa2 = time.gmtime(aa2 / 1000)
            strt1 = time.gmtime(t1)
            strt2 = time.gmtime(t2)
            strt3 = time.gmtime(t3)
            
            print(time.asctime(straa1))
            print(time.asctime(straa2))
            print(time.asctime(strt1))
            print(time.asctime(strt2))
            print(time.asctime(strt3))
            ''' ======================================================= '''
            ''' ================= authorization UI exp ================ '''
            tkExp()
            authorizationInterface()
                        
            ''' ======================================================= '''

            ''' ================= Google workspace development start ================ '''
            ''' Google drive file details '''
            exc_txt = "\nAn exception occurred - unable to access Google sheet"
            googleAuth = get_ini_data("GOOGLE")
            googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
            
            ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
            exc_txt = "\nAn exception occurred - unable to authenticate with Google"
            gSheets = googleSheet()
                        
            ''' 
            Use the connetion to Google Drive API to read sheet data
            Find file ID of file used for development 
            '''
            sheetID = googleDriveFiles["Google IDs"]["Market Data"]["Development"]
            print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Data"]["Development"]))
            print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Data"]["Production"]))
            cellRange = 'Stock Information!A2:C999'
            cellValues = gSheets.readGoogleSheet(sheetID, cellRange)
            
            mktData = MarketData("AAPL", periodType="month", period="1", frequencyType="daily", frequency="1")
            for candle in mktData:
                print("Market data: symbol: {}, date/time: {} {}, open: {}, close: {}, volume: {}".format( \
                      mktData.symbol, candle.candleDateValue, candle.candleDateTimeStr, \
                      candle.candleOpen, candle.candleClose, candle.candleVolume) )

            options = OptionChain("AAPL", optionType="Both", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            for option in options:
                print("Option chain for {}: type: {} expiration: {} strike: {} bid: {} ask: {}".format( \
                        option.symbol, option.putCall, option.expirationDate, \
                        option.strikePrice, option.bidPrice, option.askPrice))
            ''' ================= Google workspace development end ================ '''

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
        
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

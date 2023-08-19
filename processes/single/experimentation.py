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

''' Google workspace requirements start '''
import os.path
import datetime as dt
import json
import pandas as pd
import re
from decimal import *

from GoogleSheets import googleSheet

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

from tkinter import *
from tkinter import ttk
from tkinter.ttk import Combobox

import numpy as np
import pandas as pd
import pickle

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing

import tensorflow as tf
from tensorflow import keras
import autokeras as ak

# The ID and range of a sample spreadsheet.
EXP_SPREADSHEET_ID = "1XJNEWZ0uDdjCOvxJItYOhjq9kOhYtacJ3epORFn_fm4"
DAILY_OPTIONS = "1T0yNe6EkLLpwzg_rLXQktSF1PCx1IcCX9hq9Xc__73U"
        
PROCESS_CONFIGS = "processes"
MODEL_CONFIGS = "models"
PROCESS_ID = "name"
PROCESS_DESCRIPTION = "Description"
MODEL = "model"
RUN = "run"
MODEL_FILE = "file"
OUTPUT_FIELDS = "Outputs"
CONFIG = "json"



def autoKeras():
    
    house_dataset = fetch_california_housing()
    df = pd.DataFrame(np.concatenate((house_dataset.data, house_dataset.target.reshape(-1, 1)), axis=1), \
                      columns=house_dataset.feature_names + ["Price"], )
    train_size = int(df.shape[0] * 0.9)
    df[:train_size].to_csv("train.csv", index=False)
    df[train_size:].to_csv("eval.csv", index=False)
    train_file_path = "train.csv"
    test_file_path = "eval.csv"
    
    # Initialize the structured data regressor.
    reg = ak.StructuredDataRegressor(overwrite=True, max_trials=3)  # It tries 3 different models.
    # Feed the structured data regressor with training data.
    # The path to the train.csv file.
    # The name of the label column.
    reg.fit(train_file_path,"Price",epochs=10,)
    # Predict with the best model.
    predicted_y = reg.predict(test_file_path)
    # Evaluate the best model with testing data.
    print(reg.evaluate(test_file_path, "Price"))
    
    '''
    Data Format
    The AutoKeras StructuredDataRegressor is quite flexible for the data format.

    The example above shows how to use the CSV files directly. Besides CSV files, 
    it also supports numpy.ndarray, pandas.DataFrame or tf.data.Dataset. 
    The data should be two-dimensional with numerical or categorical values.

    For the regression targets, it should be a vector of numerical values. 
    AutoKeras accepts numpy.ndarray, pandas.DataFrame, or pandas.Series.

    The following examples show how the data can be prepared with 
    numpy.ndarray, pandas.DataFrame, and tensorflow.data.Dataset.
    '''
    # x_train as pandas.DataFrame, y_train as pandas.Series
    x_train = pd.read_csv(train_file_path)
    print(type(x_train))  # pandas.DataFrame
    y_train = x_train.pop("Price")
    print(type(y_train))  # pandas.Series

    # You can also use pandas.DataFrame for y_train.
    y_train = pd.DataFrame(y_train)
    print(type(y_train))  # pandas.DataFrame

    # You can also use numpy.ndarray for x_train and y_train.
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    print(type(x_train))  # numpy.ndarray
    print(type(y_train))  # numpy.ndarray

    # Preparing testing data.
    x_test = pd.read_csv(test_file_path)
    y_test = x_test.pop("Price")

    # It tries 10 different models.
    reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
    # Feed the structured data regressor with training data.
    reg.fit(x_train, y_train, epochs=10)
    # Predict with the best model.
    predicted_y = reg.predict(x_test)
    # Evaluate the best model with testing data.
    print(reg.evaluate(x_test, y_test))   
    
    ''' The following code shows how to convert numpy.ndarray to tf.data.Dataset. '''
    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    reg = ak.StructuredDataRegressor(max_trials=3, overwrite=True)
    # Feed the tensorflow Dataset to the regressor.
    reg.fit(train_set, epochs=10)
    # Predict with the best model.
    predicted_y = reg.predict(test_set)
    # Evaluate the best model with testing data.
    print(reg.evaluate(test_set))
    
    '''
    You can also specify the column names and types for the data as follows. 
    The column_names is optional if the training data already have the column names, 
    e.g. pandas.DataFrame, CSV file. 
    Any column, whose type is not specified will be inferred from the training data.
    '''
    # Initialize the structured data regressor.
    reg = ak.StructuredDataRegressor(
        column_names=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude",],
        column_types={"MedInc": "numerical", "Latitude": "numerical"},
        max_trials=10,  # It tries 10 different models.
        overwrite=True,
        )
    
    return

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
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(tf.keras.layers.Dropout(rate=0.2))

    model.add(tf.keras.layers.RepeatVector(trainX.shape[1]))

    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(trainX.shape[2])))
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

def saveMachineLearningSignal(signals):
    exc_txt = "\nAn exception occurred - saving machine learning signal"
    try:
        EXP_SPREADSHEET_ID = "1XJNEWZ0uDdjCOvxJItYOhjq9kOhYtacJ3epORFn_fm4"
        HEADER_RANGE = 'ML Signals!A1:Z2'
        DATA_RANGE = 'ML Signals!A3:Z'

        ''' read sheet current cells - do not overwrite these '''
        gSheet = googleSheet()
        result = gSheet.googleSheet.values().get(spreadsheetId=EXP_SPREADSHEET_ID, range=DATA_RANGE).execute()
        values = result.get('values', [])

        for signal in signals:
            outputStr = signal['outputs'][0]
            predictionStr = str(signal['prediction'][0])
            newSignal = [signal['name'], signal['symbol'], outputStr, dt.datetime.now().strftime("%m/%d/%y"), predictionStr]
            values.append(newSignal)
        
        requestBody = {'values': values}
        result = gSheet.googleSheet.values().update(spreadsheetId=EXP_SPREADSHEET_ID, range=DATA_RANGE, \
                                       valueInputOption="USER_ENTERED", body=requestBody).execute()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def BollingerBandPrediction(name, controls):
    exc_txt = "\nAn exception occurred - BollingerBandPrediction"
    try:
        features = ""
        timeSteps = 0
        scalerFile = ""
        modelFile = ""
        outputs = ""
        threshold = 0.9
        for control in controls:
            if 'file' in control:
                modelFile = control['file']
            elif 'features' in control:
                features = control['features']
            elif 'timeSteps' in control:
                timeSteps = control['timeSteps']
            elif 'scaler' in control:
                scalerFile =  control['scaler']
            elif 'Outputs' in control:
                outputs =  control['Outputs']
            elif 'threshold' in control:
                threshold =  control['threshold']
            elif 'featureFile' in control:
                inputFileSpec =  control['featureFile']

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
            saveMachineLearningSignal(signals)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def MACDTrendPrediction(trainedModel, controls):
    exc_txt = "\nAn exception occurred - MACDTrendPrediction"
    try:
        pass
    
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)
    
    return

def mlPredictions(dfRunControl):
    exc_txt = "\nAn exception occurred - mlPredictions"
    
    try:
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        models = aiwork + '\\' + localDirs['trainedmodels']

        for ndx in range (len(dfRunControl)):
            if dfRunControl.iloc[ndx][MODEL]:
                if dfRunControl.iloc[ndx][RUN]:
                    if dfRunControl.iloc[ndx][PROCESS_ID] == "Bollinger Band":
                        BollingerBandPrediction(dfRunControl.iloc[ndx][PROCESS_ID], dfRunControl.iloc[ndx]['json']['controls'])
                    elif dfRunControl.iloc[ndx][PROCESS_ID] == "MACD Trend":
                        MACDTrendPrediction(dfRunControl.iloc[ndx]['json']['controls'])
                    else:
                        print("WIP ==============\n\tPredictions using {} are not implemented\n==================".format(dfRunControl.iloc[ndx][PROCESS_ID])) 
            '''
                else:
                    print("Process named {} is a machine learning model but was not selected".format(dfRunControl.iloc[ndx][PROCESS_ID]))
            else:
                print("Process named {} is not a machine learning model".format(dfRunControl.iloc[ndx][PROCESS_ID]))
            '''
            ndx += 1
            
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def tkInterExp():
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

def iniRead():

    print("\n======================================================= jsonExp enter")
    exc_txt = "\nAn exception occurred - jsonExp"
    
    try:
        ''' access the <user>\AppData\Local\Develoment\<appName>.ini file '''
        
        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to retrieve <user>\AppData\Local\Develoment\<appName>.ini"
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        gitdir = localDirs['git']
        models = localDirs['trainedmodels']
        print("Local computer directories - localDirs: {}\n\taiwork: {}\n\tgitdir: {}\n\ttrained models: {}".format(localDirs, aiwork, gitdir, models))
    
        ''' identify information related to TD Ameritrade / Schwab APIs '''
        exc_txt = "\nAn exception occurred - unable to retrieve TD Ameritrade authentication information"
        tdaAuth = get_ini_data("TDAMERITRADE")
        print("TD Ameritrade authentication\n\ttdaAuth: {}\n\tauthentication: {}".format(tdaAuth, tdaAuth["authentication"]))

        ''' Google APIs '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleAuth = get_ini_data("GOOGLE")
        print("Google authentication\n\ttoken: {}\n\tcredentials: {}".format(googleAuth["token"], googleAuth["credentials"]))
        
        ''' read application specific configuration file '''
        exc_txt = "\nAn exception occurred - unable to process configuration file"
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        print("appConfig: {}".format(appConfig))
        
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

if __name__ == '__main__':
    print("==================================================== Code experimentation starting")
    
    processCtrl = iniRead()                   # Experimentation / development of configuration json file
    
    if True:
        processCtrl = tkInterExp()
        
        for ndx in range(len(processCtrl)):
            print("Process: {}: run = {}".format(processCtrl.iloc[ndx][PROCESS_ID], processCtrl.iloc[ndx][RUN]))
        mlPredictions(processCtrl)             # Use trained models to make predictions

    else:
        dataFiles = buildListofMarketDataFiles()

        digitalSreeni_180()
        linear_regression()
        sine_wave_regression()
        autoKeras()
        pass
    
    print("\n==================================================== Code experimentation ending")

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
import json
import pandas as pd
import re
from decimal import *

from configuration import get_ini_data
from configuration import read_config_json

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
RUN = "run"
MODEL_FILE = "file"
OUTPUT_FIELDS = "Outputs"
CONFIG = "json"

MMDDYYYY = 0
YYYYMMDD = 1


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

def BollingerBandPrediction(symbol, dataFile):
    
    prediction = False
    if os.path.exists(dataFile[0]):
        pass

    return prediction

def MACDTrendPrediction(symbol, dataFile):
    
    prediction = [0.2, 0.6, 0.2]
    if os.path.exists(dataFile[0]):
        pass
    
    return prediction

def mlPredictions(dictML):
    print("\n======================================================= mlPredictions enter")
    exc_txt = "\nAn exception occurred - mlPredictions"
    
    try:
    
        for model in iter(dictML):
            print("Prediction {}: {}".format(model, dictML[model]))
            
        localDirs = get_ini_data("LOCALDIRS")
        gitdir = localDirs['git']
        #aiwork = localDirs['aiwork']
        #models = localDirs['trainedmodels']
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        appDefaults = appConfig["defaults"]
        modelControlDefaults = appDefaults["mlModels"]
        modelPrep = appConfig["models"]
        for model in modelControlDefaults:
            notFound = True
            for ndxModel in iter(dictML):
                if ndxModel == model["name"]:
                    notFound = False
                    if dictML[model["name"]]:
                        if model["name"] == "Bollinger Band":
                            dataFiles = buildListofMarketDataFiles()
                            for sym in dataFiles.index:
                                dataFile = dataFiles.loc[sym]
                                prediction = BollingerBandPrediction(sym, dataFile)
                        elif model["name"] == "MACD Trend":
                            dataFiles = buildListofMarketDataFiles()
                            for sym in dataFiles.index:
                                dataFile = dataFiles.loc[sym]
                                prediction = MACDTrendPrediction(sym, dataFile)
                        else:
                            raise NameError("Predictions using {} are not implemented".format(model["name"])) 
                    break
            if notFound:
                raise NameError(exc_txt) 

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return

def tkInterExp(processCtrl):
    print("\n======================================================= tkInterExp enter")
    exc_txt = "\nAn exception occurred - tkInterExp"
    
    try:
        
        ROW_1 = 50
        ROW_2 = 100
        ROW_3 = 200
        ROW_BUTTON = 300
        ROW_HEIGHT = 20
        
        COL_1 = 100
        COL_2 = 200
        COL_3 = 300
        
        FORM_WIDTH = '600'
        FORM_HEIGHT = '400'
        FORM_BORDER = '10'
        FORM_GEOMETRY = FORM_WIDTH + 'x' + FORM_HEIGHT + "+" + FORM_BORDER + "+" + FORM_BORDER

        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to access local AppData information"
        localDirs = get_ini_data("LOCALDIRS")
        gitdir = localDirs['git']
        aiwork = localDirs['aiwork']
        trainedModels = localDirs['trainedmodels']
        print("trainedmodels: {}".format((aiwork + "\\" + trainedModels)))

        ''' read application specific configuration file '''
        exc_txt = "\nAn exception occurred - unable to process configuration file"
        config_data = get_ini_data("DAILY_PROCESS")
        appConfig = read_config_json(gitdir + config_data['config'])
        
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
            print(process[PROCESS_ID])
            newd={PROCESS_ID : process[PROCESS_ID], \
                  PROCESS_DESCRIPTION : process[PROCESS_DESCRIPTION], \
                  RUN : process[RUN], \
                  CONFIG : process}
            processCtrl.loc[len(processCtrl)]=newd
            processCheck[ndx] = IntVar()
            processButton[ndx] = Checkbutton(window, text = processCtrl.loc[ndx][PROCESS_ID], variable = processCheck[ndx])
            processButton[ndx].place(x=COL_1, y=ROW_2 + (ROW_HEIGHT * ndx))
            ndx += 1

        lblML=Label(window, text="Make machine learning predictions", fg='blue', font=("ariel", 10))
        lblML.configure(bg="white")
        lblML.place(x=COL_3, y=(ROW_2 - ROW_HEIGHT))

        modelDetails = appConfig[MODEL_CONFIGS]
        mlCheck = [IntVar()] * len(modelDetails)
        mlButton = [None] * len(processDetails)
        ndx = 0
        for model in modelDetails:
            print(model[PROCESS_ID])
            newd={PROCESS_ID : model[PROCESS_ID], \
                  PROCESS_DESCRIPTION : model[PROCESS_DESCRIPTION], \
                  RUN : model[RUN], \
                  MODEL_FILE : model[MODEL_FILE], \
                  OUTPUT_FIELDS : model[OUTPUT_FIELDS], \
                  CONFIG : model}
            processCtrl.loc[len(processCtrl)]=newd
            mlCheck[ndx] = IntVar()
            mlButton[ndx] = Checkbutton(window, text = processCtrl.loc[ndx + len(processCheck)][PROCESS_ID], variable = mlCheck[ndx])
            mlButton[ndx].place(x=COL_3, y=ROW_2 + (ROW_HEIGHT * ndx))
            ndx += 1

        ''' Select Google sheet file to use for market options details '''
        localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close
        radioButton = [None] * len(jsonGoogle["Google sheets"])
        radioValue = [IntVar()] * len(jsonGoogle["Google sheets"])
        ndx = 0
        for sheet in jsonGoogle["Google sheets"]:
            print("Name: {}, ID: {}".format(sheet["name"], sheet["file ID"]))
            radioButton[ndx] = Radiobutton(window, text=sheet["name"], variable = radioValue[ndx])
            radioButton[ndx].place(x=COL_3, y=ROW_3 + (ROW_HEIGHT * ndx))
            ndx += 1

        ''' =================== create button to process inputs =================== '''
        def go_button():
            for ndx in range (len(processCheck)):
                if processCheck[ndx].get() == 1:
                    processCtrl.loc[ndx][RUN] = True
            for ndx in range (len(mlCheck)):
                if mlCheck[ndx].get() == 1:
                    processCtrl.loc[ndx + len(processCheck)][RUN] = True
            window.quit()

        btn=Button(window, command=go_button, text="Run processes selected", fg='blue')
        btn.place(x=COL_2, y=ROW_BUTTON)

        ''' =================== Interact with user =================== '''
        window.geometry(FORM_GEOMETRY)
        window.mainloop()

        print("User Interface completing")

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return processCtrl

def strToInt(value):
    try:
        value = value.replace(",", "")
        return int(value)
    
    except ValueError:
        return np.nan

def strToFloat(value):
    try:
        value = value.replace("'", "")
        value = value.replace(",", "")
        return float(value)
    
    except ValueError:
        return np.nan

def dollarStrToFloat(value):
    try:
        if value == "-":
            value = 0.0
        else:
            value = value.replace("'", "")
            value = value.replace(",", "")
            value = value.replace("$", "")
        return float(value)
    
    except ValueError:
        return np.nan

def pctStrToFloat(value):
    try:
        if value == "-":
            fVal = 0.0
        else:
            value = value.replace("'", "")
            value = value.replace(",", "")
            value = value.replace("%", "")
            fVal = float(value)
            fVal = fVal / 100
        return float(fVal)
    
    except ValueError:
        return np.nan

def strToBool(value):
    try:
        if value == 'TRUE':
            return True
        elif value == 'FALSE':
            return False

    except ValueError:
        return np.nan
        
def sheetStrToDate(value, fmt, dummy):
    try:
        yyyy = datetime.MINYEAR
        mm = 1
        dd = 1
        sheetDate = date(yyyy, mm, dd)
        
        mo = re.search("/", value)
        if mo == None:
            mo = re.search("-", value)
            if mo == None:
                pass
            else:
                seperator = "-"
        else:
            seperator = "/"

        if value == "-":
            pass
        else:
            elems = re.split(seperator, value)
            if fmt == MMDDYYYY:
                yyyy = int(elems[2])
                mm = int(elems[0])
                dd = int(elems[1])
            elif fmt == YYYYMMDD:
                yyyy = int(elems[0])
                mm = int(elems[1])
                dd = int(elems[2])
            else:
                pass
            sheetDate = date(yyyy, mm, dd)
        
        return sheetDate

    except ValueError:
        return sheetDate
        
def seperateSymbol(value):
    try:
        subStrs = re.split(',', value)
        return subStrs[0]

    except ValueError:
        return np.nan

def loadMarketOptionsFile():
    
    ''' Find local file directories '''
    localDirs = get_ini_data("LOCALDIRS")
    workDir = localDirs['aiwork']
    optionsFile = workDir + "\\potential_option_trades.csv"

    if os.path.isfile(optionsFile):
        dfOptions = pd.read_csv(optionsFile)

        ''' convert strings to appropriate data types '''
        dfOptions['expiration'] = dfOptions['expiration'].apply(sheetStrToDate, args=(YYYYMMDD, np.NaN))
        
    else:
        print("File containing options does not exist")

    return dfOptions

def loadHoldings(sheet):
    
    READ_RANGE = 'Holdings!A1:ZZ'
    
    result = sheet.values().get(spreadsheetId=EXP_SPREADSHEET_ID, range=READ_RANGE).execute()
    values = result.get('values', [])
    if not values:
        print('\tNo data found.')

    dfHoldings = pd.DataFrame(data=values[1:], columns=values[0])
    
    ''' convert strings to appropriate data types '''
    dfHoldings['Current Holding'] = dfHoldings['Current Holding'].apply(strToInt)
    dfHoldings['Purchase $'] = dfHoldings['Purchase $'].apply(dollarStrToFloat)
    dfHoldings['Optioned'] = dfHoldings['Optioned'].apply(strToBool)
    
    ''' use the symbols as the index '''
    dfHoldings = dfHoldings.set_index('Symbol')
    
    return dfHoldings

def loadMarketDetails(sheet):
    
    READ_RANGE = 'TD Import Inf!A1:ZZ'                        

    result = sheet.values().get(spreadsheetId=EXP_SPREADSHEET_ID, range=READ_RANGE).execute()
    values = result.get('values', [])
    if not values:
        print('\tNo data found.')

    dfMarketDetails = pd.DataFrame(data=values[1:], columns=values[0])

    ''' convert strings to appropriate data types '''
    dfMarketDetails['Symbol'] = dfMarketDetails['Symbol'].apply(seperateSymbol)
    dfMarketDetails['Day Change %'] = dfMarketDetails['Day Change %'].apply(pctStrToFloat)
    dfMarketDetails['Last Price'] = dfMarketDetails['Last Price'].apply(dollarStrToFloat)
    dfMarketDetails['Dividend Yield %'] = dfMarketDetails['Dividend Yield %'].apply(pctStrToFloat)
    dfMarketDetails['Dividend $'] = dfMarketDetails['Dividend $'].apply(dollarStrToFloat)
    dfMarketDetails['Dividend Date'] = dfMarketDetails['Dividend Date'].apply(sheetStrToDate, args=(MMDDYYYY, np.NaN))
    dfMarketDetails['P/E Ratio'] = dfMarketDetails['P/E Ratio'].apply(strToFloat)
    dfMarketDetails['52 Week High'] = dfMarketDetails['52 Week High'].apply(dollarStrToFloat)
    dfMarketDetails['52 Week Low'] = dfMarketDetails['52 Week Low'].apply(dollarStrToFloat)
    dfMarketDetails['Volume'] = dfMarketDetails['Volume'].apply(strToInt)
    # Sectore remains a string
    dfMarketDetails['Earnings Date'] = dfMarketDetails['Earnings Date'].apply(sheetStrToDate, args=(MMDDYYYY, np.NaN))

    ''' use the symbols as the index '''
    dfMarketDetails = dfMarketDetails.set_index('Symbol')
    
    return dfMarketDetails
    
def prepMktOptionsForSheets(mktOptions):
    sheetMktOption = mktOptions
    
    sheetMktOption['Dividend Date'] = sheetMktOption['Dividend Date'].apply(date.isoformat)
    sheetMktOption['Earnings Date'] = sheetMktOption['Earnings Date'].apply(date.isoformat)
    sheetMktOption['expiration'] = sheetMktOption['expiration'].apply(date.isoformat)
        
    return sheetMktOption

def parseSheetErr(errText):
    '''
    <HttpError 400 when requesting https://sheets.googleapis.com/v4/spreadsheets/1XJNEWZ0uDdjCOvxJItYOhjq9kOhYtacJ3epORFn_fm4:batchUpdate?alt=json 
    returned "Invalid requests[0].addSheet: A sheet with the name "chandra42" already exists. Please enter another name.". 
    Details: "Invalid requests[0].addSheet: A sheet with the name "chandra42" already exists. Please enter another name.">
    '''
    
    return 

def openGoogleSheetService():
    """
    Google sheets API documentation
    
    https://cloud.google.com/apis/docs/client-libraries-explained
    https://cloud.google.com/python/docs/reference
    
    https://google-auth.readthedocs.io/en/stable/user-guide.html
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    
    https://developers.google.com/sheets/api/guides/concepts
    https://developers.google.com/sheets/api/reference/rest
    https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/cells
    
    https://googleapis.github.io/google-api-python-client/docs/dyn/sheets_v4.spreadsheets.html
    """
    
    # If modifying these scopes, delete the file token.json.
    SCOPE_LIMITED = ['https://www.googleapis.com/auth/drive.file']
    SCOPE_RO = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SCOPE_RW = ['https://www.googleapis.com/auth/spreadsheets']

    exc_txt = "\nAn exception occurred - unable to open Google sheets service"

    try:

        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
    
        ''' Google APIs '''
        googleAuth = get_ini_data("GOOGLE")
        GoogleTokenPath = aiwork + "\\" + googleAuth["token"]
        credentialsPath = aiwork + "\\" + googleAuth["credentials"]
    
        '''
        The file token.json stores the user's access and refresh tokens, and is
        created automatically when the authorization flow completes for the first
        time.
        '''
        creds = None

        if os.path.exists(GoogleTokenPath):
            creds = Credentials.from_authorized_user_file(GoogleTokenPath, SCOPE_RW)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(credentialsPath, SCOPE_RW)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(GoogleTokenPath, 'w') as token:
                token.write(creds.to_json())
    
        service = build('sheets', 'v4', credentials=creds)
        gsheet = service.spreadsheets()

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return gsheet

def marketOprionsPopulate(row, col, dfHoldings, dfMarketData):
    exc_txt = "Unable to determine purchase price"
    
    try:
        symbol = row['symbol']
        if symbol in dfHoldings.index:
            if col == 'Purchase $' or col == 'Current Holding':
                value = dfHoldings.loc[symbol][col]
            if col == 'Earnings Date' or col == 'Dividend Date':
                value = dfMarketData.loc[symbol][col]
        else:
            if col == 'Purchase $' or col == 'Current Holding':
                value = 0.0
            if col == 'Earnings Date' or col == 'Dividend Date':
                yyyy = datetime.MINYEAR
                mm = 1
                dd = 1
                sheetDate = date(yyyy, mm, dd)
                value = sheetDate

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  value

def dividendRisk(row):
    exc_txt = "Unable to determine risk due to dividend date"
    
    try:
        now = date.today()
        if now > row['Dividend Date']:
            riskStr = "2 - Div Past"
        elif row['expiration'] > row['Dividend Date']:            
            riskStr = "9 - Dividend"
        else:
            riskStr = "3 - TBD"

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  riskStr

def earningsRisk(row):
    exc_txt = "Unable to determine risk due to dividend date"
    
    try:
        now = date.today()
        if now > row['Earnings Date']:
            riskStr = "1 - Earnings Past"
        elif row['expiration'] <= row['Earnings Date']:            
            riskStr = "2 - No earnings"
        elif row['expiration'] > row['Earnings Date']:            
            riskStr = "9 - Earnings"
        else:
            riskStr = "3 - Unknown"

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  riskStr

def optionQty(row):
    exc_txt = "Unable to determine qty of options to trade"
    
    try:
        if row['strategy'] == "Covered call":
            optQty = int(row['Current Holding'] / 100)
        elif row['strategy'] == "Cash Secured Put":            
            optQty = int(25000 / (row['underlying Price'] * 100))
        else:
            optQty = 0

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  optQty

def annual_yield(discount, face_value, days_to_maturity):
    """
      Calculates the annual yield of a discount security.
    
      Args:
        discount: The discount on the security.
        face_value: The face value of the security.
        days_to_maturity: The number of days until the security matures.
    
      Returns:
        The annual yield of the security.
    """
    
    annual_yield = (discount / face_value) * (365 / days_to_maturity) * 100
    
    return annual_yield

def maxGainApy(row):
    exc_txt = "Unable to determine maximum gain APY of options if traded"
    try:
        now = date.today()
        exp = row['expiration']
        dtm = timedelta()
        dtm = exp - now
        gain = row['Max Profit']
        if row['Risk Management'] == "Current holding":
            ''' yielddisc(today(),C{},((S{}*100)*E{}),((S{}*100)*E{})+U{})
            YIELDDISC(
                settlement,      now
                maturity,        exp
                price,           s * e * 100
                redemption,      (s * e * 100) + u
                [day_count_convention])
            YIELDDISC(DATE(2010,01,02),DATE(2010,12,31),98.45,100)
            col c = row['expiration']
            col s = row['Qty']
            col e = row['underlying Price']
            col u = row['Max Profit']
            '''
            futureVal = (row['Qty'] * 100 * row['underlying Price']) + row['Max Profit']
        else:
            ''' yielddisc(today(),C{},V{},V{}+U{}))
            col c = row['expiration']
            col v = row['Risk Management']
            col u = row['Max Profit']
            '''
            futureVal = row['Risk Management'] + row['Max Profit']

        if gain == 0.0 or futureVal == 0.0 or dtm.days == 0:
            maxGainApy = 0.0
        else:
            maxGainApy = annual_yield(gain, futureVal, dtm.days)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  maxGainApy

def riskManagement(row):
    exc_txt = "Unable to determine risk management of options if traded"
    
    try:
        if row['strategy'] == "Covered call":
            riskMgmt = "Current holding"
        elif row['strategy'] == "Cash Secured Put":            
            #riskMgmt = "${}".format(row['Qty'] * 100 * row['strike Price'])
            riskMgmt = row['Qty'] * 100 * row['strike Price']
        else:
            riskMgmt = "TBD"

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  riskMgmt

def maxProfit(row):
    exc_txt = "Unable to calculate max profit"    
    try:
        profit = row['premium'] - row['commission']

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  profit

def lossVProfit(row):
    exc_txt = "Unable to calculate loss vs. profit"
    try:
        lvp = row['OTM Probability'] - row['probability of loss']

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  lvp

def premium(row):
    exc_txt = "Unable to calculate premium"
    try:
        prem = row['Qty'] * 100 *  row['bid']

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  prem

def commission(row):
    exc_txt = "Unable to calculate commission"
    try:
        com = row['Qty'] * 0.65

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return  com

def calculateFields(mktOptions, dfHoldings, dfMarketData):
    exc_txt = "Unable to determine calculated field values"
    
    try:
        ''' The order of these updates is important as the later ones use the values calculated earlier '''
        mktOptions['Purchase $'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Purchase $', dfHoldings, dfMarketData))
        mktOptions['Earnings Date'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Earnings Date', dfHoldings, dfMarketData))
        mktOptions['Dividend Date'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Dividend Date', dfHoldings, dfMarketData))
        mktOptions['Current Holding'] = mktOptions.apply(marketOprionsPopulate, axis=1, args=('Current Holding', dfHoldings, dfMarketData))
        mktOptions['Dividend'] = mktOptions.apply(dividendRisk, axis=1)
        mktOptions['Earnings'] = mktOptions.apply(earningsRisk, axis=1)
        mktOptions['Qty'] = mktOptions.apply(optionQty, axis=1)
        mktOptions['Risk Management'] = mktOptions.apply(riskManagement, axis=1)
        mktOptions['premium'] = mktOptions.apply(premium, axis=1)
        mktOptions['commission'] = mktOptions.apply(commission, axis=1)
        mktOptions['Max Profit'] = mktOptions.apply(maxProfit, axis=1)
        mktOptions['Loss vs. Profit'] = mktOptions.apply(lossVProfit, axis=1)
        mktOptions['max gain APY'] = mktOptions.apply(maxGainApy, axis=1)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return mktOptions

def eliminateLowReturnOptions(mktOptions):
    exc_txt = "An error occurred eliminating options"
    
    try:
        pass

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return mktOptions

def workbookExp():

    optionsCols = [ \
        "Symbol", \
        "Strategy", \
        "Expiration", \
        "Days to Expiration", \
        "Share price", \
        "Closing Price", \
        "Strike Price", \
        "Break Even", \
        "Bid", \
        "Ask", \
        "OTM Probability", \
        "volatility", \
        "ADX (Price trend est.)", \
        "Probability of Loss", \
        "Purchase $", \
        "Earnings", \
        "Dividend", \
        "Current Holding", \
        "Qty", \
        "Max Gain APY", \
        "Max Profit", \
        "Risk Management", \
        "Loss vs. profit", \
        "premium", \
        "commission", \
        "Earnings Date", \
        "Dividend Date", \
        "delta", \
        "gamma", \
        "theta", \
        "vega", \
        "rho", \
        "in The Money", \
        "expiration Date", \
        "ROI", \
        "Max Loss", \
        "Preferred outcome", \
        "Preferred Result", \
        "Unfavored Result" \
    ]

    ''' option evaluation sheet column numbers '''
    COL_symbol =  0
    COL_strategy = 1
    COL_expiration = 2
    COL_days_To_Expiration = 3
    COL_underlying_Price = 4
    COL_Closing_Price = 5
    COL_strike_Price = 6
    COL_break_even = 7
    COL_bid = 8
    COL_ask = 9
    COL_OTM_Probability = 10
    COL_volatility = 11
    COL_ADX = 12
    COL_probability_of_loss = 13
    COL_Purchase_dol = 14
    COL_Earnings = 15
    COL_Dividend = 16
    COL_Current_Holding = 17
    COL_Qty = 18
    COL_max_gain_APY = 19
    COL_Max_Profit = 20
    COL_Risk_Management = 21
    COL_Loss_vs_Profit = 22
    COL_premium = 23
    COL_commission = 24
    COL_Earnings_Date = 25
    COL_dividend_Date = 26
    COL_delta = 27
    COL_gamma = 28
    COL_theta = 29
    COL_vega = 30
    COL_rho = 31
    COL_in_The_Money = 32
    COL_expiration_Date = 33
    COL_ROI = 34
    COL_Max_Loss = 35
    COL_Preferred_Outcome = 36
    COL_Preferred_Result = 37
    COL_Unfavored_Result = 38

    COMMISSION_THRESHOLD = '0.65'

    '''
    hide rows where Qty = 0
    hide rows (A-Z) where ITM is TRUE
    hide rows where Concern = Dividend or Earnings
    highlighted where risk management > $50,000
    hide rows where max gain APY < 15%
    hide rows where max profit <$500
    sort rows by strategy/symbol/expiration/max gain APY
    
    https://developers.google.com/sheets/api/guides/values
    
    '''
    
    print("\n======================================================= workbookExp enter")
    exc_txt = "\nAn exception occurred - workbookExp"
    
    try:
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']

        ''' Google APIs '''
        exc_txt = "\nAn exception occurred - unable to retrieve Google authentication information"
        googleAuth = get_ini_data("GOOGLE")
        '''
        GoogleTokenPath = aiwork + "\\" + googleAuth["token"]
        credentialsPath = aiwork + "\\" + googleAuth["credentials"]
        '''
        
        exc_txt = "\nAn exception occurred - unable to authenticate with Google"
        
        sheet = openGoogleSheetService()
        
        ''' ================== Call the Sheets API - values - read data ================== '''
        READ_RANGE = 'Read Test!A1:CZ'                        
        result = ""

        result = sheet.values().get(spreadsheetId=EXP_SPREADSHEET_ID, range=READ_RANGE).execute()
        print("\tmajorDimension: {}".format(result.get('majorDimension')))
        print("\trange: {}".format(result.get('range')))

        values = result.get('values', [])
        if not values:
            print('\tNo data found.')
        else:
            rowNum = 1
            for row in values:
                if len(row) < 2:
                    print("\tBlank row in calls A-C")
                else:
                    print('\trow: {}, col A: {}'.format(rowNum, row[0]))
                rowNum += 1
            
        ''' ================== Call the Sheets API - update - write data ================== '''        
        WRITE_RANGE_TITLE = 'Write Test!A1:CZ'                        
        WRITE_RANGE_DATA = 'Write Test!A2:CZ'                        
        
        mktOptions = loadMarketOptionsFile()
        ''' replace NaN with 0 '''
        mktOptions = mktOptions.fillna(0)
        dfHoldings = loadHoldings(sheet)
        dfMarketData = loadMarketDetails(sheet)        
        mktOptions = calculateFields(mktOptions, dfHoldings, dfMarketData)
        mktOptions = eliminateLowReturnOptions(mktOptions)

        ''' valueInputOption = USER_ENTERED or RAW    '''
        sheetMktOptions = prepMktOptionsForSheets(mktOptions)
        cellValues = sheetMktOptions.values.tolist()
        requestBody = {'values': cellValues}
        result = sheet.values().update(spreadsheetId=EXP_SPREADSHEET_ID, range=WRITE_RANGE_DATA, \
                    valueInputOption="USER_ENTERED", body=requestBody).execute()
        print("\tspreadsheetId: {}".format(result.get('spreadsheetId')))
        print("\tupdatedRange: {}".format(result.get('updatedRange')))
        print("\tupdatedRows: {}".format(result.get('updatedRows')))
        print("\tupdatedColumns: {}".format(result.get('updatedColumns')))
        print("\tupdatedCells: {}".format(result.get('updatedCells')))
        
        colTitles = mktOptions.columns
        colTitleList = [colTitles.values.tolist()]
        requestBody = {'values': colTitleList}
        result = sheet.values().update(spreadsheetId=EXP_SPREADSHEET_ID, range=WRITE_RANGE_TITLE, \
                    valueInputOption="USER_ENTERED", body=requestBody).execute()
                    
        #COL_bid = 8
        #COL_ask = 9
        ''' read back and identify written data '''
        result = sheet.values().get(spreadsheetId=EXP_SPREADSHEET_ID, range=WRITE_RANGE_DATA).execute()
        values = result.get('values', [])
        if not values:
            print('\tNo data found.')
        else:
            rowNum = 1
            for row in values:
                rowNum += 1
            print("Read back {} rows".format(rowNum))
        
        ''' ================== Call the Sheets API - batchUpdate - add a new tab ================== '''
        requests = list([])
        addSheetRequest = {"properties" : {"title": "chandra42"}}
        requests.append({"addSheet" : addSheetRequest})
        requestBody = {"requests" : requests}
        print("\nbatchUpdate requestBody:\n\t{}".format(requestBody))
        result = sheet.batchUpdate(spreadsheetId=EXP_SPREADSHEET_ID, body=requestBody).execute()

    except HttpError as err:
        parseSheetErr(err)
        sys.exit(err)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        sys.exit(exc_txt + "\n\t" + exc_str)

    return

def setPythonPath(gitdir):
    
    ''' Set python path for executing stand alone scripts '''
    pPath = gitdir + "\\chandra\\processes\\single"
    pPath += ";"
    pPath += gitdir + "\\chandra\\processes\\multiprocess"
    pPath += ";"
    pPath += gitdir + "\\chandra\\processes\\child"
    pPath += ";"
    pPath += gitdir + "\\chandra\\utility"
    pPath += ";"
    pPath += gitdir + "\\chandra\\technical_analysis"
    pPath += ";"
    pPath += gitdir + "\\chandra\\td_ameritrade"
    pPath += ";"
    pPath += gitdir + "\\chandra\\machine_learning"
    pPath += ";"
    pPath += gitdir + "\\chandra\\unit_test"
    os.environ["PYTHONPATH"] = pPath

    return

def jsonExp():

    print("\n======================================================= jsonExp enter")
    exc_txt = "\nAn exception occurred - jsonExp"
    
    try:
        ''' access the <user>\AppData\Local\Develoment\<appName>.ini file '''
        
        ''' Find local file directories '''
        exc_txt = "\nAn exception occurred - unable to retrieve <user>\AppData\Local\Develoment\<appName>.ini"
        localDirs = get_ini_data("LOCALDIRS")
        gitdir = localDirs['git']
        aiwork = localDirs['aiwork']
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
        
        localGoogleProject = open(aiwork + "\\Google_Project_Local.json", "rb")
        jsonGoogle = json.load(localGoogleProject)
        localGoogleProject.close

        cols = [PROCESS_ID, PROCESS_DESCRIPTION, RUN, MODEL_FILE, OUTPUT_FIELDS, CONFIG]
        processCtrl = pd.DataFrame(columns=cols)

    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        exc_txt = exc_txt + "\n\t" + exc_str
        sys.exit(exc_txt)

    return processCtrl

if __name__ == '__main__':
    print("==================================================== Code experimentation starting")
    
    '''
    Update market data: base on tda_derivative_data_master
    read option chains: base on assess_trading_signals
    '''
    
    if True:
        workbookExp()               # Experimentation with Google sheet access
        
    else:
        dataFiles = buildListofMarketDataFiles()
        processCtrl = jsonExp()                   # Experimentation / development of configuration json file
        processCtrl = tkInterExp(processCtrl)                # Experimentation with user interface        

        for ndx in range(len(processCtrl)):
            print("Process: {}: run = {}".format(processCtrl.loc[ndx][PROCESS_ID], processCtrl.loc[ndx][RUN]))
        mlPredictions(processCtrl)             # Use trained models to make predictions

        digitalSreeni_180()
        linear_regression()
        sine_wave_regression()
        autoKeras()
        pass
    
    print("\n==================================================== Code experimentation ending")

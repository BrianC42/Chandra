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


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import fetch_california_housing

import tensorflow as tf
from tensorflow import keras
import autokeras as ak

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

if __name__ == '__main__':
    #digitalSreeni_180()
    linear_regression()
    sine_wave_regression()
    autoKeras()
    
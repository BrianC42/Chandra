'''
Created on Jan 15, 2021

@author: Brian
'''
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

from tda_api_library import format_tda_datetime

from configuration_constants import JSON_TENSORFLOW
from configuration_constants import JSON_AUTOKERAS
from configuration_constants import JSON_OPTIMIZER
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_METRICS
from configuration_constants import JSON_VISUALIZATIONS
from configuration_constants import JSON_VISUALIZE_TRAINING_FIT
from configuration_constants import JSON_VISUALIZE_TARGET_SERIES

from TrainingDataAndResults import TRAINING_TENSORFLOW
from TrainingDataAndResults import TRAINING_AUTO_KERAS
from TrainingDataAndResults import INPUT_LAYERTYPE_DENSE
from TrainingDataAndResults import INPUT_LAYERTYPE_RNN
from TrainingDataAndResults import INPUT_LAYERTYPE_CNN

def visualize_fit(d2r, plot_on_axis):
    nx_optimizer = nx.get_node_attributes(d2r.graph, JSON_OPTIMIZER)[d2r.mlNode]
    nx_loss = nx.get_node_attributes(d2r.graph, JSON_LOSS)[d2r.mlNode]
    nx_metrics = nx.get_node_attributes(d2r.graph, JSON_METRICS)[d2r.mlNode]

    str_l7 = 'Opt:{:s}'.format(nx_optimizer)    
    str_p4 = ',loss:{:s}'.format(nx_loss)
    str_p5 = ',metrics:{:s}'.format(nx_metrics[0])
    str_p6 = ',epochs:{:d}'.format(d2r.fitting.params['epochs'])
    str_p7 = ',steps:{:d}'.format(d2r.fitting.params['steps'])
    str_params = "\n" + str_l7 + str_p4 + str_p5 + str_p6 + str_p7
    
    plot_on_axis.set_title("Fitting history" + str_params)
    plot_on_axis.plot(d2r.fitting.epoch, d2r.fitting.history['loss'], label='Training loss')
    plot_on_axis.plot(d2r.fitting.epoch, d2r.fitting.history['val_loss'], label='Validation loss')
    plot_on_axis.set_xlabel("Epochs")
    plot_on_axis.set_ylabel("loss")
    plot_on_axis.legend()
        
    return
    
def plotTargetValues(d2r, plot_on_axis):
    plot_on_axis.set_title("Target data series")
    plot_on_axis.set_xlabel("time periods")
    plot_on_axis.set_ylabel("Data Values")
    if d2r.seriesDataType == "TDADateTime":
        '''
        x_tdaDatesStr = []
        for tdaDate in d2r.rawData['DateTime']:
            x_tdaDatesStr.append(format_tda_datetime(tdaDate))
        '''
        x_dates = d2r.rawData['DateTime'].copy()
        x_dates2 = x_dates[range(0, len(x_dates), int(len(x_dates)/10))]
        x_ticks = []
        x_tickLabels = []
        for tdaDate in x_dates2:
            x_ticks.append(tdaDate)
            x_tickLabels.append(format_tda_datetime(tdaDate))
        #x_ticks = d2r.rawData['DateTime'][range(0, len(), 100)]
        y_targets = d2r.rawData['Close']
        label=d2r.dataSeriesIDFields[0]
        plot_on_axis.xaxis.set_ticks(x_ticks)
        plot_on_axis.xaxis.set_ticklabels(x_tickLabels)
        plot_on_axis.plot(x_dates, y_targets, label=label, linestyle='solid')

    '''
    plot_on_axis.plot(range(len(d2r.rawTargets)), d2r.rawTargets[:], label='Test series')
    testData = d2r.rawData.iloc[(len(d2r.rawTargets) - len(d2r.testY)) : ]
    testDateTimes = testData.iloc[:, 0]
    tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    plt.xticks(tmark, tmarkDates, rotation=20)
    plot_on_axis.legend()
    plot_on_axis.grid(True)
    '''
    return 

def visualizeAnomalyDistribution(d2r, prediction, plot_on_axis):
    print("\n=======================WIP ====================\n\tdisplay distribution of forecast error")
    plot_on_axis.set_title("Data series anomaly distribution")
    plot_on_axis.set_xlabel("prediction error")
    plot_on_axis.set_ylabel("error count")

    #testMAE = np.mean(np.abs(prediction[:, 0] - d2r.testY))
    testError = np.abs(prediction[:, 0] - d2r.testY)
    n, bins, patches = plot_on_axis.hist(testError, bins=50)
    #plot_on_axis.set_xticks(np.arange(0, max(testError), max(testError)/10))
    '''
    max_trainMAE = 0.2  #or Define 90% value of max as threshold.

    #Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(d2r.testX[20:])
    anomaly_df['testMAE'] = testMAE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
    anomaly_df['Close'] = d2r.testX[20:]['Close']

    #Plot testMAE vs max_trainMAE
    sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['testMAE'])
    sns.lineplot(x=anomaly_df['Date'], y=anomaly_df['max_trainMAE'])

    anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

    #Plot anomalies
    sns.lineplot(x=anomaly_df['Date'], y=d2r.scaler.inverse_transform(anomaly_df['Close']))
    sns.scatterplot(x=anomalies['Date'], y=d2r.scaler.inverse_transform(anomalies['Close']), color='r')
    '''
    return

def selectDateAxisLabels(dateTimes):
    testDates = []    
    for dt in dateTimes:
        testDates.append(format_tda_datetime(dt))
        
    tmark = list(range(0, len(testDates), int(len(testDates)/10)))
    
    tmarkDates = []    
    for ndx in range(0, len(testDates)):
        if ndx in tmark:
            tmarkDates.append(testDates[ndx])
        
    return tmark, tmarkDates

def visualizeTestVsForecast(d2r, prediction, axis1, axis2):
    RECENT = 40
    FORECAST = 0
    
    axis1.set_title("WIP")
    axis1.set_xlabel("time periods")
    axis1.set_ylabel("Data Value")
    axis1.plot(range(len(prediction)-RECENT, len(prediction)), \
               d2r.testY[len(prediction)-RECENT : ], \
               label='Test series')
    axis1.plot(range(len(prediction)-RECENT, len(prediction)), \
               prediction[len(prediction)-RECENT : , FORECAST], \
               linestyle='dashed', label='Prediction - 1 period')
    testData = d2r.data.iloc[(len(d2r.data) - RECENT) : ]
    testDateTimes = testData.iloc[:, 0]
    tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    axis1.legend()
    axis1.grid(True)

    axis2.set_title("Data series vs. Predictions")
    axis2.set_xlabel("time periods")
    axis2.set_ylabel("Data Value")
    axis2.plot(range(len(prediction)), d2r.testY[:], label='Test series')
    axis2.plot(range(len(prediction)), prediction[:, FORECAST], linestyle='dashed', label='Prediction - 1 period')
    testData = d2r.data.iloc[(len(d2r.data) - len(d2r.testY)) : ]
    testDateTimes = testData.iloc[:, 0]
    tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    plt.xticks(tmark, tmarkDates, rotation=20)
    axis2.legend()
    axis2.grid(True)

    return

def visualize_dense(d2r):
    prediction = d2r.model.predict(x=d2r.testX)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    visualize_fit(d2r, axs[1, 0])

    '''
    axs[0, 1].set_title("Training Data")
    axs[0, 1].scatter(d2r.trainX, d2r.trainY)
        
    axs[0, 2].set_title("Testing Data")
    axs[0, 2].set_xlabel("Feature")
    axs[0, 2].set_ylabel("Target")
    axs[0, 2].scatter(d2r.testX, d2r.testY)
    
    axs[1, 1].set_title("Prediction")
    axs[1, 1].set_xlabel("Feature")
    axs[1, 1].set_ylabel("Target")
    axs[1, 1].scatter(d2r.testX, prediction, label='Prediction', linestyle='dashed')
    axs[1, 1].scatter(d2r.testX, d2r.testY, label='Test data')
    axs[1, 1].legend()
    '''

    visualizeTestVsForecast(d2r, prediction, axs[0, 1], axs[1, 1])
        
    axs[1, 2].set_title("Extrapolation")
    x_min = np.min(d2r.testX)
    x_max = np.max(d2r.testX)
    iterable = ((x_max + (((x_max - x_min) / 10) * x)) for x in range(100))
    x_predict = np.fromiter(iterable, float)
    y_predict = d2r.model.predict(x=x_predict)
    axs[1, 2].scatter(x_predict, y_predict)
    axs[1, 2].set_xlabel("Feature")
    axs[1, 2].set_ylabel("Target")

    plt.tight_layout()
    plt.show()
    return 

def visualize_rnn(d2r):
    prediction = d2r.model.predict(d2r.testX)

    '''
    Screen 1
        Training parameters
        Fitting results
    '''
    fig1, axs1 = plt.subplots(2, 1)
    fig1.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    
    visualize_fit(d2r, axs1[1])
    
    plt.tight_layout()
    plt.show()

    '''
    Screen 2
        Testing data and model predictions
        Two timeframes with focus on the most recent period
    '''
    fig2, axs2 = plt.subplots(2, 1)
    fig2.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    
    visualizeTestVsForecast(d2r, prediction, axs2[0], axs2[1])
    
    plt.tight_layout()
    plt.show()
    
    '''
    Screen 3
        Distribution of prediction errors
        Test data with samples generating the largest errors highlighted
    '''
    fig3, axs3 = plt.subplots(1, 2)
    fig3.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    
    '''
    visualizeAnomalyDistribution(d2r, prediction, axs3[0])
    '''

    ''' =======================WIP ==================== '''
    print("\n=======================WIP ====================\n\tinverse_transform to display test target value")
    if d2r.normalized == True:
        data = d2r.scaler.inverse_transform(d2r.normalizedData)
    else:
        data = d2r.data
    ''' =============================================== '''
        
    ''' d2r.testY is normalized '''
    axs3[1].plot(range(len(prediction)), d2r.testY[:], label='Test series')
    axs3[1].grid(True)
    
    testData = d2r.data.iloc[(len(d2r.data) - len(d2r.testY)) : ]
    testDateTimes = testData.iloc[:, 0]
    tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    #plt.xticks(tmark, tmarkDates, rotation=20)
    #fig3.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()
    return 

def visualize_cnn(d2r):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    visualize_fit(d2r, axs[0, 0])
    
    plt.tight_layout()
    plt.show()
    return 

def visualize_ak(d2r):
    print("\n=======================WIP ====================\n\tAutoKeras evaluation and visualization")
    d2r.testY=d2r.testY['Close'].to_numpy()
    
    d2r.model = d2r.model.export_model()
    d2r.model.summary()
    prediction = d2r.model.predict(x=d2r.testX)

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    visualize_fit(d2r, axs[1, 0])
    visualizeTestVsForecast(d2r, prediction, axs[0, 1], axs[1, 1])
    
    plt.tight_layout()
    plt.show()
    return

def evaluate_and_visualize(d2r):
    d2r.evaluation = d2r.model.evaluate(x=d2r.testX, y=d2r.testY)
    prediction = d2r.model.predict(d2r.testX)

    if d2r.trainer == TRAINING_TENSORFLOW:
        if d2r.modelType == INPUT_LAYERTYPE_DENSE:
            pass
            #visualize_dense(d2r)
        elif d2r.modelType == INPUT_LAYERTYPE_RNN:
            pass
            #visualize_rnn(d2r)
        elif d2r.modelType == INPUT_LAYERTYPE_CNN:
            pass
            #visualize_cnn(d2r)
    elif d2r.trainer == TRAINING_AUTO_KERAS:
        d2r.model = d2r.model.export_model()
        d2r.model.summary()
        #visualize_ak(d2r)
        
    nx_visualizations = nx.get_node_attributes(d2r.graph, JSON_VISUALIZATIONS)[d2r.mlNode]
    for vizualize in nx_visualizations:
        if vizualize == JSON_VISUALIZE_TRAINING_FIT:
            fig1, axs1 = plt.subplots(1, 1)
            fig1.suptitle("Training results of: " + d2r.mlNode, fontsize=14, fontweight='bold')
            visualize_fit(d2r, axs1)
            plt.tight_layout()
            plt.show()
        elif vizualize == JSON_VISUALIZE_TARGET_SERIES:
            fig1, axs1 = plt.subplots(1, 1)
            fig1.suptitle("Training targets for: " + d2r.mlNode, fontsize=14, fontweight='bold')
            plotTargetValues(d2r, axs1)
            plt.tight_layout()
            plt.show()
    
    return
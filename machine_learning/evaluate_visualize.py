'''
Created on Jan 15, 2021

@author: Brian
'''
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from matplotlib.pyplot import tight_layout

def visualize_fit(d2r):
    '''
    fig1, axs1 = plt.subplots(2, 1)
    fig1.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    
    fig = plt.figure(tight_layout=True)
    '''
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 1)
    fig.suptitle("Training results", fontsize=14, fontweight='bold')

    axLoss = fig.add_subplot(gs[0, 0])
    axAccuracy = fig.add_subplot(gs[1, 0])
    
    nx_optimizer = nx.get_node_attributes(d2r.graph, JSON_OPTIMIZER)[d2r.mlNode]
    nx_loss = nx.get_node_attributes(d2r.graph, JSON_LOSS)[d2r.mlNode]
    nx_metrics = nx.get_node_attributes(d2r.graph, JSON_METRICS)[d2r.mlNode]

    str_l7 = 'Opt:{:s}'.format(nx_optimizer)    
    str_p4 = ',loss:{:s}'.format(nx_loss)
    str_p5 = ',metrics:{:s}'.format(nx_metrics[0])
    str_p6 = ',epochs:{:d}'.format(d2r.fitting.params['epochs'])
    str_p7 = ',steps:{:d}'.format(d2r.fitting.params['steps'])
    str_params = "\n" + str_l7 + str_p4 + str_p5 + str_p6 + str_p7
    axLoss.set_title("Fitting history" + str_params)
    '''
    evaluationLoss = d2r.evaluation[0]
    evaluationAccuracy = d2r.evaluation[1]
    '''
    axLoss.plot(d2r.fitting.epoch, d2r.fitting.history['loss'], label='Training loss')
    axLoss.plot(d2r.fitting.epoch, d2r.fitting.history['val_loss'], label='Validation loss')
    axLoss.set_xlabel("Epochs")
    axLoss.set_ylabel("loss")
    axLoss.legend()

    axAccuracy.plot(d2r.fitting.epoch, d2r.fitting.history['accuracy'], label='Training accuracy')
    axAccuracy.plot(d2r.fitting.epoch, d2r.fitting.history['val_accuracy'], label='Validation accuracy')
    axAccuracy.set_xlabel("Epochs")
    axAccuracy.set_ylabel("Accuracy")
    axAccuracy.legend()

    plt.tight_layout()
    plt.show()
    
    return
    
def plotTargetValues(d2r, axis):
    axis.set_title("Target data series")
    axis.set_xlabel("time periods")
    axis.set_ylabel("Data Values")
    if d2r.seriesDataType == "TDADateTime":
        '''
        d2r.dataSeriesIDFields and d2r.rawTargets are lists.
        d2r.dataSeriesIDFields returns a dataframe (including column header)
        d2r.dataSeriesIDFields[0] returns a list without header
        '''
        x_dates = d2r.rawData[d2r.dataSeriesIDFields[0]].copy()
        x_dates2 = x_dates[range(0, len(x_dates), int(len(x_dates)/10))]
        x_ticks = []
        x_tickLabels = []
        for tdaDate in x_dates2:
            x_ticks.append(tdaDate)
            x_tickLabels.append(format_tda_datetime(tdaDate))
        y_targets = d2r.rawData[d2r.rawTargets[0]]
        label=d2r.dataSeriesIDFields[0]
        axis.xaxis.set_ticks(x_ticks)
        axis.xaxis.set_ticklabels(x_tickLabels)
        axis.plot(x_dates, y_targets, label=label, linestyle='solid')

    return 

def plotDataGroups(d2r):

    print("\nTraining shapes x:%s y:%s, validating shapes x:%s y:%s, testing shapes x:%s y:%s" % (d2r.trainX.shape, d2r.trainY.shape, d2r.validateX.shape, d2r.validateY.shape, d2r.testX.shape, d2r.testY.shape))

    if d2r.seriesDataType == "TDADateTime":
        targetCount = len(d2r.preparedTargets)
        seriesLen = len(d2r.trainX[0, :, 0])
        trainingPeriods = d2r.trainX.shape[2]
        rawFeatureCount = len(d2r.rawFeatures)
        trainingFeatureCount = len(d2r.preparedFeatures)
        
        ''' create matplotlib figure with gridspec sized to display data used to train the model '''
        plotCount = max(rawFeatureCount, trainingFeatureCount, trainingPeriods, targetCount)
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(5, plotCount)

        fig.suptitle("Data preparation for: " + d2r.mlNode, fontsize=14, fontweight='bold')

        axRawTarget = fig.add_subplot(gs[0, :])
        axPrepTarget = fig.add_subplot(gs[1, :])

        axsTargets = pd.DataFrame()
        for colLabel in range(0, targetCount):
            axsTargets.insert( colLabel, colLabel, [fig.add_subplot(gs[2, colLabel])])

        axRawFeature = pd.DataFrame()
        for colLabel in range(0, rawFeatureCount):
            axRawFeature.insert( colLabel, colLabel, [fig.add_subplot(gs[3, colLabel])])
            
        axPreparedFeatures = pd.DataFrame()
        for colLabel in range(0, trainingFeatureCount):
            axPreparedFeatures.insert( colLabel, colLabel, [fig.add_subplot(gs[4, colLabel])])
            
        ''' Full raw data '''        
        x_dates = d2r.rawData[d2r.dataSeriesIDFields[0]].copy()
        x_dates2 = x_dates[range(0, len(x_dates), int(len(x_dates)/10))]
        y_targets = d2r.rawData[d2r.rawTargets]
        x_ticks = []
        x_tickLabels = []
        for tdaDate in x_dates2:
            x_ticks.append(tdaDate)
            x_tickLabels.append(format_tda_datetime(tdaDate))
            
        ''' Training data '''        
        x_trainDates = d2r.rawData[d2r.dataSeriesIDFields[0]][ : d2r.trainLen].copy()
        x_trainDates2 = x_trainDates    [range( 0, \
                                                d2r.trainLen, \
                                                int(d2r.trainLen / 3))]
        y_preparedTrainTargets = d2r.data[d2r.preparedTargets][ : d2r.trainLen].to_numpy()
        y_rawTrainTargets = d2r.rawData[d2r.rawTargets[0]][ : d2r.trainLen].copy()
        x_ticksTrain = []
        x_tickLabelsTrain = []
        for tdaDate in x_trainDates2:
            x_ticksTrain.append(tdaDate)
            x_tickLabelsTrain.append(format_tda_datetime(tdaDate))

        ''' Validation data '''        
        x_validateDates = d2r.rawData[d2r.dataSeriesIDFields[0]][d2r.trainLen : (d2r.trainLen + d2r.validateLen)].copy()
        x_validateDates2 = x_validateDates  [range( d2r.trainLen, \
                                                    d2r.trainLen + d2r.validateLen, \
                                                    int(d2r.validateLen / 3))]
        y_rawValidateTargets = d2r.rawData[d2r.rawTargets[0]][d2r.trainLen : (d2r.trainLen + d2r.validateLen)].copy()
        y_preparedValidateTargets = d2r.data[d2r.preparedTargets][d2r.trainLen : (d2r.trainLen + d2r.validateLen)].to_numpy()
        x_ticksVal = []
        x_tickLabelsVal = []
        for tdaDate in x_validateDates2:
            x_ticksVal.append(tdaDate)
            x_tickLabelsVal.append(format_tda_datetime(tdaDate))

        ''' Testing data '''        
        x_testDates = d2r.rawData[d2r.dataSeriesIDFields[0]][(d2r.trainLen + d2r.validateLen) : ].copy()
        x_testDates2 = x_testDates  [range(d2r.trainLen + d2r.validateLen, \
                                           len(x_dates), \
                                           int(d2r.testLen / 3))]
        y_rawTestTargets = d2r.rawData[d2r.rawTargets[0]][(d2r.trainLen + d2r.validateLen) : ].copy()
        y_preparedTestTargets = d2r.data[d2r.preparedTargets][(d2r.trainLen + d2r.validateLen) : ].to_numpy()
        x_ticksTest = []
        x_tickLabelsTest = []
        for tdaDate in x_testDates2:
            x_ticksTest.append(tdaDate)
            x_tickLabelsTest.append(format_tda_datetime(tdaDate))

        label=d2r.dataSeriesIDFields[0]

        axRawTarget.set_title("Target raw data")
        axRawTarget.set_xlabel("time periods")
        axRawTarget.set_ylabel("Target Values")
        axRawTarget.xaxis.set_ticks(x_ticks)
        axRawTarget.xaxis.set_ticklabels(x_tickLabels)
        lineTrain, = axRawTarget.plot(x_trainDates, y_rawTrainTargets, label=label, linestyle='solid')
        lineVal, = axRawTarget.plot(x_validateDates, y_rawValidateTargets, label=label, linestyle='solid')
        lineTest, = axRawTarget.plot(x_testDates, y_rawTestTargets, label=label, linestyle='solid')
        axRawTarget.legend([lineTrain, lineVal, lineTest], ['Training', 'Validation', 'Testing'], loc='upper center')
        
        axPrepTarget.set_title("Target prepared data")
        axPrepTarget.set_xlabel("time periods")
        axPrepTarget.set_ylabel("Data Values")
        axPrepTarget.xaxis.set_ticks(x_ticks)
        axPrepTarget.xaxis.set_ticklabels(x_tickLabels)
        lineTrain, = axPrepTarget.plot(x_trainDates, y_preparedTrainTargets[:, 0], label=label)
        lineVal, = axPrepTarget.plot(x_validateDates, y_preparedValidateTargets[:, 0], label=label)
        lineTest, = axPrepTarget.plot(x_testDates, y_preparedTestTargets[:, 0], label=label)
        axPrepTarget.legend([lineTrain, lineVal, lineTest], ['Training', 'Validation', 'Testing'], loc='upper center')

        for ndx in range(0, targetCount):
            title = "Prepared Target: " + d2r.preparedTargets[ndx]
            axsTargets.iat[0, ndx].set_title(title, fontsize=8)
            axsTargets.iat[0, ndx].set_xlabel("time periods (training samples)")
            axsTargets.iat[0, ndx].set_ylabel("Data Values")
            axsTargets.iat[0, ndx].xaxis.set_ticks(range(seriesLen))
            axsTargets.iat[0, ndx].plot(range(seriesLen), d2r.data[d2r.preparedTargets[ndx]][0: seriesLen], label=ndx, linestyle='solid')
            
        for ndx in range(0, rawFeatureCount):
            title = "Raw: " + d2r.rawFeatures[ndx]
            axRawFeature.iat[0, ndx].set_title(title, fontsize=8)
            axRawFeature.iat[0, ndx].set_xlabel("time periods")
            axRawFeature.iat[0, ndx].set_ylabel("Data Values")
            axRawFeature.iat[0, ndx].xaxis.set_ticks(range(seriesLen))
            axRawFeature.iat[0, ndx].plot(range(seriesLen), d2r.rawData[d2r.rawFeatures[ndx]][0: seriesLen], label=d2r.rawFeatures[ndx], linestyle='solid')
            
        for ndx in range(0, trainingFeatureCount):
            title = "Prepared: " + d2r.preparedFeatures[ndx]
            '''
            for tgtndx in range(0, targetCount):
                axPreparedFeatures.iat[0, ndx].text(0.5, 0.5 - (tgtndx * 0.1), "Training target value " + "a" + " = " + "1.0")
            '''
            axPreparedFeatures.iat[0, ndx].set_title(title, fontsize=8)
            axPreparedFeatures.iat[0, ndx].set_xlabel("time periods")
            axPreparedFeatures.iat[0, ndx].set_ylabel("Data Values")
            axPreparedFeatures.iat[0, ndx].xaxis.set_ticks(range(seriesLen))
            axPreparedFeatures.iat[0, ndx].plot(range(seriesLen), d2r.data[d2r.preparedFeatures[ndx]][0: seriesLen], label=d2r.preparedFeatures[ndx], linestyle='solid')
            
        plt.tight_layout()
        plt.show()

    return 

def showNormalizationCategorization(d2r):
    if d2r.seriesDataType == "TDADateTime":
        ''' Full raw data '''        
        x_dates = d2r.rawData['DateTime'].copy()
        x_dates2 = x_dates[range(0, len(x_dates), int(len(x_dates)/3))]

        ''' Training data '''        
        x_trainDates = d2r.rawData['DateTime'][ : d2r.trainLen].copy()
        x_trainDates2 = x_trainDates    [range( 0, \
                                                d2r.trainLen, \
                                                int(d2r.trainLen / 3))]
        y_trainTargets = d2r.rawData['Close'][ : d2r.trainLen].copy()
        x_ticksTrain = []
        x_tickLabelsTrain = []
        for tdaDate in x_trainDates2:
            x_ticksTrain.append(tdaDate)
            x_tickLabelsTrain.append(format_tda_datetime(tdaDate))

        ''' Validation data '''        
        x_validateDates = d2r.rawData['DateTime'][d2r.trainLen : (d2r.trainLen + d2r.validateLen)].copy()
        x_validateDates2 = x_validateDates  [range( d2r.trainLen, \
                                                    d2r.trainLen + d2r.validateLen, \
                                                    int(d2r.validateLen / 3))]
        y_validateTargets = d2r.rawData['Close'][d2r.trainLen : (d2r.trainLen + d2r.validateLen)].copy()
        x_ticksVal = []
        x_tickLabelsVal = []
        for tdaDate in x_validateDates2:
            x_ticksVal.append(tdaDate)
            x_tickLabelsVal.append(format_tda_datetime(tdaDate))

        ''' Testing data '''        
        x_testDates = d2r.rawData['DateTime'][(d2r.trainLen + d2r.validateLen) : ].copy()
        x_testDates2 = x_testDates  [range(d2r.trainLen + d2r.validateLen, \
                                           len(x_dates), \
                                           int(d2r.testLen / 3))]
        y_testTargets = d2r.rawData['Close'][(d2r.trainLen + d2r.validateLen) : ].copy()
        x_ticksTest = []
        x_tickLabelsTest = []
        for tdaDate in x_testDates2:
            x_ticksTest.append(tdaDate)
            x_tickLabelsTest.append(format_tda_datetime(tdaDate))

        fig = plt.figure(tight_layout=True)
        fig.suptitle("Feature data for: " + d2r.mlNode, fontsize=14, fontweight='bold')
    
        gs = gridspec.GridSpec(2, 3)

        axULeft = fig.add_subplot(gs[0, 0])
        axULeft.set_title("Training data")
        axULeft.set_xlabel("time periods")
        axULeft.set_ylabel("Data Values")
        axULeft.xaxis.set_ticks(x_trainDates2)
        axULeft.xaxis.set_ticklabels(x_tickLabelsTrain)
        axULeft.plot(d2r.data['DateTime'][ : d2r.trainLen], d2r.data['MACD_flag'][ : d2r.trainLen], linestyle='solid')

        axLLeft = fig.add_subplot(gs[1, 0])
        axLLeft.set_title("Training targets")
        axLLeft.set_xlabel("time periods")
        axLLeft.set_ylabel("Data Values")
        axLLeft.xaxis.set_ticks(x_ticksTrain)
        axLLeft.xaxis.set_ticklabels(x_tickLabelsTrain)

        axUMiddle = fig.add_subplot(gs[0, 1])
        axUMiddle.set_title("Validation data")
        axUMiddle.set_xlabel("time periods")
        axUMiddle.set_ylabel("Data Values")
        axUMiddle.xaxis.set_ticks(x_ticksVal)
        axUMiddle.xaxis.set_ticklabels(x_tickLabelsVal)
    
        axLMiddle = fig.add_subplot(gs[1, 1])
        axLMiddle.set_title("Validation targets")
        axLMiddle.set_xlabel("time periods")
        axLMiddle.set_ylabel("Data Values")
        axLMiddle.xaxis.set_ticks(x_ticksVal)
        axLMiddle.xaxis.set_ticklabels(x_tickLabelsVal)
    
        axURight = fig.add_subplot(gs[0, 2])
        axURight.set_title("Testing data")
        axURight.set_xlabel("time periods")
        axURight.set_ylabel("Data Values")
        axURight.xaxis.set_ticks(x_ticksTest)
        axURight.xaxis.set_ticklabels(x_tickLabelsTest)
    
        axLRight = fig.add_subplot(gs[1, 2])
        axLRight.set_title("Testing targets")
        axLRight.set_xlabel("time periods")
        axLRight.set_ylabel("Data Values")
        axLRight.xaxis.set_ticks(x_ticksTest)
        axLRight.xaxis.set_ticklabels(x_tickLabelsTest)
    
        plt.tight_layout()
        plt.show()

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
    for ndx in range(0, len(dateTimes)):
        testDates.append(format_tda_datetime(dateTimes.iat[ndx, 0]))
        
    tmark = list(range(0, len(testDates), int(len(testDates)/10)))
    
    tmarkDates = []    
    for ndx in range(0, len(testDates)):
        if ndx in tmark:
            tmarkDates.append(testDates[ndx])
        
    return tmark, tmarkDates

def visualizeTestVsForecast(d2r, prediction, axis1, axis2):
    RECENT = len(d2r.trainX[0, :, 0]) * 3
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
    testDateTimes = testData.loc[:, d2r.dataSeriesIDFields]
    tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    axis1.legend()
    axis1.grid(True)

    axis2.set_title("Data series vs. Predictions")
    axis2.set_xlabel("time periods")
    axis2.set_ylabel("Data Value")
    axis2.plot(range(len(prediction)), d2r.testY[:], label='Test series')
    axis2.plot(range(len(prediction)), prediction[:, FORECAST], linestyle='dashed', label='Prediction - 1 period')
    testData = d2r.data.iloc[(len(d2r.data) - len(d2r.testY)) : ]
    testDateTimes = testData.loc[:, d2r.dataSeriesIDFields]
    tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    plt.xticks(tmark, tmarkDates, rotation=20)
    axis2.legend()
    axis2.grid(True)

    return

def visualize_dense(d2r, prediction):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')

    axs[0, 0].set_title("Training Data")
    axs[0, 0].set_xlabel("Sample")
    axs[0, 0].set_ylabel("Values")
    lines = []
    for featureNdx in range (0, d2r.trainX.shape[1]):
        lines.append(axs[0, 0].scatter(range(0, d2r.trainX.shape[0]), d2r.trainX[:, featureNdx], s=0.1))
    lines.append(axs[0, 0].scatter(range(0, d2r.trainX.shape[0]), d2r.trainY, s=0.1))
    axs[0, 0].legend(lines, d2r.preparedFeatures, loc='upper center')
        
    axs[0, 1].set_title("Evaluation Data")
    axs[0, 1].set_xlabel("Sample")
    axs[0, 1].set_ylabel("Values")
    lines = []
    for featureNdx in range (0, d2r.testX.shape[1]):
        lines.append(axs[0, 1].scatter(range(0, d2r.validateX.shape[0]), d2r.validateX[:, featureNdx], s=0.1))
    lines.append(axs[0, 1].scatter(range(0, d2r.validateX.shape[0]), d2r.validateY, s=0.1))
    axs[0, 1].legend(lines, d2r.preparedFeatures, loc='upper center')
    
    axs[0, 2].set_title("Testing Data")
    axs[0, 2].set_xlabel("Sample")
    axs[0, 2].set_ylabel("Values")
    lines = []
    for featureNdx in range (0, d2r.testX.shape[1]):
        lines.append(axs[0, 2].scatter(range(0, d2r.testX.shape[0]), d2r.testX[:, featureNdx], s=0.1))
    lines.append(axs[0, 2].scatter(range(0, d2r.testX.shape[0]), d2r.testY, s=0.1))
    axs[0, 2].legend(lines, d2r.preparedFeatures, loc='upper center')
    
    axs[1, 1].set_title("Prediction")
    axs[1, 1].set_xlabel("Sample")
    axs[1, 1].set_ylabel("Values")
    #axs[1, 1].scatter(d2r.testX, prediction, label='Prediction', linestyle='dashed')
    #axs[1, 1].scatter(d2r.testX, d2r.testY, label='Test data')
    #axs[1, 1].legend()

    axs[1, 2].set_title("Extrapolation")
    x_min = np.min(d2r.testX)
    x_max = np.max(d2r.testX)
    iterable = ((x_max + (((x_max - x_min) / 10) * x)) for x in range(100))
    #x_predict = np.fromiter(iterable, float)
    #y_predict = d2r.model.predict(x=x_predict)
    #axs[1, 2].scatter(x_predict, y_predict)
    #axs[1, 2].set_xlabel("Feature")
    #axs[1, 2].set_ylabel("Target")

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

def visualize_cnn(d2r, prediction):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle("CNN Test Results for - " + d2r.mlNode, fontsize=14, fontweight='bold')
    
    featureAxis = axs[0, 0]
    featureAxis.set_title("Feature Data series")
    featureAxis.set_xlabel("time periods")
    featureAxis.set_ylabel("Data Value")

    lines = []
    dataPoints = d2r.testX.shape[1]
    for batch in range(0, d2r.batches):
        for featureNdx in range (0, d2r.feature_count):
            lines.append(featureAxis.scatter(range(0, dataPoints), d2r.testX[batch, :, featureNdx], s=0.1))
        #lines.append(featureAxis.scatter(range(0, dataPoints), d2r.testY[batch, :], s=0.1))
        featureAxis.legend(lines, d2r.preparedFeatures, loc='upper center')

    predictionAxis = axs[1, 0]
    predictionAxis.set_title("Prediction series")
    predictionAxis.set_xlabel("time periods")
    predictionAxis.set_ylabel("Prediction Value")

    lines = []
    targets = prediction.shape[0]
    dataPoints = prediction.shape[1]
    categories = prediction.shape[2]
    for target in range(0, targets):
        for category in range(0, categories):
            lines.append(predictionAxis.scatter(range(0, dataPoints), prediction[target, :, category], s=0.1))
            #lines.append(featureAxis.scatter(range(0, dataPoints), d2r.testY[batch, :], s=0.1))
    #predictionAxis.legend(lines, d2r.preparedFeatures, loc='upper center')

    print("\n=======================WIP ====================\n\tvisualize_cnn visualization")
    accurate = 0
    false_positive = 0
    false_negative = 0
    for batch in range(0, d2r.batches):
        for period in range(0, dataPoints):
            for target in range(0, d2r.testY.shape[2]):
                if prediction[batch, period, target] == d2r.testY[batch, period, target]:
                    accurate += 1
                elif prediction[batch, period, target] == 1:
                    false_positive += 1
                elif d2r.testY[batch, period, target] == 1:
                    false_negative += 1
    print("Accurate: %s, false negative: %s, false positive: %s" % (accurate, false_negative, false_positive))

    categories = 3
    results = np.zeros([categories, categories]  , dtype=int)
    for batch in range(0, d2r.batches):
        for period in range(0, dataPoints):
            for target in range(0, d2r.testY.shape[2]):
                
                if d2r.testY[batch, period, target] == -1:
                    if prediction[batch, period, target] == -1:
                        results[0, 0] += 1
                    elif prediction[batch, period, target] == 0:
                        results[0, 1] += 1
                    elif prediction[batch, period, target] == 1:
                        results[0, 2] += 1

                if d2r.testY[batch, period, target] == 0:
                    if prediction[batch, period, target] == -1:
                        results[1, 0] += 1
                    elif prediction[batch, period, target] == 0:
                        results[1, 1] += 1
                    elif prediction[batch, period, target] == 1:
                        results[1, 2] += 1

                if d2r.testY[batch, period, target] == 1:
                    if prediction[batch, period, target] == -1:
                        results[2, 0] += 1
                    elif prediction[batch, period, target] == 0:
                        results[2, 1] += 1
                    elif prediction[batch, period, target] == 1:
                        results[2, 2] += 1

    df_results = pd.DataFrame(results)
    print("\nPrediction result statistics\n%s\n" % (df_results.describe().transpose()))
    
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

    if d2r.trainer == TRAINING_AUTO_KERAS:
        d2r.model = d2r.model.export_model()
    d2r.evaluation = d2r.model.evaluate(x=d2r.testX, y=d2r.testY)
    prediction = d2r.model.predict(d2r.testX)

    '''
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
    '''
    
    nx_visualizations = nx.get_node_attributes(d2r.graph, JSON_VISUALIZATIONS)[d2r.mlNode]
    for vizualize in nx_visualizations:
        if vizualize == JSON_VISUALIZE_TRAINING_FIT:
            visualize_fit(d2r)
        elif vizualize == JSON_VISUALIZE_TARGET_SERIES:
            fig1, axs1 = plt.subplots(1, 1)
            fig1.suptitle("Training targets for: " + d2r.mlNode, fontsize=14, fontweight='bold')
            plotTargetValues(d2r, axs1)
            plt.tight_layout()
            plt.show()
        elif vizualize == "summary":
            d2r.model.summary()
        elif vizualize == "dataGroups":
            plotDataGroups(d2r)
        elif vizualize == "normalizationCategorization":
            showNormalizationCategorization(d2r)
        elif vizualize == "cnnResult":
            visualize_cnn(d2r, prediction)
        elif vizualize == "testVsPrediction":
            visualize_fit(d2r)

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
        elif vizualize == "denseCategorization":
            visualize_dense(d2r, prediction)
    
    return
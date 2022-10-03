'''
Created on Jan 15, 2021

@author: Brian
'''
import sys
import logging

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

def sanityCheckMACD(combined=None, npX=None, npY=None, verbose=False, stage=""):
    if combined is None:
        print("\nSanity check of MACD data: 2 numpy tables " + stage)
    else:
        print("\nSanity check of MACD data: 1 dataframe " + stage)
    
    LOOKBACK = 1
    NEGCROSS = 0
    NEUTRAL = 1
    POSCROSS = 2
    nanCnt = 0
    MACD = 0
    MACD_SIGNAL = 1
    MACD_FLAG = 0
    macdFlagCnts = [0, 0, 0]
    correctFlags = [0, 0, 0]
    errFlags = [0, 0, 0]
    if combined is None:
        for batch in range (0, npX.shape[0]):
            '''
            for ndx in range (LOOKBACK, npX.shape[1]):
                if npX[batch, ndx, :].any().isnan() or npY[batch, ndx, :].any().isnan():
                    pass
                else:
            '''
            ndx = npX.shape[1] - 1
            dayDiff = npX[batch, ndx, MACD] - npX[batch, ndx, MACD_SIGNAL]
            priorDiff = npX[batch, ndx-1, MACD] - npX[batch, ndx-1, MACD_SIGNAL]
            if npY[batch, 0, MACD_FLAG] == 0:
                macdFlagCnts[NEUTRAL] += 1
                if priorDiff > 0 and dayDiff > 0:
                    correctFlags[NEUTRAL] += 1
                elif priorDiff < 0 and dayDiff < 0:
                    correctFlags[NEUTRAL] += 1
                elif priorDiff == 0 and dayDiff == 0:
                    correctFlags[NEUTRAL] += 1
                else:
                    if verbose:
                        print("\nMACD flag: %s" % npY[batch, 0, MACD_FLAG])
                        print("MACD        batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD], npX[batch, ndx, MACD]))
                        print("MACD Signal batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD_SIGNAL], npX[batch, ndx, MACD_SIGNAL]))
                        print("batch/day %s/%s difference: %s day %s difference: %s" % (batch, ndx-1, priorDiff, ndx, dayDiff))
                    errFlags[NEUTRAL] += 1
            elif npY[batch, 0, MACD_FLAG] == -1:
                macdFlagCnts[NEGCROSS] += 1
                if (npX[batch, ndx, MACD] < npX[batch, ndx, MACD_SIGNAL]):
                    if priorDiff > 0 and dayDiff < 0:
                        correctFlags[NEGCROSS] += 1
                    else:
                        if verbose:
                            print("\nMACD flag: %s" % npY[batch, 0, MACD_FLAG])
                            print("MACD        batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD], npX[batch, ndx, MACD]))
                            print("MACD Signal batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD_SIGNAL], npX[batch, ndx, MACD_SIGNAL]))
                            print("batch/day %s/%s difference: %s day %s difference: %s" % (batch, ndx-1, priorDiff, ndx, dayDiff))
                        errFlags[NEGCROSS] += 1
                else:
                    if verbose:
                        print("\nMACD flag: %s" % npY[batch, 0, MACD_FLAG])
                        print("MACD        batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD], npX[batch, ndx, MACD]))
                        print("MACD Signal batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD_SIGNAL], npX[batch, ndx, MACD_SIGNAL]))
                        print("batch/day %s/%s difference: %s day %s difference: %s" % (batch, ndx-1, priorDiff, ndx, dayDiff))
                    errFlags[NEGCROSS] += 1
            elif npY[batch, 0, MACD_FLAG] == 1:
                macdFlagCnts[POSCROSS] += 1
                if (npX[batch, ndx, MACD] > npX[batch, ndx, MACD_SIGNAL]):
                    if priorDiff < 0 and dayDiff > 0:
                        correctFlags[POSCROSS] += 1
                    else:
                        if verbose:
                            print("\nMACD flag: %s" % npY[batch, 0, MACD_FLAG])
                            print("MACD        batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD], npX[batch, ndx, MACD]))
                            print("MACD Signal batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD_SIGNAL], npX[batch, ndx, MACD_SIGNAL]))
                            print("batch/day %s/%s difference: %s day %s difference: %s" % (batch, ndx-1, priorDiff, ndx, dayDiff))
                        errFlags[POSCROSS] += 1
                else:
                    if verbose:
                        print("\nMACD flag: %s" % npY[batch, 0, MACD_FLAG])
                        print("MACD        batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD], npX[batch, ndx, MACD]))
                        print("MACD Signal batch/ndx-1: %s/%s\t%s\t\tndx  : %s" % (batch, ndx, npX[batch, ndx-1, MACD_SIGNAL], npX[batch, ndx, MACD_SIGNAL]))
                        print("batch/day %s/%s difference: %s day %s difference: %s" % (batch, ndx-1, priorDiff, ndx, dayDiff))
                    errFlags[POSCROSS] += 1
                    
        print("\nMACD flags count:%s\nCorrect %s\nIncorrect %s" % (macdFlagCnts, correctFlags, errFlags))
    else:
        dfData = combined
    
        #for ndx in range (LOOKBACK, len(dfData)):
        for ndx in dfData.index[1:]:
            if pd.isna(dfData.at[ndx, 'MACD_flag']) or pd.isna(dfData.at[ndx-LOOKBACK, 'MACD_flag']):
                pass
            else:
                dayDiff = dfData.at[ndx, 'MACD'] - dfData.at[ndx, 'MACD_Signal']
                priorDiff = dfData.at[ndx-1, 'MACD'] - dfData.at[ndx-1, 'MACD_Signal']
                if dfData.at[ndx, 'MACD_flag'] == 0:
                    macdFlagCnts[NEUTRAL] += 1
                    if priorDiff > 0 and dayDiff > 0:
                        correctFlags[NEUTRAL] += 1
                    elif priorDiff < 0 and dayDiff < 0:
                        correctFlags[NEUTRAL] += 1
                    else:
                        print("\nMACD flag: %s" % dfData.at[ndx, 'MACD_flag'])
                        print("MACD        ndx-1: %s\t%s\t\tndx  : %s" % (ndx, dfData.at[ndx-1, 'MACD'], dfData.at[ndx, 'MACD']))
                        print("MACD Signal ndx-1: %s\t%s\t\tndx  : %s" % (ndx, dfData.at[ndx-1, 'MACD_Signal'], dfData.at[ndx, 'MACD_Signal']))
                        print("day %s difference: %s day %s difference: %s" % (ndx-1, priorDiff, ndx, dayDiff))
                        errFlags[NEUTRAL] += 1
                elif dfData.at[ndx, 'MACD_flag'] == -1:
                    macdFlagCnts[NEGCROSS] += 1
                    if (dfData.at[ndx, 'MACD'] < dfData.at[ndx, 'MACD_Signal']):
                        if priorDiff > 0 and dayDiff < 0:
                            correctFlags[NEGCROSS] += 1
                        else:
                            print("\nMACD flag: %s" % dfData.at[ndx, 'MACD_flag'])
                            print("MACD        ndx-1: %s\t%s\t\tndx  : %s" % (ndx, dfData.at[ndx-1, 'MACD'], dfData.at[ndx, 'MACD']))
                            print("MACD Signal ndx-1: %s\t%s\t\tndx  : %s" % (ndx, dfData.at[ndx-1, 'MACD_Signal'], dfData.at[ndx, 'MACD_Signal']))
                            print("day %s difference: %s day %s difference: %s" % (ndx-1, priorDiff, ndx, dayDiff))
                            errFlags[NEGCROSS] += 1
                    else:
                        errFlags[NEGCROSS] += 1
                elif dfData.at[ndx, 'MACD_flag'] == 1:
                    macdFlagCnts[POSCROSS] += 1
                    if (dfData.at[ndx, 'MACD'] > dfData.at[ndx, 'MACD_Signal']):
                        if priorDiff < 0 and dayDiff > 0:
                            correctFlags[POSCROSS] += 1
                        else:
                            print("\nMACD flag: %s" % dfData.at[ndx, 'MACD_flag'])
                            print("MACD        ndx-1: %s\t%s\t\tndx  : %s" % (ndx, dfData.at[ndx-1, 'MACD'], dfData.at[ndx, 'MACD']))
                            print("MACD Signal ndx-1: %s\t%s\t\tndx  : %s" % (ndx, dfData.at[ndx-1, 'MACD_Signal'], dfData.at[ndx, 'MACD_Signal']))
                            print("day %s difference: %s day %s difference: %s" % (ndx-1, priorDiff, ndx, dayDiff))
                            errFlags[POSCROSS] += 1
                    else:
                        errFlags[POSCROSS] += 1
                
        print("\nMACD flags count:%s\nCorrect %s\nIncorrect %s" % (macdFlagCnts, correctFlags, errFlags))
    
    return

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

def visualizeTestVsForecast(d2r, prediction):
    fig2, axs = plt.subplots(2, 1)
    fig2.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    
    if len(d2r.trainX.shape) == 3:
        samples = len(d2r.trainX[0, :, 0]) * 3
    if len(d2r.trainX.shape) == 2:
        samples = len(d2r.trainX[0, :]) * 3
    else:
        ex_txt = "training data shape is invalid"
        raise NameError(ex_txt)

    FORECAST = 0
    
    axis1 = axs[0]
    axis1.set_title("Data series vs. Predictions")
    axis1.set_xlabel("samples")
    axis1.set_ylabel("Data Value")
    axis1.plot(range(len(prediction)-samples, len(prediction)), \
               d2r.testY[len(prediction)-samples : ], \
               label='Test series')
    axis1.plot(range(len(prediction)-samples, len(prediction)), \
               prediction[len(prediction)-samples : , FORECAST], \
               linestyle='dashed', label='Prediction - 1 period')
    #testData = d2r.data.iloc[(len(d2r.data) - samples) : ]
    #testDateTimes = testData.loc[:, d2r.dataSeriesIDFields]
    #tmark, tmarkDates = selectDateAxisLabels(testDateTimes)
    axis1.legend()
    axis1.grid(True)

    print("\n============== WIP visualizeTestVsForecast - second axis ================================\n\tvisualization TBD\n================================\n")
    axis2 = axs[1]

    predVals, predInverse, predCounts = np.unique(prediction, return_inverse=True, return_counts=True)
    catVals, catInverse, catCounts = np.unique(d2r.testY, return_inverse=True, return_counts=True)    
    
    if len(catCounts) <= 10:
        dfV = pd.DataFrame(columns=['Prediction', 'Labels'])
        dfV['Prediction'] = prediction[:,0]
        dfV['Labels'] = d2r.testY[:,0]
        axis2 = sns.violinplot(x=dfV['Labels'], y=dfV['Prediction'])
        axis2.legend()
        
    else:
        print("\n============== WIP visualizeTestVsForecast - second axis ================================\n\tcontinuous label values\n================================\n")
        axis2.set_title("WIP")
        axis2.set_xlabel("time periods")
        axis2.set_ylabel("Data Value")
        low = 0
        LOW_PCT = 0.9
        accurate = 0
        high = 0
        HIGH_PCT = 1.1
        for ndx in range(0, len(prediction)):
            if prediction[ndx] < d2r.testY[ndx] * LOW_PCT:
                low += 1
            elif prediction[ndx] > d2r.testY[ndx] * HIGH_PCT:
                high += 1
            else:
                accurate += 1
        
    
    plt.tight_layout()
    plt.show()

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

def reportEvaluationMatrix(d2r, prediction):
    #print(prediction)
    #categories, counts = np.unique(d2r.data, return_counts=True)
    testCategories, testCounts = np.unique(d2r.testY, return_counts=True)
    categoryCount = len(d2r.categories)
    print("\nThe shape of the test data is [%s, %s]" % (prediction.shape[0], prediction.shape[1]))
    print("Testing labels are %s with %s distribution\n" % (testCategories, testCounts))
    
    THRESHOLD = 0.66
    results = np.zeros([len(d2r.categories), len(d2r.categories)], dtype=float)
    thresholdResults = np.zeros([len(d2r.categories), len(d2r.categories)], dtype=float)

    '''
    for batch in range(0, len(d2r.testY)):
        for sample in range (0, )
    '''
    if len(d2r.testY.shape) == 2: # dense models
        ''' ++++++++++++++++++++++++++++++++++++++++
        print("\n============== WIP =============\n\tdense and RNN model evaluation matrix assumes 3 categories of specific values\n================================\n")
        '''
        ''' dense models '''
        '''
        for batch in range(0, len(d2r.testY)):
            maxndx = np.argmax(prediction[batch])
            if d2r.testY[batch, 0] == -1:
                results[0, maxndx] += 1
            elif d2r.testY[batch, 0] == 0:
                results[1, maxndx] += 1
            elif d2r.testY[batch, 0] == 1:
                results[2, maxndx] += 1
    
            if prediction[batch, maxndx] > THRESHOLD:
                if d2r.testY[batch, 0] == -1:
                    thresholdResults[0, maxndx] += 1
                elif d2r.testY[batch, 0] == 0:
                    thresholdResults[1, maxndx] += 1
                elif d2r.testY[batch, 0] == 1:
                    thresholdResults[2, maxndx] += 1

        print("Rows represent test label values, Columns the predicted most likely choice")
        print("All results")
        pdResults = pd.DataFrame(data=results, index=['-1', '0', '1'], columns=['-1', '0', '1'])
        print(pdResults)
        
        print("\nResults above probability threshold value of %s" % THRESHOLD)
        print("Rows represent test label values, Columns the predicted most likely choice")
        pdResults = pd.DataFrame(data=thresholdResults, index=['-1', '0', '1'], columns=['-1', '0', '1'])
        print(pdResults)

        print("\n============== WIP =============\n\tflexible category values\n================================\n")
        '''

        catDict = dict()
        for ndx in range (0, len(d2r.categories)):
            catDict[d2r.categories[ndx]] = ndx

        for batch in range(0, len(d2r.testY)):
            labelNdx = int(d2r.testY[batch])
            labelNdx = catDict.get(labelNdx)
            predictionNdx = np.argmax(prediction[batch])
            
            results[labelNdx, predictionNdx] += 1
            if prediction[batch, predictionNdx] > THRESHOLD:
                thresholdResults[labelNdx, predictionNdx] += 1
        
        #print("\n=======================WIP ====================\n\tvisualize category prediction vs label\n===============================================\n")
        print("Rows represent actual test label values, Columns the predicted most likely category")
        #print("All results")
        #print(results)
        colDesc = "Label"
        hdrRow = "\t\tModel prediction\n\t"
        for predictionNdx in range (0, len(d2r.categories)):
            hdrRow = hdrRow + "\t"
            hdrRow = hdrRow + "{!s}".format(d2r.categories[predictionNdx])
        print(hdrRow)
        textRow = list()
        for labelNdx in range (0, len(d2r.categories)):
            textRow.append(colDesc[labelNdx])
            textRow[labelNdx] = textRow[labelNdx] + "\t"
            textRow[labelNdx] = textRow[labelNdx] + "{}".format(d2r.categories[labelNdx])
            textRow[labelNdx] = textRow[labelNdx] + "\t"
            for predictionNdx in range (0, len(d2r.categories)):
                textRow[labelNdx] = textRow[labelNdx] + "{0:d}".format(int(results[labelNdx, predictionNdx]))
                textRow[labelNdx] = textRow[labelNdx] + "\t"
            print(textRow[labelNdx])

    elif len(d2r.testY.shape) == 3: #CNN models
        ''' CNN models '''
        catDict = dict()
        for ndx in range (0, len(d2r.categories)):
            catDict[d2r.categories[ndx]] = ndx

        for batch in range(0, len(d2r.testY)):
            labelNdx = int(d2r.testY[batch])
            labelNdx = catDict.get(labelNdx)
            predictionNdx = np.argmax(prediction[batch])
            
            results[labelNdx, predictionNdx] += 1
            if prediction[batch, predictionNdx] > THRESHOLD:
                thresholdResults[labelNdx, predictionNdx] += 1
        
        #print("\n=======================WIP ====================\n\tvisualize category prediction vs label\n===============================================\n")
        print("Rows represent actual test label values, Columns the predicted most likely category")
        #print("All results")
        #print(results)
        colDesc = "Label"
        hdrRow = "\t\tModel prediction\n\t"
        for predictionNdx in range (0, len(d2r.categories)):
            hdrRow = hdrRow + "\t"
            hdrRow = hdrRow + "{!s}".format(d2r.categories[predictionNdx])
        print(hdrRow)
        textRow = list()
        for labelNdx in range (0, len(d2r.categories)):
            textRow.append(colDesc[labelNdx])
            textRow[labelNdx] = textRow[labelNdx] + "\t"
            textRow[labelNdx] = textRow[labelNdx] + "{}".format(d2r.categories[labelNdx])
            textRow[labelNdx] = textRow[labelNdx] + "\t"
            for predictionNdx in range (0, len(d2r.categories)):
                textRow[labelNdx] = textRow[labelNdx] + "{0:d}".format(int(results[labelNdx, predictionNdx]))
                textRow[labelNdx] = textRow[labelNdx] + "\t"
            print(textRow[labelNdx])

    return

def visualize_cnn(d2r, prediction):

    try:
        err_txt = "*** An exception occurred visualizing CNN results ***"
        if len(prediction.shape) == 2:
            categories = prediction.shape[1]
            if categories == 1:
                dfPrediction = pd.DataFrame(prediction)
                print("\nprediction\n%s" % (dfPrediction.describe().transpose()))
            else:
                d2r.visualize_categorization_samples()
        else:
            err_txt = "No preparation sequence specified"
            raise NameError(err_txt)                
    
    except Exception:
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" +exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += s
        logging.debug(exc_txt)
        sys.exit(exc_txt)

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

    d2r.evaluation = d2r.model.evaluate(x=d2r.testX, y=d2r.testY, verbose=0)
    prediction = d2r.model.predict(d2r.testX, verbose=0)

    if len(prediction.shape) == 2:
        print("\nevaluation prediction shape is [%s, %s]\n" % (prediction.shape[0], prediction.shape[1]))
    elif len(prediction.shape) == 3:
        print("\nevaluation prediction shape is [%s, %s, %s]\n" % (prediction.shape[0], prediction.shape[1], prediction.shape[2]))

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
            visualizeTestVsForecast(d2r, prediction)
        elif vizualize == "categoryMatrix":
            reportEvaluationMatrix(d2r, prediction)
        elif vizualize == "denseCategorization":
            visualize_dense(d2r, prediction)
    
    return
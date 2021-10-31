'''
Created on Oct 6, 2021

@author: brian
'''
import os
import datetime as dt
from datetime import date
from datetime import timedelta
import time
import logging
from re import sub
from decimal import Decimal

import matplotlib.pyplot as plt
#import matplotlib.dates as md 
import numpy as np
import pandas as pd
#import pickle

from configuration import get_ini_data
from configuration import read_config_json

'''
START_NDX = 0
END_NDX = 1
CLOSE_NDX = 50
'''
IGNORELIST = ["$VIX.X", "VWSYF"]
PLOT_RAWDATA = 0
PLOT_OUTLIERS = 1
PLOT_REPEATS = 2
PLOTS = 3

def dfAccessDemo():
    print("\nexamination of DataFrame functionality ------------------")
    df2 = pd.DataFrame(np.array([['C', '2021-10-01', 3], \
                                     ['IBM', '2021-10-01', 6], \
                                     ['C', '2021-10-04', 7], \
                                     ['IBM', '2021-10-04', 8], \
                                     ['IBM', '2021-10-05', 10], \
                                     ['C', '2021-10-05', 9]]),
                   columns=['symbol', 'date', 'c'])
    print(df2)
    print("\nloc[4]\n%s" % (df2.loc[4]))
    print("\nSlice loc[4:]\n%s" % (df2.loc[4:]))
    print("\ninfo: \n%s" % df2.info)
    print("\ncolumns: %s\n\tcolumn[2]=%s" % (df2.columns, df2.columns[2]))
    print("\nindex: %s\n\t index[3] = %s" % (df2.index, df2.index[3]))
    print("\naxes: %s:" % df2.axes)
    print("\nndim: %s:" % df2.ndim)
    print("\nsize: %s:" % df2.size)
    #print("shape: %s:" % df2.shape)
    df2.insert(2, 'Inserted', [100,101,102,103,104,105])
    df2.insert(2, 'Fixed', "TBD")
    print("\nExpanded\n%s" % df2)
    for label,content in df2.items():
        if label == "Inserted":
            print("\nitems - label==Inserted: %s, content:\n%s" % (label, content))
            print("\nManipulate index ------------------")
    df3 = df2.reindex(index=["A", 5, "C", 1, "E", "F"])
    print(df3)
    df3 = df3.reset_index(drop=True)
    print(df3)
    df4 = df2.set_index("symbol")
    print(df4)
    df5 = df2.set_index(["symbol","date"])
    print(df5)
    print("\n2 level loc\n%s\nInserted = %s" % \
          (df5.loc[("IBM","2021-10-01")], df5.loc[("IBM","2021-10-01")]["Inserted"]))
    print("\nMultiIndedx ------------------")
    mi1 = pd.MultiIndex.from_frame(df2)
    print(mi1)
    mi=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=['l1', 'l2'])
    cols=['c1', 'c2']
    d1=pd.DataFrame(index=mi, columns=cols)
    d1.loc[('iv1','iv2'),:] = ['c1val', 'c2val']
    print("\nback to the work at hand ------------------")
                    
    return

def IgnoreSymbols(df_sorted):
    idx=pd.IndexSlice
    for symbol  in IGNORELIST:
        if len(df_sorted.loc[idx[:, :, [symbol], :]]) > 0:
            print("Ignoring: %s" % symbol)
            df_sorted.drop(labels=symbol, level='Symbol', inplace=True)
        else:
            print("There are no samples for %s" % symbol)

    return 

def EliminateOutliers(df_pct_result):
    df_outliers = pd.DataFrame(data={'Upper': [5, 0, 0], 'Lower': [5, 0, 10]}, \
                               index=[['MACD','OBV','RS'], \
                                      ['negative Divergence', 'OBV6', 'RS1 & RS5']])
    
    analysis = df_pct_result.iloc[0].name[0]
    trigger = df_pct_result.iloc[0].name[1]
    Smax = df_pct_result.max(axis=1)
    Smin = df_pct_result.min(axis=1)
    Smax.sort_values(inplace=True)
    upper = df_outliers.loc[(analysis, trigger), "Upper"] 
    lower = df_outliers.loc[(analysis, trigger), "Lower"]
    df_return = df_pct_result
              
    for ndx in range(1, upper + 1):
        if len(Smax) - ndx >= 0:
            print("Dropping high outlier: %s, (%s, %s, %s, %s)" % \
                  (ndx, analysis, trigger, Smin.index[len(Smax) - ndx][2], Smin.index[len(Smax) - ndx][3]))
            df_return = df_return.drop([(analysis, trigger, \
                                 Smax.index[len(Smax) - ndx][2], \
                                 Smax.index[len(Smax) - ndx][3])], \
                                 inplace=False)
    
    for ndx in range(1, lower + 1):
        if ndx - 1 >= 0:
            print("Dropping low outlier: %s, (%s, %s, %s, %s)" % \
                  (ndx, analysis, trigger, Smax.index[ndx - 1][2], Smin.index[ndx - 1][3]))
            df_return.drop([(analysis, trigger, \
                                 Smin.index[ndx - 1][2], \
                                 Smin.index[ndx - 1][3])], \
                                 inplace=True)
    
    print("Clipping maximum value at 2.0")
    df_return.clip(upper=2.0, inplace=True)
    
    return df_return

def FindRepeats(df_samples, sampleIndex):
    s_return = pd.Series(dtype='UInt64')
    seq = False
    
    for ndx in range(1, len(df_samples)):
        sampleSymbol = df_samples.iloc[ndx].name[2]
        sampleDate = date.fromisoformat(df_samples.iloc[ndx].name[3])
        priorSampleSymbol = df_samples.iloc[ndx - 1].name[2]
        priorSampleDate = date.fromisoformat(df_samples.iloc[ndx - 1].name[3])
        if sampleSymbol == priorSampleSymbol:
            interval = sampleDate - priorSampleDate
            dayOfWeek = priorSampleDate.isoweekday()
            # consecutive trading days
            if interval.days == 1:
                if seq == False:
                    s_return.loc[ndx - 1] = ndx - 1
                    seq = True
            # Friday to Monday are consecutive trading days
            elif interval.days == 3 and dayOfWeek == 5:
                if seq == False:
                    s_return.loc[ndx - 1] = ndx - 1
                    seq = True
                else:
                    seq = False
            else:
                seq = False
        else:
            seq = False

    return s_return

def EvaluateSignals():
    #Read signal file
    signalFile =  "D:\\brian\AI Projects\\Trading Strategy Signals.csv"
    if os.path.isfile(signalFile):
        df_signals = pd.read_csv(signalFile)
        df_signals = df_signals.dropna()
        
        #Sort signals analysis / signal / symbol / date        
        SortOrder = ['Analysis','Trigger','Symbol','Date']
        mi_signals = pd.MultiIndex.from_frame(df_signals)

        df_sorted = df_signals.sort_values(SortOrder)
        #df_sorted.insert(0, 'row', range(0, len(df_sorted)))
        df_sorted = df_sorted.set_index(SortOrder)
        '''
        print("\ndf_sorted - Sorted matrix of identified trading opportunities:\nSorted by:\n\t%s\n%s" \
              % (SortOrder, df_sorted))
        '''
        IgnoreSymbols(df_sorted)

        TriggerGrouping = ['Analysis','Trigger']
        grpby_trigger = df_sorted.groupby(by=TriggerGrouping)
        '''
        print("\ngrpby_trigger - Groupby of trading indices grouped by\n\t%s:\n%s" \
              % (TriggerGrouping, grpby_trigger.indices))
        '''

        grpby_symbol = df_sorted.groupby(by='Symbol')
        '''
        print("\ngrpby_symbol - Groupby symbols used to pre-load market data for all symbols identified\n%s\nhas %s symbols" \
              % (grpby_symbol.indices, len(grpby_symbol.indices)))
        '''
        
        df_mkt_data = pd.DataFrame()
        print("Loading historical data")
        for symbol in grpby_symbol.indices:
            symbol_data = "D:\\brian\AI Projects\\tda\\market_analysis_data\\" + symbol + ".csv"
            if os.path.isfile(symbol_data):
                #print(symbol)
                df_add = pd.read_csv(symbol_data)
                df_add.insert(0, 'Symbol', symbol)
                df_mkt_data = pd.concat([df_mkt_data, df_add])
        #df_mkt_data.insert(0, 'row', range(0, len(df_mkt_data)))
        #df_mkt_data = df_mkt_data.insert(0, "row", 0)
        df_mkt_data = df_mkt_data.set_index(['Symbol','date'])
        #print("\nConcatenated market data for all symbols:\n%s" % df_mkt_data)
            
        #for each analysis / symbol
        for analysis in grpby_trigger.indices:
            print("Analysing: %s, %s" % (analysis[0],analysis[1]))

            '''
            print("\nTrading trigger: %s\nis identified in DataFrame indices\n%s" \
                   % (analysis, grpby_trigger.groups[analysis]))
            print("\nIndices 0,1,3: %s, %s, %s" % \
                  (grpby_trigger.groups[analysis][0],
                  grpby_trigger.groups[analysis][1],
                  grpby_trigger.groups[analysis][3]))
            '''
            df_r = pd.DataFrame(index=["Analysis", "Trigger", "Symbol", "Date"])
            df_result = pd.DataFrame()
            df_result.insert(0, "Analysis", "")
            df_result.insert(1, "Trigger", "")
            df_result.insert(2, "Symbol", "")
            df_result.insert(3, "Date", "")
            df_result = df_result.set_index(SortOrder)
            
            df_pct_result = pd.DataFrame()
            df_pct_result.insert(0, "Analysis", "")
            df_pct_result.insert(1, "Trigger", "")
            df_pct_result.insert(2, "Symbol", "")
            df_pct_result.insert(3, "Date", "")
            df_pct_result = df_pct_result.set_index(SortOrder)

            #x_tick = np.zeros(60)
            for ndx in range(0, 60):
                df_result.insert(ndx, ndx, Decimal(0.0))
                df_pct_result.insert(ndx, ndx, Decimal(1.0))
                #x_tick[ndx] = ndx
                
            for ndxT in grpby_trigger.groups[analysis]:
                '''
                trigger = df_sorted.loc[ndxT]
                tmp = df_sorted.loc[(ndxT[0], ndxT[1], ndxT[2], ndxT[3])]
                print("ndxT = (%s,%s,%s,%s)\ntmp = \n%s" % (ndxT[0],ndxT[1],ndxT[2],ndxT[3], tmp))
                if len(tmp) > 1:
                    print("Duplicate data")
                '''
                close_str = df_sorted.at[ndxT,'Close']
                close_d0 = Decimal(sub(r'[^\d.]', '', close_str))
                df_result.at[ndxT, 0] = close_d0
                df_pct_result.at[ndxT, 0] = 1.0
                #df_d0 = df_mkt_data.loc[(ndxT[2], ndxT[3]) : , : ]
                df_d0 = df_mkt_data.loc[(ndxT[2], ndxT[3]) : , 'Close' ]
                ndx = 0
                for mkt_data in df_d0.iteritems():
                    if ndx > 0:
                        close_dn = Decimal(mkt_data[1])
                        #close_dn = Decimal(sub(r'[^\d.]', '', close_str))
                        df_result.at[ndxT, ndx] = close_dn
                        df_pct_result.at[ndxT, ndx] = close_dn / close_d0
                    ndx += 1                 
                    if ndx > 59:
                        break
                    if ndx > len(df_d0):
                        break
                    
            #plot all symbol % change
            fig, axes = plt.subplots(nrows=1, ncols=PLOTS)
            fig.suptitle(analysis)
            for ndx in range(0, PLOTS):
                if ndx == PLOT_RAWDATA:
                    axes[PLOT_RAWDATA].set_title("All samples")
                    axes[PLOT_RAWDATA].set_xlabel('Days following signal')
                    axes[PLOT_RAWDATA].set_ylabel('% change')
                    axes[PLOT_RAWDATA].yaxis.grid(True)
                    for ndx2, row in df_pct_result.iterrows():
                        axes[PLOT_RAWDATA].plot(row)
                elif ndx == PLOT_OUTLIERS:
                    df_clip = EliminateOutliers(df_pct_result)
                    axes[PLOT_OUTLIERS].set_title("Outliers eliminated / clipped")
                    axes[PLOT_OUTLIERS].set_xlabel('Days following signal')
                    axes[PLOT_OUTLIERS].set_ylabel('% change')
                    axes[PLOT_OUTLIERS].yaxis.grid(True)
                    for ndx2, row in df_clip.iterrows():
                        axes[PLOT_OUTLIERS].plot(row)
                elif ndx == PLOT_REPEATS:
                    axes[PLOT_REPEATS].set_title("Symbols identified on consecutive days")
                    axes[PLOT_REPEATS].set_xlabel('Days following first signal')
                    axes[PLOT_REPEATS].set_ylabel('% change')
                    axes[PLOT_REPEATS].yaxis.grid(True)
                    s_repeats = FindRepeats(df_pct_result, SortOrder)
                    df_pct_repeats = df_pct_result.iloc[s_repeats].copy()
                    df_pct_repeats.clip(upper=2.0, inplace=True)
                    for ndx2, row in df_pct_repeats.iterrows():
                        axes[PLOT_REPEATS].plot(row)
                    
            plt.show()            
            
    return

if __name__ == '__main__':
    print ("Dave, I'm not sure if these signals are useful but we'll see\n")
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    # Get external initialization details
    app_data = get_ini_data("TDAMERITRADE")
    json_config = read_config_json(app_data['config'])
    #EODData = app_data['eod_data']
    MarketAnalysisData = app_data["market_analysis_data"]
    
    #dir_data = get_ini_data("LOCALDIRS")
    #AIWork = dir_data['aiwork']

    try:    
        log_file = json_config['logFile']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = json_config['outputFile']
        output_file = output_file + ' {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
        f_out = open(output_file, 'w')    
        
    except Exception:
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    print ("Logging to", log_file)
    logger = logging.getLogger('chandra_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Evaluating trading signals actually generated')
    
    ''' Configure Pandas for friendlier output formats '''
    # Use 3 decimal places in output display
    pd.set_option("display.precision", 3)
    # Don't wrap repr(DataFrame) across additional lines
    pd.set_option("display.expand_frame_repr", False)
    # Set max rows displayed in output to 25
    pd.set_option("display.max_rows", 25)    
    
    EvaluateSignals()

    '''     Clean up and prepare to exit     '''
    f_out.close()

    print ("\nDave, I hope this showed the signals are useful. Goodbye")

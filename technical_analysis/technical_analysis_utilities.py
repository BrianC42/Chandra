'''
Created on Jul 16, 2020

@author: Brian
'''
import pandas as pd
import numpy as np
#from numpy import NaN

#import pivottablejs
#from pivottablejs import pivot_ui
#from IPython.core.display import HTML

''' Globals '''
TIME_FRAMES = [1, 5, 10, 20, 40]
BAND_UPPER  = [   -1.0, -0.5, -0.2, -0.1, -0.05, -0.01,      0.01, 0.05,  0.1, 0.2, 0.5, 1.0, 1000.0]
BAND_LOWER  = [-1000.0, -1.0, -0.5, -0.2,  -0.1, -0.05,     -0.01, 0.01, 0.05, 0.1, 0.2, 0.5,    1.0]
COUNTS_OFFSET = 6
EVAL_COLS   = ["symbol", "segment", "classification", "analysis", "condition", "duration",   "Neg", '-5', '-4', '-3',  '-2',  '-1', 'Neutral',  '1',  '2', '3', '4', '5',  'Pos']
ZERO_DATA   = [[""     , ""       , ""              , ""        , ""         , ""        ,       0,    0,    0,    0,     0,     0,         0,    0,    0,   0,   0,   0,      0]]

def present_evaluation(eval_results, filename):
    eval_results.to_csv(filename + ".csv", index=False)
    return

def initialize_eval_results():
    eval_results = pd.DataFrame(columns=EVAL_COLS)    
    return eval_results

def add_analysis_results(cumulative_evaluation, incremental_evaluation):
    ndx = 0
    while ndx < incremental_evaluation.shape[0]:
        found, i_ndx = find_analysis_condition_duration(incremental_evaluation.at[ndx, 'symbol'], \
                                                        cumulative_evaluation, \
                                                        [incremental_evaluation.at[ndx, 'analysis'], incremental_evaluation.at[ndx, 'condition']], \
                                                        incremental_evaluation.at[ndx, 'duration'])
        if found:
            cumulative_evaluation.iloc[i_ndx, 6:18] += incremental_evaluation.iloc[ndx, 6:18]
        else:
            cumulative_evaluation = cumulative_evaluation.append(incremental_evaluation.iloc[ndx, :], ignore_index=True)
        ndx += 1
        
    return cumulative_evaluation

def find_analysis_condition_duration(symbol, eval_results, analysis_condition, duration):
    found = False
    ndx = 0
    while ndx < eval_results.shape[0]:
        if eval_results.at[ndx, 'symbol'] == symbol and \
            eval_results.at[ndx, 'analysis'] == analysis_condition[0] and \
            eval_results.at[ndx, 'condition'] == analysis_condition[1] and \
            eval_results.at[ndx, 'duration'] == duration:
            found = True
            break
        ndx += 1
    
    return found, ndx

def add_analysis_condition(symbol, eval_results, analysis_condition, duration):
    df_add = pd.DataFrame(data=ZERO_DATA, columns=EVAL_COLS)
    df_add.at[0, 'symbol'] = symbol
    df_add.at[0, 'analysis'] = analysis_condition[0]
    df_add.at[0, 'condition'] = analysis_condition[1]
    df_add.at[0, 'duration'] = duration
    eval_results = eval_results.append(df_add, ignore_index=True)

    return eval_results.shape[0]-1, eval_results

def find_sample_col(df_sample, result_duration):
    col = 'Neutral'
    if result_duration == TIME_FRAMES[0]:
        comp_col = '1 day change'
    elif result_duration == TIME_FRAMES[1]:
        comp_col = '5 day change'
    elif result_duration == TIME_FRAMES[2]:
        comp_col = '10 day change'
    elif result_duration == TIME_FRAMES[3]:
        comp_col = '20 day change'
    else:
        comp_col = '40 day change'

    band_ndx = 0
    while band_ndx < len(BAND_UPPER):
        if df_sample.at[comp_col] >= BAND_LOWER[band_ndx] and \
            df_sample.at[comp_col] < BAND_UPPER[band_ndx]:
            col = EVAL_COLS[band_ndx + COUNTS_OFFSET]
            break
        band_ndx += 1
        
    return col

def increment_sample_counts(symbol, eval_results, analysis_condition, df_sample):
    duration_ndx = 0
    while duration_ndx < len(TIME_FRAMES):
        found, result_ndx = find_analysis_condition_duration(symbol, eval_results, analysis_condition, TIME_FRAMES[duration_ndx])
        if not found:
            result_ndx, eval_results = add_analysis_condition(symbol, eval_results, analysis_condition, TIME_FRAMES[duration_ndx])
        duration_col = find_sample_col(df_sample, TIME_FRAMES[duration_ndx])
        eval_results.at[result_ndx, duration_col] += 1
        duration_ndx += 1
    return eval_results

def sample_count(symbol, df_data, eval_results):
    analysis_condition = ['sample', 'counts']
    symbol = "Baseline"
        
    ndx = 0
    while ndx < df_data.shape[0]:
        eval_results = increment_sample_counts(symbol, eval_results, analysis_condition, df_data.iloc[ndx, :])
        df_data.loc[ndx, 'segment'] = 'Baseline'
        df_data.loc[ndx, 'classification'] = 'Baseline'
        ndx += 1
    return eval_results

def eval_combinations(symbol, df_data, eval_results):
    analysis_condition = ['combination1', 'MACD+Close']
        
    ndx = 0
    while ndx < df_data.shape[0]:
        if df_data.at[ndx, 'MACD_Buy'] == True:
            if not df_data.at[ndx, 'MACD'] and \
                df_data.at[ndx, 'MACD'] < 0.25 and \
                df_data.at[ndx, 'Close'] > df_data.at[ndx, 'SMA20']:
                increment_sample_counts(symbol, eval_results, analysis_condition, df_data[ndx])
        ndx += 1
    return eval_results

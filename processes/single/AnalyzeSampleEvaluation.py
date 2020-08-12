'''
Created on Aug 8, 2020

@author: Brian
'''
import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import logging

from configuration import get_ini_data
from configuration import read_config_json

TIME_FRAMES = [1, 5, 10, 20, 40]
BAND_UPPER  = [   -1.0, -0.5, -0.2, -0.1, -0.05, -0.01,      0.01, 0.05,  0.1, 0.2, 0.5, 1.0, 1000.0]
BAND_LOWER  = [-1000.0, -1.0, -0.5, -0.2,  -0.1, -0.05,     -0.01, 0.01, 0.05, 0.1, 0.2, 0.5,    1.0]
BAND_LABELS = [-1000.0, -1.0, -0.5, -0.2,  -0.1, -0.05,      0.00, 0.05,  0.1, 0.2, 0.5, 1.0, 1000.0]
EVAL_COLS   = ["symbol", "segment", "classification", "analysis", "condition", "duration",   "Neg", '-5', '-4', '-3',  '-2',  '-1', 'Neutral',  '1',  '2', '3', '4', '5',  'Pos']
INDEX_CLASS = ["duration", "classification", "segment"]
INDEX_SEG = ["duration", "segment", "classification"]
INDEX_ANALYSIS = ["duration", "analysis", "condition"]
ANALYSIS_CATEGORIES = ["MACD","BB","Accumulation Distribution","Average Directional Index", "OBV", "Relative Strength", "sample", "SO"]
VALUE_COLS = ["Neg", '-5', '-4', '-3',  '-2',  '-1', 'Neutral',  '1',  '2', '3', '4', '5',  'Pos']
TOTAL_COLS = ["Neg", '-5', '-4', '-3',  '-2',  '-1', 'Neutral',  '1',  '2', '3', '4', '5',  'Pos', 'total']
BASELINE_COLS = ["duration", "Neg", '-5', '-4', '-3',  '-2',  '-1', 'Neutral',  '1',  '2', '3', '4', '5',  'Pos', 'total']


def read_evaluation_result(json_config):
    segmentation = json_config['evaluateoutputFile' ] +'.csv'
    if os.path.isfile(segmentation):
        df_eval = pd.read_csv(segmentation)
    return df_eval

def display_baseline(df_baseline_pct):
    x = np.arange(len(BAND_LABELS))  # the label locations
    width = 0.15  # the width of the bars
    fig, ax = plt.subplots()

    rects1 = ax.bar(x - (2*width),  df_baseline_pct.iloc[0, 1:14], width, label='1 day')
    rects2 = ax.bar(x - width,      df_baseline_pct.iloc[1, 1:14], width, label='5 day')
    rects3 = ax.bar(x,              df_baseline_pct.iloc[2, 1:14], width, label='10 day')
    rects4 = ax.bar(x + width,      df_baseline_pct.iloc[3, 1:14], width, label='20 day')
    rects5 = ax.bar(x + (2*width),  df_baseline_pct.iloc[4, 1:14], width, label='40 day')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('fraction of total samples')
    ax.set_title('Baseline distribution of samples')
    ax.set_xticks(x)
    ax.set_xticklabels(BAND_LABELS)
    ax.legend()
    
    fig.tight_layout()
    plt.show()    
    return

def display_comparison_to_baseline(df_analysis, comparison, df_baseline_pct):
    x = np.arange(len(BAND_LABELS))  # the label locations
    width = 0.2  # the width of the bars
    fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Baseline compared to ' + comparison[1] + ', ' + comparison[2])

    anal5 = ax00.bar(x - (0.5 * width), df_analysis.loc[(5,comparison[1],comparison[2])], width, label='analysis')
    base5 = ax00.bar(x + (0.5 * width), df_baseline_pct.iloc[1, 1:14], width, label='baseline')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax00.set_ylabel('fraction of total samples')
    ax00.set_title('After 5 days')
    ax00.set_xticks(x)
    ax00.set_xticklabels(BAND_LABELS)
    ax00.legend()

    anal10 = ax01.bar(x - (0.5 * width), df_analysis.loc[(10,comparison[1],comparison[2])], width, label='analysis')
    base10 = ax01.bar(x + (0.5 * width), df_baseline_pct.iloc[2, 1:14], width, label='baseline')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax01.set_ylabel('fraction of total samples')
    ax01.set_title('After 10 days')
    ax01.set_xticks(x)
    ax01.set_xticklabels(BAND_LABELS)
    ax01.legend()
    
    anal20 = ax10.bar(x - (0.5 * width), df_analysis.loc[(20,comparison[1],comparison[2])], width, label='analysis')
    base20 = ax10.bar(x + (0.5 * width), df_baseline_pct.iloc[3, 1:14], width, label='baseline')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax10.set_ylabel('fraction of total samples')
    ax10.set_title('After 20 days')
    ax10.set_xticks(x)
    ax10.set_xticklabels(BAND_LABELS)
    ax10.legend()

    anal40 = ax11.bar(x - (0.5 * width), df_analysis.loc[(40,comparison[1],comparison[2])], width, label='analysis')
    base40 = ax11.bar(x + (0.5 * width), df_baseline_pct.iloc[4, 1:14], width, label='baseline')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax11.set_ylabel('fraction of total samples')
    ax11.set_title('After 40 days')
    ax11.set_xticks(x)
    ax11.set_xticklabels(BAND_LABELS)
    ax11.legend()
    
    fig.tight_layout()
    plt.show() 
    return

def pivot(df_eval):
    cols = df_eval.columns
    
    df_baseline = df_eval[df_eval['symbol'] == 'Baseline']
    #print(df_baseline.to_string())
    df_baseline.insert(len(cols), 'total', df_baseline[VALUE_COLS].sum(axis=1))
    #print(df_baseline.to_string())

    df_baseline_pct = pd.DataFrame(index=df_baseline.index, columns=df_baseline.columns)
    row_ndx = 0
    while row_ndx < df_baseline_pct.shape[0]:
        col_ndx = 0
        while col_ndx < len(df_baseline_pct.columns):
            if df_baseline_pct.columns[col_ndx] in TOTAL_COLS:
                df_baseline_pct.iloc[row_ndx, col_ndx] = df_baseline.iloc[row_ndx, col_ndx] / df_baseline.iloc[row_ndx, 19]
            col_ndx += 1
        df_baseline_pct.at[row_ndx, 'duration'] = df_baseline.at[row_ndx, 'duration']
        row_ndx += 1
    df_baseline_pct = df_baseline_pct.drop(["symbol", "segment", "classification", "analysis", "condition"], axis=1)
    str_buff = df_baseline_pct.to_string()
    print(str_buff)
    
    display_baseline(df_baseline_pct)
        
    df_analysis = df_eval[df_eval['symbol'] != 'Baseline']
    df_analysis.insert(len(cols), 'total', df_analysis[VALUE_COLS].sum(axis=1))
    
    df_counts = pd.pivot_table(df_analysis, index=INDEX_ANALYSIS, values=TOTAL_COLS, aggfunc=np.sum)
    str_buff = df_counts.to_string()
    print(str_buff)

    df_pct = pd.DataFrame(index=df_counts.index, columns=df_counts.columns)
    row_ndx = 0
    while row_ndx < df_pct.shape[0]:
        col_ndx = 0
        while col_ndx < len(df_pct.columns):
            df_pct.iloc[row_ndx, col_ndx] = df_counts.iloc[row_ndx, col_ndx] / df_counts.iloc[row_ndx, 13]
            col_ndx += 1
        row_ndx += 1
    df_pct = df_pct[TOTAL_COLS]
    str_buff = df_pct.to_string()
    print(str_buff)
    df_pct = df_pct.drop(["total"], axis=1)
    row_ndx = 0
    while row_ndx < df_pct.shape[0] / 5:
        display_comparison_to_baseline(df_pct, df_pct.index[row_ndx], df_baseline_pct)
        row_ndx += 1

    return

if __name__ == '__main__':
    print ("Welcome back Pythia, I'm ready to get to work\n")
    now = dt.datetime.now()
    
    # Get external initialization details
    app_data = get_ini_data("CHANDRA")
    json_config = read_config_json(app_data['config'])

    try:    
        log_file = json_config['analysislogFile']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = json_config['analysisoutputFile']
        output_file = output_file + ' {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
        f_out = open(output_file, 'w')    
        
        # global parameters
        #logging.debug("Global parameters")
    
    except Exception:
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    #print ("Logging to", log_file)
    logger = logging.getLogger('chandra_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Processing evaluation results')

    df_eval = read_evaluation_result(json_config)
    pivot(df_eval)
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nI hope that gave you what you need")

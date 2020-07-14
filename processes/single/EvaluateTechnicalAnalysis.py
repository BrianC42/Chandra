'''
Created on Jul 9, 2020

@author: Brian
'''
import os
import datetime as dt
import time
import logging
import pandas as pd
import numpy as np

import matplotlib

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists

def present_evaluation(f_out, eval_results):
    print("Evaluation results of technical analysis\n%s" % eval_results.to_string())
    f_out.write ('Evaluation results of technical analysis\n%s' + eval_results.to_string())   
    return

def find_sample_index(eval_results, data_point):
    cat_range = 'Neutral'
    if not np.isnan(data_point):
        for cat_range in eval_results:
            cat_max = eval_results.at['Range Max', cat_range]
            cat_min = eval_results.at['Range Min', cat_range]
            if data_point > cat_min and data_point < cat_max:
                break
    return cat_range

def add_results_index(df_results, ndx_label):
    zero_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    result_cols = df_results.columns
    expanded_results = pd.DataFrame(data = zero_data, index=[ndx_label], columns=result_cols)
    return expanded_results

def combinations(f_out, df_data, eval_results):
    result_index = 'combination1 MACD<0.25+Close>SMA20, 10 day'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['MACD_Buy'] == True:
            if not np.isnan(nrow[1]['MACD']) and \
                nrow[1]['MACD'] < 0.25 and \
                nrow[1]['Close'] > nrow[1]['SMA20']:
                cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
                eval_results.at[result_index, cat_str] += 1
    return eval_results

def sample_count(f_out, df_data, eval_results):
    r1_index = 'Baseline 1 day'
    r5_index = 'Baseline 5 day'
    r10_index = 'Baseline 10 day'
    r20_index = 'Baseline 20 day'
    if not r1_index in eval_results.index:
        eval_results = eval_results.append(add_results_index(eval_results, r1_index))
        eval_results = eval_results.append(add_results_index(eval_results, r5_index))
        eval_results = eval_results.append(add_results_index(eval_results, r10_index))
        eval_results = eval_results.append(add_results_index(eval_results, r20_index))
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            eval_results.at[r1_index, find_sample_index(eval_results, nrow[1]['1 day change'])] += 1
            eval_results.at[r5_index, find_sample_index(eval_results, nrow[1]['5 day change'])] += 1
            eval_results.at[r10_index, find_sample_index(eval_results, nrow[1]['10 day change'])] += 1
            eval_results.at[r20_index, find_sample_index(eval_results, nrow[1]['20 day change'])] += 1
    return eval_results

def accumulation_distribution(f_out, df_data, eval_results):
    result_index = 'Accumulation Distribution, TBD'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def aroon_indicator(f_out, df_data, eval_results):
    result_index = 'Aroon Indicator, TBD'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            cat_str = find_sample_index(eval_results, nrow[1]['20 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def average_directional_index(f_out, df_data, eval_results):
    result_index = 'Average Directional Index, TBD'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def stochastic_oscillator(f_out, df_data, eval_results):
    result_index = 'Stochastic Oscillator, TBD'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if True:
            cat_str = find_sample_index(eval_results, nrow[1]['5 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def relative_strength(f_out, df_data, eval_results):
    result_index = 'Relative Strength, RS<RS SMA20, 10 day'
    rs_mean_index = 'Relative Strength, RS<RS mean, 10 day'
    if not result_index in eval_results.index:
        combo_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(combo_results)
        combo_results = add_results_index(eval_results, rs_mean_index)
        eval_results = eval_results.append(combo_results)

    rs_mean = df_data.get('Relative Strength').mean()
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['Relative Strength'] < nrow[1]['RS SMA20']:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
        if nrow[1]['Relative Strength'] < rs_mean:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[rs_mean_index, cat_str] += 1
    return eval_results

def on_balance_volume(f_out, df_data, eval_results):
    result_index = 'OBV, OBV>0, 10 day'
    if not result_index in eval_results.index:
        OBV_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(OBV_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['OBV'] > 0:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def bollinger_bands(f_out, df_data, eval_results):
    result_index = 'Bollinger Bands, Close<SMA20, 10 day'
    if not result_index in eval_results.index:
        BB_results = add_results_index(eval_results, result_index)
        eval_results = eval_results.append(BB_results)
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['Close'] < nrow[1]['SMA20']:
            cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
            eval_results.at[result_index, cat_str] += 1
    return eval_results

def macd_positive_cross(f_out, df_data, eval_results):
    r1_index = 'MACD Positive Cross - 1 day'
    r5_index = 'MACD Positive Cross - 5 day'
    r10_index = 'MACD Positive Cross - 10 day'
    r20_index = 'MACD Positive Cross - 20 day'
    if not r10_index in eval_results.index:
        #MACD_results = add_results_index(eval_results, r10_index)
        eval_results = eval_results.append(add_results_index(eval_results, r1_index))
        eval_results = eval_results.append(add_results_index(eval_results, r5_index))
        eval_results = eval_results.append(add_results_index(eval_results, r10_index))
        eval_results = eval_results.append(add_results_index(eval_results, r20_index))
        
    rows = df_data.iterrows()
    for nrow in rows:
        if nrow[1]['MACD_Buy'] == True:
            if not np.isnan(nrow[1]['MACD']):
                #cat_str = find_sample_index(eval_results, nrow[1]['10 day change'])
                eval_results.at[r1_index, find_sample_index(eval_results, nrow[1]['1 day change'])] += 1
                eval_results.at[r5_index, find_sample_index(eval_results, nrow[1]['5 day change'])] += 1
                eval_results.at[r10_index, find_sample_index(eval_results, nrow[1]['10 day change'])] += 1
                eval_results.at[r20_index, find_sample_index(eval_results, nrow[1]['20 day change'])] += 1
    return eval_results

def evaluate_technical_analysis(f_out, authentication_parameters, analysis_dir):
    logger.info('evaluate_technical_analysis ---->')
    json_authentication = tda_get_authentication_details(authentication_parameters)
    eval_results = pd.DataFrame([[-1000.0, -1.0, -0.5, -0.2, -0.1,  -0.05, -0.01, 0.01, 0.05, 0.1, 0.2, 0.5,    1.0], \
                                 [-   1.0, -0.5, -0.2, -0.1, -0.05, -0.01,  0.01, 0.05, 0.1,  0.2, 0.5, 1.0, 1000.0]], \
                                index=['Range Min', 'Range Max'], \
                                columns=['Neg', '-5', '-4', '-3', '-2', '-1', 'Neutral', '1', '2', '3', '4', '5', 'Pos'])
    for symbol in tda_read_watch_lists(json_authentication):
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            #print("File: %s" % filename)
            df_data = pd.read_csv(filename)
            eval_results = sample_count(f_out, df_data, eval_results)
            eval_results = macd_positive_cross(f_out, df_data, eval_results)
            eval_results = bollinger_bands(f_out, df_data, eval_results)
            eval_results = on_balance_volume(f_out, df_data, eval_results)
            eval_results = relative_strength(f_out, df_data, eval_results)
            eval_results = stochastic_oscillator(f_out, df_data, eval_results)
            eval_results = average_directional_index(f_out, df_data, eval_results)
            eval_results = aroon_indicator(f_out, df_data, eval_results)
            eval_results = accumulation_distribution(f_out, df_data, eval_results)
            eval_results = combinations(f_out, df_data, eval_results)
    present_evaluation(f_out, eval_results)
    logger.info('<---- evaluate_technical_analysis')
    return

if __name__ == '__main__':
    print ("What have I got in my pocket?\n")
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    # Get external initialization details
    app_data = get_ini_data("TDAMERITRADE")
    json_config = read_config_json(app_data['config'])

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
    logger = logging.getLogger('EvaluateTechnicalAnalysis_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Evaluating Technical Analysis accuracy as trading signals')

    evaluate_technical_analysis(f_out, app_data['authentication'], app_data['market_analysis_data'])

    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nWas it the One Ring or a worthless bauble?")

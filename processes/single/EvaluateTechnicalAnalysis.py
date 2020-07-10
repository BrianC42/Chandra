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
    if not np.isnan(data_point):
        for cat_range in eval_results:
            cat_max = eval_results.at['Range Max', cat_range]
            cat_min = eval_results.at['Range Min', cat_range]
            if data_point > cat_min and data_point < cat_max:
                break
    return cat_range

def on_balance_volume(f_out, df_data, eval_results):
    result_index = 'OBV'
    result_cols = eval_results.columns
    if not result_index in eval_results.index:
        zero_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        OBV_results = pd.DataFrame(data = zero_data, index=[result_index], columns=result_cols)
        eval_results = eval_results.append(OBV_results)
    return eval_results

def bollinger_bands(f_out, df_data, eval_results):
    result_index = 'Bollinger Bands'
    result_cols = eval_results.columns
    if not result_index in eval_results.index:
        zero_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        BB_results = pd.DataFrame(data = zero_data, index=[result_index], columns=result_cols)
        eval_results = eval_results.append(BB_results)
    return eval_results

def macd_positive_cross(f_out, df_data, eval_results):
    result_index = 'MACD Positive Cross'
    result_cols = eval_results.columns
    if not result_index in eval_results.index:
        zero_data = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        MACD_results = pd.DataFrame(data=zero_data, index=[result_index], columns=result_cols)
        eval_results = eval_results.append(MACD_results)
    data_rows = df_data.shape[0]
    if data_rows > 0:
        for data_row in df_data.itertuples():
            if data_row[8] == True:
                if not np.isnan(data_row[5]):
                    cat_str = find_sample_index(eval_results, data_row[5])
                    eval_results.at[result_index, cat_str] += 1
    return eval_results

def evaluate_technical_analysis(f_out, authentication_parameters, analysis_dir):
    logger.info('evaluate_technical_analysis ---->')
    json_authentication = tda_get_authentication_details(authentication_parameters)
    eval_results = pd.DataFrame([[-1000.0, -1.0, -0.5, -0.2, -0.1,  -0.05, -0.01, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0], \
                                 [-   1.0, -0.5, -0.2, -0.1, -0.05, -0.01,  0.01, 0.05, 0.1,  0.2, 0.5, 1.0, 1000.0]], \
                                index=['Range Min', 'Range Max'], \
                                columns=['Neg', '-5', '-4', '=3', '-2', '-1', 'Neutral', '1', '2', '3', '4', '5', 'Pos'])
    for symbol in tda_read_watch_lists(json_authentication):
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            #print("File: %s" % filename)
            df_data = pd.read_csv(filename)
            eval_results = macd_positive_cross(f_out, df_data, eval_results)
            eval_results = bollinger_bands(f_out, df_data, eval_results)
            eval_results = on_balance_volume(f_out, df_data, eval_results)
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

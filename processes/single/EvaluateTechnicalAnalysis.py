'''
Created on Jul 9, 2020

@author: Brian
'''
import os
import datetime as dt
import time
import logging
import pandas as pd

import matplotlib

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists

from technical_analysis_utilities import initialize_eval_results
from technical_analysis_utilities import present_evaluation
from technical_analysis_utilities import sample_count
from technical_analysis_utilities import eval_combinations
from macd import eval_macd_positive_cross
from bollinger_bands import eval_bollinger_bands
from on_balance_volume import eval_on_balance_volume
from relative_strength import eval_relative_strength
from stochastic_oscillator import eval_stochastic_oscillator
from average_directional_index import eval_average_directional_index
from accumulation_distribution import eval_accumulation_distribution
from aroon_indicator import eval_aroon_indicator

def evaluate_technical_analysis(authentication_parameters, analysis_dir):
    logger.info('evaluate_technical_analysis ---->')
    json_authentication = tda_get_authentication_details(authentication_parameters)
    eval_results = initialize_eval_results()
    
    for symbol in tda_read_watch_lists(json_authentication):
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            print("File: %s" % filename)
            df_data = pd.read_csv(filename)
            eval_results = sample_count(df_data, eval_results)
            eval_results = eval_macd_positive_cross(df_data, eval_results)
            eval_results = eval_bollinger_bands(df_data, eval_results)
            eval_results = eval_on_balance_volume(df_data, eval_results)
            eval_results = eval_relative_strength(df_data, eval_results)
            eval_results = eval_stochastic_oscillator(df_data, eval_results)
            eval_results = eval_average_directional_index(df_data, eval_results)
            eval_results = eval_aroon_indicator(df_data, eval_results)
            eval_results = eval_accumulation_distribution(df_data, eval_results)
            eval_results = eval_combinations(df_data, eval_results)
    present_evaluation(eval_results)
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

    evaluate_technical_analysis(app_data['authentication'], app_data['market_analysis_data'])

    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nWas it the One Ring or a worthless bauble?")

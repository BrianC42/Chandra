'''
Created on Jan 31, 2018

@author: Brian
'''
import logging
import os
import time

from accumulation_distribution import accumulation_distribution
from aroon_indicator import aroon_indicator
from average_directional_index import average_directional_index
from bollinger_bands import bollinger_bands
from configuration import get_ini_data
from configuration import read_config_json
import datetime as dt
from macd import macd
from on_balance_volume import on_balance_volume
import pandas as pd
from relative_strength import relative_strength
from stochastic_oscillator import stochastic_oscillator
from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists
from tda_api_library import update_tda_eod_data
from tda_derivative_data import add_change_data
from tda_derivative_data import add_trending_data


def technical_analysis(json_config, authentication_parameters, data_dir, analysis_dir):
    logger.info('technical_analysis ---->')
    '''
    Prepare and update technical analysis based on TDA market data
    df_data = accumulation_distribution(df_data[:])
    '''
    json_authentication = tda_get_authentication_details(authentication_parameters)
    for symbol in tda_read_watch_lists(json_authentication):
        filename = data_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            print("File: %s" % filename)
            df_data = pd.read_csv(filename)
            #print("EOD data for %s\n%s" % (filename, df_data))
            df_data = add_trending_data(df_data)
            df_data = add_change_data(df_data)
            df_data = macd(df_data[:], value_label="Close")
            df_data = on_balance_volume(df_data[:], value_label='Close', volume_lable='Volume')
            df_data = bollinger_bands(df_data[:], value_label="Close", sma_interval=20)
            df_data = accumulation_distribution(df_data)
            df_data = aroon_indicator(df_data)
            df_data = average_directional_index(df_data)
            df_data = stochastic_oscillator(df_data)
            df_data = relative_strength(df_data, value_label="Close", relative_to=data_dir + '\\$spx.x.csv')
            df_data.to_csv(analysis_dir + "\\" + symbol + '.csv', index=False)
            #print("Analytic data for %s\n%s" % (filename, df_data))
    logger.info('<---- technical_analysis')
    return

if __name__ == '__main__':
    print ("Affirmative, Dave. I read you\n")
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
        
        # global parameters
        #logging.debug("Global parameters")
    
    except Exception:
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    print ("Logging to", log_file)
    logger = logging.getLogger('chandra_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Updating stock data')

    #update_tda_eod_data(app_data['authentication'])
    technical_analysis(json_config, app_data['authentication'], app_data['eod_data'], app_data['market_analysis_data'])
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")

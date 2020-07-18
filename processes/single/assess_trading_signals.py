'''
Created on Jul 17, 2020

@author: Brian
'''
import os
import datetime as dt
import time
import logging
import pandas as pd

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists

from macd import trade_on_macd

def assess_trading_signals(f_out, json_config, authentication_parameters, analysis_dir):
    logger.info('technical_analysis ---->')
    '''
    Prepare and update technical analysis based on TDA market data
    df_data = accumulation_distribution(df_data[:])
    '''
    json_authentication = tda_get_authentication_details(authentication_parameters)
    for symbol in tda_read_watch_lists(json_authentication):
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            #print("File: %s" % filename)
            df_data = pd.read_csv(filename)
            guidance = df_data = trade_on_macd(symbol, df_data[:])
            if guidance[0]:
                print("Guidance: %s" % guidance)
                f_out.write("%s %s %s %s %s\n" % (guidance[1], guidance[2], guidance[3], guidance[4], guidance[5]))
                
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
    assess_trading_signals(f_out, json_config, app_data['authentication'], app_data['market_analysis_data'])
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")

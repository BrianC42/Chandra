'''
Created on Aug 19, 2020

@author: Brian

Read TD Ameritrade watchlists and save fundamental information of the list items to csv file for later analysis and use
'''
import datetime as dt
import time
import logging
import pandas as pd
from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists
from tda_api_library import tda_search_instruments

def scan_fundamentals(app_data, json_config):
    df_fundamentals = pd.DataFrame()
    tda_auth = read_config_json(json_config['tokenAttributes'])
    symbol_list = tda_read_watch_lists(tda_auth)
    fundamentals_dir = json_config['fundamentals']
    tda_throttle_time = time.time()
    tda_throttle_count = 0
    for symbol in symbol_list:
        if tda_throttle_count < 110:
            tda_throttle_count += 1
        else:
            now = time.time()
            if now - tda_throttle_time < 60:
                time.sleep(now - tda_throttle_time)
            tda_throttle_count = 0
            tda_throttle_time = time.time()
        print(symbol)
        df_symbol, fundamentals_json = tda_search_instruments(app_data['authentication'], symbol)        
        f_json = open(fundamentals_dir + symbol + ".json", 'w')
        f_json.write(fundamentals_json)
        f_json.close()
        df_fundamentals = df_fundamentals.append(df_symbol, ignore_index=True)
    df_fundamentals.to_csv(fundamentals_dir + "watchlist fundamentals.csv", index=False)

    return

if __name__ == '__main__':
    print ("Scanning asset fundamentals\n")
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    # Get external initialization details
    app_data = get_ini_data("TDAMERITRADE")
    json_config = read_config_json(app_data['config'])

    try:    
        log_file = json_config['fundamentalslog']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = json_config['fundamentalsoutputfile']
        output_file = output_file + ' {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
        f_out = open(output_file, 'w')    
        
    except Exception:
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    print ("Logging to", log_file)
    logger = logging.getLogger('ScanningFundamentals_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Scanning TD Ameritrade fundamentals')

    scan_fundamentals(app_data, json_config)

    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nGood information for further analysis")
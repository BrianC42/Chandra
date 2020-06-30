'''
Created on Jan 31, 2018

@author: Brian
'''
import os
import re
import configparser
import datetime as dt
import time
import json
import logging
import quandl

#from quandl_library import update_quandl_eod_data
from tda_api_library import update_tda_eod_data

def get_ini_data(csection):
    config_file = os.getenv('localappdata') + "\\Development\\data.ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    config.sections()
    ini_data = config[csection]
    return ini_data

def get_devdata_dir():
    logging.info('get_devdata_dir')
    devdata = get_ini_data('DEVDATA')
    devdata_dir = devdata['dir']
    return devdata_dir

def get_quandl_key():
    logging.info('get_quandl_key')
    quandl_data = get_ini_data("QUANDL")
    quandl.ApiConfig.api_key = quandl_data['key']
    return (quandl_data['key'])

def get_list_of_tickers():
    logging.info('get_list_of_tickers')
    df_tickers = ["AAPL"]
    return(df_tickers)

def save_list_of_tickers(df_tickers):
    logging.info('save_list_of_tickers')
    output_file = get_devdata_dir() + "\\TickerMetadata.csv"
    logging.debug ("Creating a file containing the list of the tickers processed ...", output_file, df_tickers)
    df_tickers.to_csv(output_file)
    return

def read_config_json(json_file) :
    print ("reading configuration details from ", json_file)
    
    json_f = open(json_file, "rb")
    json_config = json.load(json_f)
    json_f.close
    
    return (json_config)

def update_stock_data():
    logger.debug('update_stock_data ---->')
    
    df_tickers = get_list_of_tickers()
    update_tda_eod_data(df_tickers)
    
    logger.debug('<---- update_stock_data')
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
    tda_api_key = app_data['apikey']

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

    '''
        Update end of day stock prices from Quandl
        Premium service requiring a monthly fee
    
    try:
        quandl_key = get_quandl_key()
        update_stock_data(quandl_key)
    except Exception:
        print("\nUnable to set quandl API key")
    '''
    
    '''
        Update end of day stock prices from TD Ameritrade
    '''
    df_symbols = get_list_of_tickers()
    update_tda_eod_data(df_symbols, tda_api_key)
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")

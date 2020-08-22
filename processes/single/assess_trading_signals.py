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
from tda_api_library import tda_read_option_chain

from macd import trade_on_macd
from bollinger_bands import trade_on_bb
from stochastic_oscillator import trade_on_stochastic_oscillator
from on_balance_volume import trade_on_obv
from relative_strength import trade_on_relative_strength

def covered_call(symbol, df_data, df_options):
    return

def cash_secured_put(symbol, df_data, df_options):
    return 

def assess_options_chanins(symbol, df_data, df_options):
    logger.info('assess_options chains ----> %s' % symbol)
    covered_call(symbol, df_data, df_options)
    cash_secured_put(symbol, df_data, df_options)
    logger.info('<---- assess_options chains')
    return
    
def assess_trading_signals(symbol, df_data):
    logger.info('assess_trading_signals ----> %s' % symbol)
    guidance = pd.DataFrame()
    guidance = trade_on_macd(guidance, symbol, df_data[:])
    guidance = trade_on_bb(guidance, symbol, df_data[:])
    guidance = trade_on_stochastic_oscillator(guidance, symbol, df_data)
    guidance = trade_on_obv(guidance, symbol, df_data)
    guidance = trade_on_relative_strength(guidance, symbol, df_data)

    logger.info('<---- assess_trading_signals')
    return guidance

def search_for_trading_opportunities(f_out, authentication_parameters, analysis_dir):
    logger.info('searching for trading opportunities ---->')
    guidance = pd.DataFrame()
    
    json_authentication = tda_get_authentication_details(authentication_parameters)
    for symbol in tda_read_watch_lists(json_authentication):
        #print("Assessing: %s" % symbol)
        filename = analysis_dir + '\\' + symbol + '.csv'
        if os.path.isfile(filename):
            df_data = pd.read_csv(filename)
            guidance = assess_trading_signals(symbol, df_data)
            for trigger in guidance.itertuples():
                report = '{:s}, {:>8s}, {:s}, {:>8.2f}, {:s}'.format(trigger[4], trigger[2], trigger[3], trigger[6], trigger[5])
                print(report)
                f_out.write(report + "\n")
        df_options, options_json = tda_read_option_chain(app_data['authentication'], symbol)
        assess_options_chanins(symbol, df_data, df_options)
                
    logger.info('<---- searching for trading opportunities done')
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
    search_for_trading_opportunities(f_out, app_data['authentication'], app_data['market_analysis_data'])
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")

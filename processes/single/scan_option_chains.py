'''
Created on Aug 17, 2020

@author: Brian

scan TD Ameritrade watch list items for current option chain details and save to csv files 
'''
import datetime as dt
import time
import logging

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists
from tda_api_library import tda_read_option_chain

def scan_option_chains(app_data, json_config):
    tda_auth = read_config_json(json_config['tokenAttributes'])
    symbol_list = tda_read_watch_lists(tda_auth)
    options_dir = json_config['optionchains']
    for symbol in symbol_list:
        print(symbol)
        df_options, options_json = tda_read_option_chain(app_data['authentication'], symbol)
        df_options.to_csv(options_dir + symbol + ".csv", index=False)
        f_json = open(options_dir + symbol + ".json", 'w')
        f_json.write(options_json)
        f_json.close()

    return

if __name__ == '__main__':
    print ("Scanning future possibilities?\n")
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    # Get external initialization details
    app_data = get_ini_data("TDAMERITRADE")
    json_config = read_config_json(app_data['config'])

    try:    
        log_file = json_config['optionchainlog']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = json_config['optionchainoutputfile']
        output_file = output_file + ' {:4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}'.format(now.year, now.month, now.day, \
                                                                                       now.hour, now.minute, now.second) + '.txt'
        f_out = open(output_file, 'w')    
        
    except Exception:
        print("\nAn exception occurred - log file details are missing from json configuration")
        
    print ("Logging to", log_file)
    logger = logging.getLogger('ScanningOptionChains_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Scanning TD Ameritrade option chains')

    scan_option_chains(app_data, json_config)

    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nCrystal clear or clear as mud?")

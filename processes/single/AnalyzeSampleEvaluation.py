'''
Created on Aug 8, 2020

@author: Brian
'''
import os
import pandas as pd
import datetime as dt
import logging

from configuration import get_ini_data
from configuration import read_config_json

def read_evaluation_result(json_config):
    segmentation = json_config['evaluateoutputFile' ] +'.csv'
    if os.path.isfile(segmentation):
        df_eval = pd.read_csv(segmentation)
    return df_eval

if __name__ == '__main__':
    print ("Welcome back Pythia, I'm ready to get to work\n")
    '''
    Prepare the run time environment
    '''
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
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nI hope that gave you what you need")

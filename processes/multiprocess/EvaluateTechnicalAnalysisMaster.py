'''
Created on Jul 16, 2020

@author: Brian

use child processes to assess each symbol against the technical analysis implemented and save the results in
a csv file for later analysis
'''
import os
from multiprocessing import Process, Pipe
import datetime as dt
import time
import logging

from configuration import get_ini_data
from configuration import read_config_json

from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists

from EvaluateTechnicalAnalysisChild import EvaluateTechnicalAnalysisChild
from technical_analysis_utilities import initialize_eval_results
from technical_analysis_utilities import add_analysis_results
from technical_analysis_utilities import present_evaluation

def coordinate_evaluation_child_processes(authentication_parameters, analysis_dir, json_config):
    core_count = os.cpu_count()
    #print ("Starting {:d} processes:".format(core_count) )
    c_send = []
    c_rcv = []
    data_worker = []
    localDirs = get_ini_data("LOCALDIRS")
    aiwork = localDirs['aiwork']
    segmentation = aiwork + '\\' + json_config['evaluatesegmentationfile']
    
    p_ndx = 0
    while p_ndx < core_count:
        #print ("Starting pipe process", p_ndx)
        p_send, p_receive = Pipe()
        worker = Process(target=EvaluateTechnicalAnalysisChild, args=(p_receive,))
        worker.start()
        c_send.append(p_send)
        c_rcv.append(p_receive)
        data_worker.append(worker)            
        p_ndx += 1

    '''
    Reading historical data and have worker processes perform technical analyses
    '''
    print ("\nBeginning technical analysis ...")
    json_authentication = tda_get_authentication_details(authentication_parameters)
    symbol_list = tda_read_watch_lists(json_authentication)
    eval_results = initialize_eval_results()    
    idx = 0
    while idx < len(symbol_list):   
        '''================= Send slices to workers ================'''
        p_ndx = 0
        p_active = 0
        while p_ndx < core_count and idx < len(symbol_list):
            symbol = symbol_list[idx]
            c_send[p_ndx].send([analysis_dir, symbol, segmentation])
            p_ndx += 1
            idx += 1
        
        '''======== Receive processed slices from workers =========='''
        p_active = p_ndx
        p_ndx = 0
        while p_ndx < p_active:
            symbol_results = c_send[p_ndx].recv()
            eval_results = add_analysis_results(eval_results, symbol_results)
            #print ("Data from technical analysis worker: ...")
            p_ndx += 1
 
    '''================= Clean up worker processes ================='''     
    p_ndx = 0
    while p_ndx < core_count:
        c_send[p_ndx].close()
        c_rcv[p_ndx].close()
        #print ("Pipes to worker {:d} closed".format(p_ndx))
        time.sleep(2)
        if data_worker[p_ndx].is_alive():
            data_worker[p_ndx].terminate()
            #print ("Worker {:d} did not clean up and exit. Terminated".format(p_ndx))
        else:
            pass
            #print ("Worker {:d} is no longer alive".format(p_ndx))
        p_ndx += 1

    #print ("\nAll workers done")
    present_evaluation(eval_results, json_config['evaluateoutputFile'])
    return 

if __name__ == '__main__':
    print ("Affirmative, Dave. I read you\n")
    '''
    Prepare the run time environment
    '''
    now = dt.datetime.now()
    
    # Get external initialization details
    localDirs = get_ini_data("LOCALDIRS")
    aiwork = localDirs['aiwork']
    app_data = get_ini_data("TDAMERITRADE")
    json_config = read_config_json(app_data['config'])

    try:    
        log_file = aiwork + '\\' + json_config['evaluatelogFile']
        if json_config['loggingLevel'] == "debug":
            logging.basicConfig(filename=log_file, level=logging.DEBUG, format=json_config['loggingFormat'])
        elif json_config['loggingLevel'] == "info":
            logging.basicConfig(filename=log_file, level=logging.INFO, format=json_config['loggingFormat'])
        else:
            logging.basicConfig(filename=log_file, level=logging.WARNING, format=json_config['loggingFormat'])
            
        output_file = aiwork + '\\' + json_config['evaluateoutputFile']
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
    logger.info('Updating stock data')

    coordinate_evaluation_child_processes(app_data['authentication'], app_data['market_analysis_data'], json_config)
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")

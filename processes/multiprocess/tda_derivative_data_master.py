'''
Created on Jul 15, 2020

@author: Brian

Process to manage child processes to update market data and technical analysis derivative data. 
Information saved to csv files for use by other processes
'''
from multiprocessing import Process, Pipe
import os
import datetime as dt
import time
import logging

from tda_derivative_data_child import tda_derivative_data_child

from configuration import get_ini_data
from configuration import read_config_json
from tda_api_library import update_tda_eod_data
from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists

def coordinate_child_processes(authentication_parameters, data_dir, analysis_dir):
    core_count = os.cpu_count()
    #print ("Starting {:d} processes:".format(core_count) )
    c_send = []
    c_rcv = []
    data_worker = []
    
    p_ndx = 0
    while p_ndx < core_count:
        #print ("Starting pipe process", p_ndx)
        p_send, p_receive = Pipe()
        worker = Process(target=tda_derivative_data_child, args=(p_receive,))
        worker.start()
        c_send.append(p_send)
        c_rcv.append(p_receive)
        data_worker.append(worker)            
        p_ndx += 1

    '''
    Reading historical data and have worker processes perform technical analyses
    '''
    #print ("\nBeginning technical analysis ...")
    json_authentication = tda_get_authentication_details(authentication_parameters)
    #for symbol in tda_read_watch_lists(json_authentication):
    symbol_list = tda_read_watch_lists(json_authentication)
    
    idx = 0
    while idx < len(symbol_list):   
        '''================= Send slices to workers ================'''
        p_ndx = 0
        p_active = 0
        while p_ndx < core_count and idx < len(symbol_list):
            symbol = symbol_list[idx]
            c_send[p_ndx].send([data_dir, analysis_dir, symbol])
            p_ndx += 1
            idx += 1
        
        '''======== Receive processed slices from workers =========='''
        p_active = p_ndx
        p_ndx = 0
        while p_ndx < p_active:
            data_enh = c_send[p_ndx].recv()        
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
    return    
    
if __name__ == '__main__':
    print ("Affirmative, Dave. I read you\n")
    '''
    Prepare the run time environment
    '''
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
        
    #print ("Logging to", log_file)
    logger = logging.getLogger('chandra_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')
    logger.info('Updating stock data')

    update_tda_eod_data(app_data['authentication'])
    coordinate_child_processes(app_data['authentication'], app_data['eod_data'], app_data['market_analysis_data'])
    
    '''
    clean up and prepare to exit
    '''
    f_out.close()

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")

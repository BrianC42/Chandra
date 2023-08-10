'''
Created on Jul 29, 2023

@author: Brian
'''
from multiprocessing import Process, Pipe
import os
import time

import datetime as dt
from tda_api_library import tda_get_authentication_details
from tda_api_library import tda_read_watch_lists
from tda_api_library import update_tda_eod_data
from tda_derivative_data_child import tda_derivative_data_child


def updateMarketData(authentication_parameters, data_dir, analysis_dir):
    core_count = os.cpu_count()
    # print ("Starting {:d} processes:".format(core_count) )
    c_send = []
    c_rcv = []
    data_worker = []
    
    p_ndx = 0
    while p_ndx < core_count:
        # print ("Starting pipe process", p_ndx)
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
    # print ("\nBeginning technical analysis ...")
    json_authentication = tda_get_authentication_details(authentication_parameters)
    # for symbol in tda_read_watch_lists(json_authentication):
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
            # print ("Data from technical analysis worker: ...")
            p_ndx += 1
 
    '''================= Clean up worker processes ================='''     
    p_ndx = 0
    while p_ndx < core_count:
        c_send[p_ndx].close()
        c_rcv[p_ndx].close()
        # print ("Pipes to worker {:d} closed".format(p_ndx))
        time.sleep(2)
        if data_worker[p_ndx].is_alive():
            data_worker[p_ndx].terminate()
            # print ("Worker {:d} did not clean up and exit. Terminated".format(p_ndx))
        else:
            pass
            # print ("Worker {:d} is no longer alive".format(p_ndx))
        p_ndx += 1

    # print ("\nAll workers done")
    return    

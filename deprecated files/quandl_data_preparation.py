'''
Created on Jan 31, 2018

@author: Brian

*********************************************
    deprecated: based on Quandl data
*********************************************
'''
import sys
sys.path.append("../Technical_Analysis/")

from multiprocessing import Process, Pipe
import os
import datetime
import time
import logging

import pandas as pd

#from test.libregrtest.save_env import multiprocessing
from quandl_worker import quandl_worker_pipe
from quandl_library import save_enhanced_historical_data
from quandl_library import save_enhanced_symbol_data
from quandl_library import read_historical_data
from quandl_library import get_ini_data

def mp_prep_quandl_data():
    lstm_config_data = get_ini_data("LSTM")
    log_file = lstm_config_data['log']
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')
    print ("Logging to", log_file)
    logger = logging.getLogger('lstm_logger')
    log_fmt = logging.Formatter('%(asctime)s - %(name)s - %levelname - %(messages)s')    

    print ("Affirmative, Dave. I read you\n")
    
    '''.......................................................
        pipe processing
        
        1. Start one worker process per core
        2. Break work up into slices
        3. Iterate through slices
            Look for processed results from worker
                Save result
            Look for idle worker
                Send next slice of work
                If all workers busy
                    sleep
        4. Close pipes to workers (signal for them to clean up and exit)
    '''
    t_start = datetime.datetime.now()
    
    output_data = pd.DataFrame()
    core_count = os.cpu_count()
    print ("Starting {:d} processes:".format(core_count) )
    c_send = []
    c_rcv = []
    data_worker = []
    
    p_ndx = 0
    while p_ndx < core_count:
        print ("Starting pipe process", p_ndx)
        p_send, p_receive = Pipe()
        worker = Process(target=quandl_worker_pipe, args=(p_receive,))
        worker.start()
        
        c_send.append(p_send)
        c_rcv.append(p_receive)
        data_worker.append(worker)            
        p_ndx += 1

    '''
        Reading historical data and have worker processes perform technical analyses
    '''
    print ("\nBeginning technical analysis ...")
    df_data = read_historical_data(recs=200000)
    df_tickers = df_data.drop_duplicates("ticker") 

    idx = 0
    indices = df_tickers.index.get_values()

    print ("Read {1} Rows of data and discovered {0} tickers to analyze:\n".format(df_tickers.shape[0], len(df_data)), indices)

    last_slice = False
    while idx < df_tickers.shape[0]:   
        '''================= Send slices to workers ================'''
        p_ndx = 0
        p_active = core_count
        while p_ndx < core_count and last_slice == False:
            if idx < df_tickers.shape[0] - 1:
                slice_start = indices[idx]
                slice_end = indices[idx+1]
                #print ("ticker", tickers.ix[slice_start, 'ticker'], "Not last ticker. Slice:", slice_start, "to", slice_end)
            else:
                slice_start = indices[len(indices)-1]
                slice_end = len(df_data) - 1
                last_slice = True
                p_active = p_ndx
                #print ("ticker", tickers.ix[slice_start, 'ticker'], "Last ticker. Slice", slice_start, "to", slice_end)
            print ("Passing data to worker", p_ndx, slice_start, slice_end)
            c_send[p_ndx].send(df_data[slice_start:slice_end])
            p_ndx += 1
            idx += 1
        
        '''======== Receive processed slices from workers =========='''
        p_ndx = 0
        while p_ndx < core_count and p_ndx <= p_active:
            data_enh = c_send[p_ndx].recv()        
            print ("Data from technical analysis worker: ...", data_enh.ix[0, 'ticker'], " ", len(data_enh), "data points")
            save_enhanced_symbol_data(data_enh.ix[0, 'ticker'], data_enh)
            #print (data_enh.head(2), "\n", data_enh.tail(2))
            output_data = pd.concat([output_data, data_enh])
            p_ndx += 1
 
    '''================= Clean up worker processes ================='''     
    p_ndx = 0
    while p_ndx < core_count:
        c_send[p_ndx].close()
        c_rcv[p_ndx].close()
        print ("Pipes to worker {:d} closed".format(p_ndx))
        time.sleep(2)
        if data_worker[p_ndx].is_alive():
            data_worker[p_ndx].terminate()
            print ("Worker {:d} did not clean up and exit. Terminated".format(p_ndx))
        else:
            print ("Worker {:d} is no longer alive".format(p_ndx))
        p_ndx += 1

    t_end = datetime.datetime.now()
    print ("\nAll workers done", t_start, t_end)
    
    save_enhanced_historical_data(output_data)

    print ("\nDave, this conversation can serve no purpose anymore. Goodbye")
    
if __name__ == '__main__':
    mp_prep_quandl_data()

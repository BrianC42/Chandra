'''
Created on March 23, 2020

@author: Brian
'''
import logging
import datetime as dt
import time
import networkx as nx
import matplotlib.pyplot as plt

from configuration import get_ini_data
from configuration import read_config_json
from assemble_model import build_model
from configuration_graph import build_configuration_graph
from load_prepare_data import load_and_prepare_data

if __name__ == '__main__':
    print ("Good morning Dr. Chandra. I am ready for my next lesson.\n")
    
    '''
    Prepare the run time environment
    '''
    start = time.time()
    now = dt.datetime.now()
    
    # Get external initialization details
    lstm_config_data = get_ini_data("CHANDRA")
    json_config = read_config_json(lstm_config_data['config'])

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
    logger.info('Keras model for stock market analysis and prediction')
    
    nx_graph = nx.MultiDiGraph()
    build_configuration_graph(json_config, nx_graph)
    #nx.draw(nx_graph, arrows=True, with_labels=True, font_weight='bold')
    nx.draw_circular(nx_graph, arrows=True, with_labels=True, font_weight='bold')
    plt.show()
    
    ''' .......... Step 1 - Load and prepare data .........................
    ======================================================================= '''
    step1 = time.time()
    print ('\nStep 1 - Load and prepare the data for analysis')
    load_and_prepare_data(nx_graph)

    ''' ................... Step 2 - Build Model ............................
    ========================================================================= '''
    step2 = time.time()
    print ('\nStep 2 - Build Model')
    build_model(nx_graph)

    ''' ...................... Step 3 - Train the model .....................
    ========================================================================= '''
    step3 = time.time()
    print( "\nStep 3 - Train the model")
    
    ''' .................... Step 4 - Evaluate the model! ...............
    ===================================================================== '''
    step4 = time.time()
    print ("\nStep 4 - Evaluate the model!")
    
    ''' .................... Step 5 - clean up, archive and visualize accuracy! ...............
    =========================================================================================== '''
    step5 = time.time()
    print ("\nStep 5 - clean up, archive and visualize accuracy!")

    end = time.time()
    print ("")
    print ("\tStep 1 took %.1f secs to Load and prepare the data for analysis" % (step2 - step1)) 
    print ("\tStep 2 took %.1f secs to Build Model" % (step3 - step2)) 
    print ("\tStep 3 took %.1f secs to Train the model" % (step4 - step3)) 
    print ("\tStep 4 took %.1f secs to Evaluate the model" % (step5 - step4)) 
    print ("\tStep 5 took %.1f secs to Visualize accuracy, clean up and archive" % (end - step5))    

    '''
    clean up and prepare to exit
    '''
    f_out.close()
    print ('\nNow go and make us rich')

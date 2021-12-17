'''
Created on Jan 15, 2021

@author: Brian
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from TrainingDataAndResults import Data2Results as d2r

from configuration_constants import JSON_NORMALIZE_DATA
'''
deprecated
from configuration_constants import JSON_MODEL_DEPTH
from configuration_constants import JSON_NODE_COUNT
'''
#from configuration_constants import JSON_DROPOUT
#from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_ACTIVATION
from configuration_constants import JSON_MODEL_OUTPUT_ACTIVATION
from configuration_constants import JSON_OPTIMIZER
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_METRICS

def evaluate_and_visualize(d2r):
    
    loss = d2r.model.evaluate(x=d2r.testX, y=d2r.testY)
    print("evaluation loss: %s" % loss)
    
    x_min = np.min(d2r.testX)
    x_max = np.max(d2r.testX)
    
    fit_params = d2r.fitting.params
    epoch_cnt = fit_params['epochs']
    steps = fit_params['steps']
    #batch_size = fit_params['batch_size']
    #samples = fit_params['samples']
    #metrics = fit_params['metrics']
    
    '''
    deprecated
    nx_normalize = nx.get_node_attributes(d2r.graph, JSON_NORMALIZE_DATA)[d2r.mlNode]
    nx_model_depth = nx.get_node_attributes(d2r.graph, JSON_MODEL_DEPTH)[d2r.mlNode]
    nx_node_count = nx.get_node_attributes(d2r.graph, JSON_NODE_COUNT)[d2r.mlNode]
    nx_dropout = nx.get_node_attributes(d2r.graph, JSON_DROPOUT)[d2r.mlNode]
    nx_dropout_rate = nx.get_node_attributes(d2r.graph, JSON_DROPOUT_RATE)[d2r.mlNode]
    str_l3 = '\nHidden layers: {:.0f} Nodes: {:.0f}'.format(nx_model_depth, nx_node_count)
    nx_activation = nx.get_node_attributes(d2r.graph, JSON_ACTIVATION)[d2r.mlNode]
    nx_output_activation = nx.get_node_attributes(d2r.graph, JSON_MODEL_OUTPUT_ACTIVATION)[d2r.mlNode]
    '''
    nx_optimizer = nx.get_node_attributes(d2r.graph, JSON_OPTIMIZER)[d2r.mlNode]
    nx_loss = nx.get_node_attributes(d2r.graph, JSON_LOSS)[d2r.mlNode]
    nx_metrics = nx.get_node_attributes(d2r.graph, JSON_METRICS)[d2r.mlNode]
    
    str_l1 = 'Model structure'
    '''
    if nx_normalize:
        str_tf = 'True'
    else:
        str_tf = 'False'
    str_l2 = '\nData normalized: ' + str_tf
    str_l4 = '\nDropout {:.0f} Rate: {:.0f}'.format(nx_dropout, nx_dropout_rate)
    str_l5 = '\nActivations: Hidden layers: {:s}, Output: {:s}'.format(nx_activation, nx_output_activation)
    '''
    str_l7 = '\nOptimizer: {:s}'.format(nx_optimizer)
    str_structure = str_l1 + str_l7
    
    str_p0 = '\nFitting Parameters'
    #str_p1 = '\nepochs: {:.0f} batch size: {:.0f}'.format(epoch_cnt, batch_size)
    #str_p3 = '\nsamples {:.0f}, steps: {:.0f}'.format(samples, steps)
    str_p4 = '\nloss: {:s}'.format(nx_loss)
    str_p5 = '\nmetrics: {:s}'.format(nx_metrics[0])
    #str_params = str_p0 + str_p1 + str_p3 + str_p4 + str_p5
    str_params = str_p0 + str_p4 + str_p5
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')

    axs[0, 0].set_title("ML Parameters Used")
    axs[0, 0].text(0.5, 0.5, str_structure + '\n' + str_params, horizontalalignment='center', verticalalignment='center', wrap=True)
        
    axs[0, 1].set_title("Raw Data")
    axs[0, 1].scatter(d2r.testX, d2r.testY)
    axs[0, 1].set_xlabel("Feature")
    axs[0, 1].set_ylabel("Target")

    axs[0, 2].set_title("Training Data")
    axs[0, 2].scatter(d2r.testX, d2r.testY)
    axs[0, 2].set_xlabel("Feature")
    axs[0, 2].set_ylabel("Target")
    
    axs[1, 0].set_title("Fitting history")
    axs[1, 0].plot(d2r.fitting.epoch, d2r.fitting.history['loss'], label='Training loss')
    axs[1, 0].plot(d2r.fitting.epoch, d2r.fitting.history['val_loss'], label='Validation loss')
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].legend()
        
    axs[1, 1].set_title("Prediction")
    iterable = ((x_min + (((x_max - x_min) / 100) * x)) for x in range(100))
    x_predict = np.fromiter(iterable, float)
    y_predict = d2r.model.predict(x=x_predict)
    axs[1, 1].scatter(x_predict, y_predict)


    axs[1, 1].set_xlabel("Feature")
    axs[1, 1].set_ylabel("Target")
    
    axs[1, 2].set_title("Extrapolation")
    iterable = ((x_max + (((x_max - x_min) / 10) * x)) for x in range(100))
    x_predict = np.fromiter(iterable, float)
    y_predict = d2r.model.predict(x=x_predict)
    axs[1, 2].scatter(x_predict, y_predict)
    axs[1, 2].set_xlabel("Feature")
    axs[1, 2].set_ylabel("Target")
    
    plt.tight_layout()
    plt.show()
    
    return
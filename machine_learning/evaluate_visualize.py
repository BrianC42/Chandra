'''
Created on Jan 15, 2021

@author: Brian
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from configuration_constants import JSON_NORMALIZE_DATA
from configuration_constants import JSON_MODEL_DEPTH
from configuration_constants import JSON_NODE_COUNT
from configuration_constants import JSON_DROPOUT
from configuration_constants import JSON_DROPOUT_RATE
from configuration_constants import JSON_ACTIVATION

def evaluate_and_visualize(nx_graph, nx_node, k_model, x_features, y_targets, x_test, y_test, fitting):
    
    loss = k_model.evaluate(x=x_test, y=y_test)
    print("evaluation loss: %s" % loss)
    
    x_min = np.min(x_test)
    x_max = np.max(x_test)
    
    fit_params = fitting.params
    epoch_cnt = fit_params['epochs']
    batch_size = fit_params['batch_size']
    steps = fit_params['steps']
    samples = fit_params['samples']
    metrics = fit_params['metrics']
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(nx_node, fontsize=14, fontweight='bold')

    axs[0, 0].set_title("ML Parameters Used")
    nx_normalize = nx.get_node_attributes(nx_graph, JSON_NORMALIZE_DATA)[nx_node]
    nx_model_depth = nx.get_node_attributes(nx_graph, JSON_MODEL_DEPTH)[nx_node]
    nx_node_count = nx.get_node_attributes(nx_graph, JSON_NODE_COUNT)[nx_node]
    nx_dropout = nx.get_node_attributes(nx_graph, JSON_DROPOUT)[nx_node]
    nx_dropout_rate = nx.get_node_attributes(nx_graph, JSON_DROPOUT_RATE)[nx_node]
    nx_activation = nx.get_node_attributes(nx_graph, JSON_ACTIVATION)[nx_node]
    str_struct = 'Model structure\nData normalized: {:.0f}\nDepth: {:.0f} Width: {:.0f}\nDropout {:.0f} Rate: {:.0f}\nActivation: {:s}'.format(nx_normalize, nx_model_depth, nx_node_count, \
                                            nx_dropout, nx_dropout_rate, nx_activation)
    str_params = 'Fitting Parameters\nepochs: {:.0f} batch size: {:.0f}\nsteps: {:.0f}\nsamples {:.0f}\nloss: {:s}'.format(epoch_cnt, batch_size, steps, samples, metrics[1])
    axs[0, 0].text(0.5, 0.5, str_struct + '\n' + str_params, horizontalalignment='center', verticalalignment='center', wrap=True)
        
    axs[0, 1].set_title("Raw Data")
    axs[0, 1].scatter(x_features, y_targets)
    axs[0, 1].set_xlabel("Feature")
    axs[0, 1].set_ylabel("Target")

    axs[0, 2].set_title("Training Data")
    axs[0, 2].scatter(x_test, y_test)
    axs[0, 2].set_xlabel("Feature")
    axs[0, 2].set_ylabel("Target")
    
    axs[1, 0].set_title("Fitting history")
    axs[1, 0].scatter(fitting.epoch, fitting.history['loss'])
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("loss")
        
    axs[1, 1].set_title("Prediction")
    iterable = ((x_min + (((x_max - x_min) / 100) * x)) for x in range(100))
    x_predict = np.fromiter(iterable, float)
    y_predict = k_model.predict(x=x_predict)
    axs[1, 1].scatter(x_predict, y_predict)


    axs[1, 1].set_xlabel("Feature")
    axs[1, 1].set_ylabel("Target")
    
    axs[1, 2].set_title("Extrapolation")
    iterable = ((x_max + (((x_max - x_min) / 10) * x)) for x in range(100))
    x_predict = np.fromiter(iterable, float)
    y_predict = k_model.predict(x=x_predict)
    axs[1, 2].scatter(x_predict, y_predict)
    axs[1, 2].set_xlabel("Feature")
    axs[1, 2].set_ylabel("Target")
    
    plt.tight_layout()
    plt.show()
    
    return
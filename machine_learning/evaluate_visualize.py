'''
Created on Jan 15, 2021

@author: Brian
'''
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from configuration_constants import INPUT_LAYERTYPE_DENSE
from configuration_constants import INPUT_LAYERTYPE_RNN
from configuration_constants import INPUT_LAYERTYPE_CNN
from configuration_constants import JSON_OPTIMIZER
from configuration_constants import JSON_LOSS
from configuration_constants import JSON_METRICS

def visualize_fit(d2r, plot_on_axis):
    plot_on_axis.set_title("Fitting history")
    plot_on_axis.plot(d2r.fitting.epoch, d2r.fitting.history['loss'], label='Training loss')
    plot_on_axis.plot(d2r.fitting.epoch, d2r.fitting.history['val_loss'], label='Validation loss')
    plot_on_axis.set_xlabel("Epochs")
    plot_on_axis.set_ylabel("loss")
    plot_on_axis.legend()
        
    return
    
def visualize_parameters(d2r, plot_on_axis):
    nx_optimizer = nx.get_node_attributes(d2r.graph, JSON_OPTIMIZER)[d2r.mlNode]
    nx_loss = nx.get_node_attributes(d2r.graph, JSON_LOSS)[d2r.mlNode]
    nx_metrics = nx.get_node_attributes(d2r.graph, JSON_METRICS)[d2r.mlNode]
    
    str_l1 = 'Model parameters'
    str_l7 = '\nOptimizer: {:s}'.format(nx_optimizer)
    str_model = str_l1 + str_l7
    
    str_p0 = '\nFitting Parameters'
    str_p4 = '\nloss: {:s}'.format(nx_loss)
    str_p5 = '\nmetrics: {:s}'.format(nx_metrics[0])
    str_p6 = '\nepochs: {:d}'.format(d2r.fitting.params['epochs'])
    str_p7 = '\nsteps: {:d}'.format(d2r.fitting.params['steps'])
    str_params = str_p0 + str_p4 + str_p5 + str_p6 + str_p7
    
    plot_on_axis.set_title("ML Parameters Used")
    plot_on_axis.text(0.5, 0.5, str_model + '\n' + str_params, horizontalalignment='center', \
                      verticalalignment='center', wrap=True)

    return

def visualize_dense(d2r):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    visualize_parameters(d2r, axs[0, 0])
    visualize_fit(d2r, axs[1, 0])
    
    axs[0, 1].set_title("Training Data")
    axs[0, 1].scatter(d2r.trainX, d2r.trainY)
        
    axs[0, 2].set_title("Testing Data")
    axs[0, 2].set_xlabel("Feature")
    axs[0, 2].set_ylabel("Target")
    axs[0, 2].scatter(d2r.testX, d2r.testY)
    
    axs[1, 1].set_title("Prediction")
    axs[1, 1].set_xlabel("Feature")
    axs[1, 1].set_ylabel("Target")
    prediction = d2r.model.predict(x=d2r.testX)
    axs[1, 1].scatter(d2r.testX, prediction, label='Prediction', linestyle='dashed')
    axs[1, 1].scatter(d2r.testX, d2r.testY, label='Test data')
    axs[1, 1].legend()

    axs[1, 2].set_title("Extrapolation")
    x_min = np.min(d2r.testX)
    x_max = np.max(d2r.testX)
    iterable = ((x_max + (((x_max - x_min) / 10) * x)) for x in range(100))
    x_predict = np.fromiter(iterable, float)
    y_predict = d2r.model.predict(x=x_predict)
    axs[1, 2].scatter(x_predict, y_predict)
    axs[1, 2].set_xlabel("Feature")
    axs[1, 2].set_ylabel("Target")

    plt.tight_layout()
    plt.show()
    return 

def visualize_rnn(d2r):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    visualize_parameters(d2r, axs[0, 0])
    visualize_fit(d2r, axs[1, 0])
    
    axs[0, 1].set_title("Future use - outliers?")
    axs[0, 1].set_xlabel("time periods")
    axs[0, 1].set_ylabel("Data Value")
    
    axs[1, 1].set_title("Data series vs. Predictions")
    axs[1, 1].set_xlabel("time periods")
    axs[1, 1].set_ylabel("Data Value")
    prediction = d2r.model.predict(d2r.testX)
    axs[1, 1].plot(range(len(prediction)), d2r.testY, label='Test series')
    axs[1, 1].plot(range(len(prediction)), prediction[:, 0], linestyle='dashed', label='Prediction')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
    return 

def visualize_cnn(d2r):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')
    visualize_fit(d2r, axs[0, 0])
    
    plt.tight_layout()
    plt.show()
    return 

def evaluate_and_visualize(d2r):
    d2r.evaluation = d2r.model.evaluate(x=d2r.testX, y=d2r.testY)
    
    if d2r.modelType == INPUT_LAYERTYPE_DENSE:
        visualize_dense(d2r)
    elif d2r.modelType == INPUT_LAYERTYPE_RNN:
        visualize_rnn(d2r)
    elif d2r.modelType == INPUT_LAYERTYPE_CNN:
        visualize_cnn(d2r)
        
    return
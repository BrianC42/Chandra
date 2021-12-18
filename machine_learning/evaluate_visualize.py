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

def evaluate_and_visualize(d2r):
    
    loss = d2r.model.evaluate(x=d2r.testX, y=d2r.testY)
    
    x_min = np.min(d2r.testX)
    x_max = np.max(d2r.testX)
    
    fit_params = d2r.fitting.params
    epoch_cnt = fit_params['epochs']
    steps = fit_params['steps']
    nx_optimizer = nx.get_node_attributes(d2r.graph, JSON_OPTIMIZER)[d2r.mlNode]
    nx_loss = nx.get_node_attributes(d2r.graph, JSON_LOSS)[d2r.mlNode]
    nx_metrics = nx.get_node_attributes(d2r.graph, JSON_METRICS)[d2r.mlNode]
    
    str_l1 = 'Model structure'
    str_l7 = '\nOptimizer: {:s}'.format(nx_optimizer)
    str_structure = str_l1 + str_l7
    
    str_p0 = '\nFitting Parameters'
    str_p4 = '\nloss: {:s}'.format(nx_loss)
    str_p5 = '\nmetrics: {:s}'.format(nx_metrics[0])
    str_params = str_p0 + str_p4 + str_p5
    
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(d2r.mlNode, fontsize=14, fontweight='bold')

    axs[0, 0].set_title("ML Parameters Used")
    axs[0, 0].text(0.5, 0.5, str_structure + '\n' + str_params, horizontalalignment='center', verticalalignment='center', wrap=True)
        
    axs[0, 1].set_title("Training Data")
    if d2r.modelType == INPUT_LAYERTYPE_DENSE:
        axs[0, 1].scatter(d2r.trainX, d2r.trainY)
    elif d2r.modelType == INPUT_LAYERTYPE_RNN:
        axs[0, 1].set_xlabel("time periods")
        axs[0, 1].set_ylabel("Data Value")
        #axs[0, 1].plot(d2r.data[:, 0], d2r.data[:, 0])
    elif d2r.modelType == INPUT_LAYERTYPE_CNN:
        pass
    
    axs[0, 2].set_title("Testing Data")
    if d2r.modelType == INPUT_LAYERTYPE_DENSE:
        axs[0, 2].set_xlabel("Feature")
        axs[0, 2].set_ylabel("Target")
        axs[0, 2].scatter(d2r.testX, d2r.testY)
    elif d2r.modelType == INPUT_LAYERTYPE_RNN:
        '''
        axs[0, 2].set_xlabel("Feature")
        axs[0, 2].set_ylabel("Target")
        axs[0, 2].plot(d2r.testX[:, 0], d2r.testY)
        '''
    elif d2r.modelType == INPUT_LAYERTYPE_CNN:
        pass

    axs[1, 0].set_title("Fitting history")
    axs[1, 0].plot(d2r.fitting.epoch, d2r.fitting.history['loss'], label='Training loss')
    axs[1, 0].plot(d2r.fitting.epoch, d2r.fitting.history['val_loss'], label='Validation loss')
    axs[1, 0].set_xlabel("Epochs")
    axs[1, 0].set_ylabel("loss")
    axs[1, 0].legend()
        
    axs[1, 1].set_title("Prediction")
    if d2r.modelType == INPUT_LAYERTYPE_DENSE:
        axs[1, 1].set_xlabel("Feature")
        axs[1, 1].set_ylabel("Target")
        iterable = ((x_min + (((x_max - x_min) / 100) * x)) for x in range(100))
        x_predict = np.fromiter(iterable, float)
        y_predict = d2r.model.predict(x=x_predict)
        axs[1, 1].scatter(x_predict, y_predict)
    elif d2r.modelType == INPUT_LAYERTYPE_RNN:
        axs[1, 1].set_xlabel("time periods")
        axs[1, 1].set_ylabel("Data Value")
        prediction = d2r.model.predict(d2r.testX)
        axs[1, 1].plot(range(len(prediction)), d2r.testY)
        axs[1, 1].plot(range(len(prediction)), prediction[:, 0], linestyle='dashed')
    elif d2r.modelType == INPUT_LAYERTYPE_CNN:
        pass

    
    axs[1, 2].set_title("Extrapolation")
    if d2r.modelType == INPUT_LAYERTYPE_DENSE:
        iterable = ((x_max + (((x_max - x_min) / 10) * x)) for x in range(100))
        x_predict = np.fromiter(iterable, float)
        y_predict = d2r.model.predict(x=x_predict)
        axs[1, 2].scatter(x_predict, y_predict)
    elif d2r.modelType == INPUT_LAYERTYPE_RNN:
        pass
    elif d2r.modelType == INPUT_LAYERTYPE_CNN:
        pass
    axs[1, 2].set_xlabel("Feature")
    axs[1, 2].set_ylabel("Target")
    
    plt.tight_layout()
    plt.show()
    
    return
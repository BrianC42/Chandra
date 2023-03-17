'''
Created on Oct 9, 2020

@author: Brian
'''
import sys
import datetime as dt
import logging
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow import keras
''' autokeras disabled temporarily
import autokeras as ak
'''

from configuration_constants import JSON_TRAIN
from configuration_constants import JSON_BATCH
from configuration_constants import JSON_EPOCHS
from configuration_constants import JSON_VERBOSE
from configuration_constants import JSON_SHUFFLE_DATA
from configuration_constants import JSON_MODEL_FILE_DIR
from configuration_constants import JSON_ITERATION_ID

from TrainingDataAndResults import TRAINING_AUTO_KERAS

def trainModel(d2r):
    try:
        now = dt.datetime.now()
        timeStamp = ' {:4d}{:0>2d}{:0>2d} {:0>2d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day, \
                                                                        now.hour, now.minute, now.second)

        print("\nTraining shapes x:%s y:%s" % (d2r.trainX.shape, d2r.trainY.shape))
        print("validating shapes x:%s y:%s" % (d2r.validateX.shape, d2r.validateY.shape))
        print("Testing shapes x:%s y:%s" % (d2r.testX.shape, d2r.testY.shape))
        
        nx_modelIterations = nx.get_node_attributes(d2r.graph, "training iterations")[d2r.mlNode]
        iterVariables = nx_modelIterations[d2r.trainingIteration]
        iterParamters = iterVariables["iteration parameters"]

        modeFileDir = iterParamters[JSON_MODEL_FILE_DIR]
        iterationID = iterParamters[JSON_ITERATION_ID]
        
        iterTraining = iterParamters["training"]
        nx_batch = iterTraining[JSON_BATCH]
        nx_epochs = iterTraining[JSON_EPOCHS]
        nx_verbose = iterTraining[JSON_VERBOSE]
        nx_shuffle = iterTraining[JSON_SHUFFLE_DATA]
        
        iterTensorboard = iterParamters["tensorboard"]
        logDir = iterTensorboard["log file dir"]
        logFile = logDir + iterationID + timeStamp
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logFile, \
                                                              histogram_freq=1, \
                                                              write_graph=True, \
                                                              write_images=False, \
                                                              write_steps_per_second=False, \
                                                              profile_batch=0, \
                                                              embeddings_freq=0, \
                                                              embeddings_metadata=None)

        d2r.fitting = d2r.model.fit(x=d2r.trainX, y=d2r.trainY, \
                                    validation_data=(d2r.validateX, d2r.validateY), \
                                    epochs=nx_epochs, \
                                    batch_size=nx_batch, \
                                    shuffle=nx_shuffle, \
                                    verbose=nx_verbose, \
                                    callbacks=[tensorboard_callback])
    
        modelFileName = modeFileDir + iterationID + timeStamp
        
        if d2r.trainer == TRAINING_AUTO_KERAS:
            d2r.model = d2r.model.export_model()
        d2r.model.save(modelFileName)

        keras.utils.plot_model(d2r.model, to_file=modelFileName + '.png', show_shapes=True)

    except Exception:
        err_txt = "\n*** An exception occurred training the model ***\n\t"
        
        exc_info = sys.exc_info()
        exc_str = exc_info[1].args[0]
        if isinstance(exc_str, str):
            exc_txt = err_txt + "\n\t" + exc_str
        elif isinstance(exc_str, tuple):
            exc_txt = err_txt + "\n\t"
            for s in exc_str:
                exc_txt += " "
                exc_txt += s

        logging.debug(exc_txt)
        sys.exit(exc_txt)        
    
    return

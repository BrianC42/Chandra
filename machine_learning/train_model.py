'''
Created on Oct 9, 2020

@author: Brian
'''
import sys
import datetime as dt
import logging
import pickle
import networkx as nx
import tensorflow as tf
from tensorflow import keras

''' autokeras disabled temporarily
import autokeras as ak
'''

import configuration_constants as cc

from TrainingDataAndResults import TRAINING_AUTO_KERAS

def trainModel(d2r):
    try:
        now = dt.datetime.now()
        timeStamp = ' {:4d}{:0>2d}{:0>2d} {:0>2d}{:0>2d}{:0>2d}'.format(now.year, now.month, now.day, \
                                                                        now.hour, now.minute, now.second)

        nx_modelIterations = nx.get_node_attributes(d2r.graph, cc.JSON_TRAINING_ITERATIONS)[d2r.mlNode]
        iterVariables = nx_modelIterations[d2r.trainingIteration]
        iterParamters = iterVariables[cc.JSON_ITERATION_PARAMETERS]

        modeFileDir = iterParamters[cc.JSON_MODEL_FILE_DIR]
        iterationID = iterParamters[cc.JSON_ITERATION_ID]
        
        print("\n==========================\n\tTraining iteration: {}\n==========================".format(iterationID))
        print("\nTraining shapes x:{} y:{}".format(d2r.trainX.shape, d2r.trainY.shape))
        print("validating shapes x:{} y:{}".format(d2r.validateX.shape, d2r.validateY.shape))
        print("Testing shapes x:{} y:{}\n".format(d2r.testX.shape, d2r.testY.shape))
        
        iterTraining = iterParamters[cc.JSON_TRAINING]
        nx_batch = iterTraining[cc.JSON_BATCH]
        nx_epochs = iterTraining[cc.JSON_EPOCHS]
        nx_verbose = iterTraining[cc.JSON_VERBOSE]
        nx_shuffle = iterTraining[cc.JSON_SHUFFLE_DATA]
        
        iterTensorboard = iterParamters[cc.JSON_TENSORBOARD]
        logDir = iterTensorboard[cc.JSON_LOG_DIR]
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
        
        if hasattr(d2r, 'scaler'):
            scalerFile = modeFileDir + iterationID + timeStamp + cc.JSON_SCALER_ID + '.pkl'
            with open(scalerFile, 'wb') as pf:
                pickle.dump(d2r.scaler, pf)
            pf.close()
        else:
            print("No scaler to save")

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

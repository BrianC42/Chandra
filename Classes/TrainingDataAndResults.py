'''
Created on Nov 15, 2021

@author: Brian

Git access error problem
'''
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

TRAINING_TENSORFLOW = 'Tensorflow'
TRAINING_AUTO_KERAS = 'AutoKeras'

MODEL_TYPE = 'modelType'
INPUT_LAYERTYPE_DENSE = 'dense'
INPUT_LAYERTYPE_RNN = 'rnn'
INPUT_LAYERTYPE_CNN = 'cnn'

class Data2Results():
    '''
    Class to prepare training data, store training results and plot results
    
        
    Functions
        plotGraph
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.dataDict = dict()
        self.normDataDict = dict()
        self.maxDictLength = 0
        self.minDictLength = 99999999
        
    @property
    def dataDict(self):
        return self._dataDict
    
    @dataDict.setter
    def dataDict(self, dataDict):
        self._dataDict = dataDict
    
    @property
    def normDataDict(self):
        return self._normDataDict
    
    @normDataDict.setter
    def normDataDict(self, normDataDict):
        self._normDataDict = normDataDict
    
    '''
    definition graph properties
        graph - graph of nodes and edges defining the information flow and processing
        mlNode - the node in the graph containing the machine learning definition
    '''        
    @property
    def graph(self):
        return self._graph
    
    @graph.setter
    def graph(self, graph):
        self._graph = graph
    
    @property
    def mlNode(self):
        return self._mlNode
    
    @mlNode.setter
    def mlNode(self, value):
        self._mlNode = value
    
    @property
    def mlEdgeIn(self):
        return self._mlEdgeIn
    
    @mlEdgeIn.setter
    def mlEdgeIn(self, value):
        self._mlEdgeIn = value
    
    '''
    machine learning properties
        model - the compiled Keras / Tensorflow model
        fitting - Tensorflow fit results
    '''
    @property
    def trainer(self):
        return self._trainer
    
    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer
        
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model
    
    @property
    def fitting(self):
        return self._fitting
    
    @fitting.setter
    def fitting(self, fitting):
        self._fitting = fitting
    
    @property
    def evaluation(self):
        return self._evaluation
    
    @evaluation.setter
    def evaluation(self, evaluation):
        self._evaluation = evaluation
    
    @property
    def modelType(self):
        return self._modelType
    
    @modelType.setter
    def modelType(self, modelType):
        self._modelType = modelType
    
    @property
    def categorizationRegression(self):
        return self._categorizationRegression
    
    @categorizationRegression.setter
    def categorizationRegression(self, categorizationRegression):
        self._categorizationRegression = categorizationRegression
    
    @property
    def categories(self):
        return self._categories
    
    @property
    def categorieCounts(self):
        return self._counts

    def determineCategories(self):
        dfLabels = self._data[self._rawTargets]
        npLabels = np.array(dfLabels)
        self._categories, self._counts = np.unique(npLabels, return_counts=True)
    
    '''
    training data properties
        rawData - source data for samples
        data - Source data samples, modified by preparation step and used to train, evaluate and test
        trainX - sample data used to train the model. Number of samples = trainLen
        trainY - values matching the trainX samples
        testX -  sample data used to test the model. Number of samples = testLen
        testY -  values matching the testX samples
        validateX -  sample data used to validateuate the model. Number of samples = validateLen
        validateY -  values matching the validateX samples
        sequenceIDCol
        rawFeatures - list of features
        rawTargets - List of targets
        preparedFeatures - modified list of data elements (columns) to be input to the model
        preparedTargets - modified list of data elements (columns) to be input to the model for identification or prediction
    '''
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data
        
    @property
    def rawData(self):
        return self._rawData
    
    @rawData.setter
    def rawData(self, rawData):
        self._rawData = rawData
        
    @property
    def trainX(self):
        return self._trainX
    
    @trainX.setter
    def trainX(self, trainX):
        self._trainX = trainX
        
    @property
    def trainY(self):
        return self._trainY
    
    @trainY.setter
    def trainY(self, trainY):
        self._trainY = trainY
        
    @property
    def trainLen(self):
        return self._trainLen
    
    @trainLen.setter
    def trainLen(self, trainLen):
        self._trainLen = trainLen
        
    @property
    def testX(self):
        return self._testX
    
    @testX.setter
    def testX(self, testX):
        self._testX = testX
        
    @property
    def testY(self):
        return self._testY
    
    @testY.setter
    def testY(self, testY):
        self._testY = testY
        
    @property
    def testLen(self):
        return self._testLen
    
    @testLen.setter
    def testLen(self, testLen):
        self._testLen = testLen
        
    @property
    def validateX(self):
        return self._validateX
    
    @validateX.setter
    def validateX(self, validateX):
        self._validateX = validateX
        
    @property
    def validateY(self):
        return self._validateY
    
    @validateY.setter
    def validateY(self, validateY):
        self._validateY = validateY
        
    @property
    def validateLen(self):
        return self._validateLen
    
    @validateLen.setter
    def validateLen(self, validateLen):
        self._validateLen = validateLen
        
    @property
    def timeSeries(self):
        return self._timeSeries
    
    @timeSeries.setter
    def timeSeries(self, timeSeries):
        self._timeSeries = timeSeries

    @property
    def sequenceIDCol(self):
        return self._sequenceIDCol
    
    @sequenceIDCol.setter
    def sequenceIDCol(self, sequenceIDCol):
        self._sequenceIDCol = sequenceIDCol

    @property
    def rawFeatures(self):
        return self._rawFeatures
    
    @rawFeatures.setter
    def rawFeatures(self, rawFeatures):
        self._rawFeatures = rawFeatures

    @property
    def rawTargets(self):
        return self._rawTargets
    
    @rawTargets.setter
    def rawTargets(self, rawTargets):
        self._rawTargets = rawTargets

    @property
    def dataSeriesIDFields(self):
        return self._dataSeriesIDFields
    
    @dataSeriesIDFields.setter
    def dataSeriesIDFields(self, dataSeriesIDFields):
        self._dataSeriesIDFields = dataSeriesIDFields

    @property
    def seriesDataType(self):
        return self._seriesDataType
    
    @seriesDataType.setter
    def seriesDataType(self, seriesDataType):
        self._seriesDataType = seriesDataType

    @property
    def preparedFeatures(self):
        return self._preparedFeatures
    
    @preparedFeatures.setter
    def preparedFeatures(self, preparedFeatures):
        self._preparedFeatures = preparedFeatures

    @property
    def preparedTargets(self):
        return self._preparedTargets
    
    @preparedTargets.setter
    def preparedTargets(self, preparedTargets):
        self._preparedTargets = preparedTargets

    '''
    data preparation properties
    '''
    @property
    def scaler(self):
        return self._scaler
    
    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler
        
    @property
    def normalized(self):
        return self._normalized
    
    @normalized.setter
    def normalized(self, normalized):
        self._normalized = normalized

    @property
    def batches(self):
        return self._batches
    
    @batches.setter
    def batches(self, batches):
        self._batches = batches
             
    @property
    def timesteps(self):
        return self._timesteps
    
    @timesteps.setter
    def timesteps(self, timesteps):
        self._timesteps = timesteps
             
    @property
    def feature_count(self):
        return self._feature_count
    
    @feature_count.setter
    def feature_count(self, feature_count):
        self._feature_count = feature_count

    @property
    def filter_count(self):
        return self._filter_count
    
    @filter_count.setter
    def filter_count(self, filter_count):
        self._filter_count = filter_count

    @property
    def filter_size(self):
        return self._filter_size
    
    @filter_size.setter
    def filter_size(self, filter_size):
        self._filter_size = filter_size

    '''
    Functions ...
    '''
    def archiveData(self, location):
        print("\nData \n%s\nwritten to %s for training\n" % (self.data.describe().transpose(), location))
        self.data.to_csv(location, index=False)
        #"d:\\brian\\AI-Projects\\internal_flows\\TDAPrepared.csv"
          
    def plotGraph(self):
        nx.draw_circular(self.graph, arrows=True, with_labels=True, font_weight='bold')
        plt.show()        
        
    def visualize_categorization_samples(self):
        TRAIN_NDX = 0
        VALIDATE_NDX = 1
        TEST_NDX = 2
        NUM_SAMPLE_SETS = 3
        trainCategories, trainIndices, trainCounts = np.unique(self.trainY, return_index=True, return_counts=True)
        validateCategories, validateIndices, validateCounts = np.unique(self.validateY, return_index=True, return_counts=True)
        testCategories, testIndices, testCounts = np.unique(self.testY, return_index=True, return_counts=True)
    
        categories = len(trainCategories)
        fig, axs = plt.subplots(categories, NUM_SAMPLE_SETS)
        fig.suptitle("CNN Test Results for - " + self.mlNode, fontsize=14, fontweight='bold')
        
        for ndx in range (0, categories):
            featureAxis = axs[ndx, TRAIN_NDX]
            featureAxis.set_title("Feature Training Data series - category %s" % trainCategories[ndx])
            featureAxis.set_xlabel("time periods")
            featureAxis.set_ylabel("Feature Values")
            lines = []
            for feature in range(0, self.trainX.shape[2]):
                x = range(0, len(self.trainX[trainIndices[ndx], :, feature]))
                y = self.trainX[trainIndices[ndx], :, feature]
                lines.append(featureAxis.plot(x, y, label=self.preparedFeatures[feature]))
            featureAxis.legend()
    
            featureAxis = axs[ndx, VALIDATE_NDX]
            featureAxis.set_title("Feature Validation Data series - category %s" % validateCategories[ndx])
            featureAxis.set_xlabel("time periods")
            featureAxis.set_ylabel("Feature Values")
            lines = []
            for feature in range(0, self.validateX.shape[2]):
                x = range(0, len(self.validateX[validateIndices[ndx], :, feature]))
                y = self.validateX[validateIndices[ndx], :, feature]
                lines.append(featureAxis.plot(x, y, label=self.preparedFeatures[feature]))
            featureAxis.legend()
    
            featureAxis = axs[ndx, TEST_NDX]
            featureAxis.set_title("Feature Testing Data series - category %s" % testCategories[ndx])
            featureAxis.set_xlabel("time periods")
            featureAxis.set_ylabel("Feature Values")
            lines = []
            for feature in range(0, self.testX.shape[2]):
                x = range(0, len(self.testX[testIndices[ndx], :, feature]))
                y = self.testX[testIndices[ndx], :, feature]
                lines.append(featureAxis.plot(x, y, label=self.preparedFeatures[feature]))
            featureAxis.legend()
    
        plt.tight_layout()
        plt.show()
            
        return

'''
Created on Nov 15, 2021

@author: Brian

Git access error problem
'''
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
        self.nxGraph = nx_graph
        '''
    '''
    definition graph properties
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
        graph - graph of nodes and edges defining the information flow and processing
        mlNode - the node in the graph containing the machine learning definition
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
        return self.rawData
    
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
        
    '''
    Functions ...
    '''
    def archiveData(self, location):
        print("\nData \n%s\nwritten to %s for training\n" % (self.data.describe().transpose(), location))
        self.data.to_csv(location, index=False)
          
    def plotGraph(self):
        nx.draw_circular(self.graph, arrows=True, with_labels=True, font_weight='bold')
        plt.show()        
        

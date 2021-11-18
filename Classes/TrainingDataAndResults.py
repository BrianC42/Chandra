'''
Created on Nov 15, 2021

@author: Brian

Git access error problem
'''
import networkx as nx
import matplotlib.pyplot as plt

class Data2Results():
    '''
    Class to prepare training data, store training results and plot results
    
    Properties:
        graph - graph of nodes and edges defining the information flow and processing
        mlNode - the node in the graph containing the machine learning definition
        model - the compiled Keras / Tensorflow model
        fitting - Tensorflow fit results
        
        Data:
            data - Source data samples
            trainX - sample data used to train the model
            trainY - values matching the trainX samples
            testX -  sample data used to test the model
            testY -  values matching the testX samples
            evalX -  sample data used to evaluate the model
            evalY -  values matching the evalX samples
        
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
    '''    
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
        
    
    '''
    machine learning data samples
        data - Source data samples
        trainX - sample data used to train the model
        trainY - values matching the trainX samples
        testX -  sample data used to test the model
        testY -  values matching the testX samples
        evalX -  sample data used to evaluate the model
        evalY -  values matching the evalX samples
    '''
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data
        
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
    def evalX(self):
        return self._evalX
    
    @evalX.setter
    def evalX(self, evalX):
        self._evalX = evalX
        
    @property
    def evalY(self):
        return self._evalY
    
    @evalY.setter
    def evalY(self, evalY):
        self._evalY = evalY
        
    '''
    Functions ...
    '''
    def reportIn(self):
        print("d2r reporting in")
        
    def archiveData(self, location):
        print("\nData \n%s\nwritten to %s for training\n" % (self.data.describe().transpose(), location))
        self.data.to_csv(location, index=False)
          
    def plotGraph(self):
        nx.draw_circular(self.graph, arrows=True, with_labels=True, font_weight='bold')
        plt.show()        
        

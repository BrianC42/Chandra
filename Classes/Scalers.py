'''
Created on Aug 24, 2022

@author: brian
'''
import pandas as pd
import numpy as np

NORMALIZE_RELATIVE_TIME_SERIES = "rts"

class chandraScaler():
    '''
    classdocs
    '''
    scalerType = NORMALIZE_RELATIVE_TIME_SERIES
    dfFeatures = pd.DataFrame()
    dfNormalizedFeatures = pd.DataFrame()

    minNegMin = 0.0
    minNeg = dict()
    negRatio = dict()

    maxPosMax = 0.0
    maxPos = dict()
    posRatio = dict()    
    
    supportedScalers = {NORMALIZE_RELATIVE_TIME_SERIES : 0}

    def __init__(self):
        '''
        Constructor
        relativeTimeSeries:
            normalize multiple time series to the range of values (-1, 1)
            while maintaining the relative difference in values between the time series features
            guarantees that 
            if featureA[x] > featureB[x], normalizedA[x] > normalizedB[x]
            if featureA[x] < featureB[x], normalizedA[x] < normalizedB[x]
            if featureA[x] == 0 normalizedA[x] == 0
        '''
        #print("created scaler")
        return
        
    @property
    def scalerType(self):
        return self._scalerType
    
    @scalerType.setter
    def scalerType(self, scalerType):
        if scalerType in self.supportedScalers:
            self._scalerType = scalerType
        else:
            raise NameError('invalid scaler type')

    def relativeTimeSeries(self):
        #print("set normailzation type to relative time series")
        self._scalerType = NORMALIZE_RELATIVE_TIME_SERIES
        return self
    
    def transformRelativeTimeSeries(self):
        #print("Normalizing relative time series features")
        for feature in range (0, self.dfFeatures.shape[1]):
            for sample in range (0, self.dfFeatures.shape[0]):
                val = self.dfFeatures.iat[sample, feature]
                if val < 0.0:
                    nVal = (val / abs(self.minNeg[feature])) * self.negRatio[self.dfFeatures.columns[feature]]
                elif val > 0.0:
                    nVal = (val / self.maxPos[feature]) * self.posRatio[self.dfFeatures.columns[feature]]
                else:
                    nVal = val
                self.npNormalizedFeatures[sample, feature] = nVal
                
        #print("Maximum values: %s" % np.max(self.npNormalizedFeatures, axis=0))
        #print("Minimum values: %s" % np.min(self.npNormalizedFeatures, axis=0))

        for sample in range (0, self.dfFeatures.shape[0]):
            if self.dfFeatures.iat[sample, 0] > self.dfFeatures.iat[sample, 1]:
                if self.npNormalizedFeatures[sample, 0] <= self.npNormalizedFeatures[sample, 1]:
                    print("relationship changed from >: %s %s" % (self.npNormalizedFeatures[sample, 0], self.npNormalizedFeatures[sample, 1]))
            elif self.dfFeatures.iat[sample, 0] < self.dfFeatures.iat[sample, 1]:
                if self.npNormalizedFeatures[sample, 0] >= self.npNormalizedFeatures[sample, 1]:
                    print("relationship changed from <: %s %s" % (self.npNormalizedFeatures[sample, 0], self.npNormalizedFeatures[sample, 1]))
            else:
                if abs(self.npNormalizedFeatures[sample, 0] - self.npNormalizedFeatures[sample, 1]) > 0.0001:
                    print("relationship changed from ==: %s %s" % (self.npNormalizedFeatures[sample, 0], self.npNormalizedFeatures[sample, 1]))
                
        return self
    
    def fit(self, features):
        #print("Fitting %s" % features)
        if self._scalerType == NORMALIZE_RELATIVE_TIME_SERIES:
            self.dfFeatures = features
            self.minNeg = features.min(axis=0)
            self.minNegMin = features.min(axis=0).min()
            self.maxPos = features.max(axis=0)
            self.maxPosMax = features.max(axis=0).max()
            for feature in features.columns:
                if self.maxPosMax != 0.0:
                    self.posRatio[feature] = self.maxPos[feature] / self.maxPosMax
                else:
                    print("maxPosMax==0.0 for feature %s" % feature)
                    self.posRatio[feature] = 0.0
                if self.maxPosMax != 0.0:
                    self.negRatio[feature] = self.minNeg[feature] / self.minNegMin
                else:
                    print("minNegMin==0.0 for feature %s" % feature)
                    self.negRatio[feature] = 0.0
        self.npNormalizedFeatures = np.zeros((self.dfFeatures.shape[0], self.dfFeatures.shape[1]))
        #print("Positive value factors:\nmaxPosMax=\t%s\nmaxPos=\n%s" % (self.maxPosMax, self.maxPos))
        #print("Negative value factors:\nminNegMin=\t%s\nminNeg=\n%s" % (self.minNegMin, self.minNeg))
        #print("Normalization ratios:\nPositive: %s\nNegative: %s" % (self.posRatio, self.negRatio))
        return self
    
    def transform(self, features):
        #print("transforming %s" % features)
        if self._scalerType == NORMALIZE_RELATIVE_TIME_SERIES:
            self.transformRelativeTimeSeries()
        return self.npNormalizedFeatures
    
    def fit_transform(self):
        raise NameError('method not implemented')
        return 
        
    def get_feature_names_out(self):
        raise NameError('method not implemented')
        return 
        
    def get_params(self):
        raise NameError('method not implemented')
        return 
        
    def inverse_transform(self):
        raise NameError('method not implemented')
        return 

    def partial_fit(self):
        raise NameError('method not implemented')
        return 
    
    def set_params(self):
        raise NameError('method not implemented')
        return 
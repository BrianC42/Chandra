'''
Created on Aug 12, 2024

@author: brian
'''
import sys
import time
import datetime
from tensorflow.python.saved_model.revived_types import get_setter

class OptionChain(object):
    '''      classdocs       '''
    
    ''' accessible data '''
    dataServiceProvider = ""
    dataServiceProviderOptionChain = ""
    optionChainJson = ""
    
    '''
    optionList is a list of tuples with 4 key value pairs per list entry
        symbol
        expirationDate
        strikePrice
        optionDetails
    '''
    #optionList = []
    
    ''' accessible methods   '''
    
    def __init__(self, dataSource="", providerOptionChain=""):
        '''
        Constructor
        '''
        self.optionList = []
        
        self.dataServiceProvider = dataSource
        self.dataServiceProviderOptionChain = providerOptionChain
        self.optionChainJson = providerOptionChain.json()
        #self.UnderlyingSymbol = self.optionChainJson['symbol']

        for chain in ['putExpDateMap', 'callExpDateMap']:
            for exp_date, options in self.optionChainJson[chain].items():
                exp_date, daysToExp = exp_date.split(":")
                #daysToExp = int(daysToExp)
                for strike_price, options_data in options.items():
                    for option in options_data:
                        opt = {"symbol" : self.optionChainJson["symbol"], \
                               "expirationDate" : exp_date, \
                               "daysToExp" : daysToExp, \
                               "strikePrice" : strike_price, \
                               "optionDetails" : option
                               }
                        self.optionList.append(opt)

    @property
    def optionList(self):
        return self._optionList
    
    @optionList.setter
    def optionList(self, optionList):
        self._optionList = optionList


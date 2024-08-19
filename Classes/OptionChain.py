'''
Created on Aug 12, 2024

@author: brian
'''
import json

class OptionChain(object):
    '''      classdocs       '''
    
    ''' accessible data '''
    
    ''' accessible methods   '''
    
    def __init__(self):
        '''
        Constructor
        '''

    '''
    optionList is a list of tuples with 4 key value pairs per list entry
        symbol
        expirationDate
        strikePrice
        optionDetails
    '''
    @property
    def optionList(self):
        return self._optionList
    
    def convertToList(self):
        optionList = []
        for chain in ['putExpDateMap', 'callExpDateMap']:
            for exp_date, options in self.optionChainJson[chain].items():
                exp_date, daysToExp = exp_date.split(":")
                for strike_price, options_data in options.items():
                    for option in options_data:
                        opt = {"symbol" : self.optionChainJson["symbol"], \
                               "expirationDate" : exp_date, \
                               "daysToExp" : daysToExp, \
                               "strikePrice" : strike_price, \
                               "optionDetails" : option
                               }
                        optionList.append(opt)
        return optionList
    
    @optionList.setter
    def optionList(self, optionList):
        self._optionList = optionList

    @property
    def optionChainJson(self):
        return self._optionChainJson
    
    @optionChainJson.setter
    def optionChainJson(self, optionChainJson):
        self._optionChainJson = optionChainJson

    @property
    def marketDataReturn(self):
        return self._marketDataReturn
    
    @marketDataReturn.setter
    def marketDataReturn(self, marketDataReturn):
        self._marketDataReturn = marketDataReturn
        self.optionChainJson = json.loads(marketDataReturn)


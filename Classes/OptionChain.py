'''
Created on Aug 12, 2024

@author: brian
'''
import sys
import time
import datetime
from tensorflow.python.saved_model.revived_types import get_setter

class OptionChain(object):
    '''
    classdocs
    
    
    accessible data
        symbol - underlying security symbol
        contractType - CALL or PUT
        expirationDate
        strikePrice
        jsonData - all data received from the data provider
    
    accessible methods
    
    '''

    UnderlyingSymbol = ""
    contractType = ""
    expirationDate = ""
    strikePrice = ""
    dataServiceProviderData = None

    def __init__(self, symbol="", expiration="", strikePrice="", providerData=None):
        '''
        Constructor
        '''
        self.UnderlyingSymbol = symbol
        self.expirationDate = expiration
        self.strikePrice = strikePrice
        self.dataServiceProviderData = providerData
        
        self.analyzeOptionChain()
        
    '''
    @property
    def configurationFileDir(self):
        return self._configurationFileDir
    
    @configurationFileDir.setter
    def configurationFileDir(self, configurationFileDir):
        self._configurationFileDir = configurationFileDir
    '''
        
    @property
    def type(self):
        return self._dataServiceProviderData["putCall"]
    
    @property
    def UnderlyingSymbol(self):
        return self._UnderlyingSymbol
    
    @UnderlyingSymbol.setter
    def UnderlyingSymbol(self, UnderlyingSymbol):
        self._UnderlyingSymbol = UnderlyingSymbol
    
    @property
    def dataServiceProviderData(self):
        return self._dataServiceProviderData
    
    @dataServiceProviderData.setter
    def dataServiceProviderData(self, dataServiceProviderData):
        self._dataServiceProviderData = dataServiceProviderData
    
    @property
    def expirationDate(self):
        return self._expirationDate
    
    @expirationDate.setter
    def expirationDate(self, expirationDate):
        self._expirationDate = expirationDate
        
    @property
    def strikePrice(self):
        return self._strikePrice

    @strikePrice.setter
    def strikePrice(self, strikePrice):
        self._strikePrice = strikePrice
        
    def analyzeOptionChain(self):
        exc_txt = "analyzeOptionChain exception"
        try:
            '''
            for exp_date, options in apiOpt['putExpDateMap'].items():
                for strike_price, options_data in options.items():
                    for option in options_data:
                        print("Put option: symbol: {}, expiration: {}, strike: {}, bid: {}". \
                              format(apiOpt["symbol"], exp_date, strike_price, option['bid']))
            '''
            print("Analyzing option: symbol: {}, expiration: {}, strike: {} type: {}". \
                  format(self.UnderlyingSymbol, self.expirationDate, self.strikePrice, self.dataServiceProviderData["putCall"]))
            return 
        
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)


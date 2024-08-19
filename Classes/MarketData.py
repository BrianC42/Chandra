'''
Created on Jul 23, 2024

@author: brian
'''
import sys
import time
import requests
import json
import re

import datetime
from datetime import date
from functools import partial
from tkinter import *
from tkinter import ttk

from configuration import get_ini_data
from configuration import read_config_json
from OptionChain import OptionChain

class MarketData(object):
    '''
    classdocs
    
    https://datatracker.ietf.org/doc/html/rfc6749#page-10
    https://datatracker.ietf.org/doc/html/rfc6750
    '''
    
    '''
    class data
    '''
    localAPIAccessFile = ""
    localAPIAccessDetails = ""
    marketDataSymbolsList = []

    def __init__(self):
        '''         Constructor        '''
    

    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @property
    def marketDataJson(self):
        return self._marketDataJson
    
    @marketDataJson.setter
    def marketDataJson(self, marketDataJson):
        self._marketDataJson = marketDataJson

    @property
    def marketDataReturn(self):
        return self._marketDataReturn
    
    @marketDataReturn.setter
    def marketDataReturn(self, marketDataReturn):
        self._marketDataReturn = marketDataReturn
        self._marketDataJson = json.loads(marketDataReturn)

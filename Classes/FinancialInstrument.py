'''
Created on Aug 20, 2024

@author: brian
'''
import sys
import json
import datetime
import re

from configuration import get_ini_data

from financialDataServices import financialDataServices

class FinancialInstrument(object):
    '''
    Support Schwab API financial instrument details json response
            {'instruments': [
                {'fundamental': 
                    {'symbol': 'AAPL', 
                    'high52': 237.23, 
                    'low52': 164.075, 
                    'dividendAmount': 1.0, 
                    'dividendYield': 0.44269, 
                    'dividendDate': '2024-08-12 00:00:00.0', 
                    'peRatio': 34.49389, 
                    'pegRatio': 112.66734, 
                    'pbRatio': 48.06189, 
                    'prRatio': 8.43929, 
                    'pcfRatio': 23.01545, 
                    'grossMarginTTM': 45.962, 
                    'grossMarginMRQ': 46.2571, 
                    'netProfitMarginTTM': 26.4406, 
                    'netProfitMarginMRQ': 25.0043, 
                    'operatingMarginTTM': 26.4406, 
                    'operatingMarginMRQ': 25.0043, 
                    'returnOnEquity': 160.5833, 
                    'returnOnAssets': 22.6119, 
                    'returnOnInvestment': 50.98106, 
                    'quickRatio': 0.79752, 
                    'currentRatio': 0.95298, 
                    'interestCoverage': 0.0, 
                    'totalDebtToCapital': 51.3034, 
                    'ltDebtToEquity': 151.8618, 
                    'totalDebtToEquity': 129.2138, 
                    'epsTTM': 6.56667, 
                    'epsChangePercentTTM': 10.3155, 
                    'epsChangeYear': 0.0, 
                    'epsChange': 0.0, 
                    'revChangeYear': -2.8005, 
                    'revChangeTTM': 0.4349, 
                    'revChangeIn': 0.0, 
                    'sharesOutstanding': 15204137000.0, 
                    'marketCapFloat': 0.0, 
                    'marketCap': 3434462506930.0, 
                    'bookValuePerShare': 4.38227, 
                    'shortIntToFloat': 0.0, 
                    'shortIntDayToCover': 0.0, 
                    'divGrowthRate3Year': 0.0, 
                    'dividendPayAmount': 0.25, 
                    'dividendPayDate': '2024-08-15 00:00:00.0', 
                    'beta': 1.24364, 
                    'vol1DayAvg': 0.0, 
                    'vol10DayAvg': 0.0, 
                    'vol3MonthAvg': 0.0, 
                    'avg10DaysVolume': 47812576, 
                    'avg1DayVolume': 60229630, 
                    'avg3MonthVolume': 64569166, 
                    'declarationDate': '2024-08-01 00:00:00.0', 
                    'dividendFreq': 4, 
                    'eps': 6.13, 
                    'dtnVolume': 40687813, 
                    'nextDividendPayDate': '2024-11-15 00:00:00.0', 
                    'nextDividendDate': '2024-11-12 00:00:00.0', 
                    'fundLeverageFactor': 0.0
                    }, 
                'cusip': '037833100', 
                'symbol': 'AAPL', 
                'description': 'Apple Inc', 
                'exchange': 'NASDAQ', 
                'assetType': 'EQUITY'
                }
                ]
            }
    '''

    def __init__(self, symbol):
        '''
        Constructor
        '''
        try:
            localDirs = get_ini_data("LOCALDIRS")
            aiwork = localDirs['aiwork']
            financialInstrumentDetailsDir = aiwork + localDirs['market_data'] + localDirs['financial_instrument_details']
    
            exc_txt = "An exception occurred requesting financial instrument details for {}".format(symbol)
            self.financialDataServicesObj = financialDataServices()
            response = self.financialDataServicesObj.requestFinancialInstrumentDetails(symbol=symbol)
            
            self.financialInstrumentDetails = response.text
            self.financialInstrumentDetailsJson = json.loads(self.financialInstrumentDetails)
            
            exc_txt = "An exception occurred unpacking financial instrument details for {}".format(symbol)
            instruments = self.financialInstrumentDetailsJson['instruments'][0]
            fundamentals = instruments['fundamental']
            #print("Fundamentals: {}".format(fundamentals))
    
            self.symbol = symbol
            self.symbol = instruments['symbol']
            self.symbol = fundamentals['symbol']
            
            self.description = instruments['description'] 
            self.exchange = instruments['exchange']
            self.assetType = instruments['assetType']
            
            self.high52 = float(fundamentals['high52'])
            self.low52 = float(fundamentals['low52'])
            self.marketCap = float(fundamentals['marketCap']) 
            self.eps = float(fundamentals['eps']) 
            self.dtnVolume = int(fundamentals['dtnVolume'])
    
            if 'cusip' in instruments:
                self.cusip = instruments['cusip']
            else:
                self.cusip = ""
                
            if 'peRatio' in fundamentals:
                self.peRatio = float(fundamentals['peRatio'])
            else:
                self.peRatio = 0.0

            if 'dividendAmount' in fundamentals:
                self.dividendAmount = float(fundamentals['dividendAmount'])
            else:
                self.dividendAmount = 0.0

            if 'dividendPayAmount' in fundamentals:
                self.dividendPayAmount = float(fundamentals['dividendPayAmount'])
            else:
                self.dividendPayAmount = 0.0

            if 'dividendYield' in fundamentals:
                self.dividendYield = float(fundamentals['dividendYield']) / 100
            else:
                self.dividendYield = 0.0
    
            if 'sharesOutstanding' in fundamentals:
                self.sharesOutstanding = float(fundamentals['sharesOutstanding'])
            else:
                self.sharesOutstanding = ""
           
            if 'dividendPayDate' in fundamentals:
                self.dividendPayDate, time = self.split_date_time(fundamentals['dividendPayDate'])
            else:
                self.dividendPayDate = ""
    
            if 'nextDividendPayDate' in fundamentals:
                self.nextDividendPayDate, time = self.split_date_time(fundamentals['nextDividendPayDate'])
            else:
                self.nextDividendPayDate = ""
    
            if 'nextDividendDate' in fundamentals:
                self.nextDividendDate, time = self.split_date_time(fundamentals['nextDividendDate'])
            else:
                self.nextDividendDate = ""
           
        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()
            
    ''' ============ split date time string into date and time strings - start ============= '''
    def split_date_time(self, text):
        try:
            '''
            Splits a text string containing date and time formatted "yyyy-mm-dd hh:mm:ss.s" into a date string and a time string.
            
            Args:
            text: The text string to split.
            
            Returns:
            A tuple containing the date string and the time string.
            '''
        
            # Use regular expression to match the date and time pattern
            match = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}.\d)", text)
            
            if match:
                date_str = match.group(1)
                time_str = match.group(2)
                return date_str, time_str
            else:
                raise ValueError("Invalid date and time format")
    
        except ValueError as e:
            error_message = e.args[0]
            print(error_message)  # Output: Invalid date and time format    
    ''' ============ split date time string into date and time strings - end ============= '''            
            
    ''' ============ data service provider and market data controls - start ============= '''
    @property
    def financialDataServicesObj(self):
        return self._financialDataServicesObj
    
    @financialDataServicesObj.setter
    def financialDataServicesObj(self, financialDataServicesObj):
        self._financialDataServicesObj = financialDataServicesObj

    ''' ============ data service provider and market data controls - start ============= '''

    ''' ============ data service provider returned data - start ============= '''
    @property
    def financialInstrumentDetailsJson(self):
        return self._financialInstrumentDetailsJson
    
    @financialInstrumentDetailsJson.setter
    def financialInstrumentDetailsJson(self, financialInstrumentDetailsJson):
        self._financialInstrumentDetailsJson = financialInstrumentDetailsJson

    @property
    def financialInstrumentDetails(self):
        return self._financialInstrumentDetails
    
    @financialInstrumentDetails.setter
    def financialInstrumentDetails(self, financialInstrumentDetails):
        self._financialInstrumentDetails = financialInstrumentDetails
    ''' ============ data service provider returned data - end ============= '''
        

    ''' ============ data service provider returned fundamental data elements - start ============= '''
    @property
    def symbol(self):
        return self._symbol
    
    @symbol.setter
    def symbol(self, symbol):
        self._symbol = symbol

    @property
    def cusip(self):
        return self._cusip
    
    @cusip.setter
    def cusip(self, cusip):
        self._cusip = cusip

    @property
    def description(self):
        return self._description
    
    @description.setter
    def description(self, description):
        self._description = description

    @property
    def exchange(self):
        return self._exchange
    
    @exchange.setter
    def exchange(self, exchange):
        self._exchange = exchange

    @property
    def assetType(self):
        return self._assetType
    
    @assetType.setter
    def assetType(self, assetType):
        self._assetType = assetType

    @property
    def high52(self):
        return self._high52
    
    @high52.setter
    def high52(self, high52):
        self._high52 = high52
        

    @property
    def low52(self):
        return self._low52
    
    @low52.setter
    def low52(self, low52):
        self._low52 = low52

    @property
    def dividendAmount(self):
        return self._dividendAmount
    
    @dividendAmount.setter
    def dividendAmount(self, dividendAmount):
        self._dividendAmount = dividendAmount

    @property
    def dividendYield(self):
        return self._dividendYield
    
    @dividendYield.setter
    def dividendYield(self, dividendYield):
        self._dividendYield = dividendYield

    @property
    def dividendDate(self):
        return self._dividendDate
    
    @dividendDate.setter
    def dividendDate(self, dividendDate):
        self._dividendDate = dividendDate

    @property
    def peRatio(self):
        return self._peRatio
    
    @peRatio.setter
    def peRatio(self, peRatio):
        self._peRatio = peRatio

    @property
    def sharesOutstanding(self):
        return self._sharesOutstanding
    
    @sharesOutstanding.setter
    def sharesOutstanding(self, sharesOutstanding):
        self._sharesOutstanding = sharesOutstanding

    @property
    def marketCap(self):
        return self._marketCap
    
    @marketCap.setter
    def marketCap(self, marketCap):
        self._marketCap = marketCap

    @property
    def dividendPayAmount(self):
        return self._dividendPayAmount
    
    @dividendPayAmount.setter
    def dividendPayAmount(self, dividendPayAmount):
        self._dividendPayAmount = dividendPayAmount

    @property
    def dividendPayDate(self):
        return self._dividendPayDate
    
    @dividendPayDate.setter
    def dividendPayDate(self, dividendPayDate):
        self._dividendPayDate = dividendPayDate

    @property
    def eps(self):
        return self._eps
    
    @eps.setter
    def eps(self, eps):
        self._eps = eps

    @property
    def dtnVolume(self):
        return self._dtnVolume
    
    @dtnVolume.setter
    def dtnVolume(self, dtnVolume):
        self._dtnVolume = dtnVolume

    @property
    def nextDividendPayDate(self):
        return self._nextDividendPayDate
    
    @nextDividendPayDate.setter
    def nextDividendPayDate(self, nextDividendPayDate):
        self._nextDividendPayDate = nextDividendPayDate

    @property
    def nextDividendDate(self):
        return self._nextDividendDate
    
    @nextDividendDate.setter
    def nextDividendDate(self, nextDividendDate):
        self._nextDividendDate = nextDividendDate
    '''
                'pegRatio': 112.66734, 
                'pbRatio': 48.06189, 
                'prRatio': 8.43929, 
                'pcfRatio': 23.01545, 
                'grossMarginTTM': 45.962, 
                'grossMarginMRQ': 46.2571, 
                'netProfitMarginTTM': 26.4406, 
                'netProfitMarginMRQ': 25.0043, 
                'operatingMarginTTM': 26.4406, 
                'operatingMarginMRQ': 25.0043, 
                'returnOnEquity': 160.5833, 
                'returnOnAssets': 22.6119, 
                'returnOnInvestment': 50.98106, 
                'quickRatio': 0.79752, 
                'currentRatio': 0.95298, 
                'interestCoverage': 0.0, 
                'totalDebtToCapital': 51.3034, 
                'ltDebtToEquity': 151.8618, 
                'totalDebtToEquity': 129.2138, 
                'epsTTM': 6.56667, 
                'epsChangePercentTTM': 10.3155, 
                'epsChangeYear': 0.0, 
                'epsChange': 0.0, 
                'revChangeYear': -2.8005, 
                'revChangeTTM': 0.4349, 
                'revChangeIn': 0.0, 
                'marketCapFloat': 0.0, 
                'bookValuePerShare': 4.38227, 
                'shortIntToFloat': 0.0, 
                'shortIntDayToCover': 0.0, 
                'divGrowthRate3Year': 0.0, 
                'beta': 1.24364, 
                'vol1DayAvg': 0.0, 
                'vol10DayAvg': 0.0, 
                'vol3MonthAvg': 0.0, 
                'avg10DaysVolume': 47812576, 
                'avg1DayVolume': 60229630, 
                'avg3MonthVolume': 64569166, 
                'declarationDate': '2024-08-01 00:00:00.0', 
                'dividendFreq': 4, 
                'fundLeverageFactor': 0.0
    '''
    ''' ============ data service provider returned fundamental data elements - end ============= '''
    '''        

    @property
    def xxx(self):
        return self._xxx
    
    @xxx.setter
    def xxx(self, xxx):
        self._xxx = xxx
        
    '''

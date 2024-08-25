'''
Created on Aug 20, 2024

@author: brian
'''
import sys
import json
import datetime

from configuration import get_ini_data

from financialDataServices import financialDataServices

class FinancialInstrument(object):
    '''
    classdocs
    '''

    def __init__(self, symbol):
        '''
        Constructor
        '''
        exc_txt = "An exception occurred creating a financial instrument object for {}".format(symbol)
        try:
            localDirs = get_ini_data("LOCALDIRS")
            aiwork = localDirs['aiwork']
            #basicMarketDataDir = aiwork + localDirs['market_data'] + localDirs['basic_market_data']
            #augmentedMarketDataDir = aiwork + localDirs['market_data'] + localDirs['augmented_market_data']
            financialInstrumentDetailsDir = aiwork + localDirs['market_data'] + localDirs['financial_instrument_details']
            #optionChainDir = aiwork + localDirs['market_data'] + localDirs['option_chains']
    
            self.financialDataServicesObj = financialDataServices()
            response = self.financialDataServicesObj.requestFinancialInstrumentDetails(symbol=symbol)
            
            self.financialInstrumentDetails = response.text
            self.financialInstrumentDetailsJson = json.loads(self.financialInstrumentDetails)
            
            #for exp_date, options in self.financialInstrumentDetailsJson[chain].items():
                
            instruments = self.financialInstrumentDetailsJson['instruments'][0]
            fundamentals = instruments['fundamental']
    
            self.symbol = symbol
            self.symbol = instruments['symbol']
            self.symbol = fundamentals['symbol']
            
            self.description = instruments['description'] 
            self.exchange = instruments['exchange']
            self.assetType = instruments['assetType']
            
            self.high52 = fundamentals['high52']
            self.marketCap = float(fundamentals['marketCap']) 
            self.dividendPayAmount = float(fundamentals['dividendPayAmount'])
            self.eps = float(fundamentals['eps']) 
            self.dtnVolume = int(fundamentals['dtnVolume'])
    
            if 'cusip' in instruments:
                self.cusip = instruments['cusip']
            else:
                self.cusip = ""
             
            if 'dividendPayDate' in fundamentals:
                self.dividendPayDate = fundamentals['dividendPayDate']
            else:
                self.dividendPayDate = ""
    
            if 'nextDividendPayDate' in fundamentals:
                self.nextDividendPayDate = fundamentals['nextDividendPayDate']
            else:
                self.nextDividendPayDate = ""
    
            if 'nextDividendDate' in fundamentals:
                self.nextDividendDate = fundamentals['nextDividendDate']
            else:
                self.nextDividendDate = ""
    
            '''
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
        
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
            
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
                'low52': 164.075, 
                'dividendAmount': 1.0, 
                'dividendYield': 0.44269, 
                'dividendDate': '2024-08-12 00:00:00.0', 
                'peRatio': 34.49389, 
                'sharesOutstanding': 15204137000.0, 
                
                'marketCap': 3434462506930.0, 
                'dividendPayAmount': 0.25, 
                'dividendPayDate': '2024-08-15 00:00:00.0', 
                'eps': 6.13, 
                'dtnVolume': 40687813, 
                'nextDividendPayDate': '2024-11-15 00:00:00.0', 
                'nextDividendDate': '2024-11-12 00:00:00.0', 
                
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

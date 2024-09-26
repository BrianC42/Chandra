'''
Created on Sep 17, 2024

@author: brian
'''
import sys

import pandas as pd

''' Google workspace requirements start '''
'''
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
'''
''' Google workspace requirements end '''
from configuration import get_ini_data
from configuration import read_config_json

from GoogleSheets import googleSheet
from MarketData import MarketData
from OptionChain import OptionChain
from FinancialInstrument import FinancialInstrument
#from financialDataServices import financialDataServices

'''
class bas: 
    def __init__(self, name): 
        self.name = name 

    def greet(self): 
        print("Hello, my name is", self.name) 

class der(bas): 
    def __init__(self, name, age): 
        super().__init__(name) 
        self.age = age 

    def greet(self): 
        super().greet() 
        print("I am", self.age, "years old.") 

# Create an object of the derived class 
d = der("Alice", 30) 
d.greet() 
'''
class workbooks(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        try:
            exc_txt = "exception developing google workbook class"
            print("workbook class")
            
            ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
            exc_txt = "\nAn exception occurred - unable to authenticate with Google"
            self.gSheets = googleSheet()
                        
            return
        
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
        

class investments(workbooks):
    '''
    classdocs
    
    instrumentCellValues.columns
    Index(['Symbol', 'Change %', 'Last Price', 'Dividend Yield', 'Dividend',
           'Dividend Ex-Date', 'P/E Ratio', '52 Week High', '52 Week Low',
           'Volume', 'Sector', 'Next Earnings Date', 'Morningstar', 'Market Cap',
           'Schwab Equity Rating', 'Argus Rating']
           
    Missing data elements
        'Change %',
        'Sector', 'Next Earnings Date', 
        'Morningstar', 'Schwab Equity Rating', 'Argus Rating'
    
    MarketData elements
        'Last Price' : close
        'Volume'
    
    FinancialInstrument data elements
        'symbol': 'AAPL' 'Symbol'
        'high52': 237.23,  '52 Week High'
        'low52': 164.075 '52 Week Low'
        'dividendYield': 0.44269, 'Dividend Yield'
        'dividendAmount': 1.0, 'Dividend'
        'dividendDate': '2024-08-12 00:00:00.0', 'Dividend Ex-Date'
        'peRatio': 34.49389, 'P/E Ratio'
        'marketCap': 3434462506930.0, 'Market Cap'
        
        {
        'symbol': 'AAPL', 
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
    '''  

    def __init__(self):
        '''
        Constructor
        '''
        ''' Google drive file details '''
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        
        exc_txt = "\nAn exception occurred - unable to access Google sheet"
        googleAuth = get_ini_data("GOOGLE")
        googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
        
        ''' 
        Use the connetion to Google Drive API to read sheet data
        Find file ID of file used for development 
        '''
        self.sheetID = googleDriveFiles["Google IDs"]["Market Data"]["Development"]
        print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Data"]["Development"]))
        print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Data"]["Production"]))
        
        ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
        exc_txt = "\nAn exception occurred - unable to authenticate with Google"
        #gSheets = googleSheet()
        super().__init__()                    
                    
        return

    def markToMarket(self):
        print("mark to market is WIP ====================================")
        
        try:
            exc_txt = "\nAn exception occurred - unable to authenticate with Google"
            '''
            Steps:
            1. Clear the contents of the 'TD Import Inf' tab
            2. create the list of financial symbols by reading the 'Stock Information' tab
            3. create an empty dataframe
            4. for each symbol request the basic financial instrument and the latest market data from the data service provider
            5. load the dataframe information 
            6. format the dataframe data as required by Google workspace update format requirements
            7. update the workbook's 'TD Import Inf' tab
            '''
            
            ''' Step 1 - Clear the contents of the 'TD Import Inf' tab '''
            self.gSheets.clearGoogleSheet(self.sheetID, 'TD Import Inf', 'A1:Z999')
            
            ''' Step 2 - create the list of financial symbols '''
            cellRange = 'Stock Information!A2:AE999'
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange)
            print("Stock Information:\n{}".format(symbolInformation))
            
            ''' Step 3 - create a dataframe '''
            indexLabel = "Symbol"
            columnLabels = ["Symbol", "Change %", "Last Price", "Dividend Yield", "Dividend", "Dividend Ex-Date", "P/E Ratio", \
                            "52 Week High", "52 Week Low", "Volume", "Sector", "Next Earnings Date", "Morningstar", "Market Cap", \
                            "Schwab Equity Rating", "Argus Rating"]
                        
            self.df_marktomarket = pd.DataFrame(columns=columnLabels)
            self.df_marktomarket.set_index(indexLabel, drop=True, inplace=True)
            
            ''' Step 4 - request the basic financial instrument and the latest market data '''
            #for rowNdx in range(len(symbolInformation)):
            for rowNdx in range(15):
                symbol = symbolInformation.loc[rowNdx, 'Symbol']
                exc_txt = "\nAn exception occurred - with symbol: {}".format(symbol)
            
                instrumentDetails = FinancialInstrument(symbol)
                mktData = MarketData(symbol, useArchive=False, periodType="month", period="2", frequencyType="daily", frequency="1")
                candle = mktData.iloc(mktData.candleCount()-1)
                
                ''' step 5. load the dataframe information '''
                new_row = {'symbol' : symbol, \
                           'Change %' : "TBD", \
                           'Last Price' : candle.candleClose, \
                           'Dividend Yield' : instrumentDetails.dividendYield, \
                           'Dividend' : instrumentDetails.dividendAmount, \
                           'Dividend Ex-Date' : "TBD", \
                           "P/E Ratio" : instrumentDetails.peRatio, \
                            "52 Week High" : instrumentDetails.high52, \
                            "52 Week Low" : instrumentDetails.low52, \
                            "Volume" : candle.candleVolume, \
                            "Sector" : "TBD", \
                            "Next Earnings Date" : "TBD", \
                            "Morningstar" : "TBD", \
                            "Market Cap" : instrumentDetails.marketCap, \
                            "Schwab Equity Rating" : "TBD", \
                            "Argus Rating" : "TBD"
                           }

                self.df_marktomarket.loc[symbol] = new_row
            print("Mark to market dataframe\n{}".format(self.df_marktomarket))
            
            
            ''' Step 6 - prepare the dataframe information as required by Google workspace update format requirements '''
            print("WIP - format as required by google update")
            
            ''' Step 7 - update the workbook's 'TD Import Inf' tab '''
            print("WIP - update the workbook")
            
            return 
    
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

    def holdings(self):
        try:
            ''' Return a list of asset symbols categorized as current financial holdings '''
            print("holdings is WIP =====================================")
            symbolList = []
            exc_txt = "Exception occurred returning holdings list"
            cellRange = 'Stock Information!A2:AE999'
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange)
            
            print("Starting symbol information:\n{}".format(symbolInformation))
            
            for rowNdx in range(len(symbolInformation)):
                symbol = symbolInformation.loc[rowNdx, 'Symbol']
                actionCategory = symbolInformation.loc[rowNdx, 'Action Category']
                if actionCategory == "1 - Holding":
                    symbolList.append(symbol)
                    
            return symbolList
    
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
        
    def potentialBuys(self):
        try:
            ''' Return a list of asset symbols categorized as financial assets flagged as potential buys '''
            print("potentialBuys is WIP =====================================")
            symbolList = []
            exc_txt = "Exception occurred returning list of potential buys"
            cellRange = 'Stock Information!A2:AE999'
            symbolInformation = self.gSheets.readGoogleSheet(self.sheetID, cellRange)
            
            print("Starting symbol information:\n{}".format(symbolInformation))
            
            for rowNdx in range(len(symbolInformation)):
                symbol = symbolInformation.loc[rowNdx, 'Symbol']
                actionCategory = symbolInformation.loc[rowNdx, 'Action Category']
                if actionCategory == "4 - Buy":
                    symbolList.append(symbol)
            
            return symbolList
    
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
        

    def updateYahooFinanceData(self):
        print("To update Yahoo finance data, use the script in the workbook")
        
        return
    
class optionTrades(workbooks):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        ''' Google drive file details '''
        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        
        exc_txt = "\nAn exception occurred - unable to access Google sheet"
        googleAuth = get_ini_data("GOOGLE")
        googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
        
        print("file 1: {} - {}".format('development', googleDriveFiles["Google IDs"]["Market Data"]["Development"]))
        print("file 2: {} - {}".format('production', googleDriveFiles["Google IDs"]["Market Data"]["Production"]))
        
        ''' ================ Authenticate with Google workplace and establish a connection to Google Drive API ============== '''
        exc_txt = "\nAn exception occurred - unable to authenticate with Google"
        #gSheets = googleSheet()
        super().__init__()     
                       
        return
        
    def scanOptionChains(self, symbolList=[""], strategy="both", strikeCount=5, strikeRange="OTM", daysToExpiration=60):
        print("scanOptionChains is WIP")
        
        investmentSheet = investments()
        holdingList = investmentSheet.holdings()
        for ndx in range(len(holdingList)):
            symbol = holdingList[ndx]
            options = OptionChain(symbol=symbol, strategy="Call", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            print("Option details: symbol - {}, strategy - {}, strike range - {}".format(options.symbol, \
                                                                                         options.strategy, \
                                                                                         options.strikeRange))

        potentailBuys = investmentSheet.potentialBuys()
        for ndx in range(len(potentailBuys)):
            symbol = potentailBuys[ndx]
            options = OptionChain(symbol=symbol, strategy="Put", strikeCount=5, strikeRange="OTM", daysToExpiration=60)
            print("Option details: symbol - {}, strategy - {}, strike range - {}".format(options.symbol, \
                                                                                         options.strategy, \
                                                                                         options.strikeRange))
        
        return
    
    def filterOptionsChains(self, filterList=[]):
        print("filterOptionsChains is WIP\n\t{}".format(filterList))

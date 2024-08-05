'''
Created on Jul 23, 2024

@author: brian
'''
import sys
from datetime import date
import time
import requests
import json

from configuration import get_ini_data
from configuration import read_config_json

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
    marketData = []
    marketCallOptions = []
    marketPutOptions = []

    def __init__(self):
        '''
        Constructor
        '''
        exc_txt = "Unable to create MarketData object"
        
        try:
            '''
            
            '''
            print("MarketData constructor")
            self.localDirs = get_ini_data("LOCALDIRS")
            self.aiwork = self.localDirs['aiwork']
        
            ''' Schwab APIs '''
            self.schwabConfig = get_ini_data("SCHWAB")
            self.localAPIAccessFile = self.aiwork + "\\" + self.schwabConfig['config']
            self.localAPIAccessDetails = read_config_json(self.localAPIAccessFile)
            
            return
        
        except :
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
        
    def refreshAccessToken(self):
        try:
            print("refreshAccessToken")
            
            url = self.localAPIAccessDetails["APIs"]["Authorization"]["URL"]
            
            postHeaders = {"Content-Type": "application/x-www-form-urlencoded", \
                           "Authorization" : "Basic " + self.localAPIAccessDetails["EncodedClientID_Secret"]}
            
            code = "..."
            postData = {'grant_type' : "authorization_code", \
                        'code' :  code,\
                        'redirect_uri' : self.localAPIAccessDetails["RedirectURI"]}
            
            print("POST: {}\nheaders={}\ndata={}".format(url, postHeaders, postData))
            response = requests.post(url, headers=postHeaders, data=postData)

            if response.status_code == 200:
                rJSON = response.json()
            '''
            curl -X POST https://api.schwabapi.com/v1/oauth/token ^
            -H "Authorization: Basic {Base64 clientID:clientSecret}" ^
            -H "Content-Type: application/x-www-form-urlencoded" ^
            -d "grant_type=refresh_token&refresh_token={refresh_token}"
            
            {
            "expires_in":1800,
            "token_type":"Bearer",
            "scope":"api",
            "refresh_token":"...",
            "access_token":"...",
            "id_token":"..."
            }
            '''
            return True
    
        except :
            print("\n\t***** refreshAccessToken exception *****\n")
            return False        
        
    def requestRefreshAccessTokens(self):
        try:
            print("requestRefreshAccessTokens")
            
            accessToken = self.localAPIAccessDetails['tokens']['access']['access_token']
            accessTokenExpires =  self.localAPIAccessDetails['tokens']['access']['expires']
            if time.time() > accessTokenExpires:
            
                url = self.localAPIAccessDetails["APIs"]["Authorization"]["URL"]
                
                postHeaders = {"Content-Type" : "application/x-www-form-urlencoded", "Authorization" : "Basic " + self.localAPIAccessDetails["EncodedClientID_Secret"]}
                
                postData = {'grant_type' : "refresh_token", \
                           'refresh_token' : self.localAPIAccessDetails["tokens"]["refresh"]["refresh_token"]}
                
                print("POST: {}\nheaders={}\ndata={}".format(url, postHeaders, postData))
                response = requests.post(url, headers=postHeaders, data=postData)
    
                if response.status_code == 200:
                    rJSON = response.json()
                    
                    self.localAPIAccessDetails["tokens"]["refresh"]["refresh_token"] =  rJSON['refresh_token']

                    self.localAPIAccessDetails["tokens"]["access"]["obtained"] = time.time()
                    self.localAPIAccessDetails["tokens"]["access"]["expires"] = rJSON['expires_in'] + time.time()
                    self.localAPIAccessDetails["tokens"]["access"]["expires_in"] = rJSON['expires_in']
                    self.localAPIAccessDetails["tokens"]["access"]["token_type"] =  rJSON['token_type']
                    self.localAPIAccessDetails["tokens"]["access"]["scope"] =  rJSON['scope']
                    self.localAPIAccessDetails["tokens"]["access"]["access_token"] =  rJSON['access_token']
                    self.localAPIAccessDetails["tokens"]["access"]["id_token"] =  rJSON['id_token']
                    
                    self.saveSecureLocalConfiguration()
                else:
                    raise NameError('\n\taccess token refresh failed')
            return True
    
        except :
            print("\n\t***** requestRefreshAccessTokens exception *****\n")
            return False        
        
    def manageMarketDataServiceTokens(self):
        try:
            print("manageMarketDataServiceTokens")
            self.requestRefreshAccessTokens()
            
            accessToken = self.localAPIAccessDetails['tokens']['access']['access_token']
            accessTokenExpires =  self.localAPIAccessDetails['tokens']['access']['expires']
            if time.time() > accessTokenExpires:
                
                
                authorizationURL = self.localAPIAccessDetails['APIs']['Authorization']['URL']
                refreshToken = self.localAPIAccessDetails['tokens']['refresh']['refresh_token']
                refreshTokenExpires =  self.localAPIAccessDetails['tokens']['refresh']['expires']
                clientID = self.localAPIAccessDetails['clientID']
                redirecturi = self.localAPIAccessDetails['RedirectURI']
                params = {'grant_type' : 'refresh_token', \
                          'refresh_token' : refreshToken, \
                          'access_type' : '', \
                          'code' : '', \
                          'client_id' : clientID, \
                          'redirect_uri' : redirecturi}
                
                return True

                response = requests.post(authorizationURL, data=params)
                if response.ok:
                    Authorization_details = json.loads(response.text)
                    self.localAPIAccessDetails["currentToken"] = Authorization_details["access_token"]
                    self.localAPIAccessDetails["scope"] = Authorization_details["scope"]
                    self.localAPIAccessDetails["tokenObtained"] = time.time()
                    self.localAPIAccessDetails["expiresIn"] = Authorization_details["expires_in"]
                    self.localAPIAccessDetails["token_type"] = Authorization_details["token_type"]
                    #tda_update_authentication_details(self.localAPIAccessDetails)
                    
                    print ("*****\nWriting loacl Schwab configuration json file\n*****\n")
                    json_f = open(self.localAPIAccessFile, "w")
                    json.dump(self.localAPIAccessDetails, json_f, indent=1)
                    json_f.close
                else:
                    self.localAPIAccessDetails["currentToken"] = ""
                    self.localAPIAccessDetails["scope"] = ""
                    self.localAPIAccessDetails["tokenObtained"] = 0.0
                    self.localAPIAccessDetails["expiresIn"] = 0.0
                    self.localAPIAccessDetails["token_type"] = ""
                
                    raise NameError('\n\tAuthorization request response not OK\n\tresponse code={}, reason {}, {}'.format(response.status_code, response.reason, response.text))
            return True
    
        except :
            print("\n\t***** manageMarketDataServiceTokens exception *****\n")
            return False        
        
    def manageThrottling(self, apiID=""):
        try:
            print("manageThrottling")
            if apiID == "market data":
                pass
            else:
                raise Exception
            return True
    
        except :
            exc_str = "\n\t***** manageThrottling exception *****\n"
            exc_info = sys.exc_info()
            exc_txt = exc_info[1].args[0]
            sys.exit(exc_str + "\n\t" + exc_txt)
        
    def saveSecureLocalConfiguration(self):
        try:
            print ("*****\nWriting local Schwab configuration json file\n*****\n")
            with open(self.localAPIAccessFile, 'w', encoding='utf-8') as json_f:
                json.dump(self.localAPIAccessDetails, json_f, ensure_ascii=False, indent=4)
            json_f.close

            return
    
        except :
            print("\n\t***** Unable to save the updated authentication details for the market data service api *****\n")
            return False

    def requestMarketData(self, symbolList=""):
        try:
            print("requestMarketData")
            
            if self.manageThrottling("market data"):
                self.manageMarketDataServiceTokens()
                
                urlSymbol = "AAPL"
                urlPeriodType = "month"
                urlPeriod = "1"
                urlFrequencyType = "daily"
                urlFrequency = "1"
                        
                accessToken = self.localAPIAccessDetails["tokens"]["access"]["access_token"]
                hdr2 = "Bearer " + accessToken
                getHeaders = {"accept" : "application/json", "Authorization" : hdr2}
                
                url = "https://api.schwabapi.com/marketdata/v1/pricehistory"
                payload = {'symbol' : urlSymbol, \
                           'periodType' : urlPeriodType, \
                           'period' : urlPeriod, \
                           'frequencyType' : urlFrequencyType, \
                           'frequency' : urlFrequency}
                
                print("GET: {}\nheaders={}\nparams={}".format(url, getHeaders, payload))
                response = requests.get(url, headers=getHeaders, params=payload)

                if response.status_code == 200:
                    '''
                    successful request. Data returned.
                    '''
                    rJSON = response.json()
                    responseSymbol = response.text['symbol']
                    responseEmpty = response.text['empty']
                    response.candeles = response.text['candles']
                    pass
                else:
                    if response.status_code == 400:
                        descText = "Generic client error"
                    elif response.status_code == 401:
                        descText = "Unauthorized"
                    elif response.status_code == 404:
                        descText = "Not found"
                    elif response.status_code == 500:
                        descText = "Internal server error"
                    print("Market data request failed - code: {}, {}".format(response.status_code, descText))
                    raise Exception

            '''
            {
            "candles": [
                {
                  "open": 207.72,
                  "high": 212.7,
                  "low": 206.59,
                  "close": 208.14,
                  "volume": 80727006,
                  "datetime": 1719205200000
                },
                {
                .
                .
                .
                }
              ],
              "symbol": "AAPL",
              "empty": false
            }
            
            {
            "errors": [
                {
                  "id": "6808262e-52bb-4421-9d31-6c0e762e7dd5",
                  "status": "400",
                  "title": "Bad Request",
                  "detail": "Missing header",
                  "source": {
                    "header": "Authorization"
                  }
                },
                {
                  "id": "0be22ae7-efdf-44d9-99f4-f138049d76ca",
                  "status": "400",
                  "title": "Bad Request",
                  "detail": "Search combination should have min of 1.",
                  "source": {
                    "pointer": [
                      "/data/attributes/symbols",
                      "/data/attributes/cusips",
                      "/data/attributes/ssids"
                    ]
                  }
                },
                {
                  "id": "28485414-290f-42e2-992b-58ea3e3203b1",
                  "status": "400",
                  "title": "Bad Request",
                  "detail": "valid fields should be any of all,fundamental,reference,extended,quote,regular or empty value",
                  "source": {
                    "parameter": "fields"
                  }
                }
              ]
            }
            '''
            return True
    
        except :
            print("\n\t***** requestMarketData exception *****\n")
            return False

    def requestMarketCalls(self):
        try:
            print("requestMarketCalls")
            return True
    
        except :
            print("\n\t***** TBD *****\n")
            return False

    def requestMarketPuts(self):
        try:
            print("requestMarketPuts")
            return True
    
        except :
            print("\n\t***** TBD *****\n")
            return False


'''
Created on Aug 15, 2024

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

class financialDataServices(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        exc_txt = "Unable to create MarketData object"
        try:
            #print("MarketData constructor")
            self.localDirs = get_ini_data("LOCALDIRS")
            self.aiwork = self.localDirs['aiwork']
        
            #basicMarketDataDir = aiwork + localDirs['market_data'] + localDirs['basic_market_data']
            #augmentedMarketDataDir = aiwork + localDirs['market_data'] + localDirs['augmented_market_data']
            #financialInstrumentDetailsDir = aiwork + localDirs['market_data'] + localDirs['financial_instrument_details']
            #optionChainDir = aiwork + localDirs['market_data'] + localDirs['option_chains']

            ''' Schwab APIs '''
            self.schwabConfig = get_ini_data("SCHWAB")
            self.localAPIAccessFile = self.aiwork + "\\" + self.schwabConfig['config']
            self.localAPIAccessDetails = read_config_json(self.localAPIAccessFile)
            
            return
        
        except :
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
        
    '''    ================ OAuth flow tkInter UI development - start ===================== '''
    def formatCurl(self, redirecturi, curlText):
        encodedIDSecret = self.localAPIAccessDetails["EncodedClientID_Secret"]
        reduri = redirecturi.get(1.0, END)
        
        print("redirect uri:", reduri)
            
        authCode = re.search(r'code=(.*?)&session', reduri)
        self.localAPIAccessDetails["tokens"]["authorization"]["authorization_code"] = authCode.group(1)
        self.localAPIAccessDetails["tokens"]["authorization"]["obtained"] = time.time()
        self.localAPIAccessDetails["tokens"]["authorization"]["expires_in"] = 30
        self.localAPIAccessDetails["tokens"]["authorization"]["expires"] = time.time() + 30
        print("Authorization:", authCode)
        
        if authCode:
            print('curl -X POST https://api.schwabapi.com/v1/oauth/token ^')
            print('-H "Authorization: Basic ' + encodedIDSecret + '" ^')
            print('-H "Content-Type: application/x-www-form-urlencoded" ^')
            print('-d "grant_type=authorization_code&code=' + authCode.group(1) + '&redirect_uri=https://127.0.0.1"')
            
            curlCmd = 'curl -X POST https://api.schwabapi.com/v1/oauth/token ^\n' + \
                      '-H "Authorization: Basic ' + encodedIDSecret + '" ^\n' + \
                      '-H "Content-Type: application/x-www-form-urlencoded" ^\n' + \
                      '-d "grant_type=authorization_code&code=' + authCode.group(1) + '&redirect_uri=https://127.0.0.1"\n'
            
            curlText.replace(1.0, END, curlCmd)
            
            print("Text set to:\n", curlText.get(1.0, END))
        
        else:
            pass
    
        return
    
    def saveAuthorizations(self, authorizationResponse):
        resp = authorizationResponse.get(1.0, END)
        print("Authorization response:\n", resp)
        
        respJson = json.loads(resp)
        
        acquired = time.time()
        self.localAPIAccessDetails["tokens"]["refresh"]["obtained"] = acquired
        self.localAPIAccessDetails["tokens"]["access"]["obtained"] = acquired
        # (60 * 30) reduction to minimize chances of expiration during a process
        self.localAPIAccessDetails["tokens"]["refresh"]["expires_in"] = (7*24*60*60) - (60*30)
        refreshExpires = acquired + self.localAPIAccessDetails["tokens"]["refresh"]["expires_in"]
        self.localAPIAccessDetails["tokens"]["refresh"]["expires"] = refreshExpires
        # (60 * 5) reduction to minimize chances of expiration during an access request
        accessExpires = acquired + respJson["expires_in"] - (60*5)
        self.localAPIAccessDetails["tokens"]["access"]["expires"] = accessExpires

        self.localAPIAccessDetails["tokens"]["refresh"]["refresh_token"] =  respJson['refresh_token']

        self.localAPIAccessDetails["tokens"]["access"]["expires_in"] = respJson['expires_in']
        self.localAPIAccessDetails["tokens"]["access"]["token_type"] =  respJson['token_type']
        self.localAPIAccessDetails["tokens"]["access"]["scope"] =  respJson['scope']
        self.localAPIAccessDetails["tokens"]["access"]["access_token"] =  respJson['access_token']
        self.localAPIAccessDetails["tokens"]["access"]["id_token"] =  respJson['id_token']
        
        print("access token\n", respJson['access_token'])
        print("expires in: ", respJson["expires_in"])
        print("token scope: ", respJson["scope"])
        print("token type: ", respJson["token_type"])
        print("refresh token:\n", respJson["refresh_token"])
        print("id token\n", respJson['id_token'])
        print("acquired: ", acquired)
        print("refresh expires", refreshExpires)
        print("access expires", accessExpires)
        
        self.saveSecureLocalConfiguration()

        return
    
    def authorizationInterface(self, ):
        try:
            exc_txt = "Exception occurred displaying the authorization code UI"
            ROW_OAUTH_INSTRUCTION = 1
            ROW_OAUTH_URI = 2
            ROW_PASTE_INSTRUCTION = 3
            ROW_OAUTH_REDIRECTION = 4
            ROW_FORMAT_BUTTON = 5
            ROW_PASTE_CURL_CMD = 6
            ROW_CURL_TEXT = 7
            TBD1 = 8
            ROW_AUTHORIZATION_RESPONSE = 10
            ROW_PASTE_RESPONSE_LABEL = 9
            ROW_SAVE_AUTHORIZATION_RESPONSE = 11
    
            OAuthService = 'https://api.schwabapi.com/v1/oauth/authorize'
            clientID = '?client_id=' + self.localAPIAccessDetails["clientID"]
            redirectUri = '&redirect_uri=' + 'https://127.0.0.1'
            encodedIDSecret = "EncodedClientID_Secret"
            
            OAuthFlowService = OAuthService + clientID + redirectUri
    
            # Create the main window
            uiRoot = Tk()
            uiRoot.title("Authorization OAuth Flow Data")
            uiRoot.columnconfigure(0, weight=1)
            uiRoot.rowconfigure(0, weight=1)
    
            mainframe = ttk.Frame(uiRoot, padding="3 3 12 12")
            mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
    
            uri = StringVar()
            redirectUri = StringVar()
            
            oauthURI = Text(mainframe, width=100, height=4)
            oauthURI.insert(1.0, OAuthFlowService)
            oauthURI.grid(column=1, row=ROW_OAUTH_URI, sticky=(W, E))
            
            Inst1 = "Preparation:\n\tOpen a web browser page\n\tOpen the Windows cmd shell"
            Inst2 = "\nStep 1: Paste the following uri into the browser search bar to initiate the OAuth flow"
            Inst3 = "\nStep 2: When the authenticating server redirects the browser paste the search bar uri to the redirect box"
            Inst4 = "\nStep 3: Click the Format Post button"
            Inst5 = "\nStep 4: WITHIN 30 SECONDS paste the curl cmd lines into the Windows cmd shell"
            Inst6 = "\nStep 5: Paste the response from the authentication server into the Authentication box"
            Inst7 = "\nStep 6: Click the Save Authorization button"
            instructions = Inst1 + Inst2 +  Inst3 +  Inst4 +  Inst5 +  Inst6 +  Inst7
            
            step1Inst = "Paste redirect uri here"
            step4Inst = "paste the curl command below into a cmd window WITHIN 30 SECONDS"
            step5Inst = "Paste the response from the authentication server below"
            
            ttk.Label(mainframe, text=instructions).grid(column=1, row=ROW_OAUTH_INSTRUCTION, sticky=W)
            ttk.Label(mainframe, text=step1Inst).grid(column=1, row=ROW_PASTE_INSTRUCTION, sticky=W)
            ttk.Label(mainframe, text=step4Inst).grid(column=1, row=ROW_PASTE_CURL_CMD, sticky=W)
            ttk.Label(mainframe, text=step5Inst).grid(column=1, row=ROW_PASTE_RESPONSE_LABEL, sticky=W)
            
            # multi-line text box to hold the formatted curl cmd
            curlTxt = Text(mainframe, width=100, height=6)
            curlTxt.grid(column=1, row=ROW_CURL_TEXT, sticky=(W, E))
            
            # multi-line text box and button to receive the redirection uri
            uri_entry = Text(mainframe, width=100, height=4)
            uri_entry.grid(column=1, row=ROW_OAUTH_REDIRECTION, sticky=(W, E))
            # button to format the redirection uri into an OAuth post for the authentication server
            ttk.Button(mainframe, text="Format post", command=partial(self.formatCurl, uri_entry, curlTxt)). \
                        grid(column=1, row=ROW_FORMAT_BUTTON, sticky=W)
            
            # multi-line text box to paste the authorization server response into
            authorizationResponse = Text(mainframe, width=100, height=4)
            authorizationResponse.grid(column=1, row=ROW_AUTHORIZATION_RESPONSE, sticky=(W, E))
            # button to save the authorization response for future processing
            ttk.Button(mainframe, text="Save Authorization", command=partial(self.saveAuthorizations, authorizationResponse)). \
                        grid(column=1, row=ROW_SAVE_AUTHORIZATION_RESPONSE, sticky=W)
    
            for child in mainframe.winfo_children(): 
                child.grid_configure(padx=5, pady=5)
            
            uri_entry.focus()
            #uiRoot.bind("<Return>", formatCurl)
        
            # Start the GUI event loop
            uiRoot.mainloop()
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
        return
    '''    ================ OAuth flow tkInter UI development - end ===================== '''

    
    def refreshAccessToken(self):
        exc_txt = "refreshAccessToken exception"
        try:
            #print("refreshAccessToken")
            
            url = self.localAPIAccessDetails["APIs"]["Authorization"]["URL"]
            
            postHeaders = {"Content-Type": "application/x-www-form-urlencoded", \
                           "Authorization" : "Basic " + self.localAPIAccessDetails["EncodedClientID_Secret"]}
            
            code = "..."
            postData = {'grant_type' : "refresh_token", \
                        'refresh_token' :  self.localAPIAccessDetails["tokens"]["refresh"]["refresh_token"]}
            
            #print("POST: {}\nheaders={}\ndata={}".format(url, postHeaders, postData))
            response = requests.post(url, headers=postHeaders, data=postData)

            if response.status_code == 200:
                respJson = response.json()
                
                acquired = time.time()
                self.localAPIAccessDetails["tokens"]["access"]["obtained"] = acquired
                # (60 * 5) reduction to minimize chances of expiration during an access request
                accessExpires = acquired + respJson["expires_in"] - (60*5)
                self.localAPIAccessDetails["tokens"]["access"]["expires"] = accessExpires
        
                self.localAPIAccessDetails["tokens"]["refresh"]["refresh_token"] =  respJson['refresh_token']
        
                self.localAPIAccessDetails["tokens"]["access"]["expires_in"] = respJson['expires_in']
                self.localAPIAccessDetails["tokens"]["access"]["token_type"] =  respJson['token_type']
                self.localAPIAccessDetails["tokens"]["access"]["scope"] =  respJson['scope']
                self.localAPIAccessDetails["tokens"]["access"]["access_token"] =  respJson['access_token']
                self.localAPIAccessDetails["tokens"]["access"]["id_token"] =  respJson['id_token']
                
                '''
                print("access token\n", respJson['access_token'])
                print("expires in: ", respJson["expires_in"])
                print("token scope: ", respJson["scope"])
                print("token type: ", respJson["token_type"])
                print("refresh token:\n", respJson["refresh_token"])
                print("id token\n", respJson['id_token'])
                print("acquired: ", acquired)
                print("access expires", accessExpires)
                '''
                
                self.saveSecureLocalConfiguration()
            else:
                if response.status_code == 400:
                    exc_txt = "Generic client error"
                else:
                    exc_txt = "Error code = {}\n".format(response.status_code)
            return True
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
                
    def manageMarketDataServiceTokens(self):
        try:
            #print("manageMarketDataServiceTokens")
            
            refreshTokenExpires = self.localAPIAccessDetails['tokens']['refresh']['expires']
            if time.time() > refreshTokenExpires:
                self.authorizationInterface()
            else:
                accessTokenExpires = self.localAPIAccessDetails['tokens']['access']['expires']
                if time.time() > accessTokenExpires:
                    self.refreshAccessToken()
            
            return True
    
        except :
            print("\n\t***** manageMarketDataServiceTokens exception *****\n")
            return False        
        
    def manageThrottling(self, apiID=""):
        try:
            #print("manageThrottling")
            if apiID == "market data":
                pass
            elif apiID == "call option":
                pass
            elif apiID == "put option":
                pass
            elif apiID == "financial instruments":
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
            #print ("*****\nWriting local Schwab configuration json file\n*****\n")
            with open(self.localAPIAccessFile, 'w', encoding='utf-8') as json_f:
                json.dump(self.localAPIAccessDetails, json_f, ensure_ascii=False, indent=4)
            json_f.close

            return
    
        except :
            print("\n\t***** Unable to save the updated authentication details for the market data service api *****\n")
            return False

    def requestFinancialInstrumentDetails(self, symbol=""):
        exc_txt = "An exception occurred requesting financial instrument details"
        try:
            #print("requestFinancialInstrumentDetails")
            
            self.manageThrottling("financial instruments")
            self.manageMarketDataServiceTokens()
            
            accessToken = self.localAPIAccessDetails["tokens"]["access"]["access_token"]
            hdr2 = "Bearer " + accessToken
            getHeaders = {"accept" : "application/json", "Authorization" : hdr2}
            
            url = "https://api.schwabapi.com/marketdata/v1/instruments"
            payload = {'symbol' : symbol, 'projection' : 'fundamental'}
            
            #print("GET: {}\nheaders={}\nparams={}".format(url, getHeaders, payload))
            response = requests.get(url, headers=getHeaders, params=payload)

            if response.status_code == 200:
                '''  -----------  successful request. Data returned.  --------------  '''
                pass
            else:
                if response.status_code == 400:
                    exc_txt = "Generic client error"
                    '''
                    {
                    "errors": [
                        {
                          "id": "6808262e-52bb-4421-9d31-6c0e762e7dd5",
                          "status": "400",
                          "title": "Bad Request",
                          "detail": "Missing header",
                          "detail": "Search combination should have min of 1.",
                          "detail": "valid fields should be any of all,fundamental,reference,extended,quote,regular or empty value",
                          "source": {
                            "header": "Authorization"
                          }
                        },
                    '''
                elif response.status_code == 401:
                    exc_txt = "Unauthorized"
                elif response.status_code == 404:
                    exc_txt = "Not found"
                elif response.status_code == 500:
                    exc_txt = "Internal server error"
                else:
                    exc_txt = "Unrecognized error"
                print("Market data request failed - code: {}, {}".format(response.status_code, exc_txt))
                raise Exception

            return response
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

            
    def requestMarketData(self, symbol="", periodType="", period="", frequencyType="", frequency="", startDate=None, endDate=None):
        exc_txt = "An exception occurred requesting market data"
        try:
            #print("requestMarketData")
            
            self.manageThrottling("market data")
            self.manageMarketDataServiceTokens()
            
            accessToken = self.localAPIAccessDetails["tokens"]["access"]["access_token"]
            hdr2 = "Bearer " + accessToken
            getHeaders = {"accept" : "application/json", "Authorization" : hdr2}
            
            url = "https://api.schwabapi.com/marketdata/v1/pricehistory"
            if startDate == None or endDate == None:
                payload = {'symbol' : symbol, \
                           'periodType' : periodType, \
                           'period' : period, \
                           'frequencyType' : frequencyType, \
                           'frequency' : frequency}
            else:
                '''
                https://api.schwabapi.com/marketdata/v1/pricehistory?
                symbol=AAPL&
                periodType=month&
                period=1&frequencyType=daily&frequency=1&
                startDate=1699855200000&endDate=1724341256000
                '''
                payload = {'symbol' : symbol, \
                           'periodType' : periodType, \
                           'period' : period, \
                           'frequencyType' : frequencyType, \
                           'frequency' : frequency, \
                           'startDate' : startDate, \
                           'endDate' : endDate}
            
            #print("GET: {}\nheaders={}\nparams={}".format(url, getHeaders, payload))
            response = requests.get(url, headers=getHeaders, params=payload)

            if response.status_code == 200:
                '''  -----------  successful request. Data returned.  --------------  '''
                pass
            else:
                if response.status_code == 400:
                    exc_txt = "Generic client error"
                    '''
                    {
                    "errors": [
                        {
                          "id": "6808262e-52bb-4421-9d31-6c0e762e7dd5",
                          "status": "400",
                          "title": "Bad Request",
                          "detail": "Missing header",
                          "detail": "Search combination should have min of 1.",
                          "detail": "valid fields should be any of all,fundamental,reference,extended,quote,regular or empty value",
                          "source": {
                            "header": "Authorization"
                          }
                        },
                    '''
                elif response.status_code == 401:
                    exc_txt = "Unauthorized"
                elif response.status_code == 404:
                    exc_txt = "Not found"
                elif response.status_code == 500:
                    exc_txt = "Internal server error"
                else:
                    exc_txt = "Unrecognized error"
                raise Exception

            return response
    
        except Exception:
            #print("Market data request failed - code: {}, {}".format(response.status_code, exc_txt))
            sys.exit("Market data request failed - code: {}, {}".format(response.status_code, exc_txt))

    def requestOptionChain(self, type="Both", symbol="", strikeCount=0, range="OTM", daysToExpiration=0):
        exc_txt = "An exception occurred requesting option chain"
        try:
            #print("requestOptionChain")
            
            if type == "Call":
                contractType = "CALL"
            elif type == "Put":
                contractType = "PUT"
            elif type == "Both":
                contractType = "ALL"
            else:
                raise Exception                
            
            # From date(pattern: yyyy-MM-dd)
            # To date (pattern: yyyy-MM-dd)
            # OptionChain strategy. Default is SINGLE. ANALYTICAL allows the use of 
            # volatility, underlyingPrice, interestRate, and daysToExpiration params to calculate theoretical values.
            now = time.time()
            dtNow = datetime.datetime.fromtimestamp(now)
            strNow = dtNow.strftime("%Y-%m-%d")
            duration = daysToExpiration * (60*60*24)
            optionExpires = now + duration
            dtExp = datetime.datetime.fromtimestamp(optionExpires)
            strExp = dtExp.strftime("%Y-%m-%d")
            #print("{} options expiring between {} and {}".format(contractType, strNow, strExp))
            
            if type == "Call":
                self.manageThrottling("call option")
            elif type == "Put":
                self.manageThrottling("put option")
                
            self.manageMarketDataServiceTokens()
            
            accessToken = self.localAPIAccessDetails["tokens"]["access"]["access_token"]
            hdr2 = "Bearer " + accessToken
            getHeaders = {"accept" : "application/json", "Authorization" : hdr2}
            
            url = "https://api.schwabapi.com/marketdata/v1/chains"
            payload = {'symbol' : symbol, \
                       'contractType' : contractType, \
                       'strikeCount' : strikeCount, \
                       'range' : range, \
                       'fromDate' : strNow, \
                       'toDate' : strExp
                       }
            
            #print("GET: {}\nheaders={}\nparams={}".format(url, getHeaders, payload))
            response = requests.get(url, headers=getHeaders, params=payload)

            if response.status_code == 200:
                '''  -----------  successful request. Data returned.  --------------  '''
                
                pass
                
            else:
                if response.status_code == 400:
                    exc_txt = "Generic client error"
                    '''
                    {
                    "errors": [
                        {
                          "id": "6808262e-52bb-4421-9d31-6c0e762e7dd5",
                          "status": "400",
                          "title": "Bad Request",
                          "detail": "Missing header",
                          "detail": "Search combination should have min of 1.",
                          "detail": "valid fields should be any of all,fundamental,reference,extended,quote,regular or empty value",
                          "source": {
                            "header": "Authorization"
                          }
                        },
                    '''
                elif response.status_code == 401:
                    exc_txt = "Unauthorized"
                elif response.status_code == 404:
                    exc_txt = "Not found"
                elif response.status_code == 500:
                    exc_txt = "Internal server error"
                else:
                    exc_txt = "Unrecognized error"
                print("Option request failed - code: {}, {}".format(response.status_code, exc_txt))
                raise Exception

            return response
    
        except ValueError:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
        
'''
Created on Aug 6, 2023

@author: Brian
        Google sheets API documentation
        
        https://cloud.google.com/apis/docs/client-libraries-explained
        https://cloud.google.com/python/docs/reference
        
        https://google-auth.readthedocs.io/en/stable/user-guide.html
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        
        https://developers.google.com/sheets/api/guides/concepts
        https://developers.google.com/sheets/api/reference/rest
        https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets/cells
        
        https://googleapis.github.io/google-api-python-client/docs/dyn/sheets_v4.spreadsheets.html

        from google.auth.transport.requests import Request
            
            
'''
import sys
import os.path
import re

import datetime
from datetime import date

import numpy as np
import pandas as pd

from configuration import get_ini_data
#from configuration import read_config_json

'''    https://github.com/googleapis/google-api-python-client/blob/main/docs/oauth-installed.md '''
from google.auth.transport.requests import Request
import requests

#from google_auth_oauthlib.flow import Flow
from google_auth_oauthlib.flow import InstalledAppFlow

from google.oauth2.credentials import Credentials
#from google.oauth2.credentials import UserAccessTokenCredentials

from googleapiclient.discovery import build
#from googleapiclient.errors import HttpError


class googleSheet():
    '''
    classdocs
    '''

    ''' 
    class data 
    '''
    ''' credentials '''
    creds = None
    ''' token '''
    token = None

    '''  '''
    googleDriveFiles = ""

    # If modifying these scopes, delete the file token.json.
    SCOPE_LIMITED = ['https://www.googleapis.com/auth/drive.file']
    SCOPE_RO = ['https://www.googleapis.com/auth/spreadsheets.readonly']
    SCOPE_RW = ['https://www.googleapis.com/auth/spreadsheets']
    
    MMDDYYYY = 0
    YYYYMMDD = 1

    '''
    Class data
        self.localDirs local file structure details
        self.googleAuth local file folder containing Google authentication files
        self.aiwork local file folder for AI and machine learning configuration files and data
        self.GoogleTokenPath directory containing Google token file
        self.credentialsPath directory containing Google credentials file
        self.sheetService Google service object
        self.googleSheet Google sheet service object
    '''
    batchRequests = list([])

    def __init__(self):
        '''
        Constructor
        googleAuth = get_ini_data("GOOGLE")
        print("Google authentication\n\ttoken: {}\n\tcredentials: {}".format(googleAuth["token"], googleAuth["credentials"]))

        localDirs = get_ini_data("LOCALDIRS")
        aiwork = localDirs['aiwork']
        googleDriveFiles = read_config_json(aiwork + "\\" + googleAuth['fileIDs'])
        '''

        self.openGoogleSheetService()
        return

    def gCellStrToInt(self, value):
        '''
        convert a string from a Google sheet cell formatted as an integer value to a python int
        '''
        try:
            value = value.replace(",", "")
            return int(value)
        
        except ValueError:
            return np.nan
    
    def gCellStrToFloat(self, value):
        '''
        convert a string from a Google sheet cell formatted as a floating point number value to a python float
        '''
        try:
            value = value.replace("'", "")
            value = value.replace(",", "")
            return float(value)
        
        except ValueError:
            return np.nan
    
    def gCellDollarStrToFloat(self, value):
        '''
        convert a string from a Google sheet cell formatted as a dollar value to a python float
        '''
        try:
            if value == "-":
                value = 0.0
            else:
                value = value.replace("'", "")
                value = value.replace(",", "")
                value = value.replace("$", "")
            return float(value)
        
        except ValueError:
            return np.nan
    
    def gCellPctStrToFloat(self, value):
        '''
        convert a string from a Google sheet cell formatted as a percentage value to a python float
        '''
        try:
            if value == "-":
                fVal = 0.0
            else:
                value = value.replace("'", "")
                value = value.replace(",", "")
                value = value.replace("%", "")
                fVal = float(value)
                fVal = fVal / 100
            return float(fVal)
        
        except ValueError:
            return np.nan
    
    def gCellStrToBool(self, value):
        '''
        convert a string from a Google sheet cell formatted as a boolean to a python boolean object
        '''
        try:
            if value == 'TRUE':
                return True
            elif value == 'FALSE':
                return False
    
        except ValueError:
            return np.nan
            
    def gCellDateStrToDate(self, value, fmt, dummy):
        '''
        convert a string from a Google sheet cell formatted as a date to a python date object
        '''
        try:
            yyyy = datetime.MINYEAR
            mm = 1
            dd = 1
            sheetDate = date(yyyy, mm, dd)
            
            mo = re.search("/", value)
            if mo == None:
                mo = re.search("-", value)
                if mo == None:
                    pass
                else:
                    seperator = "-"
            else:
                seperator = "/"
    
            if value == "-":
                pass
            else:
                elems = re.split(seperator, value)
                if fmt == 'MMDDYYYY':
                    yyyy = int(elems[2])
                    mm = int(elems[0])
                    dd = int(elems[1])
                elif fmt == 'YYYYMMDD':
                    yyyy = int(elems[0])
                    mm = int(elems[1])
                    dd = int(elems[2])
                else:
                    pass
                sheetDate = date(yyyy, mm, dd)
            
            return sheetDate
    
        except ValueError:
            return sheetDate
            
    def readGoogleSheet(self, spreadsheetID, readRange):
        ''' read Google sheet data and return it as a DataFrame '''
        exc_txt = "\nAn exception occurred - unable to read Google sheet values"
        try:
            result = ""
    
            result = self.googleSheet.values().get(spreadsheetId=spreadsheetID, range=readRange).execute()
            #print("\tmajorDimension: {}".format(result.get('majorDimension')))
            #print("\trange: {}".format(result.get('range')))
    
            values = result.get('values', [])
            if not values:
                print('\tNo data found.')
            else:
                if len(values) < 2:
                    dfVals = pd.DataFrame(values)
                else:
                    dfVals = pd.DataFrame(values[1:], columns=values[0])
        
        except Exception:
            #exc_info = sys.exc_info()
            #exc_str = exc_info[1].args[0]
            sys.exit(exc_txt)
    
        return dfVals
    
    def updateGoogleSheet(self, spreadsheetID, writeRange, updateData):
        ''' update Google sheet data '''
        exc_txt = "\nAn exception occurred - unable to write data to Google sheet"
        try:
            cellValues = updateData.values.tolist()
            if len(updateData.shape) == 1:
                requestBody = {'values': [cellValues]}
            else:
                requestBody = {'values': cellValues}
            result = self.googleSheet.values().update(spreadsheetId=spreadsheetID, range=writeRange, \
                                           valueInputOption="USER_ENTERED", body=requestBody).execute()
            '''
            print("\tspreadsheetId: {}".format(result.get('spreadsheetId')))
            print("\tupdatedRange: {}".format(result.get('updatedRange')))
            print("\tupdatedRows: {}".format(result.get('updatedRows')))
            print("\tupdatedColumns: {}".format(result.get('updatedColumns')))
            print("\tupdatedCells: {}".format(result.get('updatedCells')))
            '''
            
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    
        return result.get('updatedCells')
    
    def addGoogleSheet(self, spreadsheetID, sheetTitle):
        ''' update Google workbook - add sheet '''
        exc_txt = "\nAn exception occurred attempting to add a new sheet"
        try:
            addSheetRequest = {"properties" : {"title": sheetTitle}}
            self.batchRequests.append({"addSheet" : addSheetRequest})
            requestBody = {"requests" : self.batchRequests}
            #print("\nbatchUpdate requestBody:\n\t{}".format(requestBody))
            result = self.googleSheet.batchUpdate(spreadsheetId=spreadsheetID, body=requestBody).execute()
            self.batchRequests.clear()
            
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    
        return

    def openGoogleSheetService(self):
        '''
        Use local file credentials and tokens to authenticate with Google and establish a service to access sheets
        '''
        exc_txt = "\nAn exception occurred - unable to open Google sheets service"
    
        try:
            self.localDirs = get_ini_data("LOCALDIRS")
            self.aiwork = self.localDirs['aiwork']
        
            ''' Google APIs '''
            self.googleAuth = get_ini_data("GOOGLE")
            self.GoogleTokenPath = self.aiwork + "\\" + self.googleAuth["token"]
            self.credentialsPath = self.aiwork + "\\" + self.googleAuth["credentials"]
        
            '''
            The file token.json stores the user's access and refresh tokens, and is
            created automatically when the authorization flow completes for the first time
            '''
            self.creds = None
    
            if os.path.exists(self.GoogleTokenPath):
                self.creds = Credentials.from_authorized_user_file(self.GoogleTokenPath, self.SCOPE_RW)
                
            # If there are no (valid) credentials available, let the user log in.
            if not self.creds or not self.creds.valid:
                print("WIP - unable to get code to refresh access token to work\n\tthrows exception")
                '''
                if self.creds and self.creds.expired and self.creds.refresh_token:
                    self.creds.refresh(Request())
                    print("Google project access token refreshed")
                else:
                '''
                flow = InstalledAppFlow.from_client_secrets_file(self.credentialsPath, self.SCOPE_RW)
                self.creds = flow.run_local_server(port=0)
                print("Google project - new access token received")
                # Save the credentials for the next run
                with open(self.GoogleTokenPath, 'w') as self.token:
                    self.token.write(self.creds.to_json())
            
            '''
            With valid credentials
            '''
            self.sheetService = build('sheets', 'v4', credentials=self.creds)
            self.googleSheet = self.sheetService.spreadsheets()
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    
        return self.googleSheet

            
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
        
        https://github.com/googleapis/google-api-python-client/blob/main/docs/start.md
            
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

'''
object (SpreadsheetProperties)
object (Sheet)
object (NamedRange)
object (DeveloperMetadata)
object (DataSource)
object (DataSourceRefreshSchedule)
'''
class SpreadsheetProperties():
    def __init__(self):
        self.title = ""
        self.locale = ""
        self.autoRecalc = ""
        self.timeZone = ""
        self.cellFormat = ""
        self.iterativeCalculationSettings = ""
        self.spreadsheetTheme = ""
        self.importFunctionsAllowed = False
        self.jsonStr = ""
        return
    
    def __str__(self):
        '''
        {
          "title": string,
          "locale": string,
          "autoRecalc": enum (RecalculationInterval),
          "timeZone": string,
          "defaultFormat": {
            object (CellFormat)
          },
          "iterativeCalculationSettings": {
            object (IterativeCalculationSettings)
          },
          "spreadsheetTheme": {
            object (SpreadsheetTheme)
          },
          "importFunctionsExternalUrlAccessAllowed": boolean
        }
        '''
        self.jsonStr = "{"
        self.jsonStr = self.jsonStr + '"title":' + self.title + ','
        self.jsonStr = self.jsonStr + '"locale":' + self.locale + ','
        self.jsonStr = self.jsonStr + 'autoRecalc": enum (' + self.autoRecalc + '),'
        self.jsonStr = self.jsonStr + '"timeZone":' + self.timeZone + ','
        self.jsonStr = self.jsonStr + '"defaultFormat": {object (' + self.cellFormat + ')},'
        self.jsonStr = self.jsonStr + '"iterativeCalculationSettings": {object (' + self.iterativeCalculationSettings + ')},'
        self.jsonStr = self.jsonStr + '"spreadsheetTheme": {object (' + self.spreadsheetTheme + ')},'
        self.jsonStr = self.jsonStr + '"importFunctionsExternalUrlAccessAllowed":{}'.format(self.importFunctionsAllowed)
        self.jsonStr = self.jsonStr + "}"
        
        return self.jsonStr

'''
class Sheet():
    def __init__(self):
        return

class NamedRange():
    def __init__(self):
        return

class DeveloperMetadata():
    def __init__(self):
        return

class DataSource():
    def __init__(self):
        return

class DataSourceRefreshSchedule():
    def __init__(self):
        return
'''
    
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
        
            return dfVals
    
        except Exception:
            #exc_info = sys.exc_info()
            #exc_str = exc_info[1].args[0]
            sys.exit(exc_txt)
    
    def updateGoogleSheet(self, spreadsheetID, writeRange, updateData):
        ''' update Google sheet data '''
        exc_txt = "\nAn exception occurred - unable to write data to Google sheet"
        try:
            self.openGoogleSheetService()

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
            
            return result.get('updatedCells')
        
        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[1])
            sys.exit()
    
    def readSheetMetadata(self, spreadsheetID):
        ''' read metadata for Google sheet '''
        '''
        {
        'spreadsheetId': 'xxx', 
        'properties': {
            'title': 'Options Trading', 
            'locale': 'en_US', 
            'autoRecalc': 'ON_CHANGE', 
            'timeZone': 'America/New_York', 
            'defaultFormat': {
                'backgroundColor': {'red': 1, 'green': 1, 'blue': 1}, 
                'padding': {'top': 2, 'right': 3, 'bottom': 2, 'left': 3}, 
                'verticalAlignment': 'BOTTOM', 
                'wrapStrategy': 'OVERFLOW_CELL', 
                'textFormat': {
                    'foregroundColor': {}, 
                    'fontFamily': 'arial,sans,sans-serif', 
                    'fontSize': 10, 
                    'bold': False, 
                    'italic': False, 
                    'strikethrough': False, 
                    'underline': False, 
                    'foregroundColorStyle': {
                        'rgbColor': {}
                        }
                    }, 
                'backgroundColorStyle': {
                    'rgbColor': {'red': 1, 'green': 1, 'blue': 1}
                    }
                }, 
            'spreadsheetTheme': {
                'primaryFontFamily': 'Arial', 
                'themeColors': [
                    {'colorType': 'TEXT', 'color': {'rgbColor': {}}}, 
                    {'colorType': 'BACKGROUND', 'color': {'rgbColor': {'red': 1, 'green': 1, 'blue': 1}}}, 
                    {'colorType': 'ACCENT1', 'color': {'rgbColor': {'red': 0.25882354, 'green': 0.52156866, 'blue': 0.95686275}}}, 
                    {'colorType': 'ACCENT2', 'color': {'rgbColor': {'red': 0.91764706, 'green': 0.2627451, 'blue': 0.20784314}}}, 
                    {'colorType': 'ACCENT3', 'color': {'rgbColor': {'red': 0.9843137, 'green': 0.7372549, 'blue': 0.015686275}}}, 
                    {'colorType': 'ACCENT4', 'color': {'rgbColor': {'red': 0.20392157, 'green': 0.65882355, 'blue': 0.3254902}}}, 
                    {'colorType': 'ACCENT5', 'color': {'rgbColor': {'red': 1, 'green': 0.42745098, 'blue': 0.003921569}}}, 
                    {'colorType': 'ACCENT6', 'color': {'rgbColor': {'red': 0.27450982, 'green': 0.7411765, 'blue': 0.7764706}}}, 
                    {'colorType': 'LINK', 'color': {'rgbColor': {'red': 0.06666667, 'green': 0.33333334, 'blue': 0.8}}}
                ]
                }
            }, 
        'sheets': [
            {'properties': {
                'sheetId': 1118594284, 
                'title': 'For Review', 
                'index': 0, 
                'sheetType': 'GRID', 
                'gridProperties': {
                    'rowCount': 4541, 
                    'columnCount': 41, 
                    'frozenRowCount': 7
                    }
                }, 
                'conditionalFormats': [
                    {'ranges': [
                        {
                        'sheetId': 1118594284, 
                        'startRowIndex': 0, 
                        'endRowIndex': 4, 
                        'startColumnIndex': 6, 
                        'endColumnIndex': 7
                        }, {'sheetId': 1118594284, 'startRowIndex': 4, 'endRowIndex': 5, 'startColumnIndex': 8, 'endColumnIndex': 9}, {'sheetId': 1118594284, 'startRowIndex': 5, 'endRowIndex': 7, 'startColumnIndex': 6, 'endColumnIndex': 7}, {'sheetId': 1118594284, 'startRowIndex': 7, 'endRowIndex': 4541, 'startColumnIndex': 5, 'endColumnIndex': 6}], 'booleanRule': {'condition': {'type': 'NUMBER_GREATER', 'values': [{'userEnteredValue': '250'}]}, 'format': {'backgroundColor': {'red': 1, 'green': 0.6}, 'backgroundColorStyle': {'rgbColor': {'red': 1, 'green': 0.6}}}}}, {'ranges': [{'sheetId': 1118594284, 'startRowIndex': 7, 'endRowIndex': 4541, 'startColumnIndex': 11, 'endColumnIndex': 12}], 'booleanRule': {'condition': {'type': 'NUMBER_GREATER_THAN_EQ', 'values': [{'userEnteredValue': '25'}]}, 'format': {'backgroundColor': {'green': 1}, 'backgroundColorStyle': {'rgbColor': {'green': 1}}}}}]}, {'properties': {'sheetId': 1364636417, 'title': '01-03', 'index': 1, 'sheetType': 'GRID', 'gridProperties': {'rowCount': 969, 'columnCount': 53}}, 'merges': [{'sheetId': 1364636417, 'startRowIndex': 0, 'endRowIndex': 1, 'startColumnIndex': 0, 'endColumnIndex': 39}, {'sheetId': 1364636417, 'startRowIndex': 5, 'endRowIndex': 6, 'startColumnIndex': 0, 'endColumnIndex': 39}, {'sheetId': 1364636417, 'startRowIndex': 8, 'endRowIndex': 9, 'startColumnIndex': 0, 'endColumnIndex': 39}, {'sheetId': 1364636417, 'startRowIndex': 14, 'endRowIndex': 15, 'startColumnIndex': 0, 'endColumnIndex': 39}, {'sheetId': 1364636417, 'startRowIndex': 17, 'endRowIndex': 18, 'startColumnIndex': 0, 'endColumnIndex': 39}], 'conditionalFormats': [{'ranges': [{'sheetId': 1364636417, 'startRowIndex': 1, 'endRowIndex': 5, 'startColumnIndex': 6, 'endColumnIndex': 7}, {'sheetId': 1364636417, 'startRowIndex': 2, 'endRowIndex': 5, 'startColumnIndex': 5, 'endColumnIndex': 6}, {'sheetId': 1364636417, 'startRowIndex': 6, 'endRowIndex': 8, 'startColumnIndex': 6, 'endColumnIndex': 7}, {'sheetId': 1364636417, 'startRowIndex': 9, 'endRowIndex': 14, 'startColumnIndex': 6, 'endColumnIndex': 7}, {'sheetId': 1364636417, 'startRowIndex': 10, 'endRowIndex': 14, 'startColumnIndex': 5, 'endColumnIndex': 6}, {'sheetId': 1364636417, 'startRowIndex': 15, 'endRowIndex': 17, 'startColumnIndex': 6, 'endColumnIndex': 7}, {'sheetId': 1364636417, 'startRowIndex': 16, 'endRowIndex': 17, 'startColumnIndex': 5, 'endColumnIndex': 6}, {'sheetId': 1364636417, 'startRowIndex': 18, 'endRowIndex': 19, 'startColumnIndex': 6, 'endColumnIndex': 7}, {'sheetId': 1364636417, 'startRowIndex': 20, 'endRowIndex': 21, 'startColumnIndex': 5, 'endColumnIndex': 6}], 'booleanRule': {'condition': {'type': 'NUMBER_GREATER', 'values': [{'userEnteredValue': '250'}]}, 'format': {'backgroundColor': {'red': 1, 'green': 0.6}, 'backgroundColorStyle': {'rgbColor': {'red': 1, 'green': 0.6}}}}}, {'ranges': [{'sheetId': 1364636417, 'startRowIndex': 2, 'endRowIndex': 5, 'startColumnIndex': 12, 'endColumnIndex': 13}, {'sheetId': 1364636417, 'startRowIndex': 10, 'endRowIndex': 14, 'startColumnIndex': 12, 'endColumnIndex': 13}, {'sheetId': 1364636417, 'startRowIndex': 12, 'endRowIndex': 14, 'startColumnIndex': 11, 'endColumnIndex': 12}, {'sheetId': 1364636417, 'startRowIndex': 16, 'endRowIndex': 17, 'startColumnIndex': 12, 'endColumnIndex': 13}, {'sheetId': 1364636417, 'startRowIndex': 20, 'endRowIndex': 21, 'startColumnIndex': 11, 'endColumnIndex': 12}], 'booleanRule': {'condition': {'type': 'NUMBER_GREATER_THAN_EQ', 'values': [{'userEnteredValue': '25'}]}, 'format': {'backgroundColor': {'green': 1}, 'backgroundColorStyle': {'rgbColor': {'green': 1}}}}}]}, {'properties': {'sheetId': 52376853, 'title': '20241006 210200', 'index': 2, 'sheetType': 'GRID', 'gridProperties': {'rowCount': 1000, 'columnCount': 26}}}, {'properties': {'sheetId': 1094591553, 'title': '20241007 074657', 'index': 3, 'sheetType': 'GRID', 'gridProperties': {'rowCount': 1000, 'columnCount': 26}}}, {'properties': {'sheetId': 1104673250, 'title': '20241007 075633', 'index': 4, 'sheetType': 'GRID', 'gridProperties': {'rowCount': 1000, 'columnCount': 26}}}, {'properties': {'sheetId': 301106324, 'title': '20241007 080157', 'index': 5, 'sheetType': 'GRID', 'gridProperties': {'rowCount': 1000, 'columnCount': 26}}}, {'properties': {'sheetId': 1985521564, 'title': '20241007 080459', 'index': 6, 'sheetType': 'GRID', 'gridProperties': {'rowCount': 1000, 'columnCount': 26}}}, 
            {'properties': {
                'sheetId': 1739849692, 
                'title': '20241007 080637', 
                'index': 7, 
                'sheetType': 'GRID', 
                'gridProperties': {
                    'rowCount': 1000, 
                    'columnCount': 26
                    }
                }
            }], 
        'spreadsheetUrl': 'https://docs.google.com/spreadsheets/d/17UzJLEI9ThNY4_Ulg3szPYRV3ArmcGERt_nijhZpRT0/edit'
        }
        '''
        exc_txt = "\nAn exception occurred attempting access the workbook metadata"
        try:
            self.sheets_metadata = self.googleSheet.get(spreadsheetId=spreadsheetID).execute()

        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[1])
            sys.exit()
    
    def deleteGoogleSheetTab(self, spreadsheetID, sheetTitle):
        ''' delete a tab from a Google workbook '''
        exc_txt = "\nAn exception occurred attempting to delete tab {}".format(sheetTitle)
        try:
            self.readSheetMetadata(spreadsheetID)
            for sheet in self.sheets_metadata['sheets']:
                if sheet['properties']['title'] == sheetTitle:
                    deleteSheetRequest = {"sheetId" : sheet['properties']['sheetId']}
                    self.batchRequests.append({"deleteSheet" : deleteSheetRequest})
                    requestBody = {"requests" : self.batchRequests}
                    #print("\nbatchUpdate requestBody:\n\t{}".format(requestBody))
                    result = self.googleSheet.batchUpdate(spreadsheetId=spreadsheetID, body=requestBody).execute()
                    self.batchRequests.clear()
                    break

        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[1])
            sys.exit()
            
    def addGoogleSheet(self, spreadsheetID, sheetTitle):
        ''' update Google workbook - add sheet '''
        exc_txt = "\nAn exception occurred attempting to add a new sheet {}".format(sheetTitle)
        try:
            self.deleteGoogleSheetTab(spreadsheetID, sheetTitle)            
            addSheetRequest = {"properties" : {"title": sheetTitle}}
            self.batchRequests.append({"addSheet" : addSheetRequest})
            requestBody = {"requests" : self.batchRequests}
            #print("\nbatchUpdate requestBody:\n\t{}".format(requestBody))
            result = self.googleSheet.batchUpdate(spreadsheetId=spreadsheetID, body=requestBody).execute()
            self.batchRequests.clear()
            
            return

        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[1])
            sys.exit()
    
    def clearGoogleSheet(self, spreadsheetID, tabName, clearedRange):
        ''' create a new Google workbook '''
        exc_txt = "\nAn exception occurred attempting to clear a sheet"
        #print("SheetID: {}, tab: {}, range: {}".format(spreadsheetID, tabName, clearedRange))
        try:
            #https://developers.google.com/sheets/api/reference/rest/v4/spreadsheets.values/clear
            result = self.googleSheet.values().clear(spreadsheetId=spreadsheetID, range=tabName + "!" + clearedRange).execute()
            return

        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            for ndx in range (len(exc_info)):
                print(exc_info[ndx])
            sys.exit()
    
    def createGoogleWorkbook(self):
        ''' create a new Google workbook '''
        print("createGoogleWorkbook is WIP - new workbook is created in the home Drive folder")
        exc_txt = "\nAn exception occurred attempting to create a new workbook"
        try:
            t = SpreadsheetProperties()
            print(t)
            
            requestBody = {}
            result = self.googleSheet.create(body=requestBody).execute()
            
            return

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
    
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
                #print("WIP - unable to get code to refresh access token to work\n\tthrows exception")
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
    
            return self.googleSheet

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            sys.exit(exc_txt + "\n\t" + exc_str)
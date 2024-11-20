'''
Created on Oct 22, 2024

@author: brian

Object class to build and process the user interface for the daily process

Uses the json file ExperimentalDailyProcess to build the UI widgets
'''
import os
import multiprocessing
import sys
import json
import re

'''
import tkinter as tk
import tkinter
'''
from tkinter import *
from tkinter import ttk

from configuration import get_ini_data
from configuration import read_config_json

from TrainingDataAndResults import Data2Results as d2r

from configuration import read_processing_network_json
from configuration_graph import build_configuration_graph

from executeProcessingNodes import executeProcessingNodes
from pretrainedModels import rnnCategorization, rnnPrediction

from Workbooks import investments, optionTrades, mlEvaluations
from MarketData import MarketData, BasicMarketDataArchive, EnrichedMarketDataArchive

class processParameter(object):
    '''
    classdocs
    '''

    def __init__(self, controlFrame, parameterName, defaultValue, rowNum):
        '''
        Constructor
        '''        
        self.controlFrame = controlFrame
        self.parameterName = parameterName
        self.defaultValue = defaultValue
        self.rowNum = rowNum
        
        self.parameterNameWidget = ttk.Label(self.controlFrame, text=self.parameterName)
        self.parameterNameWidget.grid(column=0, row=rowNum, sticky=(W))
        
        self.parameterValue = StringVar()
        self.parameterValueWidget = ttk.Entry(self.controlFrame, textvariable=self.parameterValue, width=80)
        self.parameterValueWidget.grid(column=1, row=rowNum, sticky=(W))
        self.setValue(self.defaultValue)
        
    ''' Class methods '''
    def setValue(self, valueStr):
        self.parameterValueWidget.delete(0, 'end')
        self.parameterValueWidget.insert(0, valueStr)
        
    ''' Class data '''
    @property
    def controlFrame(self):
        return self._controlFrame
    
    @controlFrame.setter
    def controlFrame(self, controlFrame):
        self._controlFrame = controlFrame

    @property
    def parameterValueWidget(self):
        return self._parameterValueWidget
    
    @parameterValueWidget.setter
    def parameterValueWidget(self, parameterValueWidget):
        self._parameterValueWidget = parameterValueWidget

    @property
    def defaultValue(self):
        return self._defaultValue
    
    @defaultValue.setter
    def defaultValue(self, defaultValue):
        self._defaultValue = defaultValue

    @property
    def parameterValue(self):
        return self._parameterValue
    
    @parameterValue.setter
    def parameterValue(self, parameterValue):
        self._parameterValue = parameterValue

    @property
    def parameterName(self):
        return self._parameterName
    
    @parameterName.setter
    def parameterName(self, parameterName):
        self._parameterName = parameterName


class DailyProcessFrame(object):
    '''
    classdocs
    '''

    def __init__(self, mainframe, colNdx, rowNdx):
        '''
        Constructor
        '''        
        self.layoutFrame = ttk.Frame(mainframe, borderwidth=2, relief='sunken')
        self.layoutFrame.grid(column=colNdx, row=rowNdx, sticky=(W, E, N))
        self.embeddedFrameCount = 0
        
    @property
    def layoutFrame(self):
        return self._layoutFrame
    
    @layoutFrame.setter
    def layoutFrame(self, layoutFrame):
        self._layoutFrame = layoutFrame

    @property
    def embeddedFrameCount(self):
        return self._embeddedFrameCount
    
    @embeddedFrameCount.setter
    def embeddedFrameCount(self, embeddedFrameCount):
        self._embeddedFrameCount = embeddedFrameCount

class DailyProcessControl(object):
    '''
    classdocs
    '''
    implementedProcesses = ["Investment Tracking", "Options", "Enriched data archive", "Market data archive", \
                            "MACD Trend", "Bollinger Bands", \
                            "Networks", "ExperimentaL Network"]
    plannedProcesses = ["MultiModelNetwork"]

    def processOnOff(self):
        if self.executeProcessFlag.get() == "run":
            if self.processName in self.implementedProcesses:
                pass
            elif self.processName in self.plannedProcesses:
                print("{} is not yet implemented".format(self.processName))
            else:
                print("{} is not recognized".format(self.processName))
        return
        
    def __init__(self, processControlSpecification, framesGrid):
        '''
        Constructor
        '''        
        try:
            exc_txt = "\nAn exception occurred building the user interface controls for a process"
            processRowNdx = int(processControlSpecification["groupRow"]) - 1
            processColNdx = int(processControlSpecification["groupColumn"]) - 1
            self.processName = processControlSpecification["name"]
            self.processDescription = processControlSpecification["Description"]
            
            self.parentProcessFrame = framesGrid[processRowNdx][processColNdx]
            self.controlFrame = ttk.Frame(self.parentProcessFrame.layoutFrame)
            self.controlFrame.grid(column=0, row=self.parentProcessFrame.embeddedFrameCount, sticky=(W, E))
            self.parentProcessFrame.embeddedFrameCount = self.parentProcessFrame.embeddedFrameCount + 1
            self.processParameterList = []
            
            exc_txt = "\nAn exception occurred building the user interface controls for - {}".format(self.processName)
            
            DESCRIPTION_AND_SELECTION_ROW = 0
            DESCRIPTION_COL = 0
            SELECTION_COL = 1
            CONTROLCOL = 0
            DEFAULTCOL = 1
            controlRow = 2
            
            ''' Process name label '''
            ttk.Label(self.controlFrame, text=self.processName).grid(column=DESCRIPTION_COL, row=DESCRIPTION_AND_SELECTION_ROW, sticky=(W))
            
            ''' Process run check box '''
            self.executeProcessFlag = StringVar()
            ttk.Checkbutton(self.controlFrame, text='Execute', \
                            command=self.processOnOff, variable=self.executeProcessFlag, \
                            onvalue='run', offvalue='pass').grid(column=SELECTION_COL, row=DESCRIPTION_AND_SELECTION_ROW, sticky=(W))
            
            ''' Build UI for each configurable  for the process in the configuration json '''
            if "controls" in processControlSpecification:
                for control in processControlSpecification["controls"]:
                    ''' extract control parameter name and default value '''
                    for parameterName, defaultText in control.items():
                        pass
                    procParm = processParameter(self.controlFrame, parameterName, defaultText, controlRow)
                    self.processParameterList.append(procParm)
                    controlRow = controlRow + 1
                    
            return
        
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

    @property
    def processName(self):
        return self._processName
    
    @processName.setter
    def processName(self, processName):
        self._processName = processName

    @property
    def processDescription(self):
        return self._processDescription
    
    @processDescription.setter
    def processDescription(self, processDescription):
        self._processDescription = processDescription

    @property
    def executeProcessFlag(self):
        return self._executeProcessFlag
    
    @executeProcessFlag.setter
    def executeProcessFlag(self, executeProcessFlag):
        self._executeProcessFlag = executeProcessFlag

    @property
    def processParameterList(self):
        return self._processParameterList
    
    @processParameterList.setter
    def processParameterList(self, processParameterList):
        self._processParameterList = processParameterList

class DailyProcessUI(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        try:
            ''' Find local file directories '''
            exc_txt = "\nAn exception occurred - unable to identify localization details"
            localDirs = get_ini_data("LOCALDIRS")
            gitdir = localDirs['git']
        
            ''' read application specific configuration file '''
            exc_txt = "\nAn exception occurred - unable to access process configuration file"
            self.config_data = get_ini_data("DAILY_PROCESS")
            self.appConfig = read_config_json(gitdir + self.config_data['config'])
            
            ''' ============== experimentation json config file ============== '''
            #self.appConfig = read_config_json(gitdir +  '\\chandra\\unit_test\\ExperimentalDailyProcess.json')
            ''' ============================================================== '''
            #print("appConfig file {}\n{}".format(self.config_data['config'], self.appConfig))
            
            ''' =============== build user interface based on configuration json file =========== '''
            self.root=Tk()
            self.root.title(self.appConfig['WindowTitle'])
            self.colCount = len(self.appConfig['groupColumns'])
            self.rowCount = self.appConfig['groupRows']
            
            self.colDecriptionLabelRow = 1
            self.doitButtonRow = self.rowCount + 2
            self.colDescription = []
            for desc in self.appConfig['groupColumns']:
                self.colDescription.append(desc)

            self.processControlList = []
            self.buildUI()
            return
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    def saveMachineLearningSignal(self, controls, signals):
        exc_txt = "\nAn exception occurred - saving machine learning signals"
        signalWorkbook = mlEvaluations()
        signalWorkbook.processName = controls.processName
        
        try:
            for param in controls.processParameterList:
                paramValue = param.parameterValue.get()
                print("control parameter - {} set to {}".format(param.parameterName, paramValue))
                
                if param.parameterName == "gsheet":
                    gsheet = paramValue
                    signalWorkbook.sheetID = gsheet
                    
                if param.parameterName == "header range":
                    signalWorkbook.headerRange = paramValue
                    
                if param.parameterName == "data range":
                    signalWorkbook.dataRange = paramValue
                    
                if param.parameterName == "Outputs":
                    outputs = outputs = re.split(',',paramValue)
                    
                if param.parameterName == "features":
                    features = paramValue
                    signalWorkbook.features = re.split(',', features)
                    
            signalWorkbook.archiveSignals(signals, outputs)
            return
    
        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def trainKerasModel(self, controls):
        exc_txt = "\nAn exception occurred - training model based on script {}".format(controls.processName)
        try:
            print("WIP =============\ntrainKerasModel is not yet implemented. Script {}".format(controls.processName))
            
            for param in controls.processParameterList:
                paramValue = param.parameterValue.get()
                print("control parameter - {} set to {}".format(param.parameterName, paramValue))
                
                if param.parameterName == "run":
                    run = paramValue
                    print("running network: {}".format(run))
                    
            localdirs = get_ini_data("LOCALDIRS")
            gitdir = localdirs['git']
            aiwork = localdirs['aiwork']
            config_data = get_ini_data("CHANDRA")
            json_config = read_config_json(gitdir + config_data['config'])

            d2ra = d2r()
        
            configurationDir = gitdir + json_config['processNetDir']
            d2ra.configurationFileDir = configurationDir
        
            configurationFile = run + ".json"
            #configurationFile = json_config['processNetConfiguration']
            processing_json = read_processing_network_json(configurationDir + configurationFile)
            d2ra.configurationFile = processing_json
            
            build_configuration_graph(d2ra, processing_json)
            executeProcessingNodes(d2ra)
            
            return
    
        except Exception:
            print(exc_txt)
            exc_info = sys.exc_info()
            if len(exc_info) > 1:
                print(exc_info[1].args[0])
            sys.exit()

    def useTrainedRNNCategorizationModel(self, controls):
        try:
            exc_txt = "\nAn exception occurred - executing trained RNN categorization model"
            print("Use trained model for process {}".format(controls.processName))
            for param in controls.processParameterList:
                paramValue = param.parameterValue.get()
                print("control parameter - {} set to {}".format(param.parameterName, paramValue))
                
                if param.parameterName == "file":
                    modelFile = paramValue
                    
                if param.parameterName == "scaler":
                    scalerFile = paramValue
                
                if param.parameterName == "Outputs":
                    outputs = re.split(',',paramValue)
                    
                if param.parameterName == "timeSteps":
                    timeSteps = int(paramValue)
                    
                if param.parameterName == "threshold":
                    thresholdStrs = re.split(',',paramValue)
                    threshold = []
                    for ndx in range (len(thresholdStrs)):       
                        threshold.append(float(thresholdStrs[ndx]))
                    
                if param.parameterName == "features":
                    features = paramValue
                    features = re.split(',', features)
                    
                if param.parameterName == "featureFile":
                    inputFileSpec = paramValue
                    inputFileSpec = re.split(',', inputFileSpec)
                    
                name = controls.processDescription
            
            signals = rnnCategorization(name, modelFile, inputFileSpec, features, \
                                     scalerFile, timeSteps, outputs, signalThreshold=threshold)
            if len(signals) > 0:
                print("{} signals returned".format(len(signals)))
                self.saveMachineLearningSignal(controls, signals)
            return
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    def useTrainedRNNPredictionModel(self, controls):
        try:
            exc_txt = "\nAn exception occurred - executing trained RNN prediction model"
            print("Use trained model for process {}".format(controls.processName))
            
            for param in controls.processParameterList:
                paramValue = param.parameterValue.get()
                print("control parameter - {} set to {}".format(param.parameterName, paramValue))
                
                if param.parameterName == "file":
                    modelFile = paramValue
                    
                if param.parameterName == "scaler":
                    scalerFile = paramValue
                
                if param.parameterName == "Outputs":
                    outputs = paramValue
                    
                if param.parameterName == "timeSteps":
                    timeSteps = int(paramValue)
                    
                if param.parameterName == "threshold":
                    threshold = float(paramValue)
                    
                if param.parameterName == "features":
                    features = paramValue
                    features = re.split(',', features)
                    
                if param.parameterName == "featureFile":
                    inputFileSpec = paramValue
                    inputFileSpec = re.split(',', inputFileSpec)
                    
                name = controls.processDescription
            signals = rnnPrediction(name, modelFile, inputFileSpec, features, \
                                     scalerFile, timeSteps, outputs, signalThreshold=threshold)
            if len(signals) > 0:
                self.saveMachineLearningSignal(controls, signals)
                
            return
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    def marketDataArchive(self):
        exc_txt = "\nAn exception occurred - daily market data process"
        try:
            print("updating market data")
            investmentSheet = investments()
            symbolList = investmentSheet.stockInformationSymbols()
            #symbolList = ["AAPL", "C", "INTC"]
            for symbol in symbolList:
                exc_txt = "\nAn exception occurred - daily market data process - symbol: {}".format(symbol)
                marketData = BasicMarketDataArchive(symbol)
                marketData.updateLocalArchive()
            return 
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

    def enrichMarketDataArchive(self):
        exc_txt = "\nAn exception occurred - calculating derived data"
        
        try:
            print("calculating derived market data")
            investmentSheet = investments()
            symbolList = investmentSheet.stockInformationSymbols()
            enrichedMarketData = EnrichedMarketDataArchive(symbolList)
            pass
        
            return 

        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
            
    def markToMarket(self):
        exc_txt = "\nAn exception occurred - mark to market process"
        try:
            investmentSheet = investments()
            investmentSheet.markToMarket()
            return 
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
        
    def optionTradeProcess(self, controls):
        exc_txt = "\nAn exception occurred - daily call option process"
        try:
            optionChains = optionTrades()

            filterList = []
            ''' set list of supported filters '''
            print("WIP ============\n\tSupported filter list should be provided by the class")
            filterNames = ["delta", "max cover", "min gain APY", "min gain $", "dividend date", "earnings date", "option quantity", "limit price"]
            for param in controls.processParameterList:
                if param.parameterName in filterNames:
                    print("param: {}, default: {} changed to {}". \
                          format(param.parameterName, param.defaultValue, param.parameterValue.get()))

                    filterNameCtrl = param.parameterValue.get()
                    filterNameJson = filterNameCtrl.replace("'", '"')
                    filterNameJson = json.loads(filterNameJson)
                    condition = filterNameJson["test"]
                    threshold = filterNameJson["threshold"]
                    filterList.append({"dataElement":param.parameterName, "condition":condition, "threshold":threshold})
                    
            ''' set option search parameters '''
            strikeCount = 25
            strikeRange = "OTM"
            daysToExpiration = 90
            optionChains.findPotentialOptionTrades(strikeCount=strikeCount, strikeRange=strikeRange, daysToExpiration=daysToExpiration, \
                                                   filterList=filterList)
            return
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)
    
    def dailyProcessUIdoit(self):
        #print("do it button pressed")
        for proc in self.processControlList:
            if proc.executeProcessFlag.get() == "run":
                print("execute process {} - {}".format(proc.processName, proc.processDescription))
                
                '''
                for param in proc.processParameterList:
                    print("param: {}, default: {} changed to {}". \
                          format(param.parameterName, param.defaultValue, param.parameterValue.get()))
                '''
                
                if proc.processName == "Investment Tracking":
                    self.markToMarket()
                elif proc.processName == "Options":
                    self.optionTradeProcess(proc)
                elif proc.processName == "Market data":
                    self.marketDataArchive()
                elif proc.processName == "Enriched data archive":
                    self.enrichMarketDataArchive()
                elif proc.processName == "Networks":
                    ''' ML networks that have been debugged (not necessarily optimized) '''
                    self.trainKerasModel(proc)
                elif proc.processName == "ExperimentaL Network":
                    ''' ML networks currently under development '''
                    self.trainKerasModel(proc)
                    
                elif proc.processName == "Bollinger Bands":
                    self.useTrainedRNNPredictionModel(proc)
                    
                elif proc.processName == "MACD Trend":
                    self.useTrainedRNNCategorizationModel(proc)
                    
                elif proc.processName == "MultiModelNetwork":
                    self.trainKerasModel(script = proc.processName)
                    
                else:
                    print("{} is not recognized".format(proc.processName))
                
                print("Process {} - {}, complete".format(proc.processName, proc.processDescription))
        return
    
    def buildFrames(self):
        ''' ================== create window frames for top level placement ============ 
        Creates self.framesGrid - a list of lists self.framesGrid[row][col]
        Adds a label widget to each frame for the frame title from the configuration file
        Adds a button frame at the bottom of the main frame for a button to execute the selected processes
        '''
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        ''' Display frame column description '''
        self.descriptionFrame = []
        self.descriptionLabel = []
        for colNdx in range(self.colCount):
            descFrame = ttk.Frame(self.mainframe)
            descFrame.grid(column=colNdx, row=0)
            descLabel = ttk.Label(descFrame, text=self.colDescription[colNdx], borderwidth=2).grid(row=0, sticky=(N))
            self.descriptionFrame.append(descFrame)
            self.descriptionLabel.append(descLabel)
        
        ''' frames within frames '''
        self.framesGrid = []
        for rowNdx in range(self.rowCount):
            framesRow = []
            for colNdx in range(self.colCount):
                #processFrame = DailyProcessFrame(self.mainframe, colNdx, rowNdx + 1, self.colDescription[colNdx])
                processFrame = DailyProcessFrame(self.mainframe, colNdx, rowNdx + 1)
                framesRow.append(processFrame)
            self.framesGrid.append(framesRow)
        
        ''' Display a button to start execution of selected processes '''
        doitButton = ttk.Button(self.mainframe, text="Perform selected processes", command=self.dailyProcessUIdoit)
        doitButton.grid(column=0, row=self.doitButtonRow, columnspan=self.colCount, rowspan=1)
            
        return 
    
    def buildUI(self):
        ''' display a user interface to solicit run time selections '''
        try:
            exc_txt = "\nAn exception occurred building the user interface"
            self.buildFrames()
    
            ''' Build UI for each process in the configuration json '''
            exc_txt = "\nAn exception occurred building the process specific user interface"
            for process in self.appConfig["processes"]:
                exc_txt = "\nAn exception occurred building the process specific user interface for {}".format(process)
                procControl = DailyProcessControl(process, self.framesGrid)
                self.processControlList.append(procControl)
                
            ''' =================== Interact with user =================== '''
            for child in self.mainframe.winfo_children(): 
                child.grid_configure(padx=5, pady=5)
            
            self.root.mainloop()
            return
    
        except Exception:
            exc_info = sys.exc_info()
            exc_str = exc_info[1].args[0]
            exc_txt = exc_txt + "\n\t" + exc_str
            sys.exit(exc_txt)

    @property
    def processControlList(self):
        return self._processControlList
    
    @processControlList.setter
    def processControlList(self, processControlList):
        self._processControlList = processControlList


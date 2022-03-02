import optparse
# import configparser
# import json
# import datetime
# import logging
# import os, traceback, sys
# 
# 
from riskmodels.runJsonCommands import *
# 
# 
# from lib.common import modelDbTransfer as modelDbTransfer1
# from lib.common.assetAddtitionNonDjango import processBulk
# from lib.common import macNonDjango
# from lib.common.contentCentralNonDjango import updateFutureRmsRecords
# from lib.common.contentCentralNonDjango import getModelDbConfig_new
# from lib.common.contentCentralNonDjango import getMarketDbConfig_new
# from lib.common.contentCentralNonDjango import runQaReports#NP: remove this line later
# from lib.common.contentCentralNonDjango import runQaReportsForEquitiesAndMutualFunds
# from lib.common.contentCentralNonDjango import emailResults
# 
# from marketdb import transfer as marketDbTransfer
# from marketdb import MarketID
# # from marketdb.qa import bulkqareports
# 
# from marketdb import ETFHelperUtility2
# 
# from macpy import CurveGenerator
# from riskmodels import Connections
# from riskmodels.creatermsid import createRmsIssueRecords
# from riskmodels import transfer as modelDbTransfer
# from riskmodels import Utilities
# 
# from phoenix.datagateway.environment import MacDatabase
# 
# import subprocess

logger = logging.getLogger('RiskModels.jobcontrol.runJsonCommands')
logger.setLevel(logging.DEBUG)
testOnly=True
 
# def doMarketDbTransfer(sid, axiomaIds, dates, commit=False, partOfTransaction=False,countries=None, 
#                sectionList=[], rates=None, marketDb=None, axiomaDb=None, 
#                config=None, assetType='equity', indexVariant=''):
#     logger.debug('in marketDbTransfer.doTransfer(): sid %s, axiomaIds %s, sectionList %s, dates %s, commit %s, partOfTransaction %s,countries %s',sid, axiomaIds, sectionList, dates, commit, partOfTransaction,countries)
#     
#     config = getMarketDbConfig_new(sid)
#     config.set('DEFAULT','axioma-ids', axiomaIds)
#     config.set('DEFAULT', 'dates',dates)
#     config.set('DEFAULT','asset-type', assetType)
#     config.set('DEFAULT','index-variant', indexVariant)    
#        
#     for section in sectionList:
#         logger.debug('section: %s',section)
# 
#         if config.has_section(section):
#             options = Utilities.Struct()
#             global testOnly
#             options.testOnly=testOnly
#             
#             logger.debug('config: %s',str(config))        
#             logger.debug('going to run MarketDB.transfer for section %s for axioma-ids %s dates %s testOnly %s', section, config.get('DEFAULT', 'axioma-ids'),config.get('DEFAULT', 'dates'), testOnly )
# 
#             transferFunction = config.get(section, 'transfer')    
#                 
#             connections = Connections.Connections(config)
#             
#             evalStr="marketDbTransfer.%s(config, '%s', connections, options)" %(transferFunction,section)
#             logger.debug('Transferring data to marketdb: going to call transfer.py\'s  %s',evalStr) 
#             
# #             
#             if marketDb is not None:
#                 connections.marketDB.finalize()
#                 connections.marketDB = marketDb
#             if axiomaDb is not None:
#                 connections.axiomaDB.finalize()
#                 connections.axiomaDB = axiomaDb
#             proc = eval(evalStr) 
#             logger.debug('transfer for section %s finished; axioma-ids %s dates %s ', section, config.get('DEFAULT', 'axioma-ids'),config.get('DEFAULT', 'dates') )
#             
#         else:    
#             logger.warning('No "%s" section in config file. Skipping' % section) 
#          
#     return "doMarketDbTransfer finished"
# 
# 
# #NP  commit=False, partOfTransaction=False, modelDb=None, marketDb=None are included as parameters to be consitent with content central methods
# # def doModelDbTransfer(connections, issueId, subIssueIds, dates, sectionList=[]):
# def doModelDbTransfer(sid, issueId, subIssueIds, dates,  commit='placeholder; not used here', partOfTransaction='placeholder; not used here', sectionList=[], modelDb=None, marketDb=None ):
# 
#     
#     config = getModelDbConfig_new(sid)
#          
#     config.set('DEFAULT','issue-ids', issueId)
#     config.set('DEFAULT', 'dates',dates) 
#     config.set('DEFAULT', 'sub-issue-ids',subIssueIds) 
#     
#     config.set('DEFAULT', 'id-chunk', str(len(subIssueIds)))
#    
#     connections = Connections.Connections(config)   
#     
#     for section in sectionList:
#         logger.debug('section: %s',section)
#     
#         if config.has_section(section):
#             transferMethod=config.get(section, 'transfer')
#             options = Utilities.Struct()
#             options.testOnly=testOnly
#                 
#             methodToCall="modelDbTransfer.%s(config, section, connections,options)"%transferMethod
#             logger.debug('modeldb transfer.py\'s methodToCall: %s',methodToCall)
#             proc = eval(methodToCall)
#         else:
#             logger.warning('No "%s" section in config file. Skipping ' % str(section))
# 
#     return "doModelDbTransfer finished"
# 
# def getModelIdData(modelDb, axiomaIdList, startDate, endDate): 
#         #NP: moved to modeldbtransfer
#         """
#             get axiomaid data
#         """   
#         startdt=Utilities.parseISODate(startDate)
#         enddt=Utilities.parseISODate(endDate)
#         
#         query="""                 
#             select i.modeldb_id,s.sub_id, s.rmg_id, i.from_dt, i.thru_dt
#             from %(issueMapTable)s i 
#             left outer join sub_issue s on s.issue_id = i.modeldb_id 
#             where  marketdb_id=:axiomaId 
#             and i.from_dt <= :enddt and :startdt <= i.thru_dt
#         """
#         
#         results = list()
#         for axiomaId in axiomaIdList:
#             issueMapTable = 'issue_map'
#             if MarketID.isFuture(axiomaId):
#                 issueMapTable = 'future_issue_map'
#             
#             modelDb.dbCursor.execute(query%{'issueMapTable':issueMapTable}, axiomaId=axiomaId, startdt=startdt, enddt=enddt)
#             r = modelDb.dbCursor.fetchall()
#             if r:
#                 results.extend(r)
#         return results 
# 
# 
# def doModelDbTransferByAxiomaIdList(axiomaIdList, startDate, endDate, dates, sectionList=None):
#     #NP: almost a copy from modelDbTransfer.py; it would be nice  to refactor this part to reuse the code
#     errors=[]
#     global modelDbConnections 
#     connections = modelDbConnections
#     
#     result=getModelIdData(connections.modelDB, axiomaIdList, startDate, endDate)
#     returnsStr=''
# 
#     if len(result) > 0:
#         issueid=','.join(set([str(iss) for iss, sub, rmg, fromdt, thrudt in result]))
#         rmgid=','.join(set([str(rmg) for iss, sub, rmg,fromdt, thrudt in result]))
#         subissueid=','.join(set([str(sub) for iss, sub, rmg,fromdt,thrudt in result]))
#         
#         if len(issueid)>0 and len(rmgid)>0 and len(subissueid)> 0:
#                 returnsStr = doModelDbTransfer(connections.modelDB.dbConnection.dsn, issueid, subissueid,       
#                                                         dates, sectionList=sectionList)
#                 logger.info('********** Done with ModelDB transfers **************')
#     else:
#         # check to see if this is one of the cases that is the universal model transfer situation.
#         # if it is, pass in the axids to the method in the issueid column
#         allFactors = False
#         for axiomaId in axiomaIdList:
#             if MarketID.isUNFactor(axiomaId):
#                 allFactors = True
#             else:
#                 if allFactors:
#                     errors.append("not all the AxiomaIds are UN Factors: %s"%str(axiomaIdList))  
#                     allFactors = False
#                     return (returnsStr,errors)
#                 
#         if allFactors:       
#             result=getUnivModelIDs(marketDb, axiomaIdList, startDate, endDate)
#             if len(result) > 0:
#                 axids=','.join(set([str(axid) for axid in result]))
#                 if len(axids) > 0:
#                     returnsStr = doModelDbTransfer(connections.marketDB.dbConnection.dsn,axids, '',       
#                                                         dates, sectionList=sectionList)
#                     logger.info('********** Done with Univ ModelDB transfers **************')
# 
#         else:
#             errors.append("There is no model data for the given axioma-id")
#             errors.append("Cannot perform another model transfers")
#     
#     logger.debug('returnsStr: %s"' % str(returnsStr))
#     logger.debug('errors: %s"' % str(errors))     
#     return (returnsStr,errors)
#    
# def runJsonCommands(commandsJsonFileName,jobid=None,testOnlyParam=False):
#     with open(commandsJsonFileName) as file:
#         commands = json.load(file)
#     
#     global testOnly
#     testOnly = testOnlyParam
#     
#     try:
#         if (commandsJsonFileName.find("macCurvesHistory")>-1):
#             (counter,results) = executeCommands(None, None, commands=commands.get('generateCurvesCommands',[]))
#             
#             logger.debug("textresult in runJsonCommands() %s: %d, %s",commandsJsonFileName, counter, results)    
#             return  (counter,results, commands)
#         
#         elif (commandsJsonFileName.find("assetAdditionETF")>-1): 
#             (errors, textresult, commands) = runAssetAdditionCommands_new(
#                             commands.get('sid',None),
#                             commands.get('assetAdditionCommands',[]),
#                             commands.get('qaReportsCommands',[]),
#                             macdbId = commands.get('macdbId',None),
#                             jobid=jobid
#                             )
#             if len(errors) > 0:
#                 logger.error("Error during assetAddition runJsonCommands(): %s",errors)
#             logger.debug("textresult in runJsonCommands(): %s",textresult)    
#             return  (errors, textresult, commands)
#         
#         elif (commandsJsonFileName.find("assetAddition")>-1): 
#             (errors, textresult, commands) = runAssetAdditionCommands(
#                             commands.get('sid',None),
#                             commands.get('assetAdditionCommands',[]),
#                             commands.get('qaReportsCommands',[])
#                             )
#             if len(errors) > 0:
#                 logger.error("Error during assetAddition runJsonCommands(): %s",errors)
#             logger.debug("textresult in runJsonCommands(): %s",textresult)    
#             return  (errors, textresult, commands)
#         elif (commandsJsonFileName.find("macAssetCoverage")>-1): 
#             (errors, textresult, commands) = runMacAssetCoverageCommands(commands)
#             if len(errors) > 0:
#                 logger.error("Error during macAssetCoverage runJsonCommands(): %s",errors)
#             logger.debug("textresult in runJsonCommands(): %s",textresult)    
#             return  (errors, textresult, commands)
#         elif (commandsJsonFileName.lower().find("morningstar")>-1): 
#             (counter,results) = executeCommands(None, None, commands=commands.get('commands',[]))
#             logger.info("results in runJsonCommands() %s: %d, %s",commandsJsonFileName, counter, results) 
#             for returnCode in results:
#                 if returnCode !=0:
#                     raise Exception('an error occured during execution the following commands, please check log file: %s', str(commands))
#             return  (counter,results, commands)
#         
# 
#         else:
#             (msg, marketDbCounter, modelDbCounter, transferResults) = runAxiomaIdCommands(
#                             commands.get('sid',None),
#                             commands.get('updateLastTradingDateCommands',[]),
#                             commands.get('marketIdUpdateCommands',[]),
#                             commands.get('marketIdTransferCommands',[]),
#                             commands.get('modelIdUpdateCommands',[]),
#                             commands.get('modelIdTransferCommands',[]),
#                             commands.get('qaReportsCommands',[])
#                             )
#             return (msg, marketDbCounter, modelDbCounter, transferResults)
#     except Exception as e:
#         logger.exception("Error during runJsonCommands(): %s",e)
#         __raiseError__(e)     
#     
# 
# def executeCommands(db, connections, commands): 
#     
#     if connections is not None:
#         marketDb = connections.marketDB
#         axiomaDb = connections.axiomaDB
#         modelDb = connections.modelDB
#     else:
#         marketDb=axiomaDb=modelDb=None   
#         
#     counter=0
#     results=[]
#     
#     for command in commands:
#         logger.debug('in executeCommands(); command %s: ', command)
#         
# #         command=command.encode('utf8')
#         if command.startswith('###'):
#             if command.startswith('### ~~~'):
#                 logger.warning('the following command is commented out and will not be executed: %s',command)
#             continue
#         
#         if command.startswith('~~~'):
#             if marketDb is not None:
#                 marketDb.flushCaches()
#             logger.info('evaluating command: %s',command)
#             result = evalCommand(connections, command)
#             logger.info('evaluating command result: %s',result)
#             results.append(result)
#         else:
#             logger.info('executing command: %s',command)
#             if command.find('axiomadb.')>-1:
#                 logger.debug('axiomadb is used here: %s',command)
#                 axiomaDb.dbCursor.execute(command) 
#             else:
#                 db.dbCursor.execute(command)   
#             
#         counter=counter+1    
#         logger.debug('in executeCommands(), commands executed: %s , result: %s', commands, str(result))   
#         
#     return [counter,results]
# 
# def evalCommand(connections, command):
#     
#     if connections is not None:
#         marketDb = connections.marketDB
#         axiomaDb = connections.axiomaDB
#         modelDb = connections.modelDB
#     else:
#         marketDb=axiomaDb=modelDb=None   
#     
#     methodToExecute = command.replace('~~~','')  
#     methodToExecute = methodToExecute.replace('marketDbTransfer.doTransfer','doMarketDbTransfer')
#     methodToExecute = methodToExecute.replace('modelDbTransfer.doTransfer','doModelDbTransfer')
#     methodToExecute = methodToExecute.replace('modelDbTransfer.doTransferByAxiomaIdList','doModelDbTransferByAxiomaIdList')
#     methodToExecute = methodToExecute.replace('modelDbTransfer.doSync','modelDbTransfer1.doSync')
#     
#     logger.debug('in evalCommand, methodToExecute: %s',methodToExecute)    
# #    CurveGenerator.generateCurves('DEV', '2011-11-15', '2019-01-08', ['EEX:F7BY'],use_research_table=False)
#     result=eval(methodToExecute)
#     logger.debug('in evalCommand %s, result: %s',methodToExecute,result)
#     if result == 0:
#         pass
#     elif not result:
#         raise Exception('Error: an error occurred during execution of '+methodToExecute)
#     return result
# 
# def runAxiomaIdCommands(sid,
#                         updateLastTradingDateCommands,
#                         marketIdUpdateCommands,
#                         marketIdTransferCommands,
#                         modelIdUpdateCommands,
#                         modelIdTransferCommands,
#                         qaReportsCommands 
#                         ):
#     t1 = datetime.datetime.today()
#     marketDbCounter=0
#     modelDbCounter=0
#     transferResults=[]
#     
#     connections = Connections.Connections(getModelDbConfig_new(sid))
#     
#     
#     global marketDbConnections, modelDbConnections    
#     marketDbConnections=modelDbConnections=connections
#     
#     marketDB = connections.marketDB
#     axiomaDB = connections.axiomaDB
#     modelDB = connections.modelDB
#     
#     try:
#     
#         if len(updateLastTradingDateCommands)>0:
#             try:
#                 logger.debug('execute updateLastTradingDateCommands %s',updateLastTradingDateCommands)
#                 (marketDbCounter1,results) = executeCommands(marketDB,connections,updateLastTradingDateCommands)
#                 marketDbCounter += marketDbCounter1
#                 logger.debug( "updateLastTradingDateCommands executed successfully")    
#                 axiomaDB.flushCaches()
#                 marketDB.flushCaches()
#             except Exception as e :
#                 logger.exception("Error during updateLastTradingDateCommands in runAxiomaIdCommands(): %s",e)
#                 __raiseError__(e)
#                 
#         if len(marketIdUpdateCommands)>0:    
#             try:
#                 logger.debug('execute marketIdUpdateCommands %s',marketIdUpdateCommands)
#                 (marketDbCounter2,results) = executeCommands(marketDB,connections,marketIdUpdateCommands)
#                 marketDbCounter += marketDbCounter2
#                 logger.debug( "marketIdUpdateCommands executed successfully")    
#                 axiomaDB.flushCaches()
#                 marketDB.flushCaches()
#             except Exception as e :
#                 logger.exception("Error during marketIdUpdateCommands in runAxiomaIdCommands(): %s",e)
#                 __raiseError__(e)
#     
#         if len(modelIdUpdateCommands)>0:
#             try:   
#                 logger.debug('execute modelIdUpdateCommands %s',modelIdUpdateCommands)
#                 (modelDbCounter1, results) = executeCommands(modelDB,connections,modelIdUpdateCommands)
#                 modelDbCounter += modelDbCounter1
#                 modelDB.flushCaches()
#             except Exception as e :
#                 logger.exception("Error during modelIdUpdateCommands in runAxiomaIdCommands(): %s",e)
#                 __raiseError__(e)
#         
#         if not testOnly:
#             logger.info('committng changes in axiomaDb, marketDb and modelDb')
#             axiomaDB.commitChanges()
#             marketDB.commitChanges()
#             modelDB.commitChanges()
#             
#         if len(marketIdTransferCommands)>0:  
#             try:
#                 logger.debug('execute marketIdTransferCommands %s',marketIdTransferCommands)
#                 (dummy, transferResult) = executeCommands(marketDB,connections,marketIdTransferCommands)
#                 transferResults.extend(transferResult)
#                 marketDB.flushCaches()
#             except Exception as e :
#                 logger.exception("Error during marketIdTransferCommands in runAxiomaIdCommands(): %s",e)
#                 __raiseError__(e)
#                 
#         if len(modelIdTransferCommands)>0:      
#             try:  
#                 logger.debug('execute modelIdTransferCommands %s',modelIdTransferCommands)
#                 (dummy, transferResult) = executeCommands(modelDB,connections,modelIdTransferCommands)
#                 transferResults.extend(transferResult)
#                 modelDB.flushCaches()
#             except Exception as e :
#                 logger.exception("Error during modelIdTransferCommands in runAxiomaIdCommands(): %s",e)
#                 __raiseError__(e)
#         
#         if len(qaReportsCommands)>0:
#             try:#NP remove this try later
#                 command=qaReportsCommands[0].replace('~~~','') 
#                 logger.info('in executeCommands(); qaReportsCommand1: %s ', command)
#                 eval(command)
#                 logger.info( "runQaReports executed successfully")    
#             except Exception as e:
#                 logger.exception("Error: %s", e)      
#                 raise e
#                 
#         t2 = datetime.datetime.today()
#         dt = t2 - t1
#         msg=' successful;\ntime elapsed: '+str(dt)+'\n'
#         logger.debug(msg)
#         
#         if testOnly:
#             logger.info('Reverting changes because testOnly is True')
#             connections.revertAll()
#         else:
#             logger.info('Committing changes')
#             axiomaDB.commitChanges()
#             marketDB.commitChanges()
#             modelDB.commitChanges()
#         return (msg, marketDbCounter, modelDbCounter, transferResults)
#             
#     except Exception as e:
#         logger.exception("Exception in runAxiomaIdCommands(). Reverting all changes: %s",e)
#         connections.revertAll()
#         raise e
#     finally:
#         connections.finalizeAll()
# 
#     return [marketDbCounter,modelDbCounter,transferResults]
# 
# def runAssetAdditionCommands(sid, assetAdditionCommands,qaReportsCommands=[]):
#     logger.debug('in runAssetAdditionCommands() %s %s',sid, assetAdditionCommands)
#     t1 = datetime.datetime.today()
#         
#     
#     
#     
#     from lib.common.assetAddtitionNonDjango import processBulk#,getHomeClass,__insertCorrection,BEGIN_DT,END_DT, TEMP_DIR
#     errors = []
#     textresult = None
#     commands = None
#     
#     allMarketIdStr=''
# #    allMarketIdStr = 'GHHASU3681' #used for tests
#     if len(assetAdditionCommands)>0:
#         logger.info('execute assetAdditionCommands %s',assetAdditionCommands)
#         connections = Connections.Connections(getModelDbConfig_new(sid))
#         global marketDbConnections, modelDbConnections
#         marketDbConnections=modelDbConnections = connections
#         try:
#             command=assetAdditionCommands[0]
#             logger.debug('in executeCommands(); command1: %s ', command)
# 
#             (errors, textresult, commands) = eval(command)
#             logger.info( "assetAdditionCommands executed successfully")    
#             
#             if len(textresult)>0:
#                 allMarketIdStr = textresult[-1]            
#                 logger.info('allMarketIdStr: %s ', allMarketIdStr)
#         
#             if testOnly:
#                 logger.info('Reverting changes because testOnly is True')
#     #             marketDbConnections.revertAll()
#                 connections.revertAll()
#             else:
#                 logger.info('Committing changes')
#                 connections.axiomaDB.commitChanges()
#                 connections.marketDB.commitChanges()
#                 connections.modelDB.commitChanges()   
#         except Exception as e:
#                 logger.exception('Exception during processing. Reverting all changes:\n%s\n%s',e, traceback.format_exc())
#                 connections.revertAll()
#     #             modelDbConnections.revertAll()
#                 raise e
#         finally:
#             connections.finalizeAll()
#                     
#     if len(qaReportsCommands)>0 and len(allMarketIdStr)>0:
#         try:#NP remove this try later
#             command=qaReportsCommands[0]
#             logger.info('in executeCommands(); qaReportsCommand1: %s ', command)
#             eval(command)
#             logger.info( "runQaReports executed successfully")    
#         except Exception as e:
#             logger.exception("Error: %s", e)      
#             raise e
#         
#     t2 = datetime.datetime.today()
#     dt = t2 - t1
#     msg=' successful;\ntime elapsed: '+str(dt)+'\n'
#     logger.info(msg)
#     
#     return  (errors, textresult, commands)
# 
# def checkRequiredParams(params):
#     for param in params:
#         if param is None:
#             return False
#     return True
# 
# def runMacAssetCoverageCommands(paramDict):
#     
#     logger.debug('in runMacAssetCoverageCommands() %s ',paramDict)
#     
#     macDbId = paramDict.get('macDbId',None)
#     inputLines = paramDict.get('inputLines',None)
#     assetList = eval(paramDict.get('assetList','None'))
#     commands = paramDict.get('macAssetCoverageCommands',None)
#     username = paramDict.get('username',None)
#     
#     if not checkRequiredParams([macDbId, inputLines,assetList,commands]):
#         raise Exception('macDbId, inputLines,assetList,commands for runMacAssetCoverageCommands() cannot be None; %s'%str([macDbId, inputLines,assetList,commands]))
#     
#     if len(commands)!=1:
#         logger.exception('Exception during processing: no commands provided;')
#         return  (errors, textresult, commands)
#     
#     t1 = datetime.datetime.today()
#     errors = []
#     textresult = None
#     
#     logger.info('execute commands %s',commands)
#     command = commands[0]
#     try:
#         combinedResults = eval(command)    
#         
#         combinedResultsRowList=[]
#         for row in combinedResults:
#             combinedResultsRowList.append(','.join([str(f) for f in row])) 
#         attachmentsData=[['macAssetCoverage.csv',  '\r'.join(combinedResultsRowList)]]
#         subject="macAssetCoverage results"
#         body="macAssetCoverage results attached"
#         emailResults(username+'@qontigo.com', subject, body, attachmentsData)
#         logger.info( "runMacAssetCoverageCommands executed successfully")    
#     except Exception as e:
#             logger.exception('Exception during processing. Reverting all changes:\n%s\n%s',e, traceback.format_exc())
#             raise e
#     
#     t2 = datetime.datetime.today()
#     dt = t2 - t1
#     msg=' successful;\ntime elapsed: '+str(dt)+'\n'
#     logger.info(msg)
#     
#     return  (errors, textresult, commands)
# 
# 
# def runAssetAdditionCommands_new(sid, assetAdditionCommands,qaReportsCommands=[], macdbId = None, jobid=None):
#     logger.setLevel(logging.DEBUG)
#     logger.debug('in runAssetAdditionCommands() %s %s',sid, assetAdditionCommands)
#     
#     logger1 = logging.getLogger('marketdb.QADirect')
#     logger1.setLevel(logging.DEBUG)
#     
#     t1 = datetime.datetime.today()
#         
#     from lib.common.assetAddtitionNonDjango import processBulk#,getHomeClass,__insertCorrection,BEGIN_DT,END_DT, TEMP_DIR
#     errors = []
#     textresult = None
#     commands = None
#     
#     allMarketIdStr=''
# #    allMarketIdStr = 'GHHASU3681' #used for tests
#     if len(assetAdditionCommands)>0:
#         logger.info('execute assetAdditionCommands %s',assetAdditionCommands)
#         connections = Connections.Connections(getMarketDbConfig_new(sid))
#         macDb = None
#         try:
#             if macdbId is not None:
#                 macDb = MacDatabase(macdbId)
#                 macDb.openConnection()
#                 macDb.setAutocommit(True)
#             
#                 
#             global marketDbConnections, modelDbConnections
#             marketDbConnections=modelDbConnections = connections
#             commandNumber = 0
#             for command in assetAdditionCommands:      
#                 commandNumber += 1   
# #                 command = command.replace('<<jobid>>',str(jobid))
#                    
#                 logger.info('in executeCommands(); command#%d: %s', commandNumber, command)
#                 result = eval(command)
#                 logger.info( "command#%d executed successfully", commandNumber)    
#                 
#                 
#                 if isinstance(result, int):
#                     pass
#                 
# #                 (errors, textresult, commands)
#                 elif len(result) == 3 and len(result[1])>0:
#                     allMarketIdStr = result[1][-1]            
#                     logger.info('allMarketIdStr: %s ', allMarketIdStr)
#             
#                 if testOnly:
#                     logger.info('Reverting changes because testOnly is True')
#                     connections.revertAll()
#                 else:
#                     logger.info('Committing changes')
#                     connections.axiomaDB.commitChanges()
#                     connections.marketDB.commitChanges()
#                     connections.modelDB.commitChanges()   
#         
#             logger.info('%d assetAdditionCommands executed successfully',commandNumber)
#                     
#         except Exception as e:
#                 logger.exception('Exception during processing. Reverting all changes:\n%s\n%s',e, traceback.format_exc())
#                 connections.revertAll()
#     #             modelDbConnections.revertAll()
#                 raise e
#         finally:
#             connections.finalizeAll()
#             if macdbId is not None:
#                 macDb.finalize()
#     
#     if len(qaReportsCommands)>0 and len(allMarketIdStr)>0:                
#         try:#NP remove this try later
#             command=qaReportsCommands[0]
#             logger.info('in executeCommands(); qaReportsCommand1 %s: ', command)
# #             command=command.encode('utf8')
#             logger.debug('in executeCommands(); qaReportsCommand2 %s: ', command)
#             eval(command)
#             logger.info( "runQaReports executed successfully")    
#         except Exception as e:
#             logger.exception("Error: %s", e)      
#             raise e
#         
#     t2 = datetime.datetime.today()
#     dt = t2 - t1
#     msg=' successful;\ntime elapsed: '+str(dt)+'\n'
#     logger.info(msg)
#     
#     return  (errors, textresult, commands)
# 
#                            
# def __raiseError__(e):
#     print(e, traceback.print_exc())
#     logger.error("an error occured: %s %s", str(e), traceback.format_exc())
#     raise e


def main():
    usage = "usage: %prog [options] commands-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")

    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
        
    Utilities.processDefaultCommandLine(options, cmdlineParser, disable_existing_loggers=False)
    
    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    
    commandsFileName = args[0]
    if len(args)>1:
        jobid = args[1]
    else:
        jobid = None

    global testOnly   
    testOnly = options.testOnly
    
    logger.info('in RiskModels.jobcontrol.runJsonCommands: testOnly %s, file %s', testOnly, commandsFileName)
    try:
        retObj = runJsonCommands(commandsFileName,jobid,testOnly)
        logger.info('retObj: '+str(retObj))
    except Exception as e:
        logger.exception('Exception during processing: %s',e)
        sys.exit(1)
    logger.info('in RiskModels.jobcontrol.runJsonCommands: done')

       
if __name__=='__main__':
    main()
    
    

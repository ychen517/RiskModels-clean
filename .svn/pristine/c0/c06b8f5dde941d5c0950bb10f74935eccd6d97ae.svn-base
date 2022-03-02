'''
   Document Me!
'''

import os
import time
import logging
import configparser

from marketdb.ConfigManager import findConfig
from marketdb.MarketDB import MarketDB
from marketdb.CompuStat import CompuStatDB
from marketdb import Utilities as MkDBUtils
from marketdb.XPS import XpressFeedDB
from marketdb.QADirect import QADirect
from riskmodels.XpressfeedRM import XpressfeedData
from riskmodels.ModelDB import ModelDB

SHORT_DATE_FORMAT = '%Y-%m-%d' 


def loadConfigFile(configFile):
    config = configparser.ConfigParser()
    config.read_file(configFile)
    return config

def _getDatabaseConnections(configuration):
    connections = MkDBUtils.Struct()
    connections.marketDB = _getMarketDBInstance(configuration)
    connections.xpressfeedDB = _createXpressfeedDB(configuration)
    connections.cstatDB = _createCompuStatDB(configuration)
    connections.modelDB = _createModelDB(configuration)
    connections.qaDirect = _createQADirectConnection(configuration)
    connections.qaAudit = _createQAAuditConnection(configuration)
    return connections

def _getMarketDBInstance(configuration):
    dbInfoMap = MkDBUtils.getConfigSectionAsMap(configuration, 'MarketDB')    
    return MarketDB(user=dbInfoMap['user'], passwd=dbInfoMap['password'], sid=dbInfoMap['sid'])

def _createXpressfeedDB(config):
    if config.has_section('XpressFeedDB'):
        con = XpressFeedDB(user=config.get('XpressFeedDB', 'user'),
                            passwd=config.get('XpressFeedDB', 'password'),
                            sid=config.get('XpressFeedDB', 'sid'))
    else:
        logging.error("Missing MarketDB section in configuration file")
        con = None
    return con

def _createCompuStatDB(config, section='CompuStatDB'): 
    return CompuStatDB(user=config.get(section, 'user'),
                      passwd=config.get(section, 'password'),
                      sid=config.get(section, 'sid'))

def _createModelDB(config, section='ModelDB'): 
    return ModelDB(user=config.get(section, 'user'),
                   passwd=config.get(section, 'password'),
                   sid=config.get(section, 'sid'))

def _createQADirectConnection(config):
    if config.has_section('QADirect'):
        con = QADirect(user=config.get('QADirect', 'user'),
                       passwd=config.get('QADirect', 'password'),
                       host=config.get('QADirect', 'host'),
                       database=config.get('QADirect', 'database'))
    else:
        logging.error('Missing QADirect section in configuration file')
        con = None
    return con
    
def _createQAAuditConnection(config):
    con = QADirect(user=config.get('QAAudit', 'user'),
                   passwd=config.get('QAAudit', 'password'),
                   host=config.get('QAAudit', 'host'),
                   database=config.get('QAAudit', 'database'))
    return con

def _fileCheck(fullFileName):
    return fullFileName if os.path.exists(fullFileName) else None

def _printResults(fundDataMx, measureID):
    msg = "%s RESULT SET"%measureID.upper()
    print(msg)
    print("="*len(msg))
    print()
    print(fundDataMx)
    print()

def _printSubResults(measureID, fundDataMx, selectedSubIssueIDs):
    msg = "\n%s MEASURE RESULTS"%(measureID)
    print(msg)
    print("=" * (len(msg) + 2))
    
    #gvKeyColID = fundDataMx.getColumnIndex("GVKEY")  => E.g. gvKey =  '001690'
    subIssueIndex = fundDataMx.getColumnIndex("SUBISSUE_ID")
    msg = "\n AXIOMA_ID  |  SUBISSUE_ID | DATA_DATE | EFFT_DATE | VALUE"
    print(msg)
      
    currentID = '' 
    for i in range(fundDataMx.getRowCount()):
        if fundDataMx.getValue(i, subIssueIndex) in selectedSubIssueIDs:
            value = fundDataMx.getValue(i, fundDataMx.getColumnIndex("VALUE"))
            if abs(value) > 0.000000001:
                dataDT = fundDataMx.getValue(i, fundDataMx.getColumnIndex("DATADATE"))
                effDT =  fundDataMx.getValue(i, fundDataMx.getColumnIndex("EFFECTIVE_DATE"))
                axiomaID = fundDataMx.getValue(i, fundDataMx.getColumnIndex("MARKETDB_ID"))
                subIssueID = fundDataMx.getValue(i, fundDataMx.getColumnIndex("SUBISSUE_ID"))
                
                if axiomaID != currentID:
                    currentID = axiomaID
                    print("-"*(len(msg) + 2))
                print((" %s | %s | %s | %s | %s"%(axiomaID, subIssueID, 
                            dataDT.strftime(SHORT_DATE_FORMAT),
                            effDT.strftime(SHORT_DATE_FORMAT), value)))
   
def main():
    # 1. Setup logging config file:
    logConfigFile = findConfig("log.config", "RiskModels")
    if not os.path.exists(logConfigFile):
        raise Exception("Logging configuration file:%s does not exist."%logConfigFile)
    logging.config.fileConfig(logConfigFile)
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO')
         
    # 2. Obtain Database Info from Config file
    configFilePath = findConfig('production.config', "RiskModels")
    configuration = MkDBUtils.getConfiguration(configFilePath)
        
    # 3. Create connection to databases and instantiate Xpressfeed Data Provider
    connections = _getDatabaseConnections(configuration)
    xpsDS = XpressfeedData(connections)
    
    # 4. Simple Example: Simulate User-specified measures
    subIssueIDs = ['DQ8L3JXGU111', 'D3MPV4F52811']   # AAPL [KDWC2FL4X1] + XOM [KTK98FNXX5]
    fundDataMx = xpsDS.getFundamentalDataMatrix(subIssueIDs, 'DPS_ANN')
       
    # 5A Print out Results
    startTime=time.time()
    elapsedTime =  time.time() - startTime
    logging.info(" Elapsed Time:%s ColumnsIDs: %s Rows:%s"%(elapsedTime, fundDataMx.getColumnCount(), fundDataMx.getRowCount()))
    _printSubResults( 'DPS_ANN', fundDataMx, subIssueIDs)
    
    # 5B Broader SubIssue Set: All US SubIssues: IDs & Ranges
#     startTime=time.time()
#     USSubIssues = transfer.createSubIssueIDList('US', connections)
#     USSubIssueIDList = [x.getSubIDString() for x in USSubIssues[0]] 
#     fundDataMx = xpsDS.getFundamentalDataMatrix(USSubIssueIDList, 'DPS_ANN')
#     elapsedTime =  time.time() - startTime
#     logging.info(" Elapsed Time:%s ColumnsIDs: %s Rows:%s"%(elapsedTime, fundDataMx.getColumnCount(), fundDataMx.getRowCount()))
#     _printSubResults('DPS_ANN', fundDataMx, subIssueIDs)
        
if __name__ == '__main__':
    main()

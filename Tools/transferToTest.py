
import datetime
import logging
import numpy
import operator
import optparse
import configparser
import cx_Oracle
import sys
from riskmodels import Utilities

INPUT_TABLE_LIST=[
# tablename, colname, dbName, dependent col, dep table, dep table's colname, whereclause in dep table
('ASSET_DIM_UCP','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_TDV','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_RETURN','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_VENDOR_RETURN','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_FUND_CURRENCY','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_ESTI_CURRENCY','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_TSO','AXIOMA_ID',None,'marketdb', None,None,''),
('ASSET_DIM_FUND_NUMBER','AXIOMA_ID',None,'marketdb', None,None,''),
('FUTURE_DIM_UCP','AXIOMA_ID',None,'marketdb', None,None,''),
('FUTURE_DIM_OPEN_INTEREST','AXIOMA_ID',None,'marketdb', None,None,''),
('FUTURE_DIM_TDV','AXIOMA_ID',None,'marketdb', None,None,''),
('FUTURE_DIM_RETURN','AXIOMA_ID',None,'marketdb', None,None,''),
('SUB_ISSUE_DATA','SUB_ISSUE_ID',None,'modeldb',None, None,''),
('SUB_ISSUE_RETURN','SUB_ISSUE_ID',None,'modeldb',None, None,''),
('SUB_ISSUE_CUMULATIVE_RETURN','SUB_ISSUE_ID',None,'modeldb',None, None,''),
('SUB_ISSUE_FUND_CURRENCY','SUB_ISSUE_ID',None,'modeldb',None, None,''),
#('RMG_MARKET_PORTFOLIO','SUB_ISSUE_ID',None,'modeldb',None, None,''),
#('RMG_HISTORIC_BETA','SUB_ISSUE_ID',None,'modeldb',None, None,''),
#('RMG_HISTORIC_BETA_V3','SUB_ISSUE_ID',None,'modeldb',None, None,''),
]
#('rmi_universe','SUB_ISSUE_ID',None,'modeldb',None,None,''),
#('rmi_universe','SUB_ISSUE_ID','rms_id','modeldb','risk_model_serie','serial_id','where distribute =1'),

MARKETDB_TABLES=['ASSET_DIM_UCP','ASSET_DIM_TDV','ASSET_DIM_RETURN','ASSET_DIM_VENDOR_RETURN',
        'ASSET_DIM_FUND_CURRENCY', 'ASSET_DIM_ESTI_CURRENCY','RETURNS_QA','ASSET_DIM_TSO','ASSET_DIM_FUND_NUMBER','FUTURE_DIM_UCP',
        'FUTURE_DIM_OPEN_INTEREST', 'FUTURE_DIM_TDV','FUTURE_DIM_RETURN']

MODELDB_TABLES=[
        'SUB_ISSUE_DATA','SUB_ISSUE_RETURN','SUB_ISSUE_CUMULATIVE_RETURN','RMI_ESTU','RMI_FACTOR_EXPOSURE','RMI_PREDICTED_BETA','RMI_SPECIFIC_COVARIANCE',
        'RMI_COVARIANCE','RMI_SPECIFIC_RISK','RMI_TOTAL_RISK','RMI_UNIVERSE','RMS_FACTOR_RETURN','RMS_FACTOR_STATISTICS',
        'RMS_SPECIFIC_RETURN','RMS_STATISTICS','RMG_HISTORIC_BETA',
        'RMI_PREDICTED_BETA_V3','RMI_ESTU_V3', 'SUB_ISSUE_FUND_CURRENCY','RMS_STATISTICAL_FACTOR_RETURN','RMG_MARKET_PORTFOLIO',
        'RMG_HISTORIC_BETA_V3','RMG_PROXY_RETURN','SUB_ISSUE_ESTI_CURRENCY'
]
SPL_MARKETDB_TABLES=[ 'INDEX_CONSTITUENT','COMPOSITE_CONSTITUENT',]


def finishTransactions(connections_, options_):
    if options_.testOnly:
        logging.info('Reverting changes')
        connections_.targetModelDB.rollback()
        connections_.targetMarketDB.rollback()
    else:
        logging.info('Committing changes')
        connections_.targetModelDB.commit()
        connections_.targetMarketDB.commit()

def finalize(conn):
    conn.cursor().close()
    conn.close()

def createDBConnection(**connectParameters):
    conn= cx_Oracle.connect(connectParameters['user'],connectParameters['passwd'], connectParameters['sid'])
    conn.cursor().execute('alter session set nls_date_format="YYYY-MM-DD HH24:MI:SS"')
    conn.commit()
    return conn

def createConnections(config):
    connections=Utilities.Struct()
    connections.sourceMarketDB=createDBConnection(user=config.get('SourceMarketDB','user'),passwd=config.get('SourceMarketDB','password'),
                        sid=config.get('SourceMarketDB','sid'))
    connections.targetMarketDB=createDBConnection(user=config.get('TargetMarketDB','user'),passwd=config.get('TargetMarketDB','password'),
                        sid=config.get('TargetMarketDB','sid'))
    connections.sourceModelDB=createDBConnection(user=config.get('SourceModelDB','user'),passwd=config.get('SourceModelDB','password'),
                        sid=config.get('SourceModelDB','sid'))
    connections.targetModelDB =createDBConnection(user=config.get('TargetModelDB','user'),passwd=config.get('TargetModelDB','password'),
                        sid=config.get('TargetModelDB','sid'))
    return connections

def finalizeConnections(conn):
    finalize(conn.sourceMarketDB)
    finalize(conn.targetMarketDB)
    finalize(conn.sourceModelDB)
    finalize(conn.targetModelDB)


class SourceProcessor:
    def __init__(self, sourceDB, tableName, colName,foreignKey,dbType,depTable=None, depColumn=None, whereClause=""):
        self.cursor=sourceDB.cursor()
        self.tableName=tableName
        self.colName=colName
        self.depTable=depTable
        self.dbType=dbType
        self.depColumn=depColumn
        self.foreignKey=foreignKey
        self.whereClause=whereClause
        if dbType=='marketdb':
           self.query="""select * from %s where %s=:id""" % (tableName, colName)
        if dbType=='modeldb':
           self.query="""select * from %s where %s in 
                        (select s.sub_id from sub_issue s, issue_map m  
                                where s.issue_id=m.modeldb_id and m.marketdb_id=:id)
                        union
                         select * from %s where %s in 
                        (select s.sub_id from sub_issue s, issue_map m  
                                where s.issue_id=m.modeldb_id and m.marketdb_id=:id)
                        """ % (tableName, colName, tableName, colName)

    def getDependentList(self):
       """
           dependent table and dbtype. For now, there is only modeldb and risk_model
       """
       if not self.depTable:
           return [None]

       query=(""" select %s from %s %s""" % (self.depColumn,self.depTable, self.whereClause))
       self.cursor.execute(query)
       results=self.cursor.fetchall()
       return [r[0] for r in results]


    def getHeaders(self):
        query="""select * from %s where 0=1""" % self.tableName
        self.cursor.execute(query)
        headers=[i[0] for i in self.cursor.description]
        return headers

    def getBulkData(self, idList, depIdList):
        logging.info('  .... getBulkData %s', self.tableName)
        retvals=numpy.empty((len(depIdList),len(idList)), dtype=object)
        for depIdx, depId in enumerate(depIdList):
            for idx, axid in enumerate(idList):
                if self.foreignKey:
                    query='%s and %s=%s' % (self.query, self.foreignKey, depId)
                else:
                    query=self.query
                self.cursor.execute(query, id=axid)
                results=self.cursor.fetchall()
                retvals[depIdx,idx] = results
                logging.info('       .... %s had %d rows', axid, len(results))
        return retvals
    

class TargetProcessor:
    """ 
        dbType is either marketdb or modeldb
        dependant table indicates the table on which additionally we are dependent
       
    """
    def __init__(self, targetDB, tableName, colName, dbType, headers, depTable=None, depColumn=None):
        self.cursor=targetDB.cursor()
        if dbType=='marketdb':
            self.delQuery="""delete from %s where %s=:id""" % (tableName, colName)
        if dbType=='modeldb':
            self.delQuery="""delete from %s where %s in 
                        (select s.sub_id from sub_issue s, issue_map m  
                                where s.issue_id=m.modeldb_id and m.marketdb_id=:id
                        union
                        select s.sub_id from sub_issue s, future_issue_map m  
                                where s.issue_id=m.modeldb_id and m.marketdb_id=:id)

                        """ % (tableName, colName)
        self.dbType=dbType
        self.tableName=tableName
        self.query="""insert into %s (%s)
                values (%s)
        """ % (tableName, ','.join(['%s' % h for h in headers]),','.join([':%s' % h for h in headers]))
        self.headers=headers
        self.depColumn=depColumn
 
    def bulkProcess(self, idList, retvals): 
         # for now simply assume the retvals has only one dimension
         logging.info('  .... Target Bulk Processing %s', self.tableName)
         for depIdx in range(1):
             for idx,axid in enumerate(idList):
                 # first clean up the old data in the table for this asset
                 self.cursor.execute(self.delQuery, id=axid)        
                 if self.cursor.rowcount:
                      logging.info('       .... Deleted pre-existing %d rows for %s',  self.cursor.rowcount, axid)
                 valDicts=[dict(zip(self.headers,r)) for r in retvals[depIdx, idx]]
                 self.cursor.executemany(self.query, valDicts)
                 logging.info('       ....  %s inserted %d rows', axid, self.cursor.rowcount)

if __name__ == '__main__':
    usage = "usage: %prog [options] config-file [[section:]option=value ...]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("-f", action="store_true",
                             default=False, dest="override",
                             help="override certain aspects")
    cmdlineParser.add_option("-d", action="store_true",
                             default=False, dest="cleanup",
                             help="delete current records before computation")
    cmdlineParser.add_option("--nuke", action="store_true",
                             default=False, dest="nuke",
                             help="delete all records before computation")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="Extra debugging output")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    # process command-line options to override config file
    for arg in args_[1:]:
        fields = arg.split('=')
        if len(fields) != 2:
            logging.fatal('Incorrect command-line assignment "%s"'
                          ', exactly one "=" is required', arg)
            sys.exit(1)
        (option, value) = fields
        section = 'DEFAULT'
        if option.find(':') != -1:
            fields = option.split(':')
            if len(fields) != 2:
                logging.fatal('Incorrect command-line assignment "%s"'
                              ', at most one ":" is allowed in option "%s"',
                              arg, option)
                sys.exit(1)
            (section, option) = fields
        if section != 'DEFAULT' and not config_.has_section(section):
            logging.fatal('Section "%s" is not present in config file.',
                          section)
            sys.exit(1)
        config_.set(section, option, value)
    connections_ = createConnections(config_)
    errorCase = False
    # process selected sections
    # try an experiment with asset_dim_ucp first
    tables=config_.defaults().get('tables')
    if tables:
        tableList=tables.split(',')
    else:
        tableList= [t[0] for t in INPUT_TABLE_LIST]
    tableList=[t.upper() for t in tableList]
    logging.info('Working on tables %s', ','.join(tableList))
    try:
        axioma_ids=config_.defaults().get('axioma-ids', None)
        idList=[]
        masterIDList=[]
        if axioma_ids:
            masterIDList=axioma_ids.split(',')
        INCR=50
        # do just 50 ids at a time for now
        while masterIDList:
            idList=masterIDList[:INCR] 
            for (tableName, colName,fkCol,dbType,depTable,depColumn, whereClause) in INPUT_TABLE_LIST:
                if tableName not in tableList:
                    continue
                logging.info('Processing %s', tableName)
                if dbType=='marketdb':
                    dbconn=connections_.sourceMarketDB
                    targetDB=connections_.targetMarketDB
                elif dbType=='modeldb':
                    dbconn=connections_.sourceModelDB
                    targetDB=connections_.targetModelDB
                else:
                    # make it crash!
                    dbconn=None
                sp=SourceProcessor(dbconn, tableName, colName, fkCol, dbType, depTable, depColumn, whereClause)
                depList=sp.getDependentList() 
                #logging.info('depList=%s',depList)
                headers=sp.getHeaders()
                #logging.info('headers=%s',headers)
                retvals=sp.getBulkData(idList, depList)
                tp=TargetProcessor(targetDB, tableName, colName,dbType, headers, depTable, depColumn)
                tp.bulkProcess(idList, retvals)
                finishTransactions(connections_,options_)
            # next set of INCR assets
            masterIDList=masterIDList[INCR:]

    except Exception as e:
        logging.fatal('Exception during processing. Reverting all changes',
                      exc_info=True)
        connections_.targetModelDB.rollback()
        connections_.targetMarketDB.rollback()
        errorCase = True
    finishTransactions(connections_, options_)
    finalizeConnections(connections_)
    if errorCase:
        sys.exit(1)


import configparser
import logging
import optparse
from marketdb.Utilities import listChunkIterator
from riskmodels import Connections
from riskmodels import Utilities

def executeDelete(mdlDB, query, queryDict, tableName):
    try:
        logging.info("Deleting records from %s", tableName)
        logging.debug('%s %% %s', query, queryDict)
        mdlDB.dbCursor.execute(query, queryDict)
        logging.info("Deleted %d records from %s",
                     mdlDB.dbCursor.rowcount, tableName)
    except Exception as e:
        logging.error('Exception caught while deleting from %s',
                      tableName, exc_info=True)
        return False
    return True

def findExposureTables(mdlDB):
    """Returns a list of all model specific exposure tables.
    """
    mdlDB.dbCursor.execute("""SELECT table_name FROM user_tables
       WHERE table_name like 'RMS_%_FACTOR_EXPOSURE'""")
    return [i[0] for i in mdlDB.dbCursor.fetchall()]

def deleteSpecificCovariances(mdlDB, queryArgs, queryDict):
    """Delete the given issue IDs from the specific covariance table.
    """
    query = """DELETE FROM rmi_specific_covariance
       WHERE sub_issue1_id IN (SELECT sub_id
         FROM sub_issue WHERE issue_id IN (%(args)s))""" \
        % { 'args': queryArgs }
    status1 = executeDelete(mdlDB, query, queryDict, 'rmi_specific_covariance')
    query = """DELETE FROM rmi_specific_covariance
       WHERE sub_issue2_id IN (SELECT sub_id
         FROM sub_issue WHERE issue_id IN (%(args)s))""" \
        % { 'args': queryArgs }
    status2 = executeDelete(mdlDB, query, queryDict, 'rmi_specific_covariance')
    return status1 and status2

mdlIDcache = dict()

def isMutualFund(mdlDB, mdlID):
    if mdlID not in mdlIDcache:
        mdlDB.dbCursor.execute('SELECT marketdb_id FROM issue_map WHERE modeldb_id=:mdlid', mdlid=mdlID)
        r = mdlDB.dbCursor.fetchall()
        if r and r[0][0].startswith('O'):
            mdlIDcache[mdlID] = True
        else:
            mdlIDcache[mdlID] = False
    return mdlIDcache[mdlID]

#------------------------------------------------------------------------------
if __name__ == '__main__':
    usage = "usage: %prog [options] <config file> <issue-id>[,<issue-id>,...]"
    
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-u", "--update-database", action="store_true",
                             default=False, dest="updateDB",
                             help="run queries and commit")
    cmdlineParser.add_option("--delete-exposures", action="store_false",
                             default=True, dest="leaveExposures",
                             help="delete exposures")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
        
    configFile_ = open(args[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    connections = Connections.createConnections(config_)
    mdldbglobal_ = connections.modelDB
    
    tableNames = [
        ('sub_issue_fund_number',  'sub_issue_id', True),
        ('sub_issue_fund_currency',  'sub_issue_id', True),
        ('sub_issue_esti_currency', 'sub_issue_id', True),
        ('sub_issue_cumulative_return',  'sub_issue_id', True),
        ('sub_issue_return',  'sub_issue_id', True),
        ('sub_issue_data',  'sub_issue_id', True),
        ('sub_issue_estimate_data',  'sub_issue_id', True),
        ('rmi_estu',  'sub_issue_id', True),
        ('rmi_estu_v3',  'sub_issue_id', True),
        ('rmi_predicted_beta',  'sub_issue_id', True),
        ('rmi_specific_risk',  'sub_issue_id', True),
        ('rmi_total_risk',  'sub_issue_id', True),
        ('rmi_universe',  'sub_issue_id', True),
        ('rmg_historic_beta_v3',  'sub_issue_id', True),
        ('classification_constituent',  'issue_id', False),
        ('rms_estu_excluded', 'issue_id', False),
        ('mdl_port_rebalance_universe', 'sub_issue_id', True),
        ('mdl_port_master', 'sub_issue_id', True),
        ('mdl_port_rebalance_universe_pf', 'sub_issue_id', True),
        ('mdl_port_master_pf', 'sub_issue_id', True),
        ('sub_issue', 'issue_id', False),
        ('rms_issue', 'issue_id', False),
        ('ca_spin_off', 'child_id', False),
        ('ca_spin_off', 'parent_id', False),
        ('ca_merger_survivor', 'modeldb_id', False),
        ('issue_map', 'modeldb_id', False),
        ('issue', 'issue_id', False),
        ]
    
    modelIds = args[1]
    status = True
    mdlIdList = modelIds.split(",")
    exposureTables = [(name, 'sub_issue_id', True) for name
                      in findExposureTables(mdldbglobal_)]
    exposureTables.append(('rmi_factor_exposure',  'sub_issue_id', True))
    exposureTableNames = [a[0] for a in exposureTables]
    tableNames = exposureTables + tableNames

    for mdlIds in listChunkIterator(mdlIdList, 50):
        logging.info('Processing %s', ','.join(mdlIds))
        argNames = ['mdi%d' % i for i in range(len(mdlIds))]
        queryArgs = ','.join([':%s' % arg for arg in argNames])
        queryDict = dict(zip(argNames, mdlIds))
        
        # only non-mutual funds for specific covariances
        nonFundIds = [m for m in mdlIds if not isMutualFund(mdldbglobal_, m)]
        nonFundNames = ['mdi%d' % i for i in range(len(nonFundIds))]
        myQueryArgs = ','.join([':%s' % arg for arg in nonFundNames])
        myQueryDict = dict(zip(argNames, nonFundIds))
        if myQueryDict:
            thisStatus = deleteSpecificCovariances(mdldbglobal_,
                                                   myQueryArgs, myQueryDict)
            status = status and thisStatus

        for (tableName, columnName, useSubIssue) in tableNames:
            if options.leaveExposures and tableName in exposureTableNames:
                logging.debug('Skipping %s', tableName)
                continue
            if (tableName.startswith('rmi') or tableName.startswith('ca') or tableName.startswith('sub_issue_fund') or tableName.startswith('sub_issue_esti')):
                myQueryArgs = ','.join([':%s' % arg for arg in nonFundNames])
                myQueryDict = dict(zip(argNames, nonFundIds))
            else:
                myQueryArgs = queryArgs
                myQueryDict = queryDict
            if not myQueryDict:
                continue
            if useSubIssue:
                query = """delete from %(table)s where %(column)s IN
                   (SELECT sub_id FROM sub_issue WHERE issue_id IN (%(args)s))"""\
                    % {'table': tableName, 'column': columnName, 'args': myQueryArgs}
            else:
                query = """delete from %(table)s where %(column)s IN (%(args)s)"""\
                    % {'table': tableName, 'column': columnName, 'args': myQueryArgs}
            status = status and executeDelete(
                mdldbglobal_, query, myQueryDict, tableName)
        if options.updateDB and status:
            logging.info("Committing changes")
            mdldbglobal_.commitChanges()
        else:
            logging.info("Reverting changes")
            mdldbglobal_.revertChanges()

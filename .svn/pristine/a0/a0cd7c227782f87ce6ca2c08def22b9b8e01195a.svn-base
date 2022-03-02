# Python script to generate a report with inconsistencies between the 
# production database and application database

import logging
import optparse
import sys
import datetime
import configparser
import psycopg2
import pymssql
import smtplib
from email.mime.text import MIMEText
import datetime
from socket import gethostname
from riskmodels import Utilities
from riskmodels import ModelDB

#------------------------------------------------------------------------------
# Create the required connections from the parameters in the config file
#
def createConnections(config, sqlserver):

    mdlUser = config.get('ModelDB', 'user')
    mdlPasswd = config.get('ModelDB', 'passwd')
    mdlSID = config.get('ModelDB', 'sid')
    modelDB = ModelDB.ModelDB(user=mdlUser, passwd=mdlPasswd,
                                 sid=mdlSID)
    if sqlserver:
        section='AppDB-sqlserver'
        pgDB = pymssql.connect(user = config.get(section, 'user'), 
                            password = config.get(section, 'passwd'), 
                            host = config.get(section, 'host'),
                            port =  config.get(section, 'port'),
                            database = config.get(section, 'database'))
        pgDB.autocommit(True)
    else:
        section='AppDB'
        pgDB = psycopg2.connect(user = config.get(section, 'user'), 
                            password = config.get(section, 'passwd'), 
                            host = config.get(section, 'host'),
                            database = config.get(section, 'database'))
    return pgDB, modelDB

#------------------------------------------------------------------------------
# Email the report to the receipients specified in the config file
#
def sendEmail(config, reportString):
    msg = MIMEText(reportString)

    msg['Subject'] = "Database inconsistency as of %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    msg['From'] = 'dbconsistencychecker@axioma.com'
    receipients = [i.strip() for i in config.get('DEFAULT', 'email-receipients').split(",")]
    msg['To'] = ",".join(receipients)

    s = smtplib.SMTP('localhost')
    s.sendmail(msg['From'], receipients, msg.as_string())
    s.quit()

#------------------------------------------------------------------------------
# Create the reports to be emailed
#
def runReports(config, mdldb, appdb, dirName=None, sqlserver=False):
    msquery = config.get('ModelDB', 'ms-query')
    delquery = config.get('ModelDB', 'delete-query')
    modeldelquery = config.get('ModelDB', 'model-delete-query')
    newaddsquery = config.get('ModelDB', 'new-adds-query')
    if sqlserver:
        reportString='This is a comparison with SQL server'
    else:
        reportString='This is a comparison with Postgres'
    reportString = "%s\nAssets to merge: This section shows instances of what was previously two different assets being merged in to a single asset. So the history for the OLD_ID needs to be moved to the NEW_ID between FROM_DT and THRU_DT.\n" % reportString
    reportString = reportString + "OLD_ID|NEW_ID|FROM_DT|THRU_DT (EXCL)\n"
    logging.info("Running merger-survivor report")
    # write out the list of assets to be merged in a file
    if dirName:
        fName=dirName.rstrip('/') + '/' + 'assets_to_merge.txt'
        fhandle=open(fName,'w')
        fhandle.write("OLD_ID|NEW_ID|FROM_DT|THRU_DT (EXCL)\n")
    else:
        fhandle=None
    mdldb.dbCursor.execute(msquery)
    r = mdldb.dbCursor.fetchall()
    for (oldmdlid, mktid, fromdt, thrudt, action, newmdlid) in r:
        # If both new modelid and oldmdlid are in the database log it
        if isAssetinAppDB(appdb, oldmdlid[1: 10]) and isAssetinAppDB(appdb, newmdlid[1: 10]):
            reportString = '%s%s|%s|%s|%s\n' % (reportString, oldmdlid[1: 10], newmdlid[1: 10], fromdt.date(), thrudt.date())
            if fhandle:
                fhandle.write('%s|%s|%s|%s\n' % ( oldmdlid[1: 10], newmdlid[1: 10], fromdt.date(), thrudt.date()))
    if fhandle:
        fhandle.close()

    logging.info("Running asset deletion report")
    reportString = '%s\n\nAssets to delete from database: This section shows assets that have been removed from the database. There are many reasons leading to the decision to delete an asset but it all boils down to 1) asset history is already covered under another asset and hence it is a duplicate, or, 2) it is an asset that is of a type that should not be in the risk model.\nID\n' % reportString

    # write out the list of assets to be deleted in a file
    if dirName:
        fName=dirName.rstrip('/') + '/' + 'assets_to_delete.txt'
        fhandle=open(fName,'w')
        fhandle.write('ID\n')
    else:
        fhandle=None
        
    mdldb.dbCursor.execute(delquery)
    r = mdldb.dbCursor.fetchall()
    for (oldmdlid, mktid, fromdt, thrudt, action) in r:
        # If oldmdlid is in the database log it
        if isAssetinAppDB(appdb, oldmdlid[1: 10]):
            reportString = '%s%s\n' % (reportString, oldmdlid[1: 10])
            if fhandle:
                fhandle.write('%s\n' % oldmdlid[1:10]) 

    if fhandle:
        fhandle.close()

    logging.info("Running asset deletion from models report")
    reportString = '%s\n\nAssets to possibly delete from model(s): This section lists assets that have been removed from one or more models. The primary reasons for this are 1) a Home country assignment change was made (say a DM asset had an assignment to an EM as secondary home country that was later removed resulting in the asset leaving the EM models), or, 2) an ETF wrongly classified as an equity or an Equity ETF included in a stat model, or, 3) an asset waiting to be removed from the database but has been removed from all models as a first step.\nID|FROM_DT|THRU_DT|MODEL_MNEMONIC|NOTES\n' % reportString
    mdldb.dbCursor.execute(modeldelquery)
    r = mdldb.dbCursor.fetchall()
    for (mdlid, fromdt, thrudt, model) in r:
        if isAssetinAppDB(appdb, mdlid[1: 10], 'and a.asset_type_id = 1'):
            isComposite, etfFamilies = isCompositeAssetWithFamilies(mdldb, mdlid)
            if not isComposite:
                if not inSomeModels(mdldb, mdlid):
                    notes = "ASSET REMOVED FROM ALL MODELS; WAITING TO BE REMOVED FROM DB"
                else:
                    notes = "HOME COUNTRY ASSIGNMENT CHANGE"
            else:
                if etfFamilies.strip() == '':
                    if model[-2:] == '-S':
                        notes = "COMPOSITE ASSET REMOVED FROM STAT MODELS"
                    else:
                        notes = "COMPOSITE ASSET REMOVED FROM FUNDAMENTAL MODELS"
                else:
                    notes = "COMPOSITE ASSET in FAMILIES %s" % etfFamilies
            reportString = '%s%s|%s|%s|%s|%s\n' % (reportString, mdlid[1: 10], fromdt.date(), thrudt.date(), model, notes)

    logging.info("Running new asset additions report")
    reportString = '%s\n\nAssets added in the last 30 days not in AppDB: This sections contains assets that have been added to one of our risk models that is not currently in the application database.\nID|FROM_DT|ADD_DT|ASSET TYPE\n' % reportString
    mdldb.dbCursor.execute(newaddsquery)
    r = mdldb.dbCursor.fetchall()
    for (mdlid, fromdt, adddt, assettype) in r:
        if not isAssetinAppDB(appdb, mdlid[1: 10]):
            reportString = '%s%s|%s|%s|%s\n' % (reportString, mdlid[1: 10], fromdt.date(), adddt.date(), assettype)

    return reportString

#------------------------------------------------------------------------------
# Check if the asset is in some of the production models
#
def inSomeModels(mdldb, mdlid):
    query = """select count(*) from rms_issue a, rms_id_description b 
                where issue_id=:issid_arg and a.rms_id=b.rms_id and b.distribute=1"""
    mdldb.dbCursor.execute(query, issid_arg=mdlid)
    r = mdldb.dbCursor.fetchall()
    if r is not None and r[0][0] is not None and r[0][0] >= 1:
        return True

    return False

#------------------------------------------------------------------------------
# Check if the asset is in the application database
#
def isAssetinAppDB(appdb, symb, addToQuery=''):
    cursor = appdb.cursor()
    query = """select primary_symbol from asset a, asset_type b where primary_symbol='%s' and a.asset_type_id=b.asset_type_id %s""" % (symb, addToQuery)
    cursor.execute(query)
    r = cursor.fetchone()
    if r is not None and r[0].encode('ascii', 'ignore')==symb:
        return True
    return False

#------------------------------------------------------------------------------
# Check if the asset is a composite asset in production database
#
def isCompositeAssetWithFamilies(mdldb, mdlid):
    cursor = mdldb.dbCursor
    # Get the list of ETF packages it belongs to if it currently being delivered
    query = """select a.axioma_id, c.name from marketdb_global.composite_member a left join 
               marketdb_global.composite_member_family_map b on
               a.id=b.member_id left join
               marketdb_global.composite_family c  
               on b.family_id=c.id and c.distribute='Y'
               where axioma_id in (select marketdb_id from modeldb_global.issue_map 
                                   where modeldb_id=:mdlid_arg)"""
    cursor.execute(query, mdlid_arg=mdlid)
    r = cursor.fetchall()
    if r is not None and len(r) > 0:
        etfFamilies = [i[1] for i in r if i[1] is not None]
        return True, ",".join(etfFamilies)

    return False, ''

#------------------------------------------------------------------------------
if __name__ == '__main__':
    usage = "usage: %prog [options] <config file>"
    
    cmdlineParser = optparse.OptionParser(usage=usage)

    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--dir", action="store",
                             default='/axioma/products/current/riskmodels/dbinconsistencies', dest="dirName",
                             help="directory to store results in")
    cmdlineParser.add_option("-s", "--sql-server", action="store_true",
                             default=False, dest="sqlserver",
                             help="runReports against SQL server also")

    (options, args) = cmdlineParser.parse_args()
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    
    configFile_ = open(args[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    # process command-line options to override config file
    for arg in args[1:]:
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


    if gethostname().split('.')[0] not in ['nyc-plcncr01','nyc-plcncr02','nyc-plcncr03', 'nyc-plcncr04', 'atc-blcncr01', 'atc-blcncr02', 'atc-blcncr03', 'atc-blcncr04','cordelia', ]:
        logging.error("This programs requires access to the PostgresQL database and is permissioned only from moon,dione,larissa,ceres and helios currently")
        sys.exit(1)

    pgDB_, modelDB_ = createConnections(config_, options.sqlserver)

    reportString_ = runReports(config_, modelDB_, pgDB_, options.dirName, options.sqlserver)

    sendEmail(config_, reportString_)

#    modelDB_.close()
    pgDB_.close()


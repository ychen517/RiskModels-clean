import configparser
import cx_Oracle
import logging
import optparse
import os
import subprocess
from subprocess import TimeoutExpired
import sys
import time

from riskmodels import Connections, Utilities

FIND_QUERY="""select im.modeldb_id From modeldb_global.issue_map im where MARKETDB_ID not in (select axioma_ID from asset_ref)"""

REPORT_QUERY="""SELECT MODELDB_ID, MARKETDB_ID, AREF.FROM_DT, AREF.THRU_DT, SRC_ID, REF, ADD_DT,
       AREF.ACTION AS ASSET_REF_ACTION, ACTION_DT AS ASSET_REF_ACTION_DT, TRADING_COUNTRY
FROM MODELDB_GLOBAL.ISSUE_MAP  IM ,
    MARKETDB_GLOBAL.ASSET_REF_LOG  AREF 
WHERE IM.MARKETDB_ID = AREF.AXIOMA_ID 
AND NOT EXISTS (SELECT 1 FROM MARKETDB_GLOBAL.ASSET_REF  A WHERE A.AXIOMA_ID = IM.MARKETDB_ID) 
AND AREF.ACTION = 'DELETE'"""


def main():
    usage = "usage: %prog [options] <config file> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addLogConfigCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", "--test", action="store_true",
                             default=False, dest="testOnly",
                             help="Test only")
    cmdlineParser.add_option("--delete-exposures", action="store_false",
                             default=True, dest="leaveExposures",
                             help="delete exposures")
    default_report_file = os.path.join('..','reports','%(yyyymm)s','modelIDCleanup_%(yyyymmdd)s.txt')
    cmdlineParser.add_option("--report-file", action="store",
                             default=default_report_file, dest="reportFile",
                             help="Report filename")
    
    (options_, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    datestr = args[1]
    date = Utilities.parseISODate(datestr)
    if '%' in options_.reportFile:
        options_.reportFile = options_.reportFile % { 'yyyymm' : date.strftime('%Y%m'), 
                                                      'yyyymmdd': date.strftime('%Y%m%d') }

    configFile_ = open(args[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()

    connections_ = Connections.Connections(config_)
    mktDB = connections_.marketDB
    mktCursor = mktDB.dbCursor
    mktCursor.execute(REPORT_QUERY)
    report = mktCursor.fetchall()
    dirName=os.path.dirname(options_.reportFile)
    if dirName and not os.path.isdir(dirName):
        os.makedirs(dirName)
    i=0
    while os.path.isfile(options_.reportFile + ".v" +str(i)):
        i=i+1
    options_.reportFile=options_.reportFile+'.v'+str(i)
    reportFile = open(options_.reportFile, 'w')
    fmtFile=open(options_.reportFile+'.fmt','w')
    reportFile.write('\t\tDeletion Report for %s\n\n' % date)
    reportFile.write('-'*132)
    reportFile.write('\n')
    reportFile.write('%10.10s\t|%10.10s\t|%10.10s\t|%10.10s\t|%10.10s\t|%s\t|%s\n' % ('Model ID','Market ID','Add Date','From Date', 'Thru Date','Delete Date', 'Country'))
    fmtFile.write('Assets deleted\n')
    fmtFile.write('%10.10s|%10.10s|%10.10s|%10.10s|%10.10s|%s|%s|%s\n' % ('Model ID','Market ID','Add Date','From Date', 'Thru Date','Delete Date', 'Country', 'REF'))
    reportFile.write('-'*132)
    reportFile.write('\n')
    for r in report:
        reportFile.write('%10.10s\t|%10.10s\t|%10.10s\t|%10.10s\t|%10.10s\t|%10.10s\t|%2.2s\n' % \
                         (r[0], r[1], r[6] and r[6].date() or '(null)', r[2] and r[2].date() or '(null)',
                          r[3] and r[3].date() or '(null)', r[8] and r[8].date() or '(null)', r[9]))
        fmtFile.write('%10.10s|%10.10s|%10.10s|%10.10s|%10.10s|%10.10s|%2.2s|%s\n' % \
                         (r[0], r[1], r[6] and r[6].date() or '(null)', r[2] and r[2].date() or '(null)',
                          r[3] and r[3].date() or '(null)', r[8] and r[8].date() or '(null)', r[9], r[5]))
        reportFile.write('\tref: %s\n' % r[5])
    reportFile.close()
    fmtFile.close()
    
    mktCursor.execute(FIND_QUERY)
    #print(mktCursor.fetchall())
    modelIDs = ','.join(r[0] for r in mktCursor.fetchall())
    if len(modelIDs) == 0:
        logging.info("No assets to delete. Exiting")
        return 0
    logging.debug('modelIDs: %s', modelIDs)
    targetFile = "Tools/delMdlId.py"
    if not os.path.exists(os.path.join(os.getcwd(), targetFile)):
        logging.fatal("Can't find file %s to execute", targetFile)
        return 2
    command = ['python3', os.path.join(os.getcwd(), targetFile)]
    if not options_.testOnly:
        command.append('--update-database')
    if not options_.leaveExposures:
        command.append('--delete-exposures')
    command.append(args[0])
    command.append(modelIDs)
    logging.info('Executing %s:' % ' '.join(command))
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr,
                               universal_newlines=True)
    process.wait()
    if process.returncode != 0:
        logging.error("Status from deletion is %d", process.returncode)
    return process.returncode

if __name__ == '__main__':
    sys.exit(main())
    

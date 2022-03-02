
import datetime
from jenkinsapi import jenkins
import logging
import optparse
import pymssql
import sys
from time import sleep
import urllib.request, urllib.error, urllib.parse

from riskmodels import LegacyUtilities as Utilities

BASE_URL = 'http://%(server)s:%(port)d/'

#SP_EXECUTE = "exec MarketData.dbo.DerivedDataCurveGenNodeCntReportFilter @ShortNameFilter='%(currency)s', @LastRunDate='%(dt)s'"
SP_EXECUTE = ""

if __name__ == '__main__':
    usage = "usage: %prog [options] date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-s", "--server", action="store",
                             default="calypso", dest="server",
                             help="Server which runs derivation jobs")
    cmdlineParser.add_option("-p", "--port", action="store",
                             default="8080", dest="port", type="int",
                             help="Port on server for derivations")
    cmdlineParser.add_option("-j", "--job-name", action="store",
                             default="FuturesCurveGen_PROD_DRYRUN", dest="jobName",
                             help="Jenkins job to run")
    cmdlineParser.add_option("--db-user", action="store",
                             default="DataGenOpsFI", dest="dbUser",
                             help="MS-SQL username")
    cmdlineParser.add_option("--db-passwd", action="store",
                             default="DataGenOpsFI1234", dest="dbPasswd",
                             help="MS-SQL password")
    cmdlineParser.add_option("--db-host", action="store",
                             default="dev-mac-db", dest="dbHost",
                             help="MS-SQL hostname")
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="Don't run the jenkins commands")
 
    (options_, args_) = cmdlineParser.parse_args()

    if len(args_) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    if len(args_[0]) != 10:
        cmdlineParser.error("Invalid date %s" % args_[0])
    try:
        dt = Utilities.parseISODate(args_[0])
    except:
        cmdlineParser.error("Invalid date %s" % args_[0])
        
   
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    if not options_.testOnly:
        jenkinsObj = jenkins.Jenkins(BASE_URL % {'server': options_.server, 'port': options_.port})
        #print BASE_URL % {'server': options_.server, 'port': options_.port}
        jobObj = jenkinsObj.get_job(options_.jobName)
        #def invoke(self, securitytoken=None, block=False, skip_if_running=False, invoke_pre_check_delay=3, invoke_block_delay=15, params={}):
        #print 'CurveMatch=',currency, 'EndDate=', dt
        jobObj.invoke(block=True, invoke_block_delay=5, params={'AsOfDate': dt})
        build = jobObj.get_last_build()
        outputurl=build.baseurl.strip('/') + '/console'
        print('<a href="%s">Console output for %s</a>' % (outputurl, outputurl)) 
        if build.get_status() != 'SUCCESS':
            logging.error("Result for Constant Maturity Future is %s", build.get_status())
            sys.exit(1)
    
    if len(SP_EXECUTE)==0:
        sys.exit(0)

    # get report
    database = pymssql.connect(user=options_.dbUser,
                               password=options_.dbPasswd,
                               host=options_.dbHost)
    database.autocommit(True)
    cursor = database.cursor()
    cursor.execute(SP_EXECUTE % {'dt': dt})
    dbResults = cursor.fetchall()
    headers = [d[0] for d in cursor.description]
    maxlength = [len(str(d or '')) for d in headers]
    rowlength = [0 for d in headers]
    for row in dbResults:
        for idx,d in enumerate(row):
            if len(str(d or '')) > rowlength[idx]:
                rowlength[idx]=len(str(d or ''))
        
    for idxval,lengths in enumerate(zip(maxlength,rowlength)):
        old,new=lengths
        if old < new:
            maxlength[idxval]=rowlength[idxval]
    totallength=sum(maxlength) + 3 * len(maxlength)
    fmtstr=' | '.join(['%%%ds' % x for x in maxlength])
    errorcase=False
    print('Results:')
    print('%s' % ('-'*totallength))
    print(fmtstr % tuple(headers))
    print('%s' % ('-'*totallength))
    for row in dbResults:
        print(fmtstr % row)
        if row[0] != 'pass':
            errorcase=True
    print('%s' % ('-'*totallength))
    if errorcase:
        sys.exit(1)
    

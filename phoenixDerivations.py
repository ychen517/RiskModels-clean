
import datetime
from jenkinsapi import jenkins
import logging
import optparse
import pymssql
import sys
from time import sleep
import urllib.request, urllib.error, urllib.parse

from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities

WAIT_TIME=3600
BASE_URL = 'http://%(server)s:%(port)d/'
TARGET_URL = 'http://%(server)s:%(port)d/job/%(job)s/buildWithParameters?CurveMatch=%(currency)s&EndDate=%(dt)s'
RESULT_URL = 'http://%(server)s:%(port)d/job/%(job)s/lastBuild/api/python'

import urllib.request, urllib.parse, urllib.error
BEI_DICT={
'US':'US.USD.BEI.ZC',
'GB':'GB.GBP.BEI.ZC',
'DE':'DE.EUR.BEI.ZC',
'AU':'AU.AUD.BEI.ZC',
'FR':'FR.EUR.BEI.ZC',
'CA':'CA.CAD.BEI.ZC',
'SE':'SE.SEK.BEI.ZC',
'BR':'BR.BRL.BEI.ZC.NTNB BR.BRL.BEI.ZC.NTNC',
}

JOBS={ 
'SwedishBEI': 'Swedish BEI PROD',
'BEI': 'General BEI PROD' }

SP_EXECUTE = {
'BEI':"""
USE MARKETDATA

SELECT 
  CASE 
       WHEN COUNT(TenorEnum)=0 THEN 'fail - no nodes'
       WHEN COUNT(Quote)=0 THEN 'fail- all quotes missing'
       WHEN COUNT(Quote) < COUNT(TenorEnum) THEN 'fail- some quotes missing'
       ELSE 'pass'
  END as Message,
  COUNT(TenorEnum) as ExpectedCount,
  COUNT(Quote) as ObservedCount
from Curve
JOIN CurveNodes on CurveNodes.CurveId=Curve.CurveId
left JOIN CurveNodeQuote 
  on CurveNodeQuote.CurveNodeId=CurveNodes.CurveNodeId
  AND TradeDate='%(dt)s' -- *** TradeDate ***
where CurveTypeEnum='BEI.Zero'
and CurveShortName='%(curveshortname)s' -- *** CurveShortName ***
""",
'SwedishBEI': """
USE MarketData
SELECT case when count(*) = 1 then 'pass'
        else 'fail'
        end  status
from Curve
JOIN CurveNodes on CurveNodes.CurveId=Curve.CurveId
JOIN CurveNodeQuote on CurveNodeQuote.CurveNodeId=CurveNodes.CurveNodeId
where CurveTypeEnum='BEI.Zero'
and CurrencyEnum='%(currency)s'
AND TradeDate='%(dt)s'
""",
'CurveGen.Prod':"exec MarketData.dbo.DerivedDataCurveGenNodeCntReportFilter @ShortNameFilter='%(currency)s', @LastRunDate='%(dt)s'"
}

def getBuild(job,date, currency, jobName, curveshortname):
    """
       find the appropriate build given the jobObj, the date and currency
    """
    myBuild = None
    dt=str(date)[:10]
    for b in job.get_build_ids():
        build=job.get_build(b)

        found1 = found2 = False
        # based on the spec of jenkins parameters, hardcoding the first list element which contains the paramaters...would be nice if 
        # there was an API, but can't find one
        parameters = build._data['actions'][0]['parameters']
        for p in parameters:
            if jobName in ('BEI'):
                if p['name'] == 'TradeDate' and p['value'] == dt:
                    found1 = True
            else:
                if p['name'] == 'EndDate' and p['value'] == dt:
                    found1 = True
            # for curvegen.prod check the additional value for the match; otherwise simply allow it to pass
            if jobName not in ( 'SwedishBEI', 'BEI'):
                if p['name'] == 'CurveMatch' and p['value'] == currency:
                    found2 = True
            elif jobName in ('BEI'):
               if p['name'] == 'CurveShortName' and p['value'] == curveshortname:
                   found2 = True 
            else:
                found2 = True
        if found1 and found2:
            myBuild = build
            break
    return myBuild

def getResults(dt, currency, options_, curveshortname):
    """
       Get results, print it out and return true/false if error or not
    """
    if options_.noResults:
         return False, ""

    errorcase=False
    database = pymssql.connect(user=options_.dbUser,
                               password=options_.dbPasswd,
                               host=options_.dbHost)
    database.autocommit(True)
    cursor = database.cursor()
    cursor.execute(SP_EXECUTE[options_.jobName] % {'currency': currency,
                                 'dt': dt, 'curveshortname':curveshortname})
    dbResults = cursor.fetchall()
    if len(dbResults)==0:
        errorcase=True
        #return True, ""

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
    results='Results:'
    #print 'Results:'
    results='%s\n%s' %(results, ('-'*totallength))
    results='%s\n%s' % ( results, (fmtstr % tuple(headers)))
    results='%s\n%s' % (results,'%s' % ('-'*totallength))
    for row in dbResults:
        results='%s\n%s' % (results, (fmtstr % row))
        if row[0] != 'pass':
            errorcase=True
    results='%s\n%s' % (results, ( '-'*totallength))
    return errorcase, results

#http://calypso:8080/job/CurveGenFilter.Prod/buildWithParameters?CurveMatch=USD&EndDate=2013-08-02

if __name__ == '__main__':
    usage = "usage: %prog [options] country date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-s", "--server", action="store",
                             default="calypso", dest="server",
                             help="Server which runs derivation jobs")
    cmdlineParser.add_option("-p", "--port", action="store",
                             default="8080", dest="port", type="int",
                             help="Port on server for derivations")
    cmdlineParser.add_option("-j", "--job-name", action="store",
                             default="CurveGen.Dev", dest="jobName",
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
    cmdlineParser.add_option("--no-results", action="store_true",
                             default=False, dest="noResults",
                             help="Don't run the stored procedure for results")
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="Don't run the jenkins commands")
 
    (options_, args_) = cmdlineParser.parse_args()

    if len(args_) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    ctry = args_[0]
    if len(ctry) != 2:
        cmdlineParser.error("Invalid country %s" % ctry)
    if len(args_[1]) != 10:
        cmdlineParser.error("Invalid date %s" % args_[1])
    try:
        dt = Utilities.parseISODate(args_[1])
    except:
        cmdlineParser.error("Invalid date %s" % args_[1])
        
   
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    modelDB = ModelDB.ModelDB(user=options_.modelDBUser, passwd=options_.modelDBPasswd,
                              sid=options_.modelDBSID)
    query = """SELECT currency_code FROM RMG_CURRENCY rc
    JOIN risk_model_group rmg ON rmg.rmg_id=rc.rmg_id
    WHERE rmg.mnemonic=:ctry AND rc.from_dt<=:dt AND rc.thru_dt>:dt"""
    modelDB.dbCursor.execute(query, ctry=ctry, dt=dt)
    r = modelDB.dbCursor.fetchall()
    if not r or not r[0] or not r[0][0]:
        cmdlineParser.error("Can't find currency for country %s on date %s" % (ctry, dt))

    currency = r[0][0]
    errorcase=False
    status=None
    jenkinsObj = jenkins.Jenkins(BASE_URL % {'server': options_.server, 'port': options_.port})
    jobObj = jenkinsObj.get_job(JOBS.get(options_.jobName,options_.jobName))
    curveshortnames=['']
    if options_.jobName == 'BEI':
        curveshortnames=BEI_DICT[ctry].split(' ')
    if not options_.testOnly:
        if options_.jobName == 'BEI':
            # the BEI jobs have a different signature
            # because of BR, there might be multiple jobs to be kicked of as well
            for curveshortname in curveshortnames:
                jobObj.invoke(block=True, invoke_block_delay=5, params={'CurveShortName': curveshortname, 'TradeDate': dt})
        else:
            jobObj.invoke(block=True, invoke_block_delay=5, params={'CurveMatch': currency, 'EndDate': dt})
    builds=[]
    for curveshortname in curveshortnames:
        build= getBuild(jobObj, dt, currency, options_.jobName, curveshortname)
        if build:
            builds.append(build)
    if len(builds) == 0:
        logging.fatal('Could not find build for %s %s', dt, currency)
        sys.exit(1)

    for idx,build in enumerate(builds):
        outputurl=build.baseurl.strip('/') + '/console'
        print('<a href="%s">Console output for %s</a>' % (outputurl, outputurl)) 
   
    overallError = False

    for idx,build in enumerate(builds):
        # wait for a maximum of 1 hour to get build status, checking every 3 minutes or so; otherwise, give up and error
        status=build.get_status()
        waitTime=0
        while not status:
            logging.info('is good=%s',  build.is_good())
            logging.info('is running=%s', build.is_running())
            if not build.is_running():
                logging.warning('Build is not running anymore, so exit the wait')
                break
            build.print_data()
            waitTime += 180
            if waitTime >= WAIT_TIME:
                logging.error("Did not find status even after waiting %s seconds", WAIT_TIME)
                break
            logging.info("Sleeping for 180 seconds until we can get status")
            sleep(180) 
            status=build.get_status()
            if status == 'SUCCESS':
                break
            logging.error("Status for %s is %s", currency, status)
            # get report even if
            #exec MarketData.dbo.DerivedDataCurveGenNodeCntReportFilter @ShortNameFilter='EUR', @LastRunDate='11/14/12'
            errorcase,results=getResults(dt, currency, options_,curveshortnames[idx])
            if not errorcase:
                break
        
        logging.info('status=%s', status)
        errorcase,results=getResults(dt, currency, options_,curveshortnames[idx])
        logging.info('Done error=%s', errorcase)
        overallError = overallError or errorcase
        print(results)

    if overallError:
        sys.exit(1)
        

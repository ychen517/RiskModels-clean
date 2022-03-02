
import datetime
import logging
from jenkinsapi import api,jenkins
import optparse
import sys
from time import sleep

import generateFundamentalModel as generateFund
from riskmodels import ModelDB
from marketdb import MarketDB
from riskmodels import Utilities

WAIT_TIME = 1800

if 'urlquote' in dir(jenkins):
    _jenkinsNew=True
else:
    _jenkinsNew=False

def _get_status(b):
   if _jenkinsNew:
      if b.is_running():
          return None
      else:
          res=b.get_console()
          # hack to get last 20 characters of the file and parse it for SUCCESS or FAILURE
          if res[:25:].find('Finished: SUCCESS'):
             return 'SUCCESS'
          else:
             return 'FAILURE'
   else:
      return b.get_status()

def runNewJenkinsJob(jenkinsServer,jobName,params):
   try:
      jenkinsServer.build_job(jobName, params)
   except Exception as e :
      logging.exception('Jenkins build job return')
   sleep(15)


def runJenkins(jobName, startDt, endDt, options):
    jenkins = api.Jenkins('http://prod-mac-derived-curves:8080')
    job = jenkins[jobName]
    params={'StartDate': str(startDt), 
            'EndDate': str(endDt),
            'ModeldbUser': options.modelDBUser,
            'ModeldbSid': options.modelDBSID,
            'ModeldbPasswd': options.modelDBPasswd
            }
    params['Revision'] = options.modelRev
    if not options.testOnly:
        params['ExtraFlags'] = '--commit'
    if _jenkinsNew:
        logging.info('Running on Ubuntu 16....')
        runNewJenkinsJob(jenkins,jobName,params)
    else:
        try:
           job.invoke(params=params, block=True)
        except Exception as e :
           logging.exception('Jenkins build job return')
           sleep(15)
    job = jenkins[jobName]
    b = job.get_last_build()
    logging.info('%s',b)
    status = _get_status(b)
    waitTime = 0
    while not status:
        sleep(WAIT_TIME/10)
        waitTime += WAIT_TIME/10
        if waitTime >= WAIT_TIME:
            logging.error("Did not find status even after waiting %s seconds", WAIT_TIME)
            break
        status = _get_status(b)
    if status != 'SUCCESS':
        logging.error("Error running Jenkins build--status returned is %s", status)
    return status == 'SUCCESS'

def generateCommodityExposures(startDt, endDt, options):
    logging.info('Running exposures from %s to %s', startDt, endDt)
    return runJenkins('Commodity Factor Model - Generate Exposures', startDt, endDt, options)

def generateFactorReturns(startDt, endDt, options):
    logging.info('Running factor returns from %s to %s', startDt, endDt)
    return runJenkins('Commodity Factor Model - Estimate Factor Returns', startDt, endDt, options)

def runLoop(riskModel, dates, modelDB, marketDB, options):
    status = 0
    generateFund.options = options
    if options.runExposures:
        jenkinsStatus = generateCommodityExposures(dates[0], dates[-1], options)
        if not jenkinsStatus and not riskModel.forceRun:
            return 1
    if options.runFactors:
        jenkinsStatus = generateFactorReturns(dates[0], dates[-1], options)
        if not jenkinsStatus and not riskModel.forceRun:
            return 1
    if options.runCumFactors or options.runRisks or options.runTotalRiskBeta:
        for dt in dates:
            try:
                riskModel.setRiskModelGroupsForDate(dt)
                riskModel.setFactorsForDate(dt, modelDB)
                rmi = modelDB.getRiskModelInstance(riskModel.rms_id, dt)
                assert rmi.has_exposures, "No exposures for %s" % dt
                assert rmi.has_returns, "No returns for %s" % dt
    
                if options.runCumFactors:
                    generateFund.generateCumulativeFactorReturns(dt, riskModel, modelDB, 
                                                                 options.startCumulativeReturn and dt == dates[0])
                if options.runRisks or options.runTotalRiskBeta:
                    if options.runRisks:
                        generateFund.computeFactorSpecificRisk(dt, riskModel, modelDB, marketDB)
                    
                    if options.runTotalRiskBeta:
                        generateFund.computeTotalRisksAndBetas(dt, riskModel, modelDB, marketDB)
                    
                if options.testOnly:
                    logging.info('Reverting changes')
                    modelDB.revertChanges()
                else:
                    modelDB.commitChanges()
            except Exception:
                logging.error('Exception caught during processing', exc_info=True)
                modelDB.revertChanges()
                if not riskModel.forceRun:
                    status = 1
                    break
            logging.info('Finished %s processing for %s', options.modelName, dt)
            
    return status

if __name__ == '__main__':
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
    cmdlineParser.add_option("--risks", action="store_true",
                             default=False, dest="runRisks",
                             help="Generate factor covariances and specific risk")
    # Other options
    cmdlineParser.add_option("--all", action="store_true",
                             default=False, dest="runAll",
                             help="Full model run, do everything")
    cmdlineParser.add_option("--exposures", action="store_true",
                             default=False, dest="runExposures",
                             help="Generate estimation universe and factor exposures")
    cmdlineParser.add_option("--factors", action="store_true",
                             default=False, dest="runFactors",
                             help="Generate factor returns")
    cmdlineParser.add_option("--cum-factors", action="store_true",
                             default=False, dest="runCumFactors",
                             help="Generate cumulative factor returns")
    cmdlineParser.add_option("--start-cumulative-return", action="store_true",
                             default=False, dest="startCumulativeReturn",
                             help="On the first day, set cumulative returns to 1")
    cmdlineParser.add_option("--totalrisk-beta", action="store_true",
                             default=False, dest="runTotalRiskBeta",
                             help="Generate total risks and predicted betas")
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--dw", action="store_true",
                             default=False, dest="dontWrite",
                             help="don't even attempt to write to the database")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    if not options.modelRev:
        options.modelRev = riskModelClass.revision
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)


    if options.runAll:
        options.runExposures = True
        options.runFactors = True
        options.runCumFactors = True
        options.runRisks = True
        options.runTotalRiskBeta = True

    if options.runTotalRiskBeta:
        modelDB.setMarketCapCache(45)
    modelDB.setVolumeCache(502)

    riskModel = riskModelClass(modelDB, marketDB)
                 
    if len(args) == 1:
        dateRanges = [i.strip() for i in args[0].split(',')]
        dates = set()
        for dRange in dateRanges:
            if dRange.find(':') == -1:
                dates.add(Utilities.parseISODate(dRange))
            else:
                (startDate, endDate) = [i.strip() for i in dRange.split(':')]
                startDate = Utilities.parseISODate(startDate)
                endDate = Utilities.parseISODate(endDate)
                dates.update([startDate + datetime.timedelta(i)
                              for i in range((endDate-startDate).days + 1)])
        startDate = min(dates)
        endDate = max(dates)
        modelDates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, 
                                          excludeWeekend=True)
        dates = sorted(dates & set(modelDates))
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, 
                                     excludeWeekend=True)
    if options.verbose:
        riskModel.debuggingReporting = True
    if options.force:
        riskModel.forceRun = True
    
    status = runLoop(riskModel, dates, modelDB, marketDB, options)
    logging.info("Complete, status is %s", status == 0 and 'success' or 'failure')
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)


import datetime
import logging
import optparse
import sys
import os
import math
import io as StringIO
from multiprocessing import Process
import os.path
import numpy.ma as ma
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities

from generateEquityModel import generateCumulativeFactorReturns, generateEstimationUniverse, computeFactorSpecificRisk, computeTotalRisksAndBetas

def generateInitialFactorReturns(dates,riskModel,modelDB,marketDB, options):
    riskModel.setDatesForMacroCache([max(dates)],modelDB)
    data=riskModel.generateFactorReturns(max(dates),modelDB,marketDB,initial=True,modelDateLower=min(dates),force=options.force,skipMacroQA=options.skipMacroQA)
    
    fRetRaw=data.factorReturnsDFFinal.copy()
    badDates=sorted(set(dates).difference(fRetRaw.dropna().index))
    assert dates[-1] not in badDates
    if len(badDates)>0:
        logging.warning('Missing factor returns on %d dates up to  %s ', len(badDates),dates[-1])
    fRet=fRetRaw.reindex(index=dates).fillna(0.)
    if riskModel.debuggingReporting:
        logging.info('Sum of initial factor returns: %.6f', ma.sum(fRet.values, axis=None))
    if not options.dontWrite:
        riskModel._insertFactorReturnsFromDF(fRet,modelDB)

def generateExposures(date, riskModel, modelDB, marketDB, options):
    data=riskModel.generateExposuresAndSpecificReturns(date,modelDB,marketDB)
    if not options.dontWrite:
        rmi = riskModel.getRiskModelInstance(date, modelDB)
        riskModel.deleteExposures(rmi, modelDB)
        riskModel.insertExposures(rmi, data, modelDB, marketDB)
        riskModel._insertSpecificReturnsFromSeries(data.specificReturnsFinal,date,modelDB)
        rmi.setHasExposures(True, modelDB)

def generateFactorReturns(date, riskModel, modelDB, marketDB, options):
    data=riskModel.generateFactorReturns(date,modelDB,marketDB,initial=False,modelDateLower=date,force=options.force,skipMacroQA=options.skipMacroQA)
    if riskModel.debuggingReporting:
        logging.info('Sum of factor returns: %.6f', ma.sum(data.factorReturnsFinal, axis=None))
    if not options.dontWrite:
        rmi = riskModel.getRiskModelInstance(date, modelDB)
        riskModel._insertFactorReturnsFromSeries(data.factorReturnsFinal,date,modelDB)
        rmi.setHasReturns(True, modelDB)


def getLogFileName(dates, cmdoptions):
    prefix = None
    if cmdoptions.runFactors:
        prefix = 'factors'
    elif cmdoptions.runRisks:
        prefix = 'risks'
    elif cmdoptions.runExposures:
        prefix = 'exposures'
    elif cmdoptions.runESTU:
        prefix = 'estu'

    if prefix is None:
        prefix = str(min(dates))
    else:
        prefix = prefix + '_' + str(min(dates))


    fname = os.path.join('logs', cmdoptions.modelName+'_'+prefix + '_' + str(os.getpid())+'.log')
    return fname

def runLoop(dates, options, riskModelClass, riskModel=None, modelDB=None, marketDB=None, redirect=False):
    if redirect:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        fname = getLogFileName(dates, options)
        outfile = open(fname, 'w')
        sys.stdout = outfile
        sys.stderr = outfile
        if os.path.exists('subprocess.config'):
            config = StringIO.StringIO(open('subprocess.config', 'r').read().replace('log_file_name', fname))
            logging.config.fileConfig(config, disable_existing_loggers=0)

    ownsModelDB=False
    if modelDB is None:
        ownsModelDB=True
        modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                          user=options.modelDBUser,
                          passwd=options.modelDBPasswd)
    ownsMarketDB=False
    if marketDB is None:
        ownsMarketDB=True
        marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                         user=options.marketDBUser,
                         passwd=options.marketDBPasswd)
    if riskModel is None:
        riskModel = riskModelClass(modelDB, marketDB)
    riskModel.setDatesForMacroCache(dates,modelDB)

    if options.runExposures or options.runFactors or options.runRisks or options.runInitialFactorReturns: 
        bigCacheSize=3000
        modelDB.setVolumeCache(bigCacheSize-500) 
        modelDB.setTotalReturnCache(bigCacheSize)
        modelDB.setNotTradedIndCache(bigCacheSize)
        modelDB.setProxyReturnCache(bigCacheSize)
        modelDB.setMarketCapCache(400)
        modelDB.setSpecReturnCache(bigCacheSize)
        modelDB.setRiskFreeRateCache(bigCacheSize)
        modelDB.setFactorReturnCache(bigCacheSize)
        #Set up the currency cache, potentially from pickle
        modelDB.currencyCache = None
        if options.goFast and not options.runAll:
            fname = options.modelDBPickels.replace('YEAR', 'currencyCache.pickle')
            if os.path.exists(fname):
                modelDB.currencyCache = ModelDB.ForexCache.from_pickle(fname, marketDB)
        if modelDB.currencyCache is None:
            modelDB.createCurrencyCache(marketDB,days=bigCacheSize)

    if options.verbose:
        riskModel.debuggingReporting = True
    if options.force:
        riskModel.forceRun = True
    riskModel.runFCovOnly = options.runFCovOnly

    if options.goFast and not options.runAll:
        logging.warning('option go fast is set. this is not ok for production')
        if options.runInitialFactorReturns:
            modelDB.loadCaches(options.modelDBPickels.replace('YEAR',str(max(dates).year)))
        else:
            modelDB.loadCaches(options.modelDBPickels.replace('YEAR',str(min(dates).year)))

    if options.runInitialFactorReturns:
        try:
            logging.info('Building initial factor return history: start')
            generateInitialFactorReturns(dates,riskModel,modelDB,marketDB,options)
            modelDB.commitChanges()
            logging.info('Building initial factor return history: end')
        except:
            logging.error('Exception caught during initial factor return processing', exc_info=True)
            modelDB.revertChanges()
            status = 1
            if ownsModelDB:
                modelDB.finalize()
            if ownsMarketDB:
                marketDB.finalize()
            return status


    status = 0
    for d in dates:
        try:
            riskModel.setFactorsForDate(d, modelDB)
            if options.runESTU or options.runESTUOnly:
                generateEstimationUniverse(d, riskModel, modelDB, marketDB, options)

            if options.runFactors:
                generateFactorReturns(d, riskModel, modelDB, marketDB, options)
            
            if options.runExposures:
                generateExposures(d, riskModel, modelDB, marketDB, options)
            
            rmi = riskModel.getRiskModelInstance(d, modelDB)
            if options.runRisks:
                computeFactorSpecificRisk(rmi, d, riskModel, modelDB, marketDB, options)
            
            if options.runCumFactors:
                generateCumulativeFactorReturns(rmi, d, riskModel, modelDB, 
                        options.startCumulativeReturn and d == dates[0], options)
            
            if options.runTotalRiskBeta:
                computeTotalRisksAndBetas(rmi, d, riskModel, modelDB, marketDB, options)
            
            if options.testOnly:
                logging.info('Reverting changes')
                modelDB.revertChanges()
            else:
                modelDB.commitChanges()
            logging.info('Finished %s processing for %s', options.modelName, d)
        except Exception:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            status = 1
            break

    if ownsModelDB:
        modelDB.finalize()
    if ownsMarketDB:
        marketDB.finalize()

    return status


def runmain():
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
    cmdlineParser.add_option("--estu", action="store_true",
                             default=False, dest="runESTU",
                             help="Generate model and estimation universes")
    cmdlineParser.add_option("--estu-only", action="store_true",
                             default=False, dest="runESTUOnly",
                             help="Generate only new estimation universe structure")
    cmdlineParser.add_option("--initial-factor-returns", action="store_true",
                             default=False, dest="runInitialFactorReturns",
                             help="Generate history of factor returns prior to exposure dates.")
    cmdlineParser.add_option("--exposures", action="store_true",
                             default=False, dest="runExposures",
                             help="Generate factor exposures")
    cmdlineParser.add_option("--factors", action="store_true",
                             default=False, dest="runFactors",
                             help="Generate factor returns")
    cmdlineParser.add_option("--risks", action="store_true",
                             default=False, dest="runRisks",
                             help="Generate factor covariances and specific risk")
    # Other options
    cmdlineParser.add_option("--all", action="store_true",
                             default=False, dest="runAll",
                             help="Full model run, do everything")
    cmdlineParser.add_option("--cum-factors", action="store_true",
                             default=False, dest="runCumFactors",
                             help="Generate cumulative factor returns")
    cmdlineParser.add_option("--start-cumulative-return", action="store_true",
                             default=False, dest="startCumulativeReturn",
                             help="On the first day, set cumulative returns to 1")
    cmdlineParser.add_option("--totalrisk-beta", action="store_true",
                             default=False, dest="runTotalRiskBeta",
                             help="Generate total risks and predicted betas (not implemented)")
    cmdlineParser.add_option("--cov-only", action="store_true",
                             default=False, dest="runFCovOnly",
                             help="Skip time-consuming specific risk step")
    cmdlineParser.add_option("--modeldb-cache-dir",
                             default="modeldb_pickles/YEAR", dest="modelDBPickels",
                             help="A directory to store and load modeldb caches.")
    cmdlineParser.add_option("--go-fast", action="store_true",
                             default=False, dest="goFast",
                             help="Warmstart modeldb caches")
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--dw", action="store_true",
                             default=False, dest="dontWrite",
                             help="don't even attempt to write to the database")
    cmdlineParser.add_option("--dw-old-beta", action="store_true",
                             default=False, dest="dontWriteOldBeta",
                             help="don't even attempt to write old betas to the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    cmdlineParser.add_option("--skip-macro-qa", action="store_true",
                             default=False, dest="skipMacroQA",
                             help="Do not check quality of time series when computing factor returns.")
    cmdlineParser.add_option("--v3", "--V3", action="store_true",
                             default=False, dest="v3",
                             help="run newer versions of some code")
    cmdlineParser.add_option("--ncpu", dest="ncpu", default=1, type='int',
                             help="Number of cpus to use")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                          user=options.modelDBUser,
                          passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                             user=options.marketDBUser,
                             passwd=options.marketDBPasswd)

    if options.runAll:
        options.runESTU = True
        options.runExposures = True
        options.runFactors = True
        options.runCumFactors = True
        options.runRisks = True
        options.runTotalRiskBeta = True
    options.preliminary = False

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
        dates = sorted(modelDB.getDateRange(riskModel.rmg, startDate, endDate, 
                                     excludeWeekend=True))

    status = 0
    if options.ncpu == 1:
        status = runLoop(dates, options, riskModelClass, riskModel, modelDB, marketDB)
    else:
        numdates = int(math.ceil(len(dates)/float(options.ncpu)))
        processes = []
        for datelist in Utilities.chunks(dates, numdates):
            p = Process(target=runLoop, args=(datelist, options, riskModelClass, None, None, None, True))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)

if __name__ == '__main__':
    runmain()

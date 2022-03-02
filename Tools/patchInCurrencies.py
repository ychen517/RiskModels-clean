
import datetime
import logging
import numpy.ma as ma
import numpy
import optparse
import sys
from marketdb import MarketDB
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import MFM

def generateCumulativeFactorReturns(currDate, riskModel, modelDB, startCumReturns, options):
    logging.info('Processing cumulative factor returns for %s', currDate)
    tradingDaysList = modelDB.getDates(riskModel.rmg, currDate, 5)
    if len(tradingDaysList) == 0 or tradingDaysList[-1] != currDate:
        logging.info('%s is not a trading day. Skipping cumulative returns.',
                      currDate)
        return
    if len(tradingDaysList) == 1:
        logging.info('No previous trading day prior to %s', currDate)
    else:
        # Get risk model instances for prior days and use the closest one
        assert(len(tradingDaysList) >= 2)
        prevDates = tradingDaysList[:-1]
        prevRMIs = modelDB.getRiskModelInstances(riskModel.rms_id, prevDates)
        if len(prevRMIs) == 0:
            logging.error('Skipping %s. No model instance in prior five trading day',
                          currDate)
            return
        prevRMIs = sorted([(rmi.date, rmi) for rmi in prevRMIs])
        prevRMI = prevRMIs[-1][1]
        prevDate = prevRMI.date
    currRMI = modelDB.getRiskModelInstance(riskModel.rms_id, currDate)
    if currRMI is None or not currRMI.has_returns:
        logging.warning('Skipping %s because risk model is missing', currDate)
        return

    # Get list of current additional currency subfactors
    crmi = modelDB.getRiskModelInstance(riskModel.currencyModel.rms_id, currDate)
    assert(crmi is not None)
    (currencyReturns, currencyFactors) = \
            riskModel.currencyModel.loadCurrencyReturns(crmi, modelDB)
    currencyFactors = [c for c in currencyFactors if c in riskModel.additionalCurrencyFactors]
    currSubFactors = modelDB.getSubFactorsForDate(currDate, currencyFactors)
    currFactorReturns = modelDB.loadFactorReturnsHistory(
            riskModel.rms_id, currSubFactors, [currDate])

    # Load in current cumulative factor returns - take note of missing returns
    if options.missingOnly:
        currentCumReturns = modelDB.loadCumulativeFactorReturnsHistory(
                riskModel.rms_id, currSubFactors, [currDate])
        missingCurrentReturnsIdx = numpy.flatnonzero(ma.getmaskarray(
            currentCumReturns.data[:,0]))
        nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(
            currentCumReturns.data[:,0])==0)
        nonMissingSF = [currSubFactors[idx].factor.name for idx in nonMissingIdx]
        if len(nonMissingSF) > 0:
            logging.info('Not updating the following: %s', nonMissingSF)

    if not startCumReturns:
        riskModel.setFactorsForDate(prevDate, modelDB)
        prevSubFactors = modelDB.getSubFactorsForDate(
            prevDate, riskModel.factors)
        cumFactorReturns = modelDB.loadCumulativeFactorReturnsHistory(
            riskModel.rms_id, prevSubFactors, [prevDate])
        riskModel.setFactorsForDate(currDate, modelDB)
         
        # Map subfactors in cumulative returns to those in current returns
        # and fill missing cumulative returns with ones.
        cumFactorRetMap = dict([(i.subFactorID, j) for (i,j)
             in zip(cumFactorReturns.assets, cumFactorReturns.data[:,0])])
        mappedCumFactorReturns = Matrices.allMasked(
            (len(currFactorReturns.assets), 1))
        for (idx, subFactor) in enumerate(currFactorReturns.assets):
            mappedCumFactorReturns[idx,0] = cumFactorRetMap.get(
                subFactor.subFactorID, ma.masked)
        newCumFactorReturns = mappedCumFactorReturns \
                              * (currFactorReturns.data + 1.0)

        # If the sub-factor cumulative return is missing and the corresponding
        # factor starts today, set the cumulative return to 1.0
        missingIdx = numpy.flatnonzero(newCumFactorReturns.mask)
        for idx in missingIdx:
            subFactor = currFactorReturns.assets[idx]
            modelDB.dbCursor.execute("""SELECT rf.from_dt, rf.thru_dt
              FROM rms_factor rf
              JOIN sub_factor sf ON rf.factor_id=sf.factor_id
                WHERE sf.sub_id=:sfid AND sf.from_dt <= :dt
                AND sf.thru_dt > :dt AND rf.rms_id=:rms""",
                                     sfid=subFactor.subFactorID,
                                     dt=currDate, rms=riskModel.rms_id)
            r = modelDB.dbCursor.fetchone()
            if r is None:
                raise ValueError('cannot map sub-factor ID %d to rms_factor on %s' % (subFactor.subFactorID, currDate))
            fromDt = r[0].date()
            if prevDate < fromDt and fromDt <= currDate:
                logging.warning(
                    'factor %s starts today, setting cumulative return to 1',
                    subFactor.factor.name)
                newCumFactorReturns[idx,0] = 1.0
        missingIdx = numpy.flatnonzero(newCumFactorReturns.mask)
        if len(missingIdx) > 0:
            raise ValueError('missing cumulative factor returns for %s (%s)'
                             % (currDate, ','.join([currFactorReturns.assets[i].factor.name for i in missingIdx])))
    else:
        newCumFactorReturns = numpy.ones((len(currSubFactors), 1), float)

    
    if not options.dontWrite:

        if options.missingOnly:
            currFactorReturns.assets = numpy.take(currFactorReturns.assets, missingCurrentReturnsIdx, axis=0)
            newCumFactorReturns = ma.take(newCumFactorReturns[:,0], missingCurrentReturnsIdx, axis=0)
        else:
            newCumFactorReturns = newCumFactorReturns[:,0]

        for (idx, ret) in enumerate(newCumFactorReturns):
            logging.debug('New return: %s, %s', currFactorReturns.assets[idx].factor.name, ret)

        modelDB.updateCumulativeFactorReturns(
            riskModel.rms_id, currDate, currFactorReturns.assets,
            newCumFactorReturns)

def patchInReturns(date, riskModel, modelDB, marketDB, options):
    logging.info('Processing factor returns for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return

    # Load currency factor returns
    crmi = modelDB.getRiskModelInstance(riskModel.currencyModel.rms_id, date)
    assert(crmi is not None)
    (currencyReturns, allCurrencyFactors) = \
            riskModel.currencyModel.loadCurrencyReturns(crmi, modelDB)
    logging.info('loaded %d currencies from currency model', len(allCurrencyFactors))
    currencyFactorIdxMap = dict(zip(allCurrencyFactors, range(len(allCurrencyFactors))))
    currencyReturnsMap = dict(zip(allCurrencyFactors, currencyReturns))

    notIn = [i for i in allCurrencyFactors if i not in riskModel.factors]
    if len(notIn) > 0 and (len(riskModel.additionalCurrencyFactors) > 0):
        logging.info('%s live currencies not in the model: %s', len(notIn), [f.name for f in notIn])

    # Subset only those that are extra to the model
    currencyFactors = [c for c in allCurrencyFactors if c in riskModel.additionalCurrencyFactors]
    currSubFactors = modelDB.getSubFactorsForDate(date, currencyFactors)

    # Load in current cumulative factor returns - take note of missing returns
    missingCumReturns = list(currSubFactors)
    if options.missingOnly:
        currentCumReturns = modelDB.loadCumulativeFactorReturnsHistory(
                riskModel.rms_id, currSubFactors, [date])
        missingCurrentReturnsIdx = numpy.flatnonzero(ma.getmaskarray(
            currentCumReturns.data[:,0]))
        missingCumReturns = numpy.take(currSubFactors, missingCurrentReturnsIdx, axis=0)
        nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(currentCumReturns.data[:,0])==0)
        nonMissingSF = [currSubFactors[idx].factor.name for idx in nonMissingIdx]
        if len(nonMissingSF) > 0:
            logging.info('Not updating the following: %s', nonMissingSF)

    currSubFactorList = []
    currencyFactorReturnList = []
    # Loop round and pick out all valid returns
    for (cf, sf) in zip(currencyFactors, currSubFactors):
        if sf not in missingCumReturns:
            continue
        value = currencyReturnsMap[cf]
        currSubFactorList.append(sf)
        currencyFactorReturnList.append(value)
        logging.debug('New returns: %s, %s', sf, value)

    # Add currency returns to model factor returns table
    if len(currSubFactorList) > 0:
        modelDB.deleteFactorReturns(rmi, subFactors=currSubFactorList)
        modelDB.insertFactorReturns(riskModel.rms_id, date,
                currSubFactorList, currencyFactorReturnList)
    else:
        logging.info('No returns to update')

def patchInCovariances(date, riskModel, modelDB, marketDB, options):
    logging.info('Processing factor/specific risk for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return

    # Deal with covariances
    data = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB)
    modelDB.deleteRMIFactorCovMatrix(rmi)
    riskModel.insertFactorCovariances(
            rmi, data.factorCov, data.subFactors, modelDB)

def runLoop(riskModel, dates, modelDB, marketDB, options):
    status = 0
    steps = ''
    for d in dates:
        try:
            riskModel.setFactorsForDate(d, modelDB)
            
            if options.runFactors:
                patchInReturns(d, riskModel, modelDB, marketDB, options)
                if d == dates[0] or len(steps) == 0:
                    steps += 'returns,'
            
            if options.runCumFactors:
                generateCumulativeFactorReturns(d, riskModel, modelDB, 
                        options.startCumulativeReturn and d == dates[0], options)
                if d == dates[0] or len(steps) == 0:
                    steps += 'cumrets,'
            
            if options.runRisks:
                patchInCovariances(d, riskModel, modelDB, marketDB, options)
                if d == dates[0] or len(steps) == 0:
                    steps += 'cov,'
            
            if options.testOnly:
                logging.info('Reverting changes')
                modelDB.revertChanges()
            else:
                modelDB.commitChanges()
            logging.info('Finished %s [%s] processing for %s', options.modelName, steps, d)
        except Exception:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            if not riskModel.forceRun:
                status = 1
                break
    return status

def runmain():
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
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
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)
    riskModel = riskModelClass(modelDB, marketDB)

    if options.startCumulativeReturn:
        options.runCumFactors = True

    options.statModel = False
    if riskModel.isStatModel():
        options.statModel = True

    if options.runAll:
        options.runFactors = True
        options.runCumFactors = True
        options.runRisks = True

    if options.runFactors and not options.runRisks:
        modelDB.factorReturnCache = None
        modelDB.totalReturnCache = None

    modelDB.setVolumeCache(502)

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
    options.missingOnly = True
    if options.force:
        riskModel.forceRun = True
        options.missingOnly = False
    
    status = runLoop(riskModel, dates, modelDB, marketDB, options)
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)

if __name__ == '__main__':
    runmain()

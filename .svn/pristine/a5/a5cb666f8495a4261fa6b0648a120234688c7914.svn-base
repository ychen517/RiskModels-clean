
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

def generateEstimationUniverse(date, riskModel, modelDB, marketDB):
    logging.info('Processing model and estimation universes for %s', date)
    data = riskModel.generate_model_universe(date, modelDB, marketDB)
    if not options.dontWrite:
        riskModel.deleteInstance(date, modelDB)
        rmi = riskModel.createInstance(date, modelDB)
        rmi.setIsFinal(not options.preliminary, modelDB)
        riskModel.insertEstimationUniverse(rmi, data.universe,
                data.estimationUniverseIdx, modelDB, 
                getattr(data, 'ESTUQualify', None))
        modelDB.insertExposureUniverse(rmi, data.universe)

def generateExposures(date, riskModel, modelDB, marketDB, addNew=False):
    logging.info('Processing exposures for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return
    data = riskModel.generateExposureMatrix(date, modelDB, marketDB)
    if not options.dontWrite:
        if addNew:
            riskModel.insertExposures(rmi, data, modelDB, marketDB, update=True)
        else:
            riskModel.deleteExposures(rmi, modelDB)
            riskModel.insertExposures(rmi, data, modelDB, marketDB)
        rmi.setHasExposures(True, modelDB)

def generateFactorAndSpecificReturns(date, riskModel, modelDB, marketDB):
    logging.info('Processing factor/specific return for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return
    if not rmi.has_exposures:
        logging.warning('No exposures in risk model instance for %s, skipping',
                     date)
        return
    result = riskModel.generateFactorSpecificReturns(
                    modelDB, marketDB, date)
    if not options.dontWrite:
        modelDB.deleteFactorReturns(rmi)
        modelDB.deleteSpecificReturns(rmi)
        modelDB.deleteRMSStatistics(rmi)
        modelDB.deleteRMSFactorStatistics(rmi)
        riskModel.insertFactorReturns(
                date, result.factorReturns, modelDB)
        riskModel.insertSpecificReturns(date, result.specificReturns,
                result.exposureMatrix.getAssets(), modelDB)
        riskModel.insertRegressionStatistics(date,
                result.regressionStatistics,
                None,
                result.adjRsquared, None, modelDB)
        riskModel.insertEstimationUniverseWeights(rmi,
                result.regression_ESTU, modelDB)
        rmi.setHasReturns(True, modelDB)
    logging.info('Finished processing factor/specific return for %s', date)

def generateCumulativeFactorReturns(currDate, riskModel, modelDB, startCumReturns):
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
    currSubFactors = modelDB.getSubFactorsForDate(
        currDate, riskModel.factors)
    currFactorReturns = modelDB.loadFactorReturnsHistory(
        riskModel.rms_id, currSubFactors, [currDate])
    if not startCumReturns:
        riskModel.setFactorsForDate(prevDate, modelDB)
        prevSubFactors = modelDB.getSubFactorsForDate(
            prevDate, riskModel.factors)
        cumFactorReturns = modelDB.loadCumulativeFactorReturnsHistory(
            riskModel.rms_id, prevSubFactors, [prevDate])
        riskModel.setFactorsForDate(currDate, modelDB)
        # Map subfactors in cumulative returns to those in current returns
        # and fill missing cumulative returns with ones.
        cumFactorRetMap = dict(
            [(i.subFactorID, j) for (i,j)
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
    
    modelDB.updateCumulativeFactorReturns(
        riskModel.rms_id, currDate, currFactorReturns.assets,
        newCumFactorReturns[:,0])

def computeFactorSpecificRisk(date, riskModel, modelDB, marketDB):
    logging.info('Processing factor/specific risk for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return
    if not rmi.has_returns:
        logging.warning('No factor returns in risk model instance for %s, skipping',
                     date)
        return
    data = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB)
    if not options.dontWrite:
        modelDB.deleteRMIFactorSpecificRisk(rmi)
        riskModel.insertFactorCovariances(
                rmi, data.factorCov, data.subFactors, modelDB)
        riskModel.insertSpecificRisks(
                rmi, data.specificVars, data.subIssues, data.specificCov, modelDB)
        rmi.setHasRisks(True, modelDB)

def computeTotalRisksAndBetas(date, riskModel, modelDB, marketDB, v3=False):
    logging.info('Processing total risks and betas for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return
    if not rmi.has_risks:
        logging.warning('Incomplete risk model instance for %s, skipping', date)
        return
    modelData = Utilities.Struct()
    modelData.exposureMatrix = riskModel.loadExposureMatrix(rmi, modelDB)
    modelData.exposureMatrix.fill(0.0)
    (modelData.specificRisk, modelData.specificCovariance) = \
                        riskModel.loadSpecificRisks(rmi, modelDB)
    (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
        rmi, modelDB)
    modelData.factorCovariance = factorCov
    modelData.factors = factors
    if v3:
        predictedBeta = riskModel.computePredictedBetaV3(
            date, modelData, modelDB, marketDB)
        modelDB.deleteRMIPredictedBeta(rmi, v3=True)
        modelDB.insertRMIPredictedBetaV3(rmi, predictedBeta)
    else:
        totalRisk = riskModel.computeTotalRisk(modelData, modelDB)
        predictedBeta = riskModel.computePredictedBeta(
                date, modelData, modelDB, marketDB)
         
        modelDB.deleteRMIPredictedBeta(rmi)
        modelDB.insertRMIPredictedBeta(rmi, predictedBeta)
        modelDB.deleteRMITotalRisk(rmi)
        modelDB.insertRMITotalRisk(rmi, totalRisk)

def runLoop(riskModel, riskModelExp, dates, modelDB, marketDB, options):
    status = 0
    for d in dates:
        try:
            riskModel.setFactorsForDate(d, modelDB)
            if options.runESTU:
                generateEstimationUniverse(d, riskModel, modelDB, marketDB)
            
            if options.runExposures:
                riskModelExp.setFactorsForDate(d, modelDB)
                generateExposures(d, riskModelExp, modelDB, marketDB, options.addNewExposures)
            
            if options.runFactors:
                generateFactorAndSpecificReturns(d, riskModel, modelDB, marketDB)
            
            if options.runCumFactors:
                generateCumulativeFactorReturns(d, riskModel, modelDB, 
                        options.startCumulativeReturn and d == dates[0])
            
            if options.runRisks:
                computeFactorSpecificRisk(d, riskModel, modelDB, marketDB)
            
            if options.runTotalRiskBeta:
                computeTotalRisksAndBetas(d, riskModel, modelDB, marketDB, v3=options.v3)
            
            if options.testOnly:
                logging.info('Reverting changes')
                modelDB.revertChanges()
            else:
                modelDB.commitChanges()
            logging.info('Finished %s processing for %s', options.modelName, d)
        except Exception:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            if not riskModel.forceRun:
                status = 1
                break
    return status

if __name__ == '__main__':
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
    cmdlineParser.add_option("--estu", action="store_true",
                             default=False, dest="runESTU",
                             help="Generate model and estimation universe")
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
    cmdlineParser.add_option("--exp-all", action="store_true",
                             default=False, dest="rebuildAllExposures",
                             help="Rebuild entire large exposure table")
    cmdlineParser.add_option("--exp-add", action="store_true",
                             default=False, dest="addNewExposures",
                             help="Add or replace selected exposures")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    cmdlineParser.add_option("--v3", "--V3", action="store_true",
                             default=False, dest="v3",
                             help="run newer versions of some code")
    cmdlineParser.add_option("--preliminary", action="store_true",
                             default=False, dest="preliminary",
                             help="Preliminary run--ignore DR assets")
    
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
    if options.startCumulativeReturn:
        options.runCumFactors = True
    if options.runAll:
        options.runESTU = True
        options.runExposures = True
        options.runFactors = True
        options.runCumFactors = True
        options.runRisks = True
        options.runTotalRiskBeta = True

    if options.runFactors and not options.runRisks:
        modelDB.factorReturnCache = None
        modelDB.totalReturnCache = None

    if options.runTotalRiskBeta:
        modelDB.setMarketCapCache(45)
    modelDB.setVolumeCache(502)

    riskModel = riskModelClass(modelDB, marketDB)
    if options.rebuildAllExposures:
        riskModelExp = riskModelClass(modelDB, marketDB, expTreat='rebuild')
    elif options.addNewExposures:
        riskModelExp = riskModelClass(modelDB, marketDB, expTreat='addNew')
    else:
        riskModelExp = riskModel
                 
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
        riskModelExp.debuggingReporting = True
    if options.force:
        riskModel.forceRun = True
        riskModelExp.forceRun = True
    
    status = runLoop(riskModel, riskModelExp, dates, modelDB, marketDB, options)
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)

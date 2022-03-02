
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
from riskmodels import AssetProcessor
import riskmodels.LegacyProcessReturns as ProcessReturns

def runLoop(riskModel, dates, modelDB, marketDB, options):
    status = 0
    for d in dates:
        try:
            riskModel.setFactorsForDate(d, modelDB)
            rmi = modelDB.getRiskModelInstance(riskModel.rms_id, d)
            subFactors = modelDB.getSubFactorsForDate(d, riskModel.factors)

            # Load exposure matrix
            expM = riskModel.loadExposureMatrix(rmi, modelDB)

            # Determine home country info and flag DR-like instruments
            data = AssetProcessor.process_asset_information(
                    d, expM.getAssets(), riskModel.rmg, modelDB, marketDB,
                    checkHomeCountry=(riskModel.SCM==0),
                    legacyDates=riskModel.legacyMCapDates,
                    nurseryRMGList=riskModel.nurseryRMGs,
                    numeraire_id=riskModel.numeraire.currency_id,
                    forceRun=riskModel.forceRun)

            # Get estimation universe
            ids_ESTU = riskModel.loadEstimationUniverse(rmi, modelDB)
            ids_ESTU = [n for n in ids_ESTU if n in data.assetIdxMap]
            estu = [data.assetIdxMap[n] for n in ids_ESTU]

            # Market caps
            mcapDates = modelDB.getDates(riskModel.rmg, d, 19)
            mcaps = ma.filled(modelDB.getAverageMarketCaps(
                mcapDates, data.universe, riskModel.numeraire.currency_id, marketDB), 0.0)

            # Load asset returns
            returnsHistory = 1
            returnsProcessor = ProcessReturns.assetReturnsProcessor(
                    riskModel.rmg, data.universe, data.rmgAssetMap, data.assetTypeDict,
                    debuggingReporting=riskModel.debuggingReporting,
                    numeraire_id=riskModel.numeraire.currency_id,
                    returnsTimingID=riskModel.returnsTimingId)
            assetReturnMatrix = returnsProcessor.process_returns_history(
                    d, returnsHistory, modelDB, marketDB,
                    drCurrMap=data.drCurrData, loadOnly=True)

            # Load factor returns
            factorReturns = modelDB.loadFactorReturnsHistory(
                    riskModel.rms_id, subFactors, [d])
            specificReturns = modelDB.loadSpecificReturnsHistory(
                    riskModel.rms_id, data.universe, [d])

            # ISC info
            subIssueGroups = modelDB.getIssueCompanyGroups(d, data.universe, marketDB)
            specificReturns.nameDict = modelDB.getIssueNames(
                    specificReturns.dates[0], data.universe, marketDB)
            scores = riskModel.score_linked_assets(d, data.universe, modelDB, marketDB,
                    subIssueGroups=subIssueGroups)

            # Covariance matrix and specific risk
            (specificRisk, specificCovariance) = \
                    riskModel.loadSpecificRisks(rmi, modelDB)
            (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
                    rmi, modelDB)

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
                             default=False, dest="loadAllExposures",
                             help="Rebuild entire large exposure table")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    cmdlineParser.add_option("--v3", "--V3", action="store_true",
                             default=False, dest="v3",
                             help="run newer versions of some code")
    
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

    if options.loadAllExposures:
        riskModel = riskModelClass(modelDB, marketDB, expTreat='rebuild')
    else:
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
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)

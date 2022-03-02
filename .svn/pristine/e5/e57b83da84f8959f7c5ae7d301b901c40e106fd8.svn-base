
import logging
import optparse
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def computeCurrencyReturns(modelDB, marketDB, riskModel, date, options):
    """Computes currency returns.
    """
    if options.dontWrite:
        returns = riskModel.computeCurrencyReturns(date, modelDB, marketDB)
    else:
        riskModel.deleteInstance(date, modelDB)
        returns = riskModel.computeCurrencyReturns(date, modelDB, marketDB)
        rmi = riskModel.createInstance(date, modelDB)
        riskModel.insertCurrencyReturns(date, returns, modelDB)
        rmi.setHasReturns(True, modelDB)
        rmi.setIsFinal(True, modelDB)

def computeCurrencyStatRisk(modelDB, marketDB, riskModel, date, options):
    """Computes currency risks for a statistical currency risk model.
    """
    rmi = modelDB.getRiskModelInstance(riskModel.rms_id, date)
    if rmi != None:
        if options.dontWrite:
            results = riskModel.generateStatisticalModel(date, modelDB, marketDB)
        else:
            riskModel.deleteStatisticalModel(rmi, modelDB)
            results = riskModel.generateStatisticalModel(date, modelDB, marketDB)
            # exposures
            modelDB.insertExposureUniverse(rmi, results.universe)
            riskModel.insertExposures(rmi, results, modelDB, marketDB)
            riskModel.insertEstimationUniverse(rmi, results.universe, results.estimationUniverseIdx, modelDB)
            rmi.setHasExposures(True, modelDB)

            # factor & specific returns
            subIssues = results.srMatrix.assets
            riskModel.insertSpecificReturns(date, results.srMatrix.data[:,0],
                                            results.srMatrix.assets, modelDB)
            rmi.setHasReturns(True, modelDB)

            # factor covariances & specific risks
            riskModel.insertFactorCovariances(rmi, results.factorCov, modelDB)
            riskModel.insertSpecificRisks(rmi, results.specificVars,
                                        subIssues, modelDB)
            rmi.setHasRisks(True, modelDB)
    else:
        logging.error('No risk model instance for %s, skipping', date)

def processDay(riskModel, options, d, modelDB, marketDB):
    logging.info('Processing %s', d)
    riskModel.setRiskModelGroupsForDate(d)
    if options.runReturns:
        computeCurrencyReturns(modelDB, marketDB, riskModel, d, options)
    if options.runRisk:
        computeCurrencyStatRisk(modelDB, marketDB, riskModel, d, options)
    if options.testOnly:
        logging.info('Reverting changes')
        modelDB.revertChanges()
    else:
        modelDB.commitChanges()
    logging.info('Finished processing currencies %s' % d)

def main():
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--no-returns", action="store_false",
                         default=True, dest="runReturns",
                         help="Don't process the currency returns")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--no-risk", action="store_false",
                         default=True, dest="runRisk",
                         help="Don't process the currency risks")
    cmdlineParser.add_option("--dw", action="store_true",
                            default=False, dest="dontWrite",
                            help="don't even attempt to write to the database")
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
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
    startDate = Utilities.parseISODate(args[0])
    riskModel = riskModelClass(modelDB, marketDB)
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])
    
    dateList = modelDB.getDateRange(riskModel.rmg, startDate, endDate, True)
    if options.verbose:
        riskModel.debuggingReporting = True
    for d in dateList:
        try:
            processDay(riskModel, options, d, modelDB, marketDB)
        except Exception:
            modelDB.revertChanges()
            logging.error('Exception during processing', exc_info=True)
            modelDB.finalize()
            return 1
    
    modelDB.finalize()
    return 0

if __name__ == '__main__':
    sys.exit(main())


import logging
import optparse
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def processDay(riskModel, options, date, modelDB, marketDB):
    logging.info('Processing %s', date)
    rmi = modelDB.getRiskModelInstance(riskModel.rms_id, date)
    if not rmi:
        rmi = modelDB.createRiskModelInstance(riskModel.rms_id, date)#    riskModel.setRiskModelGroupsForDate(d)
    if options.runReturns:
        riskModel.transferReturns(date, rmi, options, modelDB, marketDB)
    if options.runRisks:
        riskModel.setFactorsForDate(date, modelDB)

        data=riskModel.computeCov(date, modelDB, marketDB)
        if not options.dontWrite:
            modelDB.deleteRMIFactorSpecificRisk(rmi)
            riskModel.insertFactorCovariances(rmi, data.factorCov, data.subFactors, modelDB)
            rmi.setHasRisks(True,modelDB)
    rmi.setIsFinal(True, modelDB)
    if options.testOnly:
        logging.info('Reverting changes')
        #modelDB.revertChanges()
    else:
        modelDB.commitChanges()
    logging.info('Finished processing %s %s' % (riskModel.name, date))

def main():
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--run-returns", action="store_true",
                         default=False, dest="runReturns",
                         help="Don't process the returns")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--risks", action="store_true",
                         default=False, dest="runRisks",
                         help="Generate factor covariances")
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
    riskModel.rmg = modelDB.getAllRiskModelGroups()
#    riskModel.setRiskModelGroupsForDate(startDate)
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])
    
    dateList = modelDB.getDateRange(riskModel.rmg, startDate, endDate, True)
    if len(dateList) == 0:
        logging.info('%s is not a trading day' % str(endDate))
    if options.verbose:
        riskModel.debuggingReporting = True
    for date in dateList:
        try:
            processDay(riskModel, options, date, modelDB, marketDB)
        except Exception:
            modelDB.revertChanges()
            logging.error('Exception during processing', exc_info=True)
            modelDB.finalize()
            return 1
    
    if options.testOnly:
        logging.info('Reverting changes')
        modelDB.revertChanges()
    else:
        modelDB.commitChanges()
    modelDB.finalize()
    return 0

if __name__ == '__main__':
    sys.exit(main())

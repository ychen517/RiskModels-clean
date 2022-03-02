
import logging
import numpy.ma as ma
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def writeHeader(outFile, date, stats, options):
    """Write the appropriate header line.
    """
    if options.summaryStats:
        outFile.write('date, adjusted R-squared\n')
    elif options.factorTValues or options.factorTProbs or options.factorReturns:
        outFile.write('date')
        for f in stats[1]:
            outFile.write(', %s' % f.name)
        outFile.write('\n')

def writeData(outFile, date, stats, options):
    """Write one line of data based on the options.
    """
    if options.summaryStats:
        if stats[2] == None:
            outFile.write('%s, --\n')
        else:
            outFile.write('%s, %f\n' % (str(date), stats[2]))
    elif options.factorTValues:
        outFile.write('%s' % str(date))
        for i in range(stats[0].shape[0]):
            if stats[0][i,1] is ma.masked:
                outFile.write(', --')
            else:
                outFile.write(', %f' % stats[0][i,1])
        outFile.write('\n')
    elif options.factorTProbs:
        outFile.write('%s' % str(date))
        for i in range(stats[0].shape[0]):
            if stats[0][i,2] is ma.masked:
                outFile.write(', --')
            else:
                outFile.write(', %f' % stats[0][i,2])
        outFile.write('\n')
    elif options.factorReturns:
        outFile.write('%s' % str(date))
        for i in range(stats[0].shape[0]):
            if stats[0][i] is ma.masked:
                outFile.write(', --')
            else:
                outFile.write(', %f' % stats[0][i])
        outFile.write('\n')

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD> <outfile>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--summary-statistics", action="store_true",
                             default=False, dest="summaryStats",
                             help="print summary statistics")
    cmdlineParser.add_option("--factor-t-values", action="store_true",
                             default=False, dest="factorTValues",
                             help="print factor t statistics")
    cmdlineParser.add_option("--factor-t-probs", action="store_true",
                             default=False, dest="factorTProbs",
                             help="print factor t probabilities")
    cmdlineParser.add_option("--factor-returns", action="store_true",
                             default=False, dest="factorReturns",
                             help="print factor returns")

    (options, args) = cmdlineParser.parse_args()
    if len(args) != 3:
        cmdlineParser.error("Incorrect number of arguments")
    if options.summaryStats + options.factorTValues + options.factorTProbs\
           + options.factorReturns != 1:
        cmdlineParser.error("Exactly one option must be given")

    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    modelDB.factorReturnCache = None
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    startDate = Utilities.parseISODate(args[0])
    endDate = Utilities.parseISODate(args[1])

    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate)
    outFileName = args[2]
    outFile = open(outFileName, 'w')
    firstLine = True
    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi != None and rmi.has_returns:
            logging.info('Processing %s' % str(d))
            if options.factorReturns:
                stats = riskModel.loadFactorReturns(d, modelDB)
            else:
                stats = riskModel.loadRegressionStatistics(d, modelDB)
            if firstLine:
                firstLine = False
                writeHeader(outFile, d, stats, options)
            writeData(outFile, d, stats, options) 
        else:
            logging.error('No risk model instance on %s' % str(d))

    outFile.close()
    modelDB.finalize()

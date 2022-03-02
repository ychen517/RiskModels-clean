
import logging
import numpy
import os
import numpy.ma as ma
import optparse
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def writeData(outFile, date, stats):
    """Write one line of data based on the options.
    """
    outFile.write(', %f' % stats[0][i])
    outFile.write('\n')

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("-o", "--output-file-name", action="store",
                             default=None, dest="targetFile",
                             help="output file name")
    cmdlineParser.add_option("--use-numeraire", action="store_true",
                             default=False, dest="useNumeraire",
                             help="output returns in base currency")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")

    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    modelDB.totalReturnCache = None
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])

    if options.useNumeraire:
        baseCurrencyID = riskModel.numeraire.currency_id
        modelDB.createCurrencyCache(marketDB)
    else:
        baseCurrencyID = None

    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate)
    dates = [d for d in dates if d.weekday() <= 4]
    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi == None:
            logging.error('No risk model instance on %s' % str(d))
            continue
        if not rmi.has_exposures:
            logging.error('No exposures for risk model instance on %s' % str(d))
            continue
        logging.info('Processing %s' % str(d))
        idList = modelDB.getRiskModelInstanceUniverse(rmi)
        dailyReturns = modelDB.loadTotalReturnsHistory(
                               riskModel.rmg, d, idList, 0, baseCurrencyID)
        validAssetIndices = numpy.flatnonzero(ma.getmaskarray(dailyReturns.data[:,0])==0)
        if options.targetFile:
            outFileName=options.targetFile
        else:
            outFileName = '%s/returns-%d%02d%02d.csv' % (options.targetDir, d.year, d.month, d.day)
        dirName=os.path.dirname(outFileName)
        if not os.path.exists(dirName):
            try:
                os.makedirs(dirName)
            except OSError:
                excstr=str(sys.exc_info()[1])
                if excstr.find('File exists') >= 0 and excstr.find(dirName) >= 0:
                    logging.info('Error can be ignored - %s' % excstr)
                else:
                    raise

        outFile = open(outFileName, 'w')
        for i in validAssetIndices:
            axiomaID = idList[i].getModelID()
            outFile.write('%s, %e\n' % (axiomaID.getPublicID(),
                                        dailyReturns.data[i]))
        outFile.close()

    modelDB.finalize()

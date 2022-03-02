
import logging
import numpy
import numpy.ma as ma
import optparse
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
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")

    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    modelDB.totalReturnCache = None
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    startDate = Utilities.parseISODate(args[0])
    endDate = Utilities.parseISODate(args[1])

    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate)
    for d in dates:
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi == None:
            logging.error('No risk model instance on %s' % str(d))
            continue
        if not rmi.has_exposures:
            logging.error('No exposures for risk model instance on %s' % str(d))
            continue
        logging.info('Processing %s' % str(d))
        idList = modelDB.getRiskModelInstanceUniverse(rmi)
        mcapDates = modelDB.getDates(riskModel.rmg, d, 4)
        marketCaps = modelDB.getAverageMarketCaps(mcapDates, idList, None, marketDB)
        validAssetIndices = numpy.flatnonzero(ma.getmaskarray(marketCaps) == 0)
        outFileName = 'mktcaps-%d%02d%02d.csv' % (d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        for i in validAssetIndices:
            axiomaID = idList[i].getModelID()
            outFile.write('%s, %e\n' % (axiomaID.getPublicID(),
                                        marketCaps[i]))
        outFile.close()

    modelDB.finalize()

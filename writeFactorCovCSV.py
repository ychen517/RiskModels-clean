
import logging
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def writeFactorCovCSV(date, covMatrix, factors, outFile):
    """Write a CSV factor covariance file to outFile.
    """
    outFile.write("%4d%02d%02d Annualized Variance/Covariance in units of percent-squared.  Numeraire: USA\n" % (date.year, date.month, date.day))
    outFile.write('"NAME"')
    for factor in factors:
       shortName = factor.name
       outFile.write(',"%s"' % shortName)
    outFile.write("\n")

    for i in range(len(factors)):
       shortName = factors[i].name
       outFile.write('"%s"' % shortName)
       for j in range(len(factors)):
          outFile.write(', %f' % (10000.0*covMatrix[i,j]))
       outFile.write("\n")

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="exposureDir",
                             help="directory for output files")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")

    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    startDate = Utilities.parseISODate(args[0])
    endDate = Utilities.parseISODate(args[1])

    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate)
    dates.reverse()

    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi != None and rmi.has_risks:
            logging.info('Processing %s' % str(d))
            (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
                rmi, modelDB)
            outFileName = '%s/factorcov-%d%02d%02d.csv' % (options.exposureDir,
                                                           d.year, d.month, d.day)
            outFile = open(outFileName, 'w')
            writeFactorCovCSV(d, factorCov, factors, outFile)
            outFile.close()
        else:
            logging.error('No risk model instance on %s' % str(d))

    modelDB.finalize()

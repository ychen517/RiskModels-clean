
import numpy.ma as ma
import numpy
import logging
import optparse
from marketdb import MarketDB
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels import Utilities

def writeExposureCSV(date, expMatrix, cusipMap, marketCap, estU, svDict,
                     prices, outFile):
    """Write a CSV exposure file to outFile.
    """
    estUDict = dict(zip(estU, list(range(len(estU)))))
    outFile.write("%4d%02d%02d\n" % (date.year, date.month, date.day))
    outFile.write("""AXIOMAID,CUSIP,PBETALOC,CNTRYEXP,SRISK%,TRISK%"""
                  + """,VALUE,LEVERAGE,GROWTH,SIZE,VOLTILITY,LQUIDITY"""
                  + """,STMOM,MTMOM,INDNAME1,IND1,WGT1%,INDNAME2,IND2,WGT2%"""
                  + """,INDNAME3,IND3,WGT3%,INDNAME4,IND4,WGT4%"""
                  + """,INDNAME5,IND5,WGT5%,LOC_PRICE,LOC_CAPT,ESTU"""
                  + """,CNTRY,CCY,CCYEXP\n""")
    mat = expMatrix.getMatrix()
    industryOffset = len(expMatrix.getFactorNames(ExposureMatrix.StyleFactor))
    assert(9 == industryOffset)
    assert(8 == max(expMatrix.getFactorIndices(ExposureMatrix.StyleFactor)))
    for aIdx in range(len(expMatrix.getAssets())):
        asset = expMatrix.getAssets()[aIdx]
        #if prices.data[aIdx,0] is ma.masked:
        #    continue
        if svDict != None:
            if asset in svDict:
                specificRisk = 100.0*svDict[asset]
            else:
                continue
        else:
            specificRisk = -1.0
        modelID = asset.getModelID()
        outFile.write('%s,%s' % (modelID.getPublicID(),
                                 cusipMap.get(modelID, 'N/A')))
        outFile.write(',%4.3f,%4.3f,%9.8f,,%4.3f,%4.3f,%4.3f,%4.3f,%4.3f,%4.3f,%4.3f,%4.3f' % (
            mat[expMatrix.getFactorIndex('Market Sensitivity'), aIdx],
            mat[expMatrix.getFactorIndex('Market Sensitivity'), aIdx],
            specificRisk,
            mat[expMatrix.getFactorIndex('Value'), aIdx],
            mat[expMatrix.getFactorIndex('Leverage'), aIdx],
            mat[expMatrix.getFactorIndex('Growth'), aIdx],
            mat[expMatrix.getFactorIndex('Size'), aIdx],
            mat[expMatrix.getFactorIndex('Volatility'), aIdx],
            mat[expMatrix.getFactorIndex('Liquidity'), aIdx],
            mat[expMatrix.getFactorIndex('Short-Term Momentum'), aIdx],
            mat[expMatrix.getFactorIndex('Medium-Term Momentum'), aIdx]))
        numInd = 0
        for f in numpy.flatnonzero(mat[:,aIdx]):
            if expMatrix.checkFactorType(expMatrix.getFactorNames()[f],
                                         ExposureMatrix.IndustryFactor):
                numInd += 1
                fName = expMatrix.getFactorNames()[f]
                outFile.write(',%s,%d,%d' % (
                    fName, f - industryOffset, mat[f,aIdx] * 100))
        while numInd < 5:
            outFile.write(',,,')
            numInd += 1
        price = prices.data[aIdx,0]
        if prices.data[aIdx,0] is ma.masked:
            price = -999.0
        outFile.write(',%.4f,%.10e,%d,US,USD,1.00' % (
            price, marketCap[aIdx], asset in estUDict))
        outFile.write('\n')

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", "--no-specific-risk", action="store_false",
                             default=True, dest="specRiskFlag",
                             help="don't use specific risk")
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
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi == None:
            logging.error('No risk model instance on %s' % str(d))
            continue
        if not rmi.has_exposures:
            logging.error('No exposure in risk model instance on %s' % str(d))
            continue
        if options.specRiskFlag and not rmi.has_risks:
            logging.error('No risks in risk model instance on %s' % str(d))
            continue
        logging.info('Processing %s' % str(d))
        estM = riskModel.loadExposureMatrix(rmi, modelDB)
        estM.fill(0.0)
        estU = riskModel.loadEstimationUniverse(rmi, modelDB)
        cusipMap = modelDB.getIssueCUSIPs(
            d, [i.getModelID() for i in estM.getAssets()], marketDB)
        mcapDates = modelDB.getDates(riskModel.rmg, d, 19)
        marketCaps = modelDB.getAverageMarketCaps(
            mcapDates, estM.getAssets(), None, marketDB)
        prices = modelDB.loadUCPHistory([d], estM.getAssets(), None)
        if options.specRiskFlag:
            svDataDict = riskModel.loadSpecificRisks(rmi, modelDB)
        else:
            svDataDict = None
        outFileName = '%s/exposure-%d%02d%02d.csv' % (options.exposureDir,
                                                      d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        writeExposureCSV(d, estM, cusipMap, marketCaps, estU, svDataDict,
                         prices, outFile)
        outFile.close()

    marketDB.finalize()
    modelDB.finalize()

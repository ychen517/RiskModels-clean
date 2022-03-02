
import math
import numpy.linalg as linalg
import numpy.ma as ma
import numpy
import logging
import optparse
from marketdb import MarketDB
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities
from riskmodels import ModelDB

#------------------------------------------------------------------------------
class RiskModelData:
    """Risk model data to use in performing daily checks on risk models.
    """
    def __init__(self, d, factors, factorReturns, factorCov, estU, expMatrix, svDict, issuerMap, returns):
        self.d = d
        self.factors = factors
        self.factorReturns = factorReturns
        self.factorCov = factorCov
        self.estU = estU
        self.expMatrix = expMatrix
        self.svDict = svDict
        self.issuerMap = issuerMap
        self.returns = returns
        
#------------------------------------------------------------------------------
class RiskModelCheckParameters:
    """Risk model data to use in performing daily checks on risk models.
    """
    def __init__(self):
        self.absoluteFactorCorrDiffTolerance = 0.05
        self.relativeFactorRiskDiffTolerance = 0.05
        self.minimumFactorCovEigenValue = 1.0e-5
        self.maxFactorReturn = 0.025
        self.relativeSpecificRiskDiffTolerance = 0.1
        self.absoluteStyleFactorDiffTolerance = 0.5

#------------------------------------------------------------------------------
def checkExposures(checkParam, currRMData, prevRMData):
    """Check day over day exposure differences.
    """

    logging.info("checkExposures: begin")
    soundAlarm = False

    styleFactors = currRMData.expMatrix.getFactorNames(
        ExposureMatrix.StyleFactor)
    currExpMat = currRMData.expMatrix.getMatrix()
    prevExpMat = prevRMData.expMatrix.getMatrix()
    assetMap = dict([(prevRMData.expMatrix.getAssets()[i].getModelID(),i) for i in range(len(prevRMData.expMatrix.getAssets()))])
    for aIdx in range(len(currRMData.expMatrix.getAssets())):
        asset = currRMData.expMatrix.getAssets()[aIdx]
        if currRMData.svDict != None:
            if asset not in currRMData.svDict:
                continue
        
        modelID = asset.getModelID()
        if modelID not in assetMap:
            continue
        
        for f in styleFactors:
            currExp = currExpMat[currRMData.expMatrix.getFactorIndex(f), aIdx]
            prevExp = prevExpMat[prevRMData.expMatrix.getFactorIndex(f), assetMap[modelID]]
            if abs(currExp - prevExp) > checkParam.absoluteStyleFactorDiffTolerance:
                logging.warning("(Asset %s, Factor %s): Previous=%g Current=%g" % (modelID.getIDString(), f, prevExp, currExp))

        numPrevInd = 0
        numCurrInd = 0
        currIndName = ""
        prevIndName = ""
        for f in numpy.nonzero(currExpMat[:,aIdx]):
            factorName = currRMData.expMatrix.getFactorNames()[f]
            if currRMData.expMatrix.checkFactorType(factorName, ExposureMatrix.IndustryFactor):
                currIndName = factorName
                numCurrInd += 1

        for f in numpy.nonzero(prevExpMat[:,assetMap[modelID]]):
            factorName = prevRMData.expMatrix.getFactorNames()[f]
            if prevRMData.expMatrix.checkFactorType(factorName, ExposureMatrix.IndustryFactor):
                prevIndName = factorName
                numPrevInd += 1

        if numPrevInd != numCurrInd:
            logging.warning('%d previous industries and %d current industries for %s'
                         % (numPrevInd, numCurrInd, modelID.getIDString()))
            logging.warning('Previous industry: "%s" Current industry: "%s"'
                         % (prevIndName, currIndName))
        elif prevIndName != currIndName:
            logging.warning('Previous industry: "%s" Current industry: "%s"'
                         % (prevIndName, currIndName))

    logging.info("checkExposures: end")
    return soundAlarm

#------------------------------------------------------------------------------
def checkFactorCovEigenvalues(checkParam, rmData):
    """Check eigenvalues of factor covariance matrix to ensure matrix is
    positive definite.
    """

    logging.info("checkFactorCovEigenvalues: begin")
    soundAlarm = False
    eigenvals = numpy.sort(linalg.eigvals(rmData.factorCov))
    logging.info("Minimum 5 Eigenvalues:" +
          str([eigenvals[i] for i in range(min(5,len(eigenvals)))]))
    logging.info("Maximum 5 Eigenvalues:" +
          str([eigenvals[i] for i in range(len(eigenvals)-1, max(len(eigenvals)-6, -1),-1)]))
    minEigenval = eigenvals[0]
    if (minEigenval < checkParam.minimumFactorCovEigenValue):
        logging.warning('Minimum eigenvalue of factor covariance matrix is: %g  This is less than the minimum allowable value of %g'
                     % (minEigenval, checkParam.minimumFactorCovEigenValue))
    logging.info("checkFactorCovEigenvalues: end")

    return soundAlarm
                              
#------------------------------------------------------------------------------
def checkFactorCovDiff(checkParam, currRMData, prevRMData):
    """Perform day-over-day comparison of current factor covariance
    matrix.
    """

    logging.info("checkFactorCovDiff: begin")
    soundAlarm = False
    
    if len(currRMData.factors) != len(prevRMData.factors):
        logging.warning('RiskModel[%d] contains %d factors and RiskModel[%d] contains %d factors\n' % (currRMData.d, len(currRMData.factors), prevRMData.d, len(prevRMData.factors)))

    factorMap = {}
    index = 0
    for factor in prevRMData.factors:
        factorMap[factor.name] = index
        index += 1

    # Check difference in factor volatilities and create list of factor risks
    currFactorRisk = [0.0 for i in range(len(currRMData.factors))]
    prevFactorRisk = [0.0 for i in range(len(currRMData.factors))]
    for i in range(len(currRMData.factors)):
        iFactor = currRMData.factors[i].name
        if iFactor not in factorMap:
            continue
        iIndex = factorMap[iFactor]
        currRisk = math.sqrt(currRMData.factorCov[i, i])
        prevRisk = math.sqrt(prevRMData.factorCov[iIndex, iIndex])
        currFactorRisk[i] = currRisk
        prevFactorRisk[iIndex] = prevRisk
        riskDiff = (currRisk - prevRisk) / prevRisk
        if abs(riskDiff) > checkParam.relativeFactorRiskDiffTolerance:
            logging.warning("Risk of factor %s has changed significantly on %s:  PrevValue is %f; NewValue is %f" % (iFactor, currRMData.d, prevRisk, currRisk))
        
    # Check difference in correlations of factors common between today
    # and yesterday
    for i in range(len(currRMData.factors)):
        iFactor = currRMData.factors[i].name
        if iFactor not in factorMap:
            continue
        iIndex = factorMap[iFactor]
        for j in range(i, len(currRMData.factors)):
            jFactor = currRMData.factors[j].name
            if jFactor not in factorMap:
                continue
            jIndex = factorMap[jFactor]
            currCorr = currRMData.factorCov[i,j] / (currFactorRisk[i] * currFactorRisk[j])
            prevCorr = prevRMData.factorCov[iIndex, jIndex] / (prevFactorRisk[iIndex] * prevFactorRisk[jIndex])
            corrDiff = currCorr - prevCorr
            if abs(corrDiff) > checkParam.absoluteFactorCorrDiffTolerance:
                logging.warning("Correlation between %s and %s has changed significantly on %s:  PrevValue is %f; NewValue is %f" % (iFactor, jFactor, currRMData.d, prevCorr, currCorr))
    logging.info("checkFactorCovDiff: end")

    return soundAlarm

#------------------------------------------------------------------------------
def checkFactorReturns(checkParam, currRMData):
    logging.info("checkFactorReturns: begin")
    soundAlarm = False
    frData = currRMData.factorReturns
    for i in range(frData[0].shape[0]):
        if abs(frData[0][i]) > checkParam.maxFactorReturn:
            logging.warning("Factor return is abnormally large on %s.\nReturn of factor %s is %f%%"
                         % (currRMData.d, frData[1][i].name, frData[0][i] * 100.0))
    logging.info("checkFactorReturns: end")

    return soundAlarm

#------------------------------------------------------------------------------
def checkSpecificRisks(checkParam, currRMData, prevRMData):
    logging.info("checkSpecificRisks: begin")
    soundAlarm = False
    for asset, currSR in currRMData.svDict.items():
        if currSR < 0.0:
            soundAlarm = True
            logging.error("Specific Risk of %s is %g" % (asset.getModelID().getIDString(), currSR))
        if asset in prevRMData.svDict:
            prevSR = prevRMData.svDict[asset]
            if abs(currSR - prevSR)/prevSR > checkParam.relativeSpecificRiskDiffTolerance:
                logging.warning("Specific Risk of %s(%s) is %g; used to be %g" % (asset.getModelID().getIDString(), currRMData.issuerMap.get(asset.getModelID(), ''), currSR, prevSR))
    logging.info("checkSpecificRisks: end")
    
    return soundAlarm
    
def basicDataChecks(checkParam, currRMData, prevRMData):
    logging.info("basicDataChecks: begin")
    soundAlarm = False

    date = currRMData.d
    currExpMatrix = currRMData.expMatrix
    estU = [currRMData.estU[i].getSubIDString() for i in range(len(currRMData.estU))]

    # Check for missing industries
    indExp = ma.take(currExpMatrix.getMatrix(), currExpMatrix.getFactorIndices(ExposureMatrix.IndustryFactor), axis=0)
    missing = numpy.nonzero(numpy.sum(indExp, axis=0).filled(0.0) < 1.0)

    if len(missing) > 0:
        soundAlarm = True
        for i in missing:
            asset = currExpMatrix.getAssets()[i]
            logging.warning('|%s|DODGY missing industry|%s|'
                    % (str(date), asset.getModelID().getIDString()))

    # Check on market caps
    universe = currExpMatrix.getAssets()
    mcapDates = modelDB.getDates(riskModel.rmg, date, 1)
    MC2Day = ma.filled(modelDB.loadMarketCapsHistory(
            mcapDates, universe, None), 0.0)
    issueMapPairs = modelDB.getIssueMapPairs(date)
    issueMap = dict(issueMapPairs)
    marketIssues = [issueMap[i.getModelID()] for i in universe]
    tso = marketDB.getSharesOutstanding([date], marketIssues)
    ucp = marketDB.getPrices([date], marketIssues, riskModel.rmg.currency_code)
    tso = tso.data[:,0].filled(0.0)
    ucp = ucp.data[:,0].filled(0.0)
    
    marketCapChange = 100 * (MC2Day[:,1] - MC2Day[:,0]) / MC2Day[:,0]
    qa = ma.masked_where(marketCapChange > 75.0, marketCapChange)
    for i in numpy.flatnonzero(ma.getmaskarray(ma.masked_where(qa < -50.0, qa))):
        idX = universe[i].getSubIDString()
        if MC2Day[i,1] == 0.0:
            logging.warning('|%s|DODGY mkt cap missing|%s|estU|%s|UCP|%s|TSO|%s|'
                    % (str(date), idX, idX in estU, ucp[i], tso[i]))
        else:
            if MC2Day[i,0] != 0.0:
                logging.warning('|%s|DODGY mkt cap change|%s|estU|%s|CURR|%s|PREV|%s|DIFF|%s|'
                        % (str(date), idX, idX in estU, MC2Day[i,1], MC2Day[i,0],
                            marketCapChange[i]))

    # Report on suspicious returns
    returns = currRMData.returns.data[:,0]
    qa = ma.masked_where(returns.flat > 2.0, returns.flat)

    for i in numpy.flatnonzero(ma.getmaskarray(ma.masked_where(qa < -0.75, qa))):
        soundAlarm = True
        idX = currRMData.returns.assets[i].getSubIDString()
        logging.warning("|%s|DODGY return|%s|estU|%s|%s|"
            % (date, currRMData.returns.assets[i].getSubIDString(),
                idX in estU, returns.flat[i]))

    # Report on suspicious factor returns
    frData = currRMData.factorReturns
    for i in range(frData[0].shape[0]):
        if abs(frData[0][i]) > checkParam.maxFactorReturn:
            soundAlarm = True
            logging.warning("|%s|DODGY factor return|%s|%f|"
                    % (date, frData[1][i].name, frData[0][i] * 100.0))

    # Report on suspicious specific risks
    for asset, currSR in currRMData.svDict.items():
        if currSR < 0.05 or currSR > 1.5:
            soundAlarm = True
            logging.warning("|%s|DODGY specific risk|%s|%s|"
                    % (date, asset.getModelID().getIDString(), currSR))

    logging.info("basicDataChecks: end")
    return soundAlarm
#------------------------------------------------------------------------------
def checkDay(options, checkParam, currRMData, prevRMData):
    soundAlarm = checkFactorReturns(checkParam, currRMData)
    soundAlarm = checkFactorCovDiff(checkParam, currRMData, prevRMData)
    soundAlarm = checkFactorCovEigenvalues(checkParam, currRMData)
    soundAlarm = checkExposures(checkParam, currRMData, prevRMData)
    soundAlarm = checkSpecificRisks(checkParam, currRMData, prevRMData)
    #soundAlarm = basicDataChecks(checkParam, currRMData, prevRMData)


#------------------------------------------------------------------------------
def buildRiskModelData(modelDB, riskModel, d):
    rmi = riskModel.getRiskModelInstance(d, modelDB)
    if rmi != None and rmi.has_risks:
        logging.info('Processing %s' % str(d))
        (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
            rmi, modelDB)
        expM = riskModel.loadExposureMatrix(rmi, modelDB)
        expM.fill(0.0)
        estU = riskModel.loadEstimationUniverse(rmi, modelDB)
        svDataDict = riskModel.loadSpecificRisks(rmi, modelDB)
        factorReturns = riskModel.loadFactorReturns(d, modelDB)
        assetIDs = []
        for asset in expM.getAssets():
            assetIDs.append(asset.getModelID())
        issuerMap = modelDB.getIssueIssuers(d, assetIDs, marketDB)
        returns = modelDB.loadTotalReturnsHistory( riskModel.rmg,
                d, expM.getAssets(), 0, riskModel.currency)
        rmData = RiskModelData(d, factors, factorReturns, factorCov,
                               estU, expM, svDataDict, issuerMap, returns)
    else:
        logging.error('No risk model instance on %s' % str(d))
        
    return rmData


#------------------------------------------------------------------------------
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
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    startDate = Utilities.parseISODate(args[0])
    endDate = Utilities.parseISODate(args[1])
    
    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate)
    checkParam = RiskModelCheckParameters()
    
    firstDate = True
    for d in dates:
        currRMData = buildRiskModelData(modelDB, riskModel, d)
        
        if firstDate:
            firstDate = False
            prevDay = modelDB.getPreviousTradingDay(riskModel.rmg, d)
            prevRMData = buildRiskModelData(modelDB, riskModel, prevDay)
        else:
            prevRMData = currRMData

        checkDay(options, checkParam, currRMData, prevRMData)
    
    modelDB.finalize()
    marketDB.finalize()

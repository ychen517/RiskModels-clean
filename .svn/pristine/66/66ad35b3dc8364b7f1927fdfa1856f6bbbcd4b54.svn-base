import datetime
import logging
import numpy.ma as ma
import numpy
import optparse
import sys
import os
from marketdb import MarketDB
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import MFM
from riskmodels import EquityModel

def generateEstimationUniverse(date, riskModel, modelDB, marketDB, options):
    logging.info('Processing model and estimation universes for %s', date)

    # Nuke the existing model data for given date
    if not options.dontWrite:
        riskModel.deleteInstance(date, modelDB)
        rmi = riskModel.createInstance(date, modelDB)
        rmi.setIsFinal(not options.preliminary, modelDB)

    # Build the estimation universe
    data = riskModel.generate_model_universe(date, modelDB, marketDB)

    # If not writing anything to the DB, finish here
    if options.dontWrite:
        return True

    # Update the estimation universe(s)
    if hasattr(riskModel, 'estuMap') and riskModel.estuMap is not None:
        riskModel.insertEstimationUniverse(rmi, modelDB)
    else:
        riskModel.insertEstimationUniverse(rmi, data.universe,
                data.estimationUniverseIdx, modelDB,
                getattr(data, 'ESTUQualify', None))
    
    if hasattr(riskModel, 'insertLegacyEstimationUniverse'):
        riskModel.insertLegacyEstimationUniverse(rmi, data.universe,
                data.estimationUniverseIdx, modelDB,
                getattr(data, 'ESTUQualify', None))

    # Update model universe
    if hasattr(data, 'nurseryUniverse'):
        modelDB.insertExposureUniverse(rmi, data.universe, excludeList=data.nurseryUniverse)
    else:
        modelDB.insertExposureUniverse(rmi, data.universe)
    return True

def generateEstimationUniverseOnly(rmi, date, riskModel, modelDB, marketDB, options):
    logging.info('Processing estimation universes for %s', date)

    # Build the estimation universe
    data = riskModel.generate_model_universe(date, modelDB, marketDB)

    # If not writing anything to the DB, finish here
    if options.dontWrite:
        return True

    # Update the estimation universe(s)
    modelDB.deleteEstimationUniverse(rmi)

    if hasattr(riskModel, 'estuMap') and riskModel.estuMap is not None:
        riskModel.insertEstimationUniverse(rmi, modelDB)
    else:
        riskModel.insertEstimationUniverse(rmi, data.universe,
                data.estimationUniverseIdx, modelDB,
                getattr(data, 'ESTUQualify', None))

    if hasattr(riskModel, 'insertLegacyEstimationUniverse'):
        riskModel.insertLegacyEstimationUniverse(rmi, data.universe,
                data.estimationUniverseIdx, modelDB,
                getattr(data, 'ESTUQualify', None))
    return True

def generateExposures(rmi, date, riskModel, modelDB, marketDB, macDB, options):
    logging.info('Processing exposures for %s', date)

    # Build exposure matrix
    if riskModel.isProjectionModel():
        data = riskModel.generateExposureMatrix(date, modelDB, marketDB, macDB=macDB)
    else:
        data = riskModel.generateExposureMatrix(date, modelDB, marketDB)
    descriptorData = None
    if type(data) is list:
        descriptorData = data[1]
        data = data[0]

    if options.dontWrite:
        return True

    riskModel.deleteExposures(rmi, modelDB)
    riskModel.insertExposures(rmi, data, modelDB, marketDB, descriptorData=descriptorData)
    rmi.setHasExposures(True, modelDB)
    return True

def runModelTests(date, riskModel, modelDB, marketDB, options):
    logging.info('Running model tests for %s', date)
    riskModel.runModelTests(date, modelDB, marketDB)

def generateFactorAndSpecificReturns(rmi, date, riskModel, modelDB, marketDB, macDB, options,
        externalRun=True, weeklyRun=False, cointegrationTest=False):
    logging.info('Processing factor/specific return for %s', date)

    if not rmi.has_exposures and not riskModel.isProjectionModel():
        if riskModel.forceRun:
            logging.warning('No exposures in risk model instance for %s, skipping', date)
        else:
            logging.error('No exposures in risk model instance for %s, stopping', date)
        return False

    # Run the model regression
    if riskModel.isProjectionModel():
        if externalRun:
            result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date, macDB=macDB)
        else:
            result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date, macDB=macDB, internalRun=True)
    else:
        if cointegrationTest:
            result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date, cointTest=True)
            return True
        elif externalRun:
            result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date)
        elif weeklyRun:
            result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date, weeklyRun=True)
        else:
            result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date, internalRun=True)
    if options.dontWrite:
        return True

    if externalRun:
        # Update external factor return tables
        modelDB.deleteFactorReturns(rmi)
        modelDB.deleteRMSStatistics(rmi)
        modelDB.deleteRMSFactorStatistics(rmi)

        # Insert regression results
        riskModel.insertFactorReturns(
                date, result.factorReturns, modelDB, extraFactors=riskModel.nurseryCountries)
        if hasattr(result, 'pcttrade'):
            riskModel.insertRegressionStatistics(date,
                    result.regressionStatistics,
                    None,
                    result.adjRsquared, result.pcttrade, modelDB,
                    extraFactors=riskModel.nurseryCountries)
        else:
            riskModel.insertRegressionStatistics(date,
                    result.regressionStatistics,
                    None,
                    result.adjRsquared, None, modelDB,
                    extraFactors=riskModel.nurseryCountries)

        if riskModel.twoRegressionStructure:
            if hasattr(riskModel, 'hasInternalSpecRets') and riskModel.hasInternalSpecRets:
                # Model has dual regression structure and internal specific returns table,
                # so don't update the hasReturns flag
                updateSpecRets = 'external'
                updateReturnsFlag = False
            else:
                # Model has dual regression structure but no internal specific returns table
                # so the external table is updated by the internal returns step
                # In time we want to remove this possibility
                updateSpecRets = None
                updateReturnsFlag = False
        else:
            # Model has one regression, so updates external specific returns table only
            updateSpecRets = 'external'
            updateReturnsFlag = True

        if hasattr(result, 'robustWeightMap'):
            modelDB.deleteRobustWeights(rmi)
            modelDB.insertRobustWeights(rmi.rms_id, date, result.robustWeightMap)
    elif weeklyRun:
        # Update weekly factor return tables
        modelDB.deleteFactorReturns(rmi, flag='weekly')
        modelDB.deleteRMSStatistics(rmi, flag='weekly')
        modelDB.deleteRMSFactorStatistics(rmi, flag='weekly')

        # Insert internal regression results
        riskModel.insertFactorReturns(
                date, result.factorReturns, modelDB,
                extraFactors=riskModel.nurseryCountries, flag='weekly')
        riskModel.insertRegressionStatistics(date,
                result.regressionStatistics,
                None,
                result.adjRsquared, result.pcttrade, modelDB,
                extraFactors=riskModel.nurseryCountries, flag='weekly')
        modelDB.deleteRobustWeights(rmi, flag='weekly')
        modelDB.insertRobustWeights(rmi.rms_id, date, result.robustWeightMap, flag='weekly')
    else:
        # Update internal factor return tables
        modelDB.deleteFactorReturns(rmi, flag='internal')
        modelDB.deleteRMSStatistics(rmi, flag='internal')
        modelDB.deleteRMSFactorStatistics(rmi, flag='internal')

        # Insert internal regression results
        riskModel.insertFactorReturns(
                date, result.factorReturns, modelDB,
                extraFactors=riskModel.nurseryCountries, flag='internal')
        riskModel.insertRegressionStatistics(date,
                result.regressionStatistics,
                None,
                result.adjRsquared, result.pcttrade, modelDB,
                extraFactors=riskModel.nurseryCountries, flag='internal')

        # All models running this step will have dual regression structure,
        # so no need to check for that
        if hasattr(riskModel, 'hasInternalSpecRets') and riskModel.hasInternalSpecRets:
            # Has internal table defined, so update that
            updateSpecRets = 'internal'
            updateReturnsFlag = True
        else:
            # No internal table, so write returns to the external (original) table
            updateSpecRets = 'both'
            updateReturnsFlag = True

        modelDB.deleteRobustWeights(rmi, flag='internal')
        modelDB.insertRobustWeights(rmi.rms_id, date, result.robustWeightMap, flag='internal')

    if result.exposureMatrix is None:
        insertUniv = result.universe
    else:
        insertUniv = result.exposureMatrix.getAssets()

    if updateSpecRets == 'external':
        riskModel.insertEstimationUniverseWeights(rmi, result.regression_ESTU, modelDB)
        # Insert specific returns
        modelDB.deleteSpecificReturns(rmi)
        riskModel.insertSpecificReturns(date, result.specificReturns, insertUniv, modelDB)
        if updateReturnsFlag:
            logging.info('Updating hasReturns Flag')
            rmi.setHasReturns(True, modelDB)
    elif updateSpecRets == 'internal':
        # Insert specific returns
        modelDB.deleteSpecificReturns(rmi, internal=True)
        riskModel.insertSpecificReturns(
                    date, result.specificReturns, insertUniv, modelDB, internal=True)
        rmi.setHasReturns(True, modelDB)
        logging.info('Updating hasReturns Flag for Internal Returns step')
    elif updateSpecRets == 'both':
        # Insert internal returns into both tables
        riskModel.insertEstimationUniverseWeights(rmi, result.regression_ESTU, modelDB)
        # Insert specific returns to external table
        modelDB.deleteSpecificReturns(rmi)
        riskModel.insertSpecificReturns(date, result.specificReturns, insertUniv, modelDB)
        # Insert specific returns to internal table
        modelDB.deleteSpecificReturns(rmi, internal=True)
        riskModel.insertSpecificReturns(date, result.specificReturns, insertUniv, modelDB, internal=True)
        rmi.setHasReturns(True, modelDB)
        logging.info('Updating hasReturns Flag for Internal/External Returns step')
    else:
        logging.warning('Not updating specific returns for this step')

    return True

def computeCurrencyReturns(date, riskModel, modelDB, marketDB, options):
    """Computes currency returns.
    """
    logging.info('Processing factor/specific return for %s', date)
    returns = riskModel.computeCurrencyReturns(date, modelDB, marketDB)
    if not options.dontWrite:
        riskModel.deleteInstance(date, modelDB)
        rmi = riskModel.createInstance(date, modelDB)
        riskModel.insertCurrencyReturns(date, returns, modelDB)
        rmi.setHasReturns(True, modelDB)
        rmi.setIsFinal(True, modelDB) # currency models are always final
    return True

def computeCurrencyStatRisk(rmi, date, riskModel, modelDB, marketDB, options):
    """Computes currency risks for a statistical currency risk model.
    """
    logging.info('Processing factor/specific covariance for %s', date)
    results = riskModel.generateStatisticalModel(date, modelDB, marketDB)
    if not options.dontWrite:

        # Purge existing model
        riskModel.deleteStatisticalModel(rmi, modelDB)

        # Exposures
        modelDB.insertExposureUniverse(rmi, results.universe)
        riskModel.insertExposures(rmi, results, modelDB, marketDB)
        rmi.setHasExposures(True, modelDB)

        # Update the estimation universe
        if hasattr(riskModel, 'estuMap') and riskModel.estuMap is not None:
            riskModel.insertEstimationUniverse(rmi, modelDB)
        else:
            riskModel.insertEstimationUniverse(rmi, results.universe,
                    results.estimationUniverseIdx, modelDB)

        # Factor & specific returns
        subIssues = results.srMatrix.assets
        riskModel.insertSpecificReturns(date, results.srMatrix.data[:,0], results.srMatrix.assets, modelDB)
        rmi.setHasReturns(True, modelDB)

        # Factor covariances & specific risks
        riskModel.insertFactorCovariances(rmi, results.factorCov, modelDB)
        riskModel.insertSpecificRisks(rmi, results.specificVars, subIssues, modelDB)
        rmi.setHasRisks(True, modelDB)

    return True

def buildFMPs(rmi, date, riskModel, modelDB, marketDB, options):
    logging.info('Building FMPs for %s', date)
    if not rmi.has_exposures:
        logging.warning('No exposures in risk model instance for %s, skipping', date)
        return

    if riskModel.mnemonic[-3:] not in ('-SH' , '-MH'): 
        logging.warning('Only V3 MH and SH model can have FMPs')
        return

    # Run the model regression
    result = riskModel.generateFactorSpecificReturns(modelDB, marketDB, date, buildFMPs=True)
    if options.dontWrite or (result is None):
        return

    if hasattr(result, 'fmpMap'):
        retFlag = modelDB.deleteFMPs(rmi)
        if retFlag:
            if hasattr(riskModel,'currencies'):
                validFactors = [f for f in riskModel.factors if f not in riskModel.currencies]
            else:
                validFactors = [f for f in riskModel.factors] 
            validSFIds = [sf.subFactorID for sf in modelDB.getSubFactorsForDate(date, validFactors)]
            modelDB.insertFMPs(rmi.rms_id, date, result.fmpMap, saveIDs=validSFIds)

def generateStatisticalModel(rmi, date, riskModel, modelDB, marketDB, options):
    logging.info('Processing model for %s', date)

    # Build the stat model factors and covariances
    results = riskModel.generateStatisticalModel(date, modelDB, marketDB)
    if options.dontWrite:
        return

    # Update exposures and specific returns
    subFactors = results.frMatrix.assets
    universe = results.srMatrix.assets
    riskModel.deleteExposures(rmi, modelDB)
    riskModel.insertExposures(rmi, results, modelDB, marketDB)
    modelDB.deleteSpecificReturns(rmi)
    riskModel.insertSpecificReturns(date, results.srMatrix.data[:,0], universe, modelDB)
    rmi.setHasExposures(True, modelDB)

    # Clear out existing records
    modelDB.deleteFactorReturns(rmi)
    hasStatFactorTable = modelDB.deleteStatFactorReturns(rmi)
    if hasattr(riskModel, 'regionalDescriptorStructure'):
        assert hasStatFactorTable
    modelDB.deleteRMSStatistics(rmi)
    modelDB.deleteRMSFactorStatistics(rmi)
    modelDB.deleteRMIFactorSpecificRisk(rmi)

    # Update factor returns
    riskModel.insertFactorReturns(date, results.frMatrix.data[:,0], modelDB)
    # Update stat factor returns history
    if hasStatFactorTable:
        logging.info('Inserting %d by %d statistical factor returns',
                len(results.frMatrix.dates[:250]), results.frMatrix.data.shape[0])
        for (dtIdx, dt) in enumerate(results.frMatrix.dates[:250]):
            riskModel.insertStatFactorReturns(dt, date, results.frMatrix.data[:, dtIdx], modelDB)

    # Update regression data
    if hasattr(results, 'pctgVar'):
        riskModel.insertRegressionStatistics(
                date, results.regressionStatistics, None,
                results.adjRsquared, None, modelDB, pctgVar=results.pctgVar)
    else:
        riskModel.insertRegressionStatistics(
                date, results.regressionStatistics, None,
                results.adjRsquared, None, modelDB)
    riskModel.insertEstimationUniverseWeights(rmi, results.regression_ESTU, modelDB)
    rmi.setHasReturns(True, modelDB)

    # Update factor and specific covariances
    riskModel.insertFactorCovariances(rmi, results.factorCov, subFactors, modelDB)
    riskModel.insertSpecificRisks(rmi, results.specificVars, universe, results.specificCov, modelDB)
    rmi.setHasRisks(True, modelDB)

def generateCumulativeFactorReturns(rmi, currDate, riskModel, modelDB, startCumReturns, options):
    logging.info('Processing cumulative factor returns for %s', currDate)
    # Some checking on dates
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
        prevRMIs = sorted([(prmi.date, prmi) for prmi in prevRMIs])
        prevRMI = prevRMIs[-1][1]
        prevDate = prevRMI.date

    # Get the current factor structure and pull up the returns
    currRMI = modelDB.getRiskModelInstance(riskModel.rms_id, currDate)
    if currRMI is None or not currRMI.has_returns:
        logging.warning('Skipping %s because risk model is missing', currDate)
        return
    currSubFactors = modelDB.getSubFactorsForDate(
        currDate, riskModel.factors)
    currFactorReturns = modelDB.loadFactorReturnsHistory(
        riskModel.rms_id, currSubFactors, [currDate])

    if not startCumReturns:
        # Load in the previous cumulative factor returns
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
        # If first date, set cumulative returns to one
        newCumFactorReturns = numpy.ones((len(currSubFactors), 1), float)
    
    if not options.dontWrite:
        modelDB.updateCumulativeFactorReturns(
            riskModel.rms_id, currDate, currFactorReturns.assets,
            newCumFactorReturns[:,0])

def computeFactorSpecificRisk(rmi, date, riskModel, modelDB, marketDB, options, macDB=None):
    logging.info('Processing factor/specific risk for %s', date)
    if not rmi.has_returns:
        if riskModel.forceRun:
            logging.warning('No factor returns in risk model instance for %s, skipping', date)
        else:
            logging.error('No factor returns in risk model instance for %s, stopping', date)
        return False

    # Buld covariance matrices
    if riskModel.isProjectionModel():
        if hasattr(options, 'noDVA') and options.noDVA:
            data = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, macDB=macDB, dvaToggle=True)
        else:
            data = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, macDB=macDB)
    else:
        if hasattr(options, 'noDVA') and options.noDVA:
            data = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, dvaToggle=True)
        else:
            data = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB)

    # Run the risk code again - first with DVA turned off, then Newey West
    dvaScale = None
    if hasattr(data, 'estuRisk'):

        # Turn off DVA
        saveFlag = riskModel.runFCovOnly
        saveFlag2 = riskModel.debuggingReporting
        riskModel.runFCovOnly = True
        riskModel.debuggingReporting = False
        if riskModel.isProjectionModel():
            noDVAData = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, macDB=macDB, dvaToggle=True)
        else:
            noDVAData = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, dvaToggle=True)
        # Report on DVA scale factor
        dvaScale = data.estuRisk / noDVAData.estuRisk
        logging.info('***************************************************** DVA scale factor: %.4f', dvaScale)

        # Turn off Newey West
        if riskModel.isProjectionModel():
            noNWData = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, macDB=macDB, nwToggle=True)
        else:
            noNWData = riskModel.generateFactorSpecificRisk(date, modelDB, marketDB, nwToggle=True)
        # Report on NW scale factor
        nwScale = data.estuRisk / noNWData.estuRisk
        logging.info('***************************************************** NW scale factor: %.4f', nwScale)
        riskModel.runFCovOnly = saveFlag
        riskModel.debuggingReporting = saveFlag2

    if options.dontWrite:
        return True

    # Write data to DB
    if riskModel.runFCovOnly:
        modelDB.deleteRMIFactorCovMatrix(rmi)
        modelDB.deleteDVAStatistics(rmi)
    else:
        modelDB.deleteRMIFactorSpecificRisk(rmi)
        modelDB.deleteDVAStatistics(rmi)
        riskModel.insertSpecificRisks(rmi, data.specificVars, data.subIssues, data.specificCov, modelDB)

    riskModel.insertFactorCovariances(rmi, data.factorCov, data.subFactors, modelDB)
    if dvaScale is not None:
        modelDB.insertDVAStatistics(rmi, dvaScale, nwScale)

    rmi.setHasRisks(True, modelDB)
    return True

def computeTotalRisksAndBetas(rmi, date, riskModel, modelDB, marketDB, options):
    logging.info('Processing total risks and betas for %s', date)
    if not rmi.has_risks:
        logging.warning('Incomplete risk model instance for %s, skipping', date)
        return False

    # Load exposures, common factor and specific risks
    modelData = Utilities.Struct()
    modelData.exposureMatrix = riskModel.loadExposureMatrix(rmi, modelDB)
    modelData.exposureMatrix.fill(0.0)
    (modelData.specificRisk, modelData.specificCovariance) = riskModel.loadSpecificRisks(rmi, modelDB)
    (modelData.factorCovariance, modelData.factors) = riskModel.loadFactorCovarianceMatrix(rmi, modelDB)

    # Compute asset total risks and add to DB
    totalRisk = riskModel.computeTotalRisk(modelData, modelDB)
    if not options.dontWrite and not options.dontWriteOldBeta:
        modelDB.deleteRMITotalRisk(rmi)
        modelDB.insertRMITotalRisk(rmi, totalRisk)

    # Legacy asset beta
    if (not isinstance(riskModel, EquityModel.FundamentalModel)) and \
       (not isinstance(riskModel, EquityModel.StatisticalModel)) and \
       (not isinstance(riskModel, EquityModel.LinkedModel)) and \
       (not isinstance(riskModel, EquityModel.ProjectionModel)) :

        # Compute legacy betas
        predictedBeta = riskModel.computePredictedBeta(date, modelData, modelDB, marketDB)
        if not options.dontWrite and not options.dontWriteOldBeta:
            modelDB.deleteRMIPredictedBeta(rmi)
            modelDB.insertRMIPredictedBeta(rmi, predictedBeta)

        if options.v3:
            # Compute V3 betas
            predictedBeta = riskModel.computePredictedBetaV3(date, modelData, modelDB, marketDB)
            if not options.dontWrite:
                modelDB.deleteRMIPredictedBeta(rmi, v3=True)
                modelDB.insertRMIPredictedBetaV3(rmi, predictedBeta)
    else:
        # For newer models, we have improved beta code
        predictedBeta = riskModel.computePredictedBeta(date, modelData, modelDB, marketDB)
        if not options.dontWrite:
            legacyBetaTuple = [(sid, legacy_beta) for (sid, dum1, legacy_beta, dum2) in predictedBeta]
            modelDB.deleteRMIPredictedBeta(rmi)
            modelDB.insertRMIPredictedBeta(rmi, legacyBetaTuple)
            modelDB.deleteRMIPredictedBeta(rmi, v3=True)
            modelDB.insertRMIPredictedBetaV3(rmi, predictedBeta)

    return True

def extractFlatFiles(rmi, date, riskModel, modelDB, marketDB, options):
    dirName = options.extractFlatFiles
    logging.info('Extracting flatfiles to %s for %s', dirName, date)

    # Set up directories and command line args
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        cmd = 'chmod a+rxw %s' % dirName
        os.system(cmd)

    # Run the flatfile step
    args = "--ignore-missing --no-hist --new-rsk-fields --target-sub-dirs --histbeta-new \
            --file-format-version 4.0 --warn-not-crash -f"
    dbOpt = "--marketdb-user=%s --marketdb-passwd=%s --marketdb-sid=%s \
             --modeldb-user=%s --modeldb-passwd=%s --modeldb-sid=%s" % \
             (options.marketDBUser, options.marketDBPasswd, options.marketDBSID, \
             options.modelDBUser, options.modelDBPasswd, options.modelDBSID)
    cmdString = "python3 writeFlatFiles.py -l log.config -m%s %s %s -d %s %s" % (options.modelName, dbOpt, args, dirName, date)
    os.system(cmdString)

    # Zip files
    fileLoc = "%s/%d/%02d" % (dirName, date.year, date.month)
    cmdString = "find %s -name '%s*.%s.*.gz' -exec rm -f {} \;" % (fileLoc, riskModel.mnemonic, str(date).replace('-',''))
    os.system(cmdString)
    cmdString = "find %s -name 'Currencies.%s.*.gz' -exec rm -f {} \;" % (fileLoc, str(date).replace('-',''))
    os.system(cmdString)
    cmdString = "find %s -name '%s*.%s' -exec rm -f {} \;" % (fileLoc, riskModel.mnemonic, str(date).replace('-',''))
    os.system(cmdString)
    cmdString = "find %s -name '%s*.%s.*' -exec gzip -fq {} \;" % (fileLoc, riskModel.mnemonic, str(date).replace('-',''))
    os.system(cmdString)
    cmdString = "find %s -name 'Currencies.%s.*' -exec gzip -fq {} \;" % (fileLoc, str(date).replace('-',''))
    os.system(cmdString)

    # Set permissions
    os.system('chmod a+rxw %s' % fileLoc)
    cmd = 'chmod a+rxw %s/*.%s.*.gz' % (fileLoc, str(date).replace('-',''))
    os.system(cmd)
    fileLoc = "%s/%d" % (dirName, date.year)
    os.system('chmod a+rxw %s' % fileLoc)

    return True

def extractDerbyFiles(rmi, date, riskModel, modelDB, marketDB, options):
    dirName = options.extractDerbyFiles
    logging.info('Extracting Derby files to %s for %s', dirName, date)

    # Set up directories and command line args
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        cmd = 'chmod a+rxw %s' % dirName
        os.system(cmd)

    # Run the flatfile step
    args = "--no-meta --classifications= --file-format-version=4.0 --histbeta-new \
            --new-rmm-fields --new-beta-fields --derby-creator-path=/home/ops-rm/global/lib --tmp-dir=/tmp \
            --target-sub-dirs --no-subdirs --new-rsk-fields"
    dbOpt = "--marketdb-user=%s --marketdb-passwd=%s --marketdb-sid=%s \
             --modeldb-user=%s --modeldb-passwd=%s --modeldb-sid=%s" % \
             (options.marketDBUser, options.marketDBPasswd, options.marketDBSID, \
             options.modelDBUser, options.modelDBPasswd, options.modelDBSID)
    cmdString = "cd Derby; python3 createRange.py -l log.config --models %s %s %s --destination %s %s" % (options.modelName, dbOpt, args, dirName, date)
    if options.force:
        cmdString += ' -f'
    os.system(cmdString)

    # Set permissions
    fileLoc = "%s" % dirName
    os.system('chmod a+rxw %s' % fileLoc)
    fileLoc = "%s/%s" % (dirName, date.year)
    os.system('chmod a+rxw %s' % fileLoc)
    fileLoc = "%s/%d/%02d" % (dirName, date.year, date.month)
    os.system('chmod a+rxw %s' % fileLoc)
    cmd = 'chmod a+rxw %s/*-%s.jar' % (fileLoc, str(date).replace('-',''))
    os.system(cmd)

    return True

def runLoop(riskModel, dates, modelDB, marketDB, macDB, options):
    status = 0
    steps = ''
    for d in dates:
        if options.trackList is not None:
            trackList = options.trackList.split(',')
            allSubIDs = modelDB.getAllActiveSubIssues(d)
            sidStringMap = dict([(sid.getSubIDString(), sid) for sid in allSubIDs])
            riskModel.trackList = [sidStringMap[ss] for ss in trackList if ss in sidStringMap]
            riskModel.dropList = []
            riskModel.addList = []
        else:
            riskModel.trackList = []
        if len(riskModel.trackList) > 0:
            logging.info('Tracking assets: %s', ','.join([sid.getSubIDString() for sid in riskModel.trackList]))
        riskModel.compareBM = []
        if options.compareBM is not None:
            riskModel.compareBM = options.compareBM.split(',')

        try:
            runError = False
            if options.runESTU:
                riskModel.setFactorsForDate(d, modelDB)
                runFlag = generateEstimationUniverse(d, riskModel, modelDB, marketDB, options)
                assert runFlag
                if d == dates[0] or len(steps) == 0:
                    steps += 'estu,'

            if not options.currencyModel:
                rmi = riskModel.getRiskModelInstance(d, modelDB)
                if rmi is None:
                    logging.warning('No risk model instance for %s, skipping', d)
                    continue

            if options.cointegrationTests:
                riskModel.setFactorsForDate(d, modelDB)
                generateFactorAndSpecificReturns(
                        rmi, d, riskModel, modelDB, marketDB, macDB, options, cointegrationTest=True)
                if d == dates[0] or len(steps) == 0:
                    steps += 'coint-tests,'

            if options.runESTUOnly:
                riskModel.setFactorsForDate(d, modelDB)
                runFlag = generateEstimationUniverseOnly(rmi, d, riskModel, modelDB, marketDB, options)
                assert runFlag
                if d == dates[0] or len(steps) == 0:
                    steps += 'estuOnly,'
            
            if options.statModel:
                riskModel.setFactorsForDate(d, modelDB)
                if options.runExposures or options.runFactors or options.runRisks:
                    generateStatisticalModel(rmi, d, riskModel, modelDB, marketDB, options)
                    if d == dates[0] or len(steps) == 0:
                        steps += 'risks,'
                if options.flipFactors:
                    maxIter = 10
                    iters = 0
                    count = 1
                    while (count > 0) and (iters < maxIter):
                        count = flipStatFactors(rmi, d, riskModel, modelDB, marketDB, options)
                        if d == dates[0] or len(steps) == 0:
                            steps += 'flipping,'
                        iters += 1
            elif options.currencyModel:
                riskModel.setRiskModelGroupsForDate(d)
                if options.runFactors:
                    runFlag = computeCurrencyReturns(d, riskModel, modelDB, marketDB, options)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'returns,'

                if options.runRisks:
                    rmi = riskModel.getRiskModelInstance(d, modelDB)
                    if rmi is None:
                        logging.warning('No risk model instance for %s, skipping', d)
                        continue
                    runFlag = computeCurrencyStatRisk(rmi, d, riskModel, modelDB, marketDB, options)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'cov,'
            elif options.projectionModel: # re-order the sequence for projection model, remove FMP steps
                if options.runFactors:
                    if riskModel.twoRegressionStructure:
                        runFlag = generateFactorAndSpecificReturns(
                                rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=False)
                    else:
                        runFlag = generateFactorAndSpecificReturns(
                                rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=True)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'returns,'
                if options.runExposures:
                    riskModel.setFactorsForDate(d, modelDB)
                    runFlag = generateExposures(rmi, d, riskModel, modelDB, marketDB, macDB, options)
                    assert runFlag
                    if d == dates[0] or len(steps) == 0:
                        steps += 'exposures,'
                if options.runRisks:
                    riskModel.setFactorsForDate(d, modelDB)
                    runFlag = computeFactorSpecificRisk(rmi, d, riskModel, modelDB, marketDB, options, macDB=macDB)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'cov,'
                if options.runExternalFactors:
                    runFlag = generateFactorAndSpecificReturns(rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=True)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'external_returns,'
                if options.runCumFactors:
                    riskModel.setFactorsForDate(d, modelDB)
                    generateCumulativeFactorReturns(rmi, d, riskModel, modelDB,
                            options.startCumulativeReturn and d == dates[0], options)
                    if d == dates[0] or len(steps) == 0:
                        steps += 'cumrets,'
            else:

                if options.runExposures:
                    riskModel.setFactorsForDate(d, modelDB)
                    runFlag = generateExposures(rmi, d, riskModel, modelDB, marketDB, macDB, options)
                    assert runFlag
                    if d == dates[0] or len(steps) == 0:
                        steps += 'exposures,'
            
                if options.runFactors:
                    if riskModel.twoRegressionStructure:
                        runFlag = generateFactorAndSpecificReturns(
                                rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=False)
                    else:
                        runFlag = generateFactorAndSpecificReturns(
                                rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=True)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'returns,'

                if options.runWeeklyFactors:
                    runFlag = generateFactorAndSpecificReturns(
                            rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=False, weeklyRun=True)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'weekly-returns,'

                if options.runRisks:
                    riskModel.setFactorsForDate(d, modelDB)
                    runFlag = computeFactorSpecificRisk(rmi, d, riskModel, modelDB, marketDB, options, macDB=macDB)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'cov,'

                if options.runExternalFactors:
                    runFlag = generateFactorAndSpecificReturns(
                            rmi, d, riskModel, modelDB, marketDB, macDB, options, externalRun=True)
                    if not runFlag:
                        runError = True
                    if d == dates[0] or len(steps) == 0:
                        steps += 'external_returns,'

                if options.buildFMPs:
                    buildFMPs(rmi, d, riskModel, modelDB, marketDB, options)
                    if d == dates[0] or len(steps) == 0:
                        steps += 'fmps,'

                if options.runCumFactors:
                    riskModel.setFactorsForDate(d, modelDB)
                    generateCumulativeFactorReturns(rmi, d, riskModel, modelDB,
                            options.startCumulativeReturn and d == dates[0], options)
                    if d == dates[0] or len(steps) == 0:
                        steps += 'cumrets,'
            
            if options.runTotalRiskBeta:
                riskModel.setFactorsForDate(d, modelDB)
                runFlag = computeTotalRisksAndBetas(rmi, d, riskModel, modelDB, marketDB, options)
                if not runFlag:
                    runError = True
                if d == dates[0] or len(steps) == 0:
                    steps += 'betas,'
            
            if options.extractFlatFiles is not None:
                riskModel.setFactorsForDate(d, modelDB)
                runFlag = extractFlatFiles(rmi, d, riskModel, modelDB, marketDB, options)
                if not runFlag:
                    runError = True
                if d == dates[0] or len(steps) == 0:
                    steps += 'flatfiles,'

            if options.extractDerbyFiles is not None:
                riskModel.setFactorsForDate(d, modelDB)
                runFlag = extractDerbyFiles(rmi, d, riskModel, modelDB, marketDB, options)
                if not runFlag:
                    runError = True
                if d == dates[0] or len(steps) == 0:
                    steps += 'derby,'

            if options.testOnly:
                logging.info('Reverting changes')
                modelDB.revertChanges()
            else:
                modelDB.commitChanges()
            if not runError:
                logging.info('Finished %s [%s] processing for %s', options.modelName, steps, d)
            else:
                logging.warning('%s steps [%s] for %s showed errors - take a look',
                        options.modelName, steps, d) 

        except Exception:
            logging.error('Exception caught during processing', exc_info=True)
            modelDB.revertChanges()
            if not riskModel.forceRun:
                status = 1
                break
    return status

def runmain():
    import os
    print('Number of threads = ', os.getenv('OPENBLAS_NUM_THREADS'))
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    
    # Typical model generation steps
    cmdlineParser.add_option("--estu", action="store_true",
                             default=False, dest="runESTU",
                             help="Generate model and estimation universe")
    cmdlineParser.add_option("--estu-only", action="store_true",
                             default=False, dest="runESTUOnly",
                             help="Generate only new estimation universe structure")
    cmdlineParser.add_option("--exposures", action="store_true",
                             default=False, dest="runExposures",
                             help="Generate factor exposures")
    cmdlineParser.add_option("--factors", action="store_true",
                             default=False, dest="runFactors",
                             help="Generate factor returns")
    cmdlineParser.add_option("--xfactors", action="store_true",
                             default=False, dest="runExternalFactors",
                             help="Generate external factor returns")
    cmdlineParser.add_option("--wfactors", action="store_true",
                             default=False, dest="runWeeklyFactors",
                             help="Generate weekly factor returns")
    cmdlineParser.add_option("--risks", action="store_true",
                             default=False, dest="runRisks",
                             help="Generate factor covariances and specific risk")
    cmdlineParser.add_option("--flatfiles", "--ff", action="store",
                             default=None, dest="extractFlatFiles",
                             help="Extract standard set of flatfiles to location specified")
    cmdlineParser.add_option("--derby", "--df", action="store",
                             default=None, dest="extractDerbyFiles",
                             help="Extract standard set of Derby files to location specified")
    cmdlineParser.add_option("--tests", action="store",
                             default=None, dest="cointegrationTests",
                             help="Run assorted tests on data and model")
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
    cmdlineParser.add_option("--totalrisk-beta", "--trb", action="store_true",
                             default=False, dest="runTotalRiskBeta",
                             help="Generate total risks and predicted betas")
    cmdlineParser.add_option("--cov-only", action="store_true",
                             default=False, dest="runFCovOnly",
                             help="Skip time-consuming specific risk step")
    cmdlineParser.add_option("--no-dva", action="store_true",
                             default=False, dest="noDVA",
                             help="Turn off DVA in the risk step")
    cmdlineParser.add_option("--flip", action="store_true",
                             default=False, dest="flipFactors",
                             help="Flip stat factors")
    cmdlineParser.add_option("--fmp", action="store_true",
                             default=False, dest="buildFMPs",
                             help="Build model FMPs")
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="perform a lot of debugging diagnostics")
    cmdlineParser.add_option("--dw", action="store_true",
                             default=False, dest="dontWrite",
                             help="don't even attempt to write to the database")
    cmdlineParser.add_option("--dw-old-beta", action="store_true",
                             default=False, dest="dontWriteOldBeta",
                             help="don't even attempt to write old betas to the database")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override certain constraints")
    cmdlineParser.add_option("--track", action="store",
                             default=None, dest="trackList",
                             help="One or more assets whose progress we wish to track")
    cmdlineParser.add_option("--bmc", action="store",
                             default=None, dest="compareBM",
                             help="Compare estimation universe coverage of specified benchmark")
    cmdlineParser.add_option("--v3", "--V3", action="store_true",
                             default=False, dest="v3",
                             help="run newer versions of some code")
    cmdlineParser.add_option("--big-log", action="store_true",
                             default=False, dest="bigLog",
                             help="toggle between full logging output and that in config file")
    cmdlineParser.add_option("--preliminary", action="store_true",
                             default=False, dest="preliminary",
                             help="Preliminary run--ignore DR assets")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)
    # added for Projection model, only global macro projection model uses this for now.
    macDB = {'port': options.macdb_port,
             'host': options.macdb_host,
             'database': options.macdb_name,
             'dbUser': options.macdb_user,
             'dbPass': options.macdb_pwd
             }
    riskModel = riskModelClass(modelDB, marketDB)

    if options.startCumulativeReturn:
        options.runCumFactors = True

    options.currencyModel = riskModel.isCurrencyModel()
    options.statModel = riskModel.isStatModel()
    options.projectionModel = riskModel.isProjectionModel() # add special treatment for projection model

    if options.bigLog:
        for h in logging.root.handlers:
            h.setLevel(logging.NOTSET)

    if options.runAll:
        if options.currencyModel:
            options.runFactors = True
            options.runRisks = True
        else:
            options.runESTU = True
            options.runExposures = True
            options.runFactors = True
            options.runRisks = True
            options.runTotalRiskBeta = True
            if not options.statModel:
                options.buildFMPs = True
                options.runCumFactors = True
        if riskModel.twoRegressionStructure:
            options.runExternalFactors = True

    if options.currencyModel:
        options.runESTU = False
        options.runExposures = False
        options.runTotalRiskBeta = False
        options.buildFMPs = False
        options.runCumFactors = False
        options.runExternalFactors = False

    if options.runTotalRiskBeta:
        modelDB.setMarketCapCache(45)
    modelDB.setVolumeCache(502)

    if options.preliminary:
        riskModel.rollOverDescriptors = 1
    if options.cointegrationTests is not None:
        riskModel.cointHistory = options.cointegrationTests
        options.cointegrationTests = True

    if len(args) == 1:
        dateRanges = [i.strip() for i in args[0].split(',')]
        dates = set()
        for dRange in dateRanges:
            if len(dRange) == 4:
                dRange = dRange + '-01-01:' + dRange + '-12-31'
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
        modelDates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, excludeWeekend=True)
        dates = sorted(dates & set(modelDates))
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, excludeWeekend=True)

    if len(dates) < 1:
        logging.info('No valid model dates in range %s to %s', startDate, endDate)

    if options.verbose:
        riskModel.debuggingReporting = True
    riskModel.runFCovOnly = options.runFCovOnly

    if options.force:
        riskModel.forceRun = True
        riskModel.variableStyles = True
    
    status = runLoop(riskModel, dates, modelDB, marketDB, macDB, options)
    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)

if __name__ == '__main__':
    runmain()

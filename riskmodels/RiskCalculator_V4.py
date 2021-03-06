
import datetime
import numpy.ma as ma
import numpy.linalg as linalg
import numpy
import logging
import copy
from collections import defaultdict
from itertools import chain
import scipy.interpolate as spline
import pandas
from riskmodels import Utilities
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Outliers

class CompositeCovarianceMatrix2020:
    """Composite covariance matrix.
    Code for V4 regional models.
    """
    def __init__(self, varParameters, corrParameters):
        self.log = logging.getLogger('RiskCalculator_V4.CompositeCovarianceMatrix2020')
        self.subCovariances = dict()
        self.varParameters = varParameters
        self.corrParameters = corrParameters

    def configureSubCovarianceMatrix(self, position, factorCov=None, rp=None):
        """Fine-tune the covariance matrix for the 'block'
        corresponding to the given position. (starts at zero)
        eg, position=0 refers to the first factor returns series
        provided to computeFactorCovarianceMatrix().
        If factorCov is specified, this pre-computed covariance
        matrix will be used.  If a RiskParameters object rp is provided,
        it will be used to compute this factor covariance matrix.
        """
        self.subCovariances[position] = dict()
        self.subCovariances[position]['fcov'] = factorCov
        return True

    def cov_main(self, frets, covPar, matrixType='cov'):
        """Core routine for computing a factor covariance matrix.
        """
        # Set up parameters
        nwLag = covPar.NWLag
        if covPar.dateOverLap is not None:
            nwLag = 0
            logging.info('Using overlapping %d-day returns for covariance matrix', covPar.dateOverLap)

        # Get correct shape of data
        if frets.values.shape[1] > covPar.maxObs:
            frets = frets.iloc[:, 0:covPar.maxObs]
        T = len(frets.columns)
        nFacs = len(frets.index)
        self.log.info('Using %d factors by %d periods', nFacs, T)

        # Calculate exponential weights
        factorReturnWeights = Utilities.computeExponentialWeights(covPar.halfLife, T, covPar.equalWeightFlag)

        # Apply DVA to factor returns if required
        if covPar.DVAWindow is not None:
            self.log.debug('Applying Dynamic Volatility Adjustment (DVA)')
            if covPar.DVAType == 'spline':
                frData = spline_dva(frets, covPar.DVAWindow,
                        upperBound=covPar.DVAUpperBound, lowerBound=covPar.DVALowerBound,
                        downWeightEnds=covPar.downweightEnds)
            else:
                frData = frets.copy(deep=True)
        else:
            frData = frets.copy(deep=True)

        # Check for factor returns consisting entirely of zeros
        fretSum = frData.mask(abs(frData) < 1.0e-10)
        fretSum = (numpy.isfinite(fretSum).sum(axis=1) / float(T))
        fretSum = fretSum.mask(abs(fretSum)<0.01)
        goodFactorsIdx = list(fretSum[numpy.isfinite(fretSum)].index)

        # Pick out subset of non-zero factor returns
        frData = frData.loc[goodFactorsIdx, :].fillna(0.0)
        if len(goodFactorsIdx) != nFacs:
            self.log.info('%d out of %d factors are effectively zero', nFacs-len(goodFactorsIdx), nFacs)
            self.log.info('Dimension of cov matrix reduced to %d for calculation', len(goodFactorsIdx))

        # Use overlapping returns if required
        if covPar.dateOverLap is not None:
            frTmp = numpy.array(1.0 + frData.values, copy=True)
            for jdx in range(frData.values.shape[1]-1):
                frData.values[:, jdx] = numpy.product(frTmp[:, jdx:jdx+covPar.dateOverLap], axis=1) - 1.0

        # Perform Newey-West using given number of lags
        self.log.info('Number of autocorrelation lags: %d', nwLag)
        factorCov = Utilities.compute_NWAdj_covariance(
                frData.values, nwLag, weights=factorReturnWeights, deMean=covPar.deMeanFlag, axis=1)
        factorCov = (factorCov + numpy.transpose(factorCov)) / 2.0

        # Create full covariance matrix dataframe
        factorCov = pandas.DataFrame(factorCov, index=goodFactorsIdx, columns=goodFactorsIdx)
        factorCov = factorCov.reindex(index=frets.index, columns=frets.index).fillna(0.0)

        # Scale the covariance matrix if using overlapping data
        if covPar.dateOverLap is not None:
            factorCov = factorCov / float(covPar.dateOverLap)

        # Report on condition number of covariance matrix
        (eigval, eigvec) = linalg.eigh(factorCov)
        if min(eigval) <= 0.0:
            logging.info('Zero eigenvalue - infinite condition number')
        else:
            condNum = max(eigval) / min(eigval)
            logging.info('Covariance matrix condition number: %.2f', condNum)

        # Performing shrinkage
        if covPar.shrink:
            # Note: needs overhaul first
            factorCov = Utilities.optimal_shrinkage(factorCov, T)

        # Return the relevant object
        if matrixType=='std' or matrixType=='cor':
            (stdVector, factorCor) = Utilities.cov2corr(factorCov.values)
            if matrixType=='std':
                return pandas.Series(stdVector, index=factorCov.index)
            else:
                return pandas.DataFrame(factorCor, index=factorCov.index, columns=factorCov.index)
        else:
            (stdVector, covMatrix) = Utilities.cov2corr(factorCov.values, returnCov=True)
            return pandas.DataFrame(covMatrix, index=factorCov.index, columns=factorCov.index)

    def computeFactorCovarianceMatrix(self, *frMatrix):
        """Compute a 'composite' factor covariance matrix.  Each
        input is a TimeSeriesMatrix of factor returns belonging to
        the corresponding block of the covariance matrix.  Even if
        the covariances for a particular block have been pre-computed
        (eg. currencies), its factor returns series must still be
        provided.  The first column in each TimeSeriesMatrix of
        observations should be the most recent.
        """
        self.log.debug('computeFactorCovarianceMatrix: begin')

        # Aggregate all factor returns from input blocks
        allFactorReturns = None
        for fr in frMatrix:

            # Returns timing adjustment if required
            if hasattr(fr, 'adjust'):
                self.log.info('Adjusting returns')
                fr.data = fr.data + fr.adjust

            # Long mean adjustment if required
            if hasattr(fr, 'mean'):
                fr.data = fr.data.sub(fr.mean, axis=0)

            if allFactorReturns is None:
                allFactorReturns = fr.data.copy(deep=True)
            else:
                assert(len(fr.data.columns)==len(allFactorReturns.columns))
                allFactorReturns = pandas.concat([allFactorReturns, fr.data], axis=0)

        # Initialise
        k = len(allFactorReturns.index)
        blockMatrices = []
        fullCorrelationMatrix = numpy.zeros((k, k), float)
        fullStdVector = pandas.Series(numpy.zeros((k), float), index=allFactorReturns.index)

        # Compute full-size correlation matrix if there is more than one block
        if len(frMatrix) > 1:
            compositeMatrix = True
            self.log.info('Computing full size %d by %d correlation matrix *********************', k, k)
            fullCorrelationMatrix = self.cov_main(allFactorReturns, self.corrParameters, matrixType='cor')
        else:
            compositeMatrix = False
            fullCorrelationMatrix = pandas.DataFrame(fullCorrelationMatrix,
                    index=allFactorReturns.index, columns=allFactorReturns.index).fillna(0.0)

        # Loop through all the factor sub-blocks
        for (iloc, fr) in enumerate(frMatrix):
            self.log.info('Computing sub covariance matrix for chunk %d *******************', iloc)

            # Check if using a pre-computed block
            sub = self.subCovariances.get(iloc, None)
            if sub is not None:
                subCovBlock = sub.get('fcov', None)
                self.log.info('Loading pre-computed %d by %d block', len(subCovBlock.index), len(subCovBlock.columns))
                (subStdVector, subCorrelBlock) = Utilities.cov2corr(subCovBlock.values)
                subCorrelBlock = pandas.DataFrame(subCorrelBlock, index=fr.data.index, columns=fr.data.index)
                fullStdVector.loc[fr.data.index] = pandas.Series(subStdVector, index=fr.data.index)
            else:
                # Compute correlations
                self.log.info('Computing correlation %d by %d returns', len(fr.data.index), len(fr.data.columns))
                subCorrelBlock = self.cov_main(fr.data, self.corrParameters, matrixType='cor')
                # Compute variances
                self.log.info('Computing variance of %d by %d returns', len(fr.data.index), len(fr.data.columns))
                fullStdVector.loc[fr.data.index] = self.cov_main(fr.data, self.varParameters, matrixType='std').values

            # Put blocks into place
            if compositeMatrix:
                blockMatrices.append(subCorrelBlock.fillna(0.0))
            else:
                fullCorrelationMatrix = subCorrelBlock

        self.log.info('Finished computing sub covariance matrix blocks ************')

        # Perform procrustes rotation if required
        if compositeMatrix:
            fullCorrelationMatrix = Utilities.procrustes_transform(fullCorrelationMatrix.fillna(0.0), blockMatrices)

        # Combine variances with correlations
        fullCorrelationMatrix = (fullCorrelationMatrix + fullCorrelationMatrix.T) / 2.0
        factorCov = ma.transpose(ma.transpose(fullCorrelationMatrix.values*fullStdVector.values)*fullStdVector.values)
        factorCov = pandas.DataFrame(factorCov, index=allFactorReturns.index, columns=allFactorReturns.index)

        # Return matrix
        factorCov = 252.0 * factorCov.fillna(0.0)
        self.log.info('Sum of cov matrix elements: %f', ma.sum(factorCov.values, axis=None))
        self.log.debug('computeFactorCovarianceMatrix: end')
        return factorCov

class CompositeCovarianceMatrix2017(CompositeCovarianceMatrix2020):
    """Composite covariance matrix.
    Code for V3 regional models.
    """
    def __init__(self, varParameters, corrParameters):
        self.log = logging.getLogger('RiskCalculator_V4.CompositeCovarianceMatrix')
        self.subCovariances = dict()
        self.varParameters = varParameters
        self.corrParameters = corrParameters

class ComputeSpecificRisk2020:
    """Code for V4 regional models
    Computes specific risk from time-series of specific returns.
    Exponential weighting, Newey West and DVA can be applied.
    Assets with short histories are shrunk towards a cross-sectional average.
    Linked assets have specific risk and correlation computed using either
    cointegration or the naive approach, depending on their level of cointegration
    """
    def __init__(self, srParameters):
        self.log = logging.getLogger('RiskCalculator_V4.ComputeSpecificRisk2020')
        self.srPars = srParameters
        self.minSpecificVar = self.srPars.minVar / 252.0
        self.maxSpecificVar = self.srPars.maxVar / 252.0
        self.specVarFloor = 0.000025 / 252.0
        self.minCorrel = 0.25
        self.maxCorrel = 0.99995
        self.correlationConfidence = 0.67

    def computeSpecificRisks(self, specRets, assetData, mdlClass, modelDB,
            nOkRets=None, scoreDict=None, rmgList=None, excludeAssets=None, svOverlay=None):
        """Compute the specific risk values for each asset given a
        TimeSeriesMatrix of specific returns, an array of market caps,
        and a list of lists containing SubIssues that are 'linked'.
        Assumes the same parameters (history, half-life, etc) that are
        used for specific risk computation are also used for specific
        covariance computation.
        Returns an array of specific risks for each SubIssue and a
        dictionary of dictionaries, mapping SubIssues to maps of
        linked SubIssues to their specific covariances.
        """
        self.log.debug('computeSpecificRisk: begin')
        
        # Initialise
        self.debuggingReporting = mdlClass.debuggingReporting
        self.gicsDate = mdlClass.gicsDate
        self.modelDB = modelDB
        self.scoreDict = scoreDict
        self.nOkRets = nOkRets
        self.rmgList = rmgList
        self.excludeAssets = excludeAssets
        if self.scoreDict is None:
            self.scoreDict = pandas.Series([1.0]*specRets.shape[0], index=specRets.index)
        if self.nOkRets is None:
            self.nOkRets = pandas.Series([1.0]*specRets.shape[0], index=specRets.index)
        if self.rmgList is None:
            self.rmgList = mdlClass.rmg
        if self.excludeAssets is None:
            self.excludeAssets = []
        self.trackList = []
        if hasattr(mdlClass, 'trackList'):
            self.trackList = set(mdlClass.trackList).intersection(set(specRets.index))
        self.date = max(specRets.columns)
        self.scoreDict = self.scoreDict.reindex(index=specRets.index).fillna(0.0)

        # Trim extreme returns
        opms = dict()
        opms['nBounds'] = self.srPars.clipBounds
        logging.info('*************************************** MAD weighting bounds: %s', opms['nBounds'])
        outlierClass = Outliers.Outliers(opms)
        srMatrix = outlierClass.twodMAD(Utilities.df2ma(specRets), axis=1)
        specRetsClipped = pandas.DataFrame(srMatrix, index=specRets.index, columns=specRets.columns)
        if self.debuggingReporting:
            specRetsClipped.to_csv('tmp/specRetHist-clipped-%s.csv' % self.date)
            specRets.to_csv('tmp/specRetHist-%s.csv' % self.date)

        # Compute specific variances
        if svOverlay is None:
            specificVars = self.computeSpecificVariance(specRetsClipped, assetData)
        else:
            specificVars = svOverlay.copy(deep=True)

        if self.srPars.computeISC:

            # Map groups of linked companies to their sub-issues
            linkedCIDMap = dict()
            for cidGroup in assetData.getLinkedCompanyGroups():
                thisKey = tuple(cidGroup)
                linkedCIDMap[thisKey] = []
                for cid in cidGroup:
                    sidList = assetData.getCid2SubIssueMapping()[cid]
                    linkedCIDMap[thisKey].extend(sidList)

            # Add in sub-issue mappings of non-linked companies
            linkedCIDs = set(chain.from_iterable(assetData.getLinkedCompanyGroups()))
            for (cid, sidList) in assetData.getSubIssueGroups().items():
                if cid not in linkedCIDs:
                    linkedCIDMap[tuple([cid])] = sidList

            # Downweight specific return ends-of-series
            endWts = Utilities.computePyramidWeights(20, 20, specRets.values.shape[1])
            specRetsClipped = specRetsClipped.multiply(endWts, axis=1)

            # Compute specific correlations - first directly...
            specificCorrelMap = self.computeDirectCorrelations(
                    specRetsClipped, assetData, linkedCIDMap=linkedCIDMap)

            # ...and then adjusted for cointegration
            (specificVars, specificCorrelMap) = self.computeCointegratedCovariances(
                    specRetsClipped, specificVars, assetData, specificCorrelMap)
            self.checkTrackList(self.trackList, specificVars)
        else:
            specificCorrelMap = defaultdict(dict)
            linkedCIDMap = None

        # Truncate extreme (high) values for all assets
        specificVars = numpy.clip(specificVars, None, self.maxSpecificVar)

        # Truncate extreme (low) values for all assets except selected exceptions
        lowClipAssets = set(specificVars.index).difference(set(self.excludeAssets))
        specificVars[lowClipAssets] = numpy.clip(specificVars[lowClipAssets], self.minSpecificVar, None)
        
        # Put together the specific covariance matrix
        specificVars = 252.0 * specificVars
        specificCovMap = self.computeFinalCovarianceMatrix(
                assetData, specificVars, specificCorrelMap, linkedCIDMap=linkedCIDMap)

        # Debugging output
        specificRisks = ma.sqrt(specificVars.values)
        self.log.info('Specific Risk bounds (final): [%.3f, %.3f], Mean: %.3f',
                min(specificRisks), max(specificRisks), ma.average(specificRisks))
        if self.debuggingReporting:
            self.checkISCData(specificCovMap, specificVars, assetData, linkedCIDMap=linkedCIDMap)

        self.log.debug('computeSpecificRisk: end')
        return (specificVars, specificCovMap)

    def computeSpecificVariance(self, specRets, assetData):
        """ Function to compute basic historical specific variance per asset
        """
        logging.debug('computeSpecificVariance: begin')

        # Initialise
        nwLag = self.srPars.NWLag
        if self.srPars.dateOverLap is not None:
            nwLag = 0

        # Dimension checking
        nObs = specRets.values.shape[0]
        tObs = specRets.values.shape[1]
        logging.info('Computing specific risk with %d observations for %d assets with %d day halflife',
                tObs, nObs, self.srPars.halfLife)
        if tObs > self.srPars.maxObs:
            raise LookupError('Number of time periods, %d, is greater than max number of observations, %d' % \
                    (tObs, self.srPars.maxObs))

        # Compute overlapping returns if required
        srData = specRets.copy(deep=True).fillna(0.0)
        if self.srPars.dateOverLap is not None:
            logging.info('Using overlapping %d-day returns for specific variance', self.dateOverLap)
            srTmp = numpy.array(1.0 + srData.values, copy=True)
            for jdx in range(srData.values.shape[1]-1):
                srData.values[:, jdx] = numpy.product(srTmp[:, jdx:jdx+self.srPars.dateOverLap], axis=1) - 1.0

        self.log.info('Specific return bounds: [%.6f, %.6f]',
                ma.min(srData.values, axis=None), ma.max(srData.values, axis=None))

        # Compute asset specific variance
        expWeights = Utilities.computeExponentialWeights(self.srPars.halfLife, tObs, self.srPars.equalWeightFlag)
        specificVars = Utilities.compute_NWAdj_covariance(
                srData.values, nwLag, weights=expWeights, deMean=self.srPars.deMeanFlag, axis=1, varsOnly=True)
        specificVars = pandas.Series(specificVars, index=srData.index)
        self.checkTrackList(self.trackList, specificVars)

        # Adjust risk of recent listings
        if hasattr(assetData, 'getAssetAge'):
            ageDict = pandas.Series(assetData.getAssetAge())
            maxAge = int((max(specRets.columns) - min(specRets.columns)).days)
            ageIndx = (len(expWeights) * ageDict[ageDict < maxAge] / float(maxAge)).astype(int)
            ageIndx = ageIndx.clip(1, len(expWeights))
            scaleVector = pandas.Series(1.0, index=ageIndx.index)
            sumWts = numpy.sum(expWeights, axis=None)
            for sid in ageIndx.index:
                scaleVector[sid] = sumWts / numpy.sum(expWeights[:ageIndx[sid]], axis=None)
            specificVars[ageIndx.index] = specificVars[ageIndx.index] * scaleVector

        # Output info
        specificRisks = ma.sqrt(252.0 * specificVars.values)
        self.log.info('Specific risk bounds (initial): [%.3f, %.3f], Mean: %.3f',
                min(specificRisks), max(specificRisks), ma.average(specificRisks))

        # Compute cross-sectional specific variance
        if self.srPars.useBlendedRisk:
            logging.info('Shrinking towards average')
            avSpecVar = specificVars.mask(self.nOkRets<self.srPars.blendTol)
            avSpecVar = pandas.DataFrame(avSpecVar, index=avSpecVar.index, columns=[self.date])
            dummy, avSpecVar = Utilities.proxyMissingAssetDataV4(
                    self.rmgList, avSpecVar, assetData, self.modelDB, None, robust=True,
                    estu=None, countryFactor=False, industryGroupFactor=False,
                    debugging=self.debuggingReporting, gicsDate=self.gicsDate)
            avSpecVar = ma.clip(avSpecVar.squeeze(), self.minSpecificVar, self.maxSpecificVar)

            # Combine risk components
            hl = float(self.srPars.halfLife) / 2.0
            eligibleAssets = set(specRets.index).difference(self.excludeAssets)
            eligibleAssets = eligibleAssets.intersection(set(avSpecVar[numpy.isfinite(avSpecVar)].index))
            wts = 2.0**( -(self.nOkRets * tObs) / hl)
            specificVars[eligibleAssets] = (1.0-wts[eligibleAssets]) * specificVars[eligibleAssets] \
                    + (wts[eligibleAssets] * avSpecVar[eligibleAssets])

            # Treat special cases
            excludeTypes = set([assetData.getAssetType()[sid] for sid in self.excludeAssets])
            for typ in excludeTypes:
                typeSids = [sid for sid in self.excludeAssets if assetData.getAssetType()[sid]]
                medValue = specificVars[typeSids].median(axis=None)
                specificVars[typeSids] = (1.0-wts[typeSids]) * specificVars[typeSids] + (wts[typeSids] * medValue)

            nExclude = len(set(specRets.index).difference(eligibleAssets))
            if nExclude > 0:
                logging.info('%d assets excluded from specific risk proxying', len(self.excludeAssets))
            self.checkTrackList(self.trackList, specificVars)

        # Multiply values by number of annual periods
        if self.srPars.dateOverLap is not None:
            logging.info('Adjusting for any short histories...')
            specificVars = specificVars / float(self.srPars.dateOverLap)
            self.checkTrackList(self.trackList, specificVars)

        # Output info
        specificRisks = ma.sqrt(252.0 * specificVars.values)
        self.log.info('Specific risk bounds (post-shrinkage): [%.3f, %.3f], Mean: %.3f',
            min(specificRisks), max(specificRisks), ma.average(specificRisks))
        return specificVars

    def computeDirectCorrelations(self, srMatrix, assetData, linkedCIDMap=None):
        """Compute correlations directly from specific returns histories
        """
        logging.debug('computeDirectCorrelations: begin')

        # Initialise
        nObs = srMatrix.values.shape[0]
        tObs = srMatrix.values.shape[1]
        specificCorrDict = defaultdict(dict)
        weights = Utilities.computeExponentialWeights(self.srPars.halfLife, tObs, self.srPars.equalWeightFlag)

        # Get mapping including linked CIDs if it exists
        if linkedCIDMap is None:
            cidMap = assetData.getSubIssueGroups()
        else:
            cidMap = linkedCIDMap
        self.log.info("Computing correlations for %d blocks of assets", len(cidMap))

        # Loop round subblocks of assets
        for (groupId, subIssueList) in cidMap.items():

            # Compute covariance of specific returns
            specificCov = Utilities.compute_NWAdj_covariance(
                    srMatrix.loc[subIssueList, :], self.srPars.NWLag, weights=weights, deMean=self.srPars.deMeanFlag, axis=1)

            # Convert to correlation matrix
            specificCov = pandas.DataFrame(specificCov, index=subIssueList, columns=subIssueList)
            specificCorr = self.tidyCorrelMatrix(specificCov, screen=True)
            specificCorr =  numpy.clip(specificCorr, self.minCorrel, self.maxCorrel)
            for sid in subIssueList:
                specificCorr.loc[sid, sid] = 1.0

            # "Baysesian" shrinkage towards one
            specificCorr = (1.0-self.correlationConfidence) + (self.correlationConfidence*specificCorr)
            specificCorr = self.tidyCorrelMatrix(specificCorr)
            for sid1 in specificCorr.index:
                for sid2 in specificCorr.columns:
                    specificCorrDict[sid1][sid2] = specificCorr.loc[sid1][sid2]

        # Write the correlation matrix for debug output
        if self.debuggingReporting:
            fileName = 'tmp/spec-correl-direct-%s.csv' % self.date
            self.writeCorrelMatrix(specificCorrDict, fileName, cidMap, assetData)

        return specificCorrDict

    def computeCointegratedCovariances(self, srMatrix, specificVars, assetData, specificCorrDict, TOL=1.0e-12):
        """Compute ISC information using cointegration-based techniques
        """
        logging.debug('computeCointegratedCovariances: begin')

        # Initialise
        self.log.info("Computing robust correlations for %d blocks of assets", len(assetData.getSubIssueGroups()))
        cloneSet = set(assetData.getCloneMap(cloneType='hard').keys())
        specificRisks = numpy.sqrt(specificVars)
        cointCoefDict = defaultdict(dict)
        candidateVars = defaultdict(list)
        specificCorrDF = pandas.DataFrame(specificCorrDict, copy=True)
        masterRiskDict = dict()
        minIscCorrel = 0.75

        # Get cointegration stats
        adfResults = compute_cointegration_parameters(
                numpy.fliplr(Utilities.df2ma(srMatrix)), assetData.getSubIssueGroups(), list(srMatrix.index))

        # Loop round blocks of linked assets
        for (groupId, subIssueList) in assetData.getSubIssueGroups().items():

            # Initialise
            cloneSubSet = set(subIssueList).intersection(cloneSet)
            originalSubCorrDF = specificCorrDF.loc[subIssueList, subIssueList]
            subCorrMatrix = specificCorrDF.loc[subIssueList, subIssueList]

            # Initialise cointegration stat dicts
            for sid1 in subIssueList:
                for sid2 in subIssueList:
                    cointCoefDict[sid1][sid2] = None

            # Select sub-group of CID linked assets if required
            if assetData.masterClusterDict is None:
                subGroupDict = {groupId: subIssueList}
            else:
                subGroupDict = assetData.masterClusterDict[groupId]

            # Loop round subgroups and do our stuff
            for (sgKey, smallSetSubIds) in subGroupDict.items():

                # Drop clones - to be dealt with later
                smallSetSubIds = set(smallSetSubIds).difference(cloneSubSet)
                if len(smallSetSubIds) < 1:
                    continue

                # Pick out "master" asset of the sub-group and save its risk
                masterSid = self.scoreDict[smallSetSubIds].fillna(0.0).idxmax(axis=0)
                masterRisk = specificVars[masterSid]

                # Compute cointegration specific variances
                for sid1 in smallSetSubIds:
                    masterRiskDict[sid1] = masterRisk

                    # Loop round the second of each pair
                    for sid2 in smallSetSubIds:
                        if sid1 == sid2:
                            continue

                        # Pull out cointegration stats
                        if (sid1 in adfResults.coefDict) and (sid2 in adfResults.coefDict[sid1]):

                            # Grab cointegration coefficient and p-value
                            coeff = adfResults.coefDict[sid1][sid2]
                            cointCoefDict[sid1][sid2] = coeff

                            # Compute new candidate variance for linked asset
                            if sid1 == masterSid:
                                specificVar = (coeff * coeff * 252.0 * masterRisk) + adfResults.errorVarDict[sid1][sid2]
                                specificVar /= 252.0
                                if abs(specificVar-masterRisk) < abs(specificVars[sid2]-masterRisk):
                                    candidateVars[sid2].append(specificVar)
                            elif sid2 == masterSid:
                                specificVar = (coeff * coeff * 252.0 * masterRisk) + adfResults.errorVarDict[sid2][sid1]
                                specificVar /= 252.0
                                if abs(specificVar-masterRisk) < abs(specificVars[sid1]-masterRisk):
                                    candidateVars[sid1].append(specificVar)
                        else:
                            # If no cointegration data exist, set to default
                            cointCoefDict[sid1][sid2] = 1.0
                            if sid1 == masterSid:
                                candidateVars[sid2].append(masterRisk)
                            elif sid2 == masterSid:
                                candidateVars[sid1].append(masterRisk)

            # Now choose the specific risk of each candidate pair that is closest to the master
            for sid in set(subIssueList).intersection(set(candidateVars.keys())):
                if len(candidateVars[sid]) == 1:
                    specificVars[sid] = candidateVars[sid][0]
                elif abs(candidateVars[sid][0] - masterRiskDict[sid]) < abs(candidateVars[sid][1] - masterRiskDict[sid]):
                    specificVars[sid] = candidateVars[sid][0]
                else:
                    specificVars[sid] = candidateVars[sid][1]
                specificRisks[sid] = numpy.sqrt(specificVars[sid])

            # And pick the highest correlation for each pair from the available estimates
            for sid1 in subIssueList:
                for sid2 in subIssueList:
                    if sid1 == sid2:
                        continue
                    correl1 = None
                    if cointCoefDict[sid1][sid2] is not None:
                        correl1 = cointCoefDict[sid1][sid2] * specificRisks[sid1] / specificRisks[sid2]
                    correl2 = correl1
                    if cointCoefDict[sid2][sid1] is not None:
                        correl2 = cointCoefDict[sid2][sid1] * specificRisks[sid2] / specificRisks[sid1]
                    if correl2 is None:
                        continue
                    if correl1 is None:
                        correl = correl2
                    else:
                        correl = max(correl1, correl2)
                    subCorrMatrix.loc[sid1, sid2] = numpy.clip(correl, minIscCorrel, self.maxCorrel)

            # Now force direct cloning of some pairs where required
            for cln in cloneSubSet:
                if (cln not in srMatrix.index):
                    continue
                mst = assetData.getCloneMap(cloneType='hard')[cln]
                if (mst not in srMatrix.index):
                    continue

                specificVars[cln] = specificVars[mst]
                subCorrMatrix.loc[cln, mst] = self.maxCorrel
                subCorrMatrix.loc[mst, cln] = self.maxCorrel

                # Match correlations between clone and other assets with those for the master
                for sid in subIssueList:
                    if (sid != cln) and (sid != mst):
                        subCorrMatrix.loc[cln, sid] = subCorrMatrix.loc[mst, sid]
                        subCorrMatrix.loc[sid, cln] = subCorrMatrix.loc[sid, mst]

            # Force positive semi-definiteness of the correlation matrix
            subCorrMatrix = self.tidyCorrelMatrix(subCorrMatrix, eigFix=False, screen=True, fill=self.maxCorrel)

            # If we have multiple sub-groups within the main sub-group, procrustify
            if len(subGroupDict) > 1:
                for (sgKey, smallSetSubIds) in subGroupDict.items():

                    # Fix PSDness of sub-blocks and apply transformation
                    if len(smallSetSubIds) > 1:
                        subCorrMatrix.loc[smallSetSubIds, smallSetSubIds] = \
                                self.tidyCorrelMatrix(subCorrMatrix.loc[smallSetSubIds, smallSetSubIds])
                        originalSubCorrDF = Utilities.procrustes_transform(
                                originalSubCorrDF, subCorrMatrix.loc[smallSetSubIds, smallSetSubIds])
                subCorrMatrix = self.tidyCorrelMatrix(originalSubCorrDF)
            else:
                subCorrMatrix = self.tidyCorrelMatrix(subCorrMatrix)
            for sid1 in subCorrMatrix.index:
                for sid2 in subCorrMatrix.columns:
                    specificCorrDict[sid1][sid2] = subCorrMatrix.loc[sid1][sid2]

        # Write the correlation matrix for debug output
        if self.debuggingReporting:
            outfile = 'tmp/spec-correl-coint-%s.csv' % self.date
            self.writeCorrelMatrix(specificCorrDict, outfile, assetData.getSubIssueGroups(), assetData)

        return (specificVars, specificCorrDict)

    def computeFinalCovarianceMatrix(self, assetData, specificVars, specCorrDict, linkedCIDMap=None):
        """ Pieces together the specific variances and correlations to create a specific covariance matrix
        """
        logging.debug('computeFinalCovarianceMatrix: begin')

        # Initialise
        specificCovMap = defaultdict(dict)
        specificRisks = numpy.sqrt(specificVars)
        specCorrMatrix = pandas.DataFrame(specCorrDict, copy=True)
        if not self.srPars.computeISC:
            return specificCovMap

        # Get mapping including linked CIDs if it exists
        if linkedCIDMap is None:
            cidMap = assetData.getSubIssueGroups()
        else:
            cidMap = linkedCIDMap
            # Deal with groups of linked companies
            for (cidGroup, subIssueList) in cidMap.items():

                if len(cidGroup) > 1:

                    # Compute transformation matrix for combination of correlation matrices
                    originalSpecCorr = specCorrMatrix.loc[subIssueList, subIssueList]
                    newSpecCorr = self.tidyCorrelMatrix(\
                            specCorrMatrix.loc[subIssueList, subIssueList])
                    for cid in cidGroup:
                        smallSetSubIds = assetData.getCid2SubIssueMapping()[cid]
                        if len(smallSetSubIds) > 1:
                            newSpecCorr = Utilities.procrustes_transform(
                                    newSpecCorr, originalSpecCorr.loc[smallSetSubIds, smallSetSubIds])

                    # Sanity checks for PSD-ness
                    newSpecCorr = self.tidyCorrelMatrix(newSpecCorr)
                    for sid1 in subIssueList:
                        for sid2 in subIssueList:
                            specCorrDict[sid1][sid2] = newSpecCorr.loc[sid1][sid2]

        # Now construct the covariance matrix
        for (groupId, subIssueList) in cidMap.items():
            for sid1 in subIssueList:
                for sid2 in subIssueList:
                    if sid1 == sid2:
                        continue
                    specificCovMap[sid1][sid2] = specCorrDict[sid1][sid2] * specificRisks[sid1] * specificRisks[sid2]

        # Write the correlation matrix for debug output
        if self.debuggingReporting:
            outfile = 'tmp/spec-correl-final-%s.csv' % self.date
            self.writeCorrelMatrix(specCorrDict, outfile, cidMap, assetData)

            outfile = 'tmp/spec-covar-final-%s.csv' % self.date
            self.writeCorrelMatrix(specificCovMap, outfile, cidMap, assetData)

        return specificCovMap

    def tidyCorrelMatrix(self, corrMatrix, eigFix=True, screen=False, fill=0.0, min_eigenvalue=1e-8):
        # Do some processing of correlation matrix to ensure its properties are preserved

        corrMatrixData = Utilities.df2ma(corrMatrix)
        # Eliminate undesirable data types
        if screen:
            corrMatrixData =  Utilities.screen_data(corrMatrixData, fill=True, fillValue=fill)

        # Ensure PSDness
        if eigFix:
            corrMatrixData = Utilities.forcePositiveSemiDefiniteMatrix(
                        corrMatrixData, min_eigenvalue=min_eigenvalue, quiet=True)
        corrMatrixData = (corrMatrixData + numpy.transpose(corrMatrixData)) / 2.0
        corrMatrixData = Utilities.cov2corr(corrMatrixData)[1]

        return pandas.DataFrame(corrMatrixData, index=corrMatrix.index, columns=corrMatrix.columns)

    def checkISCData(self, specCovMap, specificVars, assetData, linkedCIDMap=None):
        """Do some checks and balances on ISC data
        """
        if not self.srPars.computeISC:
            return

        # Pull up groups of linked assets
        if linkedCIDMap is None:
            cidMap = assetData.getSubIssueGroups()
        else:
            cidMap = linkedCIDMap

        # Quit, if there are no groups of linked assets
        allSids = [sid for subIssueList in cidMap.values() for sid in subIssueList]
        if len(allSids) < 1:
            logging.info('No groups of linked assets')
            return

        # Initialise
        specificRisks = numpy.sqrt(specificVars)
        corrList = []
        outfile = open('tmp/isc-%s.csv' % self.date, 'w')
        outfile.write('CID,SID,Name,Type,Score,Risk,CID,SID,Name,Type,Score,Risk,Corr,TE,\n')

        # Build ISC matrices
        companyList = sorted(cidMap.keys())
        for groupId in companyList:
            subIssueList = sorted(cidMap[groupId])
            cidCovMap = defaultdict(dict)

            for (ii, sid1) in enumerate(subIssueList):
                cidCovMap[sid1][sid1] = specificVars[sid1]
                for sid2 in subIssueList[ii+1:]:

                    # Compute correlation and tracking errors per pair
                    cidCovMap[sid1][sid2] = specCovMap[sid1][sid2]
                    cidCovMap[sid2][sid1] = specCovMap[sid2][sid1]
                    correl = specCovMap[sid1][sid2] / (specificRisks[sid1] * specificRisks[sid2])
                    corrList.append(correl)
                    trackingError = numpy.sqrt(specificVars[sid1] + specificVars[sid2] - (2.0*specCovMap[sid1][sid2]))

                    # Output data to file
                    outfile.write('%s,%s,%s,%s,%.6f,%.6f,%s,%s,%s,%s,%.6f,%.6f,%.6f,%.6f,\n' % ( \
                            assetData.getSubIssue2CidMapping()[sid1], sid1.getSubIDString(),
                            assetData.getNameMap().get(sid1, '').replace(',',''), assetData.getAssetType()[sid1],
                            self.scoreDict[sid1], specificRisks[sid1],
                            assetData.getSubIssue2CidMapping()[sid2], sid2.getSubIDString(),
                            assetData.getNameMap().get(sid2, '').replace(',',''), assetData.getAssetType()[sid2],
                            self.scoreDict[sid2], specificRisks[sid2],
                            correl, trackingError))

            # Check PSDness of each ISC submatrix
            (eigval, eigvec) = linalg.eigh(pandas.DataFrame(cidCovMap))
            if min(eigval) <= 0.0:
                logging.error('Non-positive eigenvalue, %.4f in ISC matrix for %s', min(eigval), groupId)
                assert min(eigval) >= -1.0e-16

        outfile.close()

        # Sanity check on correlations
        minCorr = min(corrList)
        maxCorr = max(corrList)
        logging.info('Specific correlations: [min: %.2f, max: %.2f]', minCorr, maxCorr)
        assert (maxCorr <= 1.0) and (minCorr >= -1.0)

        return

    def writeCorrelMatrix(self, correlMatrix, fileName, cidMap, assetData):
        outfile = open(fileName, 'w')
        outfile.write('GID,SID,Name,Type,GID,SID,Name,Type,Correl,\n')
        companyList = sorted(cidMap.keys())
        for groupId in companyList:
            subIssueList = sorted(cidMap[groupId])
            for (j, sid1) in enumerate(subIssueList):
                for sid2 in subIssueList[j+1:]:
                    outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%.6f,\n' % (\
                            assetData.getSubIssue2CidMapping()[sid1],
                            sid1.getSubIDString(),
                            assetData.getNameMap().get(sid1, '').replace(',',''),
                            assetData.getAssetType().get(sid1, ''),
                            assetData.getSubIssue2CidMapping()[sid2],
                            sid2.getSubIDString(),
                            assetData.getNameMap().get(sid2, '').replace(',',''),
                            assetData.getAssetType().get(sid2, ''),
                            correlMatrix[sid1][sid2]))
        outfile.close()
        return

    def checkTrackList(self, sidList, specVars):
        for sid in sidList:
            logging.info('Asset %s has specific risk %.2f%%', sid.getSubIDString(), 100.0*numpy.sqrt(252.0*specVars[sid]))
        return

def compute_cointegration_parameters(data, subIssueGroups, subIssues, rmgAssetMap=None, TOL=1.0e-12, skipDifferentMarkets=True):
    """Given a returns series and a dataset of linked issues,
    computes the cointegration beta between each pair of
    linked assets, along with the p-value from the
    Dickey-Fuller test  as a measure of their degree of cointegration
    """
    # Initialise important stuff
    dataCopy = Utilities.screen_data(data)
    dataCopy = ma.masked_where(dataCopy<=-1.0, dataCopy)
    maskedData = ma.getmaskarray(dataCopy)
    dataCopy = ma.filled(dataCopy, 0.0)
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(subIssues)])
    coefDict = dict()
    dfCValueDict = dict()
    errorVarDict = dict()
    dfStatDict = dict()
    pValueDict = dict()
    nobsDict = dict()

    # Get mapping of asset to RMGs
    if rmgAssetMap == None:
        assetRMGMap = dict(zip(subIssues, [1]*len(subIssues)))
    else:
        assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in rmgAssetMap.items() for sid in ids])

    import statsmodels.tsa.stattools as ts
    import statsmodels.api as sm

    # Create history of pseudo-log-prices
    prices = numpy.log(1.0 + dataCopy)
    prices = Utilities.screen_data(prices)
    prices[:,0] = 1.0
    n = prices.shape[1]
    for k in range(1,n):
        prices[:,k] += prices[:,k-1]
    results = Utilities.Struct()

    # Loop round sets of linked assets
    for (groupId, subIssueList) in subIssueGroups.items():
        indices  = [assetIdxMap[n] for n in subIssueList]

        # Loop round first of each pair
        for sid1 in subIssueList:
            coefDict[sid1] = dict()
            dfCValueDict[sid1] = dict()
            errorVarDict[sid1] = dict()
            dfStatDict[sid1] = dict()
            nobsDict[sid1] = dict()
            pValueDict[sid1] = dict()
            idx1 = assetIdxMap[sid1]
            price1 = prices[idx1,:]
            mask1 = maskedData[idx1,:]

            # Loop round second of the pair
            for sid2 in subIssueList:
                if sid1 == sid2:
                    continue
                # If assets are assigned to two different markets, impose zero cointegration and skip
                if skipDifferentMarkets and (assetRMGMap[sid1] != assetRMGMap[sid2]):
                    continue
                idx2 = assetIdxMap[sid2]
                price2 = prices[idx2,:]
                mask2 = maskedData[idx2,:]
                # Locate missing values
                nonMissingIdx = numpy.flatnonzero((mask1==0) | (mask2==0))

                # Set defaults
                coefDict[sid1][sid2] = 1.0
                errorVarDict[sid1][sid2] = 0.0
                pValueDict[sid1][sid2] = 1.0
                dfCValueDict[sid1][sid2] = 1.0
                dfStatDict[sid1][sid2] = 1.0
                nobsDict[sid1][sid2] = len(price2) - len(nonMissingIdx)

                if len(nonMissingIdx) > 4:

                    # Sift out observations where both prices are missing
                    tmpPrice1 = ma.take(price1, nonMissingIdx, axis=0)
                    tmpPrice2 = ma.take(price2, nonMissingIdx, axis=0)
                    prcDiff = ma.filled(ma.sum(abs(tmpPrice1-tmpPrice2), axis=None), 0.0)

                    # Perform OLS regression on log prices
                    if prcDiff > TOL:
                        ols_result = sm.OLS(tmpPrice2, tmpPrice1).fit()
                        coef = ols_result.params[-1]
                        if (coef > 1.5) or (coef < 0.67):
                            coef = numpy.clip(coef, 0.67, 1.5)
                        error = tmpPrice2 - (coef * tmpPrice1)
                        eps = error[1:] - error[:-1]
                        errorVar = numpy.var(eps)
                    else:
                        coef = 1.0
                        error = numpy.zeros(len(tmpPrice1), float)
                        errorVar = 0.0

                    coefDict[sid1][sid2] = coef
                    errorVarDict[sid1][sid2] = errorVar
                    nobsDict[sid1][sid2] = len(error)

                    try:
                        # Dickey-Fuller test
                        errSum = ma.sum(abs(error), axis=None)
                        if errSum < TOL:
                            logging.debug('Error term all zero, skipping ADF test')
                            adfstat = -9.0
                            cvalue = -1.0
                            pvalue = 0.0
                        else:
                            adf = ts.adfuller(error, maxlag=2, autolag=None)
                            adfstat = adf[0]
                            cvalue = adf[4]['5%']
                            pvalue = adf[1]
                        adfstat = Utilities.screen_data(ma.array([adfstat]), fill=True)[0]

                        # Save the results
                        dfCValueDict[sid1][sid2] = cvalue
                        dfStatDict[sid1][sid2] = adfstat
                        pValueDict[sid1][sid2] = pvalue

                    except:
                        logging.warning('Dickey Fuller test failed for assets (%s, %s) %d obs, Beta: %s',
                                str(sid1), str(sid2), len(nonMissingIdx), coef)

                # Shrink values to take account of deficient histories
                nonMissingVals = numpy.flatnonzero((mask1==0) * (mask2==0))
                scaleFactor = len(nonMissingVals) / float(len(price2))

    results.coefDict = coefDict
    results.errorVarDict = errorVarDict
    results.dfCValueDict = dfCValueDict
    results.nobsDict = nobsDict
    results.dfStatDict = dfStatDict
    results.pValueDict = pValueDict
    return results

def spline_dva(originalData, sampleLength, upperBound=0.1, lowerBound=-0.1,
            factorIndices=None, downWeightEnds=False):
    """Applies 'secret sauce/supertrick/DVA' to a returns series.
    data should be an asset-by-time array with no masked values
    where the most recent observation is at position 0.
    If factorIndices is not None, apply dva to only those factors in factorIndices.
    If factorIndices is None, apply dva to all factors in originalData.
    """

    logging.debug('spline_dva: begin')
    if factorIndices is None:
        factorIndices = list(range(0,originalData.shape[0]))

    if len(factorIndices) == 0:
        return originalData

    if type(originalData) is pandas.DataFrame:
        data = Utilities.df2ma(originalData)
    else:
        data = ma.array(originalData, copy=True)
    sampleLength = int(sampleLength)
    logging.debug('DVA upper bound: %s, lower bound: %s', upperBound, lowerBound)
    # Set up parameters
    # T is the length of each sample used to compute statistics
    # t is the current start point of the chunk to be scaled
    # Except for the first chunk of data, overlapping chunks
    # are used of length T, centred on point t
    # The very first chunk (that which is thereafter used as a
    # reference) runs from t=0 to t=T
    T = sampleLength
    t = 0
    tMax = data.shape[1]
    tMinusT = 0

    # Initialise some extra arrays for spline interpolation
    m = int(2 * tMax / T) + 2
    tSample = numpy.zeros((m), float)
    wtSample = ma.ones((data.shape[0], m), float)
    tFull = numpy.array((list(range(tMax))), float)

    # Setting this flag to 1 will ensure that the first half of the initial
    # chunk is unscaled, set to 0 otherwise
    initialScaleIndex = 0

    # Index of values of t in vector tSample to keep track of length of vector
    loopCount = initialScaleIndex
    if initialScaleIndex != 0:
        tSample[1] = int(T/2.0)

    logging.info('DVA Reference sample Length: %d, Type=spline', sampleLength)
    dataCopy = ma.take(data, factorIndices, axis=0)

    # Loop through the chunks of returns
    while tMinusT < tMax: # loop through time series
        # Pick out our sample of returns
        tPlusT = min(t+T, tMax)
        dataSample = abs(ma.take(dataCopy, list(range(tMinusT, tPlusT)), axis=1))
        if downWeightEnds:
            endWts = Utilities.computePyramidWeights(20, 20, dataSample.shape[1])
            dataSample *= endWts

        # Derive statistics on the returns sample
        tMAD = ma.average(dataSample, axis=1)[:,numpy.newaxis] # 232 x 1 vector
        tMAD = ma.masked_where(tMAD<=0.0, tMAD)

        if t > 0:

            # Calculate scaling factor for current chunk
            Lambda = (baseMAD / tMAD).filled(1.0)
            if (upperBound is None) and (lowerBound is None):
                outlierClass = Outliers.Outliers()
                Lambda = outlierClass.twodMAD(Lambda, suppressOutput=True, nBounds=[3.0, 3.0])

            # Trim large values
            lRatio = (Lambda / prevLambda) - 1.0
            if upperBound is not None:
                Lambda = ma.where(lRatio > upperBound, (1.0+upperBound)*prevLambda, Lambda)
            if lowerBound is not None:
                Lambda = ma.where(lRatio < lowerBound, (1.0+lowerBound)*prevLambda, Lambda)

            # Add current t value to vector of sample t values
            tSample[loopCount] = min(t, tMax)

            # Add current scale factors to vector of sample scale factors
            ma.put(wtSample[:,loopCount], factorIndices, Lambda)
            prevLambda = numpy.array(Lambda, copy=1)

            logging.debug('DVA Scaling: (Min, Med, Max): (%.3f, %.3f, %.3f)',
                    numpy.min(Lambda), ma.median(Lambda, axis=None), numpy.max(Lambda))
        else:
            # For the initial chunk, set some parameters
            baseMAD = ma.array(tMAD, copy=True)
            prevLambda = ma.ones((baseMAD.shape[0],1),float)

        # Reset various stepping parameters
        if t == 0:
            t += T
            T = int(numpy.ceil(T / 2.0))
        else:
            t += T
        # A few safeguards, to prevent overreaching array bounds or tiny chunks
        tMinusT = max(t-T, 0)
        if t > tMax-20 and t < tMax:
            t = tMax
        loopCount += 1

    # Extra stuff for spline-fitting
    tSample = tSample[initialScaleIndex:loopCount]
    wtSample = wtSample[:,initialScaleIndex:loopCount]

    #Handle case where spline.splrep will throw exception
    #and skip adjustment
    if tSample.shape[0] <= 3:
        #This should only happen because of silly inputs
        #in the RiskParameters
        if type(originalData) is pandas.DataFrame:
            data = pandas.DataFrame(data, index=originalData.index, columns=originalData.columns)
        return data

    for i in factorIndices: # loop through factors, get interpolated weighted over full history
        j = int(tSample[0])
        # Compute spline coefficients over sample
        spline_coef = spline.splrep(tSample, wtSample[i,:]) # get B-spline representation of 1-D curve given by (x=tSample, y=wtSample[i,:])
        # Compute interpolated values over full history
        wtFull = spline.splev(tFull[j:], spline_coef)
        # And scale the data by the spline-fitted interpolants
        data[i,j:] *= wtFull

    logging.debug('spline_dva: end')
    if type(originalData) is pandas.DataFrame:
        data = pandas.DataFrame(data, index=originalData.index, columns=originalData.columns)
    return data

def generate_predicted_betas(\
        assets, expMatrix, factorCov, specRisk, mktPortfolio, factorList, forceRun=False, debugging=False):
    """Compute predicted beta of each SubIssue in assets using the given risk model.
    A series of  market weights (SubIssue, weight) assets is given by marketPortfolio.
    A series of betas mapped to assets is returned
    """
    logging.debug('generate_predicted_betas: begin')

    # Make sure market portfolio is covered by model
    mktLen = len(mktPortfolio.index)
    estu = list(set(specRisk.index).intersection(mktPortfolio.index))
    weights = mktPortfolio[estu].fillna(0.0)
    weights = weights / weights.sum(axis=None)

    # Report on any missing assets/empty market
    if len(estu) < mktLen:
        logging.warning('%d assets in market portfolio not in model', mktLen - len(estu))
    if len(estu) == 0:
        logging.warning('Empty market portfolio')
        if forceRun:
            return pandas.Series(0.0, index=assets)

    # Compute market portfolio specific variance
    market_sv0 = weights.multiply(specRisk[estu])
    market_sv = market_sv0.dot(market_sv0)

    # Compute market portfolio common factor variances
    market_exp = expMatrix.loc[estu, factorList].T.dot(weights)
    market_cv_exp = factorCov.loc[factorList, factorList].dot(market_exp)
    market_var = market_exp.dot(market_cv_exp) + market_sv

    # Compute asset predicted betas
    beta = pandas.Series(0.0, index=list(assets))
    fctPart = expMatrix.loc[assets, factorList].dot(market_cv_exp)
    spcPart = market_sv0.multiply(specRisk[estu]).reindex(index=assets).fillna(0.0)
    for sid in assets:
        if not Utilities.isCashAsset(sid):
            # Compute asset factor covariance with market
            beta[sid] = (fctPart[sid] + spcPart[sid]) / market_var

    if debugging:
        logging.info('Market risk is %.6f%%', 100.0 * numpy.sqrt(market_var))
        logging.info('...Market var: %.6f, Market specific var: %.6f', market_var, market_sv)
        logging.info('...Beta range: [%.6f, %.6f,  %.6f]', min(beta), ma.mean(beta), max(beta))

    logging.debug('generate_predicted_betas: end')
    return beta

def test_for_cointegration(rm, date, data, modelDB, marketDB, returns):
    # Get cointegration results
    cointResults = compute_cointegration_parameters(
            returns.data, data.subIssueGroups, returns.assets, data.rmgAssetMap,
            skipDifferentMarkets=False)

    # Initialise
    ktDict = dict()
    avStatsDict = dict()
    dr2Underlying = dict()
    metaType = dict()
    for typ in rm.allAssetTypes:
        #if typ in rm.drAssetTypes and (typ != 'DR'):
        #    metaType[typ] = 'AllDR'
        if typ in rm.commonStockTypes:
            metaType[typ] = 'COM'
        elif typ in rm.preferredStockTypes:
            metaType[typ] = 'PREF'
        else:
            metaType[typ] = typ
        avStatsDict[metaType[typ]] = defaultdict(list)

    for sid in data.dr2Underlying.keys():
        if data.dr2Underlying[sid] is not None:
            dr2Underlying[sid] = data.dr2Underlying[sid]

    # Output cointegration results
    outfile = 'tmp/coint-%s.csv' % date
    outfile = open(outfile, 'w')
    outfile.write('GID,SID1,Type1,Mkt1,Parent1,SID2,Type2,Mkt2,Parent2,DF-Stat,C-Value,N,\n')

    companyList = sorted(data.subIssueGroups.keys())
    for groupId in companyList:
        subIssueList = sorted(data.subIssueGroups[groupId])
        for (idx1, sid1) in enumerate(subIssueList):
            type1 = data.assetTypeDict.get(sid1, None)
            mkt1 = data.marketTypeDict.get(sid1, None)
            for (idx2, sid2) in enumerate(subIssueList):
                type2 = data.assetTypeDict.get(sid2, None)
                mkt2 = data.marketTypeDict.get(sid2, None)
                # Sort out what we're comparing
                pnt1 = 0
                pnt2 = 0
                if (sid1 in dr2Underlying) and dr2Underlying[sid1] == sid2:
                    pnt2 = 1
                if (sid2 in dr2Underlying) and dr2Underlying[sid2] == sid1:
                    pnt1 = 1
                if (sid1 in data.hardCloneMap) and data.hardCloneMap[sid1] == sid2:
                    pnt2 = 1
                if (sid2 in data.hardCloneMap) and data.hardCloneMap[sid2] == sid1:
                    pnt1 = 1
                if sid1 == sid2:
                    continue
                cKey1 = '%s-%s' % (sid1.getSubIDString(), sid2.getSubIDString())
                cKey2 = '%s-%s' % (sid2.getSubIDString(), sid1.getSubIDString())
                try:
                    # Output results if not already done
                    adfStat = cointResults.dfStatDict[sid1][sid2]
                    pvalue = cointResults.dfCValueDict[sid1][sid2]
                    nobs = cointResults.nobsDict[sid1][sid2]
                    if (adfStat < 0.0) and (pvalue < 0.0):
                        if (pnt1 == 0) and (pnt2 == 0):
                            ratio = adfStat / pvalue
                            avStatsDict[metaType[type1]][metaType[type2]].append(ratio)
                            avStatsDict[metaType[type2]][metaType[type1]].append(ratio)
                    if (cKey1 not in ktDict) and (cKey2 not in ktDict):
                        outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,\n' % (groupId, \
                                sid1.getSubIDString(), type1, mkt1, pnt1,
                                sid2.getSubIDString(), type2, mkt2, pnt2,
                                adfStat, pvalue, nobs))
                        ktDict[cKey1] = True
                        ktDict[cKey2] = True
                except KeyError:
                    continue
    outfile.close()

    # Output summary stats
    metaVals = list(set(metaType.values()))
    outfile = 'tmp/coint-summ-%s.csv' % date
    outfile2 = 'tmp/coint-n-%s.csv' % date
    outfile = open(outfile, 'w')
    outfile2 = open(outfile2, 'w')
    outfile.write(',')
    outfile2.write(',')
    for typ in metaVals:
        outfile.write(',%s' % typ)
        outfile2.write(',%s' % typ)
    outfile.write(',\n')
    outfile2.write(',\n')
    for (idx, aType) in enumerate(metaVals):
        if aType in avStatsDict:
            outfile.write('%s,%s,' % (date.year, aType))
            outfile2.write('%s,%s,' % (date.year, aType))
            subDict = avStatsDict[aType]
            for bType in metaVals:
                if bType in subDict:
                    ln = len(subDict[bType])
                    summary = ma.median(subDict[bType], axis=None)
                    outfile.write('%.4f,' % summary)
                    outfile2.write('%d,' % ln)
                else:
                    outfile.write(',')
                    outfile2.write(',')
            outfile.write('\n')
            outfile2.write('\n')
    outfile.close()
    outfile2.close()

    # Output DR cointegration stats
    outfile = 'tmp/coint-dr-%s.csv' % date
    outfile = open(outfile, 'w')
    outfile.write('Type,N,MedianRatio,MeanRatio,\n')
    drStatDict = defaultdict(list)

    companyList = sorted(data.subIssueGroups.keys())
    for groupId in companyList:
        subIssueList = sorted(data.subIssueGroups[groupId])
        for sid1 in subIssueList:
            if sid1 in dr2Underlying:
                sid2 = dr2Underlying[sid1]
                type_ = data.assetTypeDict.get(sid2, None)
                if type_ in metaType:
                    found = False
                    if sid1 in cointResults.dfStatDict:
                        subDict = cointResults.dfStatDict[sid1]
                        if sid2 in subDict:
                            adfStat = cointResults.dfStatDict[sid1][sid2]
                            pvalue = cointResults.dfCValueDict[sid1][sid2]
                            found = True
                    elif sid2 in cointResults.dfStatDict:
                        subDict = cointResults.dfStatDict[sid2]
                        if sid1 in subDict:
                            adfStat = cointResults.dfStatDict[sid2][sid1]
                            pvalue = cointResults.dfCValueDict[sid2][sid1]
                            found = True
                    if found:
                        ratio = adfStat / pvalue
                    else:
                        ratio = 0.0
                    drStatDict[metaType[type_]].append(ratio)
    for (idx, aType) in enumerate(metaVals):
        if aType in drStatDict:
            ln = len(drStatDict[aType])
            summary = ma.median(drStatDict[aType], axis=None)
            summ1 = ma.average(drStatDict[aType], axis=None)
            outfile.write('%s,%d,%.4f,%.4f,\n' % (aType, ln, summary, summ1))
    outfile.close()

    return

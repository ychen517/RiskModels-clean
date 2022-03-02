import logging
import numpy.ma as ma
import numpy
from riskmodels.Matrices import ExposureMatrix
from riskmodels import LegacyUtilities as Utilities
from riskmodels import Outliers

class SimpleStandardization:
    """Basic exposure standardization.  Weighted means, taken 
    across all estimation universe assets, are subtracted from
    raw exposures, which are then divided by the equal-weighted
    standard deviation.
    """
    def __init__(self, exceptionNames=None, 
                 exp_bound=5.5, mad_bound=5.2, zero_tolerance=0.25,
                 capMean=True, fancyMAD=False, debuggingReporting=False,
                 fillWithZeroList=[]
                 ):
        self.log = logging.getLogger('Standardization.SimpleStandardization')
        self.debuggingReporting = debuggingReporting
        self.exceptionNames = exceptionNames
        self.exp_bound = exp_bound
        self.mad_bound = mad_bound
        self.capMean = capMean
        self.zero_tolerance = zero_tolerance
        self.fancyMAD = fancyMAD
        if capMean:
            logging.debug('Initialising standardisation: cap-weighted mean')
        else:
            logging.info('Initialising standardisation: root cap-weighted mean')
        self.fillWithZeroList = fillWithZeroList

    def standardize(self, expM, estu, mcaps, date, writeStats=True):
        """Standardize factors.
        """

        styleIndices = self.preProcessExposureData(expM, estu)
        assetIndices = list(range(len(expM.getAssets())))
        self.standardizeInternal(styleIndices, assetIndices, expM, estu, mcaps, writeStats)

    def preProcessExposureData(self, expM, estu):
        """Determines which factors will be standardized.
        Returns the index positions of the affected factors.
        """
        # Identify dummy (binary) styles
        dummyIndices = list()
        for idx in expM.getFactorIndices(ExposureMatrix.StyleFactor):
            if Utilities.is_binary_data(ma.take(expM.getMatrix()[idx], estu, axis=0)):
                dummyIndices.append(idx)
                self.log.info('%s identified as dummy, so not MADing',
                              expM.getFactorNames()[idx])
        
        # Extract style factors for ESTU assets only
        if self.exceptionNames is None:
            self.exceptionNames = []
        exclude_idx = [expM.getFactorIndex(f) for f in self.exceptionNames]
        if len(exclude_idx) > 0 or len(dummyIndices) > 0:
            skippedIdx = set(exclude_idx).union(set(dummyIndices))
            self.log.info('Skipping %d factors (%d excluded, %d dummy)',
                        len(skippedIdx), len(exclude_idx), len(dummyIndices))
            skipNames = [expM.getFactorNames()[idx] for idx in skippedIdx]
            self.log.info('Skipped factors: %s', ','.join(skipNames))
        styleIndices = [s for s in expM.getFactorIndices(ExposureMatrix.StyleFactor) \
                        if s not in exclude_idx + dummyIndices]

        if len(styleIndices) == 0:
            return []

        # MAD the raw exposures
        if self.fancyMAD:
            opms = dict()
            opms['nBounds'] = [self.mad_bound, self.mad_bound]
            opms['zeroTolerance'] = self.zero_tolerance
            #opms['shrink'] = True
            outlierClass = Outliers.Outliers(opms)
            expMatrix = ma.transpose(ma.take(
                expM.getMatrix(), styleIndices, axis=0))
            clippedExposures = outlierClass.twodMAD(expMatrix, axis=0, estu=estu)
            clippedExposures = ma.transpose(clippedExposures)
        else:
            (clippedExposures, bounds) = Utilities.mad_dataset(
                            ma.take(expM.getMatrix(), styleIndices, axis=0),
                            -1.0 * self.mad_bound, self.mad_bound, restrict=estu, 
                            axis=1, zero_tolerance=self.zero_tolerance)

        # Update the ExposureMatrix object
        for (i,j) in enumerate(styleIndices):
            nUnique = len(numpy.unique(ma.filled(clippedExposures[i,:], 0.0)))
            if nUnique < 3:
                logging.warning('Factor %s has only %d unique values after MADing, using unclipped values',
                        expM.getFactorNames()[j], nUnique)
            else:
                expM.getMatrix()[j,:] = clippedExposures[i,:]

        return styleIndices

    def standardizeInternal(self, factorIndices, assetIndices, expM, 
                            restrict, mcaps, writeStats, bucketName=None):
        """Main routine for standardizing a subset of raw exposure
        values, given by data.  The ExposureMatrix object given by
        expM is updated directly, for factors corresponding to index
        positions specified by factorIndices.  Mean and standard deviation
        are taken over the subset of assets specified by restrict.
        """
        meanDict = dict()
        stdDict = dict()
        if len(factorIndices) == 0:
            return

        # Compute and subtract cap weighted means
        mcaps_ESTU = numpy.take(mcaps, restrict, axis=0)
        if not self.capMean:
            mcaps_ESTU = numpy.sqrt(mcaps_ESTU)
            logging.warning('Using sqrt(cap)-weighted mean')
        data = ma.take(expM.getMatrix(), factorIndices, axis=0)
        avg = numpy.zeros(len(factorIndices))
        data_restr = ma.take(data, restrict, axis=1)
        for i in range(len(factorIndices)):
            fName = expM.getFactorNames()[factorIndices[i]]
            if fName in self.fillWithZeroList:
                data_col = ma.filled(data_restr[i,:], 0.0)
            else:
                data_col = data_restr[i,:]
            good_indices = numpy.flatnonzero(ma.getmaskarray(data_col)==0)
            weights = numpy.take(mcaps_ESTU, good_indices, axis=0)
            avg[i] = ma.average(ma.take(data_col, good_indices, axis=0), axis=None, weights=weights)
        data = ma.take(data, assetIndices, axis=1)
        data = ma.transpose(data) - avg
        
        # Compute and divide by standard deviation
        stdev = numpy.zeros((data.shape[1]))
        restrictSet = set(restrict)
        restrict_subidx = [i for (i,j) in enumerate(assetIndices) if j in restrictSet]
        data_ESTU = ma.take(data, restrict_subidx, axis=0)
        for i in range(len(stdev)):
            goodData = ma.filled(data_ESTU[:,i], 0.0)
            if len(goodData) >= 2:
                stdev[i] = ma.sqrt(ma.inner(goodData, goodData) / (goodData.shape[0] - 1.0))
            else:
                self.log.warning('%s only has %d non-missing exposures',
                            expM.getFactorNames()[factorIndices[i]], len(goodData))
            if bucketName is None:
                self.log.info('%s, mean: %f, standard deviation: %f, obs: %d',
                                expM.getFactorNames()[factorIndices[i]],
                                avg[i], stdev[i], len(goodData))
            else:
                self.log.info('Bucket: %s, %s, mean: %f, standard deviation: %f, obs: %d',
                        bucketName, expM.getFactorNames()[factorIndices[i]],
                        avg[i], stdev[i], len(goodData))
        stdev = ma.masked_where(stdev <= 0.0, stdev)
        if len(data.mask.shape)==0:
            data = ma.masked_where(ma.getmaskarray(data), data)
        data /= stdev
        for i in range(len(factorIndices)):
            fName = expM.getFactorNames()[factorIndices[i]]
            meanDict[fName] = avg[i]
            stdDict[fName] = stdev[i]
        
        if self.debuggingReporting:
            indices = [i for (i,j) in enumerate(assetIndices) if j in set(restrict)]
            tmp_data = ma.take(data, indices, axis=0).filled(0.0)
            tmp_wgt = numpy.take(mcaps, [i for i in assetIndices if i in set(restrict)])
            tmp_wgt /= numpy.sum(tmp_wgt)
            logging.info('Weighted average: %s', 
                    ','.join(['%f' % n for n in numpy.dot(numpy.transpose(tmp_data), tmp_wgt)]))
            logging.info('Standard deviation: %s', 
                    ','.join(['%f' % n for n in [numpy.inner(tmp_data[:,i], tmp_data[:,i]) / (len(indices)-1) for i in range(len(factorIndices))]]))

        # Truncate the standardized exposures
        data = ma.where(data > self.exp_bound, self.exp_bound, data)
        data = ma.where(data < -1.0 * self.exp_bound, -1.0 * self.exp_bound, data)
        
        # Update ExposureMatrix object
        if writeStats:
            if hasattr(expM, 'meanDict'):
                meanDict.update(expM.meanDict)
                stdDict.update(expM.stdDict)
            expM.meanDict = meanDict
            expM.stdDict = stdDict
        for (i,j) in enumerate(factorIndices):
            expM.getMatrix()[j,assetIndices] = data[:,i]

class BucketizedStandardization(SimpleStandardization):
    """Exposure standardization using asset buckets.  Assets are
    standardized within each asset bucket such that assets therein
    have a weighted average of zero and standard deviation of 1.0.
    """
    def __init__(self, factorScopes, exceptionNames=None, 
                 exp_bound=5.5, mad_bound=5.2, capMean=True, fancyMAD=False,
                 fillWithZeroList=[]):
        self.log = logging.getLogger('Standardization.BucketizedStandardization')
        self.factorScopes = factorScopes
        SimpleStandardization.__init__(self, exceptionNames, 
                                       exp_bound, mad_bound, 
                                       capMean=capMean, 
                                       fancyMAD=fancyMAD,
                                       fillWithZeroList=fillWithZeroList)

    def standardize(self, expM, estu, mcaps, date, writeStats=True, eligibleEstu=[]):
        """Standardize factors.
        """
        styleIndices = set(self.preProcessExposureData(expM, estu))
        styleNames = numpy.take(expM.getFactorNames(), list(styleIndices), axis=0)
        for scope in self.factorScopes:
            factorIndices = [expM.getFactorIndex(f) for f in scope.factorNames if f in styleNames]
            factorIndices = [i for i in factorIndices if i in styleIndices]
            self.log.info('Standardizing %d factors as %s', 
                    len(factorIndices), scope.description)
            for (bucketDesc, assetIndices) in scope.getAssetIndices(expM, date):
                if len(assetIndices) == 0:
                    continue
                subset_ESTU = list(set(estu).intersection(assetIndices))
                if len(subset_ESTU) == 0:
                    if len(eligibleEstu) > 0:
                        logging.info('No ESTU assets for bucket %s, using eligible instead', bucketDesc)
                        subset_ESTU = list(set(eligibleEstu).intersection(assetIndices))
                    if len(subset_ESTU) == 0:
                        logging.warning('No ESTU assets for bucket %s standardisation, skipping...', bucketDesc)
                        continue
                self.log.debug('%d assets (%d ESTU) in %s',
                        len(assetIndices), len(subset_ESTU), bucketDesc)
                self.standardizeInternal(factorIndices, assetIndices, expM, 
                                         subset_ESTU, mcaps, writeStats, bucketName=bucketDesc)

class GlobalRelativeScope:
    def __init__(self, factorNames):
        self.factorNames = factorNames
        self.description = 'Global-Relative'
    def getAssetIndices(self, expM, date):
        return [('Universe', list(range(len(expM.getAssets()))))]

class CountryRelativeScope:
    def __init__(self, factorNames):
        self.factorNames = factorNames
        self.description = 'Country-Relative'
    def getAssetIndices(self, expM, date):
        mat = expM.getMatrix()
        ctyFactorIndices = expM.getFactorIndices(ExposureMatrix.CountryFactor)
        ctyAssetIndices = [numpy.flatnonzero(ma.getmaskarray(mat[i,:])==0) \
                            for i in ctyFactorIndices]
        ctyFactorNames = expM.getFactorNames(ExposureMatrix.CountryFactor)
        return list(zip(ctyFactorNames, ctyAssetIndices))

class RegionRelativeScope:
    def __init__(self, modelDB, factorNames):
        self.factorNames = factorNames
        self.description = 'Region-Relative'
        self.regionCountryMap = dict()
        for rmg in modelDB.getAllRiskModelGroups():
            region = modelDB.getRiskModelRegion(rmg.region_id)
            if region.description not in self.regionCountryMap:
                self.regionCountryMap[region.description] = list()
            self.regionCountryMap[region.description].append(rmg.description)
     
    def getAssetIndices(self, expM, date):
        mat = expM.getMatrix()
        assetGroups = list()
        for ctyList in self.regionCountryMap.values():
            ctyIndices = list()
            for c in ctyList:
                try:
                    ctyIndices.append(expM.getFactorIndex(c))
                except:
                    continue
            if len(ctyIndices) > 0:
                assetGroups.append(numpy.flatnonzero(ma.sum(
                            ma.take(mat, ctyIndices, axis=0), axis=0).filled(0.0)))
            else:
                assetGroups.append(list())
        return list(zip(list(self.regionCountryMap.keys()), assetGroups))

class IndustryRelativeScope:
    def __init__(self, industryClassification, parentName, modelDB, factorNames):
        self.factorNames = factorNames
        self.description = '%s-Relative' % parentName
        self.industryClassification = industryClassification
        roots = [r for r in industryClassification.getClassificationRoots(modelDB)]
        rootChildrenCount = sorted((len(industryClassification.\
                        getClassificationChildren(n, modelDB)), n.name) for n in roots)
        rootLevelMap = dict([(j[1], i-(len(roots)-1)) for (i,j) \
                        in enumerate(rootChildrenCount)])
        rootLevelMap[rootChildrenCount[-1][1]] = None
        parentNode = [r for r in roots if r.name == parentName][0]
        self.levelsBack = rootLevelMap[parentName]
        self.classificationNames = [n.description for n in industryClassification.\
                        getClassificationChildren(parentNode, modelDB)]
        self.modelDB = modelDB
    def getAssetIndices(self, expM, date):
        if self.levelsBack is None:
            mat = expM.getMatrix()
        else:
            mat = self.industryClassification.getExposures(date, expM.getAssets(), 
                        self.classificationNames, self.modelDB, self.levelsBack)
        assetGroups = list()
        for (i,cls) in enumerate(self.classificationNames):
            if self.levelsBack is None:
                idx = expM.getFactorIndex(cls)
            else:
                idx = i
            assetIndices = numpy.flatnonzero(ma.getmaskarray(mat[idx,:])==0)
            assetGroups.append(assetIndices)
        return list(zip(self.classificationNames, assetGroups))

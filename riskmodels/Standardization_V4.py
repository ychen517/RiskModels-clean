import logging
import numpy.ma as ma
import numpy
import pandas
from collections import defaultdict
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities
from riskmodels import Outliers
oldPD = Utilities.oldPandasVersion()

class SimpleStandardization:
    """Basic exposure standardization.  Weighted means, taken 
    across all estimation universe assets, are subtracted from
    raw exposures, which are then divided by the equal-weighted
    standard deviation.
    Note that we set lower bounds than previously, as the new outlier code is calibrated
    to standard deviations, rather than MADs, by default. So 5.5 StDevs is approximately 8 MADs
    """
    def __init__(self, exceptionNames=None, 
                 exp_bound=5.5, mad_bound=5.5, zero_tolerance=0.25,
                 debuggingReporting=False, fillWithZeroList=[], forceMAD=False,
                 ):
        self.log = logging.getLogger('Standardization.SimpleStandardization')
        self.debuggingReporting = debuggingReporting
        self.exceptionNames = exceptionNames        # Factors to exclude from standardisation
        self.exp_bound = exp_bound                  # Hard bound for clipping of final exposure value
        self.mad_bound = mad_bound                  # MAD bound for raw exposure clipping
        self.zero_tolerance = zero_tolerance        # Max % of zeros tolerated in raw exposures for MADing
        self.forceMAD = forceMAD                    # Forces MADing of excluded factors
        logging.debug('Initialising standardisation: cap-weighted mean')
        self.fillWithZeroList = fillWithZeroList    # Fill missing with zero during standardisation
                                                    # e.g. for Dividend Yield

    def standardize(self, expM, estu, mcaps, date, eligibleEstu=[], noClip_List=[]):
        """Standardize factors.
        """
        styleIndices = self.preProcessExposureData(expM, estu, date, noClip_List)
        self.standardizeInternal(styleIndices, expM, estu, mcaps)

    def preProcessExposureData(self, expM, estu, date, noClip_List=[]):
        """Determines which factors will be standardized.
        Returns the index positions of the affected factors.
        """
        # Initialise
        expM_DF = expM.toDataFrame()
        assetIdxMap = dict(zip(expM_DF.index, range(len(expM_DF.index))))
        estuIdx = [assetIdxMap[sid] for sid in estu]
        allStyleNames = expM.getFactorNames(ExposureMatrix.StyleFactor)

        # Identify dummy (binary) styles
        dummyIndices = list()
        for fname in allStyleNames:
            expos = expM_DF.loc[:, fname]
            if len(expos[numpy.isfinite(expos)].unique()) < 3:
                dummyIndices.append(fname)
                self.log.info('%s identified as dummy, so not MADing', fname)
        
        # Drop any ineligible factors
        excludedNames = []
        if self.exceptionNames is not None:
            excludedNames = [nm for nm in self.exceptionNames if nm not in dummyIndices]
        if len(excludedNames) > 0 or len(dummyIndices) > 0:
            self.log.info('Skipping %d excluded and %d dummy factors', len(excludedNames), len(dummyIndices))
            self.log.info('Excluded factors: %s', ','.join(excludedNames))

        # Get list of factors to be treated
        styleNames = [nm for nm in allStyleNames if nm not in excludedNames + dummyIndices]
        if len(styleNames) == 0:
            if self.forceMAD:
                styleNames = [s for s in allStyleNames if s not in dummyIndices]
            else:
                return []

        # MAD the raw exposures
        opms = dict()
        opms['nBounds'] = [self.mad_bound, self.mad_bound]
        opms['zeroTolerance'] = self.zero_tolerance
        outlierClass = Outliers.Outliers(opms)
        clippedExposures = outlierClass.twodMAD(Utilities.df2ma(expM_DF.loc[:, styleNames]), axis=0, estu=estuIdx)
        clippedExposures = pandas.DataFrame(clippedExposures, index=expM_DF.index, columns=styleNames)
        
        # Update the ExposureMatrix object
        for nm in styleNames:
            if nm in noClip_List:
                logging.warning('Factor %s is in the noClip List, using unclipped values', nm)
            else:
                nUnique = len(numpy.unique(clippedExposures.loc[:, nm].fillna(0.0)))
                if nUnique < 3:
                    logging.warning('Factor %s has only %d unique values after MADing, using unclipped values', nm, nUnique)
                else:
                    expM_DF.loc[:, nm] = clippedExposures.loc[:, nm]
        expM.data_ = Utilities.df2ma(expM_DF.T)

        if self.forceMAD:
            return []
        return styleNames

    def standardizeInternal(self, factorNames, expM, subIssues, estu, mcaps, bucketName=None):
        """Main routine for standardizing a subset of raw exposure values, given by data.
        The ExposureMatrix object given by expM is updated directly,
        for factors corresponding to index positions specified by factorIndices.
        Mean and standard deviation are taken over the subset of assets specified by estu.
        """

        # Initialise
        if len(factorNames) == 0:
            return
        meanDict = dict()
        stdDict = dict()
        data = expM.toDataFrame().loc[subIssues, factorNames]
        avgExp = pandas.Series(0.0, index=factorNames)
        msk = data.isnull()

        # Loop round factor exposures
        for fname, data_col in data.items():
            data_estu = data_col[estu]
            if fname in self.fillWithZeroList:
                data_estu = data_estu.fillna(0.0)

            # Get non missing data and compute mean
            good_idx = data_estu[numpy.isfinite(data_estu)].index
            if len(good_idx) > 0:
                avgExp[fname] = numpy.average(data_estu[good_idx], weights=mcaps[good_idx])

        # Subtract weighted mean from data
        dm_data = data.subtract(avgExp.fillna(0.0), axis=1)
        
        # Check we have enough values to compute stdev
        stdev = pandas.Series(0.0, index=factorNames)
        if len(estu) < 3:
            self.log.warning('ESTU only has %d non-missing exposures', len(estu))
        else:
            # Compute equal-weighted standard deviation of demeaned data
            for fname, data_col in dm_data.items():
                goodData = data_col[estu].fillna(0.0)
                stdev[fname] = numpy.sqrt(goodData.dot(goodData) / (len(estu) - 1.0))

                # Report on stats
                if bucketName is None:
                    self.log.debug('%s, mean: %f, standard deviation: %f, obs: %d', fname, avgExp[fname], stdev[fname], len(estu))
                else:
                    self.log.debug('Bucket: %s, %s, mean: %f, standard deviation: %f, obs: %d',
                            bucketName, fname, avgExp[fname], stdev[fname], len(estu))

        # Scale data by stdev
        data = dm_data.mask(msk).div(stdev.mask(stdev <= 0.0), axis=1)
        for fname in factorNames:
            meanDict[fname] = avgExp[fname]
            stdDict[fname] = stdev[fname]
        
        if self.debuggingReporting:
            tmp_data = data.loc[estu, :]
            tmp_wgt = mcaps[estu] / mcaps[estu].sum(axis=None)
            avg = ['%f' % n for n in tmp_data.dot(tmp_wgt).values]
            stdev = ['%f' % n for n in [tmp_data[:, nm].dot(tmp_data[:, nm]) / (len(estu)-1.0) for nm in factorNames]]
            if bucketName is None:
                logging.info('Weighted average: %s', ','.join(avg))
                logging.info('Standard deviation: %s', ','.join(stdev))
            else:
                logging.info('Weighted average for bucket %s: %s', bucketName, ','.join(avg))
                logging.info('Standard deviation for bucket %s: %s', bucketName, ','.join(stdev))

        # Truncate the standardized exposures
        data = data.clip(-self.exp_bound, self.exp_bound)
        
        # Update standardisation stats
        if not hasattr(expM, 'meanDict'):
            expM.meanDict = defaultdict(dict)
        if not hasattr(expM, 'stdDict'):
            expM.stdDict = defaultdict(dict)
        expM.meanDict[bucketName].update(meanDict)
        expM.stdDict[bucketName].update(stdDict)

        # Overwrite relevant exposures
        expMFull = expM.toDataFrame()
        expMFull.loc[subIssues, factorNames] = data.loc[subIssues, factorNames]
        expM.data_ = Utilities.df2ma(expMFull.T)
        return

class BucketizedStandardization(SimpleStandardization):
    """Exposure standardization using asset buckets.
    Assets are standardized within each asset bucket such that assets therein
    have a weighted average of zero and standard deviation of 1.0.
    """
    def __init__(
            self, factorScopes, exceptionNames=None, exp_bound=5.5, mad_bound=5.5, fillWithZeroList=[], forceMAD=False):
        self.log = logging.getLogger('Standardization.BucketizedStandardization')
        self.factorScopes = factorScopes        # The bucket "type" - i.e. region, sector, global etc.
        SimpleStandardization.__init__(\
                self, exceptionNames, exp_bound, mad_bound, fillWithZeroList=fillWithZeroList, forceMAD=forceMAD)

    def standardize(self, expM, estu, mcaps, date, eligibleEstu=[], noClip_List=[]):
        """Standardize factors.
        """
        # Initialise
        styleNames = self.preProcessExposureData(expM, estu, date, noClip_List)

        # Loop round the scopes and get the buckets in each
        for scope in self.factorScopes:

            factorNames = [f for f in scope.factorNames if f in styleNames]
            self.log.info('Standardizing %d factors as %s', len(factorNames), scope.description)

            # Loop round the set of buckets for each scope
            for (bucketDesc, assets) in scope.getAssets(expM, date):
                if len(assets) == 0:
                    continue
                subset_ESTU = set(estu).intersection(assets)

                # If intersection with ESTU is zero, use the larger set of eligible assets
                if len(subset_ESTU) == 0:
                    if len(eligibleEstu) > 0:
                        logging.info('No ESTU assets for bucket %s, using eligible instead', bucketDesc)
                        subset_ESTU = set(eligibleEstu).intersection(assets)
                    if len(subset_ESTU) == 0:
                        logging.warning('No eligible or ESTU assets for bucket %s standardisation, skipping...', bucketDesc)
                        continue

                # Standardise assets across the bucket
                self.log.debug('%d assets (%d ESTU) in %s', len(assets), len(subset_ESTU), bucketDesc)
                self.standardizeInternal(\
                        factorNames, expM, assets, subset_ESTU, mcaps, bucketName=bucketDesc)

class GlobalRelativeScope:
    def __init__(self, factorNames):
        self.factorNames = factorNames
        self.description = 'Global-Relative'

    def getAssets(self, expM, date):
        return [('Universe', list(expM.getAssets()))]

class RegionRelativeScope:
    def __init__(self, modelDB, factorNames):
        self.factorNames = factorNames
        self.description = 'Region-Relative'
        self.modelDB = modelDB
     
    def getAssets(self, expM, date):

        # Build mapping from region to markets
        regionCountryMap = defaultdict(list)
        for rmg in self.modelDB.getAllRiskModelGroups():
            region = self.modelDB.getRiskModelRegion(rmg.region_id)
            regionCountryMap[region].append(rmg)

        # Get list of tuples of region to associated assets
        assetGroups = list()
        expM_DF = expM.toDataFrame()
        countryFactorList = expM.getFactorNames(ExposureMatrix.CountryFactor)

        for region, ctyList in regionCountryMap.items():
            cnts = [cnt.description for cnt in ctyList if cnt.description in countryFactorList]
            if len(cnts) > 0:
                if oldPD:
                    cntSums = expM_DF.loc[:, cnts].sum(axis=1)
                else:
                    cntSums = expM_DF.loc[:, cnts].sum(axis=1, min_count=1)
                assetGroups.append((region.description, list(cntSums[numpy.isfinite(cntSums)].index)))
            else:
                assetGroups.append((region.description, []))
        return assetGroups

# The following are not used by any production models, so we can't vouch for whether they work correctly

class CountryRelativeScope:
    def __init__(self, factorNames):
        self.factorNames = factorNames
        self.description = 'Country-Relative'

    def getAssets(self, expM, date):

        # Get list of tuples of country to associated assets
        assetGroups = list()
        expM_DF = expM.toDataFrame()
        countryFactorList = expM.getFactorNames(ExposureMatrix.CountryFactor)
    
        for cty in countryFactorList:
            cntExp = expM_DF.loc[:, cty]
            assetGroups.append((cty, list(cntExp[numpy.isfinite(cntExp)].index)))

        return assetGroups

class IndustryRelativeScope:
    def __init__(self, industryClassification, parentName, modelDB, factorNames):
        self.factorNames = factorNames
        self.description = '%s-Relative' % parentName
        self.modelDB = modelDB

        # Load industry classification
        self.industryClassification = industryClassification
        roots = [r for r in industryClassification.getClassificationRoots(modelDB)]
        rootChildrenCount = sorted((len(\
                industryClassification.getClassificationChildren(n, modelDB)), n.name) for n in roots)
        rootLevelMap = dict([(j[1], i-(len(roots)-1)) for (i,j) in enumerate(rootChildrenCount)])
        rootLevelMap[rootChildrenCount[-1][1]] = None
        parentNode = [r for r in roots if r.name == parentName][0]
        self.levelsBack = rootLevelMap[parentName]
        self.classificationNames = [n.description \
                for n in industryClassification.getClassificationChildren(parentNode, modelDB)]

    def getAssets(self, expM, date):

        # Get relevant exposure matrix
        if self.levelsBack is None:
            expM_DF = expM.toDataFrame()
        else:
            expM_DF = self.industryClassification.getExposures(date, expM.getAssets(),
                        self.classificationNames, self.modelDB, self.levelsBack, returnDF=True)

        # Get list of tuples of region to associated assets
        assetGroups = list()
        for idy in self.classificationNames:
            indExp = expM_DF.loc[:, idy]
            assetGroups.append((idy, list(indExp[numpy.isfinite(indExp)].index)))

        return assetGroups

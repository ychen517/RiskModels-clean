import copy
import datetime
import time
import logging
from collections import defaultdict
import numpy
import numpy.ma as ma
import numpy.linalg as linalg
import pandas
import sys
import itertools
import os
import socket
from itertools import chain
from riskmodels.Factors import ModelFactor
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels import ProcessReturns
from riskmodels import RiskCalculator_V4
from riskmodels import Standardization_V4
from riskmodels import Utilities
from riskmodels import EstimationUniverse_V4
from riskmodels import FactorReturns
from riskmodels import AssetProcessor
from riskmodels import AssetProcessor_V4
from riskmodels import Outliers
from riskmodels import MacroFactorReturn
oldPD = Utilities.oldPandasVersion()

class FactorRiskModel(object):
    """Abstract class defining a factor risk model.
    """
    # Various factor-type defaults
    blind = []
    styles = []
    countries = []
    industries = []
    regionalIntercepts = []
    localStructureFactors = []
    macro_core = []
    macro_market_traded= []
    macro_equity = []
    macro_sectors = []
    macros = []
    nurseryRMGs = []
    nurseryCountries = []
    hiddenCurrencies = []
    naughtyList = {'ZW': ('2020-10-02', '2020-11-02', '2999-12-31', 0.00005),
                   'VE': ('2020-10-02', '2020-11-02', '2999-12-31', 0.00005)}

    # Structure parameters
    intercept = None
    industryClassification = None
    industrySchemeDict = None
    multiCountry = False
    multiCurrency = False
    legacyLMFactorStructure = False

    # Estu parameters
    allowETFs = False
    productExcludeFlag = True
    legacySAWeight = False

    # Exposure parameters
    variableStyles = False
    legacyMCapDates = False
    regionalDescriptorStructure = True
    allowMixedFreqDescriptors = True
    rmgOverride = dict()
    gicsDate = datetime.date(2016,9,1)
    useLegacyISCScores = False
    rollOverDescriptors = 0
    dummyStyleList = []
    allowMissingFactors = ['Amihud Liquidity 60 Day', 'Amihud Liquidity 125 Day']
    grandFatherLookBack = 61

    # Regression parameters
    returnsTimingId = None
    useRobustRegression = True
    twoRegressionStructure = False
    firstReturnDate = datetime.date(1995,1,1)
    useBucketedMAD = False
    zeroExposureFactorNames = []
    zeroExposureAssetTypes = []

    # Covariance parameters
    runFCovOnly = False

    # Debugging and overrides
    debuggingReporting = False
    forceRun = False
    dplace = 8

    # Asset type meta-groups
    drAssetTypes = AssetProcessor_V4.drAssetTypes
    commonStockTypes = AssetProcessor_V4.commonStockTypes
    otherAllowedStockTypes = AssetProcessor_V4.otherAllowedStockTypes
    preferredStockTypes = AssetProcessor_V4.preferredStockTypes
    fundAssetTypes = AssetProcessor_V4.fundAssetTypes
    localChineseAssetTypes = AssetProcessor_V4.localChineseAssetTypes
    intlChineseAssetTypes = AssetProcessor_V4.intlChineseAssetTypes
    etfAssetTypes = AssetProcessor_V4.etfAssetTypes
    spacTypes = AssetProcessor_V4.spacTypes
    otherAssetTypes = AssetProcessor_V4.otherAssetTypes
    allAssetTypes = drAssetTypes + commonStockTypes + otherAllowedStockTypes + preferredStockTypes +\
            fundAssetTypes + localChineseAssetTypes + intlChineseAssetTypes + etfAssetTypes +\
            otherAssetTypes
    noProxyTypes = AssetProcessor_V4.noProxyTypes

    @classmethod
    def GetFactorObjects(cls, modelDB):
        """Returns text to put into factors for this class
        """
        query = """SELECT 'ModelFactor("'||f.name||'","'||f.description||'"),'
        FROM rms_factor rf JOIN factor f ON f.factor_id=rf.factor_id
        WHERE rf.rms_id=:rms_id AND f.name NOT IN (SELECT currency_code FROM currency_instrument_map)
        ORDER BY f.name
        """
        modelDB.dbCursor.execute(query, rms_id=cls.rms_id)
        for r in modelDB.dbCursor.fetchall():
            print(r[0])

    def __init__(self, primaryIDList, modelDB, marketDB):
        logging.info('Python version: %s, Numpy version: %s, Pandas version: %s',
                sys.version.split(' ')[0], numpy.version.version, pandas.__version__)
        logging.info('Hostname: %s', socket.gethostname())
        logging.info('DB Connection: user: %s, sid: %s', modelDB.dbConnection.username, modelDB.dbConnection.tnsentry)
        logging.info('Risk model code grabbed from: %s', os.path.dirname(AssetProcessor_V4.__file__))
        rmInfo = modelDB.getRiskModelInfo(self.rm_id, self.revision)
        self.rms_id = rmInfo.serial_id
        self.name = rmInfo.name
        self.description = rmInfo.description
        self.mnemonic = rmInfo.mnemonic
        self.log.info('Initializing risk model: %s %s (%s, %s)', self.name, self.description, self.mnemonic, self.rms_id)
        self.primaryIDList = primaryIDList
        self.rmgTimeLine = rmInfo.rmgTimeLine
        self.rmg = rmInfo.rmgTimeLine       # WRONG
        # Get model numeraire and its currency ID
        if rmInfo.numeraire == 'EUR':
            dt = datetime.date(1999, 1, 1)
        else:
            dt = min([r.from_dt for r in self.rmgTimeLine])
        from riskmodels.CurrencyRisk import ModelCurrency
        self.numeraire = ModelCurrency(rmInfo.numeraire)
        self.numeraire.currency_id = \
                    marketDB.getCurrencyID(rmInfo.numeraire, dt)
        
        # Initialize modelDB currency cache
        modelDB.createCurrencyCache(marketDB)
        
        # Get model specific hacks
        self.exposureStandardization = Standardization_V4.SimpleStandardization(debuggingReporting=self.debuggingReporting)

        if self.industryClassification is not None and \
                self.industrySchemeDict is None:
            # Only override when no industry schemeDict is found
            self.industrySchemeDict = {datetime.date(1950, 1, 1):\
                                           self.industryClassification}

        self.noClip_List = [] # control whether factors need clipping

    def isRegionalModel(self):
        return (self.hasCountryFactor and self.hasCurrencyFactor)

    def isCurrencyModel(self):
        return False
    
    def isStatModel(self):
        return False

    def isLinkedModel(self):
        return False

    def isProjectionModel(self):
        return False

    def setBaseModelForDate(self, date):
        """Determine corresponding base/parent factor model class for
        the given date and Set baseModel attribute accordingly.
        """
        try:
            baseModelDate = datetime.date(1900, 1, 1)
            for dt in self.baseModelDateMap.keys():
                if date >= dt and baseModelDate <= dt:
                    baseModelDate = dt
            self.baseModel = self.baseModelDateMap[baseModelDate]
            self.log.info('Setting base model to %s (rms_id %d)',
                    self.baseModel.name, self.baseModel.rms_id)
        except:
            raise LookupError('Unable to set base model!')
    
    def set_model_instance(self, date, modelDB, rm=None, lookBack=0):
        """Set model or sub-model instance for given date, or closest precdeing date
        """
        # If no model specified, set it to the base model
        if rm is None:
            rm = self

        # Find most recent model instance within the last lookBack days
        rmi = modelDB.getRiskModelInstance(rm.rms_id, date)
        prevDt = date
        stopDt = date - datetime.timedelta(lookBack)
        while (rmi is None) and (prevDt > stopDt):
            logging.info('No %s model on %s, looking back further', rm.name, prevDt)
            prevDt -= datetime.timedelta(1)
            rmi = modelDB.getRiskModelInstance(rm.rms_id, prevDt)
        if rmi is None:
            raise LookupError('no %s model instances found from %s to %s' % (rm.name, prevDt, date))

        # Find list of non-nursery markets
        if hasattr(rm, 'nurseryRMGs'):
            rmgList = [r for r in rm.rmg if r not in rm.nurseryRMGs]
        else:
            rmgList = rm.rmg

        return rmi, rmgList

    def getDefaultAPParameters(self, quiet=False, useNursery=True):
        """Default parameters for asset processing code
        """
        ap = dict()
        ap['debugOutput'] = self.debuggingReporting
        ap['forceRun'] = self.forceRun
        ap['legacyMCapDates'] = self.legacyMCapDates
        ap['numeraire_iso'] = self.numeraire.currency_code
        ap['quiet'] = quiet
        ap['rmgOverride'] = self.rmgOverride
        if useNursery:
            ap['nurseryRMGs'] = self.nurseryRMGs
        else:
            ap['nurseryRMGs'] = []
        ap['checkHomeCountry'] = self.coverageMultiCountry
        if hasattr(self, 'trackList'):
            ap['trackList'] = self.trackList
        return ap

    def generate_model_universe(self, modelDate, modelDB, marketDB):
        """Generate risk model instance universe and estimation
        universe.  Return value is a Struct containing a universe
        attribute (list of SubIssues) containing the model universe
        and an estimationUniverse attribute (list of index positions
        corresponding to ESTU members).
        """
        self.log.debug('generate_model_universe: begin')

        estuModel = self

        # Extra processing for statistical model
        if self.isStatModel():
            assert(self.baseModel is not None)
            estuModel = self.baseModel
            self.baseModel.rollOverDescriptors = self.rollOverDescriptors
            self.baseModel.setFactorsForDate(modelDate, modelDB)
            if hasattr(self, 'trackList') and not hasattr(self.baseModel, 'trackList'):
                self.baseModel.trackList = self.trackList

            # Prune basemodel estu dict
            for key in list(self.baseModel.estuMap.keys()):
                if key not in self.estuMap:
                    del self.baseModel.estuMap[key]
                else:
                    self.estuMap[key] = self.baseModel.estuMap[key]

            # Overwrite estimation universe parameters if necessary
            if hasattr(self, 'estu_parameters'):
                self.baseModel.estu_parameters = self.estu_parameters.copy()

        # Get basic risk model instance universe
        assetData = AssetProcessor_V4.AssetProcessor(modelDate, modelDB, marketDB, self.getDefaultAPParameters())
        universe = set(assetData.getModelAssetMaster(self))

        # Load size and IPO descriptors
        descDict = dict(modelDB.getAllDescriptors())
        sizeVec = estuModel.loadDescriptors(['LnIssuerCap'], descDict, modelDate, universe, modelDB, None,
                            rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, 'LnIssuerCap']
        ipoVec = estuModel.loadDescriptors(['ISC_IPO_Score'], descDict, modelDate, universe, modelDB, None,
                        rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, 'ISC_IPO_Score']
        sizeVec = sizeVec * ipoVec

        # Drop anything with these missing descriptors
        missingSizeIds = set(sizeVec[sizeVec.isnull()].index)
        preSPAC = assetData.getSPACs(universe=universe)
        if len(missingSizeIds) > 0:

            if self.isStatModel():
                # Make an exception for non-equity ETFs in stat models
                assetTypeDict = AssetProcessor_V4.get_asset_info(\
                        modelDate, missingSizeIds, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')
                etfs = set([sid for sid in missingSizeIds if assetTypeDict[sid] in self.etfAssetTypes])
                missingSizeIds = missingSizeIds.difference(etfs)

            # Make another exception for SPACs pre-announcement date
            overLap = missingSizeIds.intersection(preSPAC)
            if len(overLap) > 0:
                missingSizeIds = missingSizeIds.difference(preSPAC)

            if len(missingSizeIds) > 0:
                universe = universe.difference(set(missingSizeIds))
                logging.warning('%d missing LnIssuerCaps dropped from model', len(missingSizeIds))

        # Process universe data relative to RMGs
        assetData.process_asset_information(self.rmg, universe=universe)
        preSPAC = preSPAC.intersection(assetData.universe)

        # Look for unrecognised asset types
        unknownAT = set(assetData.getAssetType().values()).difference(set(self.allAssetTypes))
        if len(unknownAT) > 0:
            self.log.error('Unrecognised asset types: %s', ','.join(unknownAT))

        # Build a temporary exposure matrix to house industry and country membership
        exposureMatrix = Matrices.ExposureMatrix(assetData.universe)

        # Populate industry membership if required
        if self.industryClassification is not None:
            exposureMatrix = self.generate_industry_exposures(\
                            modelDate, modelDB, marketDB, exposureMatrix, setNull=preSPAC)

        # Populate country/currency membership for regional models
        if self.hasCountryFactor:
            exposureMatrix = self.generate_binary_country_exposures(\
                            modelDate, modelDB, marketDB, assetData, exposureMatrix, setNull=preSPAC)
        if self.hasCurrencyFactor:
            exposureMatrix = self.generate_currency_exposures(\
                            modelDate, modelDB, marketDB, assetData, exposureMatrix)

        # Generate universe of eligible assets
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                modelDate, assetData.universe, self, modelDB, marketDB,
                debugOutput=self.debuggingReporting)
        assetData.eligibleUniverse = estuCls.generate_eligible_universe(assetData)

        # Call model-specific estu construction routine
        excludeFactors = [self.descFactorMap[r.description] for r in self.nurseryRMGs]
        assetData.estimationUniverse = estuModel.generate_estimation_universe(
                modelDate, assetData, exposureMatrix, modelDB, marketDB,
                excludeFactors=excludeFactors, grandfatherRMS_ID=self.rms_id)

        # Generate mid-cap/small-cap factor estus if any
        styleNames = [s.name for s in self.styles]
        scMap = dict()
        for nm in self.estuMap.keys():
            if nm in styleNames:
                scMap[self.estuMap[nm].id] = nm
        estuModel.generate_smallcap_estus(scMap, modelDate, assetData, modelDB, marketDB)

        # If there are nursery markets, generate their estimation universe
        if len(self.nurseryCountries) > 0:
            excludeFactors = [self.descFactorMap[r.description] \
                    for r in self.rmg if r not in self.nurseryRMGs]
            estuModel.generate_nursery_estu(
                    modelDate, assetData, exposureMatrix, modelDB, marketDB,
                    self.estuMap['nursery'], assetData.nurseryUniverse, excludeFactors=excludeFactors)

        # Generate China Domestic ESTU if required
        epar = Utilities.Struct()
        if 'ChinaA' in self.estuMap:
            epar.returnsTol = self.estu_parameters['ChinaATol']
            epar.includeTypes = ['AShares']
            epar.excludeTypes = None
            epar.capBound = 5
            estuModel.generate_China_ESTU(modelDate, 'ChinaA', epar, assetData, modelDB, marketDB)

        # And China Offshore ESTU (for China SCM)
        if 'ChinaOff' in self.estuMap:
            epar.returnsTol = self.estu_parameters_ChinaOff['minGoodReturns']
            epar.includeTypes = None
            epar.excludeTypes = self.localChineseAssetTypes
            epar.capBound = self.estu_parameters_ChinaOff['cap_lower_pctile']
            estuModel.generate_China_ESTU(modelDate, 'ChinaOff', epar, assetData, modelDB, marketDB)

        # Output information on assets and their characteristics
        estuCls.estimation_universe_reporting(assetData, exposureMatrix)

        # If statistical model, copy estimation universe data from base model
        if self.isStatModel():
            for key in self.baseModel.estuMap.keys():
                if key in self.estuMap:
                    self.estuMap[key] = self.baseModel.estuMap[key]

        # Some reporting of stats
        estuIdSeries = pandas.Series(dict([(self.estuMap[ky].id, ky) for ky in self.estuMap.keys()]))
        for eid in sorted(estuIdSeries.index):
            sub_estu = self.estuMap[estuIdSeries[eid]]
            if hasattr(sub_estu, 'assets'):
                mcap_ESTU = assetData.marketCaps[sub_estu.assets].sum(axis=None)
                self.log.info('%s ESTU contains %d assets, %.2f tr %s market cap',
                        sub_estu.name, len(sub_estu.assets), mcap_ESTU / 1e12, self.numeraire.currency_code)
        self.log.info('Universe contains %d assets, %.2f tr %s market cap',
                len(assetData.universe), assetData.marketCaps.sum(axis=None) / 1e12, self.numeraire.currency_code)
        self.compareWithBM(modelDate, assetData, modelDB, marketDB)

        self.log.debug('generate_model_universe: end')
        return assetData

    def compareWithBM(self, modelDate, assetData, modelDB, marketDB):
        # Compare estimation universe coverage with given benchmark
        for compareBM in self.compareBM:
            bmData = modelDB.getIndexConstituents(compareBM, modelDate, marketDB, rollBack=30)
            bmIssues = [x for x,y in bmData]
            universeMIds = set([sid.getModelID() for sid in assetData.universe])
            mcaps = pandas.Series(assetData.marketCaps[assetData.universe].values, index=universeMIds)
            totalMCap = mcaps.sum()

            # Report on BM assets not covered in the model
            notCovered = set(bmIssues).difference(universeMIds)
            if len(notCovered) > 0:
                ncMcap =  100.0 * mcaps.reindex(index=notCovered).sum() / totalMCap
                logging.info('%d out of %d %s assets not in model universe (%.2f%% of model mcap)',
                        len(notCovered), len(universeMIds), compareBM, ncMcap)

            # Report on remaining BM asset coverage in the estimation universe
            coveredSids = set(bmIssues).difference(notCovered)
            bmCap = mcaps.reindex(index=coveredSids).sum()
            totalEstu = []
            for nm in self.estuMap.keys():
                if hasattr(self.estuMap[nm], 'assets'):
                    totalEstu.extend([sid.getModelID() for sid in self.estuMap[nm].assets])
            totalEstu = set(totalEstu).intersection(universeMIds)
            estuOverlap = coveredSids.intersection(totalEstu)
            overlapMcap =  100.0 * mcaps.reindex(index=estuOverlap).sum() / bmCap
            logging.info('%d out of %d %s assets in estimation universe (%.2f%% of BM mcap)',
                    len(estuOverlap), len(coveredSids), compareBM, overlapMcap)
            estuMissing = coveredSids.difference(totalEstu)
            if len(estuMissing) > 0:
                mdl2SIDMap = dict(zip([sid.getModelID() for sid in assetData.universe], assetData.universe))
                for mid in estuMissing:
                    sid = mdl2SIDMap[mid]
                    logging.info('Missing assets: %s, %s, %s, %s, Foreign: %s, %.4f wt',
                            mid, assetData.getNameMap().get(sid, ''), assetData.getAssetType()[sid],
                            assetData.getMarketType()[sid], sid in assetData.foreign, mcaps[mid]/bmCap)
        return
        
    def load_ISC_Scores(self, date, assetData, modelDB, marketDB, returnDF=False):
        """Either loads the linked asset scores from the descriptor table
        or computes them on the fly if they don't exist
        """

        def makeScoreDict(scores):
            # Convert score array to a dict
            scoreDict = dict()
            sid2ScoreDict = dict()
            if scores.count() > 0:
                if hasattr(assetData, 'subIssueGroups'):
                    subIssueGroups = assetData.subIssueGroups
                else:
                    subIssueGroups = assetData.getSubIssueGroups()
                for (groupId, subIssueList) in subIssueGroups.items():
                    missingScores = int(scores[subIssueList].isnull().sum())
                    if missingScores == len(subIssueList):
                        logging.warning('All ISC scores zero or missing for %s', groupId)
                        scoreDict[groupId] = scores[subIssueList].fillna(1.0).values
                    else:
                        scoreDict[groupId] = scores[subIssueList].fillna(0.0).values
                    sid2ScoreDict.update(dict(zip(subIssueList, scoreDict[groupId])))
            else:
                logging.error('No valid ISC scores saved in descriptor table')
             
            return scoreDict, sid2ScoreDict

        # Load dict of all descriptors
        descDict = dict(modelDB.getAllDescriptors())

        # Choose type of score
        if not self.useLegacyISCScores:
            return self.score_linked_assets(descDict, date, assetData, modelDB, returnDF=returnDF)
        else:
            # If previous steps have not returned scores, we compute them on the fly
            from riskmodels import DescriptorExposures
            # Set subIssueGroups on assetData
            assetData.getSubIssueGroups(universe=assetData.universe)
            scores = DescriptorExposures.generate_linked_asset_scores_legacy(\
                    date, self.rmg, assetData.universe, modelDB, marketDB,
                    assetData.subIssueGroups, self.numeraire.currency_id)
            scores = pandas.Series(ma.masked_where(scores==0.0, scores), index=assetData.universe)

        if returnDF:
            return scores
        scoreDict, sid2ScoreDict = makeScoreDict(scores)
        return scoreDict

    def score_linked_assets(self, descDict, date, assetData, modelDB, tol=1.0e-15, returnDF=False):
        """Assigns scores to linked assets based on their cap and
        liquidity, in order to assist cloning or other adjustment
        of exposures, specific risks, correlations etc.
        """
        self.log.debug('Start score_linked_assets')

        # Initialise
        scoreDict = dict()
        sid2ScoreDict = dict()

        # Back-compatibility
        if type(assetData.marketCaps) is not pandas.Series:
            marketCaps = pandas.Series(assetData.marketCaps, index=assetData.universe)
        else:
            marketCaps = assetData.marketCaps.copy(deep=True)
        if hasattr(assetData, 'subIssueGroups'):
            subIssueGroups = assetData.subIssueGroups
        else:
            subIssueGroups = assetData.getSubIssueGroups()

        # Load ADV component
        scoreType = 'ISC_ADV_Score'
        logging.debug('Loading %s data', scoreType)
        advScore = self.loadDescriptors([scoreType], descDict, date, assetData.universe, modelDB, None,
                rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, scoreType]

        # Load IPO component
        scoreType = 'ISC_IPO_Score'
        logging.debug('Loading %s data', scoreType)
        ipoScore = self.loadDescriptors([scoreType], descDict, date, assetData.universe, modelDB, None,
                rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, scoreType]

        # Load % return component
        scoreType = 'ISC_Ret_Score'
        logging.debug('Loading %s data', scoreType)
        retScore = self.loadDescriptors([scoreType], descDict, date, assetData.universe, modelDB, None,
                rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, scoreType]

        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in subIssueGroups.items():

            # Score each asset by its trading volume
            volumeSubSet = advScore[subIssueList].fillna(tol).values
            volumeSubSet = numpy.where(volumeSubSet<=0.0, tol, volumeSubSet)
            volMax = numpy.max(volumeSubSet, axis=None)
            if volMax > 0.0:
                volumeSubSet = volumeSubSet / volMax
            else:
                volumeSubSet = numpy.ones((len(subIssueList)), float)

            # Now score each asset by its market cap
            mcapSubSet = marketCaps[subIssueList].fillna(tol).values
            mcapSubSet = numpy.where(mcapSubSet<=0.0, tol, mcapSubSet)
            mcapMax = numpy.max(mcapSubSet, axis=None)
            if mcapMax > 0.0:
                mcapSubSet = mcapSubSet / mcapMax
            else:
                mcapSubSet = numpy.ones((len(subIssueList)), float)

            # Now score by proportion of non-missing returns
            nMissSubSet = retScore[subIssueList].fillna(tol).values
            nMissSubSet = numpy.where(nMissSubSet<=0.0, tol, nMissSubSet)
            nMissMax = numpy.max(nMissSubSet, axis=None)
            if nMissMax > 0.0:
                nMissSubSet = nMissSubSet / nMissMax
            else:
                nMissSubSet = numpy.ones((len(subIssueList)), float)

            # Score each asset by its age
            ageSubSet = ipoScore[subIssueList].fillna(tol).values
            ageSubSet = numpy.where(ageSubSet<=0.0, tol, ageSubSet)
            ageMax = numpy.max(ageSubSet, axis=None)
            if ageMax > 0.0:
                ageSubSet = ageSubSet / ageMax
            else:
                ageSubSet = numpy.ones((len(subIssueList)), float)

            # Now combine the scores
            score = volumeSubSet * mcapSubSet * nMissSubSet * ageSubSet
            scoreDict[groupId] = score
            sid2ScoreDict.update(dict(zip(subIssueList, score)))

        if self.debuggingReporting:
            idList = []
            scoreList = []
            for groupId in sorted(subIssueGroups.keys()):
                subIssueList = sorted(subIssueGroups[groupId])
                sidList = [groupId + ':' + sid.getSubIDString() for sid in subIssueList]
                idList.extend(sidList)
                scores = [sid2ScoreDict[sid] for sid in subIssueList]
                scoreList.extend(scores)
            Utilities.writeToCSV(numpy.array(scoreList), 'tmp/new_scores-%s.csv' % date, rowNames=idList)

        self.log.debug('End score_linked_assets')
        if returnDF:
            scoreDict = pandas.Series(sid2ScoreDict)
        return scoreDict

    def getRMDatesLegacy(self, date, modelDB, numDays,
            samplingLength=250, ceiling=True):
        """Returns a list of dates such that each model geography
        contains *approximately* numDays trading dates but trading
        holidays may also be present in the sample.
        """
        if len(self.rmg) == 1:
            return (modelDB.getDates(self.rmg, date, numDays-1), 1.0)
        dateList = modelDB.getDates(self.rmg, date, samplingLength)
        nonWeekDays = len([d for d in dateList if d.weekday() > 4])
        goodRatio = 1.0 - float(nonWeekDays) / samplingLength
        if goodRatio > 0.9:
            goodRatio = 1.0
        if ceiling:
            needDays = numpy.ceil(numDays / goodRatio)
        else:
            needDays = int(numDays / goodRatio)
        dateList = modelDB.getDates(self.rmg, date, needDays-1)
        return (dateList, goodRatio)

    def generate_currency_exposures(self, modelDate, modelDB, marketDB, assetData, exposureMatrix):
        """Generate binary currency exposures.
        """
        logging.debug('generate_currency_exposures: begin')
        allCurrencies = self.currencies + self.hiddenCurrencies

        # Set up currency exposures array
        currencyExposures = pandas.DataFrame(\
                numpy.nan, index=assetData.universe, columns=[c.name for c in allCurrencies])

        # Populate currency factors, one country at a time
        for rmg in self.rmg:
            rmg_assets = set(Utilities.readMap(rmg, assetData.rmgAssetMap, set()))
            logging.debug('Computing %s currency %s exposures, %d assets',
                    rmg.mnemonic, rmg.currency_code, len(rmg_assets))
            if len(rmg_assets) > 0:
                currencyExposures.loc[rmg_assets, rmg.currency_code] = 1.0

        # Insert currency exposures into exposureMatrix
        exposureMatrix.addFactors([c.name for c in allCurrencies], currencyExposures.T, ExposureMatrix.CurrencyFactor)
        logging.debug('generate_currency_exposures: end')
        return exposureMatrix

    def generate_binary_country_exposures(self, modelDate, modelDB, marketDB, assetData, exposureMatrix, setNull=None):
        """Assign unit exposure to the country of quotation.
        """
        logging.debug('generate_binary_country_exposures: begin')
        countryExposures = pandas.DataFrame(numpy.nan, index=assetData.universe, columns=self.rmg)
        if setNull is None:
            setNull = set()

        # Populate country factors, one at a time
        for rmg in self.rmg:
            rmg_assets = set(Utilities.readMap(rmg, assetData.rmgAssetMap, set()))
            rmg_assets = rmg_assets.difference(setNull)
            logging.debug('Computing country exposures %s, %d assets', rmg.description, len(rmg_assets))
            if len(rmg_assets) > 0:
                countryExposures.loc[rmg_assets, rmg] = 1.0
        exposureMatrix.addFactors([r.description for r in self.rmg], countryExposures.T, ExposureMatrix.CountryFactor)

        logging.debug('generate_binary_country_exposures: end')
        return exposureMatrix

    def computeExcessReturns(self, date, returns, modelDB, marketDB,
                            drAssetCurrMap=None, forceCurrencyID=None):
        """Given a TimeSeriesMatrix of asset (total) returns, 
        compute excess returns by subtracting, for each asset and
        time period, the risk-free rate corresponding to its currency
        of quotation.  If a single rfCurrency is specified, however, the 
        corresponding risk-free rate will be used for all assets.
        Returns a tuple containing the TimeSeriesMatrix of excess
        returns and risk-free rates.
        """
        self.log.debug('computeExcessReturns: begin')
        riskFreeRates = ma.zeros((returns.data.shape[0],\
                returns.data.shape[1]), float)
        # Deal with the simple case first - map risk-free
        # rates based on most recent data
        # Get mapping from asset ID to currency ID
        assetCurrMap = modelDB.getTradingCurrency(returns.dates[-1], returns.assets, marketDB, returnType='id')
        if drAssetCurrMap is not None:
            for sid in drAssetCurrMap.keys():
                assetCurrMap[sid] = drAssetCurrMap[sid]
        # Report on assets which are missing a trading currency
        missingTC = [sid for sid in assetCurrMap.keys()\
                if assetCurrMap[sid] == None]
        if len(missingTC) > 0:
            self.log.warning('Missing trading currency for %d assets: %s', len(missingTC), missingTC)
        
        # Back-compatibility point
        if forceCurrencyID is not None:
            currencies = [forceCurrencyID]
        else:
            # Assume missing TCs are numeraire for want of anything better
            assetCurrMap = dict([(sid, assetCurrMap.get(sid,
                    self.numeraire.currency_id)) \
                        for sid in returns.assets])
            # Create a map of currency indices to ID
            currencies = list(set(assetCurrMap.values()))
        currencyIdxMap = dict([(j, i) for (i, j) in enumerate(currencies)])
        
        # Get currency ISO mapping - need ISOs for rfrates
        isoCodeMap = marketDB.getCurrencyISOCodeMap()
        currencyISOs = [isoCodeMap[i] for i in currencies]
        
        # Pull up history of risk-free rates
        rfHistory = modelDB.getRiskFreeRateHistory(currencyISOs, returns.dates, marketDB)
        rfHistory.data = ma.filled(rfHistory.data, 0.0)
        
        # Back-compatibility point
        if forceCurrencyID is not None:
            riskFreeRates = rfHistory.data
        else:
            assetIdxMap = dict([(j,i) for (i,j) in enumerate(returns.assets)])
            # Map risk-free rates to assets
            for (sid, cid) in assetCurrMap.items():
                idx = assetIdxMap[sid]
                riskFreeRates[assetIdxMap[sid],:] = rfHistory.data[currencyIdxMap[cid],:]
        
        # Models requiring a rfrate history of more than one day require more 
        # complex treatment. The following ensures that changes in ISO code 
        # over time are dealt with correctly
        if len(returns.dates) > 1:
            # Remove DRs from list of assets - their local currency has been
            # forced to be constant over the period
            sidList = set(returns.assets)
            if drAssetCurrMap is not None:
                sidList = sidList.difference(set(drAssetCurrMap.keys()))
            # Check for changes in trading currency for each asset
            sidList = list(sidList)
            (allCurrencyIDs, tcChangeAssets) = self.getAssetCurrencyChanges(sidList, returns.dates, modelDB, marketDB)
            
            # If there are no currency changes, we're finished
            if len(allCurrencyIDs) > 0:
                self.log.info('Trading currency changes for %s assets, %s currencies',
                              len(list(tcChangeAssets.keys())), len(allCurrencyIDs))
                allCurrencyIDs = list(allCurrencyIDs)
                currencyIdxMap = dict([(j,i) for (i,j) in enumerate(allCurrencyIDs)])
                # Get currency ISO mapping - need ISOs for rfrates
                isoCodeMap = marketDB.getCurrencyISOCodeMap()
                allCurrencyISOs = [isoCodeMap[i] for i in allCurrencyIDs]
                self.log.info('Extra risk-free rates needed for %s',
                              allCurrencyISOs)
                
                # Pull up a history of risk-free rates for all currencies
                rfHistory = modelDB.getRiskFreeRateHistory(
                        allCurrencyISOs, returns.dates, marketDB)
                rfHistory.data = ma.filled(rfHistory.data, 0.0)
                
                # Build a date-mapping to speed things
                dateIdxMap = dict([(d,i) for (i,d) in enumerate(returns.dates)])
                dt = returns.dates[-1]
                idx = None
                while dt > returns.dates[0]:
                    if dt not in dateIdxMap:
                        dateIdxMap[dt] = idx
                    idx = dateIdxMap[dt]
                    dt -= datetime.timedelta(1)
                
                # Loop through assets which have changed currency
                for (sid, tcChanges) in tcChangeAssets.items():
                    for chng in tcChanges:
                        cid = chng.id
                        d0 = dateIdxMap.get(chng.fromDt, 0)
                        d1 = dateIdxMap.get(chng.thruDt, len(returns.dates) + 1)
                        # Pull out rfrate
                        riskFreeRates[assetIdxMap[sid], d0:d1] = rfHistory.data[currencyIdxMap[cid], d0:d1]
        
        # Finally compute excess returns
        returns.data = returns.data - riskFreeRates
        self.log.debug('computeExcessReturns: end')
        return (returns, rfHistory)
    
    def computePredictedBeta(self, date, modelData, modelDB, marketDB):
        """Compute the predicted beta of the assets for the given risk
        model instance, i.e., all assets that have specific risk.
        modelData is a Struct that contains the exposure
        matrix (exposureMatrix), factor covariance matrix (factorCovariance),
        and specific risk (specificRisk) of the instance.
        The return value is a list of (SubIssue, value) tuples containing
        all assets for which a specific risk is available.
        """

        # Get our factors straight
        factorNames = [f.name for f in self.factors]
        nonCurrencyFactors = None
        if hasattr(self, 'currencies'):
            nonCurrencyFactors = [f.name for f in self.factors if f not in self.currencies]

        # Process asset country assignments
        rmi = modelDB.getRiskModelInstance(self.rms_id, date)
        universe = list(modelData.specificRisk.keys())
        rmgList = [r for r in self.rmg if r not in self.nurseryRMGs]
        assetData = AssetProcessor_V4.AssetProcessor(\
                date, modelDB, marketDB, self.getDefaultAPParameters(useNursery=False))
        assetData.process_asset_information(rmgList, universe=universe)
        preSPAC = assetData.getSPACs()

        # Initialise
        mcapSwitch = False
        localBetas = pandas.Series(numpy.nan, index=assetData.universe)
        legacyBetas = pandas.Series(numpy.nan, index=assetData.universe)
        globalBetas = pandas.Series(numpy.nan, index=assetData.universe)

        # Set up model data
        expMatrix = modelData.exposureMatrix.toDataFrame().loc[assetData.universe, :]
        specificRisk = pandas.Series(modelData.specificRisk)[assetData.universe]
        factorCovariance = pandas.DataFrame(\
                modelData.factorCovariance, index=factorNames, columns=factorNames)

        # Some rearranging for linked models
        if self.isLinkedModel():
            runningList = set(assetData.universe)
            oldAssetRMG = dict()
            for rmg in rmgList:
                if rmg.description in expMatrix.columns:
                    rmgExpos = expMatrix.loc[:, rmg.description]
                else:
                    rmgExpos = expMatrix.loc[:, self.ri2CountryMap[rmg.description]]
                rmgAssets = set(rmgExpos[numpy.isfinite(rmgExpos.mask(rmgExpos==0.0))].index)
                oldAssetRMG[rmg] = set(assetData.rmgAssetMap[rmg]).difference(rmgAssets)
                assetData.rmgAssetMap[rmg] = rmgAssets
                runningList = runningList.difference(rmgAssets)
            if len(runningList) > 0:
                for rmg, rmgAssets in oldAssetRMG.items():
                    overlap = rmgAssets.intersection(preSPAC)
                    if len(overlap) > 0:
                        logging.info('Adding %d SPACs with no country exposure to %s RMG', len(overlap), rmg.mnemonic)
                        assetData.rmgAssetMap[rmg] = assetData.rmgAssetMap[rmg].union(overlap)
                        runningList = runningList.difference(overlap)
                if len(runningList) > 0:
                    logging.info('%d linked model assets have no country exposure - probably SPACs', len(runningList))
                    localBetas[runningList] = 0.0
                    legacyBetas[runningList] = 0.0

        # Deal with global beta first
        if self.isRegionalModel():
            self.log.info('Computing %d predicted (regional) betas', len(assetData.universe))

            # Get estimation universe assets
            estu = set(self.loadEstimationUniverse(rmi, modelDB, assetData))
            estu = list(estu.intersection(assetData.universe))

            # Construct on-the-fly market portfolio
            if mcapSwitch:
                marketPortfolio = assetData.getMarketCaps()[estu]
            else:
                mcapDates = modelDB.getDates(self.rmg, date, 19)
                marketPortfolio = modelDB.getAverageMarketCaps(\
                        mcapDates, estu, self.numeraire.currency_id, marketDB, returnDF=True)

            # Compute the betas
            globalBetas = RiskCalculator_V4.generate_predicted_betas(
                    assetData.universe, expMatrix, factorCovariance, specificRisk, marketPortfolio, factorNames,
                    self.forceRun, debugging=self.debuggingReporting)

        # Loop through RMGs and compute local betas
        for rmg in rmgList:
            beta_assets = list(set(universe).intersection(assetData.rmgAssetMap[rmg]))

            # Hack for Domestic China A-Shares
            if (rmg.description == 'China') and (len(beta_assets) > 0):

                # Sort out Chinese A-Shares
                aShares = [sid for sid in beta_assets if assetData.getAssetType()[sid] in self.localChineseAssetTypes]
                self.log.info('Computing %d predicted (local) betas for Domestic China', len(aShares))

                # Get Chinese domestic market portfolio
                dcRMG = modelDB.getRiskModelGroupByISO('XC')
                aShareAMP = modelDB.getRMGMarketPortfolio(dcRMG, date, returnDF=True,
                                    amp=modelDB.convertLMP2AMP(dcRMG, date))
                assert(aShareAMP is not None)

                if len(aShareAMP) > 0:
                    # Compute Chinese A-share betas relative to domestic market
                    legacyBetas[aShares] = RiskCalculator_V4.generate_predicted_betas(
                            aShares, expMatrix, factorCovariance, specificRisk, aShareAMP, factorNames,
                            self.forceRun, debugging=self.debuggingReporting)

                    if nonCurrencyFactors is not None:
                        # Compute A-share beta in in local currency
                        localBetas[aShares] = RiskCalculator_V4.generate_predicted_betas(
                                aShares, expMatrix, factorCovariance, specificRisk, aShareAMP, nonCurrencyFactors,
                                self.forceRun, debugging=self.debuggingReporting)

                # Remove A-shares from list of Chinese assets
                beta_assets = [sid for sid in beta_assets if sid not in aShares]

            # Pick up market portfolio
            self.log.info('Computing %d predicted (local) betas for %s', len(beta_assets), rmg.description)
            amp = modelDB.getRMGMarketPortfolio(rmg, date, returnDF=True, amp=modelDB.convertLMP2AMP(rmg, date))
            assert(amp is not None)

            if (len(amp) > 0) and (len(beta_assets) > 0):
                # Compute predicted betas for given RMG
                legacyBetas[beta_assets] = RiskCalculator_V4.generate_predicted_betas(
                        beta_assets, expMatrix, factorCovariance, specificRisk, amp, factorNames,
                        self.forceRun, debugging=self.debuggingReporting)

                if nonCurrencyFactors is not None:
                    # Compute beta in local currency
                    localBetas[beta_assets] = RiskCalculator_V4.generate_predicted_betas(
                            beta_assets, expMatrix, factorCovariance, specificRisk, amp, nonCurrencyFactors,
                            self.forceRun, debugging=self.debuggingReporting)
            else:
                self.log.warning('ALERT: RMG: %s has %d assets and %d AMP weights', rmg.mnemonic, len(beta_assets), len(amp))
                if not self.forceRun:
                    assert((len(beta_assets)>0) and (len(amp)>0))

        # Set up return value dicts
        legacyBetas = legacyBetas[numpy.isfinite(legacyBetas)].to_dict()
        localBetas = localBetas[numpy.isfinite(localBetas)].to_dict()
        globalBetas = globalBetas[numpy.isfinite(globalBetas)].to_dict()
        allKeys = set(legacyBetas.keys()).union(localBetas.keys()).union(globalBetas.keys())
        predictedBetas = [(sid, \
                globalBetas.get(sid, None), legacyBetas.get(sid, None), localBetas.get(sid, None)) for sid in allKeys]

        # A bit of reporting
        if self.debuggingReporting:
            assetRMGMap = Utilities.flip_dict_of_lists(assetData.rmgAssetMap)

        if len(globalBetas) > 0:
            self.log.info('Predicted global beta bounds: [%.3f, %.3f], Median: %.3f',\
                    ma.min(list(globalBetas.values()), axis=None),
                    ma.max(list(globalBetas.values()), axis=None),
                    ma.median(list(globalBetas.values()), axis=None))
            if self.debuggingReporting:
                sidList = ['%s|%s' % (assetRMGMap[sid].rmg_id if sid in assetRMGMap else None,\
                        sid.getSubIDString()) for sid in sorted(globalBetas.keys())]
                betaList = ma.array([globalBetas[sid] for sid in sorted(globalBetas.keys())], float)[:, numpy.newaxis]
                Utilities.writeToCSV(betaList, 'tmp/globalBeta-%s-%s.csv' % (self.mnemonic, date), rowNames=sidList)

        if len(legacyBetas) > 0:
            self.log.info('Predicted legacy beta bounds: [%.3f, %.3f], Median: %.3f',\
                    ma.min(list(legacyBetas.values()), axis=None),
                    ma.max(list(legacyBetas.values()), axis=None),
                    ma.median(list(legacyBetas.values()), axis=None))
            if self.debuggingReporting:
                sidList = ['%s|%s' % (assetRMGMap[sid].rmg_id if sid in assetRMGMap else None,\
                        sid.getSubIDString()) for sid in sorted(legacyBetas.keys())]
                betaList = ma.array([legacyBetas[sid] for sid in sorted(legacyBetas.keys())], float)[:, numpy.newaxis]
                Utilities.writeToCSV(betaList, 'tmp/legacyBeta-%s-%s.csv' % (self.mnemonic, date), rowNames=sidList)
        else:
            self.log.warning('No predicted betas generated')

        if len(localBetas) > 0:
            self.log.info('Predicted local beta bounds: [%.3f, %.3f], Median: %.3f',\
                    ma.min(list(localBetas.values()), axis=None),
                    ma.max(list(localBetas.values()), axis=None),
                    ma.median(list(localBetas.values()), axis=None))
            if self.debuggingReporting:
                sidList = ['%s|%s' % (assetRMGMap[sid].rmg_id if sid in assetRMGMap else None,\
                        sid.getSubIDString()) for sid in sorted(localBetas.keys())]
                betaList = ma.array([localBetas[sid] for sid in sorted(localBetas.keys())], float)[:, numpy.newaxis]
                Utilities.writeToCSV(betaList, 'tmp/localBeta-%s-%s.csv' % (self.mnemonic, date), rowNames=sidList)

        # Report on anything missing and return
        missingBeta = set(assetData.universe).difference(allKeys)
        if len(missingBeta) > 0:
            self.log.warning('Missing predicted betas: %d out of %d total assets', \
                                len(missingBeta), len(assetData.universe))
        return predictedBetas

    def regressionReporting(self, excessReturns, result, expM, nameSubIDMap, assetIdxMap,
                            modelDate, buildFMPs=True, constrComp=None, specificRets=None):
        """Debugging reporting for regression step
        """

        # Output total and specific return info
        if self.debuggingReporting and (specificRets is not None):
            outfile = open('tmp/specificReturns-%s-%s.csv' % (self.name, modelDate), 'w')
            outfile.write('subIssue,excessReturn,specificReturn,\n')
            for (sid, ret, sret) in zip(expM.getAssets(), excessReturns, specificRets):
                if ret is ma.masked:
                    if sret is ma.masked:
                        outfile.write('%s,,,\n' % sid if isinstance(sid, str) else sid.getSubIDString())
                    else:
                        outfile.write('%s,,%.8f,\n' % (sid if isinstance(sid, str) else sid.getSubIDString(), sret))
                elif sret is ma.masked:
                    outfile.write('%s,%.8f,,\n' % (sid if isinstance(sid, str) else sid.getSubIDString(), ret))
                else:
                    outfile.write('%s,%.8f,%.8f,\n' % (sid if isinstance(sid, str) else sid.getSubIDString(), ret, sret))
            outfile.close()

        if len(self.nurseryCountries) > 0:
            factors = self.factors + self.nurseryCountries
        else:
            factors = self.factors
        # Construct FMP-derived factor returns
        tmpFRets = []
        tmpFMPRets = []
        fmpRetDict = dict()
        fmpExpDict = dict()
        allSids = []
        for i in range(len(result.factorReturns)):
            if factors[i].name not in nameSubIDMap:
                continue
            sf_id = nameSubIDMap[factors[i].name]
            if sf_id in result.fmpMap:
                fmpMap = result.fmpMap[sf_id]
                sidList = list(fmpMap.keys())
                allSids.extend(sidList)
                fmpWts = numpy.array([fmpMap[sid] for sid in sidList], float)
                sidIdxList = [assetIdxMap[sid] for sid in sidList]
                subRets = ma.filled(ma.take(excessReturns, sidIdxList, axis=0), 0.0)
                fmpFR = numpy.inner(fmpWts, subRets)
                fmpRetDict[sf_id] = fmpFR
                tmpFRets.append(result.factorReturns[i])
                tmpFMPRets.append(fmpFR)
                fmpExp = []
                if self.debuggingReporting:
                    for j in range(len(result.factorReturns)):
                        expMCol = expM.getMatrix()[expM.getFactorIndex(factors[j].name),:]
                        expMCol = ma.filled(ma.take(expMCol, sidIdxList, axis=0), 0.0)
                        fExp = numpy.inner(fmpWts, expMCol)
                        if constrComp is not None:
                            if (factors[j].name in constrComp.ccDict) and (factors[i].name in constrComp.ccDict):
                                constrCol = constrComp.ccDict[factors[i].name]
                                constrX = ma.filled(constrComp.ccXDict[factors[j].name], 0.0)
                                constrContr = numpy.inner(constrCol, constrX)
                                fExp += constrContr
                        fmpExp.append(fExp)
                    fmpExpDict[sf_id] = fmpExp

        allSids = sorted(set(allSids))
        if len(tmpFRets) > 0:
            tmpFRets = numpy.array(tmpFRets, float)
            tmpFMPRets = numpy.array(tmpFMPRets, float)
            correl = numpy.corrcoef(tmpFRets, tmpFMPRets, rowvar=False)[0,1]
            self.log.info('Correlation between factor returns and FMP returns: %.6f', correl)

        if self.debuggingReporting:
            outfile = open('tmp/factorReturns-%s-%s.csv' % (self.name, modelDate), 'w')
            outfile.write('factor,return,stderr,tstat,prob,constr_wt,fmp_ret\n')
            for i in range(len(result.factorReturns)):
                sf_id = nameSubIDMap[factors[i].name]
                fmpFR = fmpRetDict.get(sf_id, None)
                msk = numpy.flatnonzero(ma.getmaskarray(result.regressionStatistics[i, :]))
                if len(msk) > 0:
                    outfile.write('%s,%.8f,,,,,' % \
                        (factors[i].name.replace(',',''), result.factorReturns[i]))
                else:
                    outfile.write('%s,%.8f,%.8f,%.8f,%.8f,%.8f,' % \
                            (factors[i].name.replace(',',''),
                                result.factorReturns[i],
                                result.regressionStatistics[i,0],
                                result.regressionStatistics[i,1],
                                result.regressionStatistics[i,2],
                                result.regressionStatistics[i,3]))
                if fmpFR is None:
                    outfile.write(',\n')
                else:
                    outfile.write('%.8f,\n' % fmpFR)

            outfile.close()

            outfile = open('tmp/fmp-exposures.csv', 'w')
            for i in range(len(result.factorReturns)):
                outfile.write(',%s' % factors[i].name.replace(',',''))
            outfile.write(',\n')
            for i in range(len(result.factorReturns)):
                outfile.write('%s,' % factors[i].name.replace(',',''))
                sf_id = nameSubIDMap[factors[i].name]
                fmpExp = fmpExpDict.get(sf_id, [])
                for j in range(len(result.factorReturns)):
                    if len(fmpExp) == 0:
                        outfile.write(',')
                    else:
                        outfile.write('%.8f,' % fmpExp[j])
                outfile.write('\n')
            outfile.close()

        if self.debuggingReporting:
            if buildFMPs:
                outfile = open('tmp/fmp-fwd-%s-%s.csv' % (self.mnemonic, modelDate), 'w')
            else:
                outfile = open('tmp/fmp-%s-%s.csv' % (self.mnemonic, modelDate), 'w')
            outfile.write('subissue|')
            for i in range(len(result.factorReturns)):
                sf_id = nameSubIDMap[factors[i].name]
                if sf_id in result.fmpMap:
                    outfile.write('%s|' % factors[i].name.replace(',',''))
            outfile.write('\n')
            for sid in allSids:
                outfile.write('%s|' % sid.getSubIDString())
                for i in range(len(result.factorReturns)):
                    sf_id = nameSubIDMap[factors[i].name]
                    if sf_id in result.fmpMap:
                        fmpMap = result.fmpMap[sf_id]
                        if sid in fmpMap:
                            outfile.write('%.8f|' % fmpMap[sid])
                        else:
                            outfile.write('|')
                outfile.write('\n')
            outfile.close()
        return

    def computeTotalRisk(self, modelData, modelDB):
        """Compute the total risk for all assets in the factor risk model
        provided in modelData.
        modelData is a struct that contains the exposure
        matrix (exposureMatrix), factor covariance matrix (factorCovariance),
        and specific risk (specificRisk).
        The return value is a list of (SubIssue, value) tuples containing
        all assets for which a specific risk is available.
        """
        # Initialise
        factorNames = [f.name for f in self.factors]
        expMatrix = modelData.exposureMatrix.toDataFrame()
        factorCovariance = pandas.DataFrame(modelData.factorCovariance, index=factorNames, columns=factorNames)
        assets = sorted(set(modelData.specificRisk.keys()).intersection(set(expMatrix.index)))
        specRisk = pandas.Series(modelData.specificRisk)
        totalRisks = pandas.Series(numpy.nan, index=assets)

        # Compute asset specific risk
        fc_dot_exp = factorCovariance.dot(expMatrix.T)
        for sid in assets:
            fv = expMatrix.loc[sid, :].dot(fc_dot_exp.loc[:, sid])
            sr = specRisk[sid] * specRisk[sid]
            totalRisks[sid] = numpy.sqrt(fv + sr) 

        if self.debuggingReporting:
            sidList = [sid.getSubIDString() for sid in totalRisks.index]
            fname = 'tmp/totalRisks-%s.csv' % self.mnemonic
            Utilities.writeToCSV(totalRisks.values, fname, rowNames=sidList)

        return list(zip(assets, totalRisks))
    
    def returns_timing_v3(self, dateList, nonCurrencySubFactors, nonCurrencyFactorReturns, currencyFactorReturns):
        # *************** Test code for returns-timing V3

        # Initialise parameters
        factorBeta = None
        marketInterceptDifference = None

        # Test code for next generation returns-timing
        if self.useReturnsTimingV3 and self.twoRegressionStructure and (self.intercept is not None):
            regHistLen = 500
            # Load raw (external) factor returns
            externalFactorReturns = modelDB.loadFactorReturnsHistory(
                    self.rms_id, nonCurrencySubFactors, dateList[:ret.maxOmegaObs], screen=True)
            efrData = ma.filled(externalFactorReturns.data, 0.0)
            efrAss = externalFactorReturns.assets

            # Find market intercept factor
            intIdx = [fIdx for (fIdx, f) in enumerate(efrAss) if f.factor == self.intercept][0]

            # Compute each factor beta to the market intercept return
            factorBeta = dict()
            for (idx, sf) in enumerate(efrAss):
                if sf.factor in [self.intercept] + self.countries:
                    factorBeta[sf] = [0.0] * efrData.shape[1]
                    continue
                factorBeta[sf] = []
                fb = 0.0
                for i_dt in range(efrData.shape[1]):
                    facret = efrData[idx, i_dt:]
                    rawIntRet = efrData[intIdx, i_dt:]
                    if len(facret) < regHistLen:
                        factorBeta[sf].append(fb)
                    else:
                        facret = facret[:regHistLen]
                        rawIntRet = rawIntRet[:regHistLen]
                        intProd = numpy.inner(rawIntRet, rawIntRet)
                        fb = numpy.inner(facret, rawIntRet) / intProd
                        factorBeta[sf].append(fb)

            # Do the same for currency factor returns
            for (idx, sf) in enumerate(currencyFactorReturns.assets):
                factorBeta[sf] = []
                fb = 0.0
                for i_dt in range(efrData.shape[1]):
                    facret = currencyFactorReturns.data[idx, i_dt:]
                    rawIntRet = efrData[intIdx, i_dt:]
                    if len(facret) < regHistLen:
                        factorBeta[sf].append(fb)
                    else:
                        facret = facret[:regHistLen]
                        rawIntRet = rawIntRet[:regHistLen]
                        intProd = numpy.inner(rawIntRet, rawIntRet)
                        fb = numpy.inner(facret, rawIntRet) / intProd
                        factorBeta[sf].append(fb)

            # Compute the difference between internal and external market intercept returns
            intIntIdx = [fIdx for (fIdx, f) in enumerate(nonCurrencyFactorReturns.assets) \
                    if f.factor == self.intercept][0]
            marketInterceptDifference = ma.filled(nonCurrencyFactorReturns.data[intIntIdx,:], 0.0) - efrData[intIdx,:]
        # In the code we later need something like this:
        # Adjust non-market factors for returns timing if relevant
        #if factorBeta is not None:
        #    for (idx, sf) in enumerate(nonCurrencyFactorReturns.assets):
        #        nonCurrencyFactorReturns.data[idx,:] += factorBeta[sf] * marketInterceptDifference
        #    for (idx, sf) in enumerate(currencyFactorReturns.assets):
        #        currencyFactorReturns.data[idx,:] += factorBeta[sf] * marketInterceptDifference

        return marketInterceptDifference, factorBeta

    def createInstance(self, date, modelDB):
        """Creates a new risk model instance for this risk model serie
        on the given date.
        """
        return modelDB.createRiskModelInstance(self.rms_id, date)
    
    def createRiskModelSerie(cls, modelDB, marketDB):
        """Creates a new serial number in the database and initializes
        it. The revision strings are set to ''.
        The factors are linked in rms_factor.
        """
        # First check that we have all factors
        factors = []
        for factorType in [cls.styles, cls.blind, cls.macro_core, cls.macro_market_traded, cls.macro_equity, cls.macro_sectors]:
            if len(factorType) > 0:
                factors.extend([f.description for f in factorType])
        if not cls.industryClassification is None:
            factors.extend(cls.industryClassification.getDescriptions(
                modelDB))
        if len(cls.rmg) > 1:
            factors.extend([r.description for r in cls.rmg])
            currencies = set([r.currency_code for r in cls.rmg])
            factors.extend(list(currencies))
        factorIDs = modelDB.getFactors(factors)
        if len([i for i in factorIDs if i == None]) > 0:
            factorToID = list(zip(factors, factorIDs))
            print([(i,j) for (i,j) in factorToID if j is None])
            logging.fatal('Factors missing in database')
            return
        modelDB.createRiskModelSerie(
            cls.rms_id, cls.rm_id, cls.revision)
        modelDB.insertRMSFactors(cls.rms_id, factorIDs)
    
    def deleteExposures(self, rmi, modelDB, subIssues=[]):
        """Delete the exposures for the given sub-factors.
        """
        modelDB.deleteRMIExposureMatrixNew(rmi, subIssues=subIssues)
    
    def deleteInstance(self, date, modelDB):
        """Deletes the risk model instance for the given date if it exists.
        """
        rmi_id = modelDB.getRiskModelInstance(self.rms_id, date)
        if rmi_id != None:
            modelDB.deleteRiskModelInstance(rmi_id, True)
    
    def deleteRiskModelSerie(cls, modelDB):
        info = modelDB.getRiskModelInfo(cls.rm_id, cls.revision)
        assert(cls.rms_id == info.serial_id)
        modelDB.deleteRiskModelSerie(cls.rms_id, True)
    
    createRiskModelSerie = classmethod(createRiskModelSerie)
    deleteRiskModelSerie = classmethod(deleteRiskModelSerie)
    
    def generate_model_specific_exposures(self, modelDate, data, exposureMatrix, modelDB, marketDB):
        """Any model-specific factor exposure computations should
        be placed in a generate_model_specific_exposures() method
        under the corresponding FactorRiskModel class.
        Data should be a Struct containing all required data items
        for factor computation and the exposureMatrix attribute, which
        this method should ultimately return.
        """
        return exposureMatrix

    def generate_industry_exposures(self, modelDate, modelDB, marketDB, expMatrix, setNull=None):
        """Create the industry exposures for the assets in the given exposure
        matrix and adds them to the matrix.
        """
        self.log.debug('generate industry exposures: begin')
        factorList = [f.description for f in list(self.industryClassification.getLeafNodes(modelDB).values())]
        exposures = self.industryClassification.getExposures(
                modelDate, expMatrix.getAssets(), factorList, modelDB, returnDF=True)
        if setNull is not None:
            exposures.loc[setNull, :] = numpy.nan
        expMatrix.addFactors(factorList, exposures.T, ExposureMatrix.IndustryFactor)
        self.log.debug('generate industry exposures: end')
        return expMatrix

    def clone_linked_asset_exposures(self, date, assetData, expM, modelDB, marketDB, scoreMap, commonList=None):
        """Clones exposures for cross-listings/DRs etc.
        based on those of the most liquid/largest of each set
        Assets in hardCloneMap are directly cloned from their master asset
        Others are computed as a weighted average of all exposures
        within their group
        """
        self.log.debug('clone_linked_asset_exposures: begin')
        if (commonList is None):
            commonList = []
            if hasattr(self, 'wideCloneList'):
                commonList = self.wideCloneList
        preSPAC = assetData.getSPACs()

        # Pick out exposures to be cloned
        expM_DF = expM.toDataFrame()
        exposureNames = []
        cloneTypes = [ExposureMatrix.StyleFactor, 
                      ExposureMatrix.StatisticalFactor,
                      ExposureMatrix.MacroCoreFactor, 
                      ExposureMatrix.MacroMarketTradedFactor,
                      ExposureMatrix.MacroEquityFactor,
                      ExposureMatrix.MacroSectorFactor]            
        for ftype in cloneTypes:
            exposureNames += expM.getFactorNames(ftype)

        # Exclude any binary exposures for now
        for fname in exposureNames:
            expos = expM_DF.loc[:, fname]
            if len(expos[numpy.isfinite(expos)].unique()) < 3:
                exposureNames.remove(fname)
                self.log.info('%s is binary: excluding from cloning', fname)

        if not hasattr(assetData, 'hardCloneMap'):
            hardCloneMap = assetData.getCloneMap(cloneType='hard')
            subIssueGroups = assetData.getSubIssueGroups()
        else:
            hardCloneMap = assetData.hardCloneMap
            subIssueGroups = assetData.subIssueGroups

        # First deal with any assets to be cloned exactly - note that we clone every single factor exposure
        cloneList = set(hardCloneMap.keys()).intersection(set(assetData.universe))
        for clone in cloneList:
            master = hardCloneMap[clone]
            if master in assetData.universe:
                expM_DF.loc[clone, :] = expM_DF.loc[master, :]

        # Find all good exposures per factor in advance
        goodDict = dict()
        for fname in exposureNames:
            expos = expM_DF.loc[:, fname]
            goodDict[fname] = set(expos[numpy.isfinite(expos)].index)

        # Separate exposures cloned across all assets within a group
        fullExpNames = exposureNames
        commonList = set(commonList).intersection(set(fullExpNames))
        exposureNames = [n for n in exposureNames if n not in commonList]

        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in subIssueGroups.items():
            fctList = []
            subIssueListX = set(subIssueList).difference(preSPAC)
            expos = expM_DF.loc[subIssueListX, fullExpNames]

            # Pull up the asset scores
            score = scoreMap[subIssueListX]
            if len(numpy.unique(score.values)) < 2:
                score[:] = 1.0
            wgtExp = expos.multiply(score, axis=0).sum(axis=0)

            # If cloning some exposures across all linked assets in a group, do that here
            for fct in commonList:
                # Weighted average of all exposures in the set
                nonMissing = goodDict[fct].intersection(subIssueListX)
                if len(nonMissing) > 0:
                    avg = wgtExp[fct] / score[nonMissing].sum(axis=None)
                    expM_DF.loc[subIssueListX, fct] = numpy.full((len(subIssueListX)), avg)

            # Loop round the smaller subgroups
            subGroupDict = assetData.masterClusterDict[groupId]
            for (sgKey, smallSetSubIds) in subGroupDict.items():
                smallSetSubIdsX = set(smallSetSubIds).difference(preSPAC)
                if len(smallSetSubIdsX) > 1:

                    # Get asset scores
                    subScore = scoreMap[smallSetSubIdsX]
                    if len(numpy.unique(subScore.values)) < 2:
                        subScore[:] = 1.0
                    wgtExp = expos.loc[smallSetSubIdsX, :].multiply(subScore, axis=0).sum(axis=0)

                    for fct in exposureNames:
                        # Weighted average of all exposures in the set
                        nonMissing = goodDict[fct].intersection(smallSetSubIdsX)
                        if len(nonMissing) > 0:
                            avg = wgtExp[fct] / subScore[nonMissing].sum(axis=None)
                            expM_DF.loc[smallSetSubIdsX, fct] = numpy.full((len(smallSetSubIdsX)), avg)

        expM.data_ = Utilities.df2ma(expM_DF.T)
        self.log.debug('clone_linked_asset_exposures: end')
        return expM
    
    def clone_linked_asset_descriptors(\
            self, date, assetData, expM, modelDB, marketDB, scoreDict, excludeList=[]):
        """Clones descriptor exposures for cross-listings/DRs etc. based on the most liquid/largest of each set
        Replaces all missing values with that from the "master"
        """
        self.log.debug('clone_linked_asset_descriptors: begin')
        expM_DF = expM.toDataFrame()
        preSPAC = assetData.getSPACs()

        # Pick out exposures to be cloned
        exposureNames = []
        cloneTypes = [ExposureMatrix.StyleFactor,
                      ExposureMatrix.StatisticalFactor,
                      ExposureMatrix.MacroCoreFactor,
                      ExposureMatrix.MacroMarketTradedFactor,
                      ExposureMatrix.MacroEquityFactor,
                      ExposureMatrix.MacroSectorFactor]
        for ftype in cloneTypes:
            exposureNames += expM.getFactorNames(ftype)

        # Exclude any binary exposures for now
        for n in exposureNames:
            expos = expM_DF.loc[:, n]
            if len(expos[numpy.isfinite(expos)].unique()) < 3:
                self.log.info('%s is binary: excluding from cloning', n)
                exposureNames.remove(n)

        # Exclude any specified descriptors
        excludeNames = [n for n in excludeList if n in exposureNames]
        if len(excludeNames) > 0:
            exposureNames = [n for n in exposureNames if n not in excludeNames]
            self.log.info('Excluding %d descriptors: %s from cloning', len(excludeNames), excludeNames)
            
        # Find missing subissues for each factor first
        missDict = dict()
        for fname in exposureNames:
            expos = expM_DF.loc[:, fname]
            missDict[fname] = set(expos[expos.isnull()].index).difference(preSPAC)

        # Loop round sets of linked assets and pull out exposures
        nChanges = 0
        for (groupId, subIssueList) in assetData.getSubIssueGroups().items():
            subIssueList = set(subIssueList)
            for fname in exposureNames:

                # Get exposures for current subgroup and find those with missing values
                missingIds = subIssueList.intersection(missDict[fname])
                if (len(missingIds) < 1) or (len(missingIds) == len(subIssueList)):
                    continue

                # Get asset(s) with largest score
                nonMissingIds = subIssueList.difference(missingIds)
                goodScores = scoreDict[nonMissingIds]
                maxScore = goodScores.max(axis=None)
                if maxScore <= 0.0:
                    continue
                goodScores = abs(goodScores - maxScore)
                maxIds = set(goodScores[numpy.isfinite(goodScores.mask(goodScores>1.0e-12))].index)

                # Fill in missing values
                maxIds = maxIds.difference(missingIds)
                if len(maxIds) < 1:
                    continue
                expM_DF.loc[missingIds, fname] = expM_DF.loc[maxIds, fname].mean(axis=None)
                nChanges += len(missingIds)

        expM.data_ = Utilities.df2ma(expM_DF.T)
        self.log.info('Made %d descriptor changes over %d groups of linked assets', nChanges, len(assetData.getSubIssueGroups()))
        self.log.debug('clone_linked_asset_descriptors: end')
        return

    def group_linked_assets(self, date, assetData, modelDB, marketDB):
        """Sorts linked assets into subgroups within each main issuer group
        """
        self.log.debug('group_linked_assets: begin')
        # Initialise
        assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in assetData.rmgAssetMap.items() for sid in ids])
        masterClusterDict = dict()
        sidToClusterMap = dict()
        mainClusterTypes = self.commonStockTypes + self.otherAllowedStockTypes +\
                    self.fundAssetTypes + self.intlChineseAssetTypes +\
                    self.otherAllowedStockTypes + self.drAssetTypes

        # Some back-compatibility manipulation
        if not hasattr(assetData, 'hardCloneMap'):
            hardCloneMap = assetData.getCloneMap(cloneType='hard')
            forceCointMap = assetData.getCloneMap(cloneType='coint')
            noCointMap = assetData.getCloneMap(cloneType='soft')
            dr2Underlying = assetData.getDr2UnderMap()
            subIssueGroups = assetData.getSubIssueGroups()
            assetTypeDict = assetData.getAssetType()
            assetNameMap = assetData.getNameMap()
        else:
            hardCloneMap = assetData.hardCloneMap
            forceCointMap = assetData.forceCointMap
            noCointMap = assetData.noCointMap
            dr2Underlying = assetData.dr2Underlying
            subIssueGroups = assetData.subIssueGroups
            assetTypeDict = assetData.assetTypeDict
            assetNameMap = assetData.assetNameMap

        # Sift out assest which are clones etc. and put these to one side
        cloneList = set(hardCloneMap.keys())
        forcedCointegrationList = set(forceCointMap.keys())
        noCointegrationList = set(noCointMap.keys())

        # Do the same with DRs mapped to their underlying
        drList = set([sid for sid in dr2Underlying.keys() if \
                        (dr2Underlying[sid] is not None) and \
                        (dr2Underlying[sid] in assetData.universe)])

        # Loop round sets of linked assets
        for (groupId, subIssueList) in subIssueGroups.items():

            # Sift out clones etc. and DRs with underlyings
            subIssueList = set(subIssueList)
            subDRList = list(drList.intersection(subIssueList))
            subDRList = [sid for sid in subDRList if dr2Underlying[sid] in subIssueList]
            subCloneList = list(cloneList.intersection(subIssueList))
            subCloneList = [sid for sid in subCloneList if hardCloneMap[sid] in subIssueList]
            subForceList = list(forcedCointegrationList.intersection(subIssueList))
            subForceList = [sid for sid in subForceList if forceCointMap[sid] in subIssueList]
            subNoCointList = list(noCointegrationList.intersection(subIssueList))
            subNoCointList = [sid for sid in subNoCointList if noCointMap[sid] in subIssueList]

            # Get the subset of assets we will consider
            subIssueList = subIssueList.difference(set(subCloneList))
            subIssueList = subIssueList.difference(set(subForceList))
            subIssueList = subIssueList.difference(set(subNoCointList))
            subIssueList = list(subIssueList.difference(set(subDRList)))
            subGroupDict = defaultdict(list)

            # Loop round remaining assets in subgroup
            for sid in subIssueList:
                assetType = assetTypeDict[sid]
                rmg = assetRMGMap[sid]
                if type(rmg) is ModelDB.RiskModelGroup:
                    rmg = rmg.rmg_id
                if assetType in mainClusterTypes:
                    key = (rmg, 'main')
                    subGroupDict[key].append(sid)
                    sidToClusterMap[sid] = key
                elif assetType in self.preferredStockTypes:
                    key = (rmg, sid)
                    subGroupDict[key].append(sid)
                    sidToClusterMap[sid] = key
                else:
                    key = (rmg, assetType)
                    subGroupDict[key].append(sid)
                    sidToClusterMap[sid] = key
                    
            # Now map DRs to their respective groups
            for sid in subDRList:
                underSid = dr2Underlying[sid]
                if underSid in sidToClusterMap:
                    subGroupDict[sidToClusterMap[underSid]].append(sid)
                    sidToClusterMap[sid] = sidToClusterMap[underSid]

            # Do the same with clones
            for sid in subCloneList:
                masterSid = hardCloneMap[sid]
                if masterSid in sidToClusterMap:
                    subGroupDict[sidToClusterMap[masterSid]].append(sid)
                    sidToClusterMap[sid] = sidToClusterMap[masterSid]

            # And with forced cointegration pairs
            for sid in subForceList:
                siblingSid = forceCointMap[sid]
                if siblingSid in sidToClusterMap:
                    subGroupDict[sidToClusterMap[siblingSid]].append(sid)
                    sidToClusterMap[sid] = sidToClusterMap[siblingSid]

            masterClusterDict[groupId] = subGroupDict

        if self.debuggingReporting:
            outfile = open('tmp/Cointegration-Groups-%s-%s.csv' % (self.name, date), 'w')
            outfile.write('CID,Subissue,Name,Type,Subgroup,\n')
            for groupId in sorted(subIssueGroups.keys()):
                subIssueList = sorted(subIssueGroups[groupId])
                for sid in subIssueList:
                    aType = assetTypeDict[sid]
                    aName = assetNameMap.get(sid, '')
                    if sid in sidToClusterMap:
                        outfile.write('%s,%s,%s,%s,%s,\n' % \
                                (groupId, sid if isinstance(sid, str) else sid.getSubIDString(), aName, aType, sidToClusterMap[sid]))
                    else:
                        outfile.write('%s,%s,%s,%s,None,\n' % \
                                (groupId, sid if isinstance(sid, str) else sid.getSubIDString(), aName, aType))
            outfile.close()

        assetData.masterClusterDict = masterClusterDict
        assetData.sidToClusterMap = sidToClusterMap
        self.log.debug('group_linked_assets: end')
        return

    def loadDescriptors(\
            self, descList, descDict, date, subIssues, modelDB, currencyAssetMap, rollOver=0, forceRegStruct=False):
        """ load descriptor data from the DB for a given list of sub-issues, set of descriptors and date
        """

        # Initialise
        if hasattr(self,'descriptorNumeraire') and self.descriptorNumeraire is not None:
            descriptorNumeraire = self.descriptorNumeraire
        else:
            descriptorNumeraire = self.numeraire.currency_code
        okDescriptorCoverageMap = dict()
        returnDict = pandas.DataFrame(numpy.nan, index=subIssues, columns=descList)
        local_desc_list = set()
        localDescDict = dict()
        if self.regionalDescriptorStructure or forceRegStruct:
            localDescDict = dict(modelDB.getAllDescriptors(local=True))

        for ds in descList:
            if ds not in descDict:
                raise Exception('Undefined descriptor %s!' % ds)

            descID = descDict[ds]
            if (not self.regionalDescriptorStructure) and (not forceRegStruct):
                # Load older SCM descriptor data
                logging.info('Loading %s data, Rollover: %s', ds, rollOver)
                values = modelDB.loadDescriptorData(date, subIssues,
                        self.numeraire.currency_code, descID, rollOverData=rollOver,
                        returnDF=True).loc[:,date]
            else:
                if ds in localDescDict:
                    # Load local descriptors later
                    local_desc_list.add(ds)
                    continue
                else:
                    # Load numeraire-based descriptors
                    logging.info('Loading %s data, Rollover: %s', ds, rollOver)
                    values = modelDB.loadDescriptorData(date, subIssues, descriptorNumeraire,
                            descID, rollOverData=rollOver, tableName='descriptor_numeraire',
                            curr_field=descriptorNumeraire, returnDF=True).loc[:,date]

                    # If we are loading annual descriptors, look to see whether a quarterly version exists
                    # and load that value if possible to overwrite the annual number
                    ds_qtr = None
                    if ('RPF_Annual') in ds:
                        ds_qtr = ds.replace('RPF_Annual', 'RPF_AFQ')
                    elif ('Annual' in ds):
                        ds_qtr = ds.replace('Annual', 'Quarterly')
                    qtr_ID = descDict.get(ds_qtr, None)

                    if self.allowMixedFreqDescriptors and (qtr_ID is not None):
                        # Load quarterly version of descriptor
                        values_qtr = modelDB.loadDescriptorData(date, subIssues, descriptorNumeraire,
                                qtr_ID, rollOverData=rollOver, tableName='descriptor_numeraire',
                                curr_field=descriptorNumeraire, returnDF=True).loc[:,date]

                        # Overwrite annual descriptor with quarterly, where they exist
                        qtrNonMissingIdx = values_qtr[numpy.isfinite(values_qtr)].index
                        if len(qtrNonMissingIdx) > 0:
                            logging.info('Replacing %d %s descriptors with %s', len(qtrNonMissingIdx), ds, ds_qtr)
                            values[qtrNonMissingIdx] = values_qtr[qtrNonMissingIdx]
                    else:
                        # If we are loading quarterly descriptors, check to see whether any annual variants exist
                        # in order to fill-in missing quarterly numbers
                        ds_ann = None
                        if ('RPF_AFQ') in ds:
                            ds_ann = ds.replace('RPF_AFQ', 'RPF_Annual')
                        elif ('Quarterly' in ds):
                            ds_ann = ds.replace('Quarterly', 'Annual')
                        ann_ID = descDict.get(ds_ann, None)

                        if self.allowMixedFreqDescriptors and (ann_ID is not None):
                            # Check for any missing quarterly descriptor values
                            qtrMissingIdx = values[values.isnull()].index
                            if len(qtrMissingIdx) > 0:

                                # Load annual version of descriptor
                                values_ann = modelDB.loadDescriptorData(date, list(qtrMissingIdx), descriptorNumeraire,
                                        ann_ID, rollOverData=rollOver, tableName='descriptor_numeraire',
                                        curr_field=descriptorNumeraire, returnDF=True).loc[:,date]

                                # Overwrite missing quarterly value with non-missing annual value
                                annNonMissingIdx = values_ann[numpy.isfinite(values_ann)].index
                                if len(annNonMissingIdx) > 0:
                                    logging.info('Filling %d missing %s descriptors with %s', len(annNonMissingIdx), ds, ds_ann)
                                    values[annNonMissingIdx] = values_ann[annNonMissingIdx]

            # Checks on coverage
            nonMissingVals = len(values[numpy.isfinite(values)])
            if nonMissingVals / float(len(subIssues)) < 0.05:
                self.log.warning('Descriptor %s has only %3.3f percent unmasked values on date %s',
                        ds, nonMissingVals / float(len(subIssues)) * 100, date.strftime('%Y%m%d'))
                okDescriptorCoverageMap[ds] = 0
            else:
                okDescriptorCoverageMap[ds] = 1
            nUnique = len(values.fillna(0.0).unique())
            if nUnique < 0.01 * len(values):
                logging.warning('Only %d unique values for descriptor %s', nUnique, ds)

            # Add to master dataset
            returnDict.loc[:, ds] = values.copy(deep=True)

        # Load in array of local currency descriptors if necessary
        if len(local_desc_list) > 0:

            # Load in data array of local descriptors for each asset
            desc_id_list = [descDict[d] for d in local_desc_list]
            logging.info('Loading %s data, Rollover: %s', ','.join(local_desc_list), rollOver)
            localDesc = modelDB.loadLocalDescriptorData(
                    date, subIssues, currencyAssetMap, desc_id_list, rollOverData=rollOver, returnDF=True)
            returnDict.loc[subIssues, local_desc_list] = localDesc.rename(columns=dict(zip(desc_id_list, local_desc_list)))

            # Coverage checks per descriptor
            for desc in local_desc_list:
                values = returnDict.loc[:, desc]
                nonMissingVals = len(values[numpy.isfinite(values)])
                if nonMissingVals / float(len(subIssues)) < 0.05:
                    self.log.warning('Descriptor %s has only %3.3f percent unmasked values on date %s',
                            desc, nonMissingVals / float(len(subIssues)) * 100, date.strftime('%Y%m%d'))
                    okDescriptorCoverageMap[desc] = 0
                else:
                    okDescriptorCoverageMap[desc] = 1

        return returnDict, okDescriptorCoverageMap

    def loadDescriptorDataHistory(self, desc, descDict, dateList, subIssues, modelDB, currencyAssetMap=None):
        localDescDict = dict()
        if self.regionalDescriptorStructure:
            localDescDict = dict(modelDB.getAllDescriptors(local=True))
        if desc in descDict:
            descID = descDict[desc]
            if not self.regionalDescriptorStructure:
                values = modelDB.loadDescriptorDataHistory(descID, dateList, subIssues,
                            table='descriptor_exposure', curr_field=self.numeraire.currency_code)

            else:
                if desc in localDescDict:
                    assert(currencyAssetMap is not None)
                    values = modelDB.loadDescriptorDataHistory(descID, 
                        dateList, subIssues, table='descriptor_local_currency', 
                        currencyMap=currencyAssetMap)
                else:
                    values = modelDB.loadDescriptorDataHistory(descID, 
                        dateList, subIssues, table='descriptor_numeraire', 
                        curr_field=self.numeraire.currency_code)
        else:
            raise Exception('Undefined descriptor %s!' % ds)
        return values

    def getAllClassifications(self, modelDB):
        """Returns all non-root classification objects for this risk model.
        Delegates to the industryClassification object.
        """
        if not self.isStatModel():
            return self.industryClassification.getAllClassifications(modelDB)
        else:
            return list()
    
    def getAssetCurrencyChanges(self, sidList, dateList, modelDB, marketDB):
        allCurrencyIDs = set()
        tcChangeAssets = dict()
        tcHistory = modelDB.loadMarketIdentifierHistory(
                sidList, marketDB,
                'asset_dim_trading_currency', 'id', cache=modelDB.tradeCcyCache)
        # Add any assets that have experienced a change in TC
        # to a running dict - ignore the very latest state of being as that has
        # already been dealt with
        for (sid, history) in tcHistory.items():
            assetHistory = set([h for h in history if h.fromDt <= dateList[-1] \
                    and h.thruDt > dateList[0] and h.thruDt <= dateList[-1]])
            changeCurrencies = [h.id for h in assetHistory]
            if len(changeCurrencies) > 0:
                tcChangeAssets[sid] = assetHistory
                allCurrencyIDs.update(set(changeCurrencies))
        return (allCurrencyIDs, tcChangeAssets)

    def getClassificationChildren(self, parentClass, modelDB):
        """Returns a list of the children of the given node in the
        classification.
        Delegates to the industryClassification object.
        """
        if not self.isStatModel():
            return self.industryClassification.getClassificationChildren(parentClass, modelDB)
        else:
            return list()
    
    def getClassificationParents(self, childClass, modelDB):
        """Returns a list of the parents of the given node in the
        classification.
        Delegates to the industryClassification object.
        """
        if not self.isStatModel():
            return self.industryClassification.getAllParents(childClass, modelDB)
        else:
            return list()
    
    def getClassificationMember(self, modelDB):
        """Returns the classification member object for this risk model.
        Delegates to the industryClassification object.
        """
        if not self.isStatModel():
            return self.industryClassification.getClassificationMember(modelDB)
        else:
            return list()
    
    def getClassificationRoots(self, modelDB):
        """Returns the root classification objects for this risk model.
        Delegates to the industryClassification object.
        """
        if not self.isStatModel():
            return self.industryClassification.getClassificationRoots(modelDB)
        else:
            return list()
    
    def getRiskModelInstance(self, date, modelDB):
        """Returns the risk model instance corresponding to the given date.
        The return value is None if no such instance exists.
        """
        return modelDB.getRiskModelInstance(self.rms_id, date)
    
    def insertExposures(self, rmi, data, modelDB, marketDB, update=False, descriptorData=None):
        """Insert the exposure matrix into the database for the given risk model instance.
        The exposure matrix is stored in data as returned by generateExposureMatrix().
        """

        # Initialise
        if hasattr(data, 'exposureMatrix'):
            expMat = data.exposureMatrix
        else:
            expMat = data
        if hasattr(self, 'nurseryCountries') and (len(self.nurseryCountries) > 0):
            factors = self.factors + self.nurseryCountries + self.hiddenCurrencies
        elif hasattr(self, 'hiddenCurrencies') and (len(self.hiddenCurrencies) > 0):
            factors = self.factors + self.hiddenCurrencies
        else:
            factors = self.factors
        preSPAC = AssetProcessor_V4.sort_spac_assets(
                rmi.date, expMat.getAssets(), modelDB, marketDB)
        exSPAC = AssetProcessor_V4.sort_spac_assets(
                rmi.date, expMat.getAssets(), modelDB, marketDB, returnExSpac=True)
        assetType = AssetProcessor_V4.get_asset_info(
                rmi.date, expMat.getAssets(), modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')

        # Check that set of factors match what's in the DB
        subFactors = modelDB.getRiskModelInstanceSubFactors(rmi, factors)
        regions = {r.description: r.region_id for r in modelDB.getAllRiskModelRegions()}
        expMat.data_ = ma.masked_where(expMat.data_==0.0, expMat.data_)
        if len(expMat.getFactorNames()) != len(subFactors):
            namesInDB = [f.name for f in factors]
            namesInExpM = expMat.getFactorNames()
            missingFromDB = [n for n in namesInExpM if n not in namesInDB]
            if len(missingFromDB) > 0:
                self.log.error('Factors in Exp Matrix but not in DB: %s',
                        ','.join(missingFromDB))
            missingFromExpM = [n for n in namesInDB if n not in namesInExpM]
            if len(missingFromExpM) > 0:
                self.log.error('Factors in DB but not in Exp Matrix: %s',
                        ','.join(missingFromExpM)) 
        assert(len(expMat.getFactorNames()) == len(subFactors))

        # Check for missing industry or country exposures that are not SPACs
        indFacNames = expMat.getFactorNames(expMat.IndustryFactor)
        missingExSPAC = []
        if len(indFacNames) > 0:
            if oldPD:
                indData = expMat.toDataFrame().loc[:, indFacNames].sum(axis=1)
            else:
                indData = expMat.toDataFrame().loc[:, indFacNames].sum(axis=1, min_count=1)
            missingInd = set(indData[indData.isnull()].index).difference(preSPAC)
            if len(missingInd) > 0:
                for sid in missingInd:
                    if sid in exSPAC:
                        logging.warn('Post-announcement SPAC %s has missing Industry exposure: %s', sid.getSubIDString(), assetType[sid])
                        if self.isProjectionModel() or self.isLinkedModel():
                            missingExSPAC.append(sid)
                    else:
                        logging.error('Asset %s has missing Industry exposure: %s', sid.getSubIDString(), assetType[sid])
        cntFacNames = expMat.getFactorNames(expMat.CountryFactor)
        if len(cntFacNames) > 0:
            if oldPD:
                cntData = expMat.toDataFrame().loc[:, cntFacNames].sum(axis=1)
            else:
                cntData = expMat.toDataFrame().loc[:, cntFacNames].sum(axis=1, min_count=1)
            missingCnt = set(cntData[cntData.isnull()].index).difference(preSPAC)
            if len(missingCnt) > 0:
                for sid in missingCnt:
                    if sid in exSPAC:
                        logging.warn('Post-announcement SPAC %s has missing Country exposure: %s', sid.getSubIDString(), assetType[sid])
                        if self.isProjectionModel() or self.isLinkedModel():
                            missingExSPAC.append(sid)
                    else:
                        logging.error('Asset %s has missing Country exposure: %s, %s', sid.getSubIDString(), assetType[sid])

        # Write the expM to the DB
        subFactorMap = dict(zip([f.name for f in factors], subFactors))
        missingExSPAC = set(missingExSPAC)
        if len(missingExSPAC) > 0:
            logging.warn('Reassigning %d post-announcement SPACs to pre-announcement status as they\'re missing factors', len(missingExSPAC))
            preSPAC = preSPAC.union(missingExSPAC)
        modelDB.insertFactorExposureMatrixNew(rmi, expMat, subFactorMap, update, setNull=preSPAC)

        # Save standardization stats if any
        if hasattr(expMat, 'meanDict'):
            sfList = []
            regionList = []
            meanList = []
            stdevList = []
            for bucket in expMat.meanDict.keys():
                nameList = list(expMat.meanDict[bucket].keys())
                assert(set(nameList).issubset(list(subFactorMap.keys())))
                sfList.extend([subFactorMap[nm] for nm in nameList])
                meanList.extend([expMat.meanDict[bucket][nm] for nm in nameList])
                stdevList.extend([expMat.stdDict[bucket][nm] for nm in nameList])
                if bucket in regions:
                    regionList.extend([regions[bucket]]*len(nameList))
                else:
                    regionList.extend([None]*len(nameList))

            modelDB.insertStandardisationExp(
                    rmi.rms_id, rmi.date, sfList, regionList, meanList, stdevList)

        # Write the descriptor standardisation stats
        if descriptorData is not None:
            descList = []
            regionList = []
            meanList = []
            stdevList = []
            for bucket in descriptorData.meanDict.keys():
                nameList = list(descriptorData.meanDict[bucket].keys())
                assert(set(nameList).issubset(list(descriptorData.descDict.keys())))
                descList.extend([descriptorData.descDict[nm] for nm in nameList])
                meanList.extend([descriptorData.meanDict[bucket][nm] for nm in nameList])
                stdevList.extend([descriptorData.stdDict[bucket][nm] for nm in nameList])
                if bucket in regions:
                    regionList.extend([regions[bucket]]*len(nameList))
                else:
                    regionList.extend([None]*len(nameList))

            modelDB.insertStandardisationDesc(
                    rmi.rms_id, rmi.date, descList, regionList, meanList, stdevList)

        # Report any changes in factor structure
        currFactors = set([f.name for f in factors])
        dateList = modelDB.getDates(self.rmg, rmi.date, 1, excludeWeekend=True)
        if len(dateList)==2:
            prevDate = dateList[0]
            prmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
            if prmi is not None:
                self.setFactorsForDate(prevDate, modelDB)
                prevFactors = set([f.name for f in self.factors])
                joiners = currFactors.difference(prevFactors)
                leavers = prevFactors.difference(currFactors)
                if len(joiners) > 0:
                    self.log.info('%d factors joining model: %s', len(joiners), ', '.join(joiners))
                if len(leavers) > 0:
                    self.log.info('%d factors leaving model: %s', len(leavers), ', '.join(leavers))
                self.setFactorsForDate(rmi.date, modelDB)
    
    def insertFactorCovariances(self, rmi_id, factorCov, subFactors, modelDB):
        """Inserts the factor covariances into the database for the given
        risk model instance.
        factorCov is a factor-factor array of the covariances.
        subFactors is a list of sub-factor IDs.
        """
        modelDB.insertFactorCovariances(rmi_id, subFactors, factorCov)
    
    def insertFactorReturns(self, date, factorReturns, modelDB, extraFactors=[], flag=None, addedTags=None):
        """Inserts the internal factor returns into the database for the given date.
        factorReturns is an array of the return values.
        """
        if len(extraFactors) > 0:
            factors = self.factors + extraFactors
        else:
            factors = self.factors
        subFactors = modelDB.getSubFactorsForDate(date, factors)
        assert(len(subFactors) == len(factors))
        modelDB.insertFactorReturns(self.rms_id, date, subFactors, factorReturns, flag=flag, addedTags=addedTags)

    def insertStatFactorReturns(self, date, exp_date, factorReturns, modelDB):
        """Inserts the factor returns into the database for the given date.
        factorReturns is an array of the return values.
        """
        subFactors = modelDB.getSubFactorsForDate(exp_date, self.factors)
        assert(len(subFactors) == len(self.factors))
        modelDB.insertStatFactorReturns(self.rms_id, date, exp_date, subFactors, factorReturns)

    def insertRegressionStatistics(self, date, regressStats, factorNames,
            adjRsquared, pcttrade, modelDB, VIF=None, extraFactors=[], pctgVar=None, flag=None, addedTags=None):
        """Inserts the internal regression statistics into the database for the
        given date.
        regressStats is a two-dimensional  array of the per-factor
        statistics. See generateFactorSpecificReturns
        factorNames is an array containing the corresponding factor names.
        """
        if len(extraFactors) > 0:
            factors = self.factors + extraFactors
        else:
            factors = self.factors
        subFactors = modelDB.getSubFactorsForDate(date, factors)
        assert(len(subFactors) == len(factors))
        modelDB.insertRMSFactorStatistics(
                self.rms_id, date, subFactors, regressStats[:,0],
                regressStats[:,1], regressStats[:,2], regressStats[:,3], VIF, flag=flag, addedTags=None)
        modelDB.insertRMSStatistics(self.rms_id, date, adjRsquared, pcttrade=pcttrade, pctgVar=pctgVar, flag=flag)

    def insertSpecificReturns(self, date, specificReturns, subIssues,
                              modelDB, addedTags=None, estuWeights=None, internal=False):
        """Inserts the specific returns into the database for the given date.
        specificReturns is a masked array of the return values.
        subIssues is an array containing the corresponding sub-issue IDs.
        """
        assert(len(specificReturns.shape) == 1)
        assert(specificReturns.shape[0] == len(subIssues))
        indices = numpy.flatnonzero(ma.getmaskarray(specificReturns) == 0)
        subIssues = numpy.take(numpy.array(
            subIssues, dtype=object), indices, axis=0)
        specificReturns = ma.take(specificReturns, indices, axis=0)
        modelDB.insertSpecificReturns(self.rms_id, date, subIssues,
                                      specificReturns, addedTags=addedTags, estuWeights=estuWeights, internal=internal)
    
    def insertSpecificRisks(self, rmi_id, specificVariance, subIssues, specificCovariance, modelDB):
        """Inserts the specific risk into the database for the given date.
        specificVariance is a masked array of the specific variances.
        subIssues is an array containing the corresponding SubIssues.
        specificCovariance is a dictionary of dictionaries, mapping
        SubIssues to mappings of 'linked' SubIssue(s) and their covariance.
        """
        # Preprocess data
        subIssues = sorted(specificVariance.index)
        specificVariance = Utilities.df2ma(specificVariance[subIssues])
        assert(len(specificVariance.shape) == 1)
        assert(specificVariance.shape[0] == len(subIssues))

        # Write to DB
        indices = numpy.flatnonzero(ma.getmaskarray(specificVariance) == 0)
        subIssues = numpy.take(numpy.array(subIssues, dtype=object), indices, axis=0)
        specificRisks = ma.sqrt(ma.take(specificVariance, indices, axis=0))
        modelDB.insertSpecificCovariances(rmi_id, specificCovariance)
        modelDB.insertSpecificRisks(rmi_id, subIssues, specificRisks)
        return
    
    def loadCumulativeFactorReturns(self, date, modelDB):
        """Loads the cumulative factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        cumReturns = modelDB.loadCumulativeFactorReturnsHistory(
            self.rms_id, subFactors, [date]).data[:,0]
        return (cumReturns, self.factors)
    
    def insertEstimationUniverse(self, rmi, modelDB):
        """Inserts the estimation universes into the database for the given
        risk model instance.
        estuMap is a dict of estu objects, mapped by name
        """
        for estuName in self.estuMap.keys():
            if hasattr(self.estuMap[estuName], 'assets'):
                modelDB.insertEstimationUniverseV3(rmi, self.estuMap[estuName], estuName)

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs, modelDB, estuName='main'):
        """Inserts model estimation (regression) weights for estimation
        universe assets in the given risk model instance.
        Note that the list of SubIssues may be smaller than the full set
        of estimation universe assets due to treatment of non-trading
        markets, removal of missing returns assets, and additional
        'pre-processing' logic employed prior to the regression.
        """
        modelDB.insertEstimationUniverseWeightsV3(rmi, self.estuMap[estuName], estuName, subidWeightPairs)

    def loadEstimationUniverse(self, rmi_id, modelDB, data=None):
        """Loads the estimation universe(s) of the given risk model instance.
        Returns a list of sub-issue IDs.
        """
        if self.estuMap is None:
            self.estuMap = dict()
            self.estuMap['main'] = Utilities.Struct()
            self.estuMap['main'].assets = modelDB.getRiskModelInstanceESTU(rmi_id, estu_name='main')
            self.estuMap['main'].name = 'main'
        for name in self.estuMap.keys():
            if hasattr(self.estuMap[name], 'id'):
                idx = self.estuMap[name].id
                self.estuMap[name].assets = modelDB.getRiskModelInstanceESTU(rmi_id, estu_idx=idx, estu_name=name)
        
        if len(self.estuMap['main'].assets) > 0:
            estu = self.estuMap['main'].assets
        else:
            self.log.warning('Main estimation universe empty')
            estu = []

        # Ensure we have total coverage
        if data is not None:
            assetIdxMap = dict(zip(data.universe, range(len(data.universe))))
            for nm in self.estuMap.keys():
                self.estuMap[nm].assets = [sid for sid in self.estuMap[nm].assets if sid in assetIdxMap]
                self.estuMap[nm].assetIdx = [assetIdxMap[sid] for sid in self.estuMap[nm].assets]

            estu = sorted([sid for sid in estu if sid in assetIdxMap])
            data.estimationUniverse = list(estu)

        assert(len(estu) > 0)
        return estu

    def loadExposureMatrixDF(self, rmi_id, modelDB, skipAssets=False, addExtraCountries=False, assetList=None):
        expM = self.loadExposureMatrix(\
                rmi_id, modelDB, skipAssets=skipAssets, addExtraCountries=addExtraCountries, assetList=assetList)
        return expM.toDataFrame()

    def loadExposureMatrix(self, rmi_id, modelDB, skipAssets=False, addExtraCountries=False, assetList=None):
        """Loads the exposure matrix of the given risk model instance.
        Returns an ExposureMatrix object.
        To skip asset population, set skipAssets to True
        """
        if addExtraCountries and hasattr(self, 'nurseryCountries') and (len(self.nurseryCountries) > 0):
            factors = self.factors + self.nurseryCountries
        else:
            factors = self.factors
        subFactors = modelDB.getRiskModelInstanceSubFactors(rmi_id, factors)
        subFactorMap = dict([(s.factor.name, s) for s in subFactors])
        if skipAssets:
            assets = []
        elif (assetList is None) or (len(assetList) < 1):
            assets = modelDB.getRiskModelInstanceUniverse(rmi_id, returnExtra=addExtraCountries)
        else:
            assets = assetList

        # Set up an empty exposure matrix
        factorList = list()
        if not self.isStatModel():
            if self.intercept is not None:
                factorList.append((self.intercept.name, ExposureMatrix.InterceptFactor))
            if len(self.styles) > 0:
                # Local factor hack
                realStyles = [f for f in self.styles if f not in self.localStructureFactors]
                factorList.extend([(f.name, ExposureMatrix.StyleFactor) for f in realStyles])
                factorList.extend([(f.name, ExposureMatrix.LocalFactor) for f in self.localStructureFactors])
            if len(self.industries) > 0:
                factorList.extend([(f.name, ExposureMatrix.IndustryFactor) for f in self.industries])
            if len(self.regionalIntercepts) > 0:
                factorList.extend([(f.name, ExposureMatrix.RegionalIntercept) for f in self.regionalIntercepts])
            if len(self.rmg) > 1:
                if len(self.countries) > 0:
                    factorList.extend([(f.name, ExposureMatrix.CountryFactor) for f in self.countries])
                if addExtraCountries and hasattr(self, 'nurseryCountries') and (len(self.nurseryCountries) > 0):
                    factorList.extend([(f.name, ExposureMatrix.CountryFactor) for f in self.nurseryCountries])
                if len(self.currencies) > 0:
                    factorList.extend([(f.name, ExposureMatrix.CurrencyFactor) for f in self.currencies])
            if len(self.macro_core)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroCoreFactor) for f in self.macro_core])
            if len(self.macro_market_traded)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroMarketTradedFactor) for f in self.macro_market_traded])
            if len(self.macro_equity)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroEquityFactor) for f in self.macro_equity])
            if len(self.macro_sectors)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroSectorFactor) for f in self.macro_sectors])
            if len(self.macros)>0:
                factorList.extend([(f.name, ExposureMatrix.MacroFactor) for f in self.macros])

        if len(self.blind) > 0:
            factorList.extend([(f.name, ExposureMatrix.StatisticalFactor)
                               for f in self.blind])
            if self.isStatModel() and (len(self.currencies) > 0):
                factorList.extend([(f.name, ExposureMatrix.CurrencyFactor) for f in self.currencies])
        # Force factorList to have the same order as self.factors
        # and hence, the list of subFactors
        expMFactorNames = [f[0] for f in factorList]
        expMFactorMap = dict(zip(expMFactorNames, factorList))
        factorList = [expMFactorMap[f.name] for f in factors]

        expM = Matrices.ExposureMatrix(assets, factorList)
        if (not skipAssets) and (len(assets) > 0):
            # Now fill it in from the DB
            modelDB.getFactorExposureMatrixNew(rmi_id, expM, subFactorMap)
        return expM
    
    def loadFactorCovarianceMatrix(self, rmi, modelDB, returnDF=False):
        """Loads the factor-factor covariance matrix of the given risk
        model instance.
        Returns a (cov, factors) pair where cov is an m by m array
        containing the factor covariances and factors is a list of the
        m factor names.
        """
        subFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.factors)
        cov = modelDB.getFactorCovariances(rmi, subFactors)
        if returnDF:
            factorNames = [f.name for f in self.factors]
            return pandas.DataFrame(cov, index=factorNames, columns=factorNames)
        return (cov, self.factors)
    
    def loadFactorReturns(self, date, modelDB, addNursery=False, flag=None, returnDF=False):
        """Loads the factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        if addNursery:
            factors = self.factors + self.nurseryCountries
        else:
            factors = self.factors
        factorReturns = self.loadFactorReturnsHistory([date], modelDB, table_suffix=flag, factorList=factors)
        if returnDF:
            return pandas.Series(factorReturns.data[:,0], index=[f.name for f in factors])
        return (factorReturns.data[:,0], factors)
    
    def loadFactorReturnsHistory(\
            self, dateList, modelDB, table_suffix=None, screen_data=False, factorList=None, returnDF=False):
        """Loads the factor returns history between the given dates
           Only loads factor returns for all the subfactors alive the end date
        """
        # Get list of sub-factors
        if factorList is None:
            subFactors = modelDB.getSubFactorsForDate(max(dateList), self.factors)
        else:
            subFactors = modelDB.getSubFactorsForDate(max(dateList), factorList)

        # Load the data
        fr = modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, dateList, flag=table_suffix, screen=screen_data)
        if returnDF:
            return pandas.DataFrame(fr.data, index=[f.factor.name for f in subFactors], columns=dateList)
        return fr

    def loadRegressionStatistics(self, date, modelDB, addNursery=False, flag=None):
        """Loads the regression statistics from the database for the
        given date.
        The return value is (regressStats, factorName, adjRsquared).
        regressStats is a two-dimensional  array of the per-factor
        statistics. Each row contains std. error, t value and Pr(>|t|)
        in that order.
        factorNames is an array containing the corresponding factor names.
        adjRsquared is the adjusted R-squared value for the regression.
        """
        if addNursery:
            factors = self.factors + self.nurseryCountries
        else:
            factors = self.factors
        subFactors = modelDB.getSubFactorsForDate(date, factors)
        regressStats = modelDB.getRMSFactorStatistics(self.rms_id, date, subFactors, flag=flag)
        adjRsquared = modelDB.getRMSStatistics(self.rms_id, date, flag=flag)
        return (regressStats, factors, adjRsquared)
    
    def loadSpecificReturnsHistory(self, date, subissues, dates, modelDB, marketDB, internal=False):
        """ Loads history of specific returns for the given assets
        Returns a dataframe of sub-issues by dates
        """
        specificReturns = modelDB.loadSpecificReturnsHistory(\
                self.rms_id, subissues, dates, internal=internal).toDataFrame()
        # Mask pre-IPO returns as a precaution
        fromDates = Utilities.load_ipo_dates(\
                date, subissues, modelDB, marketDB, exSpacAdjust=True, returnList=True)
        return Matrices.maskByDate(specificReturns, fromDates)

    def loadSpecificRisks(self, rmi_id, modelDB):
        """Loads the specific risks of the given risk model instance.
        Returns a tuple containing a dictionary mapping SubIssues to 
        their specific risks and a dictionary of dictionaries, mapping
        SubIssues to mappings of 'linked' SubIssues to their specific
        covariances.
        """
        return (modelDB.getSpecificRisks(rmi_id), modelDB.getSpecificCovariances(rmi_id))
    
    def RMGDateLogic(self, r, date):
        """Logic for determining where in the timeline
        each RMG is
        """

        # Set defaults
        r.rmg.downWeight = 1.0

        # Fade-out date
        if r.fade_dt <= date and (abs(r.full_dt-date) > (date-r.fade_dt)):
            fadePeriod = (r.thru_dt - r.fade_dt).days
            # Exponential downweighting function
            expWeights = Utilities.computeExponentialWeights(
                    60, fadePeriod, normalize=False)
            r.rmg.downWeight = expWeights[(date - r.fade_dt).days]

        # Fade-in date
        if r.full_dt > date and ((r.full_dt-date) <= abs(date-r.fade_dt)):
            fadePeriod = (r.full_dt - r.from_dt).days
            halfLife = 30
            expWeights = Utilities.computeExponentialWeights(
                    halfLife, fadePeriod, normalize=False)
            iLoc = min((r.full_dt - date).days, int(16*halfLife))
            iLoc = min(iLoc, len(expWeights)-1)
            r.rmg.downWeight = expWeights[iLoc]

        # Hard-code downweight for Saudi Arabia
        if r.rmg.mnemonic in ['SA']:
            if self.legacySAWeight:
                wt0 = numpy.array([1.0])
                wt1 = numpy.array([0.2])
                start_date = Utilities.parseISODate('2020-04-03')
                end_date = Utilities.parseISODate('2020-04-10')
                r.rmg.downWeight = Utilities.blend_values(wt0, wt1, date, start_date, end_date)[0]
            else:
                r.rmg.downWeight = 0.2

        # Downweight data from markets in which we have no confidence
        mnem = r.rmg.mnemonic
        if mnem in self.naughtyList:
            startDate = Utilities.parseISODate(self.naughtyList[mnem][0])
            fullDate = Utilities.parseISODate(self.naughtyList[mnem][1])
            endDate = Utilities.parseISODate(self.naughtyList[mnem][2])
            if date >= startDate:
                r.rmg.mktCapCap = self.naughtyList[mnem][3]
                r.rmg.blendFactor = 1.0
                if date < fullDate:
                    r.rmg.blendFactor = float((date - startDate).days) / float((fullDate - startDate).days)
                logging.info('Restricting %s market cap to %.4f%% of total', mnem, 100.0*r.rmg.mktCapCap)
                logging.info('Blend factor for %s: %s', date, r.rmg.blendFactor)

        # Report on weighting
        if r.rmg.downWeight < 1.0:
            self.log.info('%s (RiskModelGroup %d, %s) down-weighted to %.3f%%',
                    r.rmg.description, r.rmg.rmg_id, r.rmg.mnemonic,
                    r.rmg.downWeight * 100.0)
        return

    def setRiskModelGroupsForDate(self, date):
        """Determines the risk model groups belonging to the 
        model for the given date, and their current currency,
        market status, etc.  Sets class attributes accordingly.
        """
        self.rmg = []
        for r in self.rmgTimeLine:
            if r.from_dt <= date and r.thru_dt > date:
                self.RMGDateLogic(r, date)
                self.rmg.append(r.rmg)
        for rmg in self.rmg:
            if not rmg.setRMGInfoForDate(date):
                raise Exception('Cannot determine details for %s risk model group (%d) on %s' % (rmg.description, rmg.rmg_id, str(date)))
                return False
        if len(self.rmg) > 1:
            self.currencyModel.setRiskModelGroupsForDate(date)
        logging.debug('%d rmg groups for the model', len(self.rmg))
        return True

    def reportCorrelationMatrixChanges(self, date, assetData, expM, factorCov, rmi, prmi, modelDB):
        """Compare the correlation matrices from the specified date with
        the one from the given RiskModelInstance.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        logging.info('Loading previous risk model info')
        prevFactorCov = modelDB.getFactorCovariances(prmi, subFactors, returnDF=True)
        diff = abs(factorCov - prevFactorCov).values.sum(axis=None)
        self.log.info('Day on day difference in covariance matrices: %.4f', diff)

        # Compute estimation universe factor risk
        estu = set(self.loadEstimationUniverse(rmi, modelDB, assetData)).intersection(set(assetData.universe))
        estu_exp = expM.toDataFrame().loc[estu, factorCov.index].fillna(0.0)
        estu_wgts = assetData.marketCaps[estu].fillna(0.0)
        estu_exp = estu_wgts.dot(estu_exp) / estu_wgts.sum(axis=None)
        estuFactorRisk = numpy.sqrt(estu_exp.dot(factorCov.dot(estu_exp.T)))
        self.log.info('Estimation universe factor risk: %.8f%%', 100.0*estuFactorRisk)
        self.log.info('Sum of composite cov matrix elements: %f', ma.sum(factorCov.values, axis=None))

        # Report on factor risk
        sortedFN = sorted(factorCov.index)
        tmpFactorCov = factorCov.reindex(index=sortedFN, columns=sortedFN)
        var = numpy.diag(tmpFactorCov.values)[:,numpy.newaxis]
        sqrtvar = 100.0 * numpy.sqrt(var)
        self.log.info('Factor risk: (Min, Mean, Max): (%.2f%%, %.2f%%, %.2f%%)',
                ma.min(sqrtvar, axis=None), numpy.average(sqrtvar, axis=None), ma.max(sqrtvar, axis=None))

        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            (d, corrMatrix) = Utilities.cov2corr(tmpFactorCov.values, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=sortedFN, rowNames=sortedFN, dp=8)

            # Write variances to flatfile
            varoutfile = 'tmp/%s-varFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(var, varoutfile, rowNames=sortedFN, dp=8)

            # Write final covariance matrix to flatfile
            covOutFile = 'tmp/%s-covFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(tmpFactorCov, covOutFile, columnNames=sortedFN, rowNames=sortedFN, dp=8)

        return estuFactorRisk
    
    def reportISCStability(self, date, varDict, currISCDict, prmi, modelDB):
        """Compares the ISC matrix from one day to the next
        """
        if (prmi is None) or (not prmi.has_risks):
            return
        logging.info('Loading previous ISC info')
        prevISCDict = modelDB.getSpecificCovariances(prmi)
        prevRskDict = modelDB.getSpecificRisks(prmi)
        corrList = []
        varList = []
        for (sid1, iscMap) in currISCDict.items():
            if sid1 in prevISCDict:
                if prevRskDict[sid1] != 0.0:
                    varChange = (numpy.sqrt(varDict[sid1]) / prevRskDict[sid1]) - 1.0
                    varList.append(abs(varChange))
                for (sid2, isc) in iscMap.items():
                    if sid2 in prevISCDict[sid1]:
                        mult = varDict[sid1] * varDict[sid2] * prevRskDict[sid1] * prevRskDict[sid2]
                        if mult != 0.0:
                            currCorrel = currISCDict[sid1][sid2] / numpy.sqrt(varDict[sid1]*varDict[sid2])
                            prevCorrel = prevISCDict[sid1][sid2] / (prevRskDict[sid1] * prevRskDict[sid2])
                            if prevCorrel != 0.0:
                                corrChange = (currCorrel / prevCorrel) - 1.0
                                corrList.append(abs(corrChange))

        # Output ISC stats
        if len(varList) > 0:
            logging.info('%s Day on day %% changes in ISC risk (min, mean, max): (%.2f, %.2f, %.2f)',
                    date, 100.0*min(varList), 100.0*ma.mean(varList), 100.0*max(varList))
        if len(corrList) > 0:
            logging.info('%s Day on day %% changes in ISC correlation (min, mean, max): (%.2f, %.2f, %.2f)',
                    date, 100.0*min(corrList), 100.0*ma.mean(corrList), 100.0*max(corrList))
        return

    def usesReturnsTimingAdjustment(self):
        return (self.returnsTimingId is not None)

    def validateFactorStructure(self, date=None, warnOnly=False):
        """Check that ModelFactors associated with FactorRiskModel
        are consistent with factors stored in ModelDB.
        """
        dbFactors = set(self.descFactorMap.values())
        dbFactors = [f for f in dbFactors if f.isLive(date)]
        classFactors = set(self.factors)

        notInModel = [f.name for f in dbFactors if f not in classFactors]
        notInDatabase =  [f.name for f in classFactors if f not in dbFactors]
        if len(notInModel) > 0:
            notInModel.sort()
            self.log.warning('Factors in rms_factor table but not in %s model parameters:', self.name)
            self.log.warning('%s', '|'.join(notInModel))
        if len(notInDatabase) > 0:
            notInDatabase.sort()
            self.log.warning('Factors in %s model parameters but not in rms_factor table:', self.name)
            self.log.warning('%s', '|'.join(notInDatabase))
        if (len(notInDatabase) > 0) or (len(notInModel) > 0 and not warnOnly):
            raise LookupError('Mismatch between factors in model and database on date %s' % str(date))

    def setDateList(self, date, modelDB, rmi=None):
        """Sets up the date list for covariance matrix generation
        """

       # Check that necesary model data exists
        if rmi is None:
            rmi, dummy = self.set_model_instance(date, modelDB)
        if not rmi.has_exposures or not rmi.has_returns:
            raise LookupError(
                'Exposures or returns missing in %s risk model instance for %s' % (self.name, str(date)))

        # Define min and max dates for factor covariances
        (self.minFCovObs, self.maxFCovObs) = \
                (max(self.fvParameters.minObs, self.fcParameters.minObs),
                 max(self.fvParameters.maxObs, self.fcParameters.maxObs))
        if self.fvParameters.selectiveDeMean:
            self.maxFCovObs = max(self.maxFCovObs, self.fvParameters.deMeanMaxHistoryLength)
        self.maxSRiskObs = self.srParameters.maxObs
        self.minSRiskObs = self.srParameters.minObs

        # Create date lists
        rmgList = [r for r in self.rmg if r not in self.nurseryRMGs]
        omegaDateList = modelDB.getDates(rmgList, date, self.maxFCovObs-1, excludeWeekend=True)
        omegaDateList.reverse()
        deltaDateList = modelDB.getDates(rmgList, date, self.maxSRiskObs-1, excludeWeekend=True)
        deltaDateList.reverse()

        # Set dateList to maximum that we would like - set to reverse chronological order
        if len(omegaDateList) > len(deltaDateList):
            return omegaDateList
        return deltaDateList

    def parseModelHistories(self, srm, date, dateList, modelDB, rmi=None):
        """Does the messy checking and processing of dates for model risk steps
        """
        # Check that necesary model data exists
        if rmi is None:
            rmi, rmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
            if not rmi.has_exposures or not rmi.has_returns:
                if not self.isProjectionModel: # projection model's risk step is embeded in exposure, return, and risk steps.
                    raise LookupError(
                        'Exposures or returns missing in %s risk model instance for %s' % (srm.name, str(date)))
                else:
                    if not rmi.has_returns:
                        raise LookupError(
                            'Exposures or returns missing in %s risk model instance for %s' % (srm.name, str(date)))
            # Get subset of dates valid for particular model
            srmDateList = modelDB.getDateRange(rmgList, min(dateList), max(dateList), excludeWeekend=True)
            srmDateList.reverse()
        else:
            srmDateList = dateList

        # Check that enough consecutive days have returns, try to get maximum number of observations
        rmiList = modelDB.getRiskModelInstances(srm.rms_id, srmDateList)
        okDays = [i.date == j and i.has_returns for (i,j) in zip(rmiList, srmDateList)]
        okDays.append(False)
        firstBadDay = okDays.index(False)

        badDt = None
        if len(srmDateList) > firstBadDay:
            badDt = srmDateList[firstBadDay]
        srmDateList = srmDateList[:firstBadDay]

        # Report on any problems with date histories
        if len(srmDateList) < max(self.minFCovObs, self.minSRiskObs):
            required = max(self.minFCovObs, self.minSRiskObs)
            self.log.warning('%s model returns missing on %d days, beginning %s',
                        srm.name, required - len(srmDateList), badDt)
            raise LookupError(
                '%d incomplete risk model instances for required days' % (required - len(srmDateList)))
        if len(srmDateList) < self.maxFCovObs:
            self.log.info('Using only %d of %d days of factor return history', len(srmDateList), self.maxFCovObs)
        if len(srmDateList) < self.maxSRiskObs:
            self.log.info('Using only %d of %d days of specific return history', len(srmDateList), self.maxSRiskObs)

        return badDt

    def generateFactorSpecificRisk(self, date, modelDB, marketDB, dvaToggle=False, nwToggle=False):
        """Compute the factor-factor covariance matrix and the specific variances for the risk model on the given date.
        Specific risk is computed for all assets in the exposure universe.
        It is assumed that the risk model instance for this day already exists.
        The return value is a structure with four fields:
         - subIssues: a list of sub-issue IDs
         - specificVars: an array of the specific variances corresponding to subIssues
         - subFactors: a list of sub-factor IDs
         - factorCov: a two-dimensional array with the factor-factor covariances corresponding to subFactors
        """
        # Toggle on/off DVA or Newey-West settings
        if dvaToggle:
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)
            varDVA_save = self.covarianceCalculator.varParameters.DVAWindow
            corDVA_save = self.covarianceCalculator.corrParameters.DVAWindow
            self.covarianceCalculator.varParameters.DVAWindow = None
            self.covarianceCalculator.corrParameters.DVAWindow = None
        elif nwToggle:
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)
            varNW_save = self.covarianceCalculator.varParameters.NWLag
            corNW_save = self.covarianceCalculator.corrParameters.NWLag
            self.covarianceCalculator.varParameters.NWLag = 0
            self.covarianceCalculator.corrParameters.NWLag = 0

        # Get model instance
        rmi, rmgList = self.set_model_instance(date, modelDB)

        # Process dates
        dateList = self.setDateList(date, modelDB, rmi=rmi)
        firstBadDt = self.parseModelHistories(self, date, dateList, modelDB, rmi=rmi)

        # Drop dates if there is missing model data
        if firstBadDt is not None:
            dlen = len(dateList)
            dateList = [dt for dt in dateList if dt>firstBadDt]
            logging.warn('Dropping dates from history because of missing data in %s model', self.name)
            logging.warn('Bad date: %s, history shrunk from %d to %d dates', firstBadDt, dlen, len(dateList))

        # Remove dates for which many markets are non-trading
        minMarkets = 0.5
        propTradeDict = modelDB.getRMSStatisticsHistory(self.rms_id, flag='internal', fieldName='pcttrade')
        badDates = [dt for dt in propTradeDict.keys() if \
                (propTradeDict.get(dt, 0.0) is not None) and (propTradeDict.get(dt, 0.0) < minMarkets)]
        badDates = sorted(dt for dt in badDates if (dt>=min(dateList) and dt<=max(dateList)))
        self.log.info('Removing %d dates with < %.2f%% markets trading: %s',
                    len(badDates), minMarkets, ','.join([str(d) for d in badDates]))
        dateList = [dt for dt in dateList if dt not in badDates]
        dateList.sort(reverse=True)

        # Initialise
        nonCurrencyFactors = [f for f in self.factors if f not in self.currencies]
        frData = Utilities.Struct()
        crData = Utilities.Struct()
        
        # Load up non-currency factor returns
        if self.twoRegressionStructure:
            nonCurrencyFactorReturns = self.loadFactorReturnsHistory(
                    dateList[:self.maxFCovObs], modelDB, table_suffix='internal',
                    screen_data=True, factorList=nonCurrencyFactors, returnDF=True)
        else:
            nonCurrencyFactorReturns = self.loadFactorReturnsHistory(
                    dateList[:self.maxFCovObs], modelDB, screen_data=True,
                    factorList=nonCurrencyFactors, returnDF=True)
        frData.data = nonCurrencyFactorReturns

        # Load exposure matrix
        expM = self.loadExposureMatrix(rmi, modelDB, addExtraCountries=False)

        # Code for selective demeaning of certain factors or factor types
        if self.fvParameters.selectiveDeMean:
            minHistoryLength = self.fvParameters.deMeanMinHistoryLength
            maxHistoryLength = self.fvParameters.deMeanMaxHistoryLength
            dmFactorNames = []

            # Sift out factors to be de-meaned
            for fType in self.fvParameters.deMeanFactorTypes:
                if fType in expM.factorTypes_:
                    dmFactorNames.extend(expM.getFactorNames(fType))
                else:
                    dmFactorNames.append(fType)
            means = pandas.Series(0.0, index=nonCurrencyFactorReturns.index)
            frets = nonCurrencyFactorReturns.loc[dmFactorNames, :].fillna(0.0)

            # Cut to required length, or pad with zeros if we fall short
            if len(nonCurrencyFactorReturns.columns) < minHistoryLength:
                extraLen = int(minHistoryLength - len(nonCurrencyFactorReturns.columns))
                frets = pandas.concat([frets, pandas.DataFrame(0.0, index=dmFactorNames, columns=range(extraLen))], axis=1)
            elif len(frets.columns) > maxHistoryLength:
                frets = frets.iloc[:, range(maxHistoryLength)]

            # Compute the weighted mean
            weights = Utilities.computeExponentialWeights(
                    self.fvParameters.deMeanHalfLife, len(frets.columns), equalWeightFlag=False, normalize=True)
            for fac in dmFactorNames:
                means[fac] = numpy.dot(weights, frets.loc[fac,:].values)
            frData.mean = means

        # Compute factor covariance matrix
        if not self.hasCurrencyFactor:
            # Compute regular factor covariance matrix
            factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(frData)
        else:
            # Process currency block
            modelCurrencyFactors = [f for f in self.factors if f in self.currencies]
            cfReturns, cc, currencySpecificRisk = self.process_currency_block(\
                                dateList[:self.maxFCovObs], modelCurrencyFactors, modelDB)
            self.covarianceCalculator.configureSubCovarianceMatrix(1, cc)
            crData.data = cfReturns

            # Post-process the array data a bit more, then compute cov
            factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(frData, crData)

            # Add in the currency specific variances
            for cf in currencySpecificRisk.index:
                factorCov.loc[cf, cf] += currencySpecificRisk.loc[cf] * currencySpecificRisk.loc[cf]

        # Re-order according to list of model factors
        factorNames = [f.name for f in self.factors]
        factorCov = factorCov.reindex(index=factorNames, columns=factorNames)

        # Safeguard to deal with nasty negative eigenvalues
        factorCov = Utilities.forcePositiveSemiDefiniteMatrix(factorCov, min_eigenvalue=0.0)
        factorCov = (factorCov + factorCov.T) / 2.0

        # Set up asset information data structure
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=False)
        assetData = AssetProcessor_V4.AssetProcessor(\
                date, modelDB, marketDB, self.getDefaultAPParameters(useNursery=False))
        assetData.process_asset_information(rmgList, universe=universe)

        # Report day-on-day correlation matrix changes
        if len(dateList) > 1:
            prmi = modelDB.getRiskModelInstance(self.rms_id, dateList[1])
            estuFactorRisk = self.reportCorrelationMatrixChanges(date, assetData, expM, factorCov, rmi, prmi, modelDB)
        else:
            estuFactorRisk = 0.0

        # Set up return structure
        covData = Utilities.Struct()
        covData.subIssues = assetData.universe
        covData.factorCov = factorCov.fillna(0.0).values
        covData.subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        covData.estuRisk = estuFactorRisk

        if dvaToggle:
            logging.getLogger().setLevel(origLoggingLevel)
            self.covarianceCalculator.varParameters.DVAWindow = varDVA_save
            self.covarianceCalculator.corrParameters.DVAWindow = corDVA_save
        if nwToggle:
            logging.getLogger().setLevel(origLoggingLevel)
            self.covarianceCalculator.varParameters.NWLag = varNW_save
            self.covarianceCalculator.corrParameters.NWLag = corNW_save

        # Compute specific risk
        if not self.runFCovOnly:
            covData.specificVars, covData.specificCov = self.compute_specific_risk(\
                    date, dateList, assetData, rmgList, modelDB, marketDB)

        self.log.debug('computed factor covariances')
        return covData

    def process_currency_block(self, dateList, modelCurrencyFactors, modelDB):
        """Does all the messy processing of the currency covariance matrix and factor returns,
        in order to back out an extended history of currency factor returns and separate currency
        factor from specific return and risk
        """
        # Instantiate currency model
        self.log.info('Merging currency model: %s, (factor model: %s)',
                self.currencyModel.name, self.currencyModel.isFactorModel())
        crmi, dummy = self.set_model_instance(dateList[0], modelDB, rm=self.currencyModel)
        assert(crmi is not None)

        # Get list of factors associated with currency model
        cmCurrencyFactors = self.currencyModel.getCurrencyFactors()
        cmCurrencyFactorNames = [f.name for f in cmCurrencyFactors]
        mcFactorNames = [f.name for f in modelCurrencyFactors]
        notInCM = set(mcFactorNames).difference(set(cmCurrencyFactorNames))
        if len(notInCM) > 0:
            logging.error('Currencies in the risk model but not in the currency model: %s', ','.join(notInCM))

        # Load currency statistical factor exposures and rename the x-axis from cash-asset to currency
        currencyExpM = self.currencyModel.loadExposureMatrix(crmi, modelDB, returnDF=True).fillna(0.0)
        expRemap = [c if isinstance(c, str) else c.getCashCurrency() for c in currencyExpM.index]
        currencyExpM.rename(index=dict(zip(currencyExpM.index, expRemap)), inplace=True)

        # Report on currencies missing an exposure (shouldn't be any)
        noExposures = set(cmCurrencyFactorNames).difference(set(currencyExpM.index))
        if len(noExposures) > 0:
            logging.error('No currency exposures for the following currencies: %s', ','.join(noExposures))
        currencyExpM = currencyExpM.reindex(cmCurrencyFactorNames)

        # Load currency "specific" risks and rename index from cash assets to currencies
        curSpecificRisk = self.currencyModel.loadSpecificRisks(crmi, modelDB, returnDF=True)
        srRemap = [c if isinstance(c, str) else c.getCashCurrency() for c in curSpecificRisk.index]
        curSpecificRisk.rename(index=dict(zip(curSpecificRisk.index, srRemap)), inplace=True)

        # Report on missing specific risks
        noSpecRisks = set(cmCurrencyFactorNames).difference(set(curSpecificRisk.index))
        if len(noSpecRisks) > 0:
            logging.error('No currency specific risks for the following currencies: %s', ','.join(noSpecRisks))
        curSpecificRisk = curSpecificRisk.reindex(cmCurrencyFactorNames)

        # Load currency model estimation universe and map to currency factors
        currESTU = set(c if isinstance(c, str) else c.getCashCurrency() for c in \
                self.currencyModel.loadEstimationUniverse(crmi, modelDB))
        cfESTU = [f for f in cmCurrencyFactors if f.name in currESTU]

        # Get currency model returns for estu currencies
        estuCurrencyFactorReturns = self.currencyModel.loadCurrencyFactorReturnsHistory(
                        crmi, dateList, modelDB, factorList=cfESTU, returnDF=True, screen=True).fillna(0.0)

        # Estimate currency model stat factor returns for required history length
        (currencyFR, dummy) = Utilities.ordinaryLeastSquares(
                estuCurrencyFactorReturns.values, currencyExpM.loc[[f.name for f in cfESTU]].values)

        # Reorder our data to match the currency factors as used by the regional model
        currencyExpM = currencyExpM.loc[mcFactorNames]
        curSpecificRisk = curSpecificRisk.loc[mcFactorNames]

        # Create set of currency factor returns containing only the common factor component
        currencyFR = pandas.DataFrame(currencyFR, index=currencyExpM.columns, columns=dateList)
        currencyFactorReturns = currencyExpM.dot(currencyFR)

        # Expand currency stat factor cov to full (dense) cov
        currencyCov = self.currencyModel.loadFactorCovarianceMatrix(crmi, modelDB, returnDF=True)
        cc = (currencyExpM.dot(currencyCov)).dot(currencyExpM.T) / 252.0
        return currencyFactorReturns, cc, curSpecificRisk

    def compute_specific_risk(self, date, dateList, assetData, rmgList, modelDB, marketDB):
        """ Computes specific risk and covariance over the universe of sub-issues
        """
        # Initialise
        specificCov = dict()

        # Load specific returns history
        specRetDates = dateList[:self.maxSRiskObs]
        logging.info('Loading specific returns for %d assets, %d dates from %s to %s',
                len(assetData.universe), len(specRetDates), specRetDates[0], specRetDates[-1])
        if hasattr(self, 'hasInternalSpecRets') and self.hasInternalSpecRets:
            specificReturns = self.loadSpecificReturnsHistory(date, assetData.universe, specRetDates, modelDB, marketDB, internal=True)
        else:
            specificReturns = self.loadSpecificReturnsHistory(date, assetData.universe, specRetDates, modelDB, marketDB)

        # ISC info
        scores = self.load_ISC_Scores(date, assetData, modelDB, marketDB, returnDF=True)
        self.group_linked_assets(date, assetData, modelDB, marketDB)

        # Proportion of returns traded
        numOkReturns = numpy.isfinite(specificReturns).sum(axis=1)
        numOkReturns = numOkReturns / float(numOkReturns.max(axis=None))

        assetTypeMap = pandas.Series(assetData.getAssetType())
        noProxyReturnsList = set(assetTypeMap[assetTypeMap.isin(self.noProxyTypes)].index)
        exSpacs = AssetProcessor_V4.sort_spac_assets(\
                date, assetData.universe, modelDB, marketDB, returnExSpac=True)
        noProxyReturnsList = noProxyReturnsList.difference(exSpacs)
        if len(noProxyReturnsList) > 0:
            logging.info('%d assets of type %s excluded from proxying', len(noProxyReturnsList), self.noProxyTypes)

        # Specific risk computation
        (specificVars, specificCov) = self.specificRiskCalculator.computeSpecificRisks(\
                specificReturns, assetData, self, modelDB, nOkRets=numOkReturns, scoreDict=scores,
                rmgList=rmgList, excludeAssets=noProxyReturnsList)

        # Write specific risk per asset
        if self.debuggingReporting:
            outfile = 'tmp/specificRisk-%s-%s.csv' % (self.name, date)
            outfile = open(outfile, 'w')
            outfile.write('CID,SID,Name,Type,exSPAC,ESTU,Risk,Ret%\n')
            specificRisks = 100.0 * numpy.sqrt(specificVars)
            exSpac = AssetProcessor_V4.sort_spac_assets(\
                    date, list(specificReturns.index), modelDB, marketDB, returnExSpac=True)
            for sid in specificReturns.index:
                outfile.write('%s,%s,%s,%s,' % ( \
                        assetData.getSubIssue2CidMapping().get(sid, ''), sid.getSubIDString(),
                        assetData.getNameMap().get(sid, '').replace(',',''), assetData.getAssetType().get(sid, '')))
                if sid in exSpac:
                    outfile.write('1,')
                else:
                    outfile.write('0,')
                if sid in assetData.estimationUniverse:
                    outfile.write('1,%.6f,%.2f\n' % (specificRisks[sid], numOkReturns[sid]))
                else:
                    outfile.write('0,%.6f,%.2f\n' % (specificRisks[sid], numOkReturns[sid]))
            outfile.close()
        
        if len(dateList) > 1:
            prmi = modelDB.getRiskModelInstance(self.rms_id, dateList[1])
            self.reportISCStability(date, specificVars, specificCov, prmi, modelDB)
        self.log.debug('computed specific variances')
        return specificVars, specificCov

    def assetReturnHistoryLoader(self, assetData, returnHistoryLength, modelDate, modelDB, marketDB,
            loadOnly=True, applyRT=False, fixNonTradingDays=False, cointTest=False):
        """ Function to load in returns for factor regression
        - loadOnly just loads the returns, and doesn't compute proxies
        - applyRT - aligns returns with the US market if True
        - fixNonTradingDays - computes a simple proxy for missing returns
        """
        assetTypeMap = pandas.Series(assetData.assetTypeDict)
        noProxyReturnsList = set(assetTypeMap[assetTypeMap.isin(self.noProxyTypes)].index)
        exSpacs = AssetProcessor_V4.sort_spac_assets(\
                modelDate, assetData.universe, modelDB, marketDB, returnExSpac=True)
        noProxyReturnsList = noProxyReturnsList.difference(exSpacs)
        if len(noProxyReturnsList) > 0:
            logging.info('%d assets of type %s excluded from proxying', len(noProxyReturnsList), self.noProxyTypes)

        if cointTest:
            applyRT = False
            fixNonTradingDays = False
            loadOnly = True

        assetIdxMap = dict(zip(assetData.universe, range(len(assetData.universe))))
        estimationUniverseIdx = [assetIdxMap[sid] for sid in assetData.estimationUniverse]

        # Load in history of returns
        returnsProcessor = ProcessReturns.assetReturnsProcessor(
                self.rmg, assetData.universe, assetData.rmgAssetMap,
                assetData.tradingRmgAssetMap, assetData.assetTypeDict,
                numeraire_id=self.numeraire.currency_id, tradingCurrency_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId, debuggingReporting=self.debuggingReporting,
                gicsDate=self.gicsDate, estu=estimationUniverseIdx, boT=self.firstReturnDate,
                simpleProxyRetTol=self.simpleProxyRetTol)
        returnsHistory = returnsProcessor.process_returns_history(
                modelDate, int(returnHistoryLength), modelDB, marketDB,
                drCurrMap=assetData.drCurrData, loadOnly=loadOnly, excludeWeekend=True,
                applyRT=applyRT, trimData=False, noProxyList=noProxyReturnsList,
                useAllRMGDates=self.coverageMultiCountry)

        if fixNonTradingDays:
            # Load market returns
            mktRets = modelDB.loadRMGMarketReturnHistory(returnsHistory.dates, self.rmg, useAMPs=False)
            logging.info('Filling non-trading day returns with proxy')

            # Load market proxy returns
            mktProxies = modelDB.loadReturnsTimingAdjustmentsHistory(
                    self.returnsTimingId, self.rmg, returnsHistory.dates,
                    loadAllDates=True, loadProxy=True, legacy=False)

            # Load returns timing adjustments
            timingAdj = modelDB.loadReturnsTimingAdjustmentsHistory(
                    self.returnsTimingId, self.rmg, returnsHistory.dates,
                    loadAllDates=True, legacy=False)
            mktProxies = ma.filled(mktRets.data, 0.0) + ma.filled(mktProxies.data, 0.0) + ma.filled(timingAdj.data, 0.0)

            # Trim outliers
            opms = dict()
            outlierClass = Outliers.Outliers(opms)
            mktProxies = outlierClass.twodMAD(mktProxies)

            if self.debuggingReporting:
                mktProxyDict = dict(zip(self.rmg, mktProxies[:,-1]))
                for (rmg, proxy) in mktProxyDict.items():
                    logging.info('Market %s uses proxy value %.4f for %s', rmg.mnemonic, proxy, returnsHistory.dates[-1])

            # Load asset raw betas if they exist
            betas = numpy.ones((len(assetData.universe)), float)
            descDict = dict(modelDB.getAllDescriptors())
            if 'Market_Sensitivity_104W' in descDict:
                betas = self.loadDescriptors(['Market_Sensitivity_104W'],
                        descDict, modelDate, assetData.universe, modelDB, assetData.currencyAssetMap,
                        rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, 'Market_Sensitivity_104W']
                opms = dict()
                opms['nBounds'] = [3.0, 3.0]
                outlierClass = Outliers.Outliers(opms)
                betas = ma.filled(outlierClass.twodMAD(betas.values), 1.0)

            # Create a crude asset proxy
            assetProxyReturn = numpy.zeros((len(assetData.universe), len(returnsHistory.dates)), float)
            rmgIdxMap = dict([(rmg.rmg_id,i) for (i,rmg) in enumerate(timingAdj.assets)])
            rmgIdxMap2 = dict([(rmg,i) for (i,rmg) in enumerate(timingAdj.assets)])
            homeCountryMap = dict([(sid, rmg_id) for (rmg_id, idSet) in \
                    assetData.rmgAssetMap.items() for sid in idSet])
            for (idx, sid) in enumerate(assetData.universe):
                if (sid in noProxyReturnsList):
                    continue
                homeIdx = homeCountryMap.get(sid, None)
                if homeIdx is not None and (homeIdx in rmgIdxMap):
                    rmgIdx = rmgIdxMap[homeIdx]
                    assetProxyReturn[idx,:] += (betas[idx] * mktProxies[rmgIdx,:])
                if homeIdx is not None and (homeIdx in rmgIdxMap2):
                    rmgIdx = rmgIdxMap2[homeIdx]
                    assetProxyReturn[idx,:] += (betas[idx] * mktProxies[rmgIdx,:])
            assetProxyReturn = numpy.clip(assetProxyReturn, -0.75, 2.0)

            # Blend these with the actual returns
            missingDataMask = numpy.array(returnsHistory.missingFlag, copy=True)
            returnsHistory.data = ProcessReturns.fill_and_smooth_returns(
                    returnsHistory.data, assetProxyReturn, mask=missingDataMask, preIPOFlag=returnsHistory.preIPOFlag)[0]

        if cointTest:
            RiskCalculator_V4.test_for_cointegration(self, modelDate, assetData, modelDB, marketDB, returnsHistory)

        return returnsHistory

    def build_excess_return_history(self, assetData, modelDate, modelDB, marketDB,
            loadOnly=True, applyRT=False, fixNonTradingDays=False, returnDF=False):
        """Wrapper for building history of excess returns
        The logic is complex as an asset's risk-free rate ISO
        can change over time as its trading currency changes
        Thus, we need to loop over blocks of constant ISO
        """
        # Load in history of returns
        returnsHistory = self.assetReturnHistoryLoader(
                assetData, self.returnHistory, modelDate, modelDB, marketDB,
                loadOnly=loadOnly, applyRT=applyRT, fixNonTradingDays=fixNonTradingDays)

        # And a bit of sorting and checking
        if returnsHistory.data.shape[1] > self.returnHistory:
            returnsHistory.data = returnsHistory.data[:,-self.returnHistory:]
            returnsHistory.dates = returnsHistory.dates[-self.returnHistory:]
            returnsHistory.preIPOFlag = returnsHistory.preIPOFlag[:,-self.returnHistory:]
            returnsHistory.missingFlag = returnsHistory.missingFlag[:,-self.returnHistory:]
            returnsHistory.zeroFlag = returnsHistory.zeroFlag[:,-self.returnHistory:]
        dateList = returnsHistory.dates
        nonMissingFlag = (returnsHistory.preIPOFlag==0) * (returnsHistory.missingFlag==0)
        assetData.nonMissingFlag = nonMissingFlag
        assetData.nonZeroFlag = (returnsHistory.zeroFlag==0)
        numOkReturns = ma.sum(nonMissingFlag, axis=1)
        assetData.numOkReturns = numOkReturns / float(numpy.max(numOkReturns, axis=None))
        assetData.preIPOFlag = returnsHistory.preIPOFlag

        # Gather a list of all factors that have ever existed in the model
        allFactorsEver = modelDB.getRiskModelSerieFactors(self.rms_id)
        # Determine relevant dates on which factor structure has changed
        changeDates = set([f.from_dt for f in allFactorsEver if \
                f.from_dt >= dateList[0] and f.from_dt <= dateList[-1]])
        if len(changeDates) > 0:
            self.log.info('Factor structure changes on these dates: %s',
                    ','.join(['%s' % str(d) for d in changeDates]))
        else:
            self.log.info('No factor structure changes')

        changeDates.add(dateList[0])
        changeDates.add(datetime.date(2999,12,31))
        changeDates = sorted(changeDates)

        # Loop round chunks of constant factor structure
        # Ensures that risk-free rates get mapped to correct currency ISOs
        # when the factor structure changes
        allReturns = Matrices.allMasked((len(assetData.universe),len(dateList)), float)
        startDate = changeDates[0]
        iDate0 = 0
        for endDate in changeDates[1:]:
            # Get list of dates within given range
            subDateList = sorted(d for d in dateList if startDate <= d < endDate)
            self.log.info('Using returns from %s to %s', subDateList[0], subDateList[-1])
            assert(hasattr(modelDB, 'currencyCache'))
            if len(assetData.foreign) > 0:
                # Map rmgs to their trading currency ID at endDate
                lastRangeDate = subDateList[-1]
                rmgCurrencyMap = dict()
                for r in self.rmgTimeLine:
                    if r.from_dt <= lastRangeDate \
                            and r.thru_dt > lastRangeDate:
                        ccyCode = r.rmg.getCurrencyCode(lastRangeDate)
                        ccyID = modelDB.currencyCache.getCurrencyID(ccyCode, lastRangeDate)
                        rmgCurrencyMap[r.rmg_id] = ccyID

                # Force DRs to have the trading currency of their home
                # country for this period
                drCurrData = dict()
                for (r, rmgAssets) in assetData.rmgAssetMap.items():
                    drCurrency = rmgCurrencyMap.get(r, rmgCurrencyMap.get(r.rmg_id))
                    if drCurrency is None:
                        # the model didn't cover the country earlier
                        # so get its information from the database
                        if not hasattr(r, 'rmg_id'):
                            rmg = modelDB.getRiskModelGroup(r)
                            ccyCode = rmg.getCurrencyCode(lastRangeDate)
                        else:
                            ccyCode = r.getCurrencyCode(lastRangeDate)
                        drCurrency = modelDB.currencyCache.getCurrencyID(ccyCode, lastRangeDate)
                    for sid in rmgAssets & set(assetData.foreign):
                        drCurrData[sid] = drCurrency
            else:
                drCurrData = None

            # Get local returns history for date range
            dateIdxMap = dict(zip(returnsHistory.dates, range(len(returnsHistory.dates))))
            subDateIdx = [dateIdxMap[d] for d in subDateList]
            returns = Utilities.Struct()
            returns.assets = returnsHistory.assets
            returns.data = ma.take(returnsHistory.data, subDateIdx, axis=1)
            returns.dates = list(subDateList)

            # Compute excess returns for date range
            (returns, rfr) = self.computeExcessReturns(subDateList[-1],
                    returns, modelDB, marketDB, drCurrData)

            # Copy chunk of excess returns into full returns matrix
            iDate1 = iDate0 + len(subDateList)
            allReturns[:,iDate0:iDate1] = returns.data
            iDate0 += len(subDateList)
            startDate = endDate
 
        # Copy excess returns back into returns structure
        returnsHistory.data = allReturns
        assetData.returns = returnsHistory

        if returnDF:
            returnsData = Utilities.Struct()
            returnsData.nonMissingFlag = pandas.DataFrame(\
                    assetData.nonMissingFlag, index=assetData.universe, columns=returnsHistory.dates)
            returnsData.nonZeroFlag = pandas.DataFrame(\
                    assetData.nonZeroFlag, index=assetData.universe, columns=returnsHistory.dates)
            returnsData.numOkReturns = pandas.Series(assetData.numOkReturns, index=assetData.universe)
            returnsData.preIPOFlag = pandas.DataFrame(\
                    assetData.preIPOFlag, index=assetData.universe, columns=returnsHistory.dates)
            returnsData.returns = pandas.DataFrame(\
                    returnsHistory.data, index=assetData.universe, columns=returnsHistory.dates)
            returnsData.missingFlag = pandas.DataFrame(\
                    returnsHistory.missingFlag, index=assetData.universe, columns=returnsHistory.dates)
            return returnsData

        return assetData

class FundamentalModel(FactorRiskModel):
    """Fundamental factor model
    """
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)

        # No statistical factors, of course
        self.blind = []
        self.VIF = None

        # Set up default estimation universe
        self.masterEstuMap = modelDB.getEstuMappingTable(self.rms_id)
        if self.masterEstuMap is None:
            logging.error('No estimation universe mapping defined')
            assert(self.masterEstuMap is not None)
        logging.info('Estimation universe structure: %d estus', len(self.masterEstuMap))

        # Special regression treatment for certain factors 
        modelDB.setTotalReturnCache(367)
        modelDB.setVolumeCache(150)

        # Report on important parameters
        logging.info('Using legacy market cap dates: %s', self.legacyMCapDates)
        logging.info('Using new descriptor structure: %s', self.regionalDescriptorStructure)
        logging.info('Using two regression structure: %s', self.twoRegressionStructure)
        logging.info('GICS date used: %s', self.gicsDate)
        logging.info('Earliest returns date: %s', self.firstReturnDate)
        if hasattr(self, 'coverageMultiCountry'):
            logging.info('model coverageMultiCountry: %s', self.coverageMultiCountry)
        if hasattr(self, 'hasCountryFactor'):
            logging.info('model hasCountryFactor: %s', self.hasCountryFactor)
        if hasattr(self, 'hasCurrencyFactor'):
            logging.info('model hasCurrencyFactor: %s', self.hasCurrencyFactor)
        if hasattr(self, 'applyRT2US'):
            logging.info('model applyRT2US: %s', self.applyRT2US)
        if hasattr(self, 'hasCountryFactor') and hasattr(self, 'hasCurrencyFactor'):
            logging.info('Regional model: %s', self.isRegionalModel())
        if hasattr(self, 'returnsTimingId'):
            logging.info('Returns Timing ID: %s', self.returnsTimingId)
        
    def isStatModel(self):
        return False

    def isCurrencyModel(self):
        return False

    def setFactorsForDate(self, date, modelDB):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Set up estimation universe parameters
        self.estuMap = copy.deepcopy(self.masterEstuMap)

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)
        if hasattr(self, 'baseModelDateMap'):
            self.setBaseModelForDate(date)
        else:
            self.baseModel = None

        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        self.allStyles = list(self.totalStyles)

        # Assign to new industry scheme if necessary
        if hasattr(self, 'industrySchemeDict'):
            chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
            self.industryClassification = self.industrySchemeDict[chngDates[-1]]
            self.log.debug('Using %s classification scheme, rev_dt: %s'%\
                          (self.industryClassification.name, chngDates[-1].isoformat()))

        # Create industry factors
        industries = list(self.industryClassification.getLeafNodes(modelDB).values())
        self.industries = [ModelFactor(None, f.description) for f in industries]

        if self.hasCountryFactor:
            # Create country factors
            countries = [ModelFactor(r.description, None) for r in self.rmg \
                    if r.description in self.nameFactorMap]
            self.CountryFactorRMGMap = dict(zip(countries, self.rmg))
        else:
            countries = []
            self.countryFactorRMGMap = dict()
        if self.hasCurrencyFactor:
            # Create currency factors
            allRMG = modelDB.getAllRiskModelGroups(inModels=True)
            for rmg in allRMG:
                rmg.setRMGInfoForDate(date)
            currencies = [ModelFactor(f, None) for f in set([r.currency_code for r in allRMG])]
            currencies.extend([ModelFactor('EUR', 'Euro')])
            currencies = sorted(set([f for f in currencies if f.name in self.nameFactorMap]))
        else:
            currencies = []
        # Get factor types in order
        regional = countries + currencies
        allFactors = self.allStyles + self.industries + regional
        if self.intercept is not None and self.intercept not in allFactors:
            allFactors = allFactors + [self.intercept]
        if len(self.localStructureFactors) > 0:
            allFactors = allFactors + self.localStructureFactors

        for f in allFactors:
            if f.name in self.nameFactorMap:
                dbFactor = self.nameFactorMap[f.name]
            else:
                dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            if f.name is None:
                f.name = dbFactor.name
            if f.description is None:
                f.description = dbFactor.description

        # Drop dead styles and currencies
        dropped = [s for s in self.allStyles if not s.isLive(date)]
        if len(dropped) > 0:
            self.log.info('%d styles not live in %s model: %s', len(dropped), date, dropped)
        self.styles = [s for s in self.allStyles if s.isLive(date)]
        self.currencies = [c for c in currencies if c.isLive(date)]
        self.countries = [f for f in countries if f.isLive(date)]
        self.nurseryCountries = [f for f in countries if f not in self.countries]
        self.hiddenCurrencies = [f for f in currencies if f not in self.currencies]
        self.nurseryRMGs = [self.CountryFactorRMGMap[f] for f in self.nurseryCountries]
        if len(self.nurseryRMGs) > 0:
            logging.debug('%d out of %d RMGs classed as nursery: %s',
                    len(self.nurseryRMGs), len(self.rmg),
                    ','.join([r.mnemonic for r in self.nurseryRMGs]))
        if len(self.hiddenCurrencies) > 0:
            logging.debug('%d out of %d currencies classed as hidden: %s',
                    len(self.hiddenCurrencies), len(currencies),
                    ','.join([c.name for c in self.hiddenCurrencies]))

        # Set up dicts
        self.factors = [f for f in allFactors if f.isLive(date)]
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.validateFactorStructure(date, warnOnly=self.variableStyles)
        self.allFactors = allFactors
    
    def generate_estimation_universe(self, date, assetData, exposureMatrix, modelDB, marketDB,
            excludeFactors=[], grandfatherRMS_ID=None):
        """Generic estimation universe selection criteria for regional models.
           Excludes assets:
            - smaller than a certain cap threshold, per country
            - smaller than a certain cap threshold, per industry
            - with insufficient returns over the history
        """
        if not hasattr(self, 'estu_parameters'):
            raise Exception('Undefined estu_parameters!')
        self.log.info('generate_estimation_universe_inner: begin')

        # Initialise
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
               date, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)

        # Load asset score descriptors
        descDict = dict(modelDB.getAllDescriptors())
        scoreTypes = ['ISC_Ret_Score', 'ISC_Zero_Score', 'ISC_ADV_Score']
        assetData.assetScoreDict, okDescriptorCoverageMap = self.loadDescriptors(
                scoreTypes, descDict, date, assetData.universe, modelDB,
                assetData.getCurrencyAssetMap(), rollOver=self.rollOverDescriptors)
        for typ in scoreTypes:
            exposureMatrix.addFactor(typ, assetData.assetScoreDict.loc[:, typ].fillna(0.0), ExposureMatrix.StyleFactor)

        # Initialise parameters
        if type(self.estu_parameters['minNonMissing']) is not list:
            minNonMissingList = [self.estu_parameters['minNonMissing'], None]
            simpleScreening = True
        else:
            minNonMissingList = list(self.estu_parameters['minNonMissing'])
            simpleScreening = False

        # Report initial estimation universe
        eligibleUniverse = set(assetData.eligibleUniverse)
        n = len(eligibleUniverse)
        logging.info('ESTU currently stands at %d stocks', n)

        # Remove nursery market assets
        if len(assetData.originalNurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            eligibleUniverse = eligibleUniverse.difference(assetData.originalNurseryUniverse)
            n = estuCls.report_on_changes(n, eligibleUniverse)

        # From DR/underlying pairs, get set of eligible underlying assets
        underlyingSet = set(assetData.getDr2UnderMap().values()).intersection(eligibleUniverse)
        underlying2DrMap = dict()
        if len(underlyingSet) > 0:
            drScoreDict = dict()
            # Loop round underlying assets and find 'best' DR as replacement, if any
            for dr, under in assetData.getDr2UnderMap().items():
                if (under in underlyingSet) and (under is not None):
                    drScore = assetData.assetScoreDict.loc[dr, 'ISC_Ret_Score']
                    sidScore = assetData.assetScoreDict.loc[under, 'ISC_Ret_Score']
                    if (drScore > 1.3 * sidScore) and (drScore > drScoreDict.get(under, 0.0)):
                        # If more than one DR per underlying, check which has highest score
                        underlying2DrMap[under] = dr
                        drScoreDict[under] = drScore

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for regularly-traded stocks')
        if 'minNonZero' in self.estu_parameters:
            nonSparseIds = estuCls.exclude_sparsely_traded_assets_legacy(\
                    assetData.assetScoreDict, baseEstu=eligibleUniverse,
                    minNonZero=self.estu_parameters['minNonZero'], minNonMissing=minNonMissingList[0])
        else:
            nonSparseIds = estuCls.exclude_sparsely_traded_assets(
                    assetData.assetScoreDict, baseEstu=eligibleUniverse, minGoodReturns=minNonMissingList[0])
        logging.info('Excluding %d assets with fewer than %d%% of returns trading',
                len(eligibleUniverse)-len(nonSparseIds), int(100*minNonMissingList[0]))

        if self.debuggingReporting:
            sparseIds = sorted(eligibleUniverse.difference(nonSparseIds))
            assetData.assetScoreDict.loc[sparseIds, :].to_csv('tmp/sparseData-%s-%s.csv' % (self.mnemonic, date))

        # Exclude lower percentiles of assets by ADV
        if 'ADV_percentile' in self.estu_parameters:
            highVolIds = estuCls.filter_by_user_score(
                    assetData.assetScoreDict, baseEstu=eligibleUniverse,
                    lower_pctile=self.estu_parameters['ADV_percentile'][0],
                    upper_pctile=self.estu_parameters['ADV_percentile'][1])
            nonSparseIds = nonSparseIds.intersection(highVolIds)
            logging.info('Final number of sufficiently liquid assets is %d', len(nonSparseIds))

        # Check for DRs whose underlyings have been excluded
        validDRs, exclUnders = self.get_eligible_drs(\
                    assetData, underlying2DrMap, nonSparseIds, estuCls, minNonMissingList[0])
        nonSparseIds = nonSparseIds.union(validDRs)

        # Exclude thinly traded assets
        if simpleScreening:
            logging.info('Simple screening: removing illiquid stocks from ESTU')
            estimationUniverse = eligibleUniverse.intersection(nonSparseIds)
        else:
            estimationUniverse = eligibleUniverse.union(validDRs)
            # A little more complicated  - exclude only underlying stocks with more liquid DRs here
            if len(validDRs) > 0:
                estimationUniverse = estimationUniverse.difference(exclUnders)
                logging.info('Replaced %d assets with more liquid DRs', len(validDRs))

        # Interim report
        n = estuCls.report_on_changes(n, estimationUniverse)
        mcap_ESTU = assetData.marketCaps[estimationUniverse].sum(axis=None)
        self.log.info('...Before mcap filtering: ESTU consists of %d assets, %.2f tr %s market cap',
                len(estimationUniverse), mcap_ESTU / 1e12, self.numeraire.currency_code)

        if self.estu_parameters['CapByNumber_Flag']:
            # Filter by combination of mcap and volume
            estimationUniverse = estuCls.filter_by_cap_and_volume(\
                    assetData, baseEstu=estimationUniverse,
                    hiCapQuota=self.estu_parameters['CapByNumber_hiCapQuota'],
                    loCapQuota=self.estu_parameters['CapByNumber_lowCapQuota'],
                    bufferFactor=1.2)
        else:
            estuCopy = set()
            # Here we remove illiquid assets (redundant in some cases)
            logging.info('Removing illiquid stocks from ESTU')
            estimationUniverse = estimationUniverse.intersection(nonSparseIds)
            n = estuCls.report_on_changes(n, estimationUniverse)

            # Weed out small assets by the entire market
            if self.estu_parameters['market_lower_pctile'] is not None:
                tol = self.estu_parameters['market_lower_pctile']
                logging.info('Filtering by top %.1f%% mcap on entire market', 100-tol)
                estuCopy = estuCls.exclude_by_cap_ranking(
                        assetData, baseEstu=estimationUniverse, lower_pctile=tol)

            # Weed out tiny-cap assets by country
            if self.estu_parameters['country_lower_pctile'] is not None:
                tol = self.estu_parameters['country_lower_pctile']
                logging.info('Filtering by top %.1f%% mcap on country', 100-tol)
                estu1 = estuCls.exclude_by_cap_ranking(
                        assetData, baseEstu=estimationUniverse,
                        byFactorType=ExposureMatrix.CountryFactor, lower_pctile=tol,
                        expM=exposureMatrix, excludeFactors=excludeFactors)
                estuCopy = estu1.union(estuCopy)

            # Perform similar check by industry
            if self.estu_parameters['industry_lower_pctile'] is not None:
                tol = self.estu_parameters['industry_lower_pctile']
                logging.info('Filtering by top %.1f%% mcap on industry', 100-tol)
                estu2 = estuCls.exclude_by_cap_ranking(
                        assetData, baseEstu=estimationUniverse,
                        byFactorType=ExposureMatrix.IndustryFactor, lower_pctile=tol,
                        expM=exposureMatrix, excludeFactors=excludeFactors)
                estuCopy = estu2.union(estuCopy)

            if len(estuCopy) > 0:
                estimationUniverse = set(estuCopy)

        # Report on shrunk-down estimation universe
        n = estuCls.report_on_changes(n, estimationUniverse)

        # Inflate any thin countries or industries
        logging.info('Inflating any thin factors - using most liquid (%.2f%%) assets', 100.0*minNonMissingList[0])
        estimationUniverse, hasThinFactors = estuCls.pump_up_factors(\
                assetData, exposureMatrix, currentEstu=estimationUniverse, baseEstu=nonSparseIds,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=self.estu_parameters['dummyThreshold'],
                cutOff=self.estu_parameters['inflation_cutoff'], excludeFactors=excludeFactors, quiet=True)
        n = estuCls.report_on_changes(n, estimationUniverse)

        # If specified, loop round, decreasing returns quality tolerance to eliminate thin factors
        bottomOfBarrel = minNonMissingList[1]
        returnsTol = minNonMissingList[0]
        if bottomOfBarrel is not None:

            # Compute next set of parameters
            delta = minNonMissingList[-1]
            returnsTol -= delta

            # Loop round until we either succeed or run out of rope
            while hasThinFactors and (returnsTol >= bottomOfBarrel):
                nonSparseIds = estuCls.exclude_sparsely_traded_assets(
                        assetData.assetScoreDict, baseEstu=eligibleUniverse, minGoodReturns=returnsTol)

                if len(underlying2DrMap) > 0:
                    validDRs, exclUnders = self.get_eligible_drs(
                            assetData, underlying2DrMap, estimationUniverse, estuCls, returnsTol)
                    nonSparseIds = nonSparseIds.union(validDRs)

                # Inflate any thin countries or industries
                logging.info('Inflating any thin factors - using less liquid (%.2f%%) assets', 100.0*returnsTol)
                estimationUniverse, hasThinFactors = estuCls.pump_up_factors(\
                        assetData, exposureMatrix,
                        currentEstu=estimationUniverse, baseEstu=nonSparseIds,
                        byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                        minFactorWidth=self.estu_parameters['dummyThreshold'],
                        cutOff=self.estu_parameters['inflation_cutoff'], excludeFactors=excludeFactors,
                        quiet=(returnsTol-delta>bottomOfBarrel))

                # Compute next set of parameters
                returnsTol -= delta

        # Interim reporting
        n = estuCls.report_on_changes(n, estimationUniverse)
        mcap_ESTU = assetData.marketCaps[estimationUniverse].sum(axis=None)
        self.log.info('...Before grandfathering: ESTU consists of %d assets, %.2f tr %s market cap',
                len(estimationUniverse), mcap_ESTU / 1e12, self.numeraire.currency_code)

        # Apply grandfathering
        estimationUniverse, ESTUQualify = estuCls.grandfather(
                estimationUniverse, baseEstu=eligibleUniverse, estuInstance=self.estuMap['main'],
                grandfatherRMS_ID=grandfatherRMS_ID)
        n = estuCls.report_on_changes(n, estimationUniverse)

        # If any DR/underlying pairs have both made it into the estu after all that,
        # exclude the DR
        duplicates = set([sid1 for (sid1, sid2) in assetData.getDr2UnderMap().items() \
                if (sid1 and sid2 in estimationUniverse)])
        if len(duplicates) > 0:
            logging.info('%d DRs in estu along with their underlying, excluding the former', len(duplicates))
            estimationUniverse = estimationUniverse.difference(duplicates)
            n = estuCls.report_on_changes(n, estimationUniverse)

        # Report on DRs in the estimation universe
        drsInEstu = estimationUniverse.intersection(set(assetData.getDr2UnderMap().keys()))
        if len(drsInEstu) > 0:
            logging.info('%d DRs in final estimation universe, %.2f tn (%s)',
                    len(drsInEstu), assetData.marketCaps[drsInEstu].sum(axis=None) / 1e12, self.numeraire.currency_code)

        # Report on final state
        self.log.info('Final estu contains %d assets, %.2f tn (%s)',
                    len(estimationUniverse), assetData.marketCaps[estimationUniverse].sum(axis=None) / 1e12,
                    self.numeraire.currency_code)
        self.log.debug('generate_estimation_universe_inner: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = estimationUniverse
        self.estuMap['main'].qualify = ESTUQualify

        return estimationUniverse

    def get_eligible_drs(self, assetData, underlying2DRMap, estu, estuCls, sparseTol):
        validDRs = set()
        excludedUnderlyings = set()
        if len(underlying2DRMap) < 1:
            return validDRs, excludedUnderlyings

        # Get subset of these that meet the liquidity criteria
        logging.debug('Found %d DRs whose underlying assets are eligible', len(underlying2DRMap))
        nonSparseDR = estuCls.exclude_sparsely_traded_assets(
                assetData.assetScoreDict, baseEstu=set(underlying2DRMap.values()), minGoodReturns=sparseTol)
        nsdrUnderlying = set([assetData.getDr2UnderMap()[sid] for sid in nonSparseDR])

        # Switch more liquid DR for any underlying not yet in the estimation universe
        if len(nsdrUnderlying) > 0:
            excludedUnderlyings = nsdrUnderlying.difference(estu)
            if len(excludedUnderlyings) > 0:
                logging.debug('%d DRs being considered for inclusion in place of excluded underlyings',
                        len(excludedUnderlyings))
                validDRs = set([underlying2DRMap[sid] for sid in excludedUnderlyings])

        return validDRs, excludedUnderlyings

    def generate_nursery_estu(\
                self, date, assetData, exposureMatrix, modelDB, marketDB, estuItem, extraUniverse, excludeFactors=[]):
        """Generic estimation universe selection criteria for nursery or non-core markets.
        Excludes assets:
            - smaller than a certain cap threshold, per country
            - with insufficient returns over the history
        """
        self.log.info('generate_nursery_estu: begin')

        # Initialise
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                    date, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
        eligibleUniverse = set(extraUniverse).intersection(assetData.eligibleUniverse)
        n = len(eligibleUniverse)
        logging.info('Initial %s universe stands at %d stocks', estuItem.name, len(eligibleUniverse))

        # Exclude thinly traded assets if required
        logging.info('Looking for thinly-traded stocks')
        nonSparseIds = estuCls.exclude_sparsely_traded_assets(assetData.assetScoreDict, baseEstu=eligibleUniverse, minGoodReturns=0.25)
        estu = eligibleUniverse.intersection(nonSparseIds)
        n = estuCls.report_on_changes(n, estu)

        # Weed out tiny-cap assets by country
        estu = estuCls.exclude_by_cap_ranking(assetData, baseEstu=estu, expM=exposureMatrix,
                byFactorType=ExposureMatrix.CountryFactor, lower_pctile=5, excludeFactors=excludeFactors)
        n = estuCls.report_on_changes(n, estu)

        # Inflate any thin countries
        estu, hasThinFactors = estuCls.pump_up_factors(
                assetData, exposureMatrix, currentEstu=estu, baseEstu=eligibleUniverse,
                byFactorType=[ExposureMatrix.CountryFactor], excludeFactors=excludeFactors)
        n = estuCls.report_on_changes(n, estu)

        # Apply grandfathering rules
        estu, ESTUQualify = estuCls.grandfather(
                estu, baseEstu=eligibleUniverse, estuInstance=estuItem)

        self.log.info('Final %s estu contains %d assets, %.2f tn (%s)',
                estuItem.name, len(estu), assetData.marketCaps[estu].sum(axis=None) / 1.0e12, self.numeraire.currency_code)
        estuItem.assets = estu
        estuItem.qualify = ESTUQualify
        self.log.debug('generate_non_core_estu: end')

        return estu

    def generate_smallcap_estus(self, scMap, date, assetData, modelDB, marketDB):
        """Generates estimation universes based on market cap for use by midcap and smallcap factors
        """
        if len(scMap) < 1:
            return
        self.log.info('generate_smallcap_estus: begin')

        # Initialise
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                    date, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)

        # Get total mcaps per issuer
        if not hasattr(assetData, 'mCapDF'):
            assetData.mCapDF = AssetProcessor_V4.computeTotalIssuerMarketCaps(
                    date, assetData.marketCaps, self.numeraire, modelDB, marketDB,
                    debugReport=self.debuggingReporting)

        # Set up various eligible and total universes
        runningUniverse = set(assetData.universe)
        eligibleUniverse = set(self.estuMap['main'].assets)
        issuerTotalMCap = assetData.mCapDF.loc[:, 'totalCap']
        logging.info('Smallcap ESTU pool currently stands at %d stocks', len(eligibleUniverse))

        # Loop round set of smallcap estus
        for scId in sorted(scMap.keys()):
            scName = scMap[scId]

            # Get estu info
            params = self.styleParameters[scName]
            if not hasattr(params, 'bounds'):
                logging.warning('No cap bounds for factor: %s', scName)
                continue
            logging.info('Generating %s universe, ID %d, bounds: %s', scName, scId, params.bounds)

            # Compute mcap thresholds
            capBounds = Utilities.prctile(issuerTotalMCap[eligibleUniverse].values, params.bounds)
            validSids = set(issuerTotalMCap[issuerTotalMCap.between(capBounds[0], capBounds[1])].index)
            validSids = validSids.intersection(runningUniverse)

            # Perform grandfathering
            estuSC, qualify = estuCls.grandfather(
                    validSids, baseEstu=runningUniverse, estuInstance=self.estuMap[scName])

            # Report and save values
            scCap = issuerTotalMCap[estuSC].sum(axis=None)
            logging.info('%s universe contains %d assets, (min: %.4f, max: %.4f), %.2f Bn Cap',
                    scName, len(estuSC), capBounds[0]/1.0e9, capBounds[1]/1.0e9, scCap/1.0e9)
            runningUniverse = runningUniverse.difference(estuSC)
            self.estuMap[scName].assets = estuSC
            self.estuMap[scName].qualify = qualify

        self.log.info('generate_smallcap_estus: end')
        return

    def generate_China_ESTU(self, date, estuName, estuParams, assetData, modelDB, marketDB):
        """Returns the estimation universe assets and weights for use in secondary China A regression
        """
        self.log.info('generate_China_ESTU: begin')
        # Initialise
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(
                date, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
        returnsTol = estuParams.returnsTol
        includeTypes = estuParams.includeTypes
        excludeTypes = estuParams.excludeTypes
        capBound = estuParams.capBound

        # Remove assets from the exclusion table
        logging.info('Applying exclusion table')
        elig = estuCls.apply_exclusion_list(productExclude=self.productExcludeFlag)
        n = len(elig)

        # Pick out A-shares by asset type
        elig = estuCls.exclude_by_asset_type(
                assetData, includeFields=includeTypes, excludeFields=excludeTypes, baseEstu=elig)
        n = estuCls.report_on_changes(n, elig)

        # Get rid of some exchanges we don't want
        for xCode in AssetProcessor_V4.connectExchanges:
            logging.info('Removing stocks on exchange: %s', xCode)
            elig = estuCls.exclude_by_market_type(
                    assetData, includeFields=None, excludeFields=[xCode], baseEstu=elig)
            n = estuCls.report_on_changes(n, elig)

        # Remove cloned assets
        hardCloneMap = assetData.getCloneMap(cloneType='hard')
        if len(hardCloneMap) > 0:
            logging.info('Removing cloned assets')
            elig = elig.difference(set(hardCloneMap.keys()))
            n = estuCls.report_on_changes(n, elig)

        # Exclude thinly traded assets
        logging.info('Looking for thinly-traded stocks')
        estu = estuCls.exclude_sparsely_traded_assets(\
                assetData.assetScoreDict, baseEstu=elig, minGoodReturns=returnsTol)
        n = estuCls.report_on_changes(n, estu)

        # Weed out tiny-cap assets by market
        logging.info('Filtering by top %.1f%% mcap on entire market', 100-capBound)
        estu = estuCls.exclude_by_cap_ranking(assetData, baseEstu=estu, lower_pctile=capBound)
        n = estuCls.report_on_changes(n, estu)

        # Perform grandfathering
        estu, qualify = estuCls.grandfather(estu, baseEstu=elig, estuInstance=self.estuMap[estuName])

        # Report and return
        asMCap = assetData.marketCaps[estu].sum(axis=None)
        logging.info('%s estu contains %d assets, %.2f Tn Cap', estuName, len(estu), asMCap/1.0e12)
        self.estuMap[estuName].assets = estu
        self.estuMap[estuName].qualify = qualify

        self.log.info('generate_China_ESTU: end')
        return

    def shrink_to_mean(self, modelDate, assetData, modelDB, marketDB, descriptorName, historyLength,
            values, useSector=True, useRegion=True, onlyIPOs=True):
        """ Code for shrinking exposure values towards a specified value in cases of short history due to e.g. IPO
        """
        self.log.debug('shrink_to_mean: begin')
        if not self.isRegionalModel():
            useRegion = False
        preSPAC = assetData.getSPACs()

        # Load up from dates and determine scaling factor
        if onlyIPOs:
            logging.info('Shrinking based on IPO-date')
            fromDates = Utilities.load_ipo_dates(\
                    modelDate, assetData.universe, modelDB, marketDB, exSpacAdjust=True, returnList=True)
            # Define scale factor as function of age
            distance = pandas.Series([int((modelDate - dt).days) for dt in fromDates], index=assetData.universe)
            distance = distance.mask(distance>=historyLength)
            goodVals = set(distance[distance.isnull()].index)
            scaleFactor = (distance / float(historyLength)).fillna(1.0)
        else:
            # Test code to shrink all returns-deficient values
            if self.descriptorNumeraire is not None:
                descriptorNumeraire = self.descriptorNumeraire
            else:
                descriptorNumeraire = self.numeraire.currency_code
            logging.info('Shrinking based on all missing returns')
            scoreType = 'Percent_Returns_%d_Days' % historyLength
            descDict = dict(modelDB.getAllDescriptors())
            logging.info('Loading %s data', scoreType)

            # Load % of returns traded
            retScore = self.loadDescriptors([scoreType], descDict, modelDate, assetData.universe, modelDB,
                    None, rollOver=self.rollOverDescriptors)[0].loc[:, scoreType]
            scaleFactor = retScore.loc[:, modelDate].fillna(0.0)
            goodVals = set(scaleFactor[scaleFactor.mask(scaleFactor>=1.0).isnull()].index)

        # Sort into good, missing and SPAC
        if len(preSPAC) < 0:
            goodVals = goodVals.difference(preSPAC)
        missingIds = set(values[values.isnull()].index).difference(preSPAC)
        if len(missingIds) > 0:
            logging.warning('Treating %d assets with missing %s values as having missing histories',
                    len(missingIds), descriptorName)
            scaleFactor[missingIds] = 0.0
            goodVals = goodVals.difference(set(missingIds))

        # Get sector/industry group exposures
        if useSector:
            level = 'Sectors'
        else:
            level = 'Industry Groups'
        sectorExposures = Utilities.buildGICSExposures(
                assetData.universe, modelDate, modelDB, level=level, clsDate=self.gicsDate, returnDF=True)
        sectorExposures = sectorExposures.mask(sectorExposures.fillna(0.0) < 1.0)

        # Bucket assets into regions/countries
        regionIDMap = dict()
        regionAssetMap = defaultdict(set)
        if not useRegion:
            for r in self.rmg:
                regionAssetMap[r.rmg_id] = set(assetData.rmgAssetMap[r])
        else:
            for r in self.rmg:
                regionAssetMap[r.region_id] = regionAssetMap[r.region_id].union(set(assetData.rmgAssetMap[r]))

        # Compute mean of entire estimation universe to be used if insufficient values in any region/sector bucket
        goodEstuAssets = assetData.estimationUniverse.intersection(goodVals)
        globalMean = ma.average(values[goodEstuAssets].values, axis=None, weights=assetData.marketCaps[goodEstuAssets].values)
        meanValue = pandas.Series(globalMean, index=assetData.universe)

        # Loop round countries/regions
        for (regID, rmgAssets) in regionAssetMap.items():

            # Get relevant assets and data
            if len(rmgAssets) < 1:
                logging.warning('No assets for region %s', regID)
                continue

            # Now loop round sector/industry group
            for sec in sectorExposures.columns:

                # Pick out assets exposed to sector
                sectorColumn = sectorExposures.loc[rmgAssets, sec]
                sectorIds = set(sectorColumn[numpy.isfinite(sectorColumn)].index)
                goodWtIds = sectorIds.intersection(goodEstuAssets)

                # Compute mean and populate larger array
                if len(goodWtIds) > 0:
                    meanSector = ma.average(values[goodWtIds].values, axis=0, weights=assetData.marketCaps[goodWtIds].values)
                    meanValue[sectorIds] = meanSector
                    logging.debug('Mean for %d values of %s in region %d Sector %s: %.3f',
                            len(sectorIds), descriptorName, regID, sec, meanSector)
                else:
                    logging.info('No values of %s for region %d Sector %s', descriptorName, regID, sec)

        # Shrink relevant values
        logging.info('Shrinking %d values of %s', len(assetData.universe)-len(goodVals), descriptorName)
        values = (values.fillna(0.0) * scaleFactor) + ((1.0 - scaleFactor) * meanValue.fillna(0.0))
        if len(preSPAC) > 0:
            values[preSPAC] = numpy.nan
        self.log.debug('shrink_to_mean: end')
        return values

    def proxy_missing_exposures(self, modelDate, assetData, expM, modelDB, marketDB,
           factorNames=['Value', 'Growth', 'Leverage'], clip=True, sizeVec=None, kappa=5.0, excludeList=None):
        """Fill-in missing exposure values for the factors given in
        factorNames by cross-sectional regression.  For each region,
        estimation universe assets are taken, and their exposure values
        regressed against their Size and industry (GICS sector) exposures.
        Missing values are then extrapolated based on these regression
        coefficients, and trimmed to lie within [-1.5, 1.5] to prevent the
        proxies taking on extreme values.
        """
        self.log.debug('proxy_missing_exposures: begin')

        # Prepare sector exposures, will be used for missing data proxies
        roots = self.industryClassification.getClassificationRoots(modelDB)
        root = [r for r in roots if r.name=='Sectors'][0]
        sectorNames = [n.description for n in self.\
                industryClassification.getClassificationChildren(root, modelDB)]
        sectorExposures = Matrices.ExposureMatrix(assetData.universe)
        e = self.industryClassification.getExposures(modelDate,
                assetData.universe, sectorNames, modelDB, level=-(len(roots)-1))
        sectorExposures.addFactors(sectorNames, e, ExposureMatrix.IndustryFactor)

        mat = expM.toDataFrame()
        sectorMat = sectorExposures.toDataFrame()
        estu = set(assetData.estimationUniverse).intersection(set(assetData.universe))

        if sizeVec is None:
            sizeVec = mat.loc[:, 'Size']

        for fct in factorNames:
            if (not hasattr(self.styleParameters[fct], 'fillMissing')) or \
                    (self.styleParameters[fct].fillMissing is not True):
                continue

            # Determine which assets are missing data and require proxying
            values = mat.loc[:, fct]
            missingIndices = set(values[values.isnull()].index)
            self.log.info('%d/%d assets missing %s fundamental data (%d/%d ESTU)',
                        len(missingIndices), len(values), fct, len(estu.intersection(missingIndices)), len(estu))
            if len(missingIndices)==0:
                continue

            # Loop around regions
            for (regionName, asset_list) in self.exposureStandardization.factorScopes[0].getAssets(expM, modelDate):

                if len(asset_list) < 1:
                    continue

                # Find numbers of assets available
                missing_indices = set(asset_list).intersection(missingIndices)
                good_indicesUniv = set(asset_list).difference(missingIndices)
                good_indices = good_indicesUniv.intersection(estu)

                # Check that there are enough "good" assets to estimate the proxy model
                if excludeList is not None:
                    missing_indices = missing_indices.difference(excludeList)
                if len(missing_indices) < 1:
                    logging.debug('No missing values for %s/%s', regionName, fct)
                    continue
                propGood = len(good_indices) / float(len(missing_indices))
                if propGood < 0.1:
                    good_indices = good_indicesUniv.intersection(assetData.eligibleUniverse)
                    propGood = len(good_indices) / float(len(missing_indices))
                    if propGood < 0.1:
                        if len(missing_indices) > 0:
                            self.log.warning('Too few assets (%d) in %s with %s data present', len(good_indices), regionName, fct)
                        continue

                self.log.info('Running %s proxy regression for %s (%d assets)', fct, regionName, len(good_indices))

                # Check that the data available are not deficient
                good_data = values[good_indices].mask(values==0.0)
                nUniqueValues = len(numpy.unique(good_data.values)) / float(len(asset_list))
                if nUniqueValues < 0.05:
                    self.log.warning('Not enough (%.2f%%) unique exposure values for regression', nUniqueValues)
                    continue
                if len(good_data[numpy.isfinite(good_data)]) == 0:
                    self.log.warning('All non-missing values are zero, skipping')
                    continue

                # Assemble regressand, regressor matrix and weights
                good_indices = sorted(good_indices)
                missing_indices = sorted(missing_indices)
                weights = numpy.sqrt(assetData.marketCaps[good_indices]).values
                regressand = values[good_indices].values
                regressor = ma.zeros((len(good_indices), len(sectorNames) + 1))
                regressor[:,0] = sizeVec[good_indices].fillna(0.0).values
                regressor[:,1:] = sectorMat.loc[good_indices, :].fillna(0.0).values
                regressor = ma.transpose(regressor)

                # Set up necessary data to piggy back off the factor regression code
                regPar = self.expProxyCalculator.allParameters[0]
                regPar.regFactorsIdx = list(range(regressor.shape[0]))
                regPar.regFactorNames = ['Size'] + sectorNames
                regPar.reg_estu = good_indices
                regPar.regEstuIdx = list(range(len(good_indices)))
                regPar.reg_weights = list(weights)
                regPar.iReg = 1
                self.expProxyCalculator.date = modelDate
                storeTWM = regPar.thinWeightMultiplier
                storeDT = regPar.dummyThreshold

                # Perform the regression to model exposures via a cross-sectional regression
                tmpFlag = self.debuggingReporting
                self.debuggingReporting = False
                factorTypes = [ExposureMatrix.StyleFactor] + ([ExposureMatrix.IndustryFactor] * (len(regPar.regFactorsIdx)-1))
                thinFacPar = self.expProxyCalculator.processThinFactorsForRegression(
                        self, regPar, None, regressand, assetData, False,
                        factorTypes=factorTypes, nonZeroNames=[regPar.regFactorNames[0]])
                coefs = self.expProxyCalculator.calc_Factor_Specific_Returns(
                        self, regPar, thinFacPar, assetData, regressand, regressor, None, False).factorReturns
                coefs = pandas.Series(coefs, index=regPar.regFactorNames)
                self.debuggingReporting = tmpFlag

                # Restore original values 
                regPar.dummyThreshold = storeDT
                regPar.thinWeightMultiplier = storeTWM

                # Substitute proxies for missing values
                self.log.debug('Proxying %d %s exposures for %s', len(missing_indices), fct, regionName)
                for sn in sectorNames:
                    regSectorExposures = sectorMat.loc[missing_indices, sn]
                    reg_sec_indices = set(regSectorExposures[numpy.isfinite(regSectorExposures)].index)
                    if len(reg_sec_indices)==0:
                        continue
                    proxies = sizeVec[reg_sec_indices].fillna(0.0) * coefs['Size'] + coefs[sn]
                    if clip:
                        proxies = proxies.clip(-2.0, 2.0)
                    mat.loc[reg_sec_indices, fct] = proxies

        expM.data_ = Utilities.df2ma(mat.T)
        self.log.debug('proxy_missing_exposures: end')
        return expM

    def generate_style_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        """Compute multiple-descriptor style exposures
        for assets in data.universe using descriptors from the relevant
        descriptor_exposure table(s)
        """
        self.log.debug('generate_style_exposures: begin')
        descriptorExposures = Matrices.ExposureMatrix(assetData.universe)
        preSPAC = assetData.getSPACs()

        # Get list of all descriptors needed
        descriptors = []
        for f in self.DescriptorMap.keys():
            dsList = [ds for ds in self.DescriptorMap[f] if ds[-3:] != '_md']
            descriptors.extend(dsList)
        descriptors = sorted(set(descriptors))

        # Map descriptor names to their IDs
        descDict = dict(modelDB.getAllDescriptors())

        # Pull out a proxy for size
        sizeVec = self.loadDescriptors(['LnIssuerCap'], descDict, modelDate, assetData.universe,
                modelDB, assetData.getCurrencyAssetMap(), rollOver=self.rollOverDescriptors)[0].loc[:, 'LnIssuerCap']
        missingSizeIds = sizeVec[sizeVec.isnull()].index

        # Make another exception for SPACs pre-announcement date
        overLap = missingSizeIds.intersection(preSPAC)
        if len(overLap) > 0:
            missingSizeIds = missingSizeIds.difference(preSPAC)
            logging.info('%d SPACs are missing Size descriptors - we\'re cool with that', len(overLap))

        # Check for any assets missing their size descriptor (shouldn't be any)
        if len(missingSizeIds) > 0:

            missingIds = ','.join([sid.getSubIdString() for sid in missingSizeIds])
            if self.forceRun:
                logging.warning('%d missing LnIssuerCaps', len(missingSizeIds))
                logging.warning('Missing asset IDs: %s', missingIds)
            else:
                raise ValueError('%d asset(s) missing LnIssuerCaps: %s' % \
                        (len(missingSizeIds), missingIds))

        # Load the descriptor data
        descValueDict, okDescriptorCoverageMap = self.loadDescriptors(
                        descriptors, descDict, modelDate, assetData.universe, modelDB,
                        assetData.getCurrencyAssetMap(), rollOver=self.rollOverDescriptors)

        # Populate descriptor matrix
        if len(preSPAC) > 0:
            descValueDict.loc[preSPAC, :] = numpy.nan
        for ds in descriptors:
            if (ds == 'LnIssuerCap') and (len(missingSizeIds) > 0):
                descValueDict.loc[missingSizeIds, ds] = descValueDict.loc[:, ds].min(axis=None)
            if ds in descValueDict:
                descriptorExposures.addFactor(ds, descValueDict.loc[:, ds], ExposureMatrix.StyleFactor)

        # Check that each exposure has at least one descriptor with decent coverage
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        factorMap = dict(zip([f.name for f in factors], factors))
        for fct in self.DescriptorMap.keys():
            numD = [okDescriptorCoverageMap[d] for d in self.DescriptorMap[fct]]
            numD = numpy.sum(numD, axis=0)
            if not factorMap[fct].from_dt <=  modelDate <= factorMap[fct].thru_dt:
                continue
            if numD < 1 and not self.forceRun:
                raise Exception('Factor %s has no descriptors with adequate coverage' % fct)
            logging.info('Factor %s has %d/%d  descriptors with adequate coverage', fct, numD, len(self.DescriptorMap[fct]))

        # Add country factors to descriptor exposure matrix; needed for regional-relative standardization
        country_indices = exposureMatrix.getFactorIndices(ExposureMatrix.CountryFactor)
        if len(country_indices) > 0:
            countryExposures = ma.take(exposureMatrix.getMatrix(), country_indices, axis=0)
            countryNames = exposureMatrix.getFactorNames(ExposureMatrix.CountryFactor)
            descriptorExposures.addFactors(countryNames, countryExposures, ExposureMatrix.CountryFactor)

        if self.debuggingReporting:
            if 'Market_Sensitivity_104W' in descDict:
                betas = self.loadDescriptors(['Market_Sensitivity_104W'],
                        descDict, modelDate, assetData.universe, modelDB, assetData.currencyAssetMap,
                        rollOver=self.rollOverDescriptors, forceRegStruct=True)[0].loc[:, 'Market_Sensitivity_104W']
                descriptorExposures.addFactor('predBetas', betas, ExposureMatrix.StyleFactor)
                betas = modelDB.getHistoricBetaDataV3(
                        modelDate, assetData.universe, field='value', home=1, rollOverData=True, returnDF=True)
                descriptorExposures.addFactor('histBetas', betas[assetData.universe], ExposureMatrix.StyleFactor)

            descriptorExposures.dumpToFile('tmp/raw-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, dp=self.dplace, assetData=assetData)

        # Clone DR and cross-listing exposures if required
        scores = self.load_ISC_Scores(modelDate, assetData, modelDB, marketDB, returnDF=True)
        self.clone_linked_asset_descriptors(\
                modelDate, assetData, descriptorExposures, modelDB, marketDB, scores, excludeList=self.noCloneDescriptor)

        # Standardize raw descriptors for multi-descriptor factors
        self.descriptorStandardization.standardize(descriptorExposures, assetData.estimationUniverse,
                assetData.marketCaps, modelDate, eligibleEstu=assetData.eligibleUniverse, noClip_List = self.noClip_List)

        if self.debuggingReporting:
            descriptorExposures.dumpToFile('tmp/stnd-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, dp=self.dplace, assetData=assetData)

        # Save descriptor standardisation stats to the DB
        descriptorData = Utilities.Struct()
        descriptorData.descriptors = descriptors
        descriptorData.descDict = descDict
        if hasattr(descriptorExposures, 'meanDict'):
            descriptorData.meanDict = descriptorExposures.meanDict
        else:
            descriptorData.meanDict = dict()
        if hasattr(descriptorExposures, 'stdDict'):
            descriptorData.stdDict = descriptorExposures.stdDict
        else:
            descriptorData.stdDict = dict()
        if not hasattr(self, 'DescriptorWeights'):
            self.DescriptorWeights = dict()

        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        mat = descriptorExposures.getMatrix()
        for cf in self.styles:
            params = self.styleParameters[cf.description]
            if cf.description not in self.DescriptorMap:
                self.log.warning('No descriptors for factor: %s', cf.description)
                continue
            cfDescriptors = self.DescriptorMap[cf.description]
            self.log.info('Factor %s has %d descriptor(s): %s', cf.description, len(cfDescriptors), cfDescriptors)
            valueList = []
            for d in cfDescriptors:
                valueList.append(mat[descriptorExposures.getFactorIndex(d),:])
            valueList = ma.array(valueList)

            if len(valueList) > 1:
                if hasattr(params, 'descriptorWeights'):
                    weights = params.descriptorWeights
                else:
                    weights = [1.0/float(len(valueList))] * len(valueList)
                e = ma.average(valueList, axis=0, weights=weights)
            else:
                e = valueList[0,:]
            exposureMatrix.addFactor(cf.description, e, ExposureMatrix.StyleFactor)

        # Proxy raw style exposures for assets missing data
        self.proxy_missing_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB,
                         factorNames=[cf.description for cf in self.styles], sizeVec=sizeVec, excludeList=preSPAC)

        # Generate other, model-specific factor exposures
        exposureMatrix = self.generate_model_specific_exposures(
                modelDate, assetData, exposureMatrix, modelDB, marketDB)

        self.log.debug('generate_style_exposures: end')
        return descriptorData

    def generate_domestic_china_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        """Generate Domestic China local factor.
        """
        logging.info('Building Domestic China Exposures')

        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                modelDate, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
        aShares = estuCls.exclude_by_asset_type(
                assetData, includeFields=self.localChineseAssetTypes, excludeFields=None)

        values = pandas.Series(numpy.nan, index=assetData.universe)
        aShares = aShares.difference(assetData.getSPACs())
        if len(aShares) > 0:
            logging.info('Assigning Domestic China exposure to %d assets', len(aShares))
            values[aShares] = 1.0
        exposureMatrix.addFactor('Domestic China', values, ExposureMatrix.LocalFactor)

        return exposureMatrix

    def generate_cap_bucket_exposures(self, modelDate, assetData, exposureMatrix, modelDB, marketDB):
        """Generate mid/small cap factor exposures
        """
        beta = pandas.Series(numpy.nan, index=assetData.universe)

        # Cap-based style factors here
        if not hasattr(self, 'estuMap') or self.estuMap is None:
            return exposureMatrix

        # Cap bucket factors
        dateList = modelDB.getDates(self.rmg, modelDate, self.grandFatherLookBack, excludeWeekend=True)
        styleNames = [s.name for s in self.styles]
        capBucketList = [sc for sc in self.estuMap.keys() if sc in styleNames]

        # Loop round cap buckets
        for bucket in capBucketList:
            beta = pandas.Series(numpy.nan, index=assetData.universe)
            subAssets = set(self.estuMap[bucket].assets).intersection(set(assetData.universe))
            if len(subAssets) < 1:
                logging.warning('No assets in %s universe', bucket)
            else:
                # Load set of qualified assets
                qualifiedAssets = modelDB.loadESTUQualifyHistory(
                    self.rms_id, subAssets, dateList, estuInstance=self.estuMap[bucket], returnDF=True)
                qualifiedIds = set(qualifiedAssets.index).difference(assetData.getSPACs())
                qualifiedAssets = qualifiedAssets.loc[qualifiedIds, :]
                if len(qualifiedAssets.index) < 1:
                    logging.warning('No assets qualified for %s factor', bucket)
                else:
                    if oldPD:
                        qualifiedAssets1 = qualifiedAssets.mask(qualifiedAssets==0.0).sum(axis=1)
                    else:
                        qualifiedAssets1 = qualifiedAssets.mask(qualifiedAssets==0.0).sum(axis=1, min_count=1)
                    qualifiedAssets2 = qualifiedAssets1 / float(qualifiedAssets1.max(axis=None))
                    beta[subAssets] = qualifiedAssets2[subAssets]

            beta = beta.mask(beta==0.0)
            exposureMatrix.addFactor(bucket, beta, ExposureMatrix.StyleFactor)

            # Add to list of factors not to standardise
            if self.exposureStandardization.exceptionNames is None:
                self.exposureStandardization.exceptionNames = [bucket]
            else:
                self.exposureStandardization.exceptionNames.append(bucket)
                self.exposureStandardization.exceptionNames = list(set(self.exposureStandardization.exceptionNames))

        return exposureMatrix

    def generateExposureMatrix(self, modelDate, modelDB, marketDB):
        """Generates and returns the exposure matrix for the given date.
        The exposures are not inserted into the database.
        Data is accessed through the modelDB and marketDB DAOs.
        The return is a structure containing the exposure matrix
        (exposureMatrix), the universe as a list of assets (universe),
        and a list of market capitalizations (marketCaps).
        """
        self.log.debug('generateExposureMatrix: begin')

        # Get risk model universe and market caps
        # Determine home country info and flag DR-like instruments
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        assetData = AssetProcessor_V4.AssetProcessor(modelDate, modelDB, marketDB, self.getDefaultAPParameters())
        assetData.process_asset_information(self.rmg, universe=universe)

        # Get set of pre-announcement SPACs that will have null exposures
        preSPAC = assetData.getSPACs()

        # Initialise exposure matrix
        exposureMatrix = Matrices.ExposureMatrix(assetData.universe)

        # Generate eligible universe
        estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                modelDate, assetData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
        assetData.eligibleUniverse = estuCls.generate_eligible_universe(assetData, quiet=True)

        # Compute issuer-level market caps if required
        assetData.mCapDF = AssetProcessor_V4.computeTotalIssuerMarketCaps(
                modelDate, assetData.marketCaps, self.numeraire, modelDB, marketDB, debugReport=self.debuggingReporting)

        # Add country and/or currency factors to exposure matrix as required
        if self.hasCountryFactor:
            exposureMatrix = self.generate_binary_country_exposures(
                    modelDate, modelDB, marketDB, assetData, exposureMatrix, setNull=preSPAC)
        if self.hasCurrencyFactor:
            exposureMatrix = self.generate_currency_exposures(
                    modelDate, modelDB, marketDB, assetData, exposureMatrix)

        # Generate 0/1 industry exposures
        exposureMatrix = self.generate_industry_exposures(
                modelDate, modelDB, marketDB, exposureMatrix, setNull=preSPAC)
        
        # Load estimation universe
        assetData.estimationUniverse = set(self.loadEstimationUniverse(rmi, modelDB, assetData))
        
        # Create intercept factor
        if self.intercept is not None:
            beta = pandas.Series(1.0, index=assetData.universe)
            if len(preSPAC) > 0:
                beta[preSPAC] = numpy.nan
            exposureMatrix.addFactor(self.intercept.name, beta, ExposureMatrix.InterceptFactor)

        # Build all style exposures
        descriptorData = self.generate_style_exposures(
                modelDate, assetData, exposureMatrix, modelDB, marketDB)

        # Shrink some values where there is insufficient history
        expM_DF = exposureMatrix.toDataFrame()
        for st in self.styles:
            params = self.styleParameters.get(st.name, None)
            if (params is None) or (not  hasattr(params, 'shrinkValue')):
                continue
            values = expM_DF.loc[:, st.name]
            # Check and warn of missing values
            missingIds = set(values[values.isnull()].index).difference(preSPAC)
            if len(missingIds) > 0:
                missingSIDs_notnursery = missingIds.difference(assetData.nurseryUniverse)
                self.log.warning('%d assets have missing %s data', len(missingIds), st.description)
                self.log.warning('%d non-nursery assets have missing %s data', len(missingSIDs_notnursery), st.description)
                if self.debuggingReporting:
                    self.log.info('Subissues: %s', missingIds)
                else:
                    self.log.debug('Subissues: %s', missingIds)
                if (len(missingSIDs_notnursery) > 5) and not self.forceRun \
                        and (st.name not in self.allowMissingFactors):
                    assert(len(missingSIDs_notnursery)==0)

            testNew = False
            if self.regionalDescriptorStructure and testNew:
                expM_DF.loc[:, st.name] = self.shrink_to_mean(modelDate, assetData, modelDB, marketDB,
                        st.name, params.daysBack, values, onlyIPOs=False)
            else:
                expM_DF.loc[:, st.name] = self.shrink_to_mean(modelDate, assetData, modelDB, marketDB,
                        st.name, params.daysBack, values)
        exposureMatrix.data_ = Utilities.df2ma(expM_DF.T)

        # Clone DR and cross-listing exposures if required
        scores = self.load_ISC_Scores(modelDate, assetData, modelDB, marketDB, returnDF=True)
        self.group_linked_assets(modelDate, assetData, modelDB, marketDB)
        exposureMatrix = self.clone_linked_asset_exposures(modelDate, assetData, exposureMatrix, modelDB, marketDB, scores)
        
        if self.debuggingReporting:
            exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, assetData=assetData, dp=self.dplace)

        # Standardise the raw exposure matrix
        tmpDebug = self.debuggingReporting
        self.debuggingReporting = False
        self.exposureStandardization.mad_bound = 5.395918
        self.exposureStandardization.standardize(
                exposureMatrix, assetData.estimationUniverse, assetData.marketCaps, modelDate,
                eligibleEstu=assetData.eligibleUniverse, noClip_List=self.noClip_List)

        # Check for exposures to be orthogonalised
        orthogDict = dict()
        for st in self.styles:
            params = self.styleParameters[st.name]
            if hasattr(params, 'orthog'):
                if not hasattr(params, 'sqrtWt'):
                    params.sqrtWt = True
                if not hasattr(params, 'orthogCoef'):
                    params.orthogCoef = 1.0
                if params.orthog is not None and len(params.orthog) > 0:
                    orthogDict[st.name] = (params.orthog, params.orthogCoef, params.sqrtWt)

        if len(orthogDict) > 0:
            # Orthogonalise selected factors
            exposureMatrix = Utilities.partial_orthogonalisation(\
                    modelDate, assetData, exposureMatrix, modelDB, marketDB, orthogDict)

            # Re-standardise orthogonalised factor
            if self.exposureStandardization.exceptionNames is None:
                tmpExcNames = None
            else:
                tmpExcNames = list(self.exposureStandardization.exceptionNames)
            self.exposureStandardization.exceptionNames = [st.name for st in self.styles if st.name not in orthogDict]
            self.exposureStandardization.standardize(
                    exposureMatrix, assetData.estimationUniverse, assetData.marketCaps, modelDate,
                    eligibleEstu=assetData.eligibleUniverse, noClip_List=self.noClip_List)
            self.exposureStandardization.exceptionNames = tmpExcNames

        self.debuggingReporting = tmpDebug

        # Fill any missing values with (standardised) zero
        for st in self.styles:
            params = self.styleParameters[st.name]
            # 'fillWithZero' covers items like Dividend Yield, where a large number of observations are genuinely missing
            # 'fillMissing' is a failsafe for exposures that shouldn't normally have any missing values,
            # but given the vagaries of global data, may have some from time to time
            if (hasattr(params, 'fillWithZero') and (params.fillWithZero is True)) or \
                    (hasattr(params, 'fillMissing') and (params.fillMissing is True)):
                exposureMatrix = self.fill_with_standardised_zero(modelDate, exposureMatrix, st, preSPAC)

        if self.debuggingReporting:
            exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, assetData=assetData, dp=self.dplace)

        # Check for exposures with all missing values
        expM_DF = exposureMatrix.toDataFrame()
        for st in self.styles:
            values = exposureMatrix.toDataFrame().loc[:, st.name]
            missingIds = set(values[values.isnull()].index)
            if (len(missingIds) > 0) and (st.name not in self.dummyStyleList):
                self.log.warning('Style factor %s has %d missing exposures', st.name, len(missingIds))
            if len(missingIds) == len(values):
                self.log.error('All %s values are missing', st.name)
                if not self.forceRun:
                    assert (len(missingIds) < len(values))
            
        self.log.info('Generated exposure matrix for %d assets and %d factors',
                len(exposureMatrix.assets_), len(exposureMatrix.factors_))
        self.log.debug('generateExposureMatrix: end')
        return [exposureMatrix, descriptorData]
    
    def fill_with_standardised_zero(self, modelDate, exposureMatrix, factor, keepNull):
        """Fill missing exposure values with (standardised) zero
        """
        expM_DF = exposureMatrix.toDataFrame()
        for scope in self.exposureStandardization.factorScopes:
            if factor.name not in scope.factorNames:
                continue
            for (bucket, sidList) in scope.getAssets(exposureMatrix, modelDate):
                values = expM_DF.loc[sidList, factor.name]
                nMissing = set(values[values.isnull()].index).difference(keepNull)
                if len(nMissing) > 0:
                    denom = ma.filled(exposureMatrix.stdDict[bucket][factor.name], 0.0)
                    if abs(denom) > 1.0e-6:
                        fillValue = (0.0 - exposureMatrix.meanDict[bucket][factor.name]) / denom
                        expM_DF.loc[nMissing, factor.name] = fillValue
                        logging.info('Filling %d missing values for %s with standardised zero: %.2f for region %s',
                                len(nMissing), factor.name, fillValue, bucket)
                    else:
                        logging.warning('Zero/missing standard deviation %s for %s for region %s',
                                exposureMatrix.stdDict[bucket][factor.name], factor.name, bucket)
        exposureMatrix.data_ = Utilities.df2ma(expM_DF.T)
        return exposureMatrix

    def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate, buildFMPs=False, internalRun=False, cointTest=False, weeklyRun=False):
        """Generates the factor and specific returns for the given
        date.  Assumes that the factor exposures for the previous
        trading day exist as those will be used for the exposures.
        Returns a Struct with factorReturns, specificReturns, exposureMatrix,
        regressionStatistics, and adjRsquared.  The returns are 
        arrays matching the factor and assets in the exposure matrix.
        exposureMatrix is an ExposureMatrix object.  The regression
        coefficients is a two-dimensional masked array
        where the first dimension is the number of factors used in the
        regression, and the second is the number of statistics for
        each factor that are stored in the array.
        """
        # Testing parameters for FMPs
        testFMPs = False
        nextTradDate = None

        # Important regression parameters
        applyRT = False
        if buildFMPs:
            logging.info('Generating FMPs')
            # Set up parameters for FMP calculation
            prevDate = modelDate
            if testFMPs:
                futureDates = modelDB.getDateRange(self.rmg, modelDate,
                        modelDate+datetime.timedelta(20), excludeWeekend=True)
                nextTradDate = futureDates[1]

            if not hasattr(self, 'fmpCalculator'):
                logging.warning('No FMP parameters set up, skipping')
                return None
            rcClass = self.fmpCalculator
        else:
            # Set up parameters for factor returns regression
            if weeklyRun:
                startDate = modelDate - datetime.timedelta(7)
                dateList = modelDB.getDateRange(
                        self.rmg, startDate, modelDate, excludeWeekend=True)
            else:
                dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
            if len(dateList) < 2:
                raise LookupError(
                    'No previous trading day for %s' %  str(modelDate))
            prevDate = dateList[0]

            if internalRun:
                logging.info('Generating internal factor returns')
                # Regression for internal factor returns
                if not hasattr(self, 'internalCalculator'):
                    logging.warning('No internal factor return parameters set up, skipping')
                    return None
                rcClass = self.internalCalculator
                applyRT = self.applyRT2US
            elif weeklyRun:
                logging.info('Generating weekly factor returns from %s to %s',
                        prevDate, modelDate)
                # Regression for weekly factor returns
                if not hasattr(self, 'weeklyCalculator'):
                    logging.warning('No weekly factor return parameters set up, skipping')
                    return None
                rcClass = self.weeklyCalculator
                applyRT = False
            else:
                # Regression for public factor returns
                logging.info('Generating external (public) factor returns')
                rcClass = self.returnCalculator

        # Get exposure matrix for previous trading day
        rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
        if rmi == None:
            raise LookupError(
                'no risk model instance for %s' % str(prevDate))
        if not rmi.has_exposures:
            raise LookupError(
                'no exposures in risk model instance for %s' % str(prevDate))
        self.setFactorsForDate(prevDate, modelDB)

        # Determine home country info and flag DR-like instruments
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        data = AssetProcessor.process_asset_information(
                prevDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=self.coverageMultiCountry, numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun, nurseryRMGList=self.nurseryRMGs,
                rmgOverride=self.rmgOverride)

        # Load previous day's exposure matrix
        expM = self.loadExposureMatrix(rmi, modelDB, addExtraCountries=True, assetList=data.universe)
        prevFactors = self.factors + self.nurseryCountries
        prevSubFactors = modelDB.getSubFactorsForDate(prevDate, prevFactors)
        nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in prevSubFactors]) 

        # Get map of current day's factor IDs
        self.setFactorsForDate(modelDate, modelDB)
        allFactors = self.factors + self.nurseryCountries
        subFactors = modelDB.getSubFactorsForDate(modelDate, allFactors)
        subFactorIDIdxMap = dict([(subFactors[i].subFactorID, i) for i in range(len(subFactors))])
        deadFactorNames = [s.factor.name for s in prevSubFactors if s not in subFactors]
        deadFactorIdx = [expM.getFactorIndex(n) for n in deadFactorNames]
        newFactors = [s for s in subFactors if s not in prevSubFactors]
        newFactorNames = [s.factor.name for s in newFactors]
        nameSubIDMap.update(dict([(s.factor.name, s.subFactorID) for s in newFactors]))
        if len(deadFactorIdx) > 0:
            self.log.warning('Dropped factors %s on %s', deadFactorNames, modelDate)
        if len(newFactorNames) > 0:
            self.log.warning('Adding factors %s on %s', newFactorNames, modelDate)
        # Get main estimation universe for previous day
        estu = self.loadEstimationUniverse(rmi, modelDB, data)
        if hasattr(self, 'estuMap') and self.estuMap is not None:
            if 'nursery' in self.estuMap:
                logging.info('Adding %d nursery assets to main estimation universe',
                        len(self.estuMap['nursery'].assets))
                estu = estu + self.estuMap['nursery'].assets
                self.estuMap['main'].assets = estu
                logging.info('Main estimation universe now %d assets', len(estu))
        estuIdx = [data.assetIdxMap[sid] for sid in estu]
        if hasattr(self, 'estuMap') and self.estuMap is not None and 'main' in self.estuMap.keys():
            self.estuMap['main'].assetIdx = estuIdx
            self.estuMap['main'].name = 'main'
        estimationUniverseIdx = estuIdx
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)

        # Domestic China Handling
        if 'ChinaA' in self.estuMap or 'ChinaA' in [self.estuMap[x].name if hasattr(self.estuMap[x], 'name') else None for x in self.estuMap.keys()]:
             chinaA_key = 'ChinaA' if 'ChinaA' in self.estuMap else dict([(self.estuMap[x].name,x) for x in self.estuMap.keys()])['ChinaA']
             dom_china_assets = self.estuMap[chinaA_key].assets
             dom_china_assets = list(set(dom_china_assets).intersection(data.assetIdxMap.keys()))
             if len(dom_china_assets) < 10:
                 dom_china_assets = []
             self.estuMap[chinaA_key].assets = dom_china_assets

        # Load asset returns
        if self.isRegionalModel():
            if cointTest:
                historyLength = self.cointHistory
                modelDB.notTradedIndCache = ModelDB.TimeSeriesCache(int(1.5*historyLength))
                assetReturnMatrix = self.assetReturnHistoryLoader(
                        data, historyLength, modelDate, modelDB, marketDB, cointTest=True)
                return
            elif internalRun:
                assetReturnMatrix = self.assetReturnHistoryLoader(
                        data, 10, modelDate, modelDB, marketDB, loadOnly=True,
                        applyRT=applyRT, fixNonTradingDays=True)
                missingReturnsMask = assetReturnMatrix.missingFlag
                zeroReturnsMask = assetReturnMatrix.zeroFlag
            elif weeklyRun:

                # Load daily returns and convert to weekly
                self.returnHistory = 10
                data = self.build_excess_return_history(data, modelDate, modelDB, marketDB,
                        loadOnly=True, applyRT=False, fixNonTradingDays=False)
                weeklyExcessReturns = ProcessReturns.compute_compound_returns_v4(
                        ma.filled(data.returns.data, 0.0), data.returns.dates, [prevDate, modelDate])[0]
                # Mask missing weekly returns
                nonMissingData = ma.getmaskarray(data.returns.data)==0
                weeklyDataMask = ProcessReturns.compute_compound_returns_v4(
                        nonMissingData.astype(int), data.returns.dates, [prevDate, modelDate])[0]
                weeklyExcessReturns = ma.masked_where(weeklyDataMask==0, weeklyExcessReturns)

                # Load currency returns in now rather than later
                crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
                assert (crmi is not None)
                currencyFactors = self.currencyModel.getCurrencyFactors()
                currencySubFactors = modelDB.getSubFactorsForDate(crmi.date, currencyFactors)
                currencyReturnsHistory = modelDB.loadFactorReturnsHistory(
                        crmi.rms_id, currencySubFactors, data.returns.dates)
                # Convert to weekly returns
                currencyReturns = ProcessReturns.compute_compound_returns_v4(
                        currencyReturnsHistory.data, currencyReturnsHistory.dates, [prevDate, modelDate])[0]
                # Mask missing weekly returns
                nonMissingData = ma.getmaskarray(currencyReturnsHistory.data)==0
                weeklyDataMask = ProcessReturns.compute_compound_returns_v4(
                        nonMissingData.astype(int), data.returns.dates, [prevDate, modelDate])[0]
                currencyReturns  = ma.masked_where(weeklyDataMask==0, currencyReturns)

                # Pull out latest weekly returns
                assetReturnMatrix = data.returns
                assetReturnMatrix.data = weeklyExcessReturns[:,-1][:,numpy.newaxis]
                assetReturnMatrix.dates = [modelDate]
                missingReturnsMask = ma.getmaskarray(assetReturnMatrix.data)
                currencyReturns = currencyReturns[:,-1]
            else:
                assetReturnMatrix = self.assetReturnHistoryLoader(
                        data, 2, modelDate, modelDB, marketDB, loadOnly=True,
                        applyRT=applyRT, fixNonTradingDays=False)
                missingReturnsMask = assetReturnMatrix.missingFlag
                zeroReturnsMask = assetReturnMatrix.zeroFlag
        else:
            assetReturnMatrix = self.assetReturnHistoryLoader(
                    data, 1, modelDate, modelDB, marketDB, loadOnly=True, applyRT=applyRT)
            missingReturnsMask = assetReturnMatrix.missingFlag
            zeroReturnsMask = assetReturnMatrix.zeroFlag
        missingReturnsMask = missingReturnsMask[:,-1]
        if not weeklyRun:
            assetReturnMatrix.data = assetReturnMatrix.data[:,-1][:,numpy.newaxis]
            assetReturnMatrix.dates = [assetReturnMatrix.dates[-1]]
            zeroReturnsMask = zeroReturnsMask[:,-1]

        # Do some checking on missing and zero returns
        missingReturnsIdx = numpy.flatnonzero(missingReturnsMask)
        logging.info('%d out of %d genuine missing returns in total', len(missingReturnsIdx), len(missingReturnsMask))
        zeroReturnsIdx = numpy.flatnonzero(zeroReturnsMask)
        logging.info('%d out of %d zero returns in total', len(zeroReturnsIdx), len(zeroReturnsMask))
        stillMissingReturnsIdx = numpy.flatnonzero(ma.getmaskarray(assetReturnMatrix.data))
        if len(stillMissingReturnsIdx) > 0:
            logging.info('%d missing total returns filled with zero', len(stillMissingReturnsIdx))

        # Do some checking on estimation universe returns
        suspectDay = False
        badRetList = list()
        # Report on missing returns
        missingESTURets = numpy.flatnonzero(ma.take(missingReturnsMask, estuIdx, axis=0))
        badRetList.extend(numpy.take(estu, missingESTURets, axis=0))
        propnBadRets = len(missingESTURets) / float(len(estuIdx))
        self.log.info('%.1f%% of %d main ESTU original returns missing', 100.0*propnBadRets, len(estuIdx))
        # Report on zero returns
        zeroESTURets = numpy.flatnonzero(ma.take(zeroReturnsMask, estuIdx, axis=0))
        badRetList.extend(numpy.take(estu, zeroESTURets, axis=0))
        propnBadRets = len(zeroESTURets) / float(len(estuIdx))
        self.log.info('%.1f%% of %d main ESTU final returns zero', 100.0*propnBadRets, len(estuIdx))
        badRetList = list(set(badRetList))
        logging.info('%d out of %d estu assets missing/zero', len(badRetList), len(estuIdx))
        if self.downWeightMissingReturns and internalRun:
            self.badRetList = badRetList
            logging.info('Downweighting bad return assets is on.')

        # If too many of both, set t-stats to be nuked
        propnBadRets = (len(zeroESTURets) + len(missingESTURets)) / float(len(estuIdx))
        if propnBadRets > 0.5:
            logging.warning('************************* More than 50% of actual returns missing...')
            logging.warning('************************* ... Regression statistics suspect and will be nuked')
            suspectDay = True

        # Compute excess returns
        assetReturnMatrix.data = ma.filled(assetReturnMatrix.data, 0.0)
        if not weeklyRun:
            (assetReturnMatrix, rfr) = self.computeExcessReturns(modelDate, 
                                    assetReturnMatrix, modelDB, marketDB, data.drCurrData)
            if self.debuggingReporting:
                for i in range(len(rfr.assets)):
                    if rfr.data[i,0] is not ma.masked:
                        self.log.info('Using risk-free rate of %f%% for %s', rfr.data[i,0] * 100.0, rfr.assets[i])
        excessReturns = assetReturnMatrix.data[:,0]
        
        # FMP testing stuff
        if nextTradDate is not None:
            returnsProcessor = ProcessReturns.assetReturnsProcessor(
                    self.rmg, data.universe, data.rmgAssetMap,
                    data.tradingRmgAssetMap, data.assetTypeDict,
                    numeraire_id=self.numeraire.currency_id, tradingCurrency_id=self.numeraire.currency_id,
                    returnsTimingID=self.returnsTimingId, debuggingReporting=self.debuggingReporting,
                    gicsDate=self.gicsDate, estu=estimationUniverseIdx, boT=self.firstReturnDate,
                    simpleProxyRetTol=self.simpleProxyRetTol)
            futureReturnMatrix = returnsProcessor.process_returns_history(
                    nextTradDate, 1, modelDB, marketDB,
                    drCurrMap=data.drCurrData, loadOnly=True,
                    applyRT=False, trimData=False)
            futureReturnMatrix.data = futureReturnMatrix.data[:,-1][:,numpy.newaxis]
            futureReturnMatrix.dates = [futureReturnMatrix.dates[-1]]
            futureReturnMatrix.data = ma.filled(futureReturnMatrix.data, 0.0)
                
            (futureReturnMatrix, rfr) = self.computeExcessReturns(nextTradDate,
                    futureReturnMatrix, modelDB, marketDB, data.drCurrData)
            futureExcessReturns = futureReturnMatrix.data[:,0]

        # Report on markets with non-trading day or all missing returns
        # Such will have their returns replaced either with zero or a proxy
        nukeTStatList = []
        rmgHolidayList = []
        totalESTUMarketCaps = ma.sum(ma.take(data.marketCaps, estimationUniverseIdx, axis=0), axis=None)
        tradingCaps = 0.0
        for r in self.rmg:

            # Pull out assets for each RMG
            if r.rmg_id not in data.rmgAssetMap:
                rmg_indices = []
            else:
                rmg_indices = [data.assetIdxMap[n] for n in \
                                data.rmgAssetMap[r.rmg_id].intersection(estu)]
            rmg_returns = ma.take(excessReturns, rmg_indices, axis=0)

            # Get missing returns (before any proxying) and calendar dates
            noOriginalReturns = numpy.sum(ma.take(missingReturnsMask, rmg_indices, axis=0), axis=None)
            rmgCalendarList = modelDB.getDateRange(r, assetReturnMatrix.dates[0], assetReturnMatrix.dates[-1])

            if noOriginalReturns >= 0.95 * len(rmg_returns) or modelDate not in rmgCalendarList:
                # Do some reporting and manipulating for NTD markets
                nukeTStatList.append(r.description)
                rmgHolidayList.append(r)
                rmgMissingIdx = list(set(stillMissingReturnsIdx).intersection(set(rmg_indices)))
                if len(rmgMissingIdx) > 0:
                    self.log.info('Non-trading day for %s, %d/%d returns missing',
                                r.description, noOriginalReturns, len(rmg_returns))
                else:
                    self.log.info('Non-trading day for %s, %d/%d returns imputed',
                            r.description, noOriginalReturns, len(rmg_returns))
            else:
                rmg_caps = ma.sum(ma.take(data.marketCaps, rmg_indices, axis=0), axis=None)
                tradingCaps += rmg_caps

        if internalRun:
            rmgHolidayList = None
        # Report on % of market trading today
        pcttrade = tradingCaps / totalESTUMarketCaps
        logging.info('Proportion of total ESTU market trading: %.2f', pcttrade)

        # Get industry asset buckets
        data.industryAssetMap = dict()
        for idx in expM.getFactorIndices(ExposureMatrix.IndustryFactor):
            assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
            data.industryAssetMap[idx] = numpy.take(data.universe, assetsIdx, axis=0)

        # Get indices of factors that we don't want in the regression
        if self.hasCurrencyFactor:
            currencyFactorsIdx = expM.getFactorIndices(ExposureMatrix.CurrencyFactor)
            excludeFactorsIdx = list(set(deadFactorIdx + currencyFactorsIdx))
        else:
            excludeFactorsIdx = deadFactorIdx

        # Remove any remaining empty style factors
        for idx in expM.getFactorIndices(ExposureMatrix.StyleFactor):
            assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
            if len(assetsIdx) == 0:
                self.log.warning('100%% empty factor, excluded from all regressions: %s', expM.getFactorNames()[idx])
                excludeFactorsIdx.append(idx)

        # Call nested regression routine
        returnData = rcClass.run_factor_regressions(
                self, rcClass, prevDate, excessReturns, expM, estu, data,
                excludeFactorsIdx, modelDB, marketDB, applyRT=applyRT, fmpRun=buildFMPs,
                rmgHolidayList=rmgHolidayList)
        
        # Map specific returns for cloned assets
        returnData.specificReturns = ma.masked_where(missingReturnsMask, returnData.specificReturns)
        if len(data.hardCloneMap) > 0:
            cloneList = set(data.hardCloneMap.keys()).intersection(set(data.universe))
            for sid in cloneList:
                if data.hardCloneMap[sid] in data.universe:
                    returnData.specificReturns[data.assetIdxMap[sid]] = returnData.specificReturns\
                            [data.assetIdxMap[data.hardCloneMap[sid]]]

        # Store regression results
        factorReturns = Matrices.allMasked((len(allFactors),))
        regressionStatistics = Matrices.allMasked((len(allFactors), 4))
        factorNames = [None]*len(allFactors) 
        for (fName, ret) in returnData.factorReturnsMap.items():
            idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
            if idx is not None:
                factorReturns[idx] = ret
                factorNames[idx] = fName
                if (not suspectDay) and (fName not in nukeTStatList):
                    regressionStatistics[idx,:] = returnData.regStatsMap[fName]
                else:
                    regressionStatistics[idx,-1] = returnData.regStatsMap[fName][-1]
        for fName in newFactorNames:
            idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
            if idx is not None:
                factorReturns[idx] = 0.
                factorNames[idx] = fName

        if not internalRun and not buildFMPs:
            # Calculate Variance Inflation Factors for each style factor regressed on other style factors.
            self.VIF = rcClass.VIF
        
        result = Utilities.Struct()
        result.universe = data.universe
        result.factorReturns = factorReturns
        result.factorNames = factorNames
        result.specificReturns = returnData.specificReturns
        result.exposureMatrix = expM
        result.regressionStatistics = regressionStatistics
        if suspectDay:
            logging.warning('************************* Too few assets trading: r-square will be set to missing')
            result.adjRsquared = None
        else:
            result.adjRsquared = returnData.anova.calc_adj_rsquared()
        result.pcttrade = pcttrade
        result.regression_ESTU = list(zip([result.universe[i] for i in returnData.anova.estU_],
                returnData.anova.weights_ / numpy.sum(returnData.anova.weights_)))
        result.VIF = self.VIF

        # Process robust weights
        newRWtMap = dict()
        sid2StringMap = dict([(sid if isinstance(sid, str) else sid.getSubIDString(), sid) for sid in data.universe])
        for (iReg, rWtMap) in returnData.robustWeightMap.items():
            tmpMap = dict()
            for (sidString, rWt) in rWtMap.items():
                if sidString in sid2StringMap:
                    tmpMap[sid2StringMap[sidString]] = rWt
            newRWtMap[iReg] = tmpMap
        result.robustWeightMap = newRWtMap

        # Process FMPs
        newFMPMap = dict()
        for (fName, fmpMap) in returnData.fmpMap.items():
            tmpMap = dict()
            for (sidString, fmp) in fmpMap.items():
                if sidString in sid2StringMap:
                    tmpMap[sid2StringMap[sidString]] = fmp
            newFMPMap[nameSubIDMap[fName]] = tmpMap
        result.fmpMap = newFMPMap

        # Report non-trading markets and set factor return to zero
        allFactorNames = expM.getFactorNames()
        # Pull in currency factor returns from currency model
        if self.hasCurrencyFactor:
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
            assert (crmi is not None)
            if not weeklyRun:
                (currencyReturns, currencyFactors) = \
                        self.currencyModel.loadCurrencyFactorReturns(crmi, modelDB)
            currSubFactors = modelDB.getSubFactorsForDate(modelDate, currencyFactors)
            currSubIDIdxMap = dict([(currSubFactors[i].subFactorID, i) \
                                    for i in range(len(currSubFactors))])
            self.log.info('loaded %d currencies from currency model', len(currencyFactors))

            # Lookup currency factor returns from currency model
            currencyFactors = set(self.currencies)
            for (i,j) in subFactorIDIdxMap.items():
                cidx = currSubIDIdxMap.get(i, None)
                if cidx is None:
                    if allFactors[j] in currencyFactors:
                        self.log.warning('Missing currency factor return for %s', allFactors[j].name)
                        value = 0.0
                    else:
                        continue
                else:
                    value = currencyReturns[cidx]
                result.factorReturns[j] = value

        constrComp = Utilities.Struct()
        constrComp.ccDict = returnData.ccMap
        constrComp.ccXDict = returnData.ccXMap

        if nextTradDate is not None:
            self.regressionReporting(futureExcessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
                            modelDate, buildFMPs=buildFMPs, constrComp=constrComp, specificRets=result.specificReturns)
        else:
            self.regressionReporting(excessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
                            modelDate, buildFMPs=buildFMPs, constrComp=constrComp, specificRets=result.specificReturns)

        if self.debuggingReporting:
            for (i,sid) in enumerate(data.universe):
                if abs(returnData.specificReturns[i]) > 1.5:
                    self.log.info('Large specific return for: %s, ret: %.8f', sid, returnData.specificReturns[i])
        return result
    
class StatisticalModel(FactorRiskModel):
    """Statistical factor model
    """
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        # No style or industry factors
        self.styles = []
        self.industries = []
        self.estuMap = None
        self.VIF = None

        # Report on important parameters
        logging.info('Using legacy market cap dates: %s', self.legacyMCapDates)
        logging.info('Using new descriptor structure: %s', self.regionalDescriptorStructure)
        logging.info('Using two regression structure: %s', self.twoRegressionStructure)
        logging.info('GICS date used: %s', self.gicsDate)
        logging.info('Earliest returns date: %s', self.firstReturnDate)
        if hasattr(self, 'coverageMultiCountry'):
            logging.info('model coverageMultiCountry: %s', self.coverageMultiCountry)
        if hasattr(self, 'hasCountryFactor'):
            logging.info('model hasCountryFactor: %s', self.hasCountryFactor)
        if hasattr(self, 'hasCurrencyFactor'):
            logging.info('model hasCurrencyFactor: %s', self.hasCurrencyFactor)
        if hasattr(self, 'applyRT2US'):
            logging.info('model applyRT2US: %s', self.applyRT2US)
        if hasattr(self, 'hasCountryFactor') and hasattr(self, 'hasCurrencyFactor'):
            logging.info('Regional model: %s', self.isRegionalModel())
        logging.info('Using bucketed MAD: %s', self.useBucketedMAD)
        logging.info('Allowing ETFs: %s', self.allowETFs)

    def isStatModel(self):
        return True

    def isCurrencyModel(self):
        return False

    def setFactorsForDate(self, date, modelDB):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        self.setBaseModelForDate(date)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])

        # Set up estimation universe parameters
        self.estuMap = modelDB.getEstuMappingTable(self.rms_id)
        if self.estuMap is None:
            logging.error('No estimation universe mapping defined')
            assert(self.estuMap is not None)
        logging.info('Estimation universe structure: %d estus', len(self.estuMap))

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)

        # Setup industry classification
        chngDate = list(self.industrySchemeDict.keys())[0]
        chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
        self.industryClassification = self.industrySchemeDict[chngDates[-1]]
        self.log.debug('Using %s classification scheme', self.industryClassification.name)

        if self.hasCurrencyFactor:
            # Create currency factors
            allRMG = modelDB.getAllRiskModelGroups(inModels=True)
            for rmg in allRMG:
                rmg.setRMGInfoForDate(date)
            currencies = [ModelFactor(f, None) for f in set([r.currency_code for r in allRMG])]
            currencies.extend([ModelFactor('EUR', 'Euro')])
            currencies = sorted(set([f for f in currencies if f.name in self.nameFactorMap]))
        else:
            currencies = []
        allFactors = self.blind + currencies
        for f in allFactors:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
        self.currencies = [c for c in currencies if c.isLive(date)]

        # Set up dicts
        self.factors = [f for f in allFactors if f.isLive(date)]
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.validateFactorStructure(date, warnOnly=self.variableStyles)
        self.allFactors = allFactors
    
    def regionalAPCA(self, specificReturns, expMatrix, data, factorReturns, modelDB): 

        # Bucket assets into regions
        totalEstu = [data.universe[idx] for idx in data.originalEstimationUniverseIdx]
        regionAssetMap = dict()
        for r in self.rmg:
            rmg_assets = data.rmgAssetMap[r.rmg_id]
            if r.region_id not in regionAssetMap:
                regionAssetMap[r.region_id] = list()
            regionAssetMap[r.region_id].extend(rmg_assets)

        # Loop round regions
        for reg in regionAssetMap.keys():

            # Get list of subissues for region
            localRegion = modelDB.getRiskModelRegion(reg)
            subSetIds = regionAssetMap[reg]
            subSetIdx = [data.assetIdxMap[sid] for sid in subSetIds]

            # Set up regional asset returns
            regReturns = Matrices.TimeSeriesMatrix(subSetIds, data.returns.dates)
            regReturns.data = ma.take(specificReturns, subSetIdx, axis=0)
            regReturns.missingFlag = ma.take(data.returns.missingFlag, subSetIdx, axis=0)

            # Get regional estu
            regAssetIdxMap = dict(zip(subSetIds, list(range(len(subSetIds)))))
            regEstu = set(totalEstu).intersection(set(subSetIds))
            regEstuIdx = [regAssetIdxMap[sid] for sid in regEstu]

            # Do APCA on residual returns for region
            logging.info('****************************************************************')
            logging.info('Performing APCA for region %d (%s), Factors: %d, Assets: %d',
                    reg, localRegion.name, self.regionFactorMap[localRegion.name], len(subSetIds))
            self.returnCalculator.numFactors = self.regionFactorMap[localRegion.name]
            (regExpMatrix, regFactorReturns, regSpecificReturns, regRegressANOVA, regPctgVar) = \
                    self.returnCalculator.calc_ExposuresAndReturns(regReturns, regEstuIdx, T=self.pcaHistory, flexibleOverride=True)

            # Reconstruct exposure matrix
            fullMatrix = Matrices.allMasked((expMatrix.shape[0], self.regionFactorMap[localRegion.name]))
            jDim = regExpMatrix.shape[1]
            for (idx, sidIdx) in enumerate(subSetIdx):
                fullMatrix[sidIdx, 0:jDim] = regExpMatrix[idx, :]
            expMatrix = ma.concatenate([expMatrix, fullMatrix], axis=1)

            # Make sure factor returns are large enough
            fullFactorMatrix = numpy.zeros((self.regionFactorMap[localRegion.name], factorReturns.shape[1]), float)
            iDim = regFactorReturns.shape[0]
            fullFactorMatrix[0:iDim, :] = Utilities.screen_data(regFactorReturns, fill=True)

            # Append factor returns
            factorReturns = numpy.concatenate([factorReturns, fullFactorMatrix], axis=0)
        return expMatrix, factorReturns

    def sectorAPCA(self, date, specificReturns, expMatrix, data, factorReturns, modelDB):

        # Pull out sector mapping for model's GICS scheme
        totalEstu = [data.universe[idx] for idx in data.originalEstimationUniverseIdx]
        sectorMatrix, sectorList = modelDB.getGICSExposures(date, data.universe, level='Sectors', clsDate=self.gicsDate)
        sectorMatrix = ma.masked_where(sectorMatrix==0.0, sectorMatrix)

        # Loop round sectors
        for isec, sc in enumerate(sectorList):

            # Get subset of assets for sector
            sectorAssetIdx = numpy.flatnonzero(ma.getmaskarray(sectorMatrix[:,isec])==0)
            sectorAssets = numpy.take(data.universe, sectorAssetIdx, axis=0)

            # Set up sector asset returns
            secReturns = Matrices.TimeSeriesMatrix(sectorAssets, data.returns.dates)
            secReturns.data = ma.take(specificReturns, sectorAssetIdx, axis=0)
            secReturns.missingFlag = ma.take(data.returns.missingFlag, sectorAssetIdx, axis=0)
            secReturns.preIPOFlag = ma.take(data.returns.preIPOFlag, sectorAssetIdx, axis=0)

            # Get regional estu
            secAssetIdxMap = dict(itertools.izip(sectorAssets, range(len(sectorAssets))))
            secEstu = set(totalEstu).intersection(set(sectorAssets))
            secEstuIdx = [secAssetIdxMap[sid] for sid in secEstu]

            # Do APCA on residual returns for sector
            logging.info('****************************************************************')
            logging.info('Performing APCA for sector %s, Factors: %d, Assets: %d',
                    sc, self.sectorFactorMap[sc], len(sectorAssets))
            self.returnCalculator.numFactors = self.sectorFactorMap[sc]
            (secExpMatrix, secFactorReturns, secSpecificReturns, secRegressANOVA, secPctgVar) = \
                    self.returnCalculator.calc_ExposuresAndReturns(secReturns, secEstuIdx, T=self.pcaHistory, flexibleOverride=True)

            # Reconstruct exposure matrix
            fullMatrix = Matrices.allMasked((expMatrix.shape[0], self.sectorFactorMap[sc]))
            jDim = secExpMatrix.shape[1]
            for (idx, sidIdx) in enumerate(sectorAssetIdx):
                fullMatrix[sidIdx, 0:jDim] = secExpMatrix[idx, :]
            expMatrix = ma.concatenate([expMatrix, fullMatrix], axis=1)

            # Make sure factor returns are large enough
            fullFactorMatrix = numpy.zeros((self.sectorFactorMap[sc], factorReturns.shape[1]), float)
            iDim = secFactorReturns.shape[0]
            fullFactorMatrix[0:iDim, :] = Utilities.screen_data(secFactorReturns, fill=True)

            # Append factor returns
            factorReturns = numpy.concatenate([factorReturns, fullFactorMatrix], axis=0)
        return expMatrix, factorReturns

    def generateStatisticalModel(self, modelDate, modelDB, marketDB):
        """Compute statistical factor exposures and returns
        Then combine returns with currency returns and build the
        composite covariance matrix
        """

        # Set up covariance parameters
        (self.minFCovObs, self.maxFCovObs) = \
                (max(self.fvParameters.minObs, self.fcParameters.minObs),
                 max(self.fvParameters.maxObs, self.fcParameters.maxObs))
        self.maxSRiskObs = self.srParameters.maxObs
        self.minSRiskObs = self.srParameters.minObs
        self.returnHistory = self.maxFCovObs
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
 
        # Determine home country info and flag DR-like instruments
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = sorted(modelDB.getRiskModelInstanceUniverse(rmi))
        assetData = AssetProcessor_V4.AssetProcessor(\
                modelDate, modelDB, marketDB, self.getDefaultAPParameters(useNursery=False))
        assetData.process_asset_information(self.rmg, universe=universe)

        # Get list of assets not to proxy
        assetTypeMap = pandas.Series(assetData.getAssetType())
        noProxyReturnsList = set(assetTypeMap[assetTypeMap.isin(self.noProxyTypes)].index)
        exSpacs = AssetProcessor_V4.sort_spac_assets(\
                modelDate, assetData.universe, modelDB, marketDB, returnExSpac=True)
        noProxyReturnsList = noProxyReturnsList.difference(exSpacs)
        if len(noProxyReturnsList) > 0:
            logging.info('%d assets of type %s excluded from proxying', len(noProxyReturnsList), self.noProxyTypes)

        # Load estimation universe from previous day (so it's the same as fundamental model)
        dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
        if len(dateList) < 2:
            raise LookupError(
                    'No previous trading day for %s' %  str(modelDate))
        prevDate = dateList[0]
        prev_rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
        if prev_rmi is None:
            modelStartDate = [ms.from_dt for ms in modelDB.getModelSeries(self.rm_id) \
                                    if ms.rms_id==self.rms_id][0]
            assert modelDate == modelStartDate
            estu = self.loadEstimationUniverse(rmi, modelDB, assetData)
        else:
            estu = self.loadEstimationUniverse(prev_rmi, modelDB, assetData)

        # Temporary back-compatibility fix
        assetData.assetTypeDict = assetData.getAssetType()
        assetData.currencyAssetMap = assetData.getCurrencyAssetMap()

        # Compute excess returns
        if self.isRegionalModel():
            # Regional returns loader:
            # Uses only basic proxy to fill missing values
            # Applies returns-timing to align with US market
            if hasattr(self, 'statModel21Settings') and self.statModel21Settings:
                # In case we want to approximately replicate 2.1 model settings 
                returnsData = self.build_excess_return_history(
                        assetData, modelDate, modelDB, marketDB,
                        loadOnly=True, applyRT=False, fixNonTradingDays=False, returnDF=True)
            else:
                returnsData = self.build_excess_return_history(
                        assetData, modelDate, modelDB, marketDB,
                        loadOnly=True, applyRT=True, fixNonTradingDays=True, returnDF=True)
        else:
            # SCM returns loader
            # Fills in missing returns with a proxy value
            # No returns-timing applied (other than alignment of foreign listings to home market)
            returnsData = self.build_excess_return_history(
                    assetData, modelDate, modelDB, marketDB, loadOnly=False,
                    applyRT=self.applyRT2US, fixNonTradingDays=False, returnDF=True)

        # Trim estimation universe to include only assets with returnTol % of returns extant
        if hasattr(self, 'statModelEstuTol'):
            returnTol = self.statModelEstuTol
            logging.info('Applying less aggressive estu filtering, tolerance: %.2f', returnTol)
            goodReturnsFlag = returnsData.nonMissingFlag.mul(returnsData.nonZeroFlag)
            numOkReturns = goodReturnsFlag.iloc[:,-self.pcaHistory:].sum(axis=1)
            numOkReturns = numOkReturns / float(numpy.max(numOkReturns.values, axis=None))
            okIds = set(numOkReturns[numOkReturns.mask(numOkReturns > returnTol).isnull()].index)
        else:
            # Older US/AU/JP4 models
            returnTol = 0.95
            okIds = set(returnsData.numOkReturns[returnsData.numOkReturns.mask(\
                    returnsData.numOkReturns > returnTol).isnull()].index) 
        originalEstu = sorted(estu)
        estu = sorted(okIds.intersection(estu))

        # Report on changes to estimation universe used by APCA
        diff = set(originalEstu).difference(estu)
        if len(diff) > 0:
            estuMCap = assetData.marketCaps[originalEstu].sum(axis=None)
            newEstuMCap = assetData.marketCaps[estu].sum(axis=None)
            capRatio = 1.0 - newEstuMCap / estuMCap
            logging.info('Dropping %d assets (%.2f%% by mcap) from estu with more than %d%% of returns missing',
                    len(diff), 100.0 * capRatio, int(round(100.0*(1.0-returnTol))))
            if self.debuggingReporting:
                assetRMGMap = Utilities.flip_dict_of_lists(assetData.rmgAssetMap)
                outFile = open('tmp/dropped-%s.csv' % modelDate, 'w')
                for sid in sorted(diff):
                    outFile.write('%s,%s,%.4f,%s\n' % \
                            (sid if isinstance(sid, str) else sid.getSubIdString(),
                                assetData.getAssetType()[sid],
                                returnsData.numOkReturns[sid],
                                assetRMGMap[sid].mnemonic))
                outFile.close()

        # Report on estu composition by country
        for rmg in self.rmg:
            rmgEstuAssets = set(assetData.rmgAssetMap[rmg]).intersection(estu)
            origRmgEstu = set(assetData.rmgAssetMap[rmg]).intersection(originalEstu)
            mcap_ESTU = assetData.marketCaps[rmgEstuAssets].fillna(0.0).sum(axis=None) / 1.0e9
            mcap_orig = assetData.marketCaps[origRmgEstu].fillna(0.0).sum(axis=None) / 1.0e9
            logging.info('Market %s has %d/%d estu assets, mcap: %.2f/%.2f',
                    rmg.mnemonic, len(rmgEstuAssets), len(origRmgEstu), mcap_ESTU, mcap_orig)

        # Report on number of China-A shares
        chinaA = [sid for sid in estu if assetData.getAssetType()[sid] in self.localChineseAssetTypes]
        if len(chinaA) > 0:
            logging.info('%d estimation universe assets out of %d are China-A shares', len(chinaA), len(estu))

        # Trim outliers along region/sector buckets
        opms = dict()
        if self.useBucketedMAD:
            # Currently should only be True for US4/AU4/JP4
            opms['nBounds'] = [3.0, 3.0]
            outlierClass = Outliers.Outliers(opms, industryClassificationDate=self.gicsDate)
            clippedReturns = outlierClass.bucketedMAD(
                    self.rmg, modelDate, Utilities.df2ma(returnsData.returns), assetData, modelDB, axis=0)
        else:
            if hasattr(self, 'nu_sh_stat_model') and self.nu_sh_stat_model:
                opms['nBounds'] = [25.0, 25.0, 15.0, 15.0]
            else:
                opms['nBounds'] = [15.0, 15.0, 8.0, 8.0]
            outlierClass = Outliers.Outliers(opms)
            clippedReturns = outlierClass.twodMAD(Utilities.df2ma(returnsData.returns))
        clippedReturns = pandas.DataFrame(\
                clippedReturns, index=returnsData.returns.index, columns=returnsData.returns.columns)

        if self.debuggingReporting:
            dates = [str(d) for d in returnsData.returns.columns]
            idList = [s if isinstance(s, str) else s.getSubIDString() for s in assetData.universe]
            retOutFile = 'tmp/%s-retHist-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(Utilities.df2ma(returnsData.returns), retOutFile, rowNames=idList, columnNames=dates)
            retOutFile = 'tmp/%s-retHist-clipped-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(Utilities.df2ma(clippedReturns), retOutFile, rowNames=idList, columnNames=dates)

        # Downweight some countries if required
        for r in [r for r in self.rmg if r.downWeight < 1.0]:
            rmgEstu = set(assetData.rmgAssetMap[r]).intersection(originalEstu)
            clippedReturns.loc[rmgEstu, :] *= r.downWeight
        if hasattr(assetData, 'mktCapDownWeight'):
            for (rmg, dnWt) in assetData.mktCapDownWeight.items():
                if rmg in assetData.tradingRmgAssetMap:
                    rmgEstu = set(assetData.rmgAssetMap[rmg]).intersection(originalEstu)
                    clippedReturns.loc[rmgEstu, :] *= dnWt

        # Compute exposures, factor and specific returns
        self.log.debug('Computing exposures and factor returns: begin')
        assetIdxMap = dict(zip(assetData.universe, range(len(assetData.universe))))
        estuIdx = [assetIdxMap[sid] for sid in estu]

        # Build temporary object to pass to APCA
        tmpRets = Utilities.Struct()
        tmpRets.data = Utilities.df2ma(clippedReturns)
        tmpRets.missingFlag = Utilities.df2ma(returnsData.missingFlag)
        tmpRets.preIPOFlag = Utilities.df2ma(returnsData.preIPOFlag)

        # Build Statistical model of returns
        if self.isRegionalModel():
            # Do initial "global" APCA run
            if hasattr(self, 'numGlobalFactors'):
                self.returnCalculator.numFactors = self.numGlobalFactors
            (expMatrix, factorReturns, specificReturns0, regressANOVA, pctgVar) = \
                    self.returnCalculator.calc_ExposuresAndReturns(tmpRets, estuIdx, T=self.pcaHistory)

            if self.debuggingReporting:
                dates = [str(d) for d in returnsData.returns.columns]
                idList = [s.getSubIDString() for s in assetData.universe]
                retOutFile = 'tmp/%s-retHist-Filled-%s.csv' % (self.name, modelDate)
                Utilities.writeToCSV(tmpRets.data, retOutFile, rowNames=idList, columnNames=dates)

            # Compute regional factors
            if hasattr(self, 'regionFactorMap'):
                expMatrix, factorReturns = self.regionalAPCA(
                        specificReturns0, expMatrix, assetData, factorReturns, modelDB)
        else:
            if hasattr(self, 'sectorFactorMap'):
                # Compute "global" factors
                self.returnCalculator.numFactors = self.numGlobalFactors
                (expMatrix, factorReturns, specificReturns0, regressANOVA, pctgVar) = \
                        self.returnCalculator.calc_ExposuresAndReturns(tmpRets, estuIdx, T=self.pcaHistory)

                # Compute sector factors
                expMatrix, factorReturns = self.sectorAPCA(
                        modelDate, specificReturns0, expMatrix, assetData, factorReturns, modelDB)

            else:
                (expMatrix, factorReturns, specificReturns0, regressANOVA, pctgVar) = \
                        self.returnCalculator.calc_ExposuresAndReturns(tmpRets, estuIdx, T=self.pcaHistory)

        # Build exposure matrix with currency factors
        exposureMatrix = Matrices.ExposureMatrix(assetData.universe)
        if len(self.currencies) > 0 and self.hasCurrencyFactor:
            exposureMatrix = self.generate_currency_exposures(\
                    modelDate, modelDB, marketDB, assetData, exposureMatrix)

        # Add statistical factor exposures to exposure matrix
        blindFactorNames = [f.name for f in self.blind]
        exposureMatrix.addFactors(blindFactorNames, numpy.transpose(expMatrix), ExposureMatrix.StatisticalFactor)
        factorReturns = pandas.DataFrame(factorReturns, index=blindFactorNames, columns=returnsData.returns.columns)

        # Do cloning of exposures for linked assets
        scores = self.load_ISC_Scores(modelDate, assetData, modelDB, marketDB, returnDF=True)
        self.group_linked_assets(modelDate, assetData, modelDB, marketDB)
        exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, assetData, exposureMatrix, modelDB, marketDB, scores)
        
        if self.debuggingReporting:
            exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, assetData=assetData, dp=self.dplace)
            dates = [str(d) for d in returnsData.returns.columns]
            retOutFile = 'tmp/%s-facretHist-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(Utilities.df2ma(factorReturns), retOutFile, rowNames=blindFactorNames, columnNames=dates, dp=8)

        # Compute 'real' specific returns using non-clipped returns
        exposureIdx = [exposureMatrix.getFactorIndex(n) for n in blindFactorNames]
        expMatrix = ma.transpose(ma.take(exposureMatrix.getMatrix(), exposureIdx, axis=0))
        specificReturns = Utilities.df2ma(returnsData.returns) - numpy.dot(ma.filled(expMatrix, 0.0), factorReturns)
        specificReturns = pandas.DataFrame(specificReturns, index=assetData.universe, columns=returnsData.returns.columns)

        # Map specific returns for cloned assets
        clones = assetData.getCloneMap(cloneType='hard')
        if len(clones) > 0:
            cloneList = set(clones.keys()).intersection(set(assetData.universe))
            for sid in cloneList:
                if clones[sid] in assetData.universe:
                    specificReturns.loc[sid,:] = specificReturns.loc[clones[sid], :]
        
        # Now run weighted OLS using previous day's exposures on today's returns
        regData = Utilities.Struct()
        regData.regressionStatistics = Matrices.allMasked((len(self.factors), 4))
        self.estuMapCopy = copy.deepcopy(self.estuMap)
        self.setFactorsForDate(prevDate, modelDB)
        self.estuMap = self.estuMapCopy

        # Set default values in case of regression failure
        originalEstuIdx = [assetIdxMap[sid] for sid in originalEstu]
        regressANOVA = FactorReturns.RegressionANOVA(
                Utilities.df2ma(returnsData.returns.iloc[:,-1]), Utilities.df2ma(specificReturns.iloc[:,-1]),
                self.numFactors, originalEstuIdx)
        regData.adjRsquared = 0.0
        regData.factorReturns = Matrices.allMasked((len(blindFactorNames),))

        if (prev_rmi is not None) and prev_rmi.has_risks:
            prev_expM = self.loadExposureMatrix(prev_rmi, modelDB, assetList=assetData.universe)
            if not hasattr(self, 'olsReturnClass'):
                from riskmodels import ModelParameters2017
                self.olsReturnClass = ModelParameters2017.simpleRegressionParameters(self, modelDB)

            # Call nested regression routine
            logging.info('Running weighted OLS for regression statistics')
            olsData = self.olsReturnClass.run_factor_regressions(
                    self, self.olsReturnClass, prevDate,
                    Utilities.df2ma(returnsData.returns.iloc[:,-1]), prev_expM, originalEstu,
                    assetData, [], modelDB, marketDB, applyRT=False, fmpRun=False)

            # Compute various regression statistics
            regData.factorReturns = pandas.Series(olsData.factorReturnsMap)[blindFactorNames].values
            for (idx, fName) in enumerate(blindFactorNames):
                regData.regressionStatistics[idx,:] = olsData.regStatsMap[fName]
            if olsData.anova is not None:
                regData.adjRsquared = olsData.anova.calc_adj_rsquared()
                regData.regression_ESTU = list(zip([assetData.universe[i] for i in olsData.anova.estU_],
                        olsData.anova.weights_ / numpy.sum(olsData.anova.weights_)))
            else:
                logging.info('Adjusted R-Squared=%.6f', regData.adjRsquared)
                regData.regression_ESTU = list(zip([assetData.universe[i] for i in regressANOVA.estU_],
                        regressANOVA.weights_ / numpy.sum(regressANOVA.weights_)))
        else:
            prev_expM = exposureMatrix
            logging.warning('No risk model saved for previous date: %s', prevDate)
            logging.info('Adjusted R-Squared=%.6f', regData.adjRsquared)
            regData.regression_ESTU = list(zip([assetData.universe[i] for i in regressANOVA.estU_],
                    regressANOVA.weights_ / numpy.sum(regressANOVA.weights_)))

        self.setFactorsForDate(modelDate, modelDB)

        # Output data on returns from OLS regression
        regData.fmpMap = dict()
        nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in subFactors])
        self.regressionReporting(Utilities.df2ma(returnsData.returns.iloc[:,-1]),
                regData, prev_expM, nameSubIDMap, assetIdxMap, modelDate, buildFMPs=False,
                constrComp=None, specificRets=Utilities.df2ma(specificReturns.iloc[:,-1]))
         
        # ***************NOTE that dates are now switched to REVERSE chronological order ***************
        # Check returns history lengths
        dateList = sorted(returnsData.returns.columns, reverse=True)
        if len(dateList) < max(self.minFCovObs, self.minSRiskObs):
            required = max(self.minFCovObs, self.minSRiskObs)
            self.log.warning('%d missing risk model instances for required days', required - len(dateList))
            raise LookupError('%d missing risk model instances for required days' % (required - len(dateList)))
        omegaObs = min(len(dateList), self.maxFCovObs)
        deltaObs = min(len(dateList), self.maxSRiskObs)
        self.log.info('Using %d of %d days of factor return history', omegaObs, len(dateList))
        self.log.info('Using %d of %d days of specific return history', deltaObs, len(dateList))
        
        # Set up specific returns history matrix
        if self.maskMissingSpecificReturns:
            # Newer models mask specific returns corresponding to missing total returns
            srMatrix = specificReturns.fillna(0.0).mask(returnsData.missingFlag).loc[assetData.universe, dateList[:deltaObs]]
        else:
            srMatrix = specificReturns.fillna(0.0).loc[assetData.universe, dateList[:deltaObs]]
        self.log.debug('building time-series matrices: end')

        if self.debuggingReporting:
            dates = [str(d) for d in srMatrix.columns]
            idList = [s if isinstance(s, str) else s.getSubIDString() for s in assetData.universe]
            retOutFile = 'tmp/%s-specRetHist-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(Utilities.df2ma(srMatrix), retOutFile, rowNames=idList, columnNames=dates)

        # Recompute proportion of non-missing returns over specific return history
        numOkReturns = returnsData.nonMissingFlag.loc[:, srMatrix.columns].sum(axis=1)
        numOkReturns = numOkReturns / float(numpy.max(numOkReturns.values, axis=None))

        # Compute asset specific risks
        (specificVars, specificCov) = self.specificRiskCalculator.computeSpecificRisks(\
                srMatrix, assetData, self, modelDB, nOkRets=numOkReturns,
                scoreDict=scores, rmgList=self.rmg, excludeAssets=noProxyReturnsList)

        # Report on specific covariance
        self.reportISCStability(modelDate, specificVars, specificCov, prev_rmi, modelDB)
        self.log.debug('computed specific variances')
        
        # Set up statistical factor returns history and corresponding information
        frData = Utilities.Struct()
        frData.data = factorReturns.loc[:,dateList[:omegaObs]]
        
        # Build factor covariance matrix
        if not self.hasCurrencyFactor:
            factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(frData)
        else:
            # Process currency block
            modelCurrencyFactors = [f for f in self.factors if f in self.currencies]
            cfReturns, cc, currencySpecificRisk = self.process_currency_block(\
                                dateList[:omegaObs], modelCurrencyFactors, modelDB)
            self.covarianceCalculator.configureSubCovarianceMatrix(1, cc)
            crData = Utilities.Struct()
            crData.data = cfReturns

            # Compute factor covariance
            factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(frData, crData)

            # Add in the currency specific variances
            for cf in currencySpecificRisk.index:
                factorCov.loc[cf, cf] += currencySpecificRisk.loc[cf] * currencySpecificRisk.loc[cf]

        # Re-order according to list of model factors
        factorNames = [f.name for f in self.factors]
        factorCov = factorCov.reindex(index=factorNames, columns=factorNames)
        frMatrix = factorReturns.reindex(index=blindFactorNames, columns=dateList[:omegaObs])

        # Safeguard to deal with nasty negative eigenvalues
        factorCov = Utilities.forcePositiveSemiDefiniteMatrix(factorCov, min_eigenvalue=0.0)
        factorCov = (factorCov + factorCov.T) / 2.0

        # Report day-on-day correlation matrix changes
        estuFactorRisk = self.reportCorrelationMatrixChanges(\
                modelDate, assetData, exposureMatrix, factorCov, rmi, prev_rmi, modelDB)

        if self.debuggingReporting:
            outfile = 'tmp/%s-specificRisk-%s.csv' % (self.name, dateList[0])
            outfile = open(outfile, 'w')
            outfile.write('CID,SID,Name,Type,exSPAC,ESTU,Risk,\n')
            specificRisks = 100.0 * numpy.sqrt(specificVars)
            exSpac = AssetProcessor_V4.sort_spac_assets(\
                    dateList[0], assetData.universe, modelDB, marketDB, returnExSpac=True)
            for sid in sorted(assetData.universe):
                outfile.write('%s,%s,%s,%s,' % ( \
                            assetData.getSubIssue2CidMapping().get(sid, ''), sid.getSubIDString(),
                            assetData.getNameMap().get(sid, '').replace(',',''), assetData.getAssetType().get(sid, '')))
                if sid in exSpac:
                    outfile.write('1,')
                else:
                    outfile.write('0,')
                if sid in assetData.estimationUniverse:
                    outfile.write('1,%.6f,\n' % (specificRisks[sid]))
                else:
                    outfile.write('0,%.6f,\n' % (specificRisks[sid]))
            outfile.close()

        # Add covariance matrix to return object
        frMatrix = factorReturns.loc[:, dateList[:omegaObs]]
        if self.hasCurrencyFactor:
            crmi, dummy = self.set_model_instance(dateList[0], modelDB, rm=self.currencyModel)
            cfReturns = self.currencyModel.loadCurrencyFactorReturnsHistory(
                    crmi, dateList[:omegaObs], modelDB, factorList=self.currencies, returnDF=True, screen=True).fillna(0.0)
            frMatrix = pandas.concat([frMatrix, cfReturns], axis=0)
        frMatrix = frMatrix.reindex(index=factorNames, columns=dateList[:omegaObs])
        
        # Add data to return object
        smData = Utilities.Struct()
        smData.pctgVar = pctgVar
        smData.VIF = self.VIF
        smData.regressionStatistics = regData.regressionStatistics
        smData.adjRsquared = regData.adjRsquared
        smData.regression_ESTU = regData.regression_ESTU
        smData.frMatrix = Matrices.TimeSeriesMatrix(subFactors, dateList[:omegaObs])
        smData.frMatrix.data = Utilities.df2ma(frMatrix)
        smData.srMatrix = Matrices.TimeSeriesMatrix(assetData.universe, dateList[:deltaObs])
        smData.srMatrix.data = Utilities.df2ma(srMatrix.reindex(index=assetData.universe, columns=dateList[:omegaObs]))
        smData.exposureMatrix = exposureMatrix
        smData.factorCov = factorCov.fillna(0.0).values
        smData.specificVars = specificVars
        smData.specificCov = specificCov
        self.log.debug('computed factor covariances')
        return smData

def merge_nodes(node, nodeDict, prefix, remap):
    """Merges new node in with the existing set and renames/augments where necessary
    """
    nodeCopy = Utilities.Struct(copy=node)
    nodeCopy.prefix = prefix

    # Apply name change of node if relevant
    remapped = False
    if node.name in remap:
        nodeName = remap[node.name]
        remapped = True
    else:
        nodeName = node.name

    # Merge with existing node if names match
    if nodeName in nodeDict:
        cur_node = nodeDict[nodeName]
        cur_node.srm_nodes[node.id] = nodeCopy
        cur_node.id = tuple(sorted((*cur_node.id, node.id)))
        if not remapped:
            cur_node.description = node.description
        # Consistency checks on the type of node
        assert not node.isLeaf
        assert (cur_node.isLeaf == node.isLeaf)
        assert (cur_node.isRoot == node.isRoot)

    # Else it's new, so initialise
    else:
        new_nd = Utilities.Struct(copy=node)
        new_nd.srm_nodes = dict()
        new_nd.srm_nodes[node.id] = nodeCopy
        # Do some renaming if necessary
        if node.isLeaf:
            new_nd.id = (prefix, node.id)
            new_nd.name = '%s %s' % (prefix, nodeName)
            new_nd.description = '%s %s' % (prefix, node.description)
        else:
            new_nd.id = (node.id,)
            new_nd.name = nodeName
            new_nd.revision_id = None
            new_nd.market_cls_ids = []
        nodeDict[new_nd.name] = new_nd

    return nodeDict

class LinkedModelClassification:
    def __init__(self, _riskModel, marketDB, modelDB):
        self._riskModel = _riskModel
        self._marketDB = marketDB
        self.name = 'Custom-Linked-Classification'
        self.members = dict()

        # Loop through models and collect constituent information
        for (srm, prefix, scmMarket) in _riskModel.linkedModelMap:
            logging.info('Retrieving %s model industry member classifications', srm.name)
            # Combine non-root members
            member = srm.industryClassification.getClassificationMember(modelDB)
            members = modelDB.getMdlClassificationMembers(member, srm.industryClassification.date)
            for m in members:
                self.members = merge_nodes(m, self.members, prefix, self._riskModel.industryRename)
            # Add root members
            srm_roots =  srm.industryClassification.getClassificationRoots(modelDB)
            for rt in srm_roots:
                self.members = merge_nodes(rt, self.members, prefix, self._riskModel.industryRename)

        if hasattr(self._riskModel, 'industryAddNode'):
            for (nd, nid, prefix) in self._riskModel.industryAddNode:
                new_nd = Utilities.Struct()
                new_nd.id = nid
                new_nd.name = nd
                new_nd.description = nd
                #new_nd.srm_nodes = {new_nd.id: (None, new_nd.name)}
                new_nd.revision_id = None
                new_nd.market_cls_ids = []
                new_nd.isLeaf = False
                new_nd.isRoot = False
                self.members = merge_nodes(new_nd, self.members, prefix, self._riskModel.industryRename)

    def update_nodes(self, node, nodeDict, prefix):
        """Merges new node in with the existing set and renames/augments where necessary
        """
        # Rename leaves to include model prefix
        if node.isLeaf and (prefix is not None):
            nodeName = '%s %s' % (prefix, node.name)
        else:
            nodeName = node.name

        # If name has been changed, reasign
        if nodeName in self._riskModel.industryRename:
            nodeName = self._riskModel.industryRename[nodeName]

        # Pull the correct node from our saved pool
        if nodeName not in nodeDict:
            nodeDict[nodeName] = self.members[nodeName]

        # Add the weight if it exists
        if hasattr(node, 'weight'):
            nodeDict[nodeName].weight = node.weight
        elif not node.isRoot:
            nodeDict[nodeName].weight = 1

        return nodeDict

    def output_node_info(self, node, ntype):
        """Outputs the various parts of each node for debugging
        """
        logging.info('%s ID: %s', ntype, node.id)
        logging.info('.......Name: %s', node.name)
        logging.info('.......Description: %s', node.description)
        logging.info('.......Revision ID : %s', node.revision_id)
        logging.info('.......Market_CLS_IDS: %s', len(node.market_cls_ids))
        logging.info('.......Root: %s', node.isRoot)
        logging.info('.......Leaf: %s', node.isLeaf)
        if hasattr(node, 'weight'):
            logging.info('.......Weight: %s', node.weight)
        srm_ids = [(nid, nd.prefix, nd.name) for (nid, nd) in node.srm_nodes.items()]
        for srm_info in srm_ids:
            logging.info('.......SRM Node: (%s, %s, %s)', srm_info[0], srm_info[1], srm_info[2] )
        return

    def getClassificationRoots(self, modelDB):
        """Returns the root classification objects for this risk model.
        Delegates to the industryClassification object.
        """
        roots = dict()

        # Loop round constituent models
        for (srm, prefix, scmMarket) in self._riskModel.linkedModelMap:
            logging.info('Retrieving %s model industry classification roots', srm.name)
            srm_roots =  srm.industryClassification.getClassificationRoots(modelDB)

            for rt in srm_roots:
                # Merge with existing roots
                roots = self.update_nodes(rt, roots, prefix)

        return list(roots.values())

    def getClassificationChildren(self, parentClass, modelDB):
        """Returns a list of the children of the given node in the classification.
        """
        # Initialise
        dropChild = self._riskModel.industryDropChild
        addChild = self._riskModel.industryAddChild
        childDict = dict()

        # Loop round parent nodes
        for pID, node in parentClass.srm_nodes.items():

            # Add all children for this node unless we've specified they should be dropped
            dropList = dropChild.get(node.name, [])
            for chld in modelDB.getMdlClassificationChildren(node):
                if chld.name not in dropList:
                    childDict = self.update_nodes(chld, childDict, node.prefix)

            # Add in children that have been manually added from another node
            for chld in addChild.get(node.name, []):
                if chld in self.members:
                    childDict = self.update_nodes(self.members[chld], childDict, None)

        return list(childDict.values())

    def getAllParents(self, childClass, modelDB):
        """Returns all parents of the given node in the classification.
        """
        # Initialise
        addParentDict = Utilities.flip_dict_of_lists_m2m(self._riskModel.industryAddChild)
        dropParentDict = Utilities.flip_dict_of_lists_m2m(self._riskModel.industryDropChild)
        prntDict = dict()
        prntList = []

        # Loop round child nodes
        for cID, node in childClass.srm_nodes.items():

            # Add any missing root parents
            if node.name in self._riskModel.industryAddRootParent:
                rootParents = [p for p in prntList if p.isRoot]
                if len(rootParents) < 1:
                    prntName = self._riskModel.industryAddRootParent[node.name]
                    newPrnt = self.members.get(prntName, None)
                    if newPrnt is not None:
                        logging.info('Adding new root parent for %s:', node.name)
                        prntList.append(newPrnt)
                        if self._riskModel.debuggingReporting:
                            self.output_node_info(newPrnt, 'PARENT')
                    else:
                        logging.warn('No root parent for %s', node.name)

            # Include any manually-added parents
            for prnt in addParentDict.get(node.name, []):
                prntList.append(self.members[prnt])

            # Add all official node parents unless manual override says no
            dropPrnt = dropParentDict.get(node.name, [])
            officialParentList = modelDB.getMdlClassificationAllParents(node)
            if officialParentList is not None:
                for prnt in officialParentList:
                    if (node.name not in self._riskModel.industryRename) and (prnt.name not in dropPrnt):
                        prntList.append(prnt)

        # Process the list of parents
        prntDict = dict()
        for prnt in prntList:
            prntDict = self.update_nodes(prnt, prntDict, node.prefix)

        return list(prntDict.values())

    def getAllClassifications(self, modelDB):
        """Returns all non-root classification objects of this industry classification.
        """
        cls = [val for val in self.members.values() if not val.isRoot]
        clsNameMap = dict(zip([val.name for val in cls], cls))
        secs = sorted([val.name for val in cls if val.name[-2:]=='-S'])
        igs = sorted([val.name for val in cls if val.name[-2:]=='-G'])
        other = sorted([val.name for val in cls if val.name not in secs+igs])
        return [clsNameMap[nm] for nm in secs+igs+other]

    def getAssetConstituents(self, modelDB, universe, date, level=None):
        """Returns the classification_constituent information for a list of assets from this classification on the supplied date.
        Return is a dict mapping assets to the constituent information.
        Used by flatfiles and Derby to determine source of classification (e.g. GICS)
        """
        # Build mapping of sub-models to assets
        universeMap = self._riskModel.createAsset2ModelMapping(date, universe, modelDB, self._marketDB)
        ass_const = dict()

        # Loop through models and collect constituent information
        for (srm, prefix, scmMarket) in self._riskModel.linkedModelMap:
            logging.info('Retrieving %s model industry classification constituents', srm.name)
            srm_ass_const = srm.industryClassification.getAssetConstituents(modelDB, universeMap[srm], date, level)
            ass_const.update(srm_ass_const)
        return ass_const

    def getLeafNodes(self, modelDB):
        """Returns the leaves of the classification which are used
        as industries in the risk model.
        """
        leaves = dict()
        for (srm, prefix, scmMarket) in self._riskModel.linkedModelMap:
            for lf_id, node in srm.industryClassification.getLeafNodes(modelDB).items():
                leaves = self.update_nodes(node, leaves, prefix)
        return leaves

    def getExposures(self, dt, universe, factors, modelDB, level=None, returnDF=False):
        """Returns a matrix of exposures of the assets to the classification
        scheme on the given date.
        """
        # Build mapping of sub-models to assets
        universeMap = self._riskModel.createAsset2ModelMapping(dt, universe, modelDB, self._marketDB)

        # Initialise exposure matrix
        exposureMatrix = None

        # Loop through models and collect sub-issues and estimation universes
        for (srm, prefix, scmMarket) in self._riskModel.linkedModelMap:
            logging.info('Retrieving %s model industry classifications', srm.name)

            # Load model's exposure matrix
            srm_leaves = srm.industryClassification.getLeafNodes(modelDB)
            srm_factorNames = [i.description for i in srm_leaves.values()]
            srm_expM = srm.industryClassification.getExposures(dt, universeMap[srm], srm_factorNames, modelDB, level=level)
            srm_expM_df = pandas.DataFrame(srm_expM, index=srm_factorNames, columns=universeMap[srm]).T

            # Rename factors
            for fct in srm_expM_df.columns:
                nuName = self._riskModel.baseModelFactorNameMap[(srm, fct)]
                srm_expM_df.rename(columns={fct:nuName}, inplace=True)

            # Add to running set of exposures
            if exposureMatrix is None:
                exposureMatrix = srm_expM_df.copy(deep=True)
            else:
                exposureMatrix = pandas.concat([exposureMatrix, srm_expM_df], axis=0)

        # Convert back to a masked array
        leaves = self._riskModel.industryClassification.getLeafNodes(modelDB)
        factorNames = [i.description for i in leaves.values()]
        exposureMatrix = exposureMatrix.reindex(index=universe, columns=factorNames)
        if returnDF:
            return exposureMatrix.T
        expData = Utilities.screen_data(exposureMatrix.T.values)
        expData = ma.masked_where(numpy.isnan(expData), expData)

        if self._riskModel.debuggingReporting:
            exposureMatrix.to_csv('tmp/industry-expM.csv')

        return expData

class LinkedModel(FactorRiskModel):
    """ Linked risk model
    """
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)

        # Set up default estimation universe
        self.masterEstuMap = modelDB.getEstuMappingTable(self.rms_id)
        if self.masterEstuMap is None:
            logging.error('No estimation universe mapping defined')
            assert(self.masterEstuMap is not None)
        logging.info('Estimation universe structure: %d estus', len(self.masterEstuMap))

        # Report on underlying models
        logging.info('Linking the following models...')
        for (rm, dm2, dm3) in self.linkedModelMap:
            logging.info('... (%s, %s)', rm.mnemonic, rm.rms_id)

    def isLinkedModel(self):
        return True

    def setFactorsForDate(self, date, modelDB):
        """Determine factors as union of those of underlying models
        """
        # Set up estimation universe parameters
        self.estuMap = copy.deepcopy(self.masterEstuMap)

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)

        # Initialise factors
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        styles = []
        countries = []
        currencies = []
        industries = []
        intercept = None
        regionalIntercepts = []
        localStructureFactors = []
        baseModelFactorNameMap = dict()
        self.ri2CountryMap = dict()
        # Remove country factors that are covered by SCMs
        dropCountries = [z for (x,y,z) in self.linkedModelMap if z is not None]

        # Set currency factors
        allRMG = modelDB.getAllRiskModelGroups(inModels=True)
        for rmg in allRMG:
            rmg.setRMGInfoForDate(date)
        currencies = [ModelFactor(f, None) for f in set([r.currency_code for r in allRMG])]
        currencies.extend([ModelFactor('EUR', 'Euro')])
        currencies = sorted(set([f for f in currencies if f.name in self.nameFactorMap]))
        factorTypeDict = {cur.name: ExposureMatrix.CurrencyFactor for cur in currencies}

        # Loop through models and collect factors
        for (rm, prefix, scmMarket) in self.linkedModelMap:
            logging.info('Loading factors from %s', rm.name)
            rm.setFactorsForDate(date, modelDB)

            # Deal with  styles
            for fct in rm.styles:
                fctName = '%s %s' % (prefix, fct.name)
                lmFct = ModelFactor(fctName, fctName)
                baseModelFactorNameMap[(rm, fct.name)] = lmFct.name
                factorTypeDict.update({fctName: ExposureMatrix.StyleFactor})
                styles.append(lmFct)

            # Deal with industries
            for fct in rm.industries:
                fctName = '%s %s' % (prefix, fct.name)
                lmFct = ModelFactor(fctName, fctName)
                baseModelFactorNameMap[(rm, fct.name)] = lmFct.name
                factorTypeDict.update({fctName: ExposureMatrix.IndustryFactor})
                industries.append(lmFct)

            # Deal with countries
            for fct in rm.countries:
                if (fct.name in dropCountries):
                    continue
                lmFct = ModelFactor(fct.name, fct.name)
                baseModelFactorNameMap[(rm, fct.name)] = fct.name
                factorTypeDict.update({fct.name: ExposureMatrix.CountryFactor})
                countries.append(lmFct)
                dropCountries.append(fct.name)
                    
            # Deal with intercepts - they become country or local structure factors
            if rm.intercept is not None:
                if not self.legacyLMFactorStructure:
                    fctName = '%s %s' % (prefix, rm.intercept.name)
                    lmFct = ModelFactor(fctName, fctName)
                    baseModelFactorNameMap[(rm, rm.intercept.name)] = lmFct.name
                    factorTypeDict.update({fctName: ExposureMatrix.RegionalIntercept})
                    regionalIntercepts.append(lmFct)
                    if scmMarket is not None:
                        self.ri2CountryMap[scmMarket] = fctName
                elif scmMarket is not None:
                    fctName = scmMarket
                    lmFct = ModelFactor(fctName, fctName)
                    baseModelFactorNameMap[(rm, rm.intercept.name)] = lmFct.name
                    factorTypeDict.update({fctName: ExposureMatrix.CountryFactor})
                    countries.append(lmFct)
                    dropCountries.append(fctName)
                else:
                    fctName = '%s %s' % (prefix, rm.intercept.name)
                    lmFct = ModelFactor(fctName, fctName)
                    baseModelFactorNameMap[(rm, rm.intercept.name)] = lmFct.name
                    factorTypeDict.update({fctName: ExposureMatrix.LocalFactor})
                    localStructureFactors.append(lmFct)

            # And local structure factors
            for fct in rm.localStructureFactors:
                fctName = fct.name
                if (fct.name in dropCountries):
                    continue
                lmFct = ModelFactor(fct.name, fct.name)
                baseModelFactorNameMap[(rm, fct.name)] = lmFct.name
                factorTypeDict.update({fct.name: ExposureMatrix.LocalFactor})
                localStructureFactors.append(lmFct)
                dropCountries.append(fct.name)

        # Set factor attributes
        if intercept is not None:
            allFactors = styles + industries + countries + currencies + localStructureFactors + regionalIntercepts + [intercept]
        else:
            allFactors = styles + industries + countries + currencies + localStructureFactors + regionalIntercepts
        for f in allFactors:
            if f.name in self.nameFactorMap:
                dbFactor = self.nameFactorMap[f.name]
            else:
                dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            if f.name is None:
                f.name = dbFactor.name
            if f.description is None:
                f.description = dbFactor.description

        # Save permanently
        self.nurseryCountries = []
        self.nurseryRMGs = []
        self.styles = [s for s in styles if s.isLive(date)]
        self.currencies = [c for c in currencies if c.isLive(date)]
        self.countries = [f for f in countries if f.isLive(date)]
        self.factors = [f for f in allFactors if f.isLive(date)]
        self.industries = industries
        self.localStructureFactors = localStructureFactors
        self.intercept = intercept
        self.regionalIntercepts = regionalIntercepts
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.validateFactorStructure(date, warnOnly=self.variableStyles)
        self.allFactors = allFactors
        self.baseModelFactorNameMap = baseModelFactorNameMap
        self.factorTypeDict = factorTypeDict
        logging.info('Linked model has the following %d factors on %s...', len(self.factors), date)
        logging.info('... %d style factors', len(self.styles))
        logging.info('... %d industry factors', len(self.industries))
        logging.info('... %d country factors', len(self.countries))
        logging.info('... %d currency factors', len(self.currencies))
        logging.info('... %d regional intercepts', len(self.regionalIntercepts))
        logging.info('... %d local factors', len(self.localStructureFactors))
        if self.intercept is not None:
            logging.info('... %d intercept factors', len([self.intercept]))

    def generate_model_universe(self, date, modelDB, marketDB):
        """Generate risk model instance universe and estimation universe.
        Return value is a Struct containing a universe attribute (list of SubIssues)
        containing the model universe and an estimationUniverse attribute
        (list of index positions corresponding to ESTU members).
        """
        self.log.debug('generate_model_universe: begin')

        # Initialise
        self.universeBaseMap = None
        baseData = AssetProcessor_V4.AssetProcessor(date, modelDB, marketDB, self.getDefaultAPParameters())
        asset_pool = set(baseData.getModelAssetMaster(self))

        # XXX Delete later
        if self.debuggingReporting:
            c2p = False
            p2c = False
            con = self.industryClassification.getAssetConstituents(modelDB, asset_pool, date)
            rts = self.industryClassification.getClassificationRoots(modelDB)
            runningList = []
            for rt in rts:
                #self.industryClassification.output_node_info(rt, 'ROOT')
                chldn = self.industryClassification.getClassificationChildren(rt, modelDB)
                for chld in chldn:
                    if c2p:
                        self.industryClassification.output_node_info(chld, 'CHILD')
                    prnts = self.industryClassification.getAllParents(chld, modelDB)
                    for prnt in prnts:
                        if c2p:
                            self.industryClassification.output_node_info(prnt, 'PARENT')
                        else:
                            if prnt.name not in runningList:
                                if p2c:
                                    self.industryClassification.output_node_info(prnt, 'PARENT')
                                chldn2 = self.industryClassification.getClassificationChildren(prnt, modelDB)
                                if p2c:
                                    for chld2 in chldn2:
                                        self.industryClassification.output_node_info(chld2, 'CHILD2')
                                runningList.append(prnt.name)

            mem = self.industryClassification.getAllClassifications(modelDB)
            #for m in mem:
            #   self.industryClassification.output_node_info(m, 'ALL')
            leaves = self.industryClassification.getLeafNodes(modelDB)
            #for (lf, node) in leaves.items():
            #   self.industryClassification.output_node_info(node, 'LEAF')
            indExpM = self.industryClassification.getExposures(\
                        date, asset_pool, [ind.name for ind in self.industries], modelDB)

        # Build mapping of sub-models to assets
        if self.universeBaseMap is not None:
            universeMap = dict()
            for (ky, vals) in self.universeBaseMap.items():
                universeMap[ky] = vals.intersection(asset_pool)
        else:
            universeMap  = self.createAsset2ModelMapping(date, asset_pool, modelDB, marketDB)

        # Loop through models and collect estimation universes
        estuRunning = []
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            
            # Load model estimation universe
            logging.info('Retrieving %s estimation universe *****************************', srm.name)
            srmi, srmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
            estu = srm.loadEstimationUniverse(srmi, modelDB)
            for estuName in self.estuMap.keys():
                if not hasattr(self.estuMap[estuName], 'assets'):
                    self.estuMap[estuName].assets = []

                if estuName in srm.estuMap:
                    # Add model's assets to estimation universe if they haven't already been added
                    rmEstu = set(srm.estuMap[estuName].assets).difference(set(estuRunning))
                    rmEstu = rmEstu.intersection(set(universeMap[self]))
                    self.estuMap[estuName].assets.extend(rmEstu)
                    estuRunning.extend(rmEstu)

        testData = AssetProcessor_V4.AssetProcessor(date, modelDB, marketDB, self.getDefaultAPParameters())
        testData.process_asset_information(self.rmg, universe=universeMap[self])

        if self.debuggingReporting:
            # Output information on assets and their characteristics
            testData.estimationUniverse = sorted(estuRunning)
            estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                    date, testData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
            estuCls.estimation_universe_reporting(testData, Matrices.ExposureMatrix(testData.universe))

        # Some reporting of stats
        estuIdSeries = pandas.Series(dict([(self.estuMap[ky].id, ky) for ky in self.estuMap.keys()]))
        for eid in sorted(estuIdSeries.index):
            sub_estu = self.estuMap[estuIdSeries[eid]]
            if hasattr(sub_estu, 'assets'):
                mcap_ESTU = testData.marketCaps[sub_estu.assets].sum(axis=None)
                self.log.info('%s ESTU contains %d assets, %.2f tr %s market cap',
                        sub_estu.name, len(sub_estu.assets), mcap_ESTU / 1e12, self.numeraire.currency_code)
        self.log.info('Universe contains %d assets, %.2f tr %s market cap',
                len(testData.universe), testData.marketCaps.sum(axis=None) / 1e12, self.numeraire.currency_code)

        # Create return values
        retData = Utilities.Struct()
        retData.universe = sorted(set(universeMap[self]))
        return retData

    def createAsset2ModelMapping(self, date, asset_pool, modelDB, marketDB):
        """ Map universe of linked model assets to best possible sub-model
        """

        # Initialise
        universe = defaultdict(set)
        base_universe = dict()
        deadAssets = set()
        unmappedAssets = set()
        base_universe[self] = set(asset_pool)

        # Loop through models and collect sub-issues
        for (srm, prefix, scmMarket) in self.linkedModelMap:

            # Load model universe
            logging.info('Retrieving %s model universe *****************************', srm.name)
            srmi, srmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
            srm_univ = set(modelDB.getRiskModelInstanceUniverse(srmi, returnExtra=False))

            # Drop assets that do not exist in the linked model
            deadAssets = deadAssets.union(srm_univ.difference(base_universe[self]))
            base_universe[srm] = srm_univ.intersection(base_universe[self])

            # Map the sub-model's assets to the set of markets spanned by the linked model
            tmpData = AssetProcessor_V4.AssetProcessor(
                    date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
            tmpData.process_asset_information(self.rmg, universe=base_universe[srm])
            if tmpData.universe is None:
                logging.warn('No assets mapped for %s', srm.name)
                continue

            # Loop through the sub-model's markets and add associated assets to the total
            addedAssets = set()
            for rmg in srmgList:
                addedAssets = addedAssets.union(set(tmpData.rmgAssetMap[rmg]))

            # Drop assets if we've already assigned them to their ideal home
            alreadyMapped = set(addedAssets).intersection(universe[self])
            if len(alreadyMapped) > 0:
                logging.info('%d assets from %s already mapped, dropping', len(alreadyMapped), srm.name)
                addedAssets = addedAssets.difference(alreadyMapped)

            # Update return dict
            universe[srm] = addedAssets
            universe[self] = universe[self].union(addedAssets)
            logging.info('%d assets added to universe with exposure to %s model', len(addedAssets), srm.name)

        # If there are unmapped assets attempt to map them to the best available model
        unmappedAssets = base_universe[self].difference(universe[self])

        # Output information on assets dropped from the model
        if self.debuggingReporting and len(unmappedAssets) > 0:
            mkt_df = AssetProcessor_V4.get_all_markets(unmappedAssets, date, modelDB, marketDB)           
            mkt_df.index = [x.getSubIDString() for x in mkt_df.index]
            mkt_df.to_csv('tmp/unmapped0-%s-%s.csv' % (self.mnemonic, date))

        mappedTo = dict()
        bestModel = dict()
        if len(unmappedAssets) > 0:
            logging.info('Attempting to map %s leftover assets *****************************', len(unmappedAssets))

            for (srm, prefix, scmMarket) in self.linkedModelMap:
                # Map the leftover assets to the set of markets spanned by the sub-model
                eligibleAssets = unmappedAssets.intersection(base_universe[srm])

                if len(eligibleAssets) > 0:
                    # Attempt to map to SRM's market structure
                    tmpData = AssetProcessor_V4.AssetProcessor(
                            date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
                    tmpData.process_asset_information(srm.rmg, universe=eligibleAssets)

                    # Check for assets that map to multiple models
                    # Choose model based on the hierarchy Home > Home2 > Trading country
                    for sid in tmpData.universe:
                        if sid not in mappedTo:
                            mappedTo[sid] = tmpData.mappedTo[sid]
                            bestModel[sid] = srm.name
                        elif tmpData.mappedTo[sid] > mappedTo[sid]:
                            mappedTo[sid] = tmpData.mappedTo[sid]
                            bestModel[sid] = srm.name

                    # Loop through the sub-model's markets and add the assets to the total
                    addedAssets = set()
                    for rmg in srm.rmg:
                        addedAssets = addedAssets.union(tmpData.rmgAssetMap[rmg])
                    universe[srm] = universe[srm].union(addedAssets)
                    universe[self] = universe[self].union(addedAssets)
                    
                    # Ajust the number of unmapped assets
                    logging.info('%d leftover assets added to universe with exposure to %s model',
                                        len(addedAssets), srm.name)

        # Where assets have been assigned to multiple models, drop from less suitable models
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            overLap = universe[srm].intersection(bestModel.keys())
            dropList = set()
            for sid in overLap:
                # If asset is assigned to a better market elsewhere then drop
                if bestModel[sid] != srm.name:
                    dropList.add(sid)
                    if self.debuggingReporting:
                        logging.info('%s dropped from %s in favour of %s', sid.getSubIDString(), srm.name, bestModel[sid])
            universe[srm] = universe[srm].difference(dropList)
            if len(dropList) > 0:
                logging.info('%d assets dropped from %s for a better market', len(dropList), srm.name)

        # Output information on assets dropped from the model
        if self.debuggingReporting and len(deadAssets) > 0:
            logging.info('********************** %d delisted assets in sub-models', len(deadAssets))
            deadData = AssetProcessor_V4.AssetProcessor(
                    date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
            deadData.process_asset_information(self.rmg, universe=deadAssets)
            deadData.assetData_DF = deadData.getAssetInfoDF()
            deadData.assetData_DF.to_csv('tmp/delisted-%s.csv' % date)

        # Output information on assets that couldn't be mapped
        unmappedAssets = unmappedAssets.difference(universe[self])
        if self.debuggingReporting and len(unmappedAssets) > 0:
            logging.info('********************** %d assets unmapped to a sub-model', len(unmappedAssets))
            unmappedData = AssetProcessor_V4.AssetProcessor(
                    date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
            unmappedData.process_asset_information(self.rmg, universe=unmappedAssets)
            unmappedData.assetData_DF = unmappedData.getAssetInfoDF()
            unmappedData.assetData_DF.to_csv('tmp/unmapped-%s.csv' % date)

        self.universeBaseMap = dict()
        for (ky, vals) in universe.items():
            self.universeBaseMap[ky] = set(vals)
        return universe

    def generateExposureMatrix(self, date, modelDB, marketDB):
        """Generates and returns the exposure matrix for the given date.
        """
        self.log.debug('generateExposureMatrix: begin')

        # Get risk model universe
        rmi, dummy = self.set_model_instance(date, modelDB)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=False)

        # Build mapping of sub-models to assets
        universeMap = self.createAsset2ModelMapping(date, universe, modelDB, marketDB)

        # Initialise exposure matrix
        exposureMatrix = None
        universe = sorted(list(universeMap[self]))
        assetData = AssetProcessor_V4.AssetProcessor(
                date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
        assetData.process_asset_information(self.rmg, universe=universe)
        preSPAC = assetData.getSPACs(universe=assetData.universe)

        # Loop through models and collect sub-issues and estimation universes
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            logging.info('Preparing %s model exposures', srm.name)

            # Check that underlying model exists
            srmi_id, srmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
            if not srmi_id.has_exposures:
                raise LookupError('no exposures in %s risk model instance for %s' % (srm.name, str(date)))

            # Load model's exposure matrix
            srm.setFactorsForDate(date, modelDB)
            srm_expM_df = srm.loadExposureMatrixDF(srmi_id, modelDB, addExtraCountries=False)

            # Remove unmapped assets
            localAssets = set(srm_expM_df.index).intersection(universeMap[srm])
            srm_expM_df = srm_expM_df.loc[localAssets]

            # If we have an SCM, assign a currency factor
            if scmMarket is not None:
                srm_expM_df.loc[:, srm.numeraire.currency_code] = numpy.ones(len(srm_expM_df.index), float)
                logging.info('Adding %s currency to %s model', srm.numeraire.currency_code, srm.name)
                
            # Rename factors where relevant
            for fct in srm_expM_df.columns:
                if self.factorTypeDict.get(fct, None) is ExposureMatrix.CurrencyFactor:
                    # Leave currency exposures as-is
                    continue
                if (srm, fct) in self.baseModelFactorNameMap:
                    # Rename other factors as necessary
                    nuName = self.baseModelFactorNameMap[(srm, fct)]
                    srm_expM_df.rename(columns={fct:nuName}, inplace=True)
                else:
                    # Drop country factors if already taken by another model
                    srm_expM_df.drop(fct, axis=1, inplace=True)
                    logging.info('Dropping %s from %s model', fct, srm.name)

            # Add to running set of exposures
            if exposureMatrix is None:
                exposureMatrix = srm_expM_df.copy(deep=True)
            else:
                exposureMatrix = pandas.concat([exposureMatrix, srm_expM_df], axis=0)

        # Convert back to an exposure matrix object
        factorNames = [f.name for f in self.factors]
        factorList = list(zip(factorNames, [self.factorTypeDict[f] for f in factorNames]))
        exposureMatrix = exposureMatrix.reindex(index=assetData.universe, columns=factorNames)
        if len(preSPAC) > 0:
            nonCurr = [f.name for f in self.factors if f not in self.currencies]
            exposureMatrix.loc[preSPAC, nonCurr] = numpy.nan
        epxData = Utilities.screen_data(exposureMatrix)
        exposureMatrix = Matrices.ExposureMatrix(assetData.universe, factorList=factorList)
        exposureMatrix.data_ = Utilities.df2ma(epxData.T)

        if self.debuggingReporting:
            estu = self.loadEstimationUniverse(rmi, modelDB, assetData)
            exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, date.year, date.month, date.day),
                    modelDB, marketDB, date, assetData=assetData, dp=self.dplace, compact=True)

        return exposureMatrix

    def generateFactorSpecificReturns(self, modelDB, marketDB, date, buildFMPs=False,
                        internalRun=False, cointTest=False, weeklyRun=False):
        """Loads factor returns for the underlying models, sorts and returns the entire set
        Regression statistics are returned as null
        """

        # Initialise 
        rmi, dummy = self.set_model_instance(date, modelDB)
        self.setFactorsForDate(date, modelDB)
        factorReturns = None
        specificReturns = None

        # Get model universe and asset mappings
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=False)

        # Build mapping of sub-models to assets
        universeMap = self.createAsset2ModelMapping(date, universe, modelDB, marketDB)
        universe = sorted(set(universeMap[self]))

        # Loop through models and collect factor and specific returns
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            logging.info('Preparing %s model factor returns', srm.name)

            # Check that underlying model exists
            srmi_id, srmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
            if not srmi_id.has_returns:
                raise LookupError('no factor returns in %s risk model instance for %s' % (srm.name, str(date)))
            srm.setFactorsForDate(date, modelDB)

            # Load specific returns history - external, not internal
            if internalRun:
                srm_specrets = pandas.Series(numpy.nan, index=universeMap[srm])
            else:
                logging.info('Loading specific returns for %d assets on %s', len(universeMap[srm]), date)
                srm_specrets = srm.loadSpecificReturnsHistory(date, universeMap[srm], [date], modelDB, marketDB).loc[:, date]

            # Add specific returns to running set
            if specificReturns is None:
                specificReturns = srm_specrets.copy(deep=True)
            else:
                specificReturns = pandas.concat([specificReturns, srm_specrets])

            # Load model factor returns
            if srm.twoRegressionStructure and internalRun:
                srm_facrets = srm.loadFactorReturns(date, modelDB, addNursery=False, flag='internal', returnDF=True)
            else:
                srm_facrets = srm.loadFactorReturns(date, modelDB, addNursery=False, returnDF=True)
            
            # Rename factors where relevant
            for fct in srm_facrets.index:
                # Drop currency factors
                if self.factorTypeDict.get(fct, None) is ExposureMatrix.CurrencyFactor:
                    srm_facrets.drop(fct, inplace=True)                   
                elif (srm, fct) in self.baseModelFactorNameMap:
                    # Rename other factors as appropriate
                    nuName = self.baseModelFactorNameMap[(srm, fct)]
                    srm_facrets.rename({fct:nuName}, inplace=True)
                else:
                    # Drop country factors if taken by another model
                    srm_facrets.drop(fct, inplace=True)
                    logging.info('Dropping %s from %s model', fct, srm.name)

            # Add to running set of total factor returns
            if factorReturns is None:
                factorReturns = srm_facrets.copy(deep=True)
            else:
                factorReturns = pandas.concat([factorReturns, srm_facrets])

        # Pull in currency factor returns from currency model
        if self.hasCurrencyFactor:
            crmi, dummy = self.set_model_instance(date, modelDB, rm=self.currencyModel)
            currencyReturns = self.currencyModel.loadCurrencyFactorReturns(crmi, modelDB, returnDF=True)
            self.log.info('loaded %d currencies from %s currency model', len(currencyReturns.index), self.currencyModel.name)
            factorReturns = pandas.concat([factorReturns, currencyReturns])

        # Get into necessary format to return
        factorNames = [f.name for f in self.factors]
        factorReturns = Utilities.screen_data(factorReturns.fillna(0.0).reindex(factorNames).values)
        specificReturns = Utilities.screen_data(specificReturns.reindex(universe).values)

        # Fake "r-square" - just a number for the purposes of branch testing
        logging.info('Adjusted R-Squared=%.6f', factorReturns.sum(axis=None))

        # Create return object
        result = Utilities.Struct()
        result.factorNames = factorNames
        result.factorReturns = factorReturns
        result.universe = universe
        result.specificReturns = specificReturns
        result.regressionStatistics = Matrices.allMasked((len(self.factors), 4))
        result.adjRsquared = None
        result.pcttrade = None
        result.robustWeightMap = dict()
        result.exposureMatrix = None
        result.regression_ESTU = []

        if self.debuggingReporting:
            outfile = open('tmp/factorReturns-%s-%s.csv' % (self.name, date), 'w')
            outfile.write('factor,return,stderr,tstat,prob,constr_wt,fmp_ret\n')
            for i in range(len(result.factorReturns)):
                outfile.write('%s,%.8f,%s,%s,%s,%s,,\n' % \
                        (result.factorNames[i].replace(',',''),
                            result.factorReturns[i],
                            result.regressionStatistics[i,0],
                            result.regressionStatistics[i,1],
                            result.regressionStatistics[i,2],
                            result.regressionStatistics[i,3]))
            outfile.close()

        return result

    def generateFactorSpecificRisk(self, date, modelDB, marketDB, dvaToggle=False, nwToggle=False):
        """Compute the factor-factor covariance matrix and the specific variances for the risk model
        """
        rmi, dummy = self.set_model_instance(date, modelDB)

        # Process dates
        dateList = self.setDateList(date, modelDB)
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            logging.debug('Combining %s model dates', srm.name)
            firstBadDt = self.parseModelHistories(srm, date, dateList, modelDB)

            # Drop dates if there is missing model data
            if firstBadDt is not None:
                dlen = len(dateList)
                dateList = [dt for dt in dateList if dt>firstBadDt]
                logging.warn('Dropping dates from history because of missing data in %s model', srm.name)
                logging.warn('Bad date: %s, history shrunk from %d to %d dates', firstBadDt, dlen, len(dateList))

        # Build factor covariance matrix
        factorCov = self.generateFactorCovarianceMatrix(dateList, modelDB, marketDB)

        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            sortedFN = sorted(factorCov.index)
            tmpFactorCov = factorCov.reindex(index=sortedFN, columns=sortedFN)
            (d, corrMatrix) = Utilities.cov2corr(tmpFactorCov.values, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=sortedFN, rowNames=sortedFN, dp=8)

            # Write variances to flatfile
            var = numpy.diag(tmpFactorCov.values)[:,numpy.newaxis]
            varoutfile = 'tmp/%s-varFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(var, varoutfile, rowNames=sortedFN, dp=8)
            sqrtvar = 100.0 * numpy.sqrt(var)
            self.log.info('Factor risk: (Min, Mean, Max): (%.2f%%, %.2f%%, %.2f%%)',
                    ma.min(sqrtvar, axis=None), numpy.average(sqrtvar, axis=None), ma.max(sqrtvar, axis=None))

            # Write final covariance matrix to flatfile
            covOutFile = 'tmp/%s-covFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(tmpFactorCov, covOutFile, columnNames=sortedFN, rowNames=sortedFN, dp=8)

        # Compute specific risk
        if not self.runFCovOnly:
            specificVars, specificCov, subIssues = self.compute_specific_risk(dateList[:self.maxSRiskObs], date, modelDB, marketDB)

        # Build return object
        ret = Utilities.Struct()
        ret.factorCov = factorCov.values
        ret.subFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.factors)
        if not self.runFCovOnly:
            ret.specificVars = specificVars
            ret.specificCov = specificCov
            ret.subIssues = subIssues
        return ret

    def generateFactorCovarianceMatrix(self, dateList, modelDB, marketDB):
        """Compute the factor-factor covariance matrix for the entire model while preserving
        the individual sub-model matrices
        """

        # Initialise
        factorReturnsMap = dict()
        covarianceMatrixMap = dict()

        # Loop through models and collect factor returns and covariances
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            logging.info('Preparing %s model covariances', srm.name)
            srmi, dummy = self.set_model_instance(dateList[0], modelDB, rm=srm, lookBack=31)

            # Load model factor return histories
            non_currencies = [f for f in srm.factors if f not in srm.currencies]
            if srm.twoRegressionStructure:
                srm_facrets = srm.loadFactorReturnsHistory(dateList[:self.maxFCovObs],
                        modelDB, table_suffix='internal', factorList=non_currencies, returnDF=True)
            else:
                srm_facrets = srm.loadFactorReturnsHistory(dateList[:self.maxFCovObs],
                        modelDB, table_suffix=None, factorList=non_currencies, returnDF=True)

            # Load model covariances
            srm_cov = srm.loadFactorCovarianceMatrix(srmi, modelDB, returnDF=True)
            srm_cov = srm_cov.loc[srm_facrets.index, srm_facrets.index]

            # Rename or prune factors where relevant
            for fct in srm_facrets.index:
                # Rename factors where necessary
                if (srm, fct) in self.baseModelFactorNameMap:
                    nuName = self.baseModelFactorNameMap[(srm, fct)]
                    srm_facrets.rename({fct:nuName}, inplace=True)
                    srm_cov.rename(index={fct:nuName}, columns={fct:nuName}, inplace=True)
                else:
                    # Drop currency or repeated country factors
                    srm_facrets.drop(fct, axis=1, inplace=True)
                    srm_cov.drop(fct, axis=0, inplace=True)
                    srm_cov.drop(fct, axis=1, inplace=True)
                    logging.info('Dropping %s from %s model', fct, srm.name)

            # Add to running totals
            assert srm_facrets.values.shape[0] == srm_cov.values.shape[0]
            covarianceMatrixMap[srm] = srm_cov / 252.0
            factorReturnsMap[srm] = srm_facrets

        # Pull in currency factor returns and covariances from currency model
        modelCurrencyFactors = [f for f in self.factors if f in self.currencies]
        cfReturns, cc, currencySpecificRisk = self.process_currency_block(\
                                    dateList, modelCurrencyFactors, modelDB)
        covarianceMatrixMap[self.currencyModel] = cc
        factorReturnsMap[self.currencyModel] = cfReturns

        # Process the set of sub-matrices and factor returns
        argList = []
        runningFactorList = []
        for (idx, srm) in enumerate(factorReturnsMap.keys()):
            runningFactorList.extend(factorReturnsMap[srm].index)
            argObj = Utilities.Struct()
            argObj.data = factorReturnsMap[srm]
            argList.append(argObj)
            self.covarianceCalculator.configureSubCovarianceMatrix(idx, covarianceMatrixMap[srm])
            logging.info('Setting block %d, model %s, factors: %s, observations: %d',
                    idx, srm.name, len(factorReturnsMap[srm].index), len(factorReturnsMap[srm].columns))

        # Finally, compute the factor covariance matrix
        factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(*argList)

        # Add in the currency specific variances
        for cf in currencySpecificRisk.index:
            factorCov.loc[cf, cf] += currencySpecificRisk.loc[cf] * currencySpecificRisk.loc[cf]

        # Re-order according to list of model factors
        modelFactorNames = [f.name for f in self.factors]
        factorCov = factorCov.reindex(index=modelFactorNames, columns=modelFactorNames)

        # Safeguard to deal with nasty negative eigenvalues
        factorCov = Utilities.forcePositiveSemiDefiniteMatrix(factorCov, min_eigenvalue=0.0)
        factorCov = (factorCov + factorCov.T) / 2.0

        # Report on estimated variances
        ncf = [f.name for f in self.factors if f not in self.currencies]
        stDevs = 100.0 * numpy.sqrt(numpy.diag(factorCov.loc[ncf, ncf].values))
        logging.info('Factor risk: [min: %.2f, max: %.2f, mean: %.2f]',
                min(stDevs), max(stDevs), numpy.average(stDevs))

        return factorCov

    def compute_specific_risk(self, dateList, date, modelDB, marketDB):
        """ Compute asset specific risk and ISC across model universe
        """

        # Get sub-issues active on this day
        rmi, rmgList = self.set_model_instance(date, modelDB)
        subIssues = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=False)
        specRiskDict = dict()
        srm2sidMap = dict()
        specCovDict = defaultdict(dict)

        # Get mapping of assets to sub-models
        universeMap = self.createAsset2ModelMapping(date, subIssues, modelDB, marketDB)
        universe = sorted(set(universeMap[self]))
        assetData = AssetProcessor_V4.AssetProcessor(
                date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
        assetData.process_asset_information(self.rmg, universe=universe)

        # Loop through models and collect specific risks and covariances for all assets
        for (srm, prefix, scmMarket) in self.linkedModelMap:
            logging.info('Preparing %s model specific covariances', srm.name)

            # Check model data exists
            srmi_id, srmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
            if not srmi_id.has_risks:
                raise LookupError('no risks in %s risk model instance for %s' % (srm.name, str(date)))

            # Load specifi
            srm_specRsk, srm_ISC = srm.loadSpecificRisks(srmi_id, modelDB)
            srm2sidMap[srm] = universeMap[srm]

            # Save only those risks and covariances we want from particular model
            for sid in universeMap[srm].intersection(set(srm_specRsk.keys())):
                specRiskDict[sid] = srm_specRsk[sid]
            for sid1 in universeMap[srm].intersection(set(srm_ISC.keys())):
                for sid2 in universeMap[srm].intersection(set(srm_ISC[sid1].keys())):
                    specCovDict[sid1][sid2] = srm_ISC[sid1][sid2]

        # Create mapping from asset to sub-model
        sid2srmMap = Utilities.flip_dict_of_lists(srm2sidMap)
        specRiskDict = pandas.Series(specRiskDict).reindex(index=assetData.universe)
        missingSpecRsk = list(specRiskDict[specRiskDict.isnull()].index)
        if len(missingSpecRsk) > 0:
            logging.warning('%d assets have missing specific risk', len(missingSpecRsk))
        if not self.forceRun:
            assert (len(missingSpecRsk)==0)

        # Loop through CIDs and find those that span multiple models
        runningList = set()
        for cidGroup in assetData.getLinkedCompanyGroups():
            # First deal with sets of multiple CIDs that are linked (e.g. DLCs)
            sidList = []
            for cid in cidGroup:
                sidList.extend(assetData.getCid2SubIssueMapping()[cid])
            srmSet = set([sid2srmMap[sid] for sid in sidList])
            if len(srmSet) > 1:
                runningList = runningList.union(set(sidList))

        # Next deal with issues linked by common CID
        for cid, sidList in assetData.getSubIssueGroups().items():
            srmSet = set([sid2srmMap[sid] for sid in sidList])
            if len(srmSet) > 1:
                runningList = runningList.union(set(sidList))
        logging.info('%d assets whose CIDs span multiple models', len(runningList))

        if len(runningList) > 0:
            specificReturns = None

            # Map issues to be overwritten to issuer data
            partData = AssetProcessor_V4.AssetProcessor(
                    date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
            partData.process_asset_information(self.rmg, universe=sorted(set(runningList)))

            # Loop through models and collect specific returns
            for (srm, prefix, scmMarket) in self.linkedModelMap:
                logging.info('Computing %s model specific covariances', srm.name)

                # Check model data exists
                srmi_id, srmgList = self.set_model_instance(date, modelDB, rm=srm, lookBack=31)
                if not srmi_id.has_returns:
                    raise LookupError('no returns in %s risk model instance for %s' % (srm.name, str(date)))

                # Load specific returns history
                part_univ = universeMap[srm].intersection(set(partData.universe))
                logging.info('Loading specific returns for %d assets, %d dates from %s to %s',
                        len(part_univ), len(dateList), dateList[0], dateList[-1])
                if len(part_univ) > 0:
                    if hasattr(srm, 'hasInternalSpecRets') and srm.hasInternalSpecRets:
                        srm_specrets = srm.loadSpecificReturnsHistory(date, part_univ, dateList, modelDB, marketDB, internal=True)
                    else:
                        srm_specrets = srm.loadSpecificReturnsHistory(date, part_univ, dateList, modelDB, marketDB)
                    # Add to running set of returns
                    if specificReturns is None:
                        specificReturns = srm_specrets.copy(deep=True)
                    else:
                        specificReturns = pandas.concat([specificReturns, srm_specrets])
           
            # Re-order
            specificReturns = specificReturns.reindex(index=partData.universe)

            # ISC info
            scores = self.load_ISC_Scores(date, partData, modelDB, marketDB, returnDF=True)
            self.group_linked_assets(date, partData, modelDB, marketDB)

            # Specific risk computation
            svOverlay = specRiskDict[partData.universe] * specRiskDict[partData.universe] / 252.0
            (partSpecVars, partSpecCov) = self.specificRiskCalculator.\
                    computeSpecificRisks(specificReturns, partData, self, modelDB, rmgList=rmgList,
                            nOkRets=None, scoreDict=scores, svOverlay=svOverlay)

            # Overwrite specific variances with those from the underlying model, and adjust ISC accordingly
            partSpecRsk = numpy.sqrt(partSpecVars)
            for (sid1, iscMap) in partSpecCov.items():
                for (sid2, isc) in iscMap.items():
                    specCovDict[sid1][sid2] = \
                        isc * specRiskDict[sid1] * specRiskDict[sid2] / (partSpecRsk[sid1] * partSpecRsk[sid2])

        # Create vector of specific variances to return
        specificVars = specRiskDict * specRiskDict
                
        # Write specific risk per asset
        if self.debuggingReporting:

            estu = self.loadEstimationUniverse(rmi, modelDB, assetData)
            outfile = 'tmp/specificRisk-%s-%s.csv' % (self.name, date)
            outfile = open(outfile, 'w')
            outfile.write('CID,SID,Name,Type,ESTU,Risk,\n')
            for sid in sorted(assetData.universe):
                outfile.write('%s,%s,%s,' % ( \
                        assetData.getSubIssue2CidMapping()[sid], sid.getSubIDString(),
                        assetData.getNameMap().get(sid, '').replace(',','')))
                if sid in estu:
                    outfile.write('1,%.6f,\n' % (100.0*specRiskDict[sid]))
                else:
                    outfile.write('0,%.6f,\n' % (100.0*specRiskDict[sid]))

            outfile = 'tmp/specificCov-%s-%s.csv' % (self.name, date)
            outfile = open(outfile, 'w')
            outfile.write('GID,SID,Name,Type,GID,SID,Name,Type,Covar,\n')
            sid1List = sorted(specCovDict.keys())
            for sid1 in sid1List:
                sid2List = sorted(specCovDict[sid1].keys())
                for sid2 in sid2List:
                    outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%.6f,\n' % ( \
                        assetData.getSubIssue2CidMapping()[sid1], sid1.getSubIDString(),
                        assetData.getNameMap().get(sid1, '').replace(',',''),
                        assetData.getAssetType().get(sid1, ''),
                        assetData.getSubIssue2CidMapping()[sid2], sid2.getSubIDString(),
                        assetData.getNameMap().get(sid2, '').replace(',',''),
                        assetData.getAssetType().get(sid2, ''),
                        specCovDict[sid1][sid2]))
            outfile.close()

        self.log.debug('computed specific variances')
        return specificVars, specCovDict, assetData.universe


# add Projection Model
class ProjectionModel(FactorRiskModel):
    """ Projection model
    """
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)

        # Set up default estimation universe
        self.masterEstuMap = modelDB.getEstuMappingTable(self.rms_id)
        if self.masterEstuMap is None:
            logging.error('No estimation universe mapping defined')
            assert(self.masterEstuMap is not None)
        logging.info('Estimation universe structure: %d estus', len(self.masterEstuMap))

        # Report on the base model
        logging.info('Projection Model is based on...')
        rm, dm2 = self.baseModel
        logging.info('... (%s, %s)', rm.mnemonic, rm.rms_id)

    def isProjectionModel(self):
        return True

    def setFactorsForDate(self, date, modelDB): 
        """Determine factors from the base model
        """
        # Set up estimation universe parameters
        self.estuMap = copy.deepcopy(self.masterEstuMap)

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)

        # Initialise factors
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        countries = []
        intercept = None
        currencies = []
        localStructureFactors = []
        factorTypeDict = {}
        # get factors from base model
        brm, suffix = self.baseModel
        logging.info('Loading factors from base model: %s', brm.name)
        brm.setFactorsForDate(date, modelDB)

        # Set currency factors
        if brm.hasCurrencyFactor:
            currencies = brm.currencies
            factorTypeDict = {cur.name: ExposureMatrix.CurrencyFactor for cur in currencies}

        # And local structure factors
        if brm.localStructureFactors is not None:
            localStructureFactors = brm.localStructureFactors
            factorTypeDict.update({fct.name: ExposureMatrix.LocalFactor for fct in localStructureFactors})
        else:
            localStructureFactors = []

        # Deal with countries
        if brm.countries is not None:
            countries = brm.countries
            factorTypeDict.update({fct.name: ExposureMatrix.CountryFactor for fct in countries})

        # Set factor attributes, add macro factor into the list
        if brm.intercept is not None:
            allFactors = brm.styles + brm.industries + countries + currencies + self.all_macros + [brm.intercept] + localStructureFactors
            factorTypeDict.update({brm.intercept.name: ExposureMatrix.InterceptFactor})
        else:
            allFactors = brm.styles + brm.industries + countries + currencies + self.all_macros + localStructureFactors

        for f in allFactors:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        self.macros = [m for m in self.all_macros if m.isLive(date)] # track active macro factor

        factorTypeDict.update({fct.name: ExposureMatrix.StyleFactor for fct in brm.styles})
        factorTypeDict.update({fct.name: ExposureMatrix.IndustryFactor for fct in brm.industries})
        factorTypeDict.update({fct.name: ExposureMatrix.MacroFactor for fct in self.macros})
        # Save permanently
        self.nurseryCountries = []
        self.nurseryRMGs = []
        self.styles = brm.styles
        self.currencies = currencies
        self.countries = countries
        self.factors = [f for f in allFactors if f.isLive(date)]
        self.industries = brm.industries
        self.intercept = brm.intercept
        self.localStructureFactors = localStructureFactors
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factorTypeDict = factorTypeDict
        self.validateFactorStructure(date, warnOnly=self.variableStyles)
        self.allFactors = allFactors
        logging.info('Projection model has the following %d factors on %s...', len(self.factors), date)
        logging.info('... %d style factors', len(self.styles))
        logging.info('... %d industry factors', len(self.industries))
        logging.info('... %d country factors', len(self.countries))
        logging.info('... %d currency factors', len(self.currencies))
        logging.info('... %d local factors', len(self.localStructureFactors))
        logging.info('... %d macro factors', len(self.macros))
        if self.intercept is not None:
            logging.info('... %d intercept factors', len([self.intercept]))

    def set_model_instance(self, date, modelDB, rm=None, lookBack=0):
        """Set sub-model instance for given date, or closest 
        precdeing date
        """
        # If no model specified, set it to the base model
        if rm is None:
            rm = self

        # Find most recent model instance within the last lookBack days
        rmi = modelDB.getRiskModelInstance(rm.rms_id, date)
        prevDt = date
        stopDt = date - datetime.timedelta(lookBack)
        while (rmi is None) and (prevDt > stopDt):
            logging.info('No %s model on %s, looking back further', rm.name, prevDt)
            prevDt -= datetime.timedelta(1)
            rmi = modelDB.getRiskModelInstance(rm.rms_id, prevDt)
        if rmi is None:
            raise LookupError('no %s model instances found from %s to %s' % (rm.name, prevDt, date))

        # Find list of non-nursery markets
        if hasattr(rm, 'nurseryRMGs'):
            rmgList = [r for r in rm.rmg if r not in rm.nurseryRMGs]
        else:
            rmgList = rm.rmg

        return rmi, rmgList
            
    def generate_model_universe(self, date, modelDB, marketDB):
        """Generate risk model instance universe and estimation universe.
        Return value is a Struct containing a universe attribute (list of SubIssues)
        containing the model universe and an estimationUniverse attribute
        (list of index positions corresponding to ESTU members).
        """
        self.log.debug('generate_model_universe: begin')
        # Load base model universe
        brm, suffix = self.baseModel
        logging.info('Loading %s model universe *****************************', brm.name)
        brmi, brmgList = self.set_model_instance(date, modelDB, rm=brm, lookBack=31)
        brm_univ = set(modelDB.getRiskModelInstanceUniverse(brmi, returnExtra=False))

        # extract the valid asset pool, there may be assets which was in the base model but then later excluded, for the benefit of extraction code, 
        # the following additional filter is applied to make sure only valid assets are included in the model. This will not make a difference when running
        # daily production, more for the benefit of generating the history.
        baseData = AssetProcessor_V4.AssetProcessor(date, modelDB, marketDB, self.getDefaultAPParameters())
        asset_pool = set(baseData.getModelAssetMaster(self))
        brm_univ = brm_univ.intersection(asset_pool) # exclude invalid assets

        # Process sub-issues
        retData = Utilities.Struct()
        retData.universe = sorted(set(brm_univ))

        # Load base model estimation universe
        logging.info('Retrieving %s estimation universe *****************************', brm.name)
        estu = brm.loadEstimationUniverse(brmi, modelDB)
        estuRunning = []
        for estuName in self.estuMap.keys():
            if not hasattr(self.estuMap[estuName], 'assets'):
                self.estuMap[estuName].assets = []

            if estuName in brm.estuMap:
                # Add model's assets to estimation universe if they haven't already been added
                rmEstu = set(brm.estuMap[estuName].assets)
                rmEstu = rmEstu.intersection(set(retData.universe))
                self.estuMap[estuName].assets.extend(rmEstu)
                estuRunning.extend(rmEstu)

        testData = AssetProcessor_V4.AssetProcessor(date, modelDB, marketDB, self.getDefaultAPParameters())
        testData.process_asset_information(self.rmg, universe=brm_univ)
        if self.debuggingReporting:
            # Output information on assets and their characteristics
            testData.estimationUniverse = sorted(estuRunning)
            estuCls = EstimationUniverse_V4.ConstructEstimationUniverse(\
                    date, testData.universe, self, modelDB, marketDB, debugOutput=self.debuggingReporting)
            estuCls.estimation_universe_reporting(testData, Matrices.ExposureMatrix(testData.universe))

        # Output estimation universe info to log
        for estuName in sorted(self.estuMap.keys()):
            if hasattr(self.estuMap[estuName], 'assets'):
                sidList = self.estuMap[estuName].assets
                mcap_ESTU = testData.marketCaps[sidList].sum(axis=None)
                self.log.info('%s ESTU contains %d assets, %.2f tr %s market cap',
                        estuName, len(sidList), mcap_ESTU / 1e12, self.numeraire.currency_code)
        self.log.info('Universe contains %d assets, %.2f tr %s market cap',
                len(testData.universe), testData.marketCaps.sum(axis=None) / 1e12, self.numeraire.currency_code)

        return retData

    def generateExposureMatrix(self, date, modelDB, marketDB, macDB=None):
        """Generates and returns the exposure matrix for the given date.
        """
        self.log.debug('generateExposureMatrix: begin')

        # Get risk model universe
        rmi, dummy = self.set_model_instance(date, modelDB)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=False)

        # Initialise exposure matrix
        exposureMatrix = None
        if self.debuggingReporting:
            assetData = AssetProcessor_V4.AssetProcessor(
                    date, modelDB, marketDB, self.getDefaultAPParameters(quiet=True))
            assetData.process_asset_information(self.rmg, universe=universe)
        else:
            assetData = Utilities.Struct()
            assetData.universe = sorted(set(universe))

        # Loop through models and collect sub-issues and estimation universes
        brm, suffix = self.baseModel
        logging.info('Processing model exposures with base model %s', brm.name)

        # Check that underlying model exists
        brmi_id, brmgList = self.set_model_instance(date, modelDB, rm=brm, lookBack=31)
        if not brmi_id.has_exposures:
            raise LookupError('no exposures in %s risk model instance for %s' % (brm.name, str(date)))

        # Load model's exposure matrix
        brm.setFactorsForDate(date, modelDB)
        brm_expM = brm.loadExposureMatrix(brmi_id, modelDB, addExtraCountries=False)
        brm_expM_df = brm_expM.toDataFrame()

        # compute beta
        beta_macros, transformed_cov = self.generateMacroBetas(date, modelDB, marketDB, macDB)

        # copy the base exposure matrix
        exposureMatrix = brm_expM_df.copy(deep=True)
        non_ccy_macro_factors = [f for f in self.factors if f not in self.currencies+self.macros]
        non_ccy_macro_factors_names = [i.name for i in non_ccy_macro_factors]

        active_macro_names = [i.name for i in self.macros]
        for i in active_macro_names:
            exposureMatrix[i] = pandas.Series(0.0, index=exposureMatrix.index)
        exposureMatrix[non_ccy_macro_factors_names] = exposureMatrix[non_ccy_macro_factors_names].fillna(0.0)

        modB = exposureMatrix[non_ccy_macro_factors_names].dot(beta_macros) + exposureMatrix[active_macro_names].fillna(0.0)
        exposureMatrix[active_macro_names] = modB
        ######## end of contruct exposure matrix ##########

        # Convert back to an exposure matrix object
        factorNames = [f.name for f in self.factors]
        exposureMatrix = exposureMatrix.reindex(index=assetData.universe, columns=factorNames)
        factorList = list(zip(factorNames, [self.factorTypeDict[f] for f in factorNames]))
        epxData = Utilities.screen_data(exposureMatrix)
        exposureMatrix = Matrices.ExposureMatrix(assetData.universe, factorList=factorList)
        exposureMatrix.data_ = Utilities.df2ma(epxData.T)

        if self.debuggingReporting:
            estu = self.loadEstimationUniverse(rmi, modelDB, assetData)
            exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, date.year, date.month, date.day),
                    modelDB, marketDB, date, assetData=assetData, dp=self.dplace, compact=True)
            beta_macros.to_csv('tmp/beta-%s-%04d%02d%02d.csv' \
                % (self.name, date.year, date.month, date.day),)

        return exposureMatrix

    def load_macro_factor_returns(self, macros, macro_ret_lib, date_list):
        """
        load single date macro factor returns, return an pandas series with factor name as index,
        if factor return is missing for this date, fill with 0.0 and log warning message
        """
        mf_ret_all = pandas.Series()
        for mf in macros:
            mf_ret_series = macro_ret_lib.methods[mf](date_list)
            date = date_list[0]
            if date in mf_ret_series.index: 
                mf_ret = mf_ret_series[date] # get mf return for dates (float)
            else:
                self.log.warning('no macro factor return for %s on %s, filled with 0.0', mf, str(date))
                mf_ret = 0.0 # special case where there was not return for this date, fill with 0.
            mf_ret_all = mf_ret_all.append(pandas.Series([mf_ret], index=[mf]))

        return mf_ret_all

    def load_macro_factor_returns_history(self, macros, macro_ret_lib, date_list):
        """
        load macro factor returns history, return an pandas series with factor name as index,
        if factor return is missing for this date, fill with 0.0 and log warning message
        """
        mf_ret_all = pandas.DataFrame()
        for mf in macros:
            mf_ret_series = macro_ret_lib.methods[mf](date_list)
            mf_ret_all[mf] = mf_ret_series

        return mf_ret_all

    #############################################################################
    def generateFactorSpecificReturns(self, modelDB, marketDB, date, macDB=None, buildFMPs=False,
                        internalRun=False, cointTest=False, weeklyRun=False):
        """Loads factor returns for the underlying models, sorts and returns the entire set
        Regression statistics are returned as null
        """

        # Initialise 
        rmi, dummy = self.set_model_instance(date, modelDB)
        self.setFactorsForDate(date, modelDB)

        factorReturns = None
        specificReturns = None
        # get base model factor and specific returns
        brm, suffix = self.baseModel
        logging.info('Processing base model (%s) factor returns', brm.name)

        # Check that underlying model exists
        brmi_id, brmgList = self.set_model_instance(date, modelDB, rm=brm, lookBack=31)
        if not brmi_id.has_returns:
            raise LookupError('no factor returns in %s risk model instance for %s' % (brm.name, str(date)))
        brm.setFactorsForDate(date, modelDB)

        dateList = self.setDateList(date, modelDB, rmi=brmi_id)

        # populate macro factor returns in the db first
        # add macro factor returns
        macro_ret_lib = MacroFactorReturn.MacroFactorReturn(modelDB, marketDB, macDB)
        # loop through all macro factors, extract return 
        macro_names = [f.name for f in self.macros]

        mf_ret_all = self.load_macro_factor_returns(macro_names, macro_ret_lib, dateList[:5])
        # Get model universe and asset mappings
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=False)

        # Load base model specific returns external only (flag = None is default to external return)
        if not internalRun:
            logging.info('Loading specific returns for %d assets on %s', len(universe), date)
            brm_specrets = brm.loadSpecificReturnsHistory(date, universe, [date], modelDB, marketDB, internal=False).loc[:, date]
        else:
            brm_specrets = pandas.Series(numpy.nan, index=universe)

        # Add specific returns
        specificReturns = brm_specrets.copy(deep=True)

        # Load model factor returns
        if brm.twoRegressionStructure and internalRun:
            brm_facrets = brm.loadFactorReturns(date, modelDB, addNursery=False, flag='internal', returnDF=True)
        else:
            brm_facrets = brm.loadFactorReturns(date, modelDB, addNursery=False, returnDF=True)

        # copy to factorReturns
        factorReturns = brm_facrets.copy(deep=True)
        if not internalRun:
            # compute beta, the try, except is mainly for populating the history where
            # try:
            beta_macros, transformed_cov = self.generateMacroBetas(date, modelDB, marketDB, macDB)
            factorReturns = factorReturns.subtract(beta_macros.dot(mf_ret_all), fill_value=0.0)
            #except:
            #    logging.warning('Insufficient return history to compute beta, non-macro factor returns remain the same as the base model')

        factorReturns = factorReturns.append(mf_ret_all)

        # Get into necessary format to return
        factorNames = [f.name for f in self.factors]
        factorReturns = Utilities.screen_data(factorReturns.fillna(0.0).reindex(factorNames).values)
        specificReturns = Utilities.screen_data(specificReturns.reindex(universe).values)

        # Fake "r-square" - just a number for the purposes of branch testing
        logging.info('Adjusted R-Squared=%.6f', factorReturns.sum(axis=None))

        # Create return object
        result = Utilities.Struct()
        result.factorNames = factorNames
        result.factorReturns = factorReturns
        result.universe = universe
        result.specificReturns = specificReturns
        result.regressionStatistics = Matrices.allMasked((len(self.factors)+len(self.nurseryCountries), 4))
        result.adjRsquared = None
        result.pcttrade = None
        result.robustWeightMap = dict()
        result.exposureMatrix = None
        result.regression_ESTU = []

        if self.debuggingReporting:
            outfile = open('tmp/factorReturns-%s-%s.csv' % (self.name, modelDate), 'w')
            outfile.write('factor,return,stderr,tstat,prob,constr_wt,fmp_ret\n')
            for i in range(len(result.factorReturns)):
                outfile.write('%s,%.8f,%s,%s,%s,%s,,\n' % \
                        (result.factorNames[i].replace(',',''),
                            result.factorReturns[i],
                            result.regressionStatistics[i,0],
                            result.regressionStatistics[i,1],
                            result.regressionStatistics[i,2],
                            result.regressionStatistics[i,3]))
            outfile.close()

        return result

    def generateMacroBetas(self, date, modelDB, marketDB, macDB, dvaToggle=False, nwToggle=False):
        """Compute the factor-factor covariance matrix for the risk model on the given date.
        Then use the factor-factor covariance matrix to compute beta between non-ccy factors and macro factors
        The function composites 3 major part, the first 2 parts are the same as the process of populating any factor-factor
        covariance matrix, loading facor returns, compute factor-factor cov, note here that projection model usually use
        overlapping data. The third part is to compute beta 
        The return value is a dataframe with columns being the macro factors (or any time series factors) and index being the 
        non-ccy factors (styles, industries, countries etc..)
        """

        self.log.info('############# Begin of computing beta for the projection model #############')

        # load base model
        brm, suffix = self.baseModel

        # Toggle on/off DVA or Newey-West settings
        if dvaToggle:
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)
            varDVA_save = self.covarianceCalculator.varParameters.DVAWindow
            corDVA_save = self.covarianceCalculator.corrParameters.DVAWindow
            self.covarianceCalculator.varParameters.DVAWindow = None
            self.covarianceCalculator.corrParameters.DVAWindow = None

        # NW set up
        elif nwToggle:
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)
            varNW_save = self.covarianceCalculator.varParameters.NWLag
            corNW_save = self.covarianceCalculator.corrParameters.NWLag
            self.covarianceCalculator.varParameters.NWLag = 0
            self.covarianceCalculator.corrParameters.NWLag = 0
        
        # get sub-factors and sub-issues active on this day
        rmi, rmgList = self.set_model_instance(date, modelDB)
        brmi, brmgList = brm.set_model_instance(date, modelDB)
        if brmi == None:
            raise LookupError('No base risk model instance for %s' % str(date))

        # only when base model does not have returns, show errors, required for time series regression
        if not brmi.has_returns:
            raise LookupError(
                'factor returns missing in base risk model instance for %s'
                % str(date))

        # Process dates
        dateList = self.setDateList(date, modelDB, rmi=brmi)


        for rm in [brm, self]:
            logging.debug('Combining %s model dates', rm.name)
            firstBadDt = self.parseModelHistories(rm, date, dateList, modelDB)

            # Drop dates if there is missing model data
            if firstBadDt is not None:
                dlen = len(dateList)
                dateList = [dt for dt in dateList if dt>firstBadDt]
                logging.warn('Dropping dates from history because of missing data in %s model', rm.name)
                logging.warn('Bad date: %s, history shrunk from %d to %d dates', firstBadDt, dlen, len(dateList))

        # Remove dates for which many markets are non-trading
        # Remember, datesAndMarkets is in chron. order whereas dateList is reversed
        minMarkets = 0.5
        propTradeDict = modelDB.getRMSStatisticsHistory(brm.rms_id, flag='internal', fieldName='pcttrade')
        badDates = [dt for dt in propTradeDict.keys() if \
                (propTradeDict.get(dt, 0.0) is not None) and (propTradeDict.get(dt, 0.0) < minMarkets)]
        badDates = sorted(dt for dt in badDates if (dt>=min(dateList) and dt<=max(dateList)))
        self.log.info('Removing %d dates with < %.2f%% markets trading: %s',
                    len(badDates), minMarkets*100, ','.join([str(d) for d in badDates]))
        dateList = [dt for dt in dateList if dt not in badDates]
        dateList.sort(reverse=True)
        
        # Load up non-currency factor returns - this contains the macro factors
        nonCurrencyAndMacroFactors = [f for f in self.factors if f not in self.currencies and f not in self.macros]
        nonCurrencyFactors = [f for f in self.factors if f not in self.currencies]
        frData = Utilities.Struct()
        crData = Utilities.Struct()

        # load macro factor returns, note that the current date return may be Nan if the factor return step has not complete
        # the looping dependency is accomdated in the next section where the current date macro factor will be load again.
        factor_datelist = dateList[:self.maxFCovObs]
        if self.twoRegressionStructure:
            macroFactorReturns = self.loadFactorReturnsHistory(
                    factor_datelist, modelDB, table_suffix='internal',
                    screen_data=True, factorList=self.macros, returnDF=True)
        else:
            macroFactorReturns = self.loadFactorReturnsHistory(
                    factor_datelist, modelDB, screen_data=True,
                    factorList=self.macros, returnDF=True)

        # use internal factor returns for covariance matrix, macro factors will have NaN
        if brm.twoRegressionStructure:
            nonCurrencyFactorReturns = brm.loadFactorReturnsHistory(
                    factor_datelist, modelDB, table_suffix='internal',
                    screen_data=True, factorList=nonCurrencyFactors, returnDF=True)
        else:
            nonCurrencyFactorReturns = brm.loadFactorReturnsHistory(
                    factor_datelist, modelDB, screen_data=True,
                    factorList=nonCurrencyFactors, returnDF=True)

        active_macro_names = [i.name for i in self.macros]
        ######## load macro factor returns added for macro projection model #########
        # the nonCurrencyFactorReturns object will contain the macro factors but with Null return values, the
        # following section is to extract macro factor returns and fill into the return matrice
        
        # check if the today's macro factor returns are not in the db, if not, pull from MAC db.
        current_return_date = factor_datelist[0]
        if macroFactorReturns[current_return_date].isnull().values.any():                
            # add macro factor returns
            macro_ret_lib = MacroFactorReturn.MacroFactorReturn(modelDB, marketDB, macDB)
            mf_ret_all = self.load_macro_factor_returns(active_macro_names, macro_ret_lib, current_return_date)
            macroFactorReturns[current_return_date] = mf_ret_all

        # fill current date's macro factor return
        nonCurrencyFactorReturns.loc[active_macro_names] = macroFactorReturns
        
        # pull directly from MAC with full history if some factors have more missing returns 
        # special case to treat newly joined factors with no history stored in rms_factor_internal table
        min_obs_required = max(self.fcParameters.maxObs, self.fvParameters.maxObs)
        ishistory_missing = macroFactorReturns.isnull().sum(axis=1) > 0
        available_return_size = macroFactorReturns.shape[1] - max(macroFactorReturns.isnull().sum(axis=1))
        if ishistory_missing.values.any():
            macro_ret_lib = MacroFactorReturn.MacroFactorReturn(modelDB, marketDB, macDB)
            missing_return_factors = list(macroFactorReturns.index[ishistory_missing])
            if available_return_size < min_obs_required: # only update if insufficient history is provided.
                self.log.warning('Missing return history for [%s], pulled from MAC DB',','.join([str(mf) for mf in missing_return_factors]))
                missing_mf_ret = self.load_macro_factor_returns_history(missing_return_factors, macro_ret_lib, factor_datelist)
                if len(missing_mf_ret) < min_obs_required:
                    self.log.warning('Fill missing return history for [%s] with %s observations, required %s', \
                        ','.join([str(mf) for mf in missing_return_factors]), str(len(missing_mf_ret)), str(min_obs_required))
                # update macro factor returns with history filled.
                macroFactorReturns.update(missing_mf_ret.T)

        # fill current date's macro factor return
        nonCurrencyFactorReturns.loc[active_macro_names] = macroFactorReturns
        nonCurrencyFactorReturns.fillna(0.0, inplace=True)

        frData.data = nonCurrencyFactorReturns
        # Load exposure matrix - for demean
        expM = brm.loadExposureMatrix(brmi, modelDB, addExtraCountries=False)

        # Code for selective demeaning of certain factors or factor types
        if self.fvParameters.selectiveDeMean:
            minHistoryLength = self.fvParameters.deMeanMinHistoryLength
            maxHistoryLength = self.fvParameters.deMeanMaxHistoryLength
            dmFactorNames = []

            # Sift out factors to be de-meaned
            for fType in self.fvParameters.deMeanFactorTypes:
                if fType in expM.factorTypes_:
                    dmFactorNames.extend(expM.getFactorNames(fType))
                # Special hack for Macro, since Macro is not in the base model type, only applied to projection model
                elif fType.name == 'Macro':
                    dmFactorNames.extend(active_macro_names)
                else:
                    dmFactorNames.append(fType)

            means = pandas.Series(0.0, index=nonCurrencyFactorReturns.index)
            frets = nonCurrencyFactorReturns.loc[dmFactorNames, :].fillna(0.0)

            # Cut to required length, or pad with zeros if we fall short
            if len(nonCurrencyFactorReturns.columns) < minHistoryLength:
                extraLen = int(minHistoryLength - len(nonCurrencyFactorReturns.columns))
                frets = pandas.concat([frets, pandas.DataFrame(0.0, index=dmFactorNames, columns=range(extraLen))], axis=1)
            elif len(frets.columns) > maxHistoryLength:
                frets = frets.iloc[:, range(maxHistoryLength)]

            # Compute the weighted mean
            weights = Utilities.computeExponentialWeights(
                    self.fvParameters.deMeanHalfLife, len(frets.columns), equalWeightFlag=False, normalize=True)

            for fac in dmFactorNames:
                means[fac] = numpy.dot(weights, frets.loc[fac,:].values)
            frData.mean = means

        # Compute factor covariance matrix
        if not self.hasCurrencyFactor:
            # Compute regular factor covariance matrix
            factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(frData)
        else:
            # Process currency block
            modelCurrencyFactors = [f for f in self.factors if f in self.currencies]
            cfReturns, cc, currencySpecificRisk = self.process_currency_block(\
                                dateList[:self.maxFCovObs], modelCurrencyFactors, modelDB)
            self.covarianceCalculator.configureSubCovarianceMatrix(1, cc)
            crData.data = cfReturns

            # Post-process the array data a bit more, then compute cov
            factorCov = self.covarianceCalculator.computeFactorCovarianceMatrix(frData, crData)

            # Add in the currency specific variances
            for cf in currencySpecificRisk.index:
                factorCov.loc[cf, cf] += currencySpecificRisk.loc[cf] * currencySpecificRisk.loc[cf]

        ######## Apply procruste transformation to match non-ccy and ccy block ########
    
        # load base model factor covariance matrix:
        brm_cov = brm.loadFactorCovarianceMatrix(brmi, modelDB, returnDF=True)

        transformed_cov = Utilities.procrustes_transform(factorCov, brm_cov)

        # compute beta
        non_ccy_factor_name = [i.name for i in nonCurrencyFactors]
        non_ccy_macro_factor_name = list(set(non_ccy_factor_name).difference(set(active_macro_names)))
  
        beta_macros = Utilities.compute_beta_from_cov(transformed_cov, non_ccy_macro_factor_name, active_macro_names)
        self.log.info('############# End of computing beta for the projection model #############')

        return beta_macros, transformed_cov

    def generateFactorSpecificRisk(self, date, modelDB, marketDB, macDB=None, dvaToggle=False, nwToggle=False):

        """Compute the factor-factor covariance matrix and the
        specific variances for the risk model on the given date.
        Specific risk is computed for all assets in the exposure universe.
        It is assumed that the risk model instance for this day already
        exists.
        The return value is a structure with four fields:
         - subIssues: a list of sub-issue IDs
         - specificVars: an array of the specific variances corresponding
            to subIssues
         - subFactors: a list of sub-factor IDs
         - factorCov: a two-dimensional array with the factor-factor
            covariances corresponding to subFactors
        --override from original one to suit macro projection model
        """

        # compute beta
        beta_macros, transformed_cov = self.generateMacroBetas(date, modelDB, marketDB, macDB)
        
        non_ccy_macro_factors = [f for f in self.factors if f not in self.currencies + self.macros]
        non_ccy_macro_factors_names = [i.name for i in non_ccy_macro_factors]
        active_macros = [f.name for f in self.macros]
        
        # construct cov matrix with beta:
        final_cov = self.restruct_cov_with_beta(transformed_cov, active_macros, non_ccy_macro_factors_names, beta_macros)

        
        # PSD check:
        final_cov_arr = Utilities.forcePositiveSemiDefiniteMatrix(final_cov.values, min_eigenvalue=0.0)
        final_cov_arr = (final_cov_arr + numpy.transpose(final_cov_arr))/2.0
        final_cov = pandas.DataFrame(final_cov_arr, index = final_cov.index, columns = final_cov.columns)

        sub_factors = modelDB.getSubFactorsForDate(date, self.factors)
        sub_factor_name = [i.factor.name for i in sub_factors]
        final_cov = final_cov.reindex(index=sub_factor_name, columns=sub_factor_name)
        # #################### end of projection model section ###################################

        # Set up return structure
        covData = Utilities.Struct()
        covData.factorCov = final_cov.fillna(0.0).values
        covData.subFactors = sub_factors

        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            factorNames = [f.factor.name for f in sub_factors]
            sfIdxMap = dict(zip(factorNames, list(range(len(factorNames)))))
            sortedFN = sorted(factorNames)
            sortedSFIdx = [sfIdxMap[n] for n in sortedFN]
            tmpFactorCov = numpy.take(covData.factorCov, sortedSFIdx, axis=0)
            tmpFactorCov = numpy.take(tmpFactorCov, sortedSFIdx, axis=1)

            (d, corrMatrix) = Utilities.cov2corr(tmpFactorCov, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=sortedFN, rowNames=sortedFN)
            var = numpy.diag(tmpFactorCov)[:,numpy.newaxis]
            sqrtvar = 100.0 * numpy.sqrt(var)
            self.log.info('Factor risk: (Min, Mean, Max): (%.2f%%, %.2f%%, %.2f%%)',
                    ma.min(sqrtvar, axis=None), numpy.average(sqrtvar, axis=None), ma.max(sqrtvar, axis=None))
            varoutfile = 'tmp/%s-varFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(var, varoutfile, rowNames=sortedFN)
            covOutFile = 'tmp/%s-covFinal-%s.csv' % (self.name, date)
            Utilities.writeToCSV(tmpFactorCov, covOutFile, columnNames=sortedFN, rowNames=sortedFN)
            beta_filename = 'tmp/%s-beta-%s.csv' % (self.name, date)
            Utilities.writeToCSV(beta_macros, beta_filename)


        if not self.runFCovOnly:
            # Load specific risks
            brm, suffix = self.baseModel
            logging.info('Processing %s model specific risks', brm.name)

            # Check model data exists
            brmi_id, brmgList = self.set_model_instance(date, modelDB, rm=brm, lookBack=31)
            if not brmi_id.has_risks:
                raise LookupError('no risks in %s risk model instance for %s' % (brm.name, str(date)))
            brmi = modelDB.getRiskModelInstance(brm.rms_id, date)
            brm_specRsk, brm_ISC = brm.loadSpecificRisks(brmi_id, modelDB)
            subIssues = modelDB.getRiskModelInstanceUniverse(brmi, returnExtra=False)
            subIssues = set(subIssues)
            if date >= datetime.date(2020,5,26):
                baseData = AssetProcessor_V4.AssetProcessor(date, modelDB, marketDB, self.getDefaultAPParameters())
                asset_pool = set(baseData.getModelAssetMaster(self))
                subIssues = set(subIssues).intersection(set(asset_pool))
            
            spec_rsk_series = pandas.Series(brm_specRsk)
            spec_rsk_series = spec_rsk_series[subIssues]
            specCovDict = defaultdict(dict)
            for sid1 in subIssues.intersection(set(brm_ISC.keys())):
                for sid2 in subIssues.intersection(set(brm_ISC[sid1].keys())):
                    specCovDict[sid1][sid2] = brm_ISC[sid1][sid2]

            specificVars = spec_rsk_series ** 2
            covData.specificVars = specificVars
            covData.specificCov = specCovDict
            covData.subIssues = subIssues
            self.log.info('Specific Risk bounds (final): [%.3f, %.3f], Mean: %.3f',
                            min(spec_rsk_series), max(spec_rsk_series), ma.average(spec_rsk_series))
        return covData

    def restruct_cov_with_beta(self, orginal_cov, from_factors, to_factors, beta_macros):
        """
        project from_factors to to_factors
        """
        other_factors = orginal_cov.columns.difference(to_factors).difference(from_factors)
        cov = pandas.DataFrame(0.0, index=orginal_cov.columns, columns=orginal_cov.columns)
        cov.loc[from_factors, from_factors] = orginal_cov.loc[from_factors, from_factors]
        cov.loc[other_factors, other_factors] = orginal_cov.loc[other_factors, other_factors]
        cov.loc[other_factors, from_factors] = orginal_cov.loc[other_factors, from_factors]
        cov.loc[from_factors, other_factors] = orginal_cov.loc[from_factors, other_factors]
        cov.loc[to_factors, other_factors] = orginal_cov.loc[to_factors, other_factors] - beta_macros.dot(orginal_cov.loc[from_factors, other_factors]) 
        cov.loc[other_factors, to_factors] = cov.loc[to_factors, other_factors].T
        modcov = orginal_cov.loc[to_factors, to_factors] - beta_macros.dot(orginal_cov.loc[from_factors, from_factors]).dot(beta_macros.T)
        cov.loc[to_factors, to_factors] = modcov
        cov = cov.reindex(index=orginal_cov.columns, columns=orginal_cov.columns)

        return cov

# vim: set softtabstop=4 shiftwidth=4:

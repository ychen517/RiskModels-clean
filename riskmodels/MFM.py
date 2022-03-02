import copy
import datetime
import time
import logging
import numpy
import numpy.ma as ma
import numpy.linalg as linalg
import pandas
import sys
from collections import defaultdict
from riskmodels import GlobalExposures
from riskmodels import MarketIndex
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels import ReturnCalculator
from riskmodels import RegressionToolbox
from riskmodels import ProcessReturns
from riskmodels import RiskCalculator
from riskmodels import Standardization
from riskmodels import LegacyUtilities as Utilities
from riskmodels import StyleExposures
from riskmodels import EstimationUniverse
from riskmodels import Ugliness
from riskmodels import FactorReturns
from riskmodels import LegacyFactorReturns
from riskmodels import LegacyModelParameters as ModelParameters
from riskmodels import AssetProcessor
from riskmodels import Classification
from riskmodels import Outliers
from riskmodels.Factors import *

# Temporary hack for merging in version 4 growth fix
MERGE_START_DATE = datetime.date(2018, 2, 16)
MERGE_END_DATE = datetime.date(2018, 3, 16)

class FactorRiskModel:
    """Abstract class defining a factor risk model.
    """
    # Various factor-type defaults
    blind = []
    styles = []
    countries = []
    industries = []
    localStructureFactors = []
    regionalIntercepts = []
    macro_core = []
    macro_market_traded= []
    macro_equity = []
    macro_sectors = []
    intercept = None
    industryClassification = None
    industrySchemeDict = None
    additionalCurrencyFactors = []
    nurseryRMGs = []

    # Default parameters for styles and various extras
    grandfatherParameters = [1, 0]
    returnsTimingId = None
    allCurrencies = False
    sensitivityNumeraire = 'XDR'
    proxyDividendPayout = True
    returnHistory = 250

    # Legacy settings
    xrBnds = [-15.0, 15.0]
    newExposureFormat = True
    quarterlyFundamentalData = False
    allowStockDividends = False
    allowETFs = False
    fixRMReturns = False
    removeNTDsFromRegression = True
    twoRegressionStructure = False
    legacyMCapDates = True

    # Run-mode options
    debuggingReporting = False
    dplace = 6
    forceRun = False

    # V3 model parameters
    variableStyles = False
    standardizationStats = False
    useRobustRegression = True
    fancyMAD = False
    legacyEstGrowth = False
    estuAssetTypes = ['All-Com', 'REIT']
    nDev=[8.0, 8.0]
    globalDescriptorModel = False
    nurseryCountries = []
    hardCloning = True
    noProxyType = ['NonEqETF', 'ComETF', 'StatETF']
    gicsDate = datetime.date(2014,3,1)

    # Controls use of free-float (test code currently deactivated)
    freeFloat = Utilities.Struct()
    freeFloat.useFFA = False
    freeFloat.backFill = False
    freeFloat.cache = None
    
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
        rmInfo = modelDB.getRiskModelInfo(self.rm_id, self.revision)
        self.rms_id = rmInfo.serial_id
        self.name = rmInfo.name
        self.description = rmInfo.description
        if self.rms_id < 0:
            self.newExposureFormat = True
        if self.quarterlyFundamentalData:
            self.log.info('...Using quarterly fundamental data')
        if self.newExposureFormat:
            self.log.debug('...Using new exposure table format')
        else:
            self.log.warning('...Using old exposure table format')
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
        self.modelHack = Ugliness.ModelHacker(self.__class__)
        self.exposureStandardization = Standardization.SimpleStandardization(
                debuggingReporting=self.debuggingReporting)
        self.legacyISCSwitchDate = datetime.date(2015, 2, 28)

        if self.industryClassification is not None and \
                self.industrySchemeDict is None:
            # Only override when no industry schemeDict is found
            self.industrySchemeDict = {datetime.date(1950, 1, 1):\
                                           self.industryClassification}

    def isStatModel(self):
        return False

    def isRegionalModel(self):
        return len(self.rmg) > 1

    def isCurrencyModel(self):
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
    
    def generate_eligible_universe(self, modelDate, data, modelDB, marketDB, capBounds=None,
            assetTypes=['All-Com', 'REIT'],
            excludeTypes=['ComWI', 'UnCls', 'StatETF', 'LLC', 'LP', 'NonEqETF', 'ComETF']):
        """Creates subset of eligible assets for consideration
        in estimation universes
        """
        self.log.info('generate_eligible_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)
        universe = buildEstu.assets
        n = len(universe)
        stepNo = 0
        if assetTypes is None:
            assetTypes = []
        if excludeTypes is None:
            excludeTypes = []
        logging.info('Eligible Universe currently stands at %d stocks', n)

        # Remove assets from the exclusion table
        stepNo+=1
        logging.info('Step %d: Applying exclusion table', stepNo)
        (estuIdx, nonest) = buildEstu.apply_exclusion_list(modelDate)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove cloned assets
        if len(data.hardCloneMap) > 0:
            stepNo+=1
            logging.info('Step %d: Removing cloned assets', stepNo)
            cloneIdx = [data.assetIdxMap[sid] for sid in data.hardCloneMap.keys()]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(cloneIdx, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Include by asset type field
        stepNo+=1
        logging.info('Step %d: Include by asset types %s', stepNo, ','.join(assetTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Pull out Chinese H shares and redchips to save for later
        stepNo+=1
        assetTypes = ['HShares', 'RCPlus']
        logging.info('Step %d: Getting list of Chinese assets: %s', stepNo, ','.join(assetTypes))
        (chinese, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=assetTypes, excludeFields=None,
                baseEstu = estuIdx)
        if len(chinese) > 0:
            logging.info('...Step %d: Found %d Chinese H-shares and Redchips', stepNo, len(chinese))

        # Exclude by asset type field
        stepNo+=1
        logging.info('Step %d: Exclude default asset types %s', stepNo, ','.join(excludeTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=excludeTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove various types of DRs and foreign listings
        stepNo+=1
        exTypes = ['NVDR', 'GlobalDR', 'DR', 'TDR', 'AmerDR', 'FStock', 'CDI']
        logging.info('Step %d: Exclude DR asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove trusts, funds and other odds and ends
        stepNo+=1
        exTypes = ['InvT', 'UnitT', 'CEFund', 'Misc']
        logging.info('Step %d: Exclude fund/trust asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove Chinese A and B shares
        stepNo+=1
        exTypes = ['AShares', 'BShares']
        logging.info('Step %d: Exclude Chinese asset types %s', stepNo, ','.join(exTypes))
        (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=None, excludeFields=exTypes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Remove assets classed as foreign via home market classification
        dr_indices = []
        if len(data.foreign) > 0:
            stepNo+=1
            logging.info('Step %d: Excluding foreign listings by home country mapping', stepNo)
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            (estuIdx, nonest) = buildEstu.exclude_specific_assets(dr_indices, baseEstu=estuIdx)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove assets classed as foreign via home market classification - part 2
        stepNo+=1
        logging.info('Step %d: Excluding foreign listings by market classification', stepNo)
        (estuIdx, nonest) = buildEstu.exclude_by_market_classification(
                modelDate, 'HomeCountry', 'REGIONS', [r.mnemonic for r in self.rmg], baseEstu=estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Weed out foreign issuers by ISIN country prefix
        stepNo+=1
        logging.info('Step %d: Excluding foreign listings by ISIN prefix', stepNo)
        (estuIdx, nonest)  = buildEstu.exclude_by_isin_country(
                [r.mnemonic for r in self.rmg], modelDate, baseEstu=estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Manually add H-shares and Red-Chips back into list of eligible assets
        stepNo+=1
        logging.info('Step %d: Adding back %d H-Shares and Redchips', stepNo, len(chinese))
        estuIdx = set(estuIdx).union(set(chinese))
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe up by %d and currently stands at %d stocks',
                    stepNo, len(estuIdx)-n, len(estuIdx))
            n = len(estuIdx)

        # Logic to selectively filter RTS/MICEX Russian stocks
        rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic=='RU']
        if len(rmgID) > 0:
            stepNo+=1
            logging.info('Step %d: Cleaning up Russian assets', stepNo)
            # Get lists of RTS and MICEX-quoted assets
            (rtsIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='RTS', excludeFields=None,
                    baseEstu = estuIdx)
            (micIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields='MIC', excludeFields=None,
                    baseEstu = estuIdx)
            # Find companies with multiple listings which include lines
            # on both exchanges
            rtsDropList = []
            for (groupID, sidList) in data.subIssueGroups.items():
                if len(sidList) > 1:
                    subIdxList = [data.assetIdxMap[sid] for sid in sidList]
                    rtsOverlap = set(rtsIdx).intersection(subIdxList)
                    micOverLap = set(micIdx).intersection(subIdxList)
                    if len(micOverLap) > 0:
                        rtsDropList.extend(rtsOverlap)
            estuIdx = estuIdx.difference(rtsDropList)
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Remove preferred stock except for selected markets
        prefMarketList = ['BR','CO']
        prefAssetBase = []
        mktsUsed = []
        for mkt in prefMarketList:
            # Find assets in our current estu that are in the relevant markets
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) > 0:
                prefAssetBase.extend(data.rmgAssetMap[rmgID[0]])
                mktsUsed.append(mkt)
        baseEstuIdx = [data.assetIdxMap[sid] for sid in prefAssetBase]
        baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
        if len(baseEstuIdx) > 0:
            # Find which of our subset of assets are preferred stock
            stepNo+=1
            logging.info('Step %d: Dropping preferred stocks NOT on markets: %s', stepNo, ','.join(mktsUsed))
            (prefIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=['All-Pref'], excludeFields=None,
                    baseEstu=baseEstuIdx)
            # Get rid of preferred stock in general
            (estuIdx, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, data,
                    includeFields=None, excludeFields=['All-Pref'],
                    baseEstu=estuIdx)
            # Add back in allowed preferred stock
            estuIdx = list(set(estuIdx).union(prefIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                        stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        # Get rid of some exchanges we don't want
        exchangeCodes = ['REG']
        if modelDate.year > 2009:
            exchangeCodes.extend(['XSQ','OFE','XLF']) # GB junk
        stepNo+=1
        logging.info('Step %d: Removing stocks on exchanges: %s', stepNo, ','.join(exchangeCodes))
        (estuIdx, nonest) = buildEstu.exclude_by_market_type(
                modelDate, data, includeFields=None, excludeFields=exchangeCodes,
                baseEstu = estuIdx)
        if n != len(estuIdx):
            logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
            n = len(estuIdx)

        # Limit stocks to certain exchanges in select countries
        mktExchangeMap = {'CA': ['TSE'],
                          'JP': ['TKS-S1','TKS-S2','OSE-S1','OSE-S2','TKS-MSC','TKS-M','JAS']
                         }
        if modelDate.year > 2009:
            mktExchangeMap['US'] = ['NAS','NYS','IEX','EXI']
        for mkt in mktExchangeMap.keys():
            rmgID = [rmg.rmg_id for rmg in self.rmg if rmg.mnemonic==mkt]
            if len(rmgID) < 1:
                continue
            stepNo+=1
            logging.info('Step %d: Dropping %s stocks NOT on exchanges: %s', stepNo,
                    mkt, ','.join(mktExchangeMap[mkt]))

            # Pull out subset of assets on relevant market
            baseEstuIdx = [data.assetIdxMap[sid] for sid in data.rmgAssetMap[rmgID[0]]]
            baseEstuIdx = list(set(baseEstuIdx).intersection(set(estuIdx)))
            if len(baseEstuIdx) < 1:
                continue

            # Determine whether they are on permitted exchange
            (mktEstuIdx, nonest) = buildEstu.exclude_by_market_type(
                    modelDate, data, includeFields=mktExchangeMap[mkt], excludeFields=None,
                    baseEstu = baseEstuIdx)
                 
            # Remove undesirables
            nonMktEstuIdx = set(baseEstuIdx).difference(set(mktEstuIdx))
            estuIdx = list(set(estuIdx).difference(nonMktEstuIdx))
            if n != len(estuIdx):
                logging.info('...Step %d: Eligible Universe down %d and currently stands at %d stocks',
                    stepNo, n-len(estuIdx), len(estuIdx))
                n = len(estuIdx)

        estu = [universe[idx] for idx in estuIdx]
        data.eligibleUniverseIdx = estuIdx
        logging.info('%d eligible assets out of %d total', len(estu), len(universe))
        self.log.info('generate_eligible_universe: end')
        return estu

    def estimation_universe_reporting(self, modelDate, data, modelDB, marketDB):
        # Write various files for debugging and reporting

        if not hasattr(data, 'assetTypeDict'):
            data.assetTypeDict = AssetProcessor.get_asset_info(\
                    modelDate, data.universe, modelDB, marketDB, 'ASSET TYPES', 'Axioma Asset Type')
        if not hasattr(data, 'marketTypeDict'):
            data.marketTypeDict = AssetProcessor.get_asset_info(\
                    modelDate, data.universe, modelDB, marketDB, 'REGIONS', 'Market')
        if not hasattr(data, 'hardCloneMap'):
            data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)
        if not hasattr(data, 'dr2Underlying'):
            data.dr2Underlying = modelDB.getDRToUnderlying(modelDate, data.universe, marketDB)
        if not hasattr(data, 'subIssueGroups'):
            data.subIssueGroups = modelDB.getIssueCompanyGroups(modelDate, data.universe, marketDB)
        if not hasattr(data, 'hardCloneMap'):
            data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe)
        if not hasattr(data, 'issuerMarketCaps'):
            data.issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
                    data, modelDate, self.numeraire, modelDB, marketDB,
                    debugReport=self.debuggingReporting)
            data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()
        data.sidToCIDMap = dict(zip(data.universe, [sid.getSubIDString() for sid in data.universe]))
        for (groupID, sidList) in data.subIssueGroups.items():
            for sid in sidList:
                data.sidToCIDMap[sid] = groupID

        # Get subissue to ISIN and name maps
        mdIDs = [sid.getModelID() for sid in data.universe]
        mdIDtoISINMap = modelDB.getIssueISINs(modelDate, mdIDs, marketDB)
        mdIDtoSEDOLMap = modelDB.getIssueSEDOLs(modelDate, mdIDs, marketDB)
        mdIDtoCUSIPMap = modelDB.getIssueCUSIPs(modelDate, mdIDs, marketDB)
        mdIDtoTickerMap = modelDB.getIssueTickers(modelDate, mdIDs, marketDB)
        mdIDtoNameMap = modelDB.getIssueNames(modelDate, mdIDs, marketDB)
        data.assetISINMap = dict()
        data.assetSEDOLMap = dict()
        data.assetCUSIPMap = dict()
        data.assetTickerMap = dict()
        data.assetNameMap = dict()
        for (mid, sid) in zip(mdIDs, data.universe):
            if mid in mdIDtoISINMap:
                data.assetISINMap[sid] = mdIDtoISINMap[mid]
            if mid in mdIDtoSEDOLMap:
                data.assetSEDOLMap[sid] = mdIDtoSEDOLMap[mid]
            if mid in mdIDtoCUSIPMap:
                data.assetCUSIPMap[sid] = mdIDtoCUSIPMap[mid]
            if mid in mdIDtoTickerMap:
                data.assetTickerMap[sid] = mdIDtoTickerMap[mid]
            if mid in mdIDtoNameMap:
                data.assetNameMap[sid] = mdIDtoNameMap[mid]

        mcap_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        mcap_ESTU = numpy.array(mcap_ESTU*100, int) / 100.0
        sidList = [data.universe[idx].getSubIDString() for idx in data.estimationUniverseIdx]

        # Output weights
        outName = 'tmp/estuWt-%s-%s.csv' % (self.mnemonic, modelDate)
        Utilities.writeToCSV(mcap_ESTU, outName, rowNames=sidList)

        # Output type of each asset
        typeList = [data.assetTypeDict.get(data.universe[idx], None) for idx in data.estimationUniverseIdx]
        outName = 'tmp/estuTypes-%s-%s.csv' % (self.mnemonic, modelDate)
        AssetProcessor.dumpAssetListForDebugging(outName, sidList, typeList)

        # Output list of available types in the estu
        typeDict = dict(zip(sidList, typeList))
        typeList = list(set(typeList))
        countList = []
        for atype in typeList:
            nType = len([sid for sid in sidList if typeDict[sid] == atype])
            countList.append(nType)
        outName = 'tmp/AssetTypes-ESTU-%s.csv' % self.mnemonic
        AssetProcessor.dumpAssetListForDebugging(outName, typeList, countList)

        # Output asset qualify flag
        data.exposureMatrix.dumpToFile('tmp/estu-expM-%s-%04d%02d%02d.csv'\
                % (self.name, modelDate.year, modelDate.month, modelDate.day),
                modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                assetType=data.assetTypeDict, subIssueGroups=data.subIssueGroups)

        # Look at the various levels of market cap
        capArray = Matrices.allMasked((len(data.universe), 3))
        capArray[:,0] = data.marketCaps
        capArray[:,1] = data.issuerTotalMarketCaps
        for (idx, sid) in enumerate(data.universe):
            if data.DLCMarketCap[idx] != data.issuerTotalMarketCaps[idx]:
                capArray[idx,2] = data.DLCMarketCap[idx]
        outName = 'tmp/mCap-%s-%s.csv' % (self.mnemonic, modelDate)
        colNames = ['mcap', 'issuer', 'DLC']
        sidList = [sid.getSubIDString() for sid in data.universe]
        Utilities.writeToCSV(capArray, outName, rowNames=sidList, columnNames=colNames)

        # Output estu composition by country/industry (if relevant)
        for fType in [ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor]:
            factorIdxList = data.exposureMatrix.getFactorIndices(fType)
            if len(factorIdxList) < 1:
                continue
            factorNameList = data.exposureMatrix.getFactorNames(fType)
            exp_ESTU = ma.take(data.exposureMatrix.getMatrix(), data.estimationUniverseIdx, axis=1)
            mCapList = []
            # Loop round set of factors
            for (j,i) in enumerate(factorIdxList):
                factorMCap = ma.sum(ma.filled(ma.masked_where(\
                        ma.getmaskarray(exp_ESTU[i]), mcap_ESTU), 0.0), axis=None)
                mCapList.append(factorMCap)
            outName = 'tmp/estuMCap-%s-%s.csv' % (fType.name, modelDate)
            mCapList = mCapList / numpy.sum(mCapList, axis=None)
            Utilities.writeToCSV(numpy.array(mCapList), outName, rowNames=factorNameList)

        if len(data.hardCloneMap) > 0:
            fileName = 'tmp/cloneList-%s.csv' % str(modelDate)
            outFile = open(fileName, 'w')
            outFile.write('Slave,Master,\n')
            for slv in data.hardCloneMap.keys():
                mst = data.hardCloneMap[slv]
                outFile.write('%s,%s,\n' % (slv.getSubIdString(), mst.getSubIdString()))
            outFile.close()

        if len(data.dr2Underlying) > 0:
            fileName = 'tmp/dr_2_under-%s.csv' % str(modelDate)
            outFile = open(fileName, 'w')
            outFile.write('DR,Underlying,\n')
            for slv in data.dr2Underlying.keys():
                mst = data.dr2Underlying[slv]
                if (slv is not None) and (mst is not None):
                    outFile.write('%s,%s,\n' % (slv.getSubIdString(), mst.getSubIdString()))
            outFile.close()

        # Get list of issuers with more than one line of stock in the estu
        multList = []
        count = 0
        estuSidList = set([data.universe[idx] for idx in data.estimationUniverseIdx])
        for (groupID, sidList) in data.subIssueGroups.items():
            intersectList = set(sidList).intersection(estuSidList)
            if len(intersectList) > 1:
                multList.extend(list(intersectList))
                count += 1
        if count > 0:
            logging.info('%d issuers with more than one line in the estu', count)
        fileName = 'tmp/estu-multiples-%s.csv' % str(modelDate)
        outFile = open(fileName, 'w')
        outFile.write(',CID,Name,Type,Market\n')
        for sid in multList:
            outFile.write('%s,%s,%s,%s,%s,\n' % \
                    (sid.getSubIDString(), data.sidToCIDMap[sid], data.assetNameMap.get(sid, None),
                        data.assetTypeDict.get(sid, None), data.marketTypeDict.get(sid, None)))
        outFile.close()

        return

    def checkTrackedAssets(self, universe, assetIdxMap, estuIdx):
        if not hasattr(self, 'trackList'):
            if hasattr(self, 'baseModel') and hasattr(self.baseModel, 'trackList'):
                self.trackList = self.baseModel.trackList
                self.dropList = self.baseModel.dropList
                self.addList = self.baseModel.addList
            else:
                return
        for sid in self.trackList:
            if sid in self.dropList:
                if assetIdxMap[sid] in estuIdx:
                    logging.info('Asset %s added back to estimation universe process', sid.getSubIDString())
                    self.addList.append(sid)
                    self.dropList.remove(sid)
            elif sid in self.addList:
                if assetIdxMap[sid] not in estuIdx:
                    logging.info('Asset %s dropped from estimation universe process', sid.getSubIDString())
                    self.dropList.append(sid)
                    self.addList.remove(sid)
            else:
                if assetIdxMap[sid] not in estuIdx:
                    logging.info('Asset %s dropped from estimation universe process', sid.getSubIDString())
                    self.dropList.append(sid)
        return

    def excludeAssetTypes(self, date, data, modelDB, marketDB, estuObj, baseEstu,
                            excludeTypes=['ComWI', 'UnCls', 'StatETF', 'LLC', 'LP', 'NonEqETF', 'ComETF']):
        """Fix for pre-V3 models to exclude certain undesirables from
        the estimation universe
        """
        data.assetTypeDict = AssetProcessor.get_asset_info(
                date, data.universe, modelDB, marketDB,
                'ASSET TYPES', 'Axioma Asset Type')
        (estu, nonest) = estuObj.exclude_by_asset_type(
                date, data,
                includeFields=None, excludeFields=excludeTypes, baseEstu=baseEstu)
        return estu, nonest

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
        assetCurrMap = modelDB.getTradingCurrency(returns.dates[-1],
                returns.assets, marketDB, returnType='id')
        if drAssetCurrMap is not None:
            for sid in drAssetCurrMap.keys():
                assetCurrMap[sid] = drAssetCurrMap[sid]
        # Report on assets which are missing a trading currency
        missingTC = [sid for sid in assetCurrMap.keys()\
                if assetCurrMap[sid] == None]
        if len(missingTC) > 0:
            self.log.warning('Missing trading currency for %d assets: %s',
                          len(missingTC), missingTC)
        
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
        if len(returns.dates) > 1 and not isinstance(self, StatisticalFactorModel):
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
                rfHistory.data = rfHistory.data.filled(0.0)
                
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
    
    def computePredictedBeta(self, date, modelData, 
                             modelDB, marketDB, globalBetas=False):
        """Compute the predicted beta of the assets for the given risk
        model instance, i.e., all assets that have specific risk.
        modelData is a Struct that contains the exposure
        matrix (exposureMatrix), factor covariance matrix (factorCovariance),
        and specific risk (specificRisk) of the instance.
        The return value is a list of (SubIssue, value) tuples containing
        all assets for which a specific risk is available.
        """
        # Set up MarketIndexSelector for local betas
        if not hasattr(self, 'indexSelector'):
            self.indexSelector = MarketIndex.\
                            MarketIndexSelector(modelDB, marketDB)
        
        goodAssets = list(modelData.specificRisk.keys())
        allAssets = []
        predBetas = []
        # Find home country assignments for DR-aware regional models
        if len(self.rmg) > 1:
            (univ, mcaps, dr) = self.process_asset_country_assignments(
                        date, goodAssets, numpy.zeros(len(goodAssets)), 
                        modelDB, marketDB)
        
        if globalBetas:
            self.log.info('Computing predicted (regional) betas')
            rmi = modelDB.getRiskModelInstance(self.rms_id, date)
            estu = self.loadEstimationUniverse(rmi, modelDB)
            assets = list(set(goodAssets).intersection(estu))
            mcapDates = modelDB.getDates(self.rmg, date, 19)
            mcaps = ma.filled(modelDB.getAverageMarketCaps(
                    mcapDates, assets, self.numeraire.currency_id, marketDB), 0.0)
            marketPortfolio = list(zip(assets, mcaps / ma.sum(mcaps)))
            predBetas = Utilities.generate_predicted_betas(assets,
                    modelData.exposureMatrix, modelData.factorCovariance, 
                    modelData.specificRisk, marketPortfolio, self.forceRun)
            return list(zip(assets, predBetas))

        for rmg in self.rmg:
            self.log.info('Computing predicted (local) betas for %s', rmg.description)
            try:
                rmg_assets = self.rmgAssetMap[rmg.rmg_id]
            except:
                rmg_assets = modelDB.getActiveSubIssues(rmg, date)
            beta_assets = list(set(goodAssets).intersection(rmg_assets))

            # Hack for Domestic China A-Shares
            if rmg.description == 'China':
                # If single-country China model, include B-shares too
                if len(self.rmg)==1:
                    domesticSubIssues = beta_assets
                else:
                    data = Utilities.Struct()
                    data.universe = beta_assets
                    data.marketCaps = numpy.zeros(len(beta_assets))
                    data.exposureMatrix = Matrices.ExposureMatrix(beta_assets)
                    data.exposureMatrix.addFactor(rmg.description, numpy.ones(
                            len(beta_assets)), Matrices.ExposureMatrix.CountryFactor)
                    data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(beta_assets)])
                    (a_idx, b_idx, h_idx, other) = MarketIndex.\
                            process_china_share_classes(data, date, modelDB, marketDB)
                    domesticSubIssues = [beta_assets[i] for i in a_idx]

                allRMGs = modelDB.getAllRiskModelGroups(inModels=False)
                dcRMG = [r for r in allRMGs if r.description=='Domestic China'][0]
                domesticMarketPortfolio = modelDB.getRMGMarketPortfolio(dcRMG, date, amp='XC-LMS')
                assert(domesticMarketPortfolio is not None)
                if len(domesticMarketPortfolio) > 0:
                    predBetas.extend(Utilities.generate_predicted_betas(
                            domesticSubIssues,
                            modelData.exposureMatrix, modelData.factorCovariance, 
                            modelData.specificRisk, domesticMarketPortfolio, self.forceRun,
                            debugging=self.debuggingReporting))
                    allAssets.extend(domesticSubIssues)

                if len(self.rmg) > 1:
                    beta_assets = [beta_assets[i] for i in (b_idx + h_idx + other)]
                else:
                    return list(zip(allAssets, predBetas))

            marketPortfolio = modelDB.getRMGMarketPortfolio(rmg, date, amp='%s-LMS' % rmg.mnemonic)
            assert(marketPortfolio is not None)
            if len(marketPortfolio) > 0:
                predBetas.extend(Utilities.generate_predicted_betas(beta_assets,
                        modelData.exposureMatrix, modelData.factorCovariance, 
                        modelData.specificRisk, marketPortfolio, self.forceRun,
                        debugging=self.debuggingReporting))
                allAssets.extend(beta_assets)
        # A bit of reporting
        if predBetas:
            self.log.info('Predicted beta bounds: [%.3f, %.3f], Median: %.3f',\
                          min(predBetas), max(predBetas), ma.median(predBetas))
        else:
            self.log.warning('No predicted betas generated')
        missingBeta = set(goodAssets).difference(set(allAssets))
        if len(missingBeta) > 0:
            self.log.warning('Missing predicted betas: %d out of %d total assets',\
                    len(missingBeta), len(goodAssets))
        return list(zip(allAssets, predBetas))
    
    def computePredictedBetaV3(self, date, modelData, modelDB, marketDB):
        """Compute the predicted beta of the assets for the given risk
        model instance, i.e., all assets that have specific risk.
        modelData is a Struct that contains the exposure
        matrix (exposureMatrix), factor covariance matrix (factorCovariance),
        and specific risk (specificRisk) of the instance.
        The return value is a list of (SubIssue, value) tuples containing
        all assets for which a specific risk is available.
        """
        if len(self.rmg) > 1:
            SCM = False
        else:
            SCM = True
        # Set up MarketIndexSelector for local betas
        if not hasattr(self, 'indexSelector'):
            self.indexSelector = MarketIndex.\
                            MarketIndexSelector(modelDB, marketDB)

        # Sort out some factor information
        factorIdxMap = dict(zip(modelData.factors, list(range(len(modelData.factors)))))
        currencyFactors = []
        nonCurrencyIdx = None
        if hasattr(self, 'currencies'):
            currencyFactors = [f for f in self.factors if f in self.currencies]
            if len(currencyFactors) > 0:
                nonCurrencyIdx = [factorIdxMap[f] for f in self.factors if f not in currencyFactors]
        styleFactors = [f for f in self.factors if f in self.styles]
        if len(styleFactors) > 0:
            nonStyleIdx = [factorIdxMap[f] for f in self.factors if f not in styleFactors
                    and f not in currencyFactors]
        else:
            nonStyleIdx = None

        allAssets = []
        localBetas = dict()
        legacyBetas = dict()
        globalBetas = dict()
        expMatrix = modelData.exposureMatrix
        goodAssets = list(modelData.specificRisk.keys())
         
        # Find home country assignments for DR-aware regional models
        data = AssetProcessor.process_asset_information(
                date, goodAssets, self.rmg, modelDB, marketDB,
                checkHomeCountry=(SCM==0),
                numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun)

        # Deal with global beta first
        if not SCM:
            self.log.info('Computing predicted (regional) betas')
            rmi = modelDB.getRiskModelInstance(self.rms_id, date)
            estu = self.loadEstimationUniverse(rmi, modelDB)
            assets = list(set(goodAssets).intersection(estu))
            # Construct on-the-fly market portfolio
            mcapDates = modelDB.getDates(self.rmg, date, 19)
            mcaps = ma.filled(modelDB.getAverageMarketCaps(
                    mcapDates, assets, self.numeraire.currency_id, marketDB), 0.0)
            marketPortfolio = list(zip(assets, mcaps / ma.sum(mcaps)))
            if self.debuggingReporting:
                outData = mcaps / ma.sum(mcaps)
                idList = idList = [s.getSubIDString() for s in assets]
                Utilities.writeToCSV(outData, 'tmp/betaMktPort.csv', rowNames=idList)
            # Compute the betas
            predBetas = Utilities.generate_predicted_betas(
                    goodAssets, expMatrix,
                    modelData.factorCovariance, modelData.specificRisk,
                    marketPortfolio, self.forceRun,
                    debugging=self.debuggingReporting)
            globalBetas =  dict(zip(goodAssets, predBetas))
            allAssets.extend(goodAssets)

        # Loop through RMGs and compute local betas
        for rmg in self.rmg:
            self.log.info('Computing predicted (local) betas for %s', rmg.description)
            try:
                rmg_assets = data.rmgAssetMap[rmg.rmg_id]
            except:
                rmg_assets = modelDB.getActiveSubIssues(rmg, date)
            beta_assets = list(set(goodAssets).intersection(rmg_assets))

            # Hack for Domestic China A-Shares
            if rmg.description == 'China':

                # If single-country China model, include B-shares too
                if len(self.rmg)==1:
                    domesticSubIssues = beta_assets
                else:
                    (a_idx, b_idx, h_idx, other) = Utilities.sort_chinese_market(
                            rmg.description, date, modelDB, marketDB, beta_assets)
                    domesticSubIssues = [beta_assets[i] for i in a_idx]

                # Get Chinese domestic market portfolio
                allRMGs = modelDB.getAllRiskModelGroups(inModels=False)
                dcRMG = [r for r in allRMGs if r.description=='Domestic China'][0]
                domesticMarketPortfolio = modelDB.getRMGMarketPortfolio(dcRMG, date, amp='XC-LMS')
                assert(domesticMarketPortfolio is not None)

                if len(domesticMarketPortfolio) > 0:
                    # Compute Chinese A-share betas relative to domestic market
                    predBetas = Utilities.generate_predicted_betas(
                            domesticSubIssues, expMatrix,
                            modelData.factorCovariance, modelData.specificRisk,
                            domesticMarketPortfolio,
                            self.forceRun, nonCurrencyIdx=None,
                            debugging=self.debuggingReporting)
                    predBetas = dict(zip(domesticSubIssues, predBetas))
                    legacyBetas.update(predBetas)

                    if nonCurrencyIdx is not None:
                        # Compute A-share beta in in local currency
                        predBetas = Utilities.generate_predicted_betas(
                                domesticSubIssues, expMatrix,
                                modelData.factorCovariance, modelData.specificRisk,
                                domesticMarketPortfolio,
                                self.forceRun, nonCurrencyIdx=nonCurrencyIdx,
                                debugging=self.debuggingReporting)
                        predBetas = dict(zip(domesticSubIssues, predBetas))
                        localBetas.update(predBetas)
                    allAssets.extend(domesticSubIssues)

                if len(self.rmg) > 1:
                    beta_assets = [beta_assets[i] for i in (b_idx + h_idx + other)]
                else:
                    # If Chinese SCM we are done
                    continue

            # Pick up market portfolio
            marketPortfolio = modelDB.getRMGMarketPortfolio(rmg, date, amp='%s-LMS' % rmg.mnemonic)
            assert(marketPortfolio is not None)

            if len(marketPortfolio) > 0:
                # Compute predicted betas for given RMG
                predBetas = Utilities.generate_predicted_betas(
                        beta_assets, expMatrix,
                        modelData.factorCovariance,
                        modelData.specificRisk, marketPortfolio,
                        self.forceRun, nonCurrencyIdx=None,
                        debugging=self.debuggingReporting)
                predBetas = dict(zip(beta_assets, predBetas))
                legacyBetas.update(predBetas)

                if nonCurrencyIdx is not None:
                    # Compute beta in local currency
                    predBetas = Utilities.generate_predicted_betas(
                            beta_assets, expMatrix,
                            modelData.factorCovariance,
                            modelData.specificRisk, marketPortfolio,
                            self.forceRun, nonCurrencyIdx=nonCurrencyIdx,
                            debugging=self.debuggingReporting)
                    predBetas = dict(zip(beta_assets, predBetas))
                    localBetas.update(predBetas)
                allAssets.extend(beta_assets)
        allAssets = list(set(allAssets))

        predictedBetas = [(a, globalBetas.get(a, None), legacyBetas.get(a, None),
                           localBetas.get(a, None)) for a in allAssets
                           if a in globalBetas or a in legacyBetas or a in localBetas]

        # A bit of reporting
        if len(globalBetas) > 0:
            self.log.info('Predicted global beta bounds: [%.3f, %.3f], Median: %.3f',\
                    ma.min(list(globalBetas.values()), axis=None), ma.max(list(globalBetas.values()), axis=None),
                    ma.median(list(globalBetas.values()), axis=None))

        if len(legacyBetas) > 0:
            self.log.info('Predicted legacy beta bounds: [%.3f, %.3f], Median: %.3f',\
                    ma.min(list(legacyBetas.values()), axis=None), ma.max(list(legacyBetas.values()), axis=None),
                    ma.median(list(legacyBetas.values()), axis=None))
        else:
            self.log.warning('No predicted betas generated')

        if len(localBetas) > 0:
            self.log.info('Predicted local beta bounds: [%.3f, %.3f], Median: %.3f',\
                    ma.min(list(localBetas.values()), axis=None), ma.max(list(localBetas.values()), axis=None),
                    ma.median(list(localBetas.values()), axis=None))

        missingBeta = set(goodAssets).difference(set(allAssets))
        if len(missingBeta) > 0:
            self.log.warning('Missing predicted betas: %d out of %d total assets',\
                    len(missingBeta), len(goodAssets))
        return predictedBetas

    def regressionReporting(self, excessReturns, result, expM, nameSubIDMap, assetIdxMap,
                            modelDate, buildFMPs=True, constrComp=None, specificRets=None):
        """Debugging reporting for regression step
        """

        if self.debuggingReporting and (specificRets is not None):
            outfile = open('tmp/specificReturns-%s-%s.csv' % (self.name, modelDate), 'w')
            outfile.write('subIssue,excessReturn,specificReturn,\n')
            for (sid, ret, sret) in zip(expM.getAssets(), excessReturns, specificRets):
                if ret is ma.masked:
                    if sret is ma.masked:
                        outfile.write('%s,,,\n' % sid.getSubIDString())
                    else:
                        outfile.write('%s,,%.6f,\n' % (sid.getSubIDString(), sret))
                elif sret is ma.masked:
                    outfile.write('%s,%.6f,,\n' % (sid.getSubIDString(), ret))
                else:
                    outfile.write('%s,%.6f,%.6f,\n' % (sid.getSubIDString(), ret, sret))
            outfile.close()

        # Construct FMP-derived factor returns
        tmpFRets = []
        tmpFMPRets = []
        fmpRetDict = dict()
        fmpExpDict = dict()
        allSids = []
        for i in range(len(result.factorReturns)):
            if self.factors[i].name not in nameSubIDMap:
                continue
            sf_id = nameSubIDMap[self.factors[i].name]
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
                        expMCol = expM.getMatrix()[expM.getFactorIndex(self.factors[j].name),:]
                        expMCol = ma.filled(ma.take(expMCol, sidIdxList, axis=0), 0.0)
                        fExp = numpy.inner(fmpWts, expMCol)
                        if constrComp is not None:
                            if self.factors[j].name in constrComp.ccDict:
                                constrCol = constrComp.ccDict[self.factors[i].name]
                                constrX = ma.filled(constrComp.ccXDict[self.factors[j].name], 0.0)
                                constrContr = numpy.inner(constrCol, constrX)
                                fExp += constrContr
                        fmpExp.append(fExp)
                    fmpExpDict[sf_id] = fmpExp

        allSids = list(set(allSids))
        if len(tmpFRets) > 0:
            tmpFRets = numpy.array(tmpFRets, float)
            tmpFMPRets = numpy.array(tmpFMPRets, float)
            correl = numpy.corrcoef(tmpFRets, tmpFMPRets, rowvar=False)[0,1]
            self.log.info('Correlation between factor returns and FMP returns: %.6f', correl)

        if self.debuggingReporting:
            outfile = open('tmp/factorReturns-%s-%s.csv' % (self.name, modelDate), 'w')
            outfile.write('factor,return,stderr,tstat,prob,constr_wt,fmp_ret\n')
            for i in range(len(result.factorReturns)):
                sf_id = nameSubIDMap[self.factors[i].name]
                fmpFR = fmpRetDict.get(sf_id, '')
                outfile.write('%s,%s,%s,%s,%s,%s,%s,\n' % \
                        (self.factors[i].name.replace(',',''),
                            result.factorReturns[i],
                            result.regressionStatistics[i,0],
                            result.regressionStatistics[i,1],
                            result.regressionStatistics[i,2],
                            result.regressionStatistics[i,3],
                            fmpFR))
            outfile.close()

            outfile = open('tmp/fmp-exposures.csv', 'w')
            for i in range(len(result.factorReturns)):
                outfile.write(',%s' % self.factors[i].name.replace(',',''))
            outfile.write(',\n')
            for i in range(len(result.factorReturns)):
                outfile.write('%s,' % self.factors[i].name.replace(',',''))
                sf_id = nameSubIDMap[self.factors[i].name]
                fmpExp = fmpExpDict.get(sf_id, [])
                for j in range(len(result.factorReturns)):
                    if len(fmpExp) == 0:
                        outfile.write(',')
                    else:
                        outfile.write('%s,' % fmpExp[j])
                outfile.write('\n')
            outfile.close()

        if self.debuggingReporting:
            if buildFMPs:
                outfile = open('tmp/fmp-fwd-%s-%s.csv' % (self.mnemonic, modelDate), 'w')
            else:
                outfile = open('tmp/fmp-%s-%s.csv' % (self.mnemonic, modelDate), 'w')
            outfile.write('subissue|')
            for i in range(len(result.factorReturns)):
                sf_id = nameSubIDMap[self.factors[i].name]
                if sf_id in result.fmpMap:
                    outfile.write('%s|' % self.factors[i].name.replace(',',''))
            outfile.write('\n')
            for sid in allSids:
                outfile.write('%s|' % sid.getSubIDString())
                for i in range(len(result.factorReturns)):
                    sf_id = nameSubIDMap[self.factors[i].name]
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
        assets = list(set(modelData.specificRisk.keys()).\
                    intersection(modelData.exposureMatrix.getAssets()))
        totalRisks = Utilities.compute_total_risk_assets(
            assets, modelData.exposureMatrix, modelData.factorCovariance,
            modelData.specificRisk)
        return list(zip(assets, totalRisks))
    
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
        if self.newExposureFormat:
            modelDB.deleteRMIExposureMatrixNew(rmi, subIssues=subIssues)
        else:
            modelDB.deleteRMIExposureMatrix(rmi)
    
    def deleteInstance(self, date, modelDB):
        """Deletes the risk model instance for the given date if it exists.
        """
        rmi_id = modelDB.getRiskModelInstance(self.rms_id, date)
        if rmi_id != None:
            modelDB.deleteRiskModelInstance(rmi_id, self.newExposureFormat)
    
    def deleteRiskModelSerie(cls, modelDB):
        info = modelDB.getRiskModelInfo(cls.rm_id, cls.revision)
        assert(cls.rms_id == info.serial_id)
        modelDB.deleteRiskModelSerie(cls.rms_id, cls.newExposureFormat)
    
    createRiskModelSerie = classmethod(createRiskModelSerie)
    deleteRiskModelSerie = classmethod(deleteRiskModelSerie)
    
    def generate_estimation_universe(self, modelDate, exposureData, 
                                     modelDB, marketDB, excludeIDs=None):
        """Generate estimation universe by piggybacking off the
        ESTU generation code from the associated 'base'
        model, whose FactorRiskModel object should be referenced
        in the self.baseModel attribute.  This should be primarily
        used for statistical and/or hybrid models with a
        corresponding fundamental model counterpart to avoid code
        duplication.
        """
        assert(self.baseModel is not None)
        dateList = modelDB.getDates(self.rmg, modelDate, 60,
                                    excludeWeekend=True)
        dateList = dateList[:-1]
        selfDates = [r.date for r in modelDB.getRiskModelInstances(
                        self.rms_id, dateList)]
        baseDates = [r.date for r in modelDB.getRiskModelInstances(
                        self.baseModel.rms_id, dateList)]
        missing = [d for d in selfDates if d not in set(baseDates)]
        if len(missing) > 0:
            raise LookupError(
                'Missing risk model instances for base model (%s) on %d dates from %s to %s'
                % (self.baseModel.name, len(missing), missing[0], missing[-1]))
        self.baseModel.setFactorsForDate(modelDate, modelDB)
        if len(self.rmg) > 1:
            return self.baseModel.generate_estimation_universe(
                    modelDate, exposureData, modelDB, marketDB,
                    exclude=excludeIDs)
        else:
            return self.baseModel.generate_estimation_universe(
                    modelDate, exposureData, modelDB, marketDB)
    
    def generate_model_specific_exposures(self, modelDate, data, modelDB, marketDB):
        """Any model-specific factor exposure computations should
        be placed in a generate_model_specific_exposures() method
        under the corresponding FactorRiskModel class.
        Data should be a Struct containing all required data items
        for factor computation and the exposureMatrix attribute, which
        this method should ultimately return.
        """
        return data.exposureMatrix

    def generate_industry_exposures(self, modelDate, modelDB, marketDB,
            expMatrix):
        """Create the industry exposures for the assets in the given exposure
        matrix and adds them to the matrix.
        """
        self.log.debug('generate industry exposures: begin')
        factorList = [f.description for f in list(self.\
                      industryClassification.getLeafNodes(modelDB).values())]
        exposures = self.industryClassification.getExposures(
                modelDate, expMatrix.getAssets(), factorList, modelDB)
        expMatrix.addFactors(factorList, exposures,
                             ExposureMatrix.IndustryFactor)
        self.log.debug('generate industry exposures: end')
        return expMatrix

    def generate_model_universe(self, modelDate, modelDB, marketDB):
        """Generate risk model instance universe and estimation
        universe.  Return value is a Struct containing a universe
        attribute (list of SubIssues) containing the model universe
        and an estimationUniverse attribute (list of index positions
        corresponding to ESTU members).
        """
        self.log.debug('generate_model_universe: begin')
        # Get basic risk model instance universe
        (universe, marketCaps, assetDict, foreign) = \
                self.getModelAssetMaster(modelDate, modelDB, marketDB)
        if hasattr(self, 'SCM'):
            data = AssetProcessor.process_asset_information(
                    modelDate, universe, self.rmg, modelDB, marketDB,
                    checkHomeCountry=(self.SCM==0),
                    numeraire_id=self.numeraire.currency_id,
                    legacyDates=self.legacyMCapDates,
                    forceRun=self.forceRun)
        else:
            data = Utilities.Struct()
            data.marketCaps = marketCaps
            data.universe = universe
            data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])

        if not hasattr(data, 'hardCloneMap'):
            data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)
        if not hasattr(data, 'dr2Underlying'):
            data.dr2Underlying = modelDB.getDRToUnderlying(modelDate, data.universe, marketDB)
        
        # Build a temporary exposure matrix to house things
        # like industry and country membership
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        
        # Populate industry membership if required
        if self.industryClassification is not None:
            self.generate_industry_exposures(
                    modelDate, modelDB, marketDB, data.exposureMatrix)
        
        # Populate country membership for regional models
        if len(self.rmg) > 1:
            GlobalExposures.generate_binary_country_exposures(
                    modelDate, self, modelDB, marketDB, data)
        
        # Call model-specific estu construction routine
        dr_indices = [data.assetIdxMap[n] for n in foreign]
        if len(self.rmg) > 1:
            data.estimationUniverseIdx = self.generate_estimation_universe(
                            modelDate, data, modelDB, marketDB, dr_indices)
        else:
            data.estimationUniverseIdx = self.generate_estimation_universe(
                            modelDate, data, modelDB, marketDB)

        if self.debuggingReporting:
            # Output information on assets and their characteristics
            self.estimation_universe_reporting(modelDate, data, modelDB, marketDB)

        # Some reporting of stats
        mcap_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        mcap_ESTU = ma.sum(mcap_ESTU)
        self.log.info('ESTU contains %d assets, %.2f tr %s market cap',
                      len(data.estimationUniverseIdx), mcap_ESTU / 1e12,
                      self.numeraire.currency_code)
        
        self.log.debug('generate_model_universe: end')
        return data
    
    def back_fill_linked_assets(self, returns, scoreDict, modelDB, marketDB):
        """Backfills missing returns for linked assets using their 
        more liquid partners
        Assumes currencies are already matched across linked assets
        dates are assumed to be ordered oldest to newest
        """
        self.log.info('Back-filling asset returns')
        # Loop round sets of linked assets
        data = ma.getdata(returns.data)
        maskArray = ma.getmaskarray(returns.data)
        assetIdxMap = returns.assetIdxMap
        for (groupId, subIssueList) in returns.subIssueGroups.items():
            # Pull out linked assets and rank them
            score = scoreDict[groupId]
            indices  = [assetIdxMap[n] for n in subIssueList]
            sortedIndices = [indices[j] for j in numpy.argsort(-score)]
            reverseSortedIdx = [indices[j] for j in numpy.argsort(score) \
                    if score[j] < max(score)]
            # Loop through each asset in turn, filling any missing
            # values from the best available source
            for idx in reverseSortedIdx:
                # Pick out days when asset returns are missing
                maskedRetIdx = numpy.flatnonzero(maskArray[idx,:])
                if len(maskedRetIdx) > 0:
                    t0 = maskedRetIdx[0]
                    cumret = 1.0
                    for t in maskedRetIdx:
                        # Find best available non-masked return
                        for idxS in sortedIndices:
                            if idxS == idx:
                                break
                            if not maskArray[idxS,t]:
                                data[idx,t] = data[idxS,t]
                                maskArray[idx,t] = False
                                break
                        # If consecutive missing returns, compound replacement returns
                        if t-t0 < 2:
                            if not maskArray[idx,t]:
                                cumret = (1.0 + data[idx,t]) * cumret
                            # Else scale non-missing return by previous compounded currency return
                        else:
                            if not maskArray[idx,t0+1]:
                                data[idx,t0+1] = -1.0 + (1.0 + data[idx,t0+1]) / cumret
                            else:
                                data[idx,t0+1] = -1.0 + (1.0) / cumret
                                maskArray[idx,t0+1] = False
                            if not maskArray[idx,t]:
                                cumret = 1.0 + data[idx,t]
                            else:
                                cumret = 1.0
                        t0 = t
                    # Deal with last in list
                    if t < data.shape[1]-1:
                        if not maskArray[idx,t+1]:
                            data[idx,t+1] = -1.0 + (1.0 + data[idx,t+1]) / cumret
                        else:
                            data[idx,t+1] = -1.0 + (1.0) / cumret
                            maskArray[idx,t+1] = False
        self.log.debug('Done Back-filling asset returns')
        returns.data[:] = ma.array(data, mask=maskArray)
        return returns

    def dump_return_history(self, fileName, returns, modelDB, marketDB):
        # For debugging
        if self.debuggingReporting:
            nameDict = modelDB.getIssueNames(\
                    returns.dates[-1], returns.assets, marketDB)
            dateList = [str(d) for d in returns.dates]
            filepath = 'tmp/%s' % fileName
            outfile = open(filepath, 'w')
            outfile.write(',,,')
            for d in dateList:
                outfile.write('%s,' % d)
            outfile.write('\n')
            runningSidList = []
            cidList1 = sorted([cid for cid in returns.subIssueGroups.keys() if type(cid) is str])
            cidList2 = sorted([cid for cid in returns.subIssueGroups.keys() if type(cid) is not str])
            cidList = cidList1 + cidList2
            for groupId in cidList:
                subIssueList = sorted(returns.subIssueGroups[groupId])
                for sid in subIssueList:
                    runningSidList.append(sid)
                    idx = returns.assetIdxMap[sid]
                    n = None
                    if returns.assets[idx] in nameDict:
                        n = nameDict[returns.assets[idx]].replace(',','')
                    outfile.write('%s,%s,%s,' % (groupId, sid.getSubIDString(), n))
                    for (t,d) in enumerate(returns.dates):
                        outfile.write('%12.8f,' % ma.filled(returns.data[idx,t], 0.0))
                    outfile.write('\n')
            otherSids = [sid for sid in returns.assets if sid not in runningSidList]
            for sid in otherSids:
                idx = returns.assetIdxMap[sid]
                n = None
                if returns.assets[idx] in nameDict:
                    n = nameDict[returns.assets[idx]].replace(',','')
                outfile.write('%s,%s,%s,' % (sid.getSubIDString(), sid.getSubIDString(), n))
                for (t,d) in enumerate(returns.dates):
                    fld = Utilities.p2_round(ma.filled(returns.data[idx,t], 0.0), 16)
                    outfile.write('%12.8f,' % fld)
                outfile.write('\n')
            outfile.close()
         
            # Output stats on agreement between returns
            corrList = []
            for (groupId, subIssueList) in returns.subIssueGroups.items():
                idxList = [returns.assetIdxMap[sid] for sid in subIssueList]
                retSubSet = ma.filled(ma.take(returns.data, idxList, axis=0), 0.0)
                corr = Utilities.compute_covariance(retSubSet, axis=1, corrOnly=True)
                for i in range(corr.shape[0]):
                    for j in range(i+1,corr.shape[0]):
                        corrList.append(corr[i,j])
            medianCorr = ma.median(corrList, axis=None)
            self.log.info('Run: %s, Date: %s, MED: %.12f', fileName, returns.dates[-1], medianCorr)
        return

    def compound_currency_returns(self,  returns, date, zeroMask, modelDB, marketDB):
        """ Compounds currency returns for missing asset returns into non-missing
        returns locations
        """
        self.log.info('Compounding currency and returns-timing adjustments')
        cumRets = numpy.cumproduct(ma.filled(returns.data, 0.0) + 1.0, axis=1)
        data = ma.getdata(returns.data)
        mask = ma.getmaskarray(returns.data)
        # Loop round each asset in turn
        for (i, sid) in enumerate(returns.assets):
            # Pick out days when asset returns are missing
            maskedRetIdx = numpy.flatnonzero(zeroMask[i,:])
            if len(maskedRetIdx) > 0:
                t0 = maskedRetIdx[0]
                t1 = maskedRetIdx[0]
                for t in maskedRetIdx:
                    if t-t1 > 1:
                        if not mask[i,t1+1]:
                            data[i,t1+1] = -1.0 + ((1.0 + data[i,t1+1]) * cumRets[i,t1] / cumRets[i,t0])
                        t0 = t-1
                    data[i,t] = 0.0
                    mask[i,t] = False
                    t1 = t
                # Deal with last in list
                if t < returns.data.shape[1]-1:
                    if not mask[i,t1+1]:
                        data[i,t+1] = -1.0 + ((1.0 + data[i,t1+1]) * cumRets[i,t1] / cumRets[i,t0])
                        mask[i,t+1] = False
        self.log.debug('Done Compounding currency and returns-timing adjustments')
        returns.data[:] = ma.array(data, mask=mask)
        return returns

    def process_returns_history(self, date, universe, needDays, modelDB, marketDB,
            data=None, drCurrData=None, subIssueGroups=None):
        """Does extensive processing of asset returns to alleviate the problem
        of illiquid data. Does currency conversion and returns-timing
        adjustment to cross-listings to make them "local"
        Then, compounds these returns when there is no asset return on
        a particular day, and scales the non-missing returns accordingly
        Finally, fills in missing returns from non-missing linked partners
        and, again, scales non-missing illiquid returns
        """
        daysBack = 21
        self.log.info('Back-filling %d days of asset returns', needDays+daysBack)
        # Get daily returns for the past calendar days(s)
        # Here do no currency conversion or other fancy stuff
        # We aim first to identify live stocks that haven't traded, i.e.
        # which have zero returns
        returns = modelDB.loadTotalReturnsHistory(
                self.rmg, date, universe, int(needDays+daysBack))
        # Uncomment for Python 3 testing
        #returns.data = Utilities.p2_round(returns.data, 16)
        if subIssueGroups == None:
            returns.subIssueGroups = modelDB.getIssueCompanyGroups(date, universe, marketDB)
        else:
            returns.subIssueGroups = subIssueGroups
        linkedIdSet = set() 
        for (groupId, subIssueList) in returns.subIssueGroups.items():
            linkedIdSet.update(subIssueList)
        self.dump_return_history('rets_s0_raw.csv', returns, modelDB, marketDB)
         
        # Get masked returns, corresponding to IPOs and NTDs
        ipoMask = ma.getmaskarray(returns.data)
        nMasked = len(numpy.flatnonzero(ipoMask))
        propMasked = 100.0 * float(nMasked) / numpy.size(returns.data)
        self.log.info('%d out of %d returns missing due to IPO or NTD (%4.1f%%)',
                nMasked, numpy.size(returns.data), propMasked)

        # Get zero returns - corresponding to live assets not trading
        tmpArray = ma.filled(returns.data, -999.0)
        tmpArray = ma.masked_where(tmpArray==0.0, tmpArray)
        # Quick fix to ensure only linked assets in SCMs have their
        # missing(zero) returns treated correctly for now
        if len(self.rmg) == 1:
            nonLinkedIdx = [returns.assetIdxMap[sid] for sid in \
                        universe if sid not in linkedIdSet]
            for idx in nonLinkedIdx:
                tmpArray[idx,:] = -999.0
        zeroMask = ma.getmaskarray(tmpArray)

        # Unmask records for non-trading days - some cross-listings
        # may have traded, so we treat them as normal days
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(returns.dates)])
        for rmg in self.rmg:
            asset_indices = [returns.assetIdxMap[sid] for sid in \
                    self.rmgAssetMap[rmg.rmg_id] if sid in linkedIdSet]
            rmgCalendarSet = set(modelDB.getDateRange(rmg, returns.dates[0], returns.dates[-1]))
            if len(asset_indices) > 0:
                ntd_date_indices = [dateIdxMap[d] for d in returns.dates
                        if d not in rmgCalendarSet
                        and d in dateIdxMap]
                positions = [len(returns.dates) * sIdx + dIdx for \
                        sIdx in asset_indices for dIdx in ntd_date_indices]
                ma.put(ipoMask, positions, False)
                ma.put(zeroMask, positions, True)
        nMasked2 = len(numpy.flatnonzero(ipoMask))
        self.log.info('%d returns unmasked on NTDs', nMasked-nMasked2)

        # Load asset returns again, and now do currency conversion to local market
        # currency and weekend compounding, with only pre-IPO returns masked
        subIssueGroups = returns.subIssueGroups
        returns = modelDB.loadTotalReturnsHistory(
                self.rmg, date, universe, int(needDays+daysBack), drCurrData,
                self.numeraire.currency_id, maskArray=ipoMask, compoundWeekend=True)
        # Uncomment for Python 3 testing
        #returns.data = Utilities.p2_round(returns.data, 16)
        returns.subIssueGroups = subIssueGroups
        self.dump_return_history('rets_s1_cur.csv', returns, modelDB, marketDB)

        if isinstance(drCurrData, dict):
            ids_DR = list(drCurrData.keys())
            if len(ids_DR) > 0:
                # Load full set of adjustment factors for every relevant market
                rtId = self.returnsTimingId
                if rtId == None and hasattr(self, 'specReturnTimingId'):
                    rtId = self.specReturnTimingId
                # Convert cross-listing returns to local temporal
                if rtId is not None:
                    rmgList = modelDB.getAllRiskModelGroups()
                    adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                            rtId, rmgList, returns.dates)
                    adjustments.data = ma.filled(adjustments.data, 0.0)
                    returns = self.adjustDRSpecificReturnsForTiming(
                            date, returns, modelDB, marketDB,
                            adjustments, ids_DR, rmgList)
                    self.dump_return_history(
                            'rets_s2_rtim.csv', returns, modelDB, marketDB)

        # Ditch the weekend dates and returns - not needed anymore
        returns.dates = [d for d in returns.dates if d.weekday() < 5]
        nonWkndIdx = [dateIdxMap[d] for d in returns.dates]
        returns.data = ma.take(returns.data, nonWkndIdx, axis=1)
        zeroMask = ma.take(zeroMask, nonWkndIdx, axis=1)

        # Compound currency and returns timing entries for illiquid assets
        returns = self.compound_currency_returns(
                returns, date, zeroMask, modelDB, marketDB)
        self.dump_return_history('rets_s3_curc.csv', returns, modelDB, marketDB)

        # Redo masks and mask zero entries for next stages
        # We differentiate between missing returns due to IPOs and 
        # missing due to illiquidity in case there is any consequent
        # proxy-returns fill-in
        # Uncomment for Python 3 testing
        #returns.data = Utilities.p2_round(returns.data, 16)
        ipoMask = ma.getmaskarray(returns.data)
        tmpArray = ma.filled(returns.data, -999.0)
        tmpArray = ma.masked_where(tmpArray==0.0, tmpArray)
        # SCM fix once more
        if len(self.rmg) == 1:
            nonLinkedIdx = [returns.assetIdxMap[sid] for sid in \
                    universe if sid not in linkedIdSet]
            for idx in nonLinkedIdx:
                tmpArray[idx,:] = -999.0
        zeroMask = ma.getmaskarray(tmpArray)
        returns.data = ma.masked_where(zeroMask, returns.data)
        nMasked = len(numpy.flatnonzero(ipoMask))
        propMasked = 100.0 * float(nMasked) / numpy.size(returns.data)
        self.log.info('%d out of %d returns missing due to IPOs (%4.1f%%)',
                nMasked, numpy.size(returns.data), propMasked)
        nMasked = len(numpy.flatnonzero(zeroMask))
        propMasked = 100.0 * float(nMasked) / numpy.size(returns.data)
        self.log.info('%d out of %d returns missing due to illiquidity (%4.1f%%)',
                nMasked, numpy.size(returns.data), propMasked)

        # Back fill missing returns between linked assets
        scores = self.score_linked_assets(date, returns.assets, modelDB, marketDB,
                                            subIssueGroups=returns.subIssueGroups)
        returns = self.back_fill_linked_assets(returns, scores, modelDB, marketDB)
        self.dump_return_history('rets_s4_bfil.csv', returns, modelDB, marketDB)
        allMasked = ma.getmaskarray(returns.data)
        zeroMask = zeroMask * allMasked
        ipoMask = ipoMask * allMasked
        nMasked = len(numpy.flatnonzero(ipoMask))
        propMasked = 100.0 * float(nMasked) / numpy.size(returns.data)
        self.log.info('%d out of %d returns missing due to IPOs (%4.1f%%)',
                nMasked, numpy.size(returns.data), propMasked)
        nMasked = len(numpy.flatnonzero(zeroMask))
        propMasked = 100.0 * float(nMasked) / numpy.size(returns.data)
        self.log.info('%d out of %d returns missing due to illiquidity (%4.1f%%)',
                nMasked, numpy.size(returns.data), propMasked)

        # Take only history we need
        returns.data = returns.data[:,-(needDays+1):]
        returns.dates = returns.dates[-(needDays+1):] 
        zeroMask = zeroMask[:,-(needDays+1):]
        ipoMask = ipoMask[:,-(needDays+1):]
        if len(returns.data.shape) < 2:
            returns.data = returns.data[:,numpy.newaxis]
            zeroMask = zeroMask[:,numpy.newaxis]
            ipoMask = ipoMask[:,numpy.newaxis]

        return (returns, zeroMask, ipoMask)

    def score_linked_assets(self, date, universe, modelDB, marketDB,
            daysBack=125, scale=15.0, subIssueGroups=None):
        """Assigns scores to linked assets based on their cap and 
        liquidity, in order to assist cloning or other adjustment
        of exposures, specific risks, correlations etc.
        """
        self.log.debug('Start score_linked_assets')
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(universe)])
        if subIssueGroups is None:
            subIssueGroups = modelDB.getIssueCompanyGroups(date, universe, marketDB)
        scoreDict = dict()
        # Determine how many calendar days of data we need
        dateList = self.getRMDates(date, modelDB, daysBack, 365)[0]
        # Compute average trading volume of each asset
        vol = modelDB.loadVolumeHistory(dateList, universe, self.numeraire.currency_id)
        volume = numpy.average(ma.filled(vol.data, 0.0), axis=1)
        # Load in market cpas
        (mcapDates, goodRatio) = self.getRMDates(
                date, modelDB, 20, ceiling=False)
        avgMarketCap = modelDB.getAverageMarketCaps(
                mcapDates, universe, self.numeraire.currency_id, marketDB)        

        # Load in returns and get proportion of days traded
        dateList = self.getRMDates(date, modelDB, 250, 365)[0]
        returns = modelDB.loadTotalReturnsHistory(
                self.rmg, dateList[-1], universe, 250, None)
        propNonMissing = ma.filled(ma.average(~ma.getmaskarray(returns.data), axis=1), 0.0)

        # And do something similar with from dates
        issueFromDates = modelDB.loadIssueFromDates([date], universe)
        age = [min(int((date-dt).days), 1250) for dt in issueFromDates]
        age = numpy.array(age, float)

        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in subIssueGroups.items():
            indices  = [assetIdxMap[n] for n in subIssueList]

            # Score each asset by its trading volume
            volumeSubSet = numpy.take(volume, indices, axis=0)
            if max(volumeSubSet) > 0.0:
                volumeSubSet /= max(volumeSubSet)
            else:
                volumeSubSet = numpy.ones((len(indices)), float)

            # Now score each asset by its market cap
            mcapSubSet = ma.filled(ma.take(avgMarketCap, indices, axis=0), 0.0)
            if max(mcapSubSet) > 0.0:
                mcapSubSet /= numpy.max(mcapSubSet, axis=None)
            else:
                mcapSubSet = numpy.ones((len(indices)), float)

            # Now score by proportion of non-missing returns
            nMissSubSet = numpy.take(propNonMissing, indices, axis=0)
            maxNMiss = max(nMissSubSet)
            if maxNMiss > 0.0:
                for (ii, idx) in enumerate(indices):
                    if issueFromDates[idx] < self.legacyISCSwitchDate:
                        nMissSubSet[ii] = 1.0
                    else:
                        nMissSubSet[ii] = nMissSubSet[ii] / maxNMiss
            else:
                nMissSubSet = numpy.ones((len(indices)), float)

            # Score each asset by its age
            ageSubSet = numpy.take(age, indices, axis=0)
            maxAge = max(ageSubSet)
            if maxAge > 0.0:
                for (ii, idx) in enumerate(indices):
                    if issueFromDates[idx] < self.legacyISCSwitchDate:
                        ageSubSet[ii] = 1.0
                    else:
                        ageSubSet[ii] = ageSubSet[ii] / maxAge
            else:
                ageSubSet = numpy.ones((len(indices)), float)

            # Now combine the scores and exponentially scale them
            score = volumeSubSet * mcapSubSet * nMissSubSet * ageSubSet
            score = numpy.exp(scale * (score - 1.0))
            scoreDict[groupId] = score

        if self.debuggingReporting:
            idList = []
            scoreList = []
            for (groupId, subIssueList) in subIssueGroups.items():
                sidList = [groupId + ':' + sid.getSubIDString() for sid in subIssueList]
                idList.extend(sidList)
                scoreList.extend(scoreDict[groupId])
            Utilities.writeToCSV(numpy.array(scoreList), 'tmp/scores-%s.csv' % date, rowNames=idList)

        self.log.debug('End score_linked_assets')
        return scoreDict

    def clone_linked_asset_exposures(self, date, data, modelDB, marketDB,
                    scoreDict, subIssueGroups=None):
        """Clones exposures for cross-listings/DRs etc.
        based on those of the most liquid/largest of each set
        Assets in hardCloneMap are directly cloned from their master asset
        Others are computed as a weighted average of all exposures
        within their group
        """
        self.log.debug('clone_linked_asset_exposures: begin')
        self.log.info('Using NEW ISC Treatment for exposure cloning')
        expM = data.exposureMatrix
        if subIssueGroups is None:
            subIssueGroups = modelDB.getIssueCompanyGroups(date, data.universe, marketDB)

        if not hasattr(data, 'hardCloneMap'):
            # Pick up dict of assets to be cloned from others
            data.hardCloneMap = modelDB.getClonedMap(date, data.universe, cloningOn=self.hardCloning)

        # Pick out exposures to be cloned
        exposureNames = []
        exposureIdx = []
        cloneTypes = [ExposureMatrix.StyleFactor, 
                      ExposureMatrix.StatisticalFactor,
                      ExposureMatrix.MacroCoreFactor, 
                      ExposureMatrix.MacroMarketTradedFactor,
                      ExposureMatrix.MacroEquityFactor,
                      ExposureMatrix.MacroSectorFactor]            
        for ftype in cloneTypes:
            exposureIdx += expM.getFactorIndices(ftype)
            exposureNames += expM.getFactorNames(ftype)

        # Exclude any binary exposures for now
        binaryExposureIdx = []
        binaryExposureNames = []
        for (n,fIdx) in zip(exposureNames, exposureIdx):
            if Utilities.is_binary_data(expM.getMatrix()[fIdx]):
                binaryExposureIdx.append(fIdx)
                binaryExposureNames.append(n)
                self.log.info('%s is binary: excluding from cloning', n)
        exposureIdx = [idx for idx in exposureIdx if idx not in binaryExposureIdx]
        exposureNames = [n for n in exposureNames if n not in binaryExposureNames]

        # First deal with any assets to be cloned exactly - note that
        # we clone every single factor exposure
        for sid in data.hardCloneMap.keys():
            expM.getMatrix()[:, data.assetIdxMap[sid]] = \
                    expM.getMatrix()[:, data.assetIdxMap[data.hardCloneMap[sid]]]

        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in subIssueGroups.items():
            score = scoreDict[groupId]
            indices  = [data.assetIdxMap[n] for n in subIssueList]
            cloneList = [n for n in subIssueList if n in data.hardCloneMap]

            for fIdx in exposureIdx:
                # Weighted average of all exposures in the set
                expos = ma.take(expM.getMatrix()[fIdx], indices, axis=0)
                expos = ma.average(expos, weights=score)
                for idx in indices:
                    expM.getMatrix()[fIdx, idx] = expos

        self.log.debug('clone_linked_asset_exposures: end')
        return data.exposureMatrix
    
    def clone_linked_asset_descriptors(self, date, data, expM,
                modelDB, marketDB, scoreDict, subIssueGroups=None,
                excludeList=[]):
        """Clones descriptor exposures for cross-listings/DRs etc.
        based on those of the most liquid/largest of each set
        Replaces all missing values with that from the "master"
        """
        self.log.debug('clone_linked_asset_descriptors: begin')
        if subIssueGroups is None:
            subIssueGroups = modelDB.getIssueCompanyGroups(date, data.universe, marketDB)

        # Pick out exposures to be cloned
        exposureNames = []
        exposureIdx = []
        cloneTypes = [ExposureMatrix.StyleFactor,
                      ExposureMatrix.StatisticalFactor,
                      ExposureMatrix.MacroCoreFactor,
                      ExposureMatrix.MacroMarketTradedFactor,
                      ExposureMatrix.MacroEquityFactor,
                      ExposureMatrix.MacroSectorFactor]
        for ftype in cloneTypes:
            exposureIdx += expM.getFactorIndices(ftype)
            exposureNames += expM.getFactorNames(ftype)

        # Exclude any binary exposures for now
        binaryExposureIdx = []
        binaryExposureNames = []
        for (n,fIdx) in zip(exposureNames, exposureIdx):
            if Utilities.is_binary_data(expM.getMatrix()[fIdx]):
                binaryExposureIdx.append(fIdx)
                binaryExposureNames.append(n)
                self.log.info('%s is binary: excluding from cloning', n)
        exposureIdx = [idx for idx in exposureIdx if idx not in binaryExposureIdx]
        exposureNames = [n for n in exposureNames if n not in binaryExposureNames]

        # Exclude any specified descriptors
        excludeNames = [n for n in excludeList if n in exposureNames]
        if len(excludeNames) > 0:
            exposureNames = [n for n in exposureNames if n not in excludeNames]
            exposureIdx = [expM.getFactorIndex(n) for n in exposureNames]
            self.log.info('Excluding %d descriptors: %s from cloning',
                    len(excludeNames), excludeNames)
            
        # Loop round sets of linked assets and pull out exposures
        nChanges = 0
        for (groupId, subIssueList) in subIssueGroups.items():
            indices  = [data.assetIdxMap[n] for n in subIssueList]
            score = scoreDict[groupId]

            for fIdx in exposureIdx:
                # Nuke scores for assets with missing descriptor value
                expos = ma.take(expM.getMatrix()[fIdx], indices, axis=0)
                nonMissingValues = ma.getmaskarray(expos)==0
                f_score = nonMissingValues * score
                # Get asset with largest score
                maxIndx = indices[numpy.argsort(f_score)[-1]]
                # Fill in missing values
                missingIdx = numpy.flatnonzero(nonMissingValues==0)
                exp_idx = [indices[idx] for idx in missingIdx]
                if maxIndx in exp_idx:
                    continue
                for idx in exp_idx:
                    nChanges += 1
                    expM.getMatrix()[fIdx, idx] = expM.getMatrix()[fIdx, maxIndx]

        self.log.info('Made %d descriptor changes over %d groups of linked assets',
                nChanges, len(subIssueGroups))
        self.log.debug('clone_linked_asset_descriptors: end')
        return

    def load_ISC_Scores(self, date, data, modelDB, marketDB):
        """Either loads the linked asset scores from the descriptor table
        or computes them on the fly if they don't exist
        """

        # Load dict of all descriptors
        descDict = dict(modelDB.getAllDescriptors())

        # If previous steps have not returned scores, we compute them on the fly
        logging.warning('No ISC scores saved in descriptor table')
        scoreDict = self.score_linked_assets(
                date, data.universe, modelDB, marketDB, subIssueGroups=data.subIssueGroups)
        return scoreDict

    def adjust_asset_returns_for_market_async(self, returns, date, modelDB, marketDB):
        """Adjusts an array of asset returns for market asynchronicity
        """
        self.log.info('Adjusting factor returns for market synchronicity')
        tcAssetMap = self.loadRiskModelGroupAssetMap(date, returns.assets, modelDB, marketDB, True)
        # And populate it
        adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                        self.returnsTimingId, self.rmg, returns.dates)
        adjustments = adjustments.data
        rmgIdxMap = dict(zip([r.rmg_id for r in self.rmg], range(len(self.rmg))))

        # Look for missing days
        missingDaysIdx = numpy.flatnonzero(numpy.sum(ma.getmaskarray(
                            adjustments), axis=0)==adjustments.shape[0])
        if len(missingDaysIdx) > 0:
            self.log.info('Missing returns timing adjustment for all RiskModelGroups on %d days, setting to zero',
                    len(missingDaysIdx))
            self.log.debug('Missing dates: %s', ','.join(
                    [str(returns.dates[i]) for i in missingDaysIdx]))
            adjustments[:, missingDaysIdx] = 0.0

        # Fill in missing adjustments with the median value for the trading
        # country's region
        rmgIdRegionMap = dict([(r.rmg_id, r.region_id) for r in self.rmg])
        regionList = set(rmgIdRegionMap.values())
        tcIDList = [tcID for tcID in tcAssetMap.keys() if tcID in rmgIdRegionMap]
        for reg in regionList:
            tcIDSubList = [tcID for tcID in tcIDList if rmgIdRegionMap[tcID]==reg]
            rmgIdx = [rmgIdxMap[tc] for tc in tcIDSubList]
            if len(rmgIdx) > 1:
                adjustSubSet = adjustments[rmgIdx]
                adjustMedian = ma.median(adjustSubSet, axis=0).filled(0.0)
                fadjustSubSet = adjustSubSet.filled()
                numpy.putmask(fadjustSubSet, adjustSubSet.mask, adjustMedian)
                adjustments[rmgIdx] = fadjustSubSet
        adjustments = adjustments.filled(0.0)
        for tc in tcAssetMap.keys():
            # Pick out all non-missing asset returns in relevant market
            tcIdxList = [returns.assetIdxMap[sid] for sid in tcAssetMap[tc]]
            if len(tcIdxList) > 0 and tc in rmgIdxMap:
                # Compute adjusted asset returns
                cty_returns = ma.take(returns.data, tcIdxList, axis=0)
                if len(cty_returns.shape) < 2:
                    cty_returns = cty_returns[numpy.newaxis]
                cty_returns = cty_returns + adjustments[rmgIdxMap[tc],:]
                for (m,idx) in enumerate(tcIdxList):
                    returns.data[idx,:] = cty_returns[m,:]
        return (returns, adjustments)

    def adjustDRSpecificReturnsForTiming(self, modelDate, specificReturns,
                                         modelDB, marketDB, adjustments,
                                         ids_DR, rmgList):
        """If DR-like instruments are present in the universe, adjust their
        specific returns for returns-timing.  Both specificReturns and
        adjustments should be TimeSeriesMatrix objects containing the
        specific returns and market adjustment factors time-series,
        respectively.  The adjustments array cannot contain masked values.
        Returns the adjusted specific returns.
        """
        if not self.modelHack.DRHandling:
            return specificReturns
        self.log.debug('adjustDRSpecificReturnsForTiming: begin')
        drSet = set(ids_DR)
        subIssues = specificReturns.assets

        # Determine DRs' home country versus country of quotation
        homeCountryMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                        self.rmgAssetMap.items() for sid in ids \
                        if sid in drSet])
        tradingCountryMap = dict([(sid, rmg.rmg_id) for (sid, rmg) in \
                        modelDB.getSubIssueRiskModelGroupPairs(
                        modelDate, restrict=list(ids_DR))])
        rmgIdxMap = dict([(j.rmg_id,i) for (i,j) in \
                            enumerate(adjustments.assets)])
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(subIssues)])

        # Set up market time zones
        rmgZoneMap = dict((rmg.rmg_id, rmg.gmt_offset) for rmg in rmgList)

        # Remove adjustment from home country, add from trading country
        dateLen = len(specificReturns.dates)
        self.log.info('Adjusting specific returns for %s assets' % len(ids_DR))
        sameTZ = []
        for i in range(len(ids_DR)):
            sid = ids_DR[i]
            homeIdx = homeCountryMap.get(sid, None)
            tradingIdx = tradingCountryMap.get(sid, None)
            if homeIdx is None or tradingIdx is None:
                continue
            if abs(rmgZoneMap[homeIdx] - rmgZoneMap[tradingIdx]) <= 3:
                sameTZ.append(sid)
            else:
                idx = assetIdxMap[sid]
                srAdjust = adjustments.data[rmgIdxMap[homeIdx],:dateLen] \
                        - adjustments.data[rmgIdxMap[tradingIdx],:dateLen]
                specificReturns.data[idx,:] -= srAdjust
        self.log.info('%s assets in same time-zone, not adjusted' % len(sameTZ))

        self.log.debug('adjustDRSpecificReturnsForTiming: end')
        return specificReturns

    def getAllClassifications(self, modelDB):
        """Returns all non-root classification objects for this risk model.
        Delegates to the industryClassification object.
        """
        if not isinstance(self, StatisticalFactorModel) and \
                not isinstance(self, RegionalStatisticalFactorModel) and \
                not isinstance(self, StatisticalModel) and \
                not isinstance(self, MacroeconomicModel):
            return self.industryClassification.getAllClassifications(
                modelDB)
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
        if not isinstance(self, StatisticalFactorModel) and \
                not isinstance(self, RegionalStatisticalFactorModel) and \
                not isinstance(self, StatisticalModel):
            return self.industryClassification.getClassificationChildren(
                parentClass, modelDB)
        else:
            return list()
    
    def getClassificationParents(self, childClass, modelDB):
        """Returns a list of the parents of the given node in the
        classification.
        Delegates to the industryClassification object.
        """
        if not isinstance(self, StatisticalFactorModel) and \
                not isinstance(self, RegionalStatisticalFactorModel) and \
                not isinstance(self, StatisticalModel):
            return self.industryClassification.getAllParents(
                childClass, modelDB)
        else:
            return list()
    
    def getClassificationMember(self, modelDB):
        """Returns the classification member object for this risk model.
        Delegates to the industryClassification object.
        """
        if not isinstance(self, StatisticalFactorModel) and \
                not isinstance(self, RegionalStatisticalFactorModel) and \
                not isinstance(self, StatisticalModel):
            return self.industryClassification.getClassificationMember(modelDB)
        else:
            return list()
    
    def getClassificationRoots(self, modelDB):
        """Returns the root classification objects for this risk model.
        Delegates to the industryClassification object.
        """
        if not isinstance(self, StatisticalFactorModel) and \
                not isinstance(self, RegionalStatisticalFactorModel) and \
                not isinstance(self, StatisticalModel) and \
                not isinstance(self, MacroeconomicModel):
            return self.industryClassification.getClassificationRoots(
                modelDB)
        else:
            return list()
    
    def getRiskModelInstance(self, date, modelDB):
        """Returns the risk model instance corresponding to the given date.
        The return value is None if no such instance exists.
        """
        return modelDB.getRiskModelInstance(self.rms_id, date)
    
    def getModelAssetMaster(self, date, modelDB, marketDB):
        """Determine the universe for the current model instance.
        Returns a tuple with the first entry a list of asset IDs,
        the second an array of their average market caps over the
        last twenty trading days, and the last a dictionary of
        asset ID to (identifier, issuer) pairs.
        The list of asset IDs is sorted.
        """
        self.log.info('Risk model numeraire is %s (ID %d)',
                      self.numeraire.currency_code, self.numeraire.currency_id)
        # Get all securities marked for this model at the current date
        assets = modelDB.getMetaEntity(marketDB, date, self.rms_id, self.primaryIDList)
        universe_list = sorted(assets)
        
        # Load market caps, convert to model numeraire
        (mcapDates, goodRatio) = self.getRMDates(
                date, modelDB, 20, ceiling=False)
        avgMarketCap = modelDB.getAverageMarketCaps(
                mcapDates, universe_list, self.numeraire.currency_id, marketDB)

        # Remove assets with missing market cap
        mask = ma.getmaskarray(avgMarketCap)
        new_universe = [universe_list[i] for i in range(len(universe_list))
                        if mask[i] == 0]
        missingCap = numpy.flatnonzero(mask)
        if len(missingCap) > 0:
            self.log.warning('%d assets dropped due to missing avg %d-day cap',
                          len(missingCap), len(mcapDates))
            # self.log.debug('dropped assets: %s' % (','.join([
            #                universe_list[i].getSubIDString() for i in missingCap])))
        avgMarketCap = ma.filled(avgMarketCap.compressed(), 0.0)
        
        # Get asset types
        self.assetTypeDict = AssetProcessor.get_asset_info(date, new_universe, modelDB, marketDB,
                'ASSET TYPES', 'Axioma Asset Type')
        self.marketTypeDict = AssetProcessor.get_asset_info(date, new_universe, modelDB, marketDB,
                'REGIONS', 'Market')
        if self.debuggingReporting:
            typeList = [self.assetTypeDict.get(sid,None) for sid in new_universe]
            AssetProcessor.dumpAssetListForDebugging('tmp/AssetTypeMap-%s.csv' % self.mnemonic,
                    new_universe, dataColumn=typeList)
            typeList = list(set(typeList))
            AssetProcessor.dumpAssetListForDebugging('tmp/AssetTypes-%s.csv' % self.mnemonic, typeList)
            typeList = [self.marketTypeDict.get(sid,None) for sid in new_universe]
            AssetProcessor.dumpAssetListForDebugging('tmp/MarketTypeMap-%s.csv' % self.mnemonic,
                    new_universe, dataColumn=typeList)

        # Identify possible ETFs
        etfList = [sid for sid in new_universe if self.assetTypeDict.get(sid,None) in AssetProcessor.etfAssetTypes]
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(new_universe)])
        etfIdx = [assetIdxMap[sid] for sid in etfList]
        self.log.info('Allowing %d non-composite asset ETFs into universe', len(etfIdx))

        # Remove assets w/o industry membership (excluding ETFs)
        if self.industryClassification is not None \
                and self.modelHack.checkForMissingIndustry:
            assert(len(list(self.industryClassification.getLeafNodes(modelDB).values())) > 0)
            leaves = self.industryClassification.getLeafNodes(modelDB)
            factorList = [i.description for i in leaves.values()]
            exposures = self.industryClassification.getExposures(
                date, new_universe, factorList, modelDB)
            indSum = ma.sum(exposures, axis=0).filled(0.0)
            if isinstance(self, StatisticalFactorModel) or \
                    isinstance(self, RegionalStatisticalFactorModel) or \
                    isinstance(self, StatisticalModel):
                numpy.put(indSum, etfIdx, 1.0)
#            missingInd = numpy.flatnonzero(indSum==0.0)
#            if len(missingInd) > 0:
#                self.log.warning('%d assets dropped due to missing %s classification',
#                              len(missingInd), self.industryClassification.name)
#                self.log.debug('dropped assets: %s',
#                               ','.join([new_universe[i].getSubIDString()
#                                         for i in missingInd]))
            new_universe = [new_universe[i] for (i,val)
                    in enumerate(indSum) if val != 0.0]
            avgMarketCap = numpy.take(avgMarketCap, 
                            numpy.flatnonzero(indSum), axis=0)
        
        # If regional model, deal with country assignments
        if len(self.rmg) > 1:
            (new_universe, avgMarketCap, foreign) = \
                    self.process_asset_country_assignments(
                    date, new_universe, avgMarketCap, modelDB, marketDB, True)
        else:
            foreign = list()
        retVal = (new_universe, avgMarketCap, assets, foreign)
        
        if len(new_universe) > 0:
            self.log.info('%d assets in the universe', len(new_universe))
            return retVal
        else:
            raise Exception('No assets in the universe!')
    
    def insertEstimationUniverse(self, rmi, universe, estU, modelDB, qualified=None):
        """Inserts the estimation universe into the database for the given
        risk model instance.
        universe is a list of sub-issues, estU contains the indices into
        universe of sub-issues in the estimation universe.  qualified 
        is an optional list of indices corresponding to the subset of 
        estU which genuinely qualified for membership, presumably from 
        some grandfathering criteria.
        """
        estUniv = [universe[i] for i in estU]
        if qualified is not None:
            qualified = [universe[i] for i in qualified]
        modelDB.insertEstimationUniverse(rmi, estUniv, qualified)

    def insertEstimationUniverseWeights(self, rmi, subidWeightPairs, modelDB):
        """Inserts model estimation (regression) weights for estimation
        universe assets in the given risk model instance.
        Note that the list of SubIssues may be smaller than the full set
        of estimation universe assets due to treatment of non-trading
        markets, removal of missing returns assets, and additional 
        'pre-processing' logic employed prior to the regression.
        """
        modelDB.insertEstimationUniverseWeights(rmi, subidWeightPairs)
    
    def insertExposures(self, rmi_id, data, modelDB, marketDB, update=False, descriptorData=None):
        """Insert the exposure matrix into the database for the given
        risk model instance.
        The exposure matrix is stored in data as returned by
        generateExposureMatrix().
        """
        if hasattr(self, 'nurseryCountries') and (len(self.nurseryCountries) > 0):
            factors = self.factors + self.nurseryCountries + self.hiddenCurrencies
        else:
            factors = self.factors

        subFactors = modelDB.getRiskModelInstanceSubFactors(rmi_id, factors)
        expMat = data.exposureMatrix
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
        subFactorMap = dict(zip([f.name for f in factors], subFactors))
        if self.newExposureFormat:
            modelDB.insertFactorExposureMatrixNew(rmi_id, expMat,
                                                  subFactorMap, update)
        else:
            modelDB.insertFactorExposureMatrix(rmi_id, expMat, subFactorMap)
        
        # Save standardization stats if any
        if self.standardizationStats and hasattr(expMat, 'meanDict'):
            nameList = [nm for nm in expMat.meanDict.keys() if nm in subFactorMap]
            sfList = [subFactorMap[nm] for nm in nameList]
            values = [expMat.meanDict[nm] for nm in nameList]
            modelDB.insertStandardizationMean(
                    rmi_id.rms_id, rmi_id.date, sfList, values, update)
            values = [expMat.stdDict[nm] for nm in nameList]
            modelDB.insertStandardizationStDev(
                    rmi_id.rms_id, rmi_id.date, sfList, values, update)

        if descriptorData is not None:
            modelDB.insertStandardisationStats(self.rms_id, rmi_id.date,
                    descriptorData.descriptors, descriptorData.descDict,
                    descriptorData.meanDict, descriptorData.stdDict)

        # Report any changes in factor structure
        currFactors = set([f.name for f in factors])
        dateList = modelDB.getDates(self.rmg, rmi_id.date, 
                                    1, excludeWeekend=True)
        if len(dateList)==2:
            prevDate = dateList[0]
            prmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
            if prmi is not None:
                self.setFactorsForDate(prevDate, modelDB)
                prevFactors = set([f.name for f in factors])
                joiners = currFactors.difference(prevFactors)
                leavers = prevFactors.difference(currFactors)
                if len(joiners) > 0:
                    self.log.warning('%d factors joining model: %s', len(joiners), ', '.join(joiners))
                if len(leavers) > 0:
                    self.log.warning('%d factors leaving model: %s', len(leavers), ', '.join(leavers))
                self.setFactorsForDate(rmi_id.date, modelDB)
    
    def insertFactorCovariances(self, rmi_id, factorCov, subFactors, modelDB):
        """Inserts the factor covariances into the database for the given
        risk model instance.
        factorCov is a factor-factor array of the covariances.
        subFactors is a list of sub-factor IDs.
        """
        modelDB.insertFactorCovariances(rmi_id, subFactors, factorCov)
    
    def insertFactorReturns(self, date, factorReturns, modelDB, extraFactors=[]):
        """Inserts the factor returns into the database for the given date.
        factorReturns is an array of the return values.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        assert(len(subFactors) == len(self.factors))
        modelDB.insertFactorReturns(self.rms_id, date, subFactors,
                                    factorReturns)

    def insertInternalFactorReturns(self, date, factorReturns, modelDB):
        """Inserts the internal factor returns into the database for the given date.
        factorReturns is an array of the return values.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        assert(len(subFactors) == len(self.factors))
        modelDB.insertInternalFactorReturns(self.rms_id, date, subFactors,
                factorReturns)

    def insertStatFactorReturns(self, date, exp_date, factorReturns, modelDB):
        """Inserts the factor returns into the database for the given date.
        factorReturns is an array of the return values.
        """
        subFactors = modelDB.getSubFactorsForDate(exp_date, self.factors)
        assert(len(subFactors) == len(self.factors))
        modelDB.insertStatFactorReturns(self.rms_id, date, exp_date, subFactors, factorReturns)

    def insertRegressionStatistics(self, date, regressStats, factorNames,
                                   adjRsquared, pcttrade, modelDB, VIF=None,
                                   extraFactors=[], pctgVar=None):
        """Inserts the regression statistics into the database for the
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
            regressStats[:,1], regressStats[:,2], regressStats[:,3], VIF)
        modelDB.insertRMSStatistics(self.rms_id, date, adjRsquared, pctgVar=pctgVar)

    def insertSpecificReturns(self, date, specificReturns, subIssues, modelDB):
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
        modelDB.insertSpecificReturns(self.rms_id, date, subIssues, specificReturns)
    
    def insertSpecificRisks(self, rmi_id, specificVariance, subIssues,
                            specificCovariance, modelDB):
        """Inserts the specific risk into the database for the given date.
        specificVariance is a masked array of the specific variances.
        subIssues is an array containing the corresponding SubIssues.
        specificCovariance is a dictionary of dictionaries, mapping
        SubIssues to mappings of 'linked' SubIssue(s) and their covariance.
        """
        assert(len(specificVariance.shape) == 1)
        assert(specificVariance.shape[0] == len(subIssues))
        indices = numpy.flatnonzero(ma.getmaskarray(specificVariance) == 0)
        subIssues = numpy.take(numpy.array(
            subIssues, dtype=object), indices, axis=0)
        specificRisks = ma.sqrt(ma.take(specificVariance, indices, axis=0))
        modelDB.insertSpecificCovariances(rmi_id, specificCovariance)
        modelDB.insertSpecificRisks(rmi_id, subIssues, specificRisks)
    
    def loadCumulativeFactorReturns(self, date, modelDB):
        """Loads the cumulative factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        cumReturns = modelDB.loadCumulativeFactorReturnsHistory(
            self.rms_id, subFactors, [date]).data[:,0]
        return (cumReturns, self.factors)
    
    def loadEstimationUniverse(self, rmi_id, modelDB, data=None):
        """Loads the estimation universe(s) of the given risk model instance.
        Returns a list of sub-issue IDs.
        """
        estu = modelDB.getRiskModelInstanceESTU(rmi_id)
        assert(len(estu) > 0)
        return estu

    def loadExposureMatrix(self, rmi_id, modelDB, skipAssets=False, addExtraCountries=False, assetList=[]):
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
        if len(assetList) > 0:
            assets = assetList
        else:
            if skipAssets:
                assets = []
            elif assetList is None:
                assets = modelDB.getRiskModelInstanceUniverse(rmi_id)
            else:
                assets = modelDB.getRiskModelInstanceUniverse(rmi_id, returnExtra=addExtraCountries)

        # Set up an empty exposure matrix
        factorList = list()
        if not isinstance(self, StatisticalFactorModel) and \
                not isinstance(self, RegionalStatisticalFactorModel) and \
                not isinstance(self, StatisticalModel):
            if self.intercept is not None:
                factorList.append((self.intercept.name,
                                   ExposureMatrix.InterceptFactor))
            if len(self.styles) > 0:
                # Local factor hack
                realStyles = [f for f in self.styles
                              if f not in self.localStructureFactors]
                factorList.extend([(f.name, ExposureMatrix.StyleFactor)
                                   for f in realStyles])
                factorList.extend([(f.name, ExposureMatrix.LocalFactor)
                                   for f in self.localStructureFactors])
            if len(self.industries) > 0:
                factorList.extend([(f.name, ExposureMatrix.IndustryFactor)
                                   for f in self.industries])
            if len(self.rmg) > 1:
                if len(self.countries) > 0:
                    factorList.extend([(f.name, ExposureMatrix.CountryFactor)
                                       for f in self.countries])
                if addExtraCountries and hasattr(self, 'nurseryCountries') and (len(self.nurseryCountries) > 0):
                    factorList.extend([(f.name, ExposureMatrix.CountryFactor)
                                        for f in self.nurseryCountries])
                if len(self.currencies) > 0:
                    factorList.extend([(f.name, ExposureMatrix.CurrencyFactor)
                                       for f in self.currencies])
            if len(self.macro_core)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroCoreFactor) for f in self.macro_core])

            if len(self.macro_market_traded)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroMarketTradedFactor) for f in self.macro_market_traded])
        
            if len(self.macro_equity)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroEquityFactor) for f in self.macro_equity])

            if len(self.macro_sectors)>0:
                        factorList.extend([(f.name, ExposureMatrix.MacroSectorFactor) for f in self.macro_sectors])

        if len(self.blind) > 0:
            factorList.extend([(f.name, ExposureMatrix.StatisticalFactor)
                               for f in self.blind])
            if isinstance(self, RegionalStatisticalFactorModel) or \
                        isinstance(self, StatisticalModel):
                if len(self.currencies) > 0:
                    factorList.extend([(f.name, ExposureMatrix.CurrencyFactor)
                                       for f in self.currencies])
        # Force factorList to have the same order as self.factors
        # and hence, the list of subFactors
        expMFactorNames = [f[0] for f in factorList]
        expMFactorMap = dict(zip(expMFactorNames, factorList))
        factorList = [expMFactorMap[f.name] for f in factors]

        expM = Matrices.ExposureMatrix(assets, factorList)
        if (not skipAssets) and (len(assets) > 0):
            # Now fill it in from the DB
            if self.newExposureFormat:
                modelDB.getFactorExposureMatrixNew(rmi_id, expM, subFactorMap)
            else:
                modelDB.getFactorExposureMatrix(rmi_id, expM, subFactorMap)
        return expM
    
    def loadFactorCovarianceMatrix(self, rmi, modelDB):
        """Loads the factor-factor covariance matrix of the given risk
        model instance.
        Returns a (cov, factors) pair where cov is an m by m array
        containing the factor covariances and factors is a list of the
        m factor names.
        """
        subFactors = modelDB.getRiskModelInstanceSubFactors(rmi, self.factors)
        cov = modelDB.getFactorCovariances(rmi, subFactors)
        return (cov, self.factors)
    
    def loadFactorReturns(self, date, modelDB, addNursery=False, flag=None):
        """Loads the factor returns of the given dates.
        Returns a pair of lists with factor returns and names.
        """
        if addNursery:
            factors = self.factors + self.nurseryCountries
        else:
            factors = self.factors
        subFactors = modelDB.getSubFactorsForDate(date, factors)
        factorReturns = modelDB.loadFactorReturnsHistory(
            self.rms_id, subFactors, [date], flag=flag).data[:,0]
        return (factorReturns, factors)
    
    def loadFactorReturnsHistoryAsDF(self, startDate, endDate, modelDB):
        """Loads the factor returns history between the given dates
           Only loads factor returns for all the subfactors alive the end date
           Returns a DataFrame, index is dates, columns are factor names
        """
        modelDates = modelDB.getDateRange(self.rmg, startDate, endDate, excludeWeekend=True)
        subFactors = modelDB.getSubFactorsForDate(max(modelDates), self.factors)

        fr = modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, modelDates)
        facRetDF = pandas.DataFrame(fr.data, index=[f.factor.name for f in subFactors], columns=modelDates).T

        return facRetDF

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
        regressStats = modelDB.getRMSFactorStatistics(
            self.rms_id, date, subFactors)
        adjRsquared = modelDB.getRMSStatistics(self.rms_id, date)
        return (regressStats, factors, adjRsquared)
    
    def loadRiskModelGroupAssetMap(self, modelDate, base_universe, 
                                   modelDB, marketDB, addNonModelRMGs):
        """Returns a dict mapping risk model group IDs to a set
        of assets trading in that country/market.  base_universe is the
        list of assets that will be considered for processing.
        If addNonModelRMGs is True then the risk model groups for
        assets trading outside the mode are added to the map.
        Otherwise those assets won't be mapped.
        """
        logging.debug('loadRiskModelGroupAssetMap: begin')
        rmgAssetMap = dict()
        base_universe = set(base_universe)
        
        legacyExceptionMap = {'China': 21, 'Hong Kong': 42}
        exceptionRMGAssetMap = dict()
        for rmg in self.rmg:
            rmg_assets = set(modelDB.getActiveSubIssues(rmg, modelDate))
            # Legacy compatibility point
            if not self.modelHack.DRHandling:
                tradingRMG = legacyExceptionMap.get(rmg.description, None)
                if tradingRMG is not None:
                    quotationCty = modelDB.getRiskModelGroup(tradingRMG)
                    subIssues = modelDB.getActiveSubIssues(quotationCty, modelDate)
                    clsFamily = marketDB.getClassificationFamily('REGIONS')
                    assert(clsFamily is not None)
                    clsMembers = dict([(i.name, i) for i in marketDB.\
                                        getClassificationFamilyMembers(clsFamily)])
                    clsMember = clsMembers.get('HomeCountry', None)
                    assert(clsMember is not None)
                    clsRevision = marketDB.\
                            getClassificationMemberRevision(clsMember, modelDate)
                    clsData = modelDB.getMktAssetClassifications(clsRevision, 
                                    subIssues, modelDate, marketDB)
                    sidToAdd = [sid for (sid,cls) in clsData.items() \
                                  if cls.classification.description==rmg.description]
                    rmg_assets.update(sidToAdd)
                    exceptionRMGAssetMap[quotationCty] = sidToAdd
                    logging.info('Reassigned %d stocks from %s to %s RiskModelGroup', 
                            len(sidToAdd), quotationCty.description, rmg.description)
            
            rmg_assets = rmg_assets.intersection(base_universe)
            rmgAssetMap[rmg.rmg_id] = rmg_assets
            if len(rmg_assets)==0:
                self.log.error('No assets in %s', rmg.description)
            if self.modelHack.DRHandling:
                self.log.debug('%d assets in %s (RiskModelGroup %d, %s)',
                        len(rmg_assets), rmg.description, rmg.rmg_id, rmg.mnemonic)
                 
        # Another legacy compatibility point
        if not self.modelHack.DRHandling:
            for (rmg, sidList) in exceptionRMGAssetMap.items():
                if rmg.rmg_id in rmgAssetMap:
                    rmgAssetMap[rmg.rmg_id] = rmgAssetMap[rmg.rmg_id].difference(sidList)
        if len(base_universe) != sum([len(assets) for (rmg, assets)
                                      in rmgAssetMap.items()]):
            if addNonModelRMGs:
                missing = set(base_universe)
                for assets in rmgAssetMap.values():
                    missing -= assets
                sidRMGMap = dict(modelDB.getSubIssueRiskModelGroupPairs(
                                modelDate, missing))
                for sid in missing:
                    rmg_id = sidRMGMap[sid].rmg_id
                    if rmg_id not in rmgAssetMap:
                        rmg = modelDB.getRiskModelGroup(rmg_id)
                        self.log.debug('Adding %s (%s) to rmgAssetMap',
                                       rmg.description, rmg.mnemonic)
                    rmgAssetMap.setdefault(rmg_id, set()).add(sid)
        logging.debug('loadRiskModelGroupAssetMap: end')
        return rmgAssetMap
    
    def loadSpecificRisks(self, rmi_id, modelDB):
        """Loads the specific risks of the given risk model instance.
        Returns a tuple containing a dictionary mapping SubIssues to 
        their specific risks and a dictionary of dictionaries, mapping
        SubIssues to mappings of 'linked' SubIssues to their specific
        covariances.
        """
        return (modelDB.getSpecificRisks(rmi_id), 
                modelDB.getSpecificCovariances(rmi_id))
    
    def standardizeExposures(self, exposureMatrix, estu, marketCaps, 
                             modelDate, modelDB, marketDB, dr_indices=None,
                             subIssueGroups=None, writeStats=True):
        """Standardize exposures using Median Absolute Deviation (MAD)
        Required inputs are an ExposureMatrix object, ESTU indices, 
        and market caps.  No return value; the ExposureMatrix is updated
        accordingly with the standardized values.
        """
        self.log.debug('standardizeExposures: begin')
        if self.debuggingReporting and writeStats:
            exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                       modelDB, marketDB, modelDate, estu=estu,
                       subIssueGroups=subIssueGroups, dp=self.dplace)

        # Downweight some countries if required
        for r in [r for r in self.rmg if r.downWeight < 1.0]:
            fIdx = exposureMatrix.getFactorIndex(r.description)
            indices = numpy.flatnonzero(ma.getmaskarray(
                                exposureMatrix.getMatrix()[fIdx,:])==0)
            for i in indices:
                marketCaps[i] *= r.downWeight

        # Set MAD bounds and then standardize
        if self.modelHack.widerStandardizationMADBound:
            self.exposureStandardization.mad_bound = 8.0
        else:
            self.exposureStandardization.mad_bound = 5.2

        # Set weighting scheme for de-meaning
        if self.modelHack.capMean:
            self.exposureStandardization.capMean = True
        else:
            self.exposureStandardization.capMean = False

        self.exposureStandardization.standardize(
                            exposureMatrix, estu, marketCaps, modelDate, writeStats)

        if self.debuggingReporting and writeStats:
            exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=estu,
                    subIssueGroups=subIssueGroups, dp=self.dplace)

        self.log.debug('standardizeExposures: end')

    def proxyMissingAssetReturns(self, date, returns, data, modelDB):
        """Given a TimeSeriesMatrix of asset returns with masked 
        values, fill in missing values using proxies.  For a
        single-country model, proxies are based on GICS Industry
        Groups.  For regional models, assets are bucketized into 
        region, then divied into GICS Sectors.
        Returns a tuple containing the returns array (with no 
        masked values) and the asset buckets.
        """
        self.log.debug('proxyMissingAssetReturns: begin')
        regionAssetMap = dict()
        numLevels = self.industryClassification.\
                    getNumLevels(modelDB)
        # For regional models, always use Sector-level classification
        if len(self.rmg) > 1:
            clsLevel = -(numLevels-1)
            industries = self.industryClassification.\
                         getNodesAtLevel(modelDB, 'Sectors') 
        # For single-country models, move up one tier if possible
        else:
            clsLevel = -1
            if numLevels == 3:
                industries = self.industryClassification.\
                             getNodesAtLevel(modelDB, 'Industry Groups') 
            elif numLevels == 2:
                industries = self.industryClassification.\
                             getNodesAtLevel(modelDB, 'Sectors') 
            # If single-tier classification, just use what we have
            else:
                industries = self.industryClassification.\
                             getNodesAtLevel(modelDB, None) 
                clsLevel = None
        
        industryNames = [i.description for i in industries]
        
        # Bucket assets into regions
        for r in self.rmg:
            if not hasattr(self, 'rmgAssetMap'):
                rmg_assets = data.universe
            else:
                rmg_assets = self.rmgAssetMap[r.rmg_id]
            if r.region_id not in regionAssetMap:
                regionAssetMap[r.region_id] = list()
            regionAssetMap[r.region_id].extend(rmg_assets)
        
        # Bucket regions into industries/sectors
        buckets = list()
        self.log.info('Proxying returns using %d region(s) and %d industry buckets each',
                len(list(regionAssetMap.keys())), len(industryNames))
        for subids in regionAssetMap.values():
            exposures = self.industryClassification.getExposures(
                        date, subids, industryNames, modelDB, level=clsLevel)
            industryExposures = Matrices.ExposureMatrix(subids)
            industryExposures.addFactors(industryNames,
                        exposures, ExposureMatrix.IndustryFactor)
            industryBuckets = industryExposures.bucketize(
                                ExposureMatrix.IndustryFactor)
            buckets.extend([tuple([data.assetIdxMap[subids[i]] for i in j]) \
                                for j in industryBuckets])
        
        # Replace missing returns with proxied values
        returns.data = Utilities.compute_asset_value_proxies(returns.data, 
                        buckets, restrict=data.estimationUniverseIdx)
        
        self.log.debug('proxyMissingAssetReturns: end')
        return (returns, buckets)
    
    def setFactorsForDate(self, date, modelDB=None):
        """For now, time-variant factors only apply to 
        country/currency factors in regional models.
        """
        return self.setRiskModelGroupsForDate(date)
    
    def RMGDateLogic(self, r, date):
        """Logic for determining where in the timeline
        each RMG is
        """

        # Set defaults
        r.rmg.estuDownWeight = 1.0
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
            r.rmg.estuDownWeight = r.rmg.downWeight

        # Hard-code downweight for Saudi Arabia
        if r.rmg.mnemonic == 'SA':
            r.rmg.downWeight = 0.025

        # Report on weighting
        if r.rmg.downWeight < 1.0:
            self.log.debug('%s (RiskModelGroup %d, %s) down-weighted to %.3f%%',
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
        self.regions = [r.region_id for r in self.rmg]
        if len(self.rmg) > 1 and not isinstance(self, StatisticalFactorModel):
            self.currencyModel.setRiskModelGroupsForDate(date)
        return True

    def process_asset_country_assignments(self, date, univ, mcaps, 
                                  modelDB, marketDB, enforceCoverage=False):
        """Creates map of RiskModelGroup ID to list of its SubIssues,
        and checks for assets belonging to non-model geographies.
        Assets will also be checked for home country classification, 
        and removed from the intial list of SubIssues (univ) and mcaps 
        if no valid classification is found.
        """
        self.log.debug('process_asset_country_assignments: begin')
        # Check assets' country of quotation
        self.rmgAssetMap = self.loadRiskModelGroupAssetMap(
                                    date, univ, modelDB, marketDB, False)
        self.rmcAssetMap = self.loadRiskModelGroupAssetMap(
                                    date, univ, modelDB, marketDB, False)

        # Check for non-model geography assets
        nonModelAssets = set(univ)
        for ids in self.rmgAssetMap.values():
            nonModelAssets = nonModelAssets.difference(ids)
        self.log.info('%d assets from non-model geographies', len(nonModelAssets))
        if not self.modelHack.DRHandling:
            if len(nonModelAssets) > 0 and enforceCoverage:
                self.log.warning('problematic Axioma IDs: %s', 
                        ','.join([sid.getSubIDString() for sid in nonModelAssets]))
                if not self.forceRun:
                    raise Exception('This model is not equipped with DR-handling')
            return (univ, mcaps, list())

        assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                        self.rmgAssetMap.items() for sid in ids])
        
        # Load home country classification data
        clsFamily = marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry', None)
        assert(clsMember is not None)
        clsRevision = marketDB.\
                getClassificationMemberRevision(clsMember, date)
        clsData = modelDB.getMktAssetClassifications(
                        clsRevision, univ, date, marketDB)
        clsData2 = None
        
        missing = nonModelAssets.difference(list(clsData.keys()))
        validRiskModelGroups = set([r.mnemonic for r in self.rmg])
        rmgCodeIDMap = dict([(r.mnemonic, r.rmg_id) for r in self.rmg])
        foreign = list()
        for (sid, homeCty) in clsData.items():
            # Home country not in model geographies, check for
            # possible secondary home country
            if homeCty.classification.code not in validRiskModelGroups:
                if clsData2 is None:
                    clsMember2 = clsMembers.get('HomeCountry2', None)
                    assert(clsMember is not None)
                    clsRevision2 = marketDB.\
                            getClassificationMemberRevision(clsMember2, date)
                    clsData2 = modelDB.getMktAssetClassifications(
                                    clsRevision2, univ, date, marketDB)
                homeCty2 = clsData2.get(sid, homeCty)
                # If secondary home country is legit, assign it there
                if homeCty2.classification.code in validRiskModelGroups:
                    homeCty = homeCty2
                # Drop if neither quotation nor secondary country is valid
                elif sid in nonModelAssets:
                    missing.add(sid)
                    continue
                # Keep in quotation country otherwise
                else:
                    foreign.append(sid)
                    continue
            
            # If home country differs from RiskModelGroup, reassign
            tmpCode = homeCty.classification.code
            manualOverride = False
            tradingCty = assetRMGMap.get(sid, None)
            if (not manualOverride) and (rmgCodeIDMap[homeCty.classification.code] != tradingCty):
                if tradingCty is not None:
                    self.rmgAssetMap[tradingCty].remove(sid)
                self.rmgAssetMap[rmgCodeIDMap[homeCty.classification.code]].add(sid)
                foreign.append(sid)
             
            # Additional tweak in case asset is being assigned to a currency other
            # than the trading or that of the home country
            if (not manualOverride) and (rmgCodeIDMap[tmpCode] != tradingCty):
                if tradingCty is not None:
                    self.rmcAssetMap[tradingCty].remove(sid)
                self.rmcAssetMap[rmgCodeIDMap[tmpCode]].add(sid)
                foreign.append(sid)
        
        foreign = list(set(foreign))
        if len(foreign) > 0:
            self.log.info('%d DR-like instruments and/or foreign assets',
                          len(foreign))
        if len(missing) > 0:
            self.log.warning('%d assets dropped due to missing/invalid home country',
                          len(missing))
            self.log.debug('missing Axioma IDs: %s', ', '.join([
                        sid.getSubIDString() for sid in missing]))
            #keep = [(univ[i], mcaps[i]) for i in \
            #        range(len(univ)) if univ[i] not in missing]
            #univ, mcaps= zip(*keep)
            mcaps = [mcaps[i] for i in range(len(univ)) if univ[i] not in missing]
            univ = [univ[i] for i in range(len(univ)) if univ[i] not in missing] 

        self.log.debug('process_asset_country_assignments: end')
        return (univ, mcaps, foreign)
    
    def build_regional_covariance_matrix(self, date, dateList,
                currencyFactorReturns, nonCurrencyFactorReturns,
                crmi, modelDB, marketDB):
        """Somewhat fiddly routine that sets up the relevant data
        items required for regional covariance matrix computation
        then passes everything to the RiskCalculator class.
        Handles two distinct cases: one where the currency risk
        model is a dense model, and another where a statistical
        factor model is used for the currency risks.
        """
        self.log.info('build_regional_covariance_matrix: begin')
        self.log.info('Using currency model: %s, (factor model: %s)', 
                self.currencyModel.name, self.currencyModel.isFactorModel())

        currencySubFactors = currencyFactorReturns.assets
        if self.currencyModel.isFactorModel():
            # Load currency model with full set of currencies
            cmCurrencyFactors = self.currencyModel.getCurrencyFactors()
            cmCurrencySubFactors = modelDB.getSubFactorsForDate(date, cmCurrencyFactors)

            # Load currency statistical factor exposures
            currencyExpM = self.currencyModel.loadExposureMatrix(crmi, modelDB)
            cmCurrencyFactorNames = [c.getCashCurrency() for c \
                                     in currencyExpM.getAssets()]
            currencyExpM = numpy.transpose(currencyExpM.getMatrix().filled(0.0))
            
            # Re-order exposures in same order as currency SubFactors
            currencyIdxMap = dict([(j,i) for (i,j) in enumerate(cmCurrencyFactorNames)])
            noExposures = [cf.factor.name for cf in cmCurrencySubFactors if cf.factor.name not in currencyIdxMap]
            if len(noExposures) > 0:
                logging.warning('No currency exposures for the following currencies: %s', ','.join(noExposures))
            order = [currencyIdxMap[n.factor.name] for n in cmCurrencySubFactors]
            currencyExpM = ma.take(currencyExpM, order, axis=0)
            
            # Load currency "specific" risks
            curSpecificRisk = self.currencyModel.loadSpecificRisks(crmi, modelDB)
            curSpecificRisk = dict([(si.getCashCurrency(), val) for (si, val)
                                    in curSpecificRisk.items()])
            noSpecRisks = [cf.factor.name for cf in cmCurrencySubFactors if cf.factor.name not in curSpecificRisk]
            if len(noSpecRisks) > 0:
                logging.warning('No currency specific risks for the following currencies: %s', ','.join(noSpecRisks))
            
            # Load currency model estimation universe
            currESTU = set(c.getCashCurrency() for c in \
                    self.currencyModel.loadEstimationUniverse(crmi, modelDB))
            currencySubFactorsESTU = [f for f in cmCurrencySubFactors
                            if f.factor.name in currESTU]

            # Get raw currency returns for estu currencies
            estuCurrencyFactorReturns = self.currencyModel.loadCurrencyFactorReturnsHistory(
                            currencySubFactorsESTU, dateList, modelDB)
            
            # Estimate currency stat factor returns for required history length
            indices_ESTU = [idx for (idx, s) in enumerate(cmCurrencySubFactors)
                    if s.factor.name in currESTU]
            (currencyFR, dummy) = Utilities.ordinaryLeastSquares(
                    estuCurrencyFactorReturns.data.filled(0.0),
                    numpy.take(currencyExpM, indices_ESTU, axis=0))
            
            # Re-order and subset currency exposure
            # We now want only those currencies used in the regional model
            currencyIdxMap = dict([(j.factor.name,i) for (i,j) in enumerate(
                                    cmCurrencySubFactors)])
            order = [currencyIdxMap[n.factor.name] for n in currencySubFactors]
            currencyExpM = numpy.take(currencyExpM, order, axis=0)
            
            # Expand currency stat factor cov to full (dense) cov
            currencyCov = self.currencyModel.loadFactorCovarianceMatrix(
                    crmi, modelDB)[0]
            cc = numpy.dot(numpy.dot(currencyExpM, currencyCov),
                    currencyExpM.transpose()) / 252.0
            
            # Replace raw currency returns with currency common-factor returns
            currencyFactorReturns.data = numpy.dot(currencyExpM, currencyFR)

            # Finally, compute the factor covariance matrix
            self.covarianceCalculator.configureSubCovarianceMatrix(1, cc)
            factorCov = self.covarianceCalculator.\
                    computeFactorCovarianceMatrix(
                            nonCurrencyFactorReturns, currencyFactorReturns)

            # Add in the currency specific variances
            bDim = len(nonCurrencyFactorReturns.assets)
            for (idx, currencyFactor) in enumerate(currencySubFactors):
                srisk = curSpecificRisk[currencyFactor.factor.name]
                factorCov[bDim+idx, bDim+idx] += srisk * srisk
        else:
            # Load covariances from currency risk model
            (currencyCov, currencyFactors) = self.currencyModel.\
                    loadCurrencyCovarianceMatrix(crmi, modelDB)
            self.log.info('loaded %d currencies from dense currency model', 
                        len(currencyFactors))
        
            # Keep only SubFactors in the regional model and re-order
            currencyIdxMap = dict(zip([f.name for f in currencyFactors],
                        range(len(currencyFactors))))
            order = [currencyIdxMap[f.factor.name] for f in currencySubFactors \
                        if f.factor.name in currencyIdxMap]
            cc = ma.take(currencyCov, order, axis=0)
            cc = ma.take(cc, order, axis=1)
            cc /= 252.0
            
            # Remove any silly factor returns. 
            # Back-compatibility point
            if self.modelHack.MADFactorReturns:
                (currencyFactorReturns.data, bounds) = Utilities.mad_dataset(
                        currencyFactorReturns.data, self.xrBnds[0], self.xrBnds[1],
                        axis=0, treat='clip')
                (nonCurrencyFactorReturns.data, bounds) = Utilities.mad_dataset(
                        nonCurrencyFactorReturns.data, self.xrBnds[0], self.xrBnds[1],
                        axis=0, treat='clip')
            else:
                allFactorReturns = ma.concatenate((ma.ravel(
                                    currencyFactorReturns.data),
                                    ma.ravel(nonCurrencyFactorReturns.data)))
                (ret_mad, bounds) = Utilities.mad_dataset(allFactorReturns, 
                                    -25, 25, axis=0, treat='zero')
                currencyFactorReturns.data = ma.where(
                                    currencyFactorReturns.data < bounds[0],
                                    0.0, currencyFactorReturns.data)
                currencyFactorReturns.data = ma.where(
                                    currencyFactorReturns.data > bounds[1],
                                    0.0, currencyFactorReturns.data)
                nonCurrencyFactorReturns.data = ma.where(
                                    nonCurrencyFactorReturns.data < bounds[0],
                                    0.0, nonCurrencyFactorReturns.data)
                nonCurrencyFactorReturns.data = ma.where(
                                    nonCurrencyFactorReturns.data > bounds[1],
                                    0.0, nonCurrencyFactorReturns.data)
            
            # Build the factor covariance matrix
            self.covarianceCalculator.configureSubCovarianceMatrix(1, cc)
            factorCov = self.covarianceCalculator.\
                            computeFactorCovarianceMatrix(
                            nonCurrencyFactorReturns, currencyFactorReturns)

        # Safeguard to deal with nasty eigenvalues
        factorCov = Utilities.forcePositiveSemiDefiniteMatrix(
                factorCov, min_eigenvalue=0.0)

        # Report on estimated variances
        stDevs = numpy.sqrt(100.0 * numpy.diag(factorCov))
        stDevs = stDevs[:len(nonCurrencyFactorReturns.assets)]
        logging.info('Factor risk: [min: %.2f, max: %.2f, mean: %.2f]',
                min(stDevs), max(stDevs), numpy.average(stDevs))

        self.log.info('build_regional_covariance_matrix: end')
        return factorCov
    
    def getRMDates(self, date, modelDB, numDays, 
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
    
    def build_excess_return_history(self, dateList, data, currChangeIDs,
            expM, modelDB, marketDB):
        """Wrapper for building history of excess returns
        The logic is complex as an asset's risk-free rate ISO
        can change over time as its trading currency changes
        Thus, we need to loop over blocks of constant ISO
        """
        dateLen = (dateList[-1] - dateList[0]).days
        # Pick up cross-listing data
        if len(currChangeIDs) > 0:
            dr_indices = [data.assetIdxMap[n] for n in currChangeIDs]
            (drCurrData, currCodeIDMap) = GlobalExposures.\
                    getAssetHomeCurrencyID(data.universe, dr_indices,
                            expM, dateList[-1], modelDB)
        else:
            drCurrData = None
             
        # Load returns
        zeroMask = None
        if self.modelHack.specialDRTreatment:
            # Complex returns fill-in routine
            self.log.info('Using NEW ISC Treatment for returns processing')
            (returnsHistory, zeroMask, ipoMask) = self.process_returns_history(
                    dateList[-1], data.universe, dateLen, modelDB, marketDB, 
                    drCurrData=drCurrData, subIssueGroups=data.subIssueGroups)
            ma.put(returnsHistory.data, numpy.flatnonzero(zeroMask), 0.0)
        else:
            returnsHistory = modelDB.loadTotalReturnsHistory(
                    self.rmg, dateList[-1], data.universe,
                    dateLen, drCurrData, self.numeraire.currency_id)
             
        # Pick out only returns in the original list of dates
        retDateIdxMap = dict(zip(returnsHistory.dates, range(len(returnsHistory.dates))))
        dateIdxMap = dict(zip(dateList, range(len(dateList))))
        tmpReturns = Matrices.allMasked((returnsHistory.data.shape[0], len(dateList)))
        for d in dateList:
            if d in retDateIdxMap:
                tmpReturns[:,dateIdxMap[d]] = returnsHistory.data[:,retDateIdxMap[d]]
        returnsHistory.data = ma.array(tmpReturns, copy=True)
        returnsHistory.dates = list(dateList)
         
        # Redo masks
        if zeroMask is not None and self.modelHack.fullProxyFill:
            tmpReturns = ma.filled(tmpReturns, -999.0)
            tmpReturns = ma.masked_where(tmpReturns==0.0, tmpReturns)
            returnsHistory.zeroMask = ma.getmaskarray(tmpReturns)
            returnsHistory.data = ma.masked_where(returnsHistory.zeroMask, returnsHistory.data)

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
        allReturns = Matrices.allMasked((len(data.universe),len(dateList)), float)
        startDate = changeDates[0]
        iDate0 = 0
        for endDate in changeDates[1:]:
            # Get list of dates within given range
            subDateList = sorted(d for d in dateList if startDate <= d < endDate)
            self.log.info('Using returns from %s to %s',
                    subDateList[0], subDateList[-1])
            assert(hasattr(modelDB, 'currencyCache'))
            if len(currChangeIDs) > 0:
                # Map rmgs to their trading currency ID at endDate
                lastRangeDate = subDateList[-1]
                rmgCurrencyMap = dict()
                for r in self.rmgTimeLine:
                    if r.from_dt <= lastRangeDate \
                            and r.thru_dt > lastRangeDate:
                        ccyCode = r.rmg.getCurrencyCode(lastRangeDate)
                        ccyID = modelDB.currencyCache.getCurrencyID(
                            ccyCode, lastRangeDate)
                        rmgCurrencyMap[r.rmg_id] = ccyID
                # Force DRs to have the trading currency of their home
                # country for this period
                drCurrData = dict()
                for (rmg_id, rmgAssets) in self.rmgAssetMap.items():
                    drCurrency = rmgCurrencyMap.get(rmg_id)
                    if drCurrency is None:
                        # the model didn't cover the country earlier
                        # so get its information from the database
                        rmg = modelDB.getRiskModelGroup(rmg_id)
                        ccyCode = rmg.getCurrencyCode(lastRangeDate)
                        drCurrency = modelDB.currencyCache.getCurrencyID(
                            ccyCode, lastRangeDate)
                    for sid in rmgAssets & set(currChangeIDs):
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
            
            # Copy cunk of excess returns into full returns matrix
            iDate1 = iDate0 + len(subDateList)
            allReturns[:,iDate0:iDate1] = returns.data
            iDate0 += len(subDateList)
            startDate = endDate
        
        # Copy excess returns back into returns structure
        returns.data = allReturns
        returns.dates = dateList
        returns.assetIdxMap = returnsHistory.assetIdxMap
        if hasattr(returnsHistory, 'zeroMask'):
            returns.zeroMask = returnsHistory.zeroMask
        return returns

    def reportCorrelationMatrixChanges(self, currDate, currFactorCov, prmi, modelDB):
        """Compare the correlation matrices from the specified date with
        the one from the given RiskModelInstance.
        """
        subFactors = modelDB.getSubFactorsForDate(currDate, self.factors)
        self.setFactorsForDate(prmi.date, modelDB)
        prevSubFactors = modelDB.getSubFactorsForDate(prmi.date, self.factors)
        prevSubFactors = [s for s in subFactors if s in prevSubFactors]
        self.setFactorsForDate(currDate, modelDB)
        if prevSubFactors == subFactors:
            prevFactorCov = modelDB.getFactorCovariances(prmi, prevSubFactors)
            (d, currCorrMatrix) = Utilities.cov2corr(currFactorCov, fill=True)
            (d, prevCorrMatrix) = Utilities.cov2corr(prevFactorCov, fill=True)
            diff = ma.sum(abs(currFactorCov - prevFactorCov), axis=None)
            self.log.info('Day on day difference in covariance matrices: %s', diff)
    
    def usesReturnsTimingAdjustment(self):
        return (self.returnsTimingId is not None)

    def validateFactorStructure(self, date=None, warnOnly=False):
        """Check that ModelFactors associated with FactorRiskModel
        are consistent with factors stored in ModelDB.
        """
        dbFactors = set(self.descFactorMap.values())
        if isinstance(self, RegionalFundamentalModel) or \
                isinstance(self, SingleCountryFundamentalModel) or \
                isinstance(self, RegionalStatisticalFactorModel) or \
                (isinstance(self, FundamentalModel)) or \
                (isinstance(self, StatisticalModel)):
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
            raise LookupError('Mismatch between factors in model and database')

class SingleCountryFundamentalModel(FactorRiskModel):
    """Single-country fundamental factor model
    """
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])

        self.industries = []
        # Add factor IDs and from/thru dates to factors
        modelFactors = self.styles + self.industries
        for f in modelFactors:
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            if f in self.industries:
                f.name = dbFactor.name
            if isinstance(f, CompositeFactor):
                f.descriptors = dbFactor.descriptors

        # no statistical factors, of course
        self.blind = []
        self.VIF = None
        # Special regression treatment for certain factors 
        self.zeroExposureNames = []
        self.zeroExposureTypes = []
        modelDB.setVolumeCache(150)
    
    def setFactorsForDate(self, date, modelDB):
        self.setRiskModelGroupsForDate(date)

        # Setup industry classification
        chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
        self.industryClassification = self.industrySchemeDict[chngDates[-1]]
        self.log.debug('Using %s classification scheme, rev_dt: %s'%\
                      (self.industryClassification.name, chngDates[-1].isoformat()))

        # Create industry factors
        industries = list(self.industryClassification.getLeafNodes(modelDB).values())
        self.industries = [ModelFactor(None, f.description) for f in industries]
        for f in self.industries:
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            f.name = dbFactor.name

        self.factors = self.styles + self.industries
        self.factorIDMap = dict([(f.factorID, f) for f in self.factors])
        self.validateFactorStructure(date)
    
    def generate_fundamental_exposures(self, modelDate, data, modelDB, marketDB):
        """Compute the exposure based on fundamental data for all assets
        in data.universe and add them to the exposure matrix in
        data.exposureMatrix.
        They are book to price, debt to marketcap, and growth.
        Returns a structure containing arrays with the exposures.
        The field names are value, leverage, and growth.
        If data was not available, then the corresponding value is masked.
        Note that PACE calls the leverage value debtToEquity even
        though it is based on market capitalization, not common equity.
        """
        self.log.debug('generate_fundamental_exposures: begin')
        expM = data.exposureMatrix
        # Value
        expM.addFactor('Value', StyleExposures.generate_book_to_price(
            modelDate, data, self, modelDB, marketDB, 
            useQuarterlyData=self.quarterlyFundamentalData), ExposureMatrix.StyleFactor)
        # Leverage
        expM.addFactor('Leverage', StyleExposures.generate_debt_to_marketcap(
            modelDate, data, self, modelDB, marketDB, 
            useQuarterlyData=self.quarterlyFundamentalData), ExposureMatrix.StyleFactor)
        # Growth (use proxied div payout for countries outside US/CA)
        roe = StyleExposures.generate_return_on_equity(
            modelDate, data, self, modelDB, marketDB, 
            useQuarterlyData=self.quarterlyFundamentalData,
            legacy=self.modelHack.legacyROE)
        if not self.proxyDividendPayout:
            divPayout = StyleExposures.generate_dividend_payout(
                    modelDate, data, self, modelDB, marketDB, 
                    useQuarterlyData=self.quarterlyFundamentalData)
        else:
            divPayout = StyleExposures.generate_proxied_dividend_payout(
                    modelDate, data, self, modelDB, marketDB, 
                    useQuarterlyData=self.quarterlyFundamentalData,
                    includeStock=self.allowStockDividends)
        expM.addFactor('Growth', ((1.0 - divPayout) * roe),
                       ExposureMatrix.StyleFactor)
        
        self.log.debug('generate_fundamental_exposures: end')

    def generate_md_fundamental_exposures2(self,
                    modelDate, data, modelDB, marketDB):
        """Compute multiple-descriptor fundamental style exposures
        for assets in data.universe for all CompositeFactors in self.factors.
        data should be a Struct() containing the ExposureMatrix
        object as well as any required market data, like market
        caps, asset universe, etc.
        Factor exposures are computed as the equal-weighted average
        of all the normalized descriptors associated with the factor.
        NOTE: This is a one-off version used for the TW SCM model
        """
        self.log.debug('generate_md_fundamental_exposures2: begin')
        compositeFactors = [f for f in self.factors if isinstance(f, CompositeFactor)]
        if len(compositeFactors) == 0:
            self.log.warning('No CompositeFactors found!')
            return data.exposureMatrix
        descriptors = [d for f in compositeFactors for d in f.descriptors]
        descriptorExposures = Matrices.ExposureMatrix(data.universe)
        for d in descriptors:
            if d.description == 'Book-to-Price':
                values = StyleExposures.generate_book_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
#                values = ma.masked_where(values < 0.0, values)   # Mask negative BTP
            elif d.description == 'Earnings-to-Price':
                values = StyleExposures.generate_earnings_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                            legacy=self.modelHack.legacyETP)
            elif d.description == 'Sales-to-Price':
                values = StyleExposures.generate_sales_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Debt-to-Assets':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            useTotalAssets=True)
                values = ma.where(values < 0.0, 0.0, values)   # Negative debt -> 0.0
            elif d.description == 'Debt-to-MarketCap':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Plowback times ROE':
                roe = StyleExposures.generate_return_on_equity(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
                divPayout = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            includeStock=self.allowStockDividends)
                values = (1.0 - divPayout) * roe
            elif d.description == 'Sales Growth':
                values = StyleExposures.generate_sales_growth(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Earnings Growth':
                values = StyleExposures.generate_earnings_growth(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Dividend Yield':
                values = StyleExposures.generate_dividend_yield(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None)
            else:
                raise Exception('Undefined descriptor %s!' % d)
            descriptorExposures.addFactor(
                    d.description, values, ExposureMatrix.StyleFactor)

        # Add country factors to ExposureMatrix, needed for regional-relative standardization
        country_indices = data.exposureMatrix.\
                            getFactorIndices(ExposureMatrix.CountryFactor)
        if len(country_indices) > 0:
            countryExposures = ma.take(data.exposureMatrix.getMatrix(),
                                country_indices, axis=0)
            countryNames = data.exposureMatrix.getFactorNames(ExposureMatrix.CountryFactor)
            descriptorExposures.addFactors(countryNames, countryExposures,
                                ExposureMatrix.CountryFactor)

        # Standardize raw descriptors for multi-descriptor factors
        mat = descriptorExposures.getMatrix()
        origStandardization = copy.copy(self.exposureStandardization)
        if len(self.rmg) > 1:
            self.exposureStandardization = Standardization.BucketizedStandardization(
                    [Standardization.RegionRelativeScope(
                        modelDB, [d.description for d in descriptors])])
        else:
            self.exposureStandardization = Standardization.BucketizedStandardization(
                    [Standardization.GlobalRelativeScope([d.description for d in descriptors])])
        self.standardizeExposures(descriptorExposures, data.estimationUniverseIdx,
                        data.marketCaps, modelDate, modelDB, marketDB)

        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        for cf in compositeFactors:
            self.log.debug('CompositeFactor %s has %d descriptor(s)',
                    cf.description, len(cf.descriptors))
            valueList = [mat[descriptorExposures.getFactorIndex(
                                d.description),:] for d in cf.descriptors]
            if len(valueList) > 1:
                e = ma.average(ma.array(valueList), axis=0)
            else:
                e = ma.array(valueList)[0,:]
            data.exposureMatrix.addFactor(cf.description, e, ExposureMatrix.StyleFactor)

        # Proxy (standardized) exposures for assets missing data
        self.proxy_missing_exposures(modelDate, data, modelDB, marketDB,
                         factorNames=[cf.description for cf in compositeFactors])
        self.exposureStandardization = origStandardization

        self.log.debug('generate_md_fundamental_exposures2: end')
        return data.exposureMatrix

    def proxy_missing_exposures(self, modelDate, data, modelDB, marketDB,
                                factorNames=['Value', 'Growth', 'Leverage']):
        """Fill-in missing exposure values for the factors given in
        factorNames by cross-sectional regression.  For each region,
        estimation universe assets are taken, and their exposure values
        regressed against their Size and industry (GICS sector) exposures.
        Missing values are then extrapolated based on these regression
        coefficients, and trimmed to lie within [-1.5, 1.5] to prevent the
        proxies taking on extreme values.
        """
        self.log.debug('proxy_missing_exposures: begin')
        expM = data.exposureMatrix

        # Prepare sector exposures, will be used for missing data proxies
        roots = self.industryClassification.getClassificationRoots(modelDB)
        root = [r for r in roots if r.name=='Sectors'][0]
        sectorNames = [n.description for n in self.\
                industryClassification.getClassificationChildren(root, modelDB)]
        sectorExposures = Matrices.ExposureMatrix(data.universe)
        e = self.industryClassification.getExposures(modelDate,
                data.universe, sectorNames, modelDB, level=-(len(roots)-1))
        sectorExposures.addFactors(sectorNames, e, ExposureMatrix.IndustryFactor)

        # Set up regression to proxy raw exposures for assets missing data
        dr = RegressionToolbox.DummyMarketReturns()
        rp = {'fixThinFactors': True,
              'dummyReturns': dr,
              'dummyWeights': RegressionToolbox.AxiomaDummyWeights(10.0),
              'whiteStdErrors': False}
        returnCalculator = ReturnCalculator.RobustRegression2(
                        RegressionToolbox.RegressionParameters(rp))

        mat = expM.getMatrix()
        for f in factorNames:
            # Determine which assetes are missing data and require proxying
            values = mat[expM.getFactorIndex(f),:]
            missingIndices = numpy.flatnonzero(ma.getmaskarray(values))
            estuSet = set(data.estimationUniverseIdx)
            self.log.info('%d/%d assets missing %s fundamental data (%d/%d ESTU)',
                        len(missingIndices), len(values), f,
                        len(estuSet.intersection(missingIndices)), len(estuSet))
            if len(missingIndices)==0:
                continue

            # Loop around regions
            for (regionName, asset_indices) in self.exposureStandardization.\
                    factorScopes[0].getAssetIndices(expM, modelDate):
                good_indices = set(asset_indices).difference(missingIndices)
                good_indices = good_indices.intersection(estuSet)
                good_indices = list(good_indices)
                missing_indices = list(set(asset_indices).intersection(missingIndices))
                if len(good_indices) <= 10:
                    if len(missing_indices) > 0:
                        self.log.warning('Too few assets (%d) in %s with %s data present',
                                      len(good_indices), regionName, f)
                    continue

                # Assemble regressand, regressor matrix and weights
                weights = numpy.take(data.marketCaps, good_indices, axis=0)**0.5
                regressand = ma.take(values, good_indices, axis=0)
                regressor = ma.zeros((len(good_indices), len(sectorNames) + 1))
                regressor[:,0] = ma.take(mat[expM.getFactorIndex(
                                'Size'),:], good_indices, axis=0)
                regressor[:,1:] = ma.transpose(ma.take(
                        sectorExposures.getMatrix(), good_indices, axis=1).filled(0.0))
                regressor = ma.transpose(regressor)

                # Set up thin factor correction
                tfc = RegressionToolbox.DummyAssetHandler(
                            list(range(regressor.shape[0])), ['Size'] + sectorNames,
                            returnCalculator.parameters)
                tfc.nonzeroExposuresIdx = [0]
                dr = returnCalculator.parameters.getDummyReturns()
                dr.factorIndices = tfc.idxToCheck
                dr.factorNames = tfc.factorNames
                rets = dr.computeReturns(ma.array(regressand), list(range(len(weights))), weights,
                            [data.universe[j] for j in good_indices], modelDate)
                tfc.setDummyReturns(rets)
                tfc.setDummyReturnWeights(dr.dummyRetWeights)
                returnCalculator.thinFactorAdjustment = tfc

                # Run regression to get proxy values
                self.log.info('Running %s proxy regression for %s (%d assets)',
                        f, regionName, len(good_indices))
                (t0, coefs, t1, t2, t3, t4, t5, t6) = returnCalculator.calc_Factor_Specific_Returns(
                        self, list(range(regressor.shape[1])), regressand, regressor,
                        ['Size'] + sectorNames, weights, None,
                        returnCalculator.parameters.getFactorConstraints(),
                        robustRegression=self.useRobustRegression)

                # Substitute proxies for missing values
                self.log.info('Proxying %d %s exposures for %s',
                        len(missing_indices), f, regionName)
                for i in range(len(sectorNames)):
                    regSectorExposures = sectorExposures.getMatrix()[i,missing_indices]
                    reg_sec_indices = numpy.flatnonzero(ma.getmaskarray(regSectorExposures)==0)
                    if len(reg_sec_indices)==0:
                        continue
                    reg_sec_indices = [missing_indices[j] for j in reg_sec_indices]
                    sectorCoef = coefs[i+1]
                    sizeExposures = mat[expM.getFactorIndex(
                                    'Size'), reg_sec_indices]
                    proxies = sizeExposures * coefs[0] + sectorCoef
                    proxies = ma.where(proxies < -3.0, -3.0, proxies)
                    proxies = ma.where(proxies > 3.0, 3.0, proxies)
                    mat[expM.getFactorIndex(f),reg_sec_indices] = proxies

#        assert(numpy.sum(ma.getmaskarray(ma.take(expM.getMatrix(),
#            [expM.getFactorIndex(f) for f in factorNames], axis=0)))==0)
        self.log.debug('proxy_missing_exposures: begin')
        return expM

    def generate_market_exposures(self, modelDate, data, modelDB, marketDB):
        """Compute exposures for market factors for the assets 
        in data.universe and add them to data.exposureMatrix.
        """
        self.log.debug('generate_market_exposures: begin')
        expM = data.exposureMatrix

        # get the daily returns for the last 250 trading days
        requiredDays = 250
        returns = modelDB.loadTotalReturnsHistory(
            self.rmg, modelDate, data.universe, requiredDays-1, None)
        if returns.data.shape[1] < requiredDays:
            raise LookupError('Not enough previous trading days (need %d, got %d)' % (requiredDays, returns.data.shape[1]))

        (returns.data, bounds) = Utilities.mad_dataset(
                        returns.data, self.xrBnds[0], self.xrBnds[1],
                        restrict=data.estimationUniverseIdx, axis=0, 
                        zero_tolerance=0.25, treat=self.modelHack.MADRetsTreat)
        (returns, buckets) = self.proxyMissingAssetReturns(
                                modelDate, returns, data, modelDB)
        data.returns = returns

        # Run market model for betas, etc
        if self.modelHack.hasMarketSensitivity:
            (beta, alpha, sigma, resid) = Utilities.run_market_model(
                            returns, self.rmg[0], modelDB, marketDB, 120, 
                            sw=True, clip=True, indexSelector=self.indexSelector)
            expM.addFactor(
                'Market Sensitivity', beta, ExposureMatrix.StyleFactor)

        # 'Standard' set of style factors common across all models
        expM.addFactor(
            'Size', StyleExposures.generate_size_exposures(data),
            ExposureMatrix.StyleFactor)
        expM.addFactor(
            'Liquidity', StyleExposures.generate_trading_volume_exposures(
            modelDate, data, self.rmg, modelDB, 
            currencyID=self.numeraire.currency_id,
            legacy=self.modelHack.legacyTradingVolume), ExposureMatrix.StyleFactor)
        expM.addFactor(
            'Short-Term Momentum', StyleExposures.generate_short_term_momentum(
            returns), ExposureMatrix.StyleFactor)
        expM.addFactor(
            'Medium-Term Momentum', StyleExposures.generate_medium_term_momentum(
            returns), ExposureMatrix.StyleFactor)
        expM.addFactor(
            'Volatility', StyleExposures.generate_cross_sectional_volatility(
            returns, indices=data.estimationUniverseIdx), ExposureMatrix.StyleFactor)

        # Compute model-specific factors
        expM = self.generate_model_specific_exposures(
                                    modelDate, data, modelDB, marketDB)

        self.log.debug('generate_market_exposures: end')

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
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        data = Utilities.Struct()
        data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
        (mcapDates, goodRatio) = self.getRMDates(
                            modelDate, modelDB, 20, ceiling=False)
        data.marketCaps = modelDB.getAverageMarketCaps(
                    mcapDates, data.universe, self.numeraire.currency_id, marketDB)
        if not self.forceRun:
            assert(numpy.sum(ma.getmaskarray(data.marketCaps))==0)
        data.marketCaps = numpy.array(data.marketCaps)
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        
        if not hasattr(self, 'indexSelector'):
            self.indexSelector = MarketIndex.\
                    MarketIndexSelector(modelDB, marketDB)
        
        # Compute issuer-level market caps if required
        if self.modelHack.issuerMarketCaps:
            data.issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
                    data, modelDate, self.numeraire, modelDB, marketDB,
                    debugReport=self.debuggingReporting)
            data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()
        else:
            data.issuerMarketCaps = data.marketCaps.copy()
            data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()

        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB)
        assert(len(estu) > 0)
        data.estimationUniverseIdx = [data.assetIdxMap[n] for n in estu if n in data.assetIdxMap]

        # Generate 0/1 industry exposures
        self.generate_industry_exposures(
            modelDate, modelDB, marketDB, data.exposureMatrix)

        # Generate market exposures
        self.generate_market_exposures(modelDate, data, modelDB, marketDB)

        # Generate fundamental exposures
        if not self.modelHack.scmMDFundExp:
            self.generate_fundamental_exposures(modelDate, data, modelDB, marketDB)
        else:
            self.generate_md_fundamental_exposures2(modelDate, data, modelDB, marketDB)

        # Pick up dict of assets to be cloned from others
        data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)

        # Clone DR and cross-listing exposures if required
        subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        scores = self.score_linked_assets(
                modelDate, data.universe, modelDB, marketDB,
                subIssueGroups=subIssueGroups)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores,
                subIssueGroups=subIssueGroups)
        
        # Normalize style exposures using MAD
        self.standardizeExposures(data.exposureMatrix, data.estimationUniverseIdx, 
                data.marketCaps, modelDate, modelDB, marketDB, None, subIssueGroups)

        self.log.debug('generateExposureMatrix: end')
        return data
    
    def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate,
                        buildFMPs=False, internalRun=False):
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
        each factor that are stored in the array.  Currently, there
        are three statistics; the std. error, the t-statistic, and
        the Pr(>|t|).  The rows corresponding to the factors that are 
        not estimated via the cross-sectional regression are masked.
        The adjRsquared is a floating point number giving the 
        adjusted R-squared of the cross-sectional regression used 
        to compute the factor returns.
        """
        if buildFMPs:
            prevDate = modelDate
        elif internalRun:
            return
        else:
            dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
            if len(dateList) != 2:
                raise LookupError(
                    'no previous trading day for %s' %  str(modelDate))
            prevDate = dateList[0]

        # Get exposure matrix for previous trading day
        rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
        if rmi == None:
            raise LookupError(
                'no risk model instance for %s' % str(prevDate))
        if not rmi.has_exposures:
            raise LookupError(
                'no exposures in risk model instance for %s' % str(prevDate))

        self.setFactorsForDate(prevDate, modelDB)

        expM = self.loadExposureMatrix(rmi, modelDB)
        nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in \
                    modelDB.getSubFactorsForDate(prevDate, self.factors)])

        data = Utilities.Struct()
        data.universe = expM.getAssets()
        data.exposureMatrix = expM
    
        # Get estimation universe for today
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        assetIdxMap = dict(zip(data.universe, range(len(data.universe))))
        ids_ESTU0= self.loadEstimationUniverse(rmi, modelDB)
        ids_ESTU = [n for n in ids_ESTU0 if n in assetIdxMap]
        estu = [assetIdxMap[n] for n in ids_ESTU]
        if (len(ids_ESTU) != len(ids_ESTU0)):
            self.log.warning('%d assets in estu not in model', len(ids_ESTU0)-len(ids_ESTU))
            missingID = [sid.getSubIDString() for sid in ids_ESTU0 if sid not in ids_ESTU]
            self.log.debug('%s', missingID)
        
        # get day's returns
        # Load in returns history
        if self.modelHack.specialDRTreatment:
            self.log.info('Using NEW ISC Treatment for returns processing')
            univ = expM.getAssets()
            self.rmgAssetMap = self.loadRiskModelGroupAssetMap(
                    modelDate, univ, modelDB, marketDB, False)
            (assetReturnMatrix, zeroMask, ipoMask) = self.process_returns_history(
                    modelDate, univ, 0, modelDB, marketDB, drCurrData=None)
            assetReturnMatrix.data = assetReturnMatrix.data[:,-1][:,numpy.newaxis]
            assetReturnMatrix.dates = [assetReturnMatrix.dates[-1]]
            zeroMask = zeroMask[:,-1]
            del self.rmgAssetMap
        else:
            assetReturnMatrix = modelDB.loadTotalReturnsHistory(
                self.rmg, modelDate, expM.getAssets(), 0, None)
        
        # Compute excess returns
        if buildFMPs:
            assetReturnMatrix.data = ma.filled(assetReturnMatrix.data, 0.0)
        badRets = ma.masked_where(abs(assetReturnMatrix.data)<1.0e-12, assetReturnMatrix.data)
        badRets = numpy.flatnonzero(ma.getmaskarray(ma.take(badRets, estu, axis=0)))
        self.log.info('%.1f%% of ESTU returns missing or zero',
                100.0 * len(badRets) / float(len(estu)))
        (assetReturnMatrix, rfr) = self.computeExcessReturns(modelDate, 
                                assetReturnMatrix, modelDB, marketDB)
        excessReturns = assetReturnMatrix.data[:,0]
        
        # Calculate weights
        weights = ReturnCalculator.calc_Weights(
            self.rmg, modelDB, marketDB, prevDate, ids_ESTU, self.numeraire.currency_id)

        # Prepare thin industry correction mechanism
        rp = self.returnCalculator.parameters
        if rp.getThinFactorCorrection():
            if buildFMPs:
                tfc = RegressionToolbox.DummyAssetHandler([], expM.getFactorNames(), rp)
            else:
                tfc = RegressionToolbox.DummyAssetHandler(
                            expM.getFactorIndices(ExposureMatrix.IndustryFactor), 
                            expM.getFactorNames(), rp)
            # Allow dummy assets to have nonzero style exposures
            tfc.nonzeroExposuresIdx = expM.getFactorIndices(ExposureMatrix.StyleFactor)
            # Assign returns to dummies
            dr = rp.getDummyReturns()
            dr.factorIndices = tfc.idxToCheck
            dr.factorNames = tfc.factorNames
            rets = dr.computeReturns(excessReturns, estu, 
                                     weights, expM.getAssets(), prevDate)
            tfc.setDummyReturns(rets)
            tfc.setDummyReturnWeights(dr.dummyRetWeights)
            self.returnCalculator.thinFactorAdjustment = tfc
        
        regressorMatrix = expM.getMatrix()
        useRobustRegression = self.useRobustRegression
        if buildFMPs:
            useRobustRegression = False

        # Some models may require certain exposures to be set to zero
        if self.zeroExposureNames != []:
            # Loop round the factors
            for (i, factor) in enumerate(self.zeroExposureNames):
                # First determine the type of asset (e.g. Investment Trust)
                # for which we wish to set exposures to zero
                # If not defined, or set to None, all assets are included
                zeroExpAssetIdx = []
                if self.zeroExposureTypes[i] != [] and self.zeroExposureTypes[i] != None:
                    zeroExpFactorIdx = regressorMatrix[expM.getFactorIndex(\
                            self.zeroExposureTypes[i]), :]
                    zeroExpAssetIdx = numpy.flatnonzero(ma.getmaskarray(\
                            zeroExpFactorIdx)==0.0)
                # Now pick out the factor and set the relevant
                # exposures to zero
                idx = expM.getFactorIndex(factor)
                if zeroExpAssetIdx == []:
                    regressorMatrix[idx,:] = 0.0
                    self.log.info('Zeroing all exposures for %s factor', factor)
                else:
                    factorExp = regressorMatrix[idx,:]
                    ma.put(factorExp, zeroExpAssetIdx, 0.0)
                    regressorMatrix[idx,:] = factorExp
                    self.log.info('Zeroing all %s exposures for %s factor',
                                  self.zeroExposureTypes[i], factor)
        
        # Calculate factor and asset-specific returns
        (expMatrix, factorReturns, specificReturns, reg_ANOVA, constraintWeights,
                fmpDict, fmpSubIssues, constraintComp) = \
            self.returnCalculator.calc_Factor_Specific_Returns(
            self, estu, excessReturns, regressorMatrix, 
            expM.getFactorNames(), weights, expM,
            rp.getFactorConstraints(), force=self.forceRun,
            robustRegression=useRobustRegression)

        # Map specific returns for cloned assets
        hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)
        if len(hardCloneMap) > 0:
            cloneList = [n for n in data.universe if n in hardCloneMap]
            for sid in cloneList:
                if hardCloneMap[sid] in data.universe:
                    specificReturns[assetIdxMap[sid]] = specificReturns\
                            [assetIdxMap[hardCloneMap[sid]]]

        # Here we setup the factor structure for model date    
        factorReturnsMap = dict([(fName, factorReturns[fIdx]) for fIdx, fName in \
                                     enumerate(expM.getFactorNames())])
        regressionStatistics = numpy.ma.concatenate(
            [reg_ANOVA.regressStats_,
             Matrices.allMasked((reg_ANOVA.regressStats_.shape[0], 1))],
            axis=1)
        regStatsMap = dict([(fName, regressionStatistics[fIdx, :]) for fIdx, fName in \
                                enumerate(expM.getFactorNames())])

        self.setFactorsForDate(modelDate, modelDB)
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        subFactorIDIdxMap = dict([(subFactors[i].subFactorID, i) 
                                for i in range(len(subFactors))])

        factorReturns = Matrices.allMasked((len(self.factors),))
        fmpMap = dict()
        regressionStatistics = Matrices.allMasked((len(self.factors), 4))
        for (fName, ret) in factorReturnsMap.items():
            idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
            if idx is not None:
                factorReturns[idx] = ret
                regressionStatistics[idx,:] = regStatsMap[fName]
            if fName in fmpDict:
                fmpMap[fName] = dict(zip(fmpSubIssues, fmpDict[fName].tolist()))

        # Process FMPs
        result = Utilities.Struct()
        newFMPMap = dict()
        sid2StringMap = dict([(sid.getSubIDString(), sid) for sid in data.universe])
        for (fName, fMap) in fmpMap.items():
            tmpMap = dict()
            for (sidString, fmp) in fMap.items():
                if sidString in sid2StringMap:
                    tmpMap[sid2StringMap[sidString]] = fmp
            newFMPMap[nameSubIDMap[fName]] = tmpMap
        result.fmpMap = newFMPMap

        if self.returnCalculator.parameters.getCalcVIF():
            # Calculate Variance Inflation Factors for each style factor regressed on other style factors.
            self.VIF = self.returnCalculator.calculateVIF(
                        numpy.transpose(expMatrix), reg_ANOVA.weights_, reg_ANOVA.estU_, expM.factorIdxMap_)
        
        if self.debuggingReporting:
            for (i,sid) in enumerate(data.universe):
                if abs(specificReturns[i]) > 1.5:
                    self.log.warning('Large specific return for: %s, ret: %s',
                            sid, specificReturns[i])
        result.factorReturns = factorReturns
        result.specificReturns = specificReturns
        result.exposureMatrix = expM
        result.adjRsquared = reg_ANOVA.calc_adj_rsquared()
        result.regressionStatistics = regressionStatistics
        result.universe = expM.getAssets()
        result.regression_ESTU = list(zip([result.universe[i] for i in reg_ANOVA.estU_], 
                        reg_ANOVA.weights_ / numpy.sum(reg_ANOVA.weights_)))
        result.VIF = self.VIF
        
        self.regressionReporting(excessReturns, result, expM, nameSubIDMap, assetIdxMap,
                                    modelDate, buildFMPs=buildFMPs, constrComp=constraintComp,
                                    specificRets=specificReturns)

        return result
    
    def generateFactorSpecificRisk(self, date, modelDB, marketDB):
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
        """
        # get sub-factors and sub-issues active on this day
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        assert(len(subFactors) == len(self.factorIDMap))
        rmi = modelDB.getRiskModelInstance(self.rms_id, date)
        if rmi == None:
            raise LookupError('No risk model instance for %s' % str(date))
        if not rmi.has_exposures or not rmi.has_returns:
            raise LookupError(
                'Exposures or returns missing in risk model instance for %s'
                % str(date))
        subIssues = modelDB.getRiskModelInstanceUniverse(rmi)
        
        self.log.debug('building time-series matrices: begin')
        if len(self.rmg) == 1 and self.rmg[0].mnemonic == 'TW':
            #Exclude SAT trading day
            omegaDateList = modelDB.getDates(self.rmg, date, maxOmegaObs - 1, True)
        else:
            omegaDateList = modelDB.getDates(self.rmg, date, maxOmegaObs - 1)
        omegaDateList.reverse()
        if len(self.rmg) == 1 and self.rmg[0].mnemonic == 'TW':
            #Exclude SAT trading day
            deltaDateList = modelDB.getDates(self.rmg, date, maxDeltaObs - 1, True)
        else:
            deltaDateList = modelDB.getDates(self.rmg, date, maxDeltaObs - 1)
        deltaDateList.reverse()
        # Check that enough consecutive days have returns, try to get
        # the maximum number of observations
        if len(omegaDateList) > len(deltaDateList):
            dateList = omegaDateList
        else:
            dateList = deltaDateList
            
        rmiList = modelDB.getRiskModelInstances(self.rms_id, dateList)
        okDays = [i.date == j and i.has_returns for (i,j)
                  in zip(rmiList, dateList)]
        okDays.append(False)
        firstBadDay = okDays.index(False)
        dateList = dateList[:firstBadDay]
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        if len(dateList) < maxOmegaObs:
            self.log.info('Using only %d of %d days of factor return history',
                          len(dateList), maxOmegaObs)
        if len(dateList) < maxDeltaObs:
            self.log.info('Using only %d of %d days of specific return history',
                          len(dateList), maxDeltaObs)
        specificReturns = modelDB.loadSpecificReturnsHistory(
            self.rms_id, subIssues, dateList[:maxDeltaObs])
        factorReturns = modelDB.loadFactorReturnsHistory(
            self.rms_id, subFactors, dateList[:maxOmegaObs])
        self.log.debug('building time-series matrices: end')
        
        ret = Utilities.Struct()
        ret.subIssues = subIssues
        if isinstance(self.specificRiskCalculator, \
                RiskCalculator.BrilliantSpecificRisk) or \
                isinstance(self.specificRiskCalculator, \
                RiskCalculator.BrilliantSpecificRisk2009):
            expM = None
        else:
            expM = self.loadExposureMatrix(rmi, modelDB)
        
        # Load market caps and estimation universe
        mcapDates = modelDB.getDates(self.rmg, date, 19)
        avgMarketCap = modelDB.getAverageMarketCaps(
                    mcapDates, subIssues, self.numeraire.currency_id, marketDB)
        assetIdxMap = dict(zip(subIssues, range(len(subIssues))))
        ids_ESTU = self.loadEstimationUniverse(rmi, modelDB)
        ids_ESTU = [n for n in ids_ESTU if n in assetIdxMap]
        estu = [assetIdxMap[n] for n in ids_ESTU]

        # Pick up dict of assets to be cloned from others
        hardCloneMap = modelDB.getClonedMap(date, subIssues, cloningOn=self.hardCloning)

        # Compute specific risk
        ret.specificCov = dict()
        if self.modelHack.specRiskEstuFillIn:
            srEstuFillIn = estu
        else:
            srEstuFillIn = None
        if isinstance(self.specificRiskCalculator, RiskCalculator.SparseSpecificRisk2010):
            # ISC info
            subIssueGroups = modelDB.getIssueCompanyGroups(date, subIssues, marketDB)
            specificReturns.nameDict = modelDB.getIssueNames(
                    specificReturns.dates[0], subIssues, marketDB)
            if self.modelHack.specialDRTreatment:
                self.log.info('Using NEW ISC Treatment for specific covariance')
                scores = self.score_linked_assets(date, subIssues, modelDB, marketDB,
                        subIssueGroups=subIssueGroups)
            else:
                scores = None
            # Up-to-date models
            (ret.specificVars, ret.specificCov) = self.specificRiskCalculator.\
                    computeSpecificRisks(specificReturns, avgMarketCap, subIssueGroups,
                            restrict=srEstuFillIn, scores=scores,
                            specialDRTreatment=self.modelHack.specialDRTreatment,
                            debuggingReporting=self.debuggingReporting, hardCloneMap=hardCloneMap)
        else:
            ret.specificVars = self.specificRiskCalculator.\
                    computeSpecificRisks(expM, avgMarketCap, specificReturns, estu=srEstuFillIn)
         
        self.log.debug('computed specific variances')
        # Compute factor covariance matrix
        ret.subFactors = subFactors
        ret.factorCov = self.covarianceCalculator.\
                computeFactorCovarianceMatrix(factorReturns)
        # Report day-on-day correlation matrix changes
        self.reportCorrelationMatrixChanges(date, ret.factorCov, rmiList[1], modelDB)

        # Compute estimation universe factor risk
        factorNames = [f.factor.name for f in ret.subFactors]
        if expM is None:
            expM = self.loadExposureMatrix(rmi, modelDB)
        common_ids = set(ids_ESTU).intersection(set(expM.assets_))
        estu_exp = expM.toDataFrame().loc[common_ids, factorNames].fillna(0.0)
        estu_wgts = pandas.Series(avgMarketCap, index=ret.subIssues).loc[common_ids].fillna(0.0)
        factorCov = pandas.DataFrame(ret.factorCov, index=factorNames, columns=factorNames)
        estu_exp = estu_wgts.dot(estu_exp) / numpy.sum(estu_wgts.values, axis=None)
        estuFactorRisk = numpy.sqrt(estu_exp.dot(factorCov.dot(estu_exp.T)))
        self.log.info('Estimation universe factor risk: %.8f%%', 100.0*estuFactorRisk)

        self.log.debug('computed factor covariances')
        return ret
    
class StatisticalFactorModel(FactorRiskModel):
    """Single-country statistical model
    """
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        # add factor ID to factors, matching by description
        for f in self.blind:
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
        self.factorIDMap = dict([(f.factorID, f) for f in self.blind])
        self.factors = self.blind
        self.validateFactorStructure()
        # no style or industry factors
        self.styles = []
        self.industries = []
        self.returnsMADBounds = (-10.0, 10.0)
        # Create industry scheme dict
        if not hasattr(self, 'industrySchemeDict'):
            self.industrySchemeDict = dict()
            self.industrySchemeDict[datetime.date(1950,1,1)] = self.industryClassification

    def isStatModel(self):
        return True
    
    def isCurrencyModel(self):
        return False

    def compute_EP_statistic(self, assetReturns, specificReturns, estu, de_mean=True):
        """Computes model EP statistic and 'averaged R-squared' as
        defined in Connor (1995)
        """
        assetReturns = ma.take(assetReturns, estu, axis=0)
        specificReturns = ma.take(specificReturns, estu, axis=0)
        if de_mean:
            assetReturns = ma.transpose(ma.transpose(assetReturns) - \
                            ma.average(assetReturns, axis=1))
        numerator = numpy.sum([ma.inner(e,e) for e in specificReturns], axis=0)
        denominator = numpy.sum([ma.inner(r,r) for r in assetReturns], axis=0)
        ep = 1.0 - numerator/denominator
        self.log.info('EP statistic: %f', ep)
        sse = [float(ma.inner(e,e)) for e in ma.transpose(specificReturns)]
        sst = [float(ma.inner(r,r)) for r in ma.transpose(assetReturns)]
        sst = ma.masked_where(sst==0.0, sst)
        sst = 1.0 / sst
        avg_r2 = 1.0 - ma.inner(sse, sst.filled(0.0)) / len(sse)
        self.log.info('Average R-Squared: %f', avg_r2)
        return (ep, avg_r2)
    
    def generateStatisticalModel(self, modelDate, modelDB, marketDB):
        """Calculate exposures and returns for statistical factors.
        """
        
        # Get risk model universe and market caps
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        data = Utilities.Struct()
        data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
        (mcapDates, goodRatio) = self.getRMDates(
                modelDate, modelDB, 20, ceiling=False)
        data.marketCaps = modelDB.getAverageMarketCaps(
                    mcapDates, data.universe, self.numeraire.currency_id, marketDB)
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        if not self.forceRun:
            assert(numpy.sum(ma.getmaskarray(data.marketCaps))==0)
        data.marketCaps = numpy.array(data.marketCaps)

        # Setup industry classification
        chngDate = list(self.industrySchemeDict.keys())[0]
        chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= modelDate)
        self.industryClassification = self.industrySchemeDict[chngDates[-1]]
        self.log.debug('Using %s classification scheme', self.industryClassification.name)

        subIssueGroups = modelDB.getIssueCompanyGroups(
            modelDate, data.universe, marketDB)
        data.subIssueGroups = subIssueGroups
    
        # Load historical asset returns
        if len(self.rmg) > 1:
            needDays = int(self.returnHistory / goodRatio)
            baseCurrencyID = self.numeraire.currency_id
        else:
            needDays = self.returnHistory
            baseCurrencyID = None
            
        # Pick up dict of assets to be cloned from others
        data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)
         
        if self.modelHack.specialDRTreatment:
            self.log.info('Using NEW ISC Treatment for returns processing')
            self.rmgAssetMap = self.loadRiskModelGroupAssetMap(
                    modelDate, data.universe, modelDB, marketDB, False)
            # SHS Should drCurrData be baseCurrencyID here instead of None?
            (returns, zeroMask, ipoMask) = self.process_returns_history(
                    modelDate, data.universe, needDays-1, modelDB, marketDB,
                    drCurrData=None, subIssueGroups=data.subIssueGroups)
            del self.rmgAssetMap
        else:
            returns = modelDB.loadTotalReturnsHistory(self.rmg, modelDate, 
                        data.universe, needDays-1, assetConvMap=baseCurrencyID)
        
        # If regional model, remove dates on which lots of assets don't trade
        if len(self.rmg) > 1:
            io = numpy.sum(ma.getmaskarray(returns.data), axis=0)
            goodDatesIdx = numpy.flatnonzero(io < 0.7 * len(returns.assets))
            badDatesIdx = [i for i in range(len(returns.dates)) if \
                    i not in set(goodDatesIdx) and returns.dates[i].weekday() <= 4]
            if len(badDatesIdx) > 0:
                self.log.debug('Omitting weekday dates: %s',
                               ','.join([str(returns.dates[i])
                                         for i in badDatesIdx]))
            keep = min(len(goodDatesIdx), self.returnHistory)
            goodDatesIdx = goodDatesIdx[-keep:]
            returns.dates = [returns.dates[i] for i in goodDatesIdx]
            returns.data = ma.take(returns.data, goodDatesIdx, axis=1)
    
        # Locate returns with "good enough" histories
        io = (ma.getmaskarray(returns.data)==0)
        weightSums = numpy.sum(io.astype(numpy.float)/self.returnHistory, axis=1)
        reallyGoodThreshold = 0.50
        reallyGoodAssetsIdx = numpy.flatnonzero(weightSums > reallyGoodThreshold)
        
        # Compute excess returns
        (returns, rfr) = self.computeExcessReturns(modelDate, 
                                    returns, modelDB, marketDB)

        # Load estimation universe
        ids_ESTU = self.loadEstimationUniverse(rmi, modelDB)
        assert(len(ids_ESTU) > 0)
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        ids_ESTU = [n for n in ids_ESTU if n in assetIdxMap]
        estu = [assetIdxMap[n] for n in ids_ESTU]
        data.estimationUniverseIdx = estu
        realESTU = list(set(estu).intersection(reallyGoodAssetsIdx))
        logging.info('Removing %d assets from ESTU with more than %.2f%% of returns missing',
                len(estu)-len(realESTU), reallyGoodThreshold*100.0)
        
        # Fill-in missing returns with proxied values
        mask = numpy.array(ma.getmaskarray(returns.data))
        if self.industryClassification is not None:
            (returns, buckets) = self.proxyMissingAssetReturns(
                                    modelDate, returns, data, modelDB)
        
        # Truncate extreme values according to MAD bounds
        clipped_returns = Matrices.TimeSeriesMatrix(
                                    returns.assets, returns.dates)
        (clipped_returns.data, mad_bounds) = Utilities.mad_dataset(
                    returns.data, self.returnsMADBounds[0], 
                    self.returnsMADBounds[1], realESTU, axis=0)
        
        # Downweight some countries if required
        for r in [r for r in self.rmg if r.downWeight < 1.0]:
            for sid in self.rmgAssetMap[r.rmg_id].intersection(ids_ESTU):
                clipped_returns.data[data.assetIdxMap[sid],:] *= r.downWeight
        
        # Compute exposures, factor and specific returns
        self.log.debug('computing exposures and factor returns: begin')
        (expMatrix, factorReturns, specificReturns0, regressANOVA) = \
            self.returnCalculator.calc_ExposuresAndReturns(
                        clipped_returns, modelDate, realESTU)
        factorNames = [f.name for f in self.factors]
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        data.exposureMatrix.addFactors(factorNames, numpy.transpose(expMatrix),
                ExposureMatrix.StatisticalFactor)
        # Exposure cloning for cross-listings
        scores = self.score_linked_assets(
                modelDate, data.universe, modelDB, marketDB,
                subIssueGroups=data.subIssueGroups)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores,
                subIssueGroups=data.subIssueGroups)
        
        # Compute 'real' specific returns using non-clipped returns
        # Older stat models (JP, EU, EM, WW) used the wrong specific returns.
        # Keep them that way until we regen them and they get a new rms ID.
        # Back-compatibility point
        if self.modelHack.statModelCorrectSpecRet:
            specificReturns = returns.data - numpy.dot(expMatrix, factorReturns)
        else:
            specificReturns = specificReturns0
        
        # Map specific returns for cloned assets
        if len(data.hardCloneMap) > 0:
            cloneList = [n for n in data.universe if n in data.hardCloneMap]
            for sid in cloneList:
                if data.hardCloneMap[sid] in data.universe:
                    specificReturns[data.assetIdxMap[sid],:] = specificReturns\
                            [data.assetIdxMap[data.hardCloneMap[sid]],:]

        if self.debuggingReporting:
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=estu,
                    subIssueGroups=data.subIssueGroups, dp=self.dplace)
            dates = [str(d) for d in returns.dates]
            retOutFile = 'tmp/%s-facretHist-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(factorReturns, retOutFile, rowNames=factorNames,
                    columnNames=dates)
             
        # Compute various regression statistics
        data.adjRsquared = regressANOVA.calc_adj_rsquared()
        data.regressionStatistics = regressANOVA.calc_regression_statistics(
                            factorReturns[:,-1], ma.take(
                            expMatrix, regressANOVA.estU_, axis=0))
        data.regressionStatistics = numpy.ma.concatenate(
            [data.regressionStatistics,
             Matrices.allMasked((data.regressionStatistics.shape[0], 1))],
            axis=1)
        data.regression_ESTU = list(zip([data.universe[i] for i in regressANOVA.estU_], 
                        regressANOVA.weights_ / numpy.sum(regressANOVA.weights_)))
        (ep, avg_r2) = self.compute_EP_statistic(
                        clipped_returns.data, specificReturns0, realESTU)
        
        # Compute root-cap-weighted R-squared to compare w/ cross-sectional model
        # Note that this is computed over the initial ESTU, not the 'real'
        # one going into the factor analysis
        assetCurrMap = modelDB.getTradingCurrency(
                        modelDate, ids_ESTU, marketDB)
        currencies = set(assetCurrMap.values())
        if len(currencies) > 1:
            baseCurrencyID = self.numeraire.currency_id
        else:
            baseCurrencyID = None
        weights = ReturnCalculator.calc_Weights(self.rmg, modelDB, marketDB,
                    modelDate, ids_ESTU, baseCurrencyID)
        regANOVA = ReturnCalculator.RegressionANOVA(clipped_returns.data[:,-1],
                specificReturns0[:,-1], self.numFactors, estu, weights)
        data.adjRsquared = regANOVA.calc_adj_rsquared()
        self.log.debug('computing exposures and factor returns: end')
        
        # Create TimeSeriesMatrix objects for factor and specific returns
        self.log.debug('building time-series matrices: begin')
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        dateList = returns.dates
        dateList.reverse()
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        omegaObs = min(len(dateList), maxOmegaObs)
        deltaObs = min(len(dateList), maxDeltaObs)
        self.log.info('Using %d of %d days of factor return history',
                      omegaObs, len(dateList))
        self.log.info('Using %d of %d days of specific return history',
                      deltaObs, len(dateList))
        
        data.frMatrix = Matrices.TimeSeriesMatrix(
                                subFactors, dateList[:omegaObs])
        data.frMatrix.data = ma.array(numpy.fliplr(factorReturns))[:,:omegaObs]
        data.srMatrix = Matrices.TimeSeriesMatrix(
                                returns.assets, dateList[:deltaObs])
        # Mask specific returns corresponding to missing returns
        data.srMatrix.data = ma.masked_where(
                    numpy.fliplr(mask), numpy.fliplr(specificReturns.filled(0.0)))
        data.srMatrix.data = data.srMatrix.data[:,:deltaObs]
        self.log.debug('building time-series matrices: end')
        
        # Compute factor covariances and specific risks
        data.specificCov = dict()
        if self.modelHack.specRiskEstuFillIn:
            srEstuFillIn = estu
        else:
            srEstuFillIn = None
        if isinstance(self.specificRiskCalculator, RiskCalculator.SparseSpecificRisk2010):
            # ISC info
            data.srMatrix.nameDict = modelDB.getIssueNames(
                    data.srMatrix.dates[0], returns.assets, marketDB)
            if self.modelHack.specialDRTreatment:
                self.log.info('Using NEW ISC Treatment for specific covariance')
                scores = self.score_linked_assets(modelDate, returns.assets, modelDB, marketDB,
                        subIssueGroups=data.subIssueGroups)
            else:
                scores = None
            # Up-to-date models
            (data.specificVars, data.specificCov) = self.specificRiskCalculator.\
                    computeSpecificRisks(data.srMatrix, data.marketCaps, data.subIssueGroups,
                            restrict=srEstuFillIn, scores=scores, specialDRTreatment=self.modelHack.specialDRTreatment,
                            debuggingReporting=self.debuggingReporting, hardCloneMap=data.hardCloneMap)
        else:
            data.specificVars = self.specificRiskCalculator.\
                    computeSpecificRisks(data.exposureMatrix, data.marketCaps,
                            data.srMatrix, estu=srEstuFillIn)
        self.log.debug('computed specific variances')
         
        data.factorCov = self.covarianceCalculator.\
                computeFactorCovarianceMatrix(data.frMatrix)
        self.log.debug('computed factor covariances')
        return data

class SingleCountryHybridModel(FactorRiskModel):
    """Single-country hybrid factor model
    """
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        descFactorMap = dict([(i.description, i) for i in factors])
        # Add factor IDs and from/thru dates to style factors
        for f in self.styles:
            f.factorID = descFactorMap[f.description].factorID
            f.from_dt = descFactorMap[f.description].from_dt
            f.thru_dt = descFactorMap[f.description].thru_dt
        # Create industry factors
        industries = list(self.industryClassification.getLeafNodes(
                                        modelDB).values())
        self.industries = [ModelFactor(None, f.description)
                           for f in industries]
        for f in self.industries:
            f.name = descFactorMap[f.description].name
            f.factorID = descFactorMap[f.description].factorID
            f.from_dt = descFactorMap[f.description].from_dt
            f.thru_dt = descFactorMap[f.description].thru_dt
        
        # Add factor IDs and from/thru dates to blind factors
        for f in self.blind:
            f.factorID = descFactorMap[f.description].factorID
            f.from_dt = descFactorMap[f.description].from_dt
            f.thru_dt = descFactorMap[f.description].thru_dt
        
        self.factorIDMap = dict([(f.factorID, f) for f in \
                        self.styles + self.industries + self.blind])
        assert(len(factors) == len(self.industries) + \
                        len(self.styles) + len(self.blind))
        self.factors = self.styles + self.industries + self.blind
    
    def setFactorsForDate(self, date, modelDB=None):
        """For now, time-variant factors only apply to 
        country/currency factors in regional models.
        """
        return self.setRiskModelGroupsForDate(date)
    
    def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate):
        """Same as the method for SingleCountryFundamentalModel
        except for the small hack at the end to add dummy entries
        to the exposureMatrix object for blind factors.
        """
        dateList = modelDB.getDates(self.rmg, modelDate, 1)
        if len(dateList) != 2:
            raise LookupError(
                'no previous trading day for %s' %  str(modelDate))
        prevDate = dateList[0]
        
        # Get exposure matrix for previous trading day
        rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
        if rmi == None:
            raise LookupError(
                'no risk model instance for %s' % str(prevDate))
        if not rmi.has_exposures:
            raise LookupError(
                'no exposures in risk model instance for %s' % str(prevDate))
        
        expM = self.loadExposureMatrix(rmi, modelDB)
        data = Utilities.Struct()
        data.universe = expM.getAssets()
        data.exposureMatrix = expM
        mcapDates = modelDB.getDates(self.rmg, prevDate, 19)
        data.marketCaps = modelDB.getAverageMarketCaps(
                    mcapDates, data.universe, None, marketDB)
        # Get estimation universe for today
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        assetIdxMap = dict(zip(expM.getAssets(), range(len(expM.getAssets()))))
        ids_ESTU = self.loadEstimationUniverse(rmi, modelDB)
        ids_ESTU = [n for n in ids_ESTU if n in assetIdxMap]
        estu = [assetIdxMap[n] for n in ids_ESTU]
        weights = ReturnCalculator.calc_Weights(
                            self.rmg, modelDB, marketDB, prevDate, ids_ESTU)
        
        # get day's returns
        assetReturnMatrix = modelDB.loadTotalReturnsHistory(
            self.rmg, modelDate, expM.getAssets(), 0, None)
        assetReturnMatrix.data = ma.reshape(assetReturnMatrix.data,
                        (assetReturnMatrix.data.shape[0],))
        
        # Get risk-free rate from database
        (assetReturnMatrix, rfr) = self.computeExcessReturns(modelDate, 
                                assetReturnMatrix, modelDB, marketDB)
        excessReturns = assetReturnMatrix.data[:,0]
        
        # Assemble regressor matrix of style + industry factors
        keepIndices = [i for i in range(expM.getMatrix().shape[0]) 
                                if i not in expM.getFactorIndices(ExposureMatrix.StatisticalFactor)]
        regressorMatrix = ma.take(expM.getMatrix(), keepIndices, axis=0)
        nonStatFactorNames = [f for f in expM.getFactorNames() \
                                if not expM.checkFactorType(f, ExposureMatrix.StatisticalFactor)]
        
        # Prepare thin industry correction mechanism
        rp = self.returnCalculator1.parameters
        if rp.getThinFactorCorrection():
            tfc = ReturnCalculator.DummyAssetHandler(
                        expM.getFactorIndices(ExposureMatrix.IndustryFactor), 
                        nonStatFactorNames, rp)
            # Allow dummy assets to have nonzero style exposures
            tfc.nonzeroExposuresIdx = expM.getFactorIndices(ExposureMatrix.StyleFactor)
            # Assign returns to dummies
            dr = rp.getDummyReturns()
            dr.factorIndices = tfc.idxToCheck
            dr.factorNames = tfc.factorNames
            rets = dr.computeReturns(excessReturns, estu, 
                                     weights, expM.getAssets(), prevDate)
            tfc.setDummyReturns(rets)
            self.returnCalculator1.thinFactorAdjustment = tfc
        
        # Calculate factor and asset-specific returns
        (expMatrix, factorReturns, specificReturns, reg_ANOVA) = \
                self.returnCalculator1.calc_Factor_Specific_Returns(
                        estu, excessReturns, regressorMatrix,
                        nonStatFactorNames, weights, expM,
                        robustRegression=self.useRobustRegression)
        
        # Add bogus entries for blind factors' regression stats
        tmp = Matrices.allMasked((len(self.factors), 3))
        tmp[:len(keepIndices),:] = reg_ANOVA.regressStats_
        result = Utilities.Struct()
        result.regressionStatistics = tmp
        result.regressionStatistics = numpy.ma.concatenate(
            [result.regressionStatistics,
             Matrices.allMasked((result.regressionStatistics.shape[0], 1))],
            axis=1)
        result.factorReturns = factorReturns
        result.specificReturns = specificReturns
        result.exposureMatrix = expM
        result.adjRsquared = reg_ANOVA.calc_adj_rsquared()
        # result.estimationUniverse = reg_ANOVA.estU_
        result.universe = expM.getAssets()
        return result
    
    def generate_pseudo_factor_returns(self, modelDate, expMatrix, modelDB):
        """Computes a 'pseudo-history' of statistical factor returns,
        based on the assumption that statistical exposures are constant
        through time.  Returns an asset-by-time array of factor 
        returns, not a TimeSeriesMatrix.
        """
        # Determine date range and extract specific returns
        (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
        dateList = modelDB.getDates(self.rmg, modelDate, maxOmegaObs-1)
        dateList.reverse()
        dateList = dateList[self.returnHistory:]
        residMatrix = modelDB.loadSpecificReturnsHistory(
                        self.rms_id, expMatrix.getAssets(), dateList)
        
        # Compute factor returns by regression
        regressorMatrix = numpy.transpose(ma.take(expMatrix.getMatrix(), 
                        expMatrix.getFactorIndices(ExposureMatrix.StatisticalFactor), axis=0).filled(0.0))
        (fr, err) = Utilities.ordinaryLeastSquares(
                            residMatrix.data.filled(0.0), regressorMatrix)
        
        self.log.info('Created %d days of pseudo stat history', fr.shape[1])
        return fr
    
    def generateBlindFactorSpecificReturns(self, modelDB, marketDB, modelDate):
        """Estimates factor returns for statistical factors using
        residuals from fundamental factor model and whatever
        factor analysis method is specified in self.returnCalculator2.
        Computes 'final' regression statistics too.
        """
        dateList = modelDB.getDates(self.rmg, modelDate, 1)
        if len(dateList) != 2:
            raise LookupError(
                'no previous trading day for %s' %  str(modelDate))
        prevDate = dateList[0]
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        if rmi == None:
            raise LookupError('no risk model instance for %s' % str(modelDate))
        if not rmi.has_exposures:
            raise LookupError('no exposures in risk model instance for %s' % str(modelDate))
        
        # Load time series of residuals from main regression for stat model
        dateList = modelDB.getDates(
                        self.rmg, modelDate, self.returnHistory-1)
        expM = self.loadExposureMatrix(rmi, modelDB)
        universe = expM.getAssets()
        retval = Utilities.Struct()
        residMatrix = modelDB.loadSpecificReturnsHistory(
                        self.rms_id, universe, dateList)
        
        # Weed out assets with insufficient returns history
        io = (ma.getmaskarray(residMatrix.data) == 0)
        weightSums = numpy.sum(io.astype(numpy.float)/self.returnHistory, axis=1)
        reallyGoodThreshold = 0.95
        reallyGoodAssetsIdx = numpy.flatnonzero(weightSums > reallyGoodThreshold)
        # residMatrix.data = residMatrix.data.filled(0.0)
        
        # Load estimation universe
        assetIdxMap = dict(zip(universe, list(range(len(universe)))))
        estu = [assetIdxMap[n] for n in self.loadEstimationUniverse(rmi, modelDB)
                if n in assetIdxMap]
        estu_set = set(estu)
        estu = list(estu_set.intersection(set(reallyGoodAssetsIdx)))
        
        # Truncate extreme values
        (residMatrix.data, mad_bounds) = Utilities.mad_dataset(residMatrix.data,
                self.returnsMADBounds[0], self.returnsMADBounds[1],
                estu, axis=0)
        
        # Compute blind factor returns and specific returns
        (expMatrix, blindFactorReturns, specificReturns, regressANOVA) = \
                self.returnCalculator2.calc_ExposuresAndReturns(residMatrix, modelDate, estu)
        
        # Mask specific returns corresponding to masked returns
        returnsMask = ma.getmaskarray(residMatrix.data)
        specificReturns = ma.masked_where(returnsMask, specificReturns)
        
        # Regression statistics 
        retval.regressionStatistics = Matrices.allMasked((len(self.factors), 3))
        retval.regressionStatistics[len(self.styles + self.industries):,:] = \
                    regressANOVA.calc_regression_statistics(
                    blindFactorReturns[:,-1], ma.take(
                        expMatrix, regressANOVA.estU_, axis=0))
        
        # Compute adjusted R-squared
        assetReturnMatrix = modelDB.loadTotalReturnsHistory(
                    self.rmg, modelDate, universe, 0, None)
        ids_ESTU = [universe[i] for i in estu]
        weights = ReturnCalculator.calc_Weights(
                self.rmg, modelDB, marketDB, prevDate, ids_ESTU)
        totalRegressionANOVA = ReturnCalculator.RegressionANOVA(
                    assetReturnMatrix.data[:,0], specificReturns[:,-1], 
                    len(self.factors), estu, weights)
        retval.adjRsquared = totalRegressionANOVA.calc_adj_rsquared()
        
        # Add statistical factor exposures
        factorNames = expM.getFactorNames()
        exposureMatrix = Matrices.ExposureMatrix(universe)
        exposureMatrix.addFactors(factorNames[-len(self.blind):],
                                    numpy.transpose(expMatrix), ExposureMatrix.StatisticalFactor)
        
        # Add bogus factors to ExposureMatrix object
        names = [f.name for f in self.styles + self.industries]
        exposureMatrix.addFactors(names, Matrices.allMasked( 
            (len(names), len(universe))), ExposureMatrix.StyleFactor)
        
        retval.exposureMatrix = exposureMatrix
        retval.blindFactorReturns = blindFactorReturns
        retval.specificReturns = specificReturns
        retval.factorReturns = Matrices.allMasked((len(self.factors)))
        retval.factorReturns[-self.numFactors:] = blindFactorReturns[:,-1]
        retval.universe = universe
        return retval
    
    def generateFactorSpecificRisk(self, modelDate, modelDB, 
                blindFactorReturns=None, specificReturns=None, sr_univ=None):
        """Computes the factor covariance matrix and specific variances.
        Although the fundamental factor returns have a longer history, 
        a 'pseudo-history' of blind factor returns is computed using 
        the statistical factor model exposures and fundamental factor
        model residuals.
        """
        
        # Create TimeSeriesMatrix objects for factor and specific returns
        self.log.debug('building time-series matrices: begin')
        (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
        (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        dateList = modelDB.getDates(self.rmg, modelDate, maxOmegaObs-1)
        dateList.reverse()
        if len(dateList) < max(minOmegaObs, minDeltaObs, self.returnHistory):
            required = max(minOmegaObs, minDeltaObs, self.returnHistory)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError('%d missing risk model instances for required days' \
                        % (required - len(dateList)))
        if len(dateList) < maxOmegaObs:
            self.log.info('Using only %d of %d days of factor return history',
                          len(dateList), maxOmegaObs)
        if len(dateList) < maxDeltaObs:
            self.log.info('Using only %d of %d days of specific return history',
                          len(dateList), maxDeltaObs)
        
        # Get sub-factors and sub-issues
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        expM = self.loadExposureMatrix(rmi, modelDB)
        universe = expM.getAssets()
        
        # Construct factor returns history matrix
        frMatrix = Matrices.TimeSeriesMatrix(subFactors, dateList)
        fr = modelDB.loadFactorReturnsHistory(
                        self.rms_id, subFactors, dateList)
        frMatrix.data = fr.data
        
        # Create pseudo history for stat factors
        if blindFactorReturns is not None \
                and specificReturns is not None and sr_univ is not None:
            fr_blind = self.generate_pseudo_factor_returns(modelDate, expM, modelDB)
            frMatrix.data[len(self.styles)+len(self.industries):,self.returnHistory:] = fr_blind
            frMatrix.data[len(self.styles)+len(self.industries):,:self.returnHistory] = \
                    numpy.array(numpy.fliplr(blindFactorReturns))
        
        # Construct specific returns history matrix
        srMatrix = modelDB.loadSpecificReturnsHistory(
                    self.rms_id, universe, dateList[:maxDeltaObs])
        self.log.debug('buliding time-series matrices: end')
        
        # Compute factor covariances and specific risks
        retval = Utilities.Struct()
        retval.factorCov = self.covarianceCalculator.\
                        computeFactorCovarianceMatrix(frMatrix)
        self.log.debug('computed factor covariances')
        mcapDates = modelDB.getDates(self.rmg, modelDate, 19)
        avgMarketCap = modelDB.getAverageMarketCaps(
                    mcapDates, universe, None, marketDB)
        retval.specificVars = self.specificRiskCalculator.\
                        computeSpecificRisks(expM, avgMarketCap, srMatrix)
        self.log.debug('computed specific variances')
        
        # Return Struct containing everything we need
        retval.subFactors = frMatrix.assets
        retval.subIssues = srMatrix.assets 
        return retval

class RegionalFundamentalModel(FactorRiskModel):
    """Regional (or global) model class
    """
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        
        if self.allCurrencies:
            self.allRMG = modelDB.getAllRiskModelGroups(inModels=True)

        # Add factor IDs and from/thru dates to factors
        if self.intercept is not None:
            modelFactors = self.styles + [self.intercept]
        else:
            modelFactors = self.styles

        for f in modelFactors:
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            if isinstance(f, CompositeFactor):
                f.descriptors = dbFactor.descriptors
        
        self.blind = []
        self.VIF = None
        modelDB.setTotalReturnCache(367)
        modelDB.setVolumeCache(190)
    
    def setFactorsForDate(self, date, modelDB):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)
        
        # Create country and currency factors
        self.countries = [ModelFactor(r.description, None) for r in self.rmg]
        self.currencies = [ModelFactor(f, None) for f in set([r.currency_code for r in self.rmg])]

        # Setup industry classification
        chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
        self.industryClassification = self.industrySchemeDict[chngDates[-1]]
        self.log.debug('Using %s classification scheme, rev_dt: %s'%\
                      (self.industryClassification.name, chngDates[-1].isoformat()))

        # Create industry factors
        industries = list(self.industryClassification.getLeafNodes(modelDB).values())
        self.industries = [ModelFactor(None, f.description) for f in industries]
        for f in self.industries:
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
            f.name = dbFactor.name

        # Add additional currency factors (to allow numeraire changes)
        # if necessary
        if self.allCurrencies:
            for rmg in self.allRMG:
                rmg.setRMGInfoForDate(date)
            additionalCurrencyFactors = [ModelFactor(f, None)
                    for f in set([r.currency_code for r in self.allRMG])]
            additionalCurrencyFactors.extend([ModelFactor('EUR', 'Euro')])
            self.additionalCurrencyFactors = [f for f in additionalCurrencyFactors
                    if f not in self.currencies and f.name in self.nameFactorMap]
            self.additionalCurrencyFactors = list(set(self.additionalCurrencyFactors))

        for f in self.additionalCurrencyFactors:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        self.additionalCurrencyFactors = [f for f in self.additionalCurrencyFactors
                if f not in self.currencies and f.isLive(date)]
        if len(self.additionalCurrencyFactors) > 0:
            self.currencies.extend(self.additionalCurrencyFactors)
            self.log.debug('Adding %d extra currencies: %s', 
                    len(self.additionalCurrencyFactors),
                    [f.name for f in self.additionalCurrencyFactors])
        self.currencies = sorted(self.currencies)

        # Assign factor IDs and names
        regional = self.countries + self.currencies
        for f in regional:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        allFactors = self.styles + self.industries + regional
        if self.intercept is not None:
            allFactors = [self.intercept] + allFactors
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factors = allFactors
        self.validateFactorStructure(date)
        
    def generate_pd_fundamental_exposures(self, 
                    modelDate, data, modelDB, marketDB):
        """Compute the pure descriptor exposures for assets in data.universe
        for all fundamental factors in self.factors.
        data should be a Struct() containing the ExposureMatrix
        object as well as any required market data, like market
        caps, asset universe, etc.
        """
        self.log.debug('generate_pd_fundamental_exposures: begin')
        for f in self.factors:
            if f.name == 'Value':
                btp = StyleExposures.generate_book_to_price(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData)
                btp = ma.masked_where(btp < 0.0, btp)   # Mask negative BTP
                etp = StyleExposures.generate_earnings_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                            legacy=self.modelHack.legacyETP)
                values = ma.average(ma.concatenate((btp[numpy.newaxis,:],\
                        etp[numpy.newaxis,:]),axis=0), axis=0)
            elif f.name == 'Leverage':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData, 
                            useTotalAssets=True)
                values = ma.where(values < 0.0, 0.0, values)   # Negative debt -> 0.0
            elif f.name == 'Growth':
                roe = StyleExposures.generate_return_on_equity(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData, 
                            legacy=self.modelHack.legacyROE)
                divPayout = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            includeStock=self.allowStockDividends)
                values = (1.0 - divPayout) * roe
            else:
                continue
            data.exposureMatrix.addFactor(
                   f.name, values, ExposureMatrix.StyleFactor)
        if self.debuggingReporting:
            data.exposureMatrix.dumpToFile('tmp/raw-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, dp=self.dplace)
        self.log.debug('generate_pd_fundamental_exposures: end')
        return data.exposureMatrix

    def generate_md_fundamental_exposures(self,
                    modelDate, data, modelDB, marketDB):
        """Compute multiple-descriptor fundamental style exposures 
        for assets in data.universe for all CompositeFactors in self.factors.
        data should be a Struct() containing the ExposureMatrix
        object as well as any required market data, like market
        caps, asset universe, etc.
        Factor exposures are computed as the equal-weighted average
        of all the normalized descriptors associated with the factor.
        """
        self.log.debug('generate_md_fundamental_exposures: begin')
        compositeFactors = [f for f in self.factors if isinstance(f, CompositeFactor)]
        if len(compositeFactors) == 0:
            self.log.warning('No CompositeFactors found!')
            return data.exposureMatrix
        descriptors = [d for f in compositeFactors for d in f.descriptors]
        descriptorExposures = Matrices.ExposureMatrix(data.universe)
        for d in descriptors:
            if d.description == 'Book-to-Price':
                values = StyleExposures.generate_book_to_price(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
#                values = ma.masked_where(values < 0.0, values)   # Mask negative BTP
            elif d.description == 'Earnings-to-Price':
                values = StyleExposures.generate_earnings_to_price(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                            legacy=self.modelHack.legacyETP)
            elif d.description == 'Sales-to-Price':
                values = StyleExposures.generate_sales_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Debt-to-Assets':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData, 
                            useTotalAssets=True)
                values = ma.where(values < 0.0, 0.0, values)   # Negative debt -> 0.0
            elif d.description == 'Debt-to-MarketCap':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Plowback times ROE':
                roe = StyleExposures.generate_return_on_equity(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
                divPayout = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            includeStock=self.allowStockDividends)
                values = (1.0 - divPayout) * roe
            elif d.description == 'Sales Growth':
                values = StyleExposures.generate_sales_growth(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, 
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Earnings Growth':
                values = StyleExposures.generate_earnings_growth(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None, 
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Dividend Yield':
                values = StyleExposures.generate_dividend_yield(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None)
            else:
                raise Exception('Undefined descriptor %s!' % d)
            descriptorExposures.addFactor(
                    d.description, values, ExposureMatrix.StyleFactor)

        # Add country factors to ExposureMatrix, needed for regional-relative standardization
        country_indices = data.exposureMatrix.\
                            getFactorIndices(ExposureMatrix.CountryFactor)
        countryExposures = ma.take(data.exposureMatrix.getMatrix(), 
                            country_indices, axis=0)
        countryNames = data.exposureMatrix.getFactorNames(ExposureMatrix.CountryFactor)
        descriptorExposures.addFactors(countryNames, countryExposures, 
                            ExposureMatrix.CountryFactor)

        if self.debuggingReporting:
            descriptorExposures.dumpToFile('tmp/raw-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, dp=self.dplace)

        # Standardize raw descriptors for multi-descriptor factors
        mat = descriptorExposures.getMatrix()
#        singleDescriptors = [d.description for f in compositeFactors for d in \
#                f.descriptors if len(f.descriptors)==1]
#        if len(singleDescriptors) < len(compositeFactors):
        origStandardization = copy.copy(self.exposureStandardization)
        self.exposureStandardization = Standardization.BucketizedStandardization(
                [Standardization.RegionRelativeScope(
                    modelDB, [d.description for d in descriptors])])
        self.standardizeExposures(descriptorExposures, data.estimationUniverseIdx, 
                        data.marketCaps, modelDate, modelDB, marketDB)

        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        for cf in compositeFactors:
            self.log.debug('CompositeFactor %s has %d descriptor(s)', 
                    cf.description, len(cf.descriptors))
            valueList = [mat[descriptorExposures.getFactorIndex(
                                d.description),:] for d in cf.descriptors]
            if len(valueList) > 1:
                e = ma.average(ma.array(valueList), axis=0)
            else:
                e = ma.array(valueList)[0,:]
            data.exposureMatrix.addFactor(cf.description, e, ExposureMatrix.StyleFactor)

        # Proxy (standardized) exposures for assets missing data
        self.proxy_missing_exposures(modelDate, data, modelDB, marketDB,
                         factorNames=[cf.description for cf in compositeFactors])
        self.exposureStandardization = origStandardization

        self.log.debug('generate_md_fundamental_exposures: end')
        return data.exposureMatrix

    def generate_md_fundamental_exposures_v2(self,
                    modelDate, data, modelDB, marketDB):
        """Compute multiple-descriptor fundamental style exposures 
        for assets in data.universe for all CompositeFactors in self.factors.
        data should be a Struct() containing the ExposureMatrix
        object as well as any required market data, like market
        caps, asset universe, etc.
        Factor exposures are computed as the equal-weighted average
        of all the normalized descriptors associated with the factor.
        """
        self.log.debug('generate_md_fundamental_exposures_v2: begin')
        compositeFactors = [f for f in self.factors if isinstance(f, CompositeFactor)]
        if len(compositeFactors) == 0:
            self.log.warning('No CompositeFactors found!')
            return data.exposureMatrix
        descriptors = [d for f in compositeFactors for d in f.descriptors]
        descriptorExposures = Matrices.ExposureMatrix(data.universe)
        for d in descriptors:
            if d.description == 'Book-to-Price':
                values = StyleExposures.generate_book_to_price(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
#                values = ma.masked_where(values < 0.0, values)   # Mask negative BTP
            elif d.description == 'Earnings-to-Price':
                values = StyleExposures.generate_earnings_to_price(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None, useQuarterlyData=self.quarterlyFundamentalData,
                            legacy=self.modelHack.legacyETP)
            elif d.description == 'Sales-to-Price':
                values = StyleExposures.generate_sales_to_price(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Debt-to-Assets':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData, 
                            useTotalAssets=True)
                values = ma.where(values < 0.0, 0.0, values)   # Negative debt -> 0.0
            elif d.description == 'Debt-to-MarketCap':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData)
            elif d.description == 'Plowback times ROE':
                startDate = MERGE_START_DATE
                endDate = MERGE_END_DATE
                if modelDate < startDate:
                    roe = StyleExposures.generate_return_on_equity(
                                modelDate, data, self, modelDB, marketDB, 
                                restrict=None,
                                useQuarterlyData=self.quarterlyFundamentalData)
                elif modelDate > endDate:
                    roe = StyleExposures.generate_return_on_equity_v2(
                                modelDate, data, self, modelDB, marketDB, 
                                restrict=None,
                                useQuarterlyData=self.quarterlyFundamentalData)
                else:
                    legacyRoe = StyleExposures.generate_return_on_equity(
                                modelDate, data, self, modelDB, marketDB, 
                                restrict=None,
                                useQuarterlyData=self.quarterlyFundamentalData)
                    roe = StyleExposures.generate_return_on_equity_v2(
                                modelDate, data, self, modelDB, marketDB, 
                                restrict=None,
                                useQuarterlyData=self.quarterlyFundamentalData)
                    roe = Utilities.blend_values(legacyRoe, roe, 
                                modelDate, startDate, endDate)
                divPayout = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, self, modelDB, marketDB, 
                            restrict=None,
                            useQuarterlyData=self.quarterlyFundamentalData,
                            includeStock=self.allowStockDividends)
                values = (1.0 - divPayout) * roe
            elif d.description == 'Sales Growth':
                startDate = MERGE_START_DATE
                endDate = MERGE_END_DATE
                if modelDate < startDate:
                    values = StyleExposures.generate_sales_growth(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData)
                elif modelDate > endDate:
                    values = StyleExposures.generate_sales_growth_v2(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData,
                                debug=self.debuggingReporting)
                else:
                    legacyValues = StyleExposures.generate_sales_growth(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData)
                    values = StyleExposures.generate_sales_growth_v2(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData,
                                debug=self.debuggingReporting)
                    values = Utilities.blend_values(legacyValues, values, 
                                modelDate, startDate, endDate)
            elif d.description == 'Earnings Growth':
                startDate = MERGE_START_DATE
                endDate = MERGE_END_DATE
                if modelDate < startDate:
                    values = StyleExposures.generate_earnings_growth(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData)
                elif modelDate > endDate:
                    values = StyleExposures.generate_earnings_growth_v2(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData,
                                debug=self.debuggingReporting)
                else:
                    legacyValues = StyleExposures.generate_earnings_growth(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData)
                    values = StyleExposures.generate_earnings_growth_v2(
                                modelDate, data, self, modelDB, marketDB,
                                restrict=None, 
                                useQuarterlyData=self.quarterlyFundamentalData,
                                debug=self.debuggingReporting)
                    values = Utilities.blend_values(legacyValues, values, 
                                modelDate, startDate, endDate)
            elif d.description == 'Dividend Yield':
                values = StyleExposures.generate_dividend_yield(
                            modelDate, data, self, modelDB, marketDB,
                            restrict=None)
            else:
                raise Exception('Undefined descriptor %s!' % d)
            descriptorExposures.addFactor(
                    d.description, values, ExposureMatrix.StyleFactor)

        # Add country factors to ExposureMatrix, needed for regional-relative standardization
        country_indices = data.exposureMatrix.\
                            getFactorIndices(ExposureMatrix.CountryFactor)
        countryExposures = ma.take(data.exposureMatrix.getMatrix(), 
                            country_indices, axis=0)
        countryNames = data.exposureMatrix.getFactorNames(ExposureMatrix.CountryFactor)
        descriptorExposures.addFactors(countryNames, countryExposures, 
                            ExposureMatrix.CountryFactor)

        if self.debuggingReporting:
            descriptorExposures.dumpToFile('tmp/raw-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, dp=self.dplace)

        # Standardize raw descriptors for multi-descriptor factors
        mat = descriptorExposures.getMatrix()
#        singleDescriptors = [d.description for f in compositeFactors for d in \
#                f.descriptors if len(f.descriptors)==1]
#        if len(singleDescriptors) < len(compositeFactors):
        origStandardization = copy.copy(self.exposureStandardization)
        self.exposureStandardization = Standardization.BucketizedStandardization(
                [Standardization.RegionRelativeScope(
                    modelDB, [d.description for d in descriptors])])
        self.standardizeExposures(descriptorExposures, data.estimationUniverseIdx, 
                        data.marketCaps, modelDate, modelDB, marketDB)

        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        for cf in compositeFactors:
            self.log.debug('CompositeFactor %s has %d descriptor(s)', 
                    cf.description, len(cf.descriptors))
            valueList = [mat[descriptorExposures.getFactorIndex(
                                d.description),:] for d in cf.descriptors]
            if len(valueList) > 1:
                e = ma.average(ma.array(valueList), axis=0)
            else:
                e = ma.array(valueList)[0,:]
            data.exposureMatrix.addFactor(cf.description, e, ExposureMatrix.StyleFactor)

        # Proxy (standardized) exposures for assets missing data
        self.proxy_missing_exposures(modelDate, data, modelDB, marketDB,
                         factorNames=[cf.description for cf in compositeFactors])
        self.exposureStandardization = origStandardization

        self.log.debug('generate_md_fundamental_exposures_v2: end')
        return data.exposureMatrix

    def proxy_missing_exposures(self, modelDate, data, modelDB, marketDB, 
                                factorNames=['Value', 'Growth', 'Leverage']):
        """Fill-in missing exposure values for the factors given in 
        factorNames by cross-sectional regression.  For each region,
        estimation universe assets are taken, and their exposure values
        regressed against their Size and industry (GICS sector) exposures.
        Missing values are then extrapolated based on these regression
        coefficients, and trimmed to lie within [-1.5, 1.5] to prevent the
        proxies taking on extreme values.
        """
        self.log.debug('proxy_missing_exposures: begin')
        expM = data.exposureMatrix

        # Prepare sector exposures, will be used for missing data proxies
        roots = self.industryClassification.getClassificationRoots(modelDB)
        root = [r for r in roots if r.name=='Sectors'][0]
        sectorNames = [n.description for n in self.\
                industryClassification.getClassificationChildren(root, modelDB)]
        sectorExposures = Matrices.ExposureMatrix(data.universe)
        e = self.industryClassification.getExposures(modelDate,
                data.universe, sectorNames, modelDB, level=-(len(roots)-1))
        sectorExposures.addFactors(sectorNames, e, ExposureMatrix.IndustryFactor)

        # Set up regression to proxy raw exposures for assets missing data
        dr = RegressionToolbox.DummyMarketReturns()
        rp = {'fixThinFactors': True,
              'dummyReturns': dr,
              'dummyWeights': RegressionToolbox.AxiomaDummyWeights(10.0),
              'whiteStdErrors': False}
        returnCalculator = ReturnCalculator.RobustRegression2(
                        RegressionToolbox.RegressionParameters(rp))

        mat = expM.getMatrix()
        for f in factorNames:
            # Determine which assetes are missing data and require proxying
            values = mat[expM.getFactorIndex(f),:]
            missingIndices = numpy.flatnonzero(ma.getmaskarray(values))
            estuSet = set(data.estimationUniverseIdx)
            self.log.info('%d/%d assets missing %s fundamental data (%d/%d ESTU)',
                        len(missingIndices), len(values), f,
                        len(estuSet.intersection(missingIndices)), len(estuSet))
            if len(missingIndices)==0:
                continue

            # Loop around regions
            for (regionName, asset_indices) in self.exposureStandardization.\
                    factorScopes[0].getAssetIndices(expM, modelDate):
                good_indices = set(asset_indices).difference(missingIndices)
                good_indices = good_indices.intersection(estuSet)
                good_indices = list(good_indices)
                missing_indices = list(set(asset_indices).intersection(missingIndices))
                if len(good_indices) <= 10:
                    if len(missing_indices) > 0:
                        self.log.warning('Too few assets (%d) in %s with %s data present', 
                                      len(good_indices), regionName, f)
                    continue

                # Assemble regressand, regressor matrix and weights
                weights = numpy.take(data.marketCaps, good_indices, axis=0)**0.5
                regressand = ma.take(values, good_indices, axis=0)
#                (regressand, mad_bounds) = Utilities.mad_dataset(
#                                ma.take(values, good_indices, axis=0), -4.2, 4.2)
                regressor = ma.zeros((len(good_indices), len(sectorNames) + 1))
                regressor[:,0] = ma.take(mat[expM.getFactorIndex(
                                'Size'),:], good_indices, axis=0)
                regressor[:,1:] = ma.transpose(ma.take(
                        sectorExposures.getMatrix(), good_indices, axis=1).filled(0.0))
                regressor = ma.transpose(regressor)

                # Set up thin factor correction
                tfc = RegressionToolbox.DummyAssetHandler(
                            list(range(regressor.shape[0])), ['Size'] + sectorNames,
                            returnCalculator.parameters)
                tfc.nonzeroExposuresIdx = [0]
                dr = returnCalculator.parameters.getDummyReturns()
                dr.factorIndices = tfc.idxToCheck
                dr.factorNames = tfc.factorNames
                rets = dr.computeReturns(ma.array(regressand), list(range(len(weights))), weights,
                            [data.universe[j] for j in good_indices], modelDate)
                tfc.setDummyReturns(rets)
                tfc.setDummyReturnWeights(dr.dummyRetWeights)
                returnCalculator.thinFactorAdjustment = tfc

                # Run regression to get proxy values
                self.log.info('Running %s proxy regression for %s (%d assets)', 
                        f, regionName, len(good_indices))
                (t0, coefs, t1, t2, t3, t4, t5, t6) = returnCalculator.calc_Factor_Specific_Returns(
                        self, list(range(regressor.shape[1])), regressand, regressor,
                        ['Size'] + sectorNames, weights, None,
                        returnCalculator.parameters.getFactorConstraints(),
                        robustRegression=self.useRobustRegression)

                # Substitute proxies for missing values
                self.log.info('Proxying %d %s exposures for %s', 
                        len(missing_indices), f, regionName)
                for i in range(len(sectorNames)):
                    regSectorExposures = sectorExposures.getMatrix()[i,missing_indices]
                    reg_sec_indices = numpy.flatnonzero(ma.getmaskarray(regSectorExposures)==0)
                    if len(reg_sec_indices)==0:
                        continue
                    reg_sec_indices = [missing_indices[j] for j in reg_sec_indices]
                    sectorCoef = coefs[i+1]
                    sizeExposures = mat[expM.getFactorIndex(
                                    'Size'), reg_sec_indices]
                    proxies = sizeExposures * coefs[0] + sectorCoef
                    proxies = ma.where(proxies < -2.0, -2.0, proxies)
                    proxies = ma.where(proxies > 2.0, 2.0, proxies)
                    mat[expM.getFactorIndex(f),reg_sec_indices] = proxies

#        assert(numpy.sum(ma.getmaskarray(ma.take(expM.getMatrix(), 
#            [expM.getFactorIndex(f) for f in factorNames], axis=0)))==0)
        self.log.debug('proxy_missing_exposures: begin')
        return expM

    def generate_fundamental_exposures(self, 
                    modelDate, data, modelDB, marketDB):
        """Compute the exposures for assets in data.universe using
        their sensitivities to the factor returns associated with 
        Value, Growth, and Leverage.  Fama-French type style analysis 
        is used.  Fundamental data is not required for all assets,
        and no assets should have masked/missing exposures.
        Only assets in data.estimationUniverse are used to compute
        the Fama-French factor returns.
        Value is computed using book-to-price and earnings-to-price
        factor returns, Leverage uses total debt over total assets,
        and Growth remains unchanged from the single-country models.
        """
        self.log.debug('generate_fundamental_exposures: begin')
        
        if hasattr(self, 'famaFrenchizer'):
            restrictIdx = data.estimationUniverseIdx
        else:
            restrictIdx = None
        # Load book-to-price (Value)
        btp = StyleExposures.generate_book_to_price(
                    modelDate, data, self, modelDB, marketDB, 
                    restrict=restrictIdx,
                    useQuarterlyData=self.quarterlyFundamentalData)
        btp = ma.masked_where(btp < 0.0, btp)   # Mask negative BTP
        
        # Load earnings-to-price (Value)
        etp = StyleExposures.generate_earnings_to_price(
                    modelDate, data, self, modelDB, marketDB, 
                    restrict=restrictIdx,
                    useQuarterlyData=self.quarterlyFundamentalData, 
                    legacy=self.modelHack.legacyETP)
        
        # Load debt-to-assets (Leverage)
        lev = StyleExposures.generate_debt_to_marketcap(
                    modelDate, data, self, modelDB, marketDB, 
                    restrict=restrictIdx,
                    useQuarterlyData=self.quarterlyFundamentalData, 
                    useTotalAssets=True)
        lev = ma.where(lev < 0.0, 0.0, lev)   # Negative debt -> 0.0
        
        # Growth
        roe = StyleExposures.generate_return_on_equity(
                    modelDate, data, self, modelDB, marketDB, 
                    restrict=restrictIdx,
                    useQuarterlyData=self.quarterlyFundamentalData, 
                    legacy=self.modelHack.legacyROE)
        divPayout = StyleExposures.generate_proxied_dividend_payout(
                    modelDate, data, self, modelDB, marketDB, 
                    restrict=restrictIdx,
                    useQuarterlyData=self.quarterlyFundamentalData,
                    includeStock=self.allowStockDividends)
        gro = (1.0 - divPayout) * roe
        
        if not hasattr(self, 'famaFrenchizer'):
            val = ma.average(ma.concatenate((btp[numpy.newaxis,:],\
                    etp[numpy.newaxis,:]),axis=0), axis=0)
            data.exposureMatrix.addFactor('Value', val, ExposureMatrix.StyleFactor)
            data.exposureMatrix.addFactor('Growth', gro, ExposureMatrix.StyleFactor)
            data.exposureMatrix.addFactor('Leverage', lev, ExposureMatrix.StyleFactor)
            self.log.debug('generate_fundamental_exposures: end')
            return data.exposureMatrix

        # Determine how many calendar days of data we need
        dateList = modelDB.getDates(self.rmg, modelDate, 365)
        nonWeekDays = len([d for d in dateList if d.weekday() > 4])
        goodRatio = 1.0 - float(nonWeekDays) / 365
        needDays = int(self.famaFrenchizer.daysBack / goodRatio)
        if needDays > len(data.returns.dates):
            self.log.warning('Requested %d calendar days of returns, only %d available',
                    needDays, len(data.returns.dates))
            needDays = len(data.returns.dates) - 1

        # Compute period returns (fetching cumulative returns from DB is slow)
        dateList = [d for d in data.returns.dates[-needDays-1:] \
                    if d.weekday() <= 4]
        if self.famaFrenchizer.regressionFrequency == \
                        self.famaFrenchizer.WeeklyRegression:
            period_dates = [prev for (prev, next) in \
                            zip(dateList[:-1], dateList[1:])
                            if next.weekday() < prev.weekday()]
        elif self.famaFrenchizer.regressionFrequency == \
                        self.famaFrenchizer.MonthlyRegression:
            period_dates = [prev for (prev, next) in \
                            zip(dateList[:-1], dateList[1:])
                            if next.month > prev.month or next.year > prev.year]
        if self.famaFrenchizer.regressionFrequency != \
                        self.famaFrenchizer.DailyRegression:
            period_dates = [d for d in period_dates if d > datetime.date(1996,1,1)]
            logging.info('Using %d period dates from %s to %s', 
                    len(period_dates), str(period_dates[0]), str(period_dates[-1]))
            datesIdxMap = dict([(d,i) for (i,d) in enumerate(data.returns.dates[-needDays-1:])])
            period_dates_idx = [datesIdxMap[d] for d in period_dates]
            cumulativeReturns = numpy.cumproduct(
                    data.returns.data[:,-needDays-1:].filled(0.0) + 1.0, axis=1)
            periodReturns = Matrices.TimeSeriesMatrix(
                                data.returns.assets, period_dates[1:])
            cumulativeReturns = numpy.take(cumulativeReturns, period_dates_idx, axis=1)
            periodReturns.data = ma.array(
                                cumulativeReturns[:,1:] / cumulativeReturns[:,:-1] - 1.0)
        else:
            periodReturns = Matrices.TimeSeriesMatrix(
                    data.returns.assets, data.returns.dates[-needDays-1:])
            periodReturns.data = ma.array(data.returns.data[:,-needDays-1:])
        
        # Invoke Fama-French factor computation
        factorDescList = [[btp, etp], [gro], [lev]]
        factorNames = ['Value', 'Growth', 'Leverage']
        data.exposureMatrix = self.famaFrenchizer.getExposures(
                    factorNames, factorDescList, data, modelDate, 
                    periodReturns, self, modelDB, marketDB)
        
        self.log.debug('generate_fundamental_exposures: end')
        return data.exposureMatrix

    def generateExposureMatrix(self, modelDate, modelDB, marketDB):
        """Generates the exposure matrix for the given date.
        Return a Struct containing the exposure matrix
        (exposureMatrix), the universe as a list of assets (universe),
        and a list of market capitalizations (marketCaps).
        """
        self.log.debug('generateExposureMatrix: begin')
        
        # Get risk model universe and market caps
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        data = Utilities.Struct()
        data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
        (mcapDates, goodRatio) = self.getRMDates(
                modelDate, modelDB, 20, ceiling=False)
        data.marketCaps = modelDB.getAverageMarketCaps(
                    mcapDates, data.universe, self.numeraire.currency_id, marketDB)
        if self.debuggingReporting:
            self.log.info('Universe total mcap: %s Bn',
                    ma.sum(data.marketCaps, axis=None) / 1.0e9)
        if not self.forceRun:
            assert(numpy.sum(ma.getmaskarray(data.marketCaps))==0)
        data.marketCaps = numpy.array(data.marketCaps)
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)

        # Pick up dict of assets to be cloned from others
        data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)
        
        # Build rmgAssetMap and flag DR-like assets
        (univ, mcaps, ids_DR) = self.process_asset_country_assignments(
                    modelDate, data.universe, data.marketCaps,
                    modelDB, marketDB)
        
        # Fetch trading calendars for all risk model groups
        # Start-date should depend on how long a history is required
        # for exposures computation 
        data.rmgCalendarMap = dict()
        startDate = modelDate - datetime.timedelta(365*2)
        for rmg in self.rmg:
            data.rmgCalendarMap[rmg.rmg_id] = \
                    modelDB.getDateRange(rmg, startDate, modelDate)
        
        # Compute issuer-level market caps if required
        if self.modelHack.issuerMarketCaps:
            data.issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
                        data, modelDate, self.numeraire, modelDB, marketDB,
                        debugReport=self.debuggingReporting)
            data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()
        else:
            data.issuerMarketCaps = data.marketCaps.copy()
            data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()

        # Generate country and currency exposures
        if self.countryBetas:
            # Note: this won't work, as returns haven't yet been loaded
            assert(False)
            data.exposureMatrix = GlobalExposures.generate_beta_country_exposures(
                    modelDate, self, modelDB, marketDB, data)
        else:
            data.exposureMatrix = GlobalExposures.generate_binary_country_exposures(
                    modelDate, self, modelDB, marketDB, data)
        data.exposureMatrix = GlobalExposures.generate_currency_exposures(
                modelDate, self, modelDB, marketDB, data)
        
        # Generate 0/1 industry exposures
        data.exposureMatrix = self.generate_industry_exposures(
            modelDate, modelDB, marketDB, data.exposureMatrix)
        
        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB)
        estu = [n for n in estu if n in data.assetIdxMap]
        assert(len(estu) > 0)
        data.estimationUniverseIdx = [data.assetIdxMap[n] for n in estu]
        
        # Generate market exposures
        dr_indices = [data.assetIdxMap[n] for n in ids_DR]
        data.exposureMatrix = self.generate_market_exposures(
            modelDate, data, dr_indices, modelDB, marketDB)
        
        # Generate other, model-specific factor exposures
        data.exposureMatrix = self.generate_model_specific_exposures(
            modelDate, data, modelDB, marketDB)
        
        # Create intercept factor
        if self.intercept is not None:
            if not self.simpleIntercept:
                retDays = int(120 / goodRatio)
            else:
                retDays = 0
            data.exposureMatrix.addFactor(self.intercept.name, 
                StyleExposures.generate_world_sensitivity(data.returns, 
                data.marketCaps, periodsBack=retDays, 
                indices=data.estimationUniverseIdx, simple=self.simpleIntercept), 
                ExposureMatrix.InterceptFactor)
        
        if self.debuggingReporting:
            subIssueGroups = modelDB.getIssueCompanyGroups(modelDate, data.universe, marketDB)
            data.exposureMatrix.dumpToFile('tmp/raw-expM-preClone-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=subIssueGroups, dp=self.dplace)

        # Clone DR and cross-listing exposures if required
        subIssueGroups = None
        if self.modelHack.specialDRTreatment:
            if self.modelHack.ahSharesISC:
                subIssueGroups = modelDB.getIssueCompanyGroups(
                    modelDate, data.universe, marketDB)
            else:
                excludeList = ['AShares','BShares']
                self.log.info('Do not apply ISC exposure cloning on %s'%(\
                        ','.join('%s'%k for k in excludeList)))
                subIssueGroups = modelDB.getSpecificIssueCompanyGroups(
                    modelDate, data.universe, marketDB, excludeList)
                
            scores = self.score_linked_assets(
                    modelDate, data.universe, modelDB, marketDB,
                    subIssueGroups=subIssueGroups)
            data.exposureMatrix = self.clone_linked_asset_exposures(
                    modelDate, data, modelDB, marketDB, scores,
                    subIssueGroups=subIssueGroups)

        # Normalize style exposures using MAD
        self.standardizeExposures(data.exposureMatrix, data.estimationUniverseIdx, 
                data.marketCaps, modelDate, modelDB, marketDB, dr_indices, subIssueGroups)
        
        self.log.debug('generateExposureMatrix: end')
        return data
    
    def fill_in_missing_returns(self, modelDate, returns, data, modelDB):
        """Same as proxyMissingAssetReturns(), but does not
        proxy observations corresponding to non-trading days 
        in each asset's respective market.  ie these records
        remain masked.
        """
        self.log.debug('fill_in_missing_returns: begin')
        # Replace missing values with proxies
        tmpReturns = ma.array(returns.data, copy=True)
        nonMissingArray = ma.getmaskarray(returns.data)==0
        (returns, buckets) = self.proxyMissingAssetReturns(
                                modelDate, returns, data, modelDB)
        # Mask records for genuine non-trading days
        newReturns = Matrices.allMasked(returns.data.shape)
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(returns.dates)])
        for rmg in self.rmg:
            asset_indices = [data.assetIdxMap[sid] for sid in \
                             self.rmgAssetMap[rmg.rmg_id]]
            rmgCalendarMap = data.rmgCalendarMap[rmg.rmg_id]

            if len(asset_indices) > 0:
                date_indices = [dateIdxMap[d] for d in rmgCalendarMap \
                        if d in dateIdxMap and d >= returns.dates[0]]
                ret = ma.take(ma.take(returns.data, asset_indices, axis=0), 
                            date_indices, axis=1)
                positions = [len(returns.dates) * sIdx + dIdx for \
                            sIdx in asset_indices for dIdx in date_indices]
                ma.put(newReturns, positions, ret)

        # Put back any genuine returns on non-trading days
        if self.fixRMReturns:
            newMaskedArray = ma.getmaskarray(newReturns)
            genuineReturnArray = nonMissingArray * newMaskedArray
            tmpReturns = ma.masked_where(genuineReturnArray==0, tmpReturns)
            newReturns = ma.filled(newReturns, 0.0) + ma.filled(tmpReturns, 0.0)
            genuineMissingArray = newMaskedArray * (nonMissingArray==0)
            newReturns = ma.masked_where(genuineMissingArray, newReturns)
        
        returns.data = newReturns
        reMasked = numpy.sum(ma.getmaskarray(returns.data), axis=None)
        self.log.debug('%d observations have been re-masked (%.2f %%)',
                       reMasked, 100.0 * reMasked
                       / (returns.data.shape[1] * returns.data.shape[0]))
        self.log.debug('fill_in_missing_returns: end')
        return returns

    def generate_market_exposures(self, modelDate, data, dr_indices,
                                  modelDB, marketDB):
        """Compute exposures for market factors for the assets
        in data.universe and add them to data.exposureMatrix,
        returning the resulting ExposureMatrix object.
        Additional processing logic ensures that exposures based
        on time-series market data account for markets which may trade
        on irregular days, such as weekends.
        dr_indices specifies index positions of DR-like instruments, 
        for which returns-based exposures should be computed in 
        'home' country currency.
        """
        self.log.debug('generate_market_exposures: begin')
        expM = data.exposureMatrix
        
        # Compute Size factor
        expM.addFactor('Size',
            StyleExposures.generate_size_exposures(data), ExposureMatrix.StyleFactor)
        
        # Determine how many calendar days of data we need
        dateList = modelDB.getDates(self.rmg, modelDate, 365)
        nonWeekDays = len([d for d in dateList if d.weekday() > 4])
        goodRatio = 1.0 - float(nonWeekDays) / 365
        needDays = int(self.returnHistory / goodRatio)
        oneYearNeedDays = int(250.0 / goodRatio)
        
        # Uncomment for testing
#        needDays = int(270 / goodRatio)
#        returns = Utilities.Struct()
#        returns.data = numpy.random.randn(len(expM.getAssets()), needDays)
#        returns.data = ma.array(returns.data)
#        returns.dates = dateList[-needDays:]
#        returns.assets = expM.getAssets()

        # Load DR information
        if len(dr_indices) > 0:
            (drCurrData, currCodeIDMap) = GlobalExposures.\
                getAssetHomeCurrencyID(data.universe, dr_indices, 
                                       expM, modelDate, modelDB)
        else:
            drCurrData = None

        # Get daily returns for the past calendar days(s)
        returns = modelDB.loadTotalReturnsHistory(
                self.rmg, modelDate, data.universe, needDays, drCurrData)
        if returns.data.shape[1] < needDays:
            raise LookupError('Not enough previous trading days (need %d, got %d)' % \
                            (needDays, returns.data.shape[1]))
        else:
            self.log.debug('Loaded %d days of returns (from %s) for exposures',
                           needDays + 1, returns.dates[0])
        
        # Back-compatibility point
        if self.modelHack.MADRetsForStyles:
            # Clip extreme returns
            (returns.data, bounds) = Utilities.mad_dataset(
                    returns.data, self.xrBnds[0], self.xrBnds[1], axis=0,
                    zero_tolerance=0.25, treat=self.modelHack.MADRetsTreat)
            # Fill in missing returns with proxy values
            returns = self.fill_in_missing_returns(
                    modelDate, returns, data, modelDB)
        else:
            # Mask non-trading entries with zero
            returns.data = returns.data.filled(0.0)
            returns.data = numpy.clip(returns.data, -0.75, 2.0)
       
        # Aggressively clip extreme ADR home currency returns
        if len(dr_indices) > 0 and self.modelHack.DRClipping:
            drReturns = ma.take(returns.data, dr_indices, axis=0)
            (drReturnsClipped, b) = Utilities.mad_dataset(drReturns, 
                                        -5.0, 5.0, axis=0, treat='clip')
            returns.data[dr_indices,:] = drReturnsClipped

        # Compute Volatility exposures; cannot compute it per market
        # b/c cross sectional std dev has to be taken across entire ESTU
        volDays = int(60 / goodRatio)
        volatility = StyleExposures.generate_cross_sectional_volatility(
                returns, indices=data.estimationUniverseIdx, daysBack=volDays)
        
        # Create market exposures for each market
        stm = Matrices.allMasked(len(data.universe))
        mtm = Matrices.allMasked(len(data.universe))
        liq = Matrices.allMasked(len(data.universe))
        tmpReturnsArray = returns.data[:,-oneYearNeedDays-1:]
        tmpReturnsDates = returns.dates[-oneYearNeedDays-1:]
        tmpDatesIdxMap = dict(zip(tmpReturnsDates, range(len(tmpReturnsDates))))

        for rmg in self.rmg:
            self.log.debug('Computing market style exposures for %s', rmg.description)
            # Determine list of assets in market
            rmg_assets = list(self.rmgAssetMap[rmg.rmg_id])
            asset_indices = [data.assetIdxMap[n] for n in rmg_assets]
            
            # Determine relevant trading dates for this market
            rmgCalendarMap = data.rmgCalendarMap[rmg.rmg_id]
            rmgDates = sorted(d for d in rmgCalendarMap if d in tmpDatesIdxMap)
            if len(rmgDates) < 100:
                self.log.warning('Very few trading days for %s, %d records from %s to %s', 
                        rmg.description, len(rmgDates), tmpReturnsDates[0], modelDate)
                rmgDates = [d for d in tmpReturnsDates if d.weekday() <= 4]
            dt_indices = [tmpDatesIdxMap[d] for d in rmgDates]
            
            # Take subset of returns
            rmgReturns = Matrices.TimeSeriesMatrix(rmg_assets, rmgDates)
            ret = ma.take(returns.data, asset_indices, axis=0)
            if self.fixRMReturns:
                rmgReturns.data = Utilities.compute_compound_returns_v3(\
                        ret, tmpReturnsDates, rmgDates, matchDates=True)[0]
            else:
                rmgReturns.data = ma.take(ret, dt_indices, axis=1)

            #if self.debuggingReporting:
            #    sidList = [sid.getSubIDString() for sid in rmg_assets]
            #    dtList = [str(d) for d in rmgDates]
            #    outfile = 'tmp/rets_%s.csv' % rmg.mnemonic
            #    Utilities.writeToCSV(rmgReturns.data, outfile, rowNames=sidList, columnNames=dtList)

            rmgData = Utilities.Struct()
            rmgData.universe = rmg_assets
            rmgData.marketCaps = numpy.take(data.marketCaps, asset_indices, axis=0)
            
            if len(asset_indices) > 0:
                ma.put(stm, asset_indices, StyleExposures.
                        generate_short_term_momentum(rmgReturns))
                ma.put(mtm, asset_indices, StyleExposures.
                        generate_medium_term_momentum(rmgReturns))
                ma.put(liq, asset_indices, StyleExposures.
                        generate_trading_volume_exposures(
                        modelDate, rmgData, [rmg], modelDB, 
                        self.numeraire.currency_id, daysBack=20,
                        legacy=self.modelHack.legacyTradingVolume))
                
                # Fudge volatility exposures to deal with trading calendar
                numTradingDays = len([d for d in rmgDates \
                                    if d >= returns.dates[-volDays]])
                numTradingDays = max(numTradingDays, 50)
                rmgVol = ma.take(volatility, asset_indices, axis=0)
                rmgVol = rmgVol * (float(volDays) / float(numTradingDays))**0.5
                ma.put(volatility, asset_indices, rmgVol)
                
        # Add factor exposures to exposureMatrix
        for (fname, values) in [('Short-Term Momentum', stm), 
                                ('Medium-Term Momentum', mtm), 
                                ('Volatility', volatility), 
                                ('Liquidity', liq)]:
            expM.addFactor(fname, values, ExposureMatrix.StyleFactor)
        
        # Finally, compute Exchange Rate Sensitivity factor
        retDays = int(120 / goodRatio)
        if self.modelHack.weeklyXRTFactor:
            self.log.info('Using new XRT Factor')
            expM.addFactor('Exchange Rate Sensitivity',
                    StyleExposures.generate_forex_sensitivity_v2(
                        returns, self, modelDB, retDays,
                        numeraire=self.sensitivityNumeraire),
                    ExposureMatrix.StyleFactor)
        else:
            expM.addFactor('Exchange Rate Sensitivity',
                    StyleExposures.generate_forex_sensitivity(
                        returns, self, modelDB, retDays,
                        numeraire=self.sensitivityNumeraire),
                    ExposureMatrix.StyleFactor)
        
        data.returns = returns
        self.log.debug('generate_market_exposures: end')
        return expM
    
    def generate_estimation_universe(self, modelDate, exposureData, 
                                     modelDB, marketDB, exclude=None):
        """Generic estimation universe selection criteria for
        regional models.  Excludes assets:
            - explicitly included in RMS_ESTU_EXCLUDED
            - smaller than a certain cap threshold, per country
            - smaller than a certain cap threshold, per industry
            - from non-investible market segments (eg. China A)
        """
        self.log.debug('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                        exposureData.exposureMatrix.getAssets(), self, modelDB, marketDB)
        
        # Downweight some countries if required
        assetIdxMap = dict(zip(exposureData.universe, range(len(exposureData.universe))))
        if not hasattr(self, 'rmgAssetMap'):
            self.rmgAssetMap = self.loadRiskModelGroupAssetMap(
                    modelDate, exposureData.universe, modelDB, marketDB, False)
        for r in [r for r in self.rmg if r.estuDownWeight < 1.0]:
            logging.info('Downweighting %s market caps by %.2f%%',
                    r.mnemonic, r.estuDownWeight*100)
            for sid in self.rmgAssetMap[r.rmg_id]:
                if sid in assetIdxMap:
                    exposureData.marketCaps[assetIdxMap[sid]] *= r.estuDownWeight

        # Remove assets from the exclusion table
        logging.info('Dropping assets in the exclusion table')
        (estu0, nonest0) = buildEstu.apply_exclusion_list(modelDate)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)
        
        # Remove some specific asset types
        logging.info('Excluding particular asset types')
        (estu0, nonest) = self.excludeAssetTypes(
                            modelDate, exposureData, modelDB, marketDB, buildEstu, estu0)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Remove foreign issuers and other DR-like instruments
        logging.info('Excluding foreign listings')
        if exclude is not None:
            (estu0, nonest) = buildEstu.exclude_specific_assets(exclude, baseEstu=estu0)
            self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Remove various types of DRs and foreign listings
        newDRCode = False
        if newDRCode:
            if not hasattr(exposureData, 'assetTypeDict'):
                exposureData.assetTypeDict = AssetProcessor.get_asset_info(\
                        modelDate, exposureData.universe, modelDB, marketDB,
                        'ASSET TYPES', 'Axioma Asset Type')
            drAssetTypes = ['NVDR', 'GlobalDR', 'TDR', 'AmerDR', 'CDI', 'DR']
            logging.info('Exclude DR asset types %s', ','.join(drAssetTypes))
            n = len(estu0)
            (estu0, nonest) = buildEstu.exclude_by_asset_type(
                    modelDate, exposureData, includeFields=None, excludeFields=drAssetTypes,
                    baseEstu = estu0)
            if n != len(estu0):
                logging.info('... Eligible Universe down %d and currently stands at %d stocks',
                        n-len(estu0), len(estu0))
                self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Remove cloned assets
        logging.info('Removing cloned assets')
        if len(exposureData.hardCloneMap) > 0:
            (estu0, nonest) = buildEstu.exclude_specific_assets(
                    exposureData.hardCloneMap, baseEstu=estu0)
            self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)
        
        # Remove A-Shares; manually add H-shares and Red-Chips back
        # into list of eligible assets even though they are treated as DRs
        logging.info('Removing A-shares')
        (aShares, bShares, hShares, other) = MarketIndex.\
            process_china_share_classes(exposureData, modelDate, 
                                        modelDB, marketDB)
        estu0 = set(estu0).union(set(hShares).difference(nonest0))
        estu0 = estu0.difference(aShares)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)
        
        # Deal with Thailand foreign shares and NVDR mess
        logging.info('Processing Thai shares')
        exclIndices = MarketIndex.process_southeast_asia_share_classes(
                    exposureData, modelDate, modelDB, marketDB, exclude)
        estu0 = estu0.difference(exclIndices)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)
        
        # Logic to selectively filter RTS/MICEX Russian stocks
        logging.info('Processing Russian shares')
        rts = MarketIndex.process_russian_exchanges(
                exposureData, modelDate, modelDB, marketDB)
        estu0 = estu0.difference(rts)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Get rid of some exchanges we don't want
        exchangeCodes = ['REG','TKS-ETF'] + AssetProcessor.connectExchanges
        if modelDate.year >= 2008 and self.modelHack.enhancedESTUFiltering:
            exchangeCodes.extend([
                          'OTC','NMS','USU','PSE',              # US junk
                          'XSQ','OFE','XLF',                    # GB junk
                          'FKA-MSC','FKA-Q','FKA-S1',           # JP (Fukuoka)
                          'SAP-MSC','SAP-AMB','SAP-S1',         # JP (Sapporo)
                          'NGO-MSC','NGO-C','NGO-S1','NGO-S2',  # JP (Nagoya)
                          'NII',                                # JP (Nigata)
                          ])
        logging.info('Excluding from exchanges: %s', ','.join(exchangeCodes))
        (otc, tmp) = buildEstu.exclude_by_market_classification(
                modelDate, 'Market', 'REGIONS', exchangeCodes,
                baseEstu=list(estu0), keepMissing=False)
        estu0 = estu0.difference(otc)
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Get rid of some unwanted asset types
        if self.modelHack.enhancedESTUFiltering:
            more_dr = GlobalExposures.identify_depository_receipts(
                    exposureData, modelDate, modelDB, marketDB, 
                    restrictCls=['TQA FTID Global Asset Type'])
            DSexcludeCodesList = ['ADR','GDR','ET','ETF']
            FTIDexcludeCodesList = ['A','F']
        else:
            DSexcludeCodesList = ['ET','ETF']
            FTIDexcludeCodesList = ['F']
            more_dr = list()
        logging.info('Excluding types: %s', ','.join(DSexcludeCodesList))
        (dr0, tmp) = buildEstu.exclude_by_market_classification(
                modelDate, 'DataStream2 Asset Type',
                'ASSET TYPES', DSexcludeCodesList, keepMissing=False)
        logging.info('Excluding FTI Codes: %s', ','.join(FTIDexcludeCodesList))
        (dr1, tmp) = buildEstu.exclude_by_market_classification(
                modelDate, 'TQA FTID Domestic Asset Type', 
                'ASSET TYPES', FTIDexcludeCodesList, keepMissing=False)
        more_dr += dr0 + dr1
        estu0 = list(estu0.difference(more_dr))
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu0)

        # Weed out tiny-cap assets
        logging.info('Filtering by cap across country')
        (estu1, nonest) = buildEstu.exclude_by_cap_ranking(
                exposureData, modelDate, baseEstu=estu0, byFactorType=ExposureMatrix.CountryFactor,
                lower_pctile=5, method='percentage')
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu1)
        
        # Perform similar check by industry
        logging.info('Filtering by cap across industry')
        (estu2, nonest) = buildEstu.exclude_by_cap_ranking(
                exposureData, modelDate, baseEstu=estu0, byFactorType=ExposureMatrix.IndustryFactor,
                lower_pctile=5, method='percentage')
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu2)
        
        logging.info('Combining country and industry pools')
        estu3 = list(set(estu2).union(estu1))
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu3)
        
        # Inflate any thin countries or industries
        logging.info('Filtering by cap across country')
        (estu, nonest) = buildEstu.pump_up_factors(exposureData, modelDate,
                currentEstu=estu3, baseEstu=estu0,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor])
        self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)

        # Back-compatibility point
        if self.modelHack.grandfatherEstu:
            logging.info('Incorporating grandfathering')
            # Apply grandfathering rules
            n = len(estu)
            (estu, exposureData.ESTUQualify, nonest) = buildEstu.grandfather(
                    modelDate, estu, baseEstu=estu0, addDays=self.grandfatherParameters[0],
                    remDays=self.grandfatherParameters[1])
            self.checkTrackedAssets(exposureData.universe, assetIdxMap, estu)
            if n != len(estu):
                diff = len(estu) - n
                logging.info('%d assets resurrected by grandfathering', diff)
                logging.info('ESTU currently stands at %d stocks', len(estu))
                
        mcap_ESTU = ma.take(exposureData.marketCaps, estu, axis=0)
        if self.debuggingReporting:
            # Output information on assets and their characteristics
            assetType = AssetProcessor.get_asset_info(
                    modelDate, exposureData.universe, modelDB, marketDB,
                    'ASSET TYPES', 'Axioma Asset Type')
            sidList = [exposureData.universe[idx].getSubIDString()  for idx in estu]
            # Output weights
            outName = 'tmp/estuWt-%s-%s.csv' % (self.mnemonic, modelDate)
            Utilities.writeToCSV(mcap_ESTU, outName, rowNames=sidList)
            # Output type of each asset
            typeList = [assetType.get(exposureData.universe[idx], None) for idx in estu]
            typeDict = dict(zip(sidList, typeList))
            outName = 'tmp/estuTypes-%s-%s.csv' % (self.mnemonic, modelDate)
            AssetProcessor.dumpAssetListForDebugging(outName, sidList, typeList)
            # Output list of available types in the estu
            typeList = list(set(typeList))
            countList = []
            for atype in typeList:
                nType = len([sid for sid in sidList if typeDict[sid] == atype])
                countList.append(nType)
            outName = 'tmp/AssetTypes-ESTU-%s.csv' % self.mnemonic
            AssetProcessor.dumpAssetListForDebugging(outName, typeList, countList)
            # Output asset qualify flag
            qualList = []
            for idx in estu:
                if idx in exposureData.ESTUQualify:
                    qualList.append(1)
                else:
                    qualList.append(0)
            outName = 'tmp/estuQualify-%s-%s.csv' % (self.mnemonic, modelDate)
            AssetProcessor.dumpAssetListForDebugging(outName, sidList, qualList)

        totalcap = ma.sum(mcap_ESTU, axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',
                      len(estu), totalcap, self.numeraire.currency_code)
        self.log.debug('generate_estimation_universe: end')
        return estu
    
    def generateFactorSpecificRisk(self, date, modelDB, marketDB):
        """Compute the factor-factor covariance matrix and the
        specific variances for the risk model on the given date.
        Specific risk is computed for all assets in the exposure universe.
        The risk model instance for this day should already exist.
        Returns a Struct with:
         - subIssues: a list of SubIssue objects
         - specificVars: an array of the specific variances corresponding
            to subIssues
         - subFactors: a list of SubFactor objects
         - factorCov: a two-dimensional array with the factor-factor
            covariances corresponding to subFactors
        """
        # get sub-factors and sub-issues active on this day
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        # Remember, factors are ordered: int, sty, ind, cty, cur
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        rmi = modelDB.getRiskModelInstance(self.rms_id, date)
        if rmi == None:
            raise LookupError('No risk model instance for %s' % str(date))
        if not rmi.has_exposures or not rmi.has_returns:
            raise LookupError(
                    'Exposures or returns missing in risk model instance for %s'
                    % str(date))
        subIssues = modelDB.getRiskModelInstanceUniverse(rmi)
        self.log.debug('building time-series matrices: begin')
        
        # Check if any member RMGs have weekend trading
        # If so, need to adjust list of dates fetched
        omegaDateList = modelDB.getDates(self.rmg, date, maxOmegaObs-1,
                                         excludeWeekend=True)
        omegaDateList.reverse()
        deltaDateList = modelDB.getDates(self.rmg, date, maxDeltaObs-1, 
                                         excludeWeekend=True)
        deltaDateList.reverse()
        # Check that enough consecutive days have returns, try to get
        # the maximum number of observations
        if len(omegaDateList) > len(deltaDateList):
            dateList = omegaDateList
        else:
            dateList = deltaDateList
            
        rmiList = modelDB.getRiskModelInstances(self.rms_id, dateList)
        okDays = [i.date == j and i.has_returns for (i,j)
                in zip(rmiList, dateList)]
        okDays.append(False)
        firstBadDay = okDays.index(False)
        dateList = dateList[:firstBadDay]
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                    '%d missing risk model instances for required days'
                    % (required - len(dateList)))
        if len(dateList) < maxOmegaObs:
            self.log.info('Using only %d of %d days of factor return history',
                          len(dateList), maxOmegaObs)
        if len(dateList) < maxDeltaObs:
            self.log.info('Using only %d of %d days of specific return history',
                          len(dateList), maxDeltaObs)
        
        # Remove dates for which many markets are non-trading
        minMarkets = int(0.5 * len(self.countries))
        datesAndMarkets = modelDB.getActiveMarketsForDates(
                                self.rmg, dateList[-1], dateList[0])
        datesAndMarkets = [(d,n) for (d,n) in datesAndMarkets \
                           if d.weekday() <= 4]
        
        # Remember, datesAndMarkets is in chron. order whereas dateList is reversed
        badDatesIdx = [len(dateList)-i-1 for (i,n) in \
                        enumerate(datesAndMarkets) if n[1] <= minMarkets]
        badDates = numpy.take(dateList, badDatesIdx)
        self.log.info('Removing %d dates with < %d markets trading: %s',
                      len(badDates), minMarkets, ','.join([str(d) for d in badDates]))
        goodDatesIdx = [i for i in range(len(dateList)) if i not in badDatesIdx]
        dateList = numpy.take(dateList, goodDatesIdx)
        
        specificReturns = modelDB.loadSpecificReturnsHistory(
                self.rms_id, subIssues, dateList[:maxDeltaObs])
        self.log.debug('building time-series matrices: begin')
        
        # Not all specific risk calculators require exposures
        if isinstance(self.specificRiskCalculator, \
                      RiskCalculator.BrilliantSpecificRisk) or \
                      isinstance(self.specificRiskCalculator, \
                      RiskCalculator.BrilliantSpecificRisk2009):
            expM = None
        else:
            expM = self.loadExposureMatrix(rmi, modelDB)
        ret = Utilities.Struct()
        ret.subIssues = subIssues
        
        # Load market caps and estimation universe
        (mcapDates, goodRatio) = self.getRMDates(
                date, modelDB, 20, ceiling=False)
        avgMarketCap = modelDB.getAverageMarketCaps(
                    mcapDates, subIssues, self.numeraire.currency_id, marketDB)
        assetIdxMap = dict(zip(subIssues, range(len(subIssues))))
        ids_ESTU = self.loadEstimationUniverse(rmi, modelDB)
        ids_ESTU = [n for n in ids_ESTU if n in assetIdxMap]
        estu = [assetIdxMap[n] for n in ids_ESTU]
        
        # Load returns timing adjustment factors if required
        nonCurrencySubFactors = [f for f in subFactors \
                if f.factor not in self.currencies]
        if self.usesReturnsTimingAdjustment():
            rmgList = modelDB.getAllRiskModelGroups()
            adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                self.returnsTimingId, rmgList,
                dateList[:max(maxOmegaObs, maxDeltaObs)])
            adjustments.data = adjustments.data.filled(0.0)
            if self.debuggingReporting:
                outData = adjustments.data
                mktNames = [r.mnemonic for r in rmgList]
                dtStr = [str(dt) for dt in adjustments.dates]
                sretFile = 'tmp/%s-RetTimAdj.csv' % self.name
                Utilities.writeToCSV(outData, sretFile, rowNames=mktNames,
                        columnNames=dtStr)
        else:
            adjustments = None

        # Get list of DR-like instruments
        subIssues = specificReturns.assets
        (tmp0, tmp1, ids_DR) = self.process_asset_country_assignments(
                date, subIssues, numpy.zeros(len(subIssues)),
                modelDB, marketDB)

        if self.debuggingReporting:
            mcap = numpy.array(avgMarketCap, copy=True)
            drFlag = numpy.zeros((len(subIssues)))
            dr_indices = [assetIdxMap[n] for n in ids_DR]
            numpy.put(drFlag, dr_indices, 1.0)
            outData = specificReturns.data.filled(0.0)
            outData = numpy.concatenate((outData, mcap[:,numpy.newaxis]), axis=1)
            outData = numpy.concatenate((outData, drFlag[:,numpy.newaxis]), axis=1)
            dateStr = [str(d) for d in dateList[:maxDeltaObs]]
            dateStr = dateStr + ['cap', 'dr']
            assetNames = [s.getModelID().getIDString() for s in subIssues]
            sretFile = 'tmp/%s-sret.csv' % self.name
            Utilities.writeToCSV(outData, sretFile, rowNames=assetNames,
                    columnNames=dateStr)

        # If non-empty, apply timing adjustment to DR specific returns
        if len(ids_DR) > 0 and not self.modelHack.specialDRTreatment:
            # Load full set of adjustment factors for every relevant market
            if adjustments == None:
                rtId = self.returnsTimingId
                rmgList = modelDB.getAllRiskModelGroups()
                if rtId == None and hasattr(self, 'specReturnTimingId'):
                    rtId = self.specReturnTimingId
                if rtId is not None:
                    adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                            rtId, rmgList, dateList)
                    adjustments.data = adjustments.data.filled(0.0)

            if adjustments is not None:
                specificReturns = self.adjustDRSpecificReturnsForTiming(
                        date, specificReturns, modelDB, marketDB,
                        adjustments, ids_DR, rmgList)

        # Pick up dict of assets to be cloned from others
        hardCloneMap = modelDB.getClonedMap(date, subIssues, cloningOn=self.hardCloning)

        if self.debuggingReporting:
            outData = specificReturns.data.filled(0.0)
            outData = numpy.concatenate((outData, mcap[:,numpy.newaxis]), axis=1)
            outData = numpy.concatenate((outData, drFlag[:,numpy.newaxis]), axis=1)
            sretFile = 'tmp/%s-sretAdj.csv' % self.name
            Utilities.writeToCSV(outData, sretFile, rowNames=subIssues,
                    columnNames=dateStr)

        # Compute specific risk
        ret.specificCov = dict()
        if self.modelHack.specRiskEstuFillIn:
            srEstuFillIn = estu
        else:
            srEstuFillIn = None
        if isinstance(self.specificRiskCalculator, RiskCalculator.SparseSpecificRisk2010):
            # ISC info
            exclList = list()
            subIssueGroups = modelDB.getIssueCompanyGroups(date, subIssues, marketDB)
            if not self.modelHack.ahSharesISC:
                #For China A/B shares, exclude them from the co-integration treatment
                excludeList = ['AShares','BShares']
                self.log.info('Do not apply ISC co-integration specific risk on %s'%(\
                        ','.join('%s'%s for s in excludeList)))
                clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
                assert(clsFamily is not None)
                clsMembers = dict([(i.name, i) for i in marketDB.\
                                       getClassificationFamilyMembers(clsFamily)])
                cm = clsMembers.get('Axioma Asset Type')
                assert(cm is not None)
                clsRevision = marketDB.\
                    getClassificationMemberRevision(cm, date)
                homeClsData = modelDB.getMktAssetClassifications(
                    clsRevision, subIssues, date, marketDB)
                secTypeDict = dict([(i, j.classification.code) for (i,j) in homeClsData.items()])

                for (cmpId, sidList) in subIssueGroups.items():
                    for sid in sidList:
                        secType = secTypeDict.get(sid)
                        if secType is None:
                            self.log.error('Missing Axioma Sec Type for %s'%sid.getSubIDString())
                        elif secType in excludeList:
                            exclList.append(sid)
            logging.info('Exclusion list now contains %s assets'%len(exclList))

            specificReturns.nameDict = modelDB.getIssueNames(
                    specificReturns.dates[0], subIssues, marketDB)
            if self.modelHack.specialDRTreatment:
                self.log.info('Using NEW ISC Treatment for specific covariance')
                if not self.modelHack.ahSharesISC:
                    scoreGroups = modelDB.getSpecificIssueCompanyGroups(
                        date, subIssues, marketDB, excludeList)
                    scores = self.score_linked_assets(date, subIssues, modelDB, marketDB,
                                                      subIssueGroups=scoreGroups)
                    #Here put back the score = 0.0 for assets in excludeList, but not in score
                    for (groupId, subIssueList) in subIssueGroups.items():
                        score = scores.get(groupId)
                        if score is None:
                            newScore = numpy.zeros((len(subIssueList)), float)
                            scores[groupId] = newScore
                        else:
                            newScore = numpy.zeros((len(subIssueList)), float)
                            if len(newScore) != len(scores):
                                #There are assets got excluded and didn't get score assigned
                                idxMap = dict([(j,i) for (i,j) in enumerate(scoreGroups[groupId])])
                                for (kdx, sid) in enumerate(subIssueList):
                                    if sid in idxMap:
                                        newScore[kdx] = score[idxMap[sid]]
                                scores[groupId] = newScore
                else:
                    scoreGroups = subIssueGroups
                    scores = self.score_linked_assets(date, subIssues, modelDB, marketDB,
                                                      subIssueGroups=scoreGroups)
                
            else:
                scores = None
            (ret.specificVars, ret.specificCov) = self.specificRiskCalculator.\
                    computeSpecificRisks(specificReturns, avgMarketCap,
                                         subIssueGroups, restrict=srEstuFillIn, scores=scores,
                                         specialDRTreatment=self.modelHack.specialDRTreatment,
                                         debuggingReporting=self.debuggingReporting,
                                         coinExclList=exclList, hardCloneMap=hardCloneMap)
        else:
            ret.specificVars = self.specificRiskCalculator.\
                    computeSpecificRisks(expM, avgMarketCap, specificReturns,
                            estu=srEstuFillIn)
        self.log.debug('computed specific variances')
        
        # Factor covariance matrix next
        ret.subFactors = subFactors
        if isinstance(self.covarianceCalculator, \
                      RiskCalculator.CompositeCovarianceMatrix2009)\
            or isinstance(self.covarianceCalculator,\
                        RiskCalculator.CompositeCovarianceMatrix):
            
            # Load up non-currency factor returns
            nonCurrencyFactorReturns = modelDB.loadFactorReturnsHistory(
                    self.rms_id, nonCurrencySubFactors, dateList[:maxOmegaObs])
            
            if self.debuggingReporting:
                outData = numpy.transpose(nonCurrencyFactorReturns.data.filled(0.0))
                dateStr = [str(d) for d in dateList[:maxOmegaObs]]
                assetNames = [s.factor.name for s in nonCurrencySubFactors]
                sretFile = 'tmp/%s-fret.csv' % self.name
                Utilities.writeToCSV(outData, sretFile, columnNames=assetNames,
                        rowNames=dateStr)

            # Adjust factor returns for returns-timing, if applicable
            nonCurrencyFactorReturns = self.adjustFactorReturnsForTiming(
                    date, nonCurrencyFactorReturns, adjustments, 
                    modelDB, marketDB) 

            if self.debuggingReporting and hasattr(nonCurrencyFactorReturns, 'adjust'):
                outData = nonCurrencyFactorReturns.data.filled(0.0) + \
                        nonCurrencyFactorReturns.adjust
                outData = numpy.transpose(outData)
                sretFile = 'tmp/%s-fretAdj.csv' % self.name
                Utilities.writeToCSV(outData, sretFile, columnNames=assetNames,
                        rowNames=dateStr)

            # Pull up currency subfactors and returns
            currencySubFactors = [f for f in subFactors if f.factor in self.currencies]
            currencyFactorReturns = modelDB.loadFactorReturnsHistory(
                    self.currencyModel.rms_id, currencySubFactors,
                    dateList[:maxOmegaObs])
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, date)
            assert(crmi is not None)
            
            # Post-process the array data a bit more, then compute cov
            ret.factorCov = self.build_regional_covariance_matrix(
                    date, dateList[:maxOmegaObs],
                    currencyFactorReturns, nonCurrencyFactorReturns,
                    crmi, modelDB, marketDB)
        # Do it the old simple way, if composite cov not used
        else:
            factorReturns = modelDB.loadFactorReturnsHistory(
                    self.rms_id, subFactors, dateList[:maxOmegaObs])
            (factorReturns.data, bounds) = Utilities.mad_dataset(
                    factorReturns.data, self.xrBnds[0], self.xrBnds[1],
                    treat='clip')
            ret.factorCov = self.covarianceCalculator.\
                    computeFactorCovarianceMatrix(factorReturns)
        
        # Report day-on-day correlation matrix changes
        self.reportCorrelationMatrixChanges(date, ret.factorCov, 
                                             rmiList[1], modelDB)
        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            ncFactorNames = [f.factor.name for f in nonCurrencySubFactors]
            cFactorNames = [f.factor.name for f in currencySubFactors]
            factorNames = ncFactorNames + cFactorNames
            (d, corrMatrix) = Utilities.cov2corr(ret.factorCov, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=factorNames,
                    rowNames=factorNames)
            var = numpy.diag(ret.factorCov)[:,numpy.newaxis]
            varoutfile = 'tmp/%s-varFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(var, varoutfile, rowNames=factorNames)
            covOutFile = 'tmp/%s-covFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ret.factorCov, covOutFile, columnNames=factorNames,
                    rowNames=factorNames)

        self.log.info('Sum of composite cov matrix elements: %f', 
                    ma.sum(ret.factorCov, axis=None))
        self.log.debug('computed factor covariances')
        return ret

    def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate,
                        buildFMPs=False, internalRun=False):
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
        each factor that are stored in the array.  Currently, there
        are four statistics; the std. error, the t-statistic, 
        the Pr(>|t|), and the constraint weight used in the regression.
        The rows corresponding to the factors that are 
        not estimated via the cross-sectional regression are masked.
        The adjRsquared is a floating point number giving the 
        adjusted R-squared of the cross-sectional regression used 
        to compute the factor returns.
        """

        # Testing stuff for FMPs
        testFMPs = False
        nextTradDate = None
        crudeRets = False

        if buildFMPs:
            prevDate = modelDate
            removeNTDsFromRegression = False
            futureDates = sorted(modelDB.getDateRange(self.rmg, modelDate,
                    modelDate+datetime.timedelta(20), excludeWeekend=True))
            if testFMPs:
                nextTradDate = futureDates[1]
        elif internalRun:
            return
        else:
            # Determine previous trading date that's a weekday
            dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
            if len(dateList) < 2:
                raise LookupError(
                        'no previous trading day for %s' %  str(modelDate))
            prevDate = dateList[0]
            removeNTDsFromRegression = self.removeNTDsFromRegression
        
        # Get exposure matrix for previous trading day
        rmi = modelDB.getRiskModelInstance(self.rms_id, prevDate)
        if rmi == None:
            raise LookupError(
                'no risk model instance for %s' % str(prevDate))
        if not rmi.has_exposures:
            raise LookupError(
                'no exposures in risk model instance for %s' % str(prevDate))
        self.setFactorsForDate(prevDate, modelDB)
        expM = self.loadExposureMatrix(rmi, modelDB)
        nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in \
                    modelDB.getSubFactorsForDate(prevDate, self.factors)])
        self.setFactorsForDate(modelDate, modelDB)
        
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        subFactorIDIdxMap = dict([(subFactors[i].subFactorID, i) 
                                for i in range(len(subFactors))])
        
        # Load estimation universe and compute regression weights
        univ = expM.getAssets()
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        assetIdxMap = dict(zip(univ, range(len(univ))))
        estuRet = self.loadEstimationUniverse(rmi, modelDB)

        ids_ESTU = [n for n in estuRet if n in assetIdxMap]
        if (len(ids_ESTU) != len(estuRet)):
            self.log.warning('%d assets in estu not in model', len(estuRet)-len(ids_ESTU))
            missingID = [sid.getSubIDString() for sid in estuRet if sid not in ids_ESTU]
            self.log.debug('%s', missingID)
        estu = [assetIdxMap[n] for n in ids_ESTU]
        weights_ESTU = ReturnCalculator.calc_Weights(self.rmg, modelDB, marketDB,
                            prevDate, ids_ESTU, self.numeraire.currency_id)
        estuArray = None
        
        # Determine home country info and flag DR-like instruments
        (tmp0, tmp1, ids_DR) = self.process_asset_country_assignments(
                    modelDate, univ, numpy.zeros(len(univ)), 
                    modelDB, marketDB)

        # Downweight some countries if required
        assetIdxMap_ESTU = dict([(j,i) for (i,j) in enumerate(ids_ESTU)])
        for r in [r for r in self.rmg if r.downWeight < 1.0]:
            for sid in self.rmgAssetMap[r.rmg_id].intersection(ids_ESTU):
                weights_ESTU[assetIdxMap_ESTU[sid]] *= r.downWeight
        
        # Get info on DRs and similar items
        if len(ids_DR) > 0:
            dr_indices = [assetIdxMap[n] for n in ids_DR]
            (drCurrData, currCodeIDMap) = GlobalExposures.\
                    getAssetHomeCurrencyID(univ, dr_indices, expM,
                            prevDate, modelDB)
        else:
            drCurrData = None

        if self.debuggingReporting:
            expM.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                            % (self.name, modelDate.year, modelDate.month, modelDate.day),
                            modelDB, marketDB, modelDate, estu=estu,
                            subIssueGroups=None, dp=self.dplace)

        # Load in returns history
        if self.modelHack.specialDRTreatment:
            self.log.info('Using NEW ISC Treatment for %s returns processing', modelDate)
            (assetReturnMatrix, zeroMask, ipoMask) = self.process_returns_history(
                    modelDate, univ, 0, modelDB, marketDB, drCurrData=drCurrData)
            assetReturnMatrix.data = assetReturnMatrix.data[:,-1][:,numpy.newaxis]
            assetReturnMatrix.dates = [assetReturnMatrix.dates[-1]]
            zeroMask = zeroMask[:,-1]
        else:
            assetReturnMatrix = modelDB.loadTotalReturnsHistory(
                    self.rmg, modelDate, univ, 0, drCurrData,
                    self.numeraire.currency_id)

        if self.debuggingReporting:
            dates = [str(modelDate)]
            idList = [s.getSubIDString() for s in univ]
            retOutFile = 'tmp/%s-retHist.csv' % self.name
            Utilities.writeToCSV(assetReturnMatrix.data, retOutFile, rowNames=idList, columnNames=dates)

        if not removeNTDsFromRegression:
            missingReturnData = ma.getmaskarray(assetReturnMatrix.data[:,0])
            assetReturnMatrix.data = ma.filled(assetReturnMatrix.data, 0.0)

        # Load risk-free rates, compute excess returns
        (assetReturnMatrix, rfr) = self.computeExcessReturns(modelDate,
                assetReturnMatrix, modelDB, marketDB, drCurrData)
        for i in range(len(rfr.assets)):
            if rfr.data[i,0] is not ma.masked:
                if self.debuggingReporting:
                    self.log.info('Using risk-free rate of %f%% for %s',
                            rfr.data[i,0] * 100.0, rfr.assets[i])
                else:
                    self.log.debug('Using risk-free rate of %f%% for %s',
                            rfr.data[i,0] * 100.0, rfr.assets[i])
        excessReturns = assetReturnMatrix.data[:,0]

        # FMP Testing code
        if nextTradDate is not None:
            if crudeRets:
                futureReturnMatrix = modelDB.loadTotalReturnsHistory(
                        self.rmg, nextTradDate, univ, 0, self.numeraire.currency_id)
                futureReturnMatrix.data = ma.filled(futureReturnMatrix.data, 0.0)
            else:
                (futureReturnMatrix, dummy1, dummy2) = self.process_returns_history(
                        nextTradDate, univ, 0, modelDB, marketDB, drCurrData=drCurrData)
                futureReturnMatrix.data = futureReturnMatrix.data[:,-1][:,numpy.newaxis]
                futureReturnMatrix.dates = [futureReturnMatrix.dates[-1]]
                futureReturnMatrix.data = ma.filled(futureReturnMatrix.data, 0.0)

            (futureReturnMatrix, rfr) = self.computeExcessReturns(nextTradDate,
                    futureReturnMatrix, modelDB, marketDB, drCurrData)
            futureExcessReturns = futureReturnMatrix.data[:,0]

        # If fewer than 5% of a market's ESTU assets have returns
        # consider it non-trading and remove from regression
        nonTradingMarketsIdx = []
        for r in self.rmg:
            rmg_indices = [assetIdxMap[n] for n in self.\
                            rmgAssetMap[r.rmg_id].intersection(ids_ESTU)]
            rmg_returns = ma.take(excessReturns, rmg_indices, axis=0)
            if self.debuggingReporting:
                medReturn = ma.masked_where(abs(rmg_returns) < 1.0e-12, rmg_returns)
                medReturn = ma.median(abs(medReturn))
                self.log.info('Date: %s, RMG: %s, Median Absolute Return: %s',
                        modelDate, r.mnemonic, ma.filled(medReturn, 0.0))
            noReturns = numpy.sum(ma.getmaskarray(rmg_returns))
            allZeroReturns = ma.masked_where(abs(rmg_returns) < 1.0e-12, rmg_returns)
            allZeroReturns = numpy.sum(ma.getmaskarray(allZeroReturns))
            rmgCalendarList = modelDB.getDateRange(r,
                    assetReturnMatrix.dates[0], assetReturnMatrix.dates[-1])
            if noReturns >= 0.95 * len(rmg_returns) or modelDate not in rmgCalendarList:
                self.log.info('Non-trading day for %s, %d/%d returns missing',
                              r.description, noReturns, len(rmg_returns))
                if expM.checkFactorType(r.description, ExposureMatrix.CountryFactor):
                    fIdx = expM.getFactorIndex(r.description)
                    # Here we either do or don't drop non-trading markets
                    if removeNTDsFromRegression:
                        nonTradingMarketsIdx.append(fIdx)
                    else:
                        rmg_returns = ma.filled(rmg_returns, 0.0)
                        for (i, idx) in enumerate(rmg_indices):
                            excessReturns[idx] = rmg_returns[i]
            elif allZeroReturns == len(rmg_returns):
                self.log.warning('All returns are zero for %s' \
                        % r.description)
        
        # Get indices of factors that we don't want in the regression
        currencyFactorsIdx = expM.getFactorIndices(ExposureMatrix.CurrencyFactor)
        excludeFactorsIdx = set(currencyFactorsIdx + nonTradingMarketsIdx)

        if self.debuggingReporting:
            dates = [str(modelDate)]
            idList = [s.getSubIDString() for s in univ]
            retOutFile = 'tmp/%s-exretHist.csv' % self.name
            Utilities.writeToCSV(excessReturns, retOutFile, rowNames=idList, columnNames=dates)

        # Call nested regression routine
        (factorReturnsMap, specificReturns, regStatsMap, anova, fmpMap, constrComp) = \
                self.run_nested_regression(prevDate, excessReturns, expM, estu, 
                      weights_ESTU, univ, excludeFactorsIdx, modelDB, marketDB,
                      nestedEstu=estuArray, buildFMPs=buildFMPs,
                      removeNTDsFromRegression=removeNTDsFromRegression)
        
        # Map specific returns for cloned assets
        if not removeNTDsFromRegression:
            specificReturns = ma.masked_where(missingReturnData, specificReturns)
        hardCloneMap = modelDB.getClonedMap(modelDate, univ, cloningOn=self.hardCloning)
        if len(hardCloneMap) > 0:
            cloneList = [n for n in univ if n in hardCloneMap]
            for sid in cloneList:
                if hardCloneMap[sid] in univ:
                    specificReturns[assetIdxMap[sid]] = specificReturns\
                            [assetIdxMap[hardCloneMap[sid]]]

        if self.debuggingReporting:
            dateStr = [str(modelDate)]
            assetNames = [s.getModelID().getIDString() for s in univ]
            sretFile = 'tmp/%s-sret.csv' % self.name
            Utilities.writeToCSV(specificReturns, sretFile, rowNames=assetNames, columnNames=dateStr)

        # Store regression results
        factorReturns = Matrices.allMasked((len(self.factors),))
        regressionStatistics = Matrices.allMasked((len(self.factors), 4))
        for (fName, ret) in factorReturnsMap.items():
            idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
            if idx is not None:
                factorReturns[idx] = ret
                regressionStatistics[idx,:] = regStatsMap[fName]
        result = Utilities.Struct()
        result.universe = univ
        result.factorReturns = factorReturns
        result.specificReturns = specificReturns
        result.exposureMatrix = expM
        result.regressionStatistics = regressionStatistics
        result.adjRsquared = anova.calc_adj_rsquared()
        result.regression_ESTU = list(zip([result.universe[i] for i in anova.estU_], 
                        anova.weights_ / numpy.sum(anova.weights_)))
        result.VIF = self.VIF
        
        # Report non-trading markets and set factor return to zero
        allFactorNames = expM.getFactorNames()
        if len(nonTradingMarketsIdx) > 0:
            nonTradingMarketNames = ', '.join([allFactorNames[i] \
                                    for i in nonTradingMarketsIdx])
            self.log.info('%d non-trading market(s): %s',
                          len(nonTradingMarketsIdx), nonTradingMarketNames)
            for i in nonTradingMarketsIdx:
                idx = subFactorIDIdxMap[nameSubIDMap[allFactorNames[i]]]
                result.factorReturns[idx] = 0.0
        
        # Process FMPs
        newFMPMap = dict()
        sid2StringMap = dict([(sid.getSubIDString(), sid) for sid in univ])
        for (fName, fMap) in fmpMap.items():
            tmpMap = dict()
            for (sidString, fmp) in fMap.items():
                if sidString in sid2StringMap:
                    tmpMap[sid2StringMap[sidString]] = fmp
            newFMPMap[nameSubIDMap[fName]] = tmpMap
        result.fmpMap = newFMPMap

        # Pull in currency factor returns from currency model
        crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
        assert(crmi is not None)
        (currencyReturns, currencyFactors) = \
                self.currencyModel.loadCurrencyFactorReturns(crmi, modelDB)
        currSubFactors = modelDB.getSubFactorsForDate(modelDate, currencyFactors)
        currSubIDIdxMap = dict([(currSubFactors[i].subFactorID, i) \
                                for i in range(len(currSubFactors))])
        self.log.info('loaded %d currencies from currency model', len(currencyFactors))
        
        # Lookup currency factor returns from currency model
        currencyFactors = set(self.currencies)
        notIn = [i for i in currSubIDIdxMap.keys() if i not in subFactorIDIdxMap.keys()]
        if len(notIn) > 0 and (len(self.additionalCurrencyFactors) > 0):
            notInNames = [currSubFactors[currSubIDIdxMap[ni]].factor.name for ni in notIn]
            self.log.info('%s live currencies not in the model: %s', len(notIn), notInNames)
        for (i,j) in subFactorIDIdxMap.items():
            cidx = currSubIDIdxMap.get(i, None)
            if cidx is None:
                if self.factors[j] in currencyFactors:
                    self.log.warning('Missing currency factor return for %s',
                                  self.factors[j].name)
                    value = 0.0
                else:
                    continue
            else:
                value = currencyReturns[cidx]
            result.factorReturns[j] = value
        
        if nextTradDate is not None:
            self.regressionReporting(futureExcessReturns, result, expM, nameSubIDMap,
                    assetIdxMap, modelDate, buildFMPs=buildFMPs, constrComp=constrComp)
        else:
            self.regressionReporting(excessReturns, result, expM, nameSubIDMap,
                    assetIdxMap, modelDate, buildFMPs=buildFMPs, constrComp=constrComp,
                    specificRets=specificReturns)
        
        return result

    def createGlobalMarketAdjustmentTerm(self):
        assert(self.intercept is not None and
               self.usesReturnsTimingAdjustment())
        return ModelDB.RiskModelGroup(0, self.intercept.name, 
                                      self.name[:2], 0, None, None)

    def adjustFactorReturnsForTiming(self, modelDate, factorReturns,
                                     adjustments, modelDB, marketDB, rms_id=None):
        """Sets up an array of returns-timing adjustment factors that
        will be added to the factor returns time-series.
        Both factorReturns and adjustments should be TimeSeriesMatrix 
        objects containing the factor returns and market adjustment 
        factors time-series, respectively.  The adjustments array 
        cannot contain masked values.
        Returns the factor returns TimeSeries matrix with an extra
        'adjust' attribute appended.
        """
        if not self.usesReturnsTimingAdjustment():
            return factorReturns
        if rms_id is None:
            rms_id = self.rms_id
        self.log.debug('adjustFactorReturnsForTiming: begin')
        factorIdxMap = dict([(f.factor.name, i) for (i, f)
                             in enumerate(factorReturns.assets)])
        rmgIndices = [factorIdxMap[rmg.description] for rmg in self.rmg]
        adjustArray = numpy.zeros(factorReturns.data.shape[0:2], float)
        # Only keep adjustments relevant to model geographies
        rmgIdxMap = dict([(j.rmg_id,i) for (i,j) in \
                            enumerate(adjustments.assets)])
        modelCountryAdjustments = numpy.take(adjustments.data, 
                            [rmgIdxMap[r.rmg_id] for r in self.rmg], axis=0)
        # Weighted country factor return should sum to zero
        # so take out the weighted average and add it to the global term
        dateList = factorReturns.dates
        dateLen = len(dateList)
        weights = modelDB.loadRMSFactorStatisticHistory(rms_id, 
                    factorReturns.assets, dateList, 'regr_constraint_weight')
        weights = weights.data.filled(0.0)
        marketAdjTerm = numpy.sum(weights[rmgIndices,:] * \
                    modelCountryAdjustments[:,:dateLen], axis=0)
        adjustArray[rmgIndices,:] = modelCountryAdjustments[:,:dateLen] - marketAdjTerm
        adjustArray[factorIdxMap[self.intercept.description],:] = marketAdjTerm

        factorReturns.adjust = adjustArray
        if self.debuggingReporting:
            adjNames = [f.factor.name for f in factorReturns.assets]
            dateFieldNames = [str(d) for d in dateList]
            outData = numpy.transpose(adjustArray)
            adjOutFile = 'tmp/%s-adjHist.csv' % self.name
            Utilities.writeToCSV(outData, adjOutFile, columnNames=adjNames,
                                 rowNames=dateFieldNames)

        self.log.debug('adjustFactorReturnsForTiming: end')
        return factorReturns

    def generate_residual_regression_ESTU(self, indices_ESTU, weights_ESTU, iter,
                                          modelDate, expM, modelDB, marketDB):
        """Returns the estimation universe assets and weights
        to be used for the given loop of the nested model regression.
        By default all the nested regressions use the same ESTU,
        but this method can be replaced by whatever logic is
        specific to the model class.
        """
        self.log.debug('generate_residual_regression_ESTU: begin')
        self.log.debug('generate_residual_regression_ESTU: end')
        return (indices_ESTU, weights_ESTU)
        
    def run_nested_regression(self, date, excessReturns, expMatrix, 
                             estu, weights, universe, excludeIndices, 
                             modelDB, marketDB, nestedEstu=None,
                             buildFMPs=False, removeNTDsFromRegression=True):
        """Performs the dirty work of shuffling around arrays and
        such that is required for the nested regressions.  Also 
        computes 'aggregate' regression statistics for the model, such
        as r-square, t-stats, etc.  Returns a tuple containing a
        map of factor names to factor return values, a specific returns
        array, a map of factor names to regression statistics, and 
        a ReturnCalculator.RegressionANOVA object housing the results.
        """
        self.log.debug('run_nested_regression: begin')
        # Get regression paramters
        rp = self.returnCalculator.parameters
        regressionOrder = rp.getRegressionOrder()
        useRobustRegression = self.useRobustRegression
        if buildFMPs:
            useRobustRegression = False
        
        # Set up some data items to be used later
        allFactorNames = expMatrix.getFactorNames()
        factorReturnsMap = dict()
        regressStatsMap = dict()
        regressionReturns = ma.array(excessReturns)
        regressorMatrix = expMatrix.getMatrix()
        constraintComponent = None
        ANOVA_data = list()
        fmpMap = dict()

        # Create list of all factors going into any regression
        factorNameIdxMap = dict([(f,i) for (i,f) in enumerate(
                            allFactorNames) if i not in excludeIndices])
        
        # Identify possible dummy style factors
        dummyStyles = set()
        for i in expMatrix.getFactorIndices(ExposureMatrix.StyleFactor):
            if Utilities.is_binary_data(regressorMatrix[i,:]):
                dummyStyles.add(allFactorNames[i])
        
        # Loop round however many regressions are required
        for iReg in range(len(regressionOrder)):
            self.log.info('Beginning nested regression, loop %d', iReg+1)
            self.log.info('Factors in loop: %s', 
                        ', '.join([f.name for f in regressionOrder[iReg]]))

            # Get estimation universe and weights for this loop
            if nestedEstu is not None:
                estu = nestedEstu[iReg].indices
                weights = nestedEstu[iReg].weights
            else:
                (estu, weights) = self.generate_residual_regression_ESTU(
                        estu, weights, iReg+1, date, expMatrix, modelDB, marketDB)
            
            # Determine which factors will go into this regression loop
            regFactorNames = list()
            standAloneFactorNames = set([ff.name for f in regressionOrder[iReg+1:] \
                                    for ff in f if isinstance(ff, ModelFactor)])
            for obj in regressionOrder[iReg]:
                try:
                    fList = expMatrix.getFactorNames(obj)
                except:
                    fList = [obj.name]
                regFactorNames.extend([f for f in fList if \
                        f in factorNameIdxMap and f not in standAloneFactorNames])
            regFactorsIdx = [factorNameIdxMap[f] for f in regFactorNames]
            regMatrix = ma.take(regressorMatrix, regFactorsIdx, axis=0)
            
            # If ESTU assets have no returns, warn and skip
            checkReturns = ma.take(excessReturns, estu, axis=0)
            checkReturns = ma.masked_where(abs(checkReturns) < 1e-12, checkReturns)
            badReturns = numpy.sum(ma.getmaskarray(checkReturns))
            if badReturns >= 0.99 * len(estu):
                self.log.warning('No returns for nested regression loop %d ESTU, skipping', 
                                iReg + 1)
                bogusANOVA = ReturnCalculator.RegressionANOVA(
                            excessReturns, numpy.zeros(len(excessReturns)), 
                            len(regFactorsIdx), list(range(len(excessReturns))))
                ANOVA_data.append(bogusANOVA)
                specificReturns = ma.array(regressionReturns, copy=True)
                for fName in regFactorNames:
                    factorReturnsMap[fName] = 0.0
                    regressStatsMap[fName] = Matrices.allMasked(4)
                continue
            
            # Deal with thin factors (industry, country, dummy styles)
            thinTestIdx = list()
            if rp.getThinFactorCorrection() and not buildFMPs:
                # Create list of factors to be tested for thinness
                thinTestIdx = [i for (i,f) in enumerate(regFactorNames) \
                               if f in dummyStyles]
                factorTypesForThin = [fType for fType in regressionOrder[iReg] \
                        if fType in (ExposureMatrix.IndustryFactor, 
                                     ExposureMatrix.CountryFactor)]
                factorNamesForThin = set([f for fType in factorTypesForThin \
                                            for f in expMatrix.getFactorNames(fType)])
                if len(factorTypesForThin) > 0:
                    for (i,f) in enumerate(regFactorNames):
                        if f in factorNamesForThin:
                            thinTestIdx.append(i)

            tfc = RegressionToolbox.DummyAssetHandler(thinTestIdx, regFactorNames, rp)
            if len(thinTestIdx) > 0:
                tfc.nonzeroExposuresIdx = []
                # Allow dummy assets to have nonzero style exposures
                for f in regressionOrder[iReg]:
                    try:
                        fType = expMatrix.getFactorType(f.name)
                    except:
                        fType = f
                    if fType in (ExposureMatrix.StyleFactor, ExposureMatrix.InterceptFactor):
                        styleFactorsIdx = [i for i in range(len(regFactorsIdx)) if \
                            expMatrix.getFactorType(regFactorNames[i]) in 
                            (ExposureMatrix.StyleFactor, ExposureMatrix.InterceptFactor)]
                        tfc.nonzeroExposuresIdx = styleFactorsIdx
                        break
                # Assign return to dummies
                dr = rp.getDummyReturns()
                # If this regression doesn't involve industry factors, use market return
                if ExposureMatrix.IndustryFactor not in factorTypesForThin:
                    dr = RegressionToolbox.DummyMarketReturns()
                # If it involves multiple factors, including industries, use
                # market for non-industry factors, and classification-based 
                # returns for industries
                elif dr != RegressionToolbox.DummyMarketReturns() \
                            and len(factorTypesForThin) > 1:
                    mktret = RegressionToolbox.DummyMarketReturns()
                    mktret.factorIndices = tfc.idxToCheck
                    mktret.factorNames = tfc.factorNames
                    rets = mktret.computeReturns(regressionReturns, estu, 
                                         weights, universe, date)
                    tfc.setDummyReturns(rets)
                    tfc.setDummyReturnWeights(mktret.dummyRetWeights)
                dr.factorIndices = tfc.idxToCheck
                dr.factorNames = tfc.factorNames
                rets = dr.computeReturns(regressionReturns, estu, 
                                         weights, universe, date)
                tfc.setDummyReturns(rets)
                tfc.setDummyReturnWeights(dr.dummyRetWeights)
            self.returnCalculator.thinFactorAdjustment = tfc
            
            # Finally, run the regression
            (realRegMatrix, factorReturns, specificReturns, 
             reg_ANOVA, constraintWeight, fmpDict, fmpSubIssues,
             constrComp) = self.returnCalculator.\
                                    calc_Factor_Specific_Returns(
                        self, estu, regressionReturns, regMatrix, regFactorNames, 
                        weights, expMatrix, rp.getFactorConstraints()[iReg],
                        robustRegression=useRobustRegression,
                        removeNTDsFromRegression=removeNTDsFromRegression)
            if iReg > 0:
                # Skip resiudal regression FMP for now as it doesn't really make sense
                fmpDict = dict()
            else:
                constraintComponent = constrComp
            
            # Assuming that 1st round of regression is done on styles, industries, market etc.,
            # and second round regression is done on Domestic China etc.,
            if iReg==0 and self.returnCalculator.parameters.getCalcVIF():
                exposureMatrix = expMatrix.getMatrix().filled(0.0)
                self.VIF = self.returnCalculator.calculateVIF(exposureMatrix, reg_ANOVA.weights_, reg_ANOVA.estU_, expMatrix.factorIdxMap_)
            
            # Pass residuals to next regression loop
            regressionReturns = specificReturns

            # Keep record of factor returns and regression statistics
            ANOVA_data.append(reg_ANOVA)
            for j in range(len(regFactorsIdx)):
                fName = regFactorNames[j]
                factorReturnsMap[fName] = factorReturns[j]
                if fName in fmpDict:
                    fmpMap[fName] = dict(zip(fmpSubIssues, fmpDict[fName].tolist()))
                values = Matrices.allMasked(4)
                values[:3] = reg_ANOVA.regressStats_[j,:]
                if fName in constraintWeight:
                    values[3] = constraintWeight[fName]
                regressStatsMap[fName] = values

        # Regression ANOVA -- depends on whether all loops use same ESTU
        if len(ANOVA_data) == 1:
            sameESTU = True
        elif [sorted(ANOVA_data[i].estU_) == sorted(ANOVA_data[i-1].estU_) for i in range(1,len(ANOVA_data))] == [True] * (len(ANOVA_data)-1):
            sameESTU = True
        else:
            sameESTU = False
        if sameESTU:
            self.log.info('%d regression loops all use same ESTU, computing ANOVA', 
                            len(regressionOrder))
            numFactors = sum([n.nvars_ for n in ANOVA_data])
            regWeights = numpy.average(numpy.array(
                            [n.weights_ for n in ANOVA_data]), axis=0)
        else:
            self.log.info('ANOVA will assume same ESTU for all %d regression loops', 
                            len(regressionOrder))
            numFactors = ANOVA_data[0].nvars_
            regWeights = ANOVA_data[0].weights_
        reg_ESTU = ANOVA_data[0].estU_
        anova = ReturnCalculator.RegressionANOVA(excessReturns, 
                        specificReturns, numFactors, reg_ESTU, regWeights)

        self.log.debug('run_nested_regression: end')
        return (factorReturnsMap, specificReturns, regressStatsMap, anova, fmpMap, constraintComponent)

class RegionalStatisticalFactorModel(FactorRiskModel):
    """Regional statistical model
    """
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])

        if self.allCurrencies:
            self.allRMG = modelDB.getAllRiskModelGroups(inModels=True)

        # Add factor ID to factors, matching by description
        for f in self.blind:
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
        # No style or industry factors
        self.styles = []
        self.industries = []
        self.returnsMADBounds = (-10.0, 10.0)
    
    def isStatModel(self):
        return True

    def isCurrencyModel(self):
        return False

    def setFactorsForDate(self, date, modelDB=None):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        self.setBaseModelForDate(date)

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)
        self.currencies = []
        
        # Setup industry classification
        chngDate = list(self.industrySchemeDict.keys())[0]
        chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
        self.industryClassification = self.industrySchemeDict[chngDates[-1]]
        self.log.debug('Using %s classification scheme', self.industryClassification.name)

        # Create currency factors
        self.currencies = [ModelFactor(f, None)
                for f in set([r.currency_code for r in self.rmg])]
        # Add additional currency factors (to allow numeraire changes)
        # if necessary
        if self.allCurrencies:
            for rmg in self.allRMG:
                rmg.setRMGInfoForDate(date)
            additionalCurrencyFactors = [ModelFactor(f, None)
                    for f in set([r.currency_code for r in self.allRMG])]
            additionalCurrencyFactors.extend([ModelFactor('EUR', 'Euro')])
            self.additionalCurrencyFactors = [f for f in additionalCurrencyFactors
                    if f not in self.currencies and f.name in self.nameFactorMap]
            self.additionalCurrencyFactors = list(set(self.additionalCurrencyFactors))

        for f in self.additionalCurrencyFactors:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        self.additionalCurrencyFactors = [f for f in self.additionalCurrencyFactors
                if f not in self.currencies and f.isLive(date)]
        if len(self.additionalCurrencyFactors) > 0:
            self.currencies.extend(self.additionalCurrencyFactors)
            self.log.debug('Adding %d extra currencies: %s',
                    len(self.additionalCurrencyFactors),
                    [f.name for f in self.additionalCurrencyFactors])
        self.currencies = sorted(self.currencies)

        for f in self.currencies:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        allFactors = self.blind + self.currencies
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factors = allFactors
        self.validateFactorStructure(date)
    
    def compute_EP_statistic(self, assetReturns, specificReturns, estu, de_mean=True):
        """Computes model EP statistic and 'averaged R-squared' as
        defined in Connor (1995)
        """
        assetReturns = ma.take(assetReturns, estu, axis=0)
        specificReturns = ma.take(specificReturns, estu, axis=0)
        if de_mean:
            assetReturns = ma.transpose(ma.transpose(assetReturns) - \
                            ma.average(assetReturns, axis=1))
        numerator = numpy.sum([ma.inner(e,e) for e in specificReturns], axis=0)
        denominator = numpy.sum([ma.inner(r,r) for r in assetReturns], axis=0)
        ep = 1.0 - numerator/denominator
        self.log.info('EP statistic: %f', ep)
        sse = [float(ma.inner(e,e)) for e in ma.transpose(specificReturns)]
        sst = [float(ma.inner(r,r)) for r in ma.transpose(assetReturns)]
        sst = ma.masked_where(sst==0.0, sst)
        sst = 1.0 / sst
        avg_r2 = 1.0 - ma.inner(sse, sst.filled(0.0)) / len(sse)
        self.log.info('Average R-Squared: %f', avg_r2)
        return (ep, avg_r2)
    
    def generateStatisticalModel(self, modelDate, modelDB, marketDB):
        """Compute statistical factor exposures and returns
        Then combine returns with currency returns and build the
        composite covariance matrix
        """
        # Get risk model universe and market caps
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        data = Utilities.Struct()
        data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
        (mcapDates, goodRatio) = self.getRMDates(modelDate, modelDB, 20)
        data.marketCaps = modelDB.getAverageMarketCaps(
                    mcapDates, data.universe, self.numeraire.currency_id, marketDB)
        if not self.forceRun:
            assert(numpy.sum(ma.getmaskarray(data.marketCaps))==0)
        data.marketCaps = numpy.array(data.marketCaps)
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        # Pick up dict of assets to be cloned from others
        data.hardCloneMap = modelDB.getClonedMap(modelDate, data.universe, cloningOn=self.hardCloning)

        if self.modelHack.ahSharesISC:
            data.subIssueGroups = modelDB.getIssueCompanyGroups(modelDate, data.universe, marketDB)
        else:
            excludeList = ['AShares','BShares']
            self.log.info('Do not apply ISC specific covariance on %s'%(\
                    ','.join('%s'%s for s in excludeList)))
            data.subIssueGroups = modelDB.getSpecificIssueCompanyGroups(
                modelDate, data.universe, marketDB, excludeList)
        baseCurrencyID = self.numeraire.currency_id
    
        # Determine dates for returns history
        needDays = int(1.05 * self.returnHistory / goodRatio)
        dateList = modelDB.getDates(self.rmg, modelDate, needDays-1)
         
        # Get list of DR ids
        (tmp1, tmp2s, foreign) = self.process_asset_country_assignments(
                modelDate, data.universe, data.marketCaps,
                modelDB, marketDB)
          
        # Build exposure matrix with currency factors
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        self.addCurrencyFactorExposures(modelDate, data, modelDB, marketDB)

        # Compute excess returns
        returns = self.build_excess_return_history(
            dateList, data, foreign, data.exposureMatrix, modelDB, marketDB)
        
        if self.debuggingReporting:
            dates = [str(d) for d in returns.dates]
            idList = [s.getSubIDString() for s in data.universe]
            retOutFile = 'tmp/%s-exretHist.csv' % self.name
            Utilities.writeToCSV(returns.data, retOutFile, rowNames=idList, columnNames=dates)

        # Load estimation universe
        ids_ESTU = self.loadEstimationUniverse(rmi, modelDB)
        assert(len(ids_ESTU) > 0)
        ids_ESTU = [n for n in ids_ESTU if n in data.assetIdxMap]
        estu = [data.assetIdxMap[n] for n in ids_ESTU]
        data.estimationUniverseIdx = estu

        # Returns timing adjustment
        if self.usesReturnsTimingAdjustment():
            (returns, adjustments) = self.adjust_asset_returns_for_market_async(
                    returns, modelDate, modelDB, marketDB)
        
        # Remove dates on which lots of assets don't trade
        io = numpy.sum(ma.getmaskarray(returns.data), axis=0)
        goodDatesIdx = numpy.flatnonzero(io < 0.7 * len(returns.assets))
        badDatesIdx = [i for i in range(len(returns.dates)) if \
                i not in set(goodDatesIdx) and returns.dates[i].weekday() <= 4]
        if len(badDatesIdx) > 0:
            self.log.debug('Omitting weekday dates: %s',
                    ','.join([str(returns.dates[i])
                        for i in badDatesIdx]))
        keep = min(len(goodDatesIdx), self.returnHistory)
        goodDatesIdx = goodDatesIdx[-keep:]
        returns.dates = [returns.dates[i] for i in goodDatesIdx]
        returns.data = ma.take(returns.data, goodDatesIdx, axis=1)
        if hasattr(returns, 'zeroMask'):
            returns.zeroMask = ma.take(returns.zeroMask, goodDatesIdx, axis=1)
        
        if self.debuggingReporting:
            dates = [str(d) for d in returns.dates]
            idList = [s.getSubIDString() for s in data.universe]
            retOutFile = 'tmp/%s-retHist.csv' % self.name
            Utilities.writeToCSV(returns.data, retOutFile, rowNames=idList,
                    columnNames=dates)
            if self.usesReturnsTimingAdjustment():
                adjGood = ma.take(adjustments, goodDatesIdx, axis=1)
                adjNames = [r.description for r in self.rmg]
                adjOutFile = 'tmp/%s-adjHist.csv' % self.name
                Utilities.writeToCSV(adjGood, adjOutFile, rowNames=adjNames,
                        columnNames=dates)

        # Locate returns with "good enough" histories
        io = (ma.getmaskarray(returns.data)==0)
        weightSums = numpy.sum(io.astype(numpy.float)/self.returnHistory, axis=1)
        reallyGoodThreshold = 0.50
        reallyGoodAssetsIdx = numpy.flatnonzero(weightSums > reallyGoodThreshold)
        realESTU = list(set(estu).intersection(reallyGoodAssetsIdx))
        logging.info('Removing %d assets from ESTU with more than %.2f%% of returns missing',
                len(estu)-len(realESTU), reallyGoodThreshold*100.0)
        
        # Fill-in missing returns with proxied values
        if hasattr(returns, 'zeroMask') and not self.modelHack.fullProxyFill:
            ma.put(returns.data, numpy.flatnonzero(returns.zeroMask), 0.0)
        mask = numpy.array(ma.getmaskarray(returns.data))
        if self.industryClassification is not None:
            (returns, buckets) = self.proxyMissingAssetReturns(
                    modelDate, returns, data, modelDB)

        # Truncate extreme values according to MAD bounds
        clipped_returns = Matrices.TimeSeriesMatrix(returns.assets, returns.dates)
        (clipped_returns.data, mad_bounds) = Utilities.mad_dataset(
                    returns.data, self.returnsMADBounds[0], 
                    self.returnsMADBounds[1], realESTU, axis=0)
        
        # Downweight some countries if required
        for r in [r for r in self.rmg if r.downWeight < 1.0]:
            for sid in self.rmgAssetMap[r.rmg_id].intersection(ids_ESTU):
                clipped_returns.data[data.assetIdxMap[sid],:] *= r.downWeight

        # Compute exposures, factor and specific returns
        self.log.debug('Computing exposures and factor returns: begin')
        (expMatrix, factorReturns, specificReturns0, regressANOVA) = \
            self.returnCalculator.calc_ExposuresAndReturns(
                        clipped_returns, modelDate, realESTU)
        
        # Add statistical factor exposures to exposure matrix
        factorNames = [f.name for f in self.blind]
        data.exposureMatrix.addFactors(factorNames, numpy.transpose(expMatrix),
                ExposureMatrix.StatisticalFactor)
        if self.modelHack.specialDRTreatment:
            scores = self.score_linked_assets(
                    modelDate, data.universe, modelDB, marketDB,
                    subIssueGroups=data.subIssueGroups)
            data.exposureMatrix = self.clone_linked_asset_exposures(
                    modelDate, data, modelDB, marketDB, scores,
                    subIssueGroups=data.subIssueGroups)
        else:
            scores = None
        
        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[n] for n in foreign]
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=estu,
                    subIssueGroups=data.subIssueGroups, dp=self.dplace)
            dates = [str(d) for d in returns.dates]
            retOutFile = 'tmp/%s-facretHist-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(factorReturns, retOutFile, rowNames=factorNames,
                    columnNames=dates)

        # Compute 'real' specific returns using non-clipped returns
        exposureIdx = [data.exposureMatrix.getFactorIndex(n) for n in factorNames]
        expMatrix = numpy.transpose(ma.take(data.exposureMatrix.getMatrix(),
                            exposureIdx, axis=0))
        specificReturns = returns.data - numpy.dot(ma.filled(expMatrix, 0.0), factorReturns)
        if hasattr(returns, 'zeroMask') and not self.modelHack.fullProxyFill:
            ma.put(specificReturns, numpy.flatnonzero(returns.zeroMask), 0.0)
        
        # Map specific returns for cloned assets
        if len(data.hardCloneMap) > 0:
            cloneList = [n for n in data.universe if n in data.hardCloneMap]
            for sid in cloneList:
                if data.hardCloneMap[sid] in data.universe:
                    specificReturns[data.assetIdxMap[sid],:] = specificReturns\
                            [data.assetIdxMap[data.hardCloneMap[sid]],:]

        if self.debuggingReporting:
            dateStr = [str(d) for d in returns.dates]
            assetNames = [s.getModelID().getIDString() for s in data.universe]
            sretFile = 'tmp/%s-sret.csv' % self.name
            Utilities.writeToCSV(specificReturns, sretFile, rowNames=assetNames, columnNames=dateStr)

        # Compute various regression statistics
        data.regressionStatistics = Matrices.allMasked((len(self.factors), 4))
        regressStats = regressANOVA.calc_regression_statistics(
                            factorReturns[:,-1], ma.take(
                            expMatrix, regressANOVA.estU_, axis=0))
        data.regressionStatistics[:len(self.blind),:3] = regressStats
        data.regression_ESTU = list(zip([data.universe[i] for i in regressANOVA.estU_], 
                        regressANOVA.weights_ / numpy.sum(regressANOVA.weights_)))
        (ep, avg_r2) = self.compute_EP_statistic(
                        clipped_returns.data, specificReturns0, realESTU)
        
        # Compute root-cap-weighted R-squared to compare w/ cross-sectional model
        # Note that this is computed over the initial ESTU, not the 'real'
        # one going into the factor analysis
        weights = ReturnCalculator.calc_Weights(self.rmg, modelDB, marketDB,
                    modelDate, [data.universe[i] for i in estu], baseCurrencyID)
        regANOVA = ReturnCalculator.RegressionANOVA(clipped_returns.data[:,-1],
                specificReturns0[:,-1], self.numFactors, estu, weights)
        data.adjRsquared = regANOVA.calc_adj_rsquared()
        self.log.debug('Computing exposures and factor returns: end')
        
        # Create TimeSeriesMatrix objects for factor and specific returns
        self.log.debug('Building time-series matrices: begin')
        # Set up covariance parameters
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        # Check returns history lengths
        # Note that dates are switched to reverse chronological order
        dateList = returns.dates
        dateList.reverse()
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        omegaObs = min(len(dateList), maxOmegaObs)
        deltaObs = min(len(dateList), maxDeltaObs)
        self.log.info('Using %d of %d days of factor return history',
                      omegaObs, len(dateList))
        self.log.info('Using %d of %d days of specific return history',
                      deltaObs, len(dateList))
        
        # Set up specific returns history matrix
        data.srMatrix = Matrices.TimeSeriesMatrix(
                                returns.assets, dateList[:deltaObs])
        # Mask specific returns corresponding to missing returns
        data.srMatrix.data = ma.masked_where(
                numpy.fliplr(mask), numpy.fliplr(ma.filled(specificReturns, 0.0)))
        data.srMatrix.data = data.srMatrix.data[:,:deltaObs]
        self.log.debug('building time-series matrices: end')

        # Compute specific variances
        data.specificCov = dict()
        if self.modelHack.specRiskEstuFillIn:
            srEstuFillIn = estu
        else:
            srEstuFillIn = None
        if isinstance(self.specificRiskCalculator, RiskCalculator.SparseSpecificRisk2010):
            # ISC info
            exclList = list()
            data.subIssueGroups = modelDB.getIssueCompanyGroups(modelDate, returns.assets, marketDB)
            if not self.modelHack.ahSharesISC:
                #For China A/B shares, exclude them from the co-integration treatment
                excludeList = ['AShares','BShares']
                self.log.info('Do not apply ISC co-integration specific risk on %s'%(\
                        ','.join('%s'%s for s in excludeList)))
                clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
                assert(clsFamily is not None)
                clsMembers = dict([(i.name, i) for i in marketDB.\
                                       getClassificationFamilyMembers(clsFamily)])
                cm = clsMembers.get('Axioma Asset Type')
                assert(cm is not None)
                clsRevision = marketDB.\
                    getClassificationMemberRevision(cm, modelDate)
                homeClsData = modelDB.getMktAssetClassifications(
                    clsRevision, returns.assets, modelDate, marketDB)
                secTypeDict = dict([(i, j.classification.code) for (i,j) in homeClsData.items()])

                for (cmpId, sidList) in data.subIssueGroups.items():
                    for sid in sidList:
                        secType = secTypeDict.get(sid)
                        if secType is None:
                            self.log.error('Missing Axioma Sec Type for %s'%sid.getSubIDString())
                        elif secType in excludeList:
                            exclList.append(sid)
            logging.info('Exclusion list now contains %s assets'%len(exclList))

            data.srMatrix.nameDict = modelDB.getIssueNames(
                modelDate, returns.assets, marketDB)

            if self.modelHack.specialDRTreatment:
                self.log.info('Using NEW ISC Treatment for specific covariance')
                if not self.modelHack.ahSharesISC:
                    scoreGroups = modelDB.getSpecificIssueCompanyGroups(
                        modelDate, returns.assets, marketDB, excludeList)
                    scores = self.score_linked_assets(modelDate, returns.assets, modelDB, marketDB,
                                                      subIssueGroups=scoreGroups)
                    #Here put back the score = 0.0 for assets in excludeList, but not in score
                    for (groupId, subIssueList) in data.subIssueGroups.items():
                        score = scores.get(groupId)
                        if score is None:
                            newScore = numpy.zeros((len(subIssueList)), float)
                            scores[groupId] = newScore
                        else:
                            newScore = numpy.zeros((len(subIssueList)), float)
                            if len(newScore) != len(scores):
                                #There are assets got excluded and didn't get score assigned
                                idxMap = dict([(j,i) for (i,j) in enumerate(scoreGroups[groupId])])
                                for (kdx, sid) in enumerate(subIssueList):
                                    if sid in idxMap:
                                        newScore[kdx] = score[idxMap[sid]]
                                scores[groupId] = newScore
                else:
                    scores = self.score_linked_assets(modelDate, returns.assets, modelDB, marketDB,
                                                      subIssueGroups=data.subIssueGroups)
            else:
                scores = None
            (data.specificVars, data.specificCov) = self.specificRiskCalculator.\
                    computeSpecificRisks(data.srMatrix, data.marketCaps,
                                         data.subIssueGroups, restrict=srEstuFillIn, scores=scores,
                                         specialDRTreatment=self.modelHack.specialDRTreatment,
                                         debuggingReporting=self.debuggingReporting,
                                         coinExclList=exclList, hardCloneMap=data.hardCloneMap)
        else:
            data.specificVars = self.specificRiskCalculator.\
                    computeSpecificRisks(data.exposureMatrix, data.marketCap, 
                                         data.srMatrix, 
                                         estu=srEstuFillIn)
        self.log.debug('computed specific variances')
        
        # More complex factor covariance part now
        # Set up statistical factor returns history and corresponding
        # information
        nonCurrencySubFactors = [f for f in subFactors \
                if f.factor not in self.currencies]
        nonCurrencyFactorReturns = Matrices.TimeSeriesMatrix(
                nonCurrencySubFactors, dateList[:omegaObs])
        nonCurrencyFactorReturns.data = ma.array(numpy.fliplr(factorReturns))[:,:omegaObs]
        
        # Load currency returns
        currencySubFactors = [f for f in subFactors \
                if f.factor in self.currencies]
        currencyFactorReturns = self.currencyModel.loadCurrencyFactorReturnsHistory(
                currencySubFactors, dateList[:maxOmegaObs], modelDB)
        # Copy currency factor returns, since they may get overwritten during
        # the cov matrix calculation
        tmpCFReturns = ma.array(currencyFactorReturns.data, copy=True)
        
        # Load currency risk model and compute factor cov
        crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
        assert(crmi is not None)
        data.factorCov = self.build_regional_covariance_matrix(\
                    modelDate, dateList[:maxOmegaObs],
                    currencyFactorReturns, nonCurrencyFactorReturns,
                    crmi, modelDB, marketDB)

        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            ncFactorNames = [f.factor.name for f in nonCurrencySubFactors]
            cFactorNames = [f.factor.name for f in currencySubFactors]
            factorNames = ncFactorNames + cFactorNames
            (d, corrMatrix) = Utilities.cov2corr(data.factorCov, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=factorNames,
                    rowNames=factorNames)

            # Output asset risks
            exposureIdx = [data.exposureMatrix.getFactorIndex(n) for n in factorNames]
            expMatrix = ma.filled(ma.take(data.exposureMatrix.getMatrix(), exposureIdx, axis=0), 0.0)
            assetFactorCov = numpy.dot(numpy.transpose(expMatrix), numpy.dot(data.factorCov, expMatrix))
            totalVar = numpy.diag(assetFactorCov) + data.specificVars
            sidList = [sid.getSubIDString() for sid in returns.assets]
            fname = 'tmp/%s-totalRisks-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ma.sqrt(totalVar), fname, rowNames=sidList)
            fname = 'tmp/%s-specificRisks-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ma.sqrt(data.specificVars), fname, rowNames=sidList)

        # Add covariance matrix to return object
        data.frMatrix = Matrices.TimeSeriesMatrix(
                subFactors, dateList[:omegaObs])
        data.frMatrix.data[:len(self.blind),:] = nonCurrencyFactorReturns.data
        data.frMatrix.data[len(self.blind):,:] = tmpCFReturns
        
        self.log.debug('computed factor covariances')
        return data

    def addCurrencyFactorExposures(self, modelDate, data, modelDB, marketDB):
        data.exposureMatrix = GlobalExposures.generate_currency_exposures(
                modelDate, self, modelDB, marketDB, data)

class FactorRiskModelv3(FactorRiskModel):
    def generateFactorSpecificRisk(self, date, modelDB, marketDB):
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
        """
        # get sub-factors and sub-issues active on this day
        (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
        (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
        (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
        (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        (selectiveDeMean, deMeanFactorTypes, deMeanHalfLife,
                deMeanMinHistoryLength, deMeanMaxHistoryLength) = \
                self.vp.getSelectiveDeMeanParameters()
        if selectiveDeMean:
            maxOmegaObs = max(maxOmegaObs, deMeanMaxHistoryLength)
        
        subFactors = modelDB.getSubFactorsForDate(date, self.factors)
        #assert(len(subFactors) == len(self.factorIDMap))
        rmi = modelDB.getRiskModelInstance(self.rms_id, date)
        if rmi == None:
            raise LookupError('No risk model instance for %s' % str(date))
        if not rmi.has_exposures or not rmi.has_returns:
            raise LookupError(
                'Exposures or returns missing in risk model instance for %s'
                % str(date))
        subIssues = modelDB.getRiskModelInstanceUniverse(rmi)
        self.rmg = [r for r in self.rmg if r not in self.nurseryRMGs]
        data = AssetProcessor.process_asset_information(
                date, subIssues, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun)
                #nurseryRMGList=self.nurseryRMGs)
        
        self.log.debug('building time-series matrices: begin')
        omegaDateList = modelDB.getDates(self.rmg, date, maxOmegaObs-1, excludeWeekend=True)
        omegaDateList.reverse()
        deltaDateList = modelDB.getDates(self.rmg, date, maxDeltaObs-1, excludeWeekend=True)
        deltaDateList.reverse()
        # Check that enough consecutive days have returns, try to get
        # the maximum number of observations
        if len(omegaDateList) > len(deltaDateList):
            dateList = omegaDateList
        else:
            dateList = deltaDateList
        
        rmiList = modelDB.getRiskModelInstances(self.rms_id, dateList)
        okDays = [i.date == j and i.has_returns for (i,j)
                  in zip(rmiList, dateList)]

        okDays.append(False)
        firstBadDay = okDays.index(False)
        dateList = dateList[:firstBadDay]
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        if len(dateList) < maxOmegaObs:
            self.log.info('Using only %d of %d days of factor return history',
                          len(dateList), maxOmegaObs)
        if len(dateList) < maxDeltaObs:
            self.log.info('Using only %d of %d days of specific return history',
                          len(dateList), maxDeltaObs)

        # Remove dates for which many markets are non-trading
        if not self.SCM:
            minMarkets = int(0.5 * len(self.countries))
            datesAndMarkets = modelDB.getActiveMarketsForDates(
                                    self.rmg, dateList[-1], dateList[0])
            datesAndMarkets = [(d,n) for (d,n) in datesAndMarkets \
                                    if d.weekday() <= 4]

            # Remember, datesAndMarkets is in chron. order whereas dateList is reversed
            badDatesIdx = [len(dateList)-i-1 for (i,n) in \
                            enumerate(datesAndMarkets) if n[1] <= minMarkets]
            badDates = numpy.take(dateList, badDatesIdx)
            self.log.info('Removing %d dates with < %d markets trading: %s',
                            len(badDates), minMarkets, ','.join([str(d) for d in badDates]))
            goodDatesIdx = [i for i in range(len(dateList)) if i not in badDatesIdx]
            dateList = numpy.take(dateList, goodDatesIdx)

        ret = Utilities.Struct()
        ret.subIssues = subIssues
        dateList = list(dateList)
        
        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB, data)
        estuIdx = [data.assetIdxMap[sid] for sid in estu]

        # Load returns timing adjustment factors if required
        if self.usesReturnsTimingAdjustment():
            rmgList = modelDB.getAllRiskModelGroups()
            adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                self.returnsTimingId, rmgList,
                dateList[:max(maxOmegaObs, maxDeltaObs)])
            adjustments.data = adjustments.data.filled(0.0)
        else:
            adjustments = None

        # Compute specific risk
        # Identify specific returns where the total return is missing
        dateListCopy = dateList[:maxDeltaObs]
        dateListCopy.reverse()
        returns = modelDB.loadTotalReturnsHistoryV3(
                self.rmg, date, subIssues, int(maxDeltaObs),
                dateList=dateListCopy, excludeWeekend=True)
        zeroRetsFlag = ma.getmaskarray(ma.masked_where(abs(returns.data)<1.0e-12, returns.data))
        nonMissingFlag = (returns.preIPOFlag==0) * (returns.notTradedFlag==0) * (zeroRetsFlag==0)
        numOkReturns = ma.sum(nonMissingFlag, axis=1)
        numOkReturns = numOkReturns / float(numpy.max(numOkReturns, axis=None))

        # Load specific returns history
        specificReturns = modelDB.loadSpecificReturnsHistory(
                        self.rms_id, subIssues, dateList[:maxDeltaObs])
        ret = Utilities.Struct()
        ret.subIssues = subIssues
        ret.specificCov = dict()
         
        # ISC info
        subIssueGroups = modelDB.getIssueCompanyGroups(date, subIssues, marketDB)
        specificReturns.nameDict = modelDB.getIssueNames(
                specificReturns.dates[0], subIssues, marketDB)
        scores = self.load_ISC_Scores(date, data, modelDB, marketDB)

        # Extra tweaking for linked assets
        if not self.SCM:
            maxNRets = numpy.max(numOkReturns, axis=None)
            entireUniverse = modelDB.getAllActiveSubIssues(date)
            allSubIssueGroups = modelDB.getIssueCompanyGroups(
                    date, entireUniverse, marketDB)
            for (groupId, subIssueList) in allSubIssueGroups.items():
                for sid in subIssueList:
                    if sid in data.assetIdxMap:
                        numOkReturns[data.assetIdxMap[sid]] = maxNRets
         
        # If non-empty, apply timing adjustment to DR specific returns
        if len(data.foreign) > 0:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            # Load full set of adjustment factors for every relevant market
            if adjustments == None:
                rtId = self.returnsTimingId
                rmgList = modelDB.getAllRiskModelGroups()
                if rtId == None and hasattr(self, 'specReturnTimingId'):
                    rtId = self.specReturnTimingId
                if rtId is not None:
                    adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                            rtId, rmgList, dateList[:max(maxOmegaObs, maxDeltaObs)])
                    adjustments.data = adjustments.data.filled(0.0)

            if adjustments is not None:
                specificReturns = self.adjustDRSpecificReturnsForTiming(
                        date, specificReturns, modelDB, marketDB,
                        adjustments, data.foreign, rmgList, data)

        # Specific risk computation
        (ret.specificVars, ret.specificCov) = self.specificRiskCalculator.\
                computeSpecificRisks(specificReturns, data,
                        subIssueGroups, self.rmg, modelDB,
                        nOkRets=numOkReturns, restrict=estuIdx, scores=scores,
                        debuggingReporting=self.debuggingReporting,
                        hardCloneMap=data.hardCloneMap,
                        gicsDate=self.gicsDate)
        self.log.debug('computed specific variances')
         
        factorReturns = modelDB.loadFactorReturnsHistory(
                self.rms_id, subFactors, dateList[:maxOmegaObs], screen=True)
         
        expM = self.loadExposureMatrix(rmi, modelDB, addExtraCountries=False,
                #(len(self.nurseryRMGs)>0),
                assetList=data.universe)

        # Test code for selective demeaning of certain factors or factor types
        if selectiveDeMean:
            fname2Idx = dict((f.factor.name, idx) for (idx, f) in enumerate(factorReturns.assets))
            dmFactorNames = []
            for fType in deMeanFactorTypes:
                if fType in expM.factorTypes_:
                    dmFactorNames.extend(expM.getFactorNames(fType))
                else:
                    dmFactorNames.append(fType)
            indices = [fname2Idx[fn] for fn in dmFactorNames]
            means = numpy.zeros((len(factorReturns.assets)), float)
            if len(factorReturns.dates) < deMeanMinHistoryLength:
                frets = numpy.zeros((
                    len(factorReturns.assets), deMeanMinHistoryLength), float)
                frets[:,:factorReturns.data.shape[1]] = factorReturns.data
            else:
                frets = factorReturns.data[:,:deMeanMaxHistoryLength]
            weights = Utilities.computeExponentialWeights(
                    deMeanHalfLife, frets.shape[1], equalWeightFlag=False, normalize=True)
            for idx in indices:
                means[idx] = numpy.dot(weights, frets[idx,:])
            factorReturns.mean = means

        # Compute factor covariance matrix
        if self.SCM:
            (shrinkType, shrinkFactor) = self.cp.getShrinkageParameters()
            if shrinkType is not None:
                self.compute_shrinkage_matrix(shrinkType, shrinkFactor, subFactors,
                                    date, expM, modelDB)

            ret.factorCov = self.covarianceCalculator.\
                computeFactorCovarianceMatrix(factorReturns)

            (resampleType, resampleIters) = self.cp.getResampleType()
            if resampleType is not None:
                (eigval, eigvec) = linalg.eigh(ret.factorCov / 252.0)
                eigval = ma.masked_where(eigval<=0.0, eigval)
                eigval = ma.filled(eigval, 0.0)
                eigvec = ma.filled(eigvec, 0.0)
                averageFCov = numpy.zeros(ret.factorCov.shape, float)
                numpy.random.seed(int(time.mktime(date.timetuple())))

                if not self.debuggingReporting:
                    l = logging.getLogger().getEffectiveLevel()
                    logging.getLogger().setLevel(logging.ERROR)

                for itr in range(resampleIters):
                    rVec = numpy.random.normal(size=factorReturns.data.shape)
                    rVec = numpy.dot(numpy.diag(numpy.sqrt(eigval)), rVec)
                    factorReturns.data = numpy.dot(eigvec, rVec)
                    rsFactorCov = self.covarianceCalculator.\
                            computeFactorCovarianceMatrix(factorReturns)
                    averageFCov += rsFactorCov
                averageFCov = averageFCov / float(resampleIters)

                if not self.debuggingReporting:
                    logging.getLogger().setLevel(l)

                if resampleType == 'fancy':
                    y = float(factorReturns.data.shape[0]) / float(factorReturns.data.shape[1])
                    alphaInv = numpy.sqrt(1.0 - y)
                    ret.factorCov = ((1.0 + alphaInv) * ret.factorCov) - (alphaInv * averageFCov)
                else:
                    ret.factorCov = averageFCov
        else:
            nonCurrencySubFactors = [f for f in subFactors \
                    if f.factor not in self.currencies]
            # Load up non-currency factor returns
            nonCurrencyFactorReturns = modelDB.loadFactorReturnsHistory(
                    self.rms_id, nonCurrencySubFactors, dateList[:maxOmegaObs], screen=True)

            # Adjust factor returns for returns-timing, if applicable
            nonCurrencyFactorReturns = self.adjustFactorReturnsForTiming(
                    date, nonCurrencyFactorReturns, adjustments,
                    modelDB, marketDB)

            # Pull up currency subfactors and returns
            currencySubFactors = [f for f in subFactors if f.factor in self.currencies]
            currencyFactorReturns = modelDB.loadFactorReturnsHistory(
                    self.currencyModel.rms_id, currencySubFactors,
                    dateList[:maxOmegaObs], screen=True)
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, date)
            assert(crmi is not None)

            # Post-process the array data a bit more, then compute cov
            factorCov = self.build_regional_covariance_matrix(
                    date, dateList[:maxOmegaObs],
                    currencyFactorReturns, nonCurrencyFactorReturns,
                    crmi, modelDB, marketDB)
            tmpSFs = nonCurrencySubFactors + currencySubFactors
            subFactorIdxMap = dict(zip(tmpSFs, list(range(len(tmpSFs)))))
            factorOrder = [subFactorIdxMap[s] for s in subFactors]
            factorCov = numpy.take(factorCov, factorOrder, axis=0)
            ret.factorCov = numpy.take(factorCov, factorOrder, axis=1)

        # Report day-on-day correlation matrix changes
        self.reportCorrelationMatrixChanges(date, ret.factorCov,
                                             rmiList[1], modelDB)
        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            factorNames = [f.factor.name for f in subFactors]
            (d, corrMatrix) = Utilities.cov2corr(ret.factorCov, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=factorNames,
                    rowNames=factorNames)
            var = numpy.diag(ret.factorCov)[:,numpy.newaxis]
            sqrtvar = 100.0 * numpy.sqrt(var)
            self.log.info('Factor risk: (Min, Mean, Max): (%.2f%%, %.2f%%, %.2f%%)',
                    ma.min(sqrtvar, axis=None), numpy.average(sqrtvar, axis=None),
                    ma.max(sqrtvar, axis=None))
            varoutfile = 'tmp/%s-varFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(var, varoutfile, rowNames=factorNames)
            covOutFile = 'tmp/%s-covFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ret.factorCov, covOutFile, columnNames=factorNames,
                    rowNames=factorNames)

        self.log.info('Sum of composite cov matrix elements: %f',
                    ma.sum(ret.factorCov, axis=None))

        ret.subFactors = subFactors
        self.log.debug('computed factor covariances')
        return ret
    
    def compute_shrinkage_matrix(self, shrinkType, shrinkFactor, subFactors, date,
                    expM, modelDB, daysBack=2500, halflife=500, fadeLength=21):
        """Applies shrinkage to sample correlation matrix.
        Shrinkage intensity is given by shrinkFactor
        If shrinkType is 'complex', we use shrinkage to long-term averages within sectors
        and within styles. Otherwise, we shrink towards the identity matrix
        """

        # Initialise
        shrinkArray = numpy.eye(len(subFactors), dtype=float)
        sfNameList = [sf.factor.name for sf in subFactors]
        sfNameMap = dict(zip(sfNameList, subFactors))
        sfNameIdxMap = dict(zip(sfNameList, list(range(len(subFactors)))))

        # Set up dates
        shrinkDateList = modelDB.getDates(self.rmg, date, daysBack, excludeWeekend=True)
        shrinkDateList.reverse()
        rmiList = modelDB.getRiskModelInstances(self.rms_id, shrinkDateList)
        okDays = [i.date == j and i.has_returns for (i,j) in zip(rmiList, shrinkDateList)]
        okDays.append(False)
        firstBadDay = okDays.index(False)
        shrinkDateList = shrinkDateList[:firstBadDay]
        T = len(shrinkDateList)

        # Get weights
        endWeights = Utilities.computePyramidWeights(fadeLength, fadeLength, T)
        expWeights = Utilities.computeExponentialWeights(halflife, T)
        weights = endWeights * expWeights

        if shrinkType != 'simple':
            self.log.info('%d dates used for shrinkage', T)
            # Get mapping of industries to sectors
            sectors = self.industryClassification.getClassificationParents('Sectors', modelDB)
            sectorNameMap = dict(zip([p.name for p in sectors], sectors))
            sectorIndMap = dict()
            for f in self.industryClassification.getLeafNodes(modelDB).values():
                ancestor = Utilities.findIndustryParent(
                        self.industryClassification, modelDB, f, list(sectorNameMap.keys()))
                if ancestor in sectorIndMap:
                    sectorIndMap[ancestor].append(f.name)
                else:
                    sectorIndMap[ancestor] = [f.name]
                
            # Compute long-term correlations within sectors
            for sec in sectorIndMap.keys():
                secSFList = [sfNameMap[nm] for nm in sectorIndMap[sec]]
                # Load up factor returns history
                if hasattr(self, 'internalFactorReturns') and self.internalFactorReturns:
                    secReturns = modelDB.loadFactorReturnsHistory(
                            self.rms_id, secSFList, shrinkDateList, flag='internal', screen=True)
                else:
                    secReturns = modelDB.loadFactorReturnsHistory(
                            self.rms_id, secSFList, shrinkDateList, screen=True)

                # Compute the correlations and paste them into the larger matrix
                secReturns.data = secReturns.data * weights
                longTermCorrel = numpy.corrcoef(secReturns.data)
                for (idx, ind1) in enumerate(sectorIndMap[sec]):
                    indIdx1 = sfNameIdxMap[ind1]
                    for (jdx, ind2) in enumerate(sectorIndMap[sec]):
                        indIdx2 = sfNameIdxMap[ind2]
                        shrinkArray[indIdx1, indIdx2] = longTermCorrel[idx, jdx]
                        shrinkArray[indIdx2, indIdx1] = longTermCorrel[jdx, idx]

            # Now do the same for style factors
            shrinkTypes = [ExposureMatrix.StyleFactor, ExposureMatrix.InterceptFactor]
            styleNames = []
            for fType in shrinkTypes:
                if fType in expM.factorTypes_:
                    styleNames.extend(expM.getFactorNames(fType))
                else:
                    styleNames.append(fType)
            styleSFList = [sfNameMap[nm] for nm in styleNames]

            # Load up factor returns history
            if hasattr(self, 'internalFactorReturns') and self.internalFactorReturns:
                styleReturns = modelDB.loadFactorReturnsHistory(
                        self.rms_id, styleSFList, shrinkDateList, flag='internal',
                        screen=True)
            else:
                styleReturns = modelDB.loadFactorReturnsHistory(
                        self.rms_id, styleSFList, shrinkDateList, screen=True)

            # Compute long-term correlations across styles
            styleReturns.data = styleReturns.data * weights
            longTermCorrel = numpy.corrcoef(styleReturns.data)
            for (idx, ind1) in enumerate(styleNames):
                indIdx1 = sfNameIdxMap[ind1]
                for (jdx, ind2) in enumerate(styleNames):
                    indIdx2 = sfNameIdxMap[ind2]
                    indIdx2 = sfNameIdxMap[ind2]
                    shrinkArray[indIdx1, indIdx2] = longTermCorrel[idx, jdx]
                    shrinkArray[indIdx2, indIdx1] = longTermCorrel[jdx, idx]

        self.covarianceCalculator.shrinkMatrix = shrinkArray
        self.covarianceCalculator.shrinkFactor = shrinkFactor
        self.log.info('Shrink type: %s, shrinkage factor: %s',
                shrinkType, shrinkFactor)
        return

    def assetReturnHistoryLoader(self, data, returnHistory, modelDate, modelDB, marketDB):
        """ Function to load in returns for factor regression
        """
        # Load in history of returns
        returnsProcessor = ProcessReturns.assetReturnsProcessor(
                self.rmg, data.universe, data.rmgAssetMap,
                data.tradingRmgAssetMap, data.assetTypeDict,
                debuggingReporting=self.debuggingReporting,
                numeraire_id=self.numeraire.currency_id,
                tradingCurrency_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId,
                gicsDate=self.gicsDate)
        returnsHistory = returnsProcessor.process_returns_history(
                modelDate, int(returnHistory), modelDB, marketDB,
                drCurrMap=data.drCurrData, loadOnly=False, excludeWeekend=True,
                applyRT=(self.SCM==False), trimData=False, noProxyList=data.noProxyList,
                useAllRMGDates=(self.SCM==False))
        return returnsHistory

    def build_excess_return_history_v3(self, data, modelDate, modelDB, marketDB):
        """Wrapper for building history of excess returns
        The logic is complex as an asset's risk-free rate ISO
        can change over time as its trading currency changes
        Thus, we need to loop over blocks of constant ISO
        """
        # Load in history of returns
        returnsHistory = self.assetReturnHistoryLoader(
                data, self.returnHistory, modelDate, modelDB, marketDB)

        # And a bit of sorting and checking
        if returnsHistory.data.shape[1] > self.returnHistory:
            returnsHistory.data = returnsHistory.data[:,-self.returnHistory:]
            returnsHistory.dates = returnsHistory.dates[-self.returnHistory:]
            returnsHistory.preIPOFlag = returnsHistory.preIPOFlag[:,-self.returnHistory:]
            returnsHistory.missingFlag = returnsHistory.missingFlag[:,-self.returnHistory:]
        dateList = returnsHistory.dates
        maskedData = numpy.array(ma.getmaskarray(returnsHistory.data))
        nonMissingFlag = (returnsHistory.preIPOFlag==0) * (returnsHistory.missingFlag==0)
        data.nonMissingFlag = nonMissingFlag
        numOkReturns = ma.sum(nonMissingFlag, axis=1)
        data.numOkReturns = numOkReturns / float(numpy.max(numOkReturns, axis=None))
        data.preIPOFlag = returnsHistory.preIPOFlag

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
        allReturns = Matrices.allMasked((len(data.universe),len(dateList)), float)
        allMasked = numpy.zeros((len(data.universe),len(dateList)), bool)
        startDate = changeDates[0]
        iDate0 = 0
        for endDate in changeDates[1:]:
            # Get list of dates within given range
            subDateList = sorted(d for d in dateList if startDate <= d < endDate)
            self.log.info('Using returns from %s to %s',
                    subDateList[0], subDateList[-1])
            assert(hasattr(modelDB, 'currencyCache'))
            if len(data.foreign) > 0:
                # Map rmgs to their trading currency ID at endDate
                lastRangeDate = subDateList[-1]
                rmgCurrencyMap = dict()
                for r in self.rmgTimeLine:
                    if r.from_dt <= lastRangeDate \
                            and r.thru_dt > lastRangeDate:
                        ccyCode = r.rmg.getCurrencyCode(lastRangeDate)
                        ccyID = modelDB.currencyCache.getCurrencyID(
                            ccyCode, lastRangeDate)
                        rmgCurrencyMap[r.rmg_id] = ccyID
                # Force DRs to have the trading currency of their home
                # country for this period
                drCurrData = dict()
                for (rmg_id, rmgAssets) in data.rmgAssetMap.items():
                    drCurrency = rmgCurrencyMap.get(rmg_id)
                    if drCurrency is None:
                        # the model didn't cover the country earlier
                        # so get its information from the database
                        rmg = modelDB.getRiskModelGroup(rmg_id)
                        ccyCode = rmg.getCurrencyCode(lastRangeDate)
                        drCurrency = modelDB.currencyCache.getCurrencyID(
                            ccyCode, lastRangeDate)
                    for sid in rmgAssets & set(data.foreign):
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
            mask = ma.take(maskedData, subDateIdx, axis=1)

            # Compute excess returns for date range
            (returns, rfr) = self.computeExcessReturns(subDateList[-1],
                    returns, modelDB, marketDB, drCurrData)

            # Copy chunk of excess returns into full returns matrix
            iDate1 = iDate0 + len(subDateList)
            allReturns[:,iDate0:iDate1] = returns.data
            allMasked[:,iDate0:iDate1] = mask
            iDate0 += len(subDateList)
            startDate = endDate
 
        # Copy excess returns back into returns structure
        returnsHistory.data = allReturns
        data.returns = returnsHistory
        data.maskedData = allMasked
        if hasattr(data, 'estimationUniverse'):
            data.realESTU = data.estimationUniverseIdx
        return data

class FundamentalModel(FactorRiskModelv3):
    """Fundamental factor model
    """
    ### XXX Test code here
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)

        # No statistical factors, of course
        self.blind = []
        self.VIF = None

        # Set up default estimation universe
        self.estuMap = None

        # Special regression treatment for certain factors 
        self.zeroExposureNames = []
        self.zeroExposureTypes = []
        modelDB.setTotalReturnCache(367)
        modelDB.setVolumeCache(150)
        self.legacyISCSwitchDate = datetime.date(1960, 1, 1)

    def setFactorsForDate(self, date, modelDB):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Set up estimation universe parameters
        self.estuMap = modelDB.getEstuMappingTable(self.rms_id)
        if self.estuMap is None:
            logging.error('No estimation universe mapping defined')
            assert(self.estuMap is not None)
        logging.info('Estimation universe structure: %d estus', len(self.estuMap))

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)
        if hasattr(self, 'baseModelDateMap'):
            self.setBaseModelForDate(date)
        else:
            self.baseModel = None

        if len(self.rmg) < 2:
            self.SCM = True
        else:
            self.SCM = False

        factors = modelDB.getRiskModelSerieFactors(self.rms_id)
        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        self.allStyles = list(self.styles)

        # Assign to new industry scheme if necessary
        if hasattr(self, 'industrySchemeDict'):
            chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
            self.industryClassification = self.industrySchemeDict[chngDates[-1]]
            self.log.info('Using %s classification scheme, rev_dt: %s'%\
                          (self.industryClassification.name, chngDates[-1].isoformat()))

        # Create industry factors
        industries = list(self.industryClassification.getLeafNodes(modelDB).values())
        self.industries = [ModelFactor(None, f.description) for f in industries]

        if self.SCM:
            countries = []
            currencies = []
            self.countryRMGMap = dict()
        else:
            # Create country factors
            countries = [ModelFactor(r.description, None) for r in self.rmg \
                    if r.description in self.nameFactorMap]
            self.CountryFactorRMGMap = dict(zip(countries, self.rmg))

            # Create currency factors
            allRMG = modelDB.getAllRiskModelGroups(inModels=True)
            for rmg in allRMG:
                rmg.setRMGInfoForDate(date)
            currencies = [ModelFactor(f, None) for f in set([r.currency_code for r in allRMG])]
            currencies.extend([ModelFactor('EUR', 'Euro')])
            currencies = sorted(list(set([f for f in currencies if f.name in self.nameFactorMap])))

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
            self.log.info('%d styles not live in current model: %s', len(dropped), dropped)
        self.styles = [s for s in self.allStyles if s.isLive(date)]
        self.currencies = [c for c in currencies if c.isLive(date)]
        self.countries = [f for f in countries if f.isLive(date)]
        self.nurseryCountries = [f for f in countries if f not in self.countries]
        self.hiddenCurrencies = [f for f in currencies if f not in self.currencies]
        self.nurseryRMGs = [self.CountryFactorRMGMap[f] for f in self.nurseryCountries]
        if len(self.nurseryRMGs) > 0:
            logging.info('%d out of %d RMGs classed as nursery: %s',
                    len(self.nurseryRMGs), len(self.rmg),
                    [r.mnemonic for r in self.nurseryRMGs])
        if len(self.hiddenCurrencies) > 0:
            logging.info('%d out of %d currencies classed as hidden: %s',
                    len(self.hiddenCurrencies), len(currencies),
                    ','.join([c.name for c in self.hiddenCurrencies]))

        # Set up dicts
        self.factors = [f for f in allFactors if f.isLive(date)]
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.validateFactorStructure(date, warnOnly=self.variableStyles)
        self.allFactors = allFactors
    
    def generate_model_universe(self, modelDate, modelDB, marketDB):
        """Generate risk model instance universe and estimation
        universe.  Return value is a Struct containing a universe
        attribute (list of SubIssues) containing the model universe
        and an estimationUniverse attribute (list of index positions
        corresponding to ESTU members).
        """
        self.log.debug('generate_model_universe: begin')
        # Get basic risk model instance universe
        universe = AssetProcessor.getModelAssetMaster(
                self, modelDate, modelDB, marketDB)

        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                numeraire_id=self.numeraire.currency_id,
                legacyDates=self.legacyMCapDates,
                forceRun=self.forceRun,
                nurseryRMGList=self.nurseryRMGs)

        # Compute the various levels of market cap
        AssetProcessor.computeTotalIssuerMarketCaps(
                data, modelDate, self.numeraire, modelDB, marketDB,
                debugReport=self.debuggingReporting)

        # Build a temporary exposure matrix to house things
        # like industry and country membership
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)

        # Populate industry membership if required
        if self.industryClassification is not None:
            self.generate_industry_exposures(
                    modelDate, modelDB, marketDB, data.exposureMatrix)

        # Populate country membership for regional models
        if len(self.rmg) > 1:
            self.generate_binary_country_exposures(
                    modelDate, modelDB, marketDB, data)
    
        # Generate universe of eligible assets
        data.eligibleUniverse = self.generate_eligible_universe(
                modelDate, data, modelDB, marketDB, assetTypes=self.estuAssetTypes)

        # Call model-specific estu construction routine
        excludeFactors = [self.descFactorMap[r.description] for r in self.nurseryRMGs]
        data.estimationUniverseIdx = self.generate_estimation_universe(
                modelDate, data, modelDB, marketDB, excludeFactors=excludeFactors)

        # Generate mid-cap/small-cap factor estus if any
        styleNames = [s.name for s in self.styles]
        scMap = dict()
        if self.estuMap is not None:
            for nm in self.estuMap.keys():
                # Sort relevant factors into order of importance
                if nm in styleNames:
                    scMap[self.estuMap[nm].id] = nm
            self.generate_smallcap_estus(
                    scMap, modelDate, data, modelDB, marketDB)

        # If there are nursery markets, generate their estimation universe
        if len(self.nurseryCountries) > 0:
            excludeFactors = [self.descFactorMap[r.description] \
                    for r in self.rmg if r not in self.nurseryRMGs]
            data.nurseryEstu = self.generate_non_core_estu(
                    modelDate, data, modelDB, marketDB, 'nursery',
                    data.nurseryUniverse, excludeFactors=excludeFactors)

        # Generate China Domestic ESTU if required
        if self.estuMap is not None:
            if 'ChinaA' in self.estuMap:
                self.generate_China_A_ESTU(modelDate, data, modelDB, marketDB)

        if self.debuggingReporting:
            # Output information on assets and their characteristics
            self.estimation_universe_reporting(modelDate, data, modelDB, marketDB)

        # Some reporting of stats
        mcap_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        mcap_ESTU = numpy.array(mcap_ESTU*100, int) / 100.0
        mcap_ESTU = ma.sum(mcap_ESTU)
        self.log.info('ESTU contains %d assets, %.2f tr %s market cap',
                      len(data.estimationUniverseIdx), mcap_ESTU / 1e12,
                      self.numeraire.currency_code)

        self.log.debug('generate_model_universe: end')
        return data

    def generate_estimation_universe(self, modelDate, data, modelDB, marketDB,
            excludeFactors=[]):
        """Generic estimation universe selection criteria for
        regional models.  Excludes assets:
            - smaller than a certain cap threshold, per country
            - smaller than a certain cap threshold, per industry
            - with insufficient returns over the history
        """
        self.log.info('generate_estimation_universe: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        originalEligibleUniverse = list(data.eligibleUniverse)
        originalEligibleUniverseIdx = [data.assetIdxMap[sid] for sid in originalEligibleUniverse]
        n = len(originalEligibleUniverse)
        logging.info('ESTU currently stands at %d stocks', n)

        # Remove nursery market assets
        if len(data.nurseryUniverse) > 0:
            logging.info('Checking for assets from nursery markets')
            ns_indices = [data.assetIdxMap[sid] for sid in data.nurseryUniverse]
            (eligibleUniverseIdx, nonest) = buildEstu.exclude_specific_assets(
                    ns_indices, baseEstu=originalEligibleUniverseIdx)
            if n != len(eligibleUniverseIdx):
                n = len(eligibleUniverseIdx)
                logging.info('ESTU currently stands at %d stocks', n)
        else:
            eligibleUniverseIdx = originalEligibleUniverseIdx

        # Report on thinly-traded assets over the entire universe
        logging.info('Looking for thinly-traded stocks')
        (nonSparseIdx, sparse) = buildEstu.exclude_thinly_traded_assets(
                                modelDate, data, baseEstu=universeIdx,
                                maskZeroWithNoADV=False, minNonZero=0.05)
        data.nonSparse = numpy.take(buildEstu.assets, nonSparseIdx, axis=0)

        # Exclude thinly traded assets
        estuIdx = list(set(eligibleUniverseIdx).intersection(set(nonSparseIdx)))
        if n != len(estuIdx):
            n = len(estuIdx)
            logging.info('ESTU currently stands at %d stocks', n)

        # Weed out tiny-cap assets by market
        lowerBound = 1
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (estuIdx0, nonest) = buildEstu.exclude_by_cap_ranking(
                data, modelDate, baseEstu=estuIdx,
                lower_pctile=lowerBound, method='percentage')
                #weight='rootCap')

        # Weed out tiny-cap assets by country
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on country', 100-lowerBound)
        (estuIdx1, nonest) = buildEstu.exclude_by_cap_ranking(
                data, modelDate, baseEstu=estuIdx,
                byFactorType=ExposureMatrix.CountryFactor,
                lower_pctile=lowerBound, method='percentage',
                excludeFactors=excludeFactors)
                #weight='rootCap')

        # Perform similar check by industry
        logging.info('Filtering by top %d%% mcap on industry', 100-lowerBound)
        (estuIdx2, nonest) = buildEstu.exclude_by_cap_ranking(
                data, modelDate, baseEstu=estuIdx,
                byFactorType=ExposureMatrix.IndustryFactor,
                lower_pctile=lowerBound, method='percentage',
                excludeFactors=excludeFactors)
                #weight='rootCap')

        estuIdx = set(estuIdx2).union(estuIdx1)
        estuIdx = list(estuIdx.union(estuIdx0))
        if n != len(estuIdx):
            n = len(estuIdx)
            logging.info('ESTU currently stands at %d stocks', n)

        # Inflate any thin countries or industries
        minFactorWidth=2*self.returnCalculator.allParameters[0].getThinFactorInformation().dummyThreshold
        logging.info('Inflating any thin factors')
        (estuIdx, nonest) = buildEstu.pump_up_factors(data, modelDate,
                currentEstu=estuIdx, baseEstu=eligibleUniverseIdx,
                byFactorType=[ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor],
                minFactorWidth=minFactorWidth, excludeFactors=excludeFactors)
        if n != len(estuIdx):
            n = len(estuIdx)
            logging.info('ESTU currently stands at %d stocks', n)

        # Apply grandfathering rules
        logging.info('Incorporating grandfathering')
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleUniverseIdx,
                estuInstance=self.estuMap['main'])

        totalcap = ma.sum(ma.take(data.marketCaps, estuIdx, axis=0), axis=0) / 1e9
        self.log.info('Final estu contains %d assets, %.2f bn (%s)',
                      len(estuIdx), totalcap, self.numeraire.currency_code)
        self.log.debug('generate_estimation_universe: end')

        # If we have a family of estimation universes, populate the main estu accordingly
        self.estuMap['main'].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap['main'].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]

        return estuIdx

    def generate_non_core_estu(self, modelDate, data, modelDB, marketDB,
            estuName, extraUniverse, excludeFactors=[]):
        """Generic estimation universe selection criteria for
        nursery or non-core markets.  Excludes assets:
            - smaller than a certain cap threshold, per country
            - with insufficient returns over the history
        """
        self.log.info('generate_nursery_estu: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in data.eligibleUniverse if sid in extraUniverse]
        n = len(eligibleUniverseIdx)
        logging.info('%s ESTU currently stands at %d stocks', estuName, n)

        # Exclude thinly traded assets if required
        nonSparseIdx = [data.assetIdxMap[sid] for sid in data.nonSparse]
        estuIdx = list(set(eligibleUniverseIdx).intersection(set(nonSparseIdx)))
        if n != len(estuIdx):
            logging.info('Removing illiquid assets')
            n = len(estuIdx)
            logging.info('%s ESTU currently stands at %d stocks', estuName, n)

        # Weed out tiny-cap assets by country
        (estuIdx, nonest) = buildEstu.exclude_by_cap_ranking(
                data, modelDate, baseEstu=estuIdx,
                byFactorType=ExposureMatrix.CountryFactor,
                lower_pctile=5, method='percentage',
                excludeFactors=excludeFactors)

        if n != len(estuIdx):
            n = len(estuIdx)
            logging.info('%s ESTU currently stands at %d stocks', estuName, n)

        # Inflate any thin countries
        (estuIdx, nonest) = buildEstu.pump_up_factors(data, modelDate,
                currentEstu=estuIdx, baseEstu=eligibleUniverseIdx,
                byFactorType=[ExposureMatrix.CountryFactor],
                excludeFactors=excludeFactors)
        if n != len(estuIdx):
            n = len(estuIdx)
            logging.info('%s ESTU currently stands at %d stocks', estuName, n)

        # Apply grandfathering rules
        (estuIdx, ESTUQualify, nonest) = buildEstu.grandfather(
                modelDate, estuIdx, baseEstu=eligibleUniverseIdx,
                estuInstance=self.estuMap[estuName])

        totalcap = ma.sum(ma.take(data.marketCaps, estuIdx, axis=0), axis=0) / 1e9
        self.log.info('Final %s estu contains %d assets, %.2f bn (%s)',
                      estuName, len(estuIdx), totalcap, self.numeraire.currency_code)
        self.estuMap[estuName].assets = [buildEstu.assets[idx] for idx in estuIdx]
        self.estuMap[estuName].qualify = [buildEstu.assets[idx] for idx in ESTUQualify]
        self.log.debug('generate_non_core_estu: end')

        return estuIdx

    def generate_smallcap_estus(self, scMap, modelDate, data, modelDB, marketDB):
        """Generates estimation universes based on market cap
        for use by midcap and smallcap factors
        """
        if len(scMap) < 1:
            return
        self.log.info('generate_smallcap_estus: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)

        # Set up various eligible and total universes
        universeIdx = list(range(len(buildEstu.assets)))
        subSetIdx = set(universeIdx)
        eligibleUniverse = list(self.estuMap['main'].assets)
        eligibleUniverseIdx = [data.assetIdxMap[sid] for sid in eligibleUniverse]
        eligibleCap = ma.filled(ma.take(
            data.issuerTotalMarketCaps, eligibleUniverseIdx, axis=0), 0.0)
        n = len(eligibleUniverse)
        logging.info('ESTU currently stands at %d stocks', n)

        idList = sorted(scMap.keys())

        # Loop round factors
        for idx in idList:
            sc = scMap[idx]
            tmpCap = ma.filled(ma.array(data.issuerTotalMarketCaps, copy=True), 0.0)
            logging.info('Generating %s universe, ID %d', sc, idx)

            # Mask everything outside the current percentile bounds
            params = self.styleParameters[sc]
            if not hasattr(params, 'bounds'):
                logging.warning('No cap bounds for factor: %s', sc)
                continue

            capBounds = Utilities.prctile(eligibleCap, params.bounds)
            if params.bounds[0] > 0.0:
                tmpCap = ma.masked_where(tmpCap <= capBounds[0], tmpCap)
            if params.bounds[1] < 100.0:
                tmpCap = ma.masked_where(tmpCap > capBounds[1], tmpCap)
            factorIdx = numpy.flatnonzero(ma.getmaskarray(tmpCap)==0)
            factorIdx = list(set(factorIdx).intersection(subSetIdx))

            # Perform grandfathering
            (estuSC, qualify, nonest) = buildEstu.grandfather(
                    modelDate, factorIdx,
                    baseEstu=list(subSetIdx),
                    estuInstance=self.estuMap[sc])

            scCap = ma.sum(ma.take(data.issuerTotalMarketCaps, estuSC, axis=0), axis=None)
            logging.info('%s universe contains %d assets, (min: %.4f, max: %.4f), %.2f Bn Cap',
                    sc, len(estuSC), capBounds[0]/1.0e9, capBounds[1]/1.0e9, scCap/1.0e9)
            subSetIdx = subSetIdx.difference(set(estuSC))
            self.estuMap[sc].assets = [buildEstu.assets[idx] for idx in estuSC]
            self.estuMap[sc].qualify = [buildEstu.assets[idx] for idx in qualify]

        self.log.info('generate_smallcap_estus: end')
        return

    def generate_China_A_ESTU(self, modelDate, data, modelDB, marketDB):
        """Returns the estimation universe assets and weights
        for use in secondary China A regression
        """
        self.log.info('generate_China_A_ESTU: begin')
        buildEstu = EstimationUniverse.ConstructEstimationUniverse(
                data.universe, self, modelDB, marketDB)

        # Remove assets from the exclusion table
        logging.info('...Applying exclusion table')
        (baseEstu, nonest) = buildEstu.apply_exclusion_list(modelDate)

        # Remove cloned assets
        if len(data.hardCloneMap) > 0:
            logging.info('...Removing cloned assets')
            cloneIdx = [data.assetIdxMap[sid] for sid in data.hardCloneMap.keys()]
            (baseEstu, nonest) = buildEstu.exclude_specific_assets(cloneIdx, baseEstu=baseEstu)

        # Pick out A-shares by asset type
        (baseEstu, nonest) = buildEstu.exclude_by_asset_type(
                modelDate, data, includeFields=['AShares'], excludeFields=None,
                baseEstu=baseEstu)
        logging.info('...%d China A shares found', len(baseEstu))
        n = len(baseEstu)

        # Exclude thinly traded assets
        logging.info('Looking for thinly-traded stocks')
        (aShares, sparse) = buildEstu.exclude_thinly_traded_assets(
                modelDate, data, baseEstu=baseEstu, maskZeroWithNoADV=False,
                minNonZero=0.05)
        if n != len(aShares):
            n = len(aShares)
            logging.info('A-share ESTU currently stands at %d stocks', n)

        # Weed out tiny-cap assets by market
        lowerBound = 5
        logging.info('Filtering by top %d%% mcap on entire market', 100-lowerBound)
        (aShares, nonest) = buildEstu.exclude_by_cap_ranking(
                data, modelDate, baseEstu=aShares,
                lower_pctile=lowerBound, method='percentage')
        if n != len(aShares):
            n = len(aShares)
            logging.info('A-share ESTU currently stands at %d stocks', n)

        # Perform grandfathering
        (aShares, qualify, nonest) = buildEstu.grandfather(
                modelDate, aShares, baseEstu=baseEstu,
                estuInstance=self.estuMap['ChinaA'])
        asMCap = ma.sum(ma.take(data.issuerTotalMarketCaps, aShares, axis=0), axis=None)

        logging.info('A-share estu contains %d assets, %.2f Bn Cap',
                len(aShares), asMCap/1.0e9)
        self.estuMap['ChinaA'].assets = [buildEstu.assets[idx] for idx in aShares]
        self.estuMap['ChinaA'].qualify = [buildEstu.assets[idx] for idx in qualify]

        self.log.info('generate_China_A_ESTU: end')
        return

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
        for name in self.estuMap.keys():
            idx = self.estuMap[name].id
            logging.info('Loading %s estimation universe, ID: %s', name, idx)
            self.estuMap[name].assets = modelDB.getRiskModelInstanceESTU(rmi_id, estu_idx=idx)
        if len(self.estuMap['main'].assets) > 0:
            estu = self.estuMap['main'].assets
        else:
            self.log.warning('Main estimation universe empty')
            estu = []
                
        # Ensure we have total coverage
        if data is not None:
            for nm in self.estuMap.keys():
                self.estuMap[nm].assets = [sid for sid in self.estuMap[nm].assets if sid in data.assetIdxMap]
                self.estuMap[nm].assetIdx = [data.assetIdxMap[sid] for sid in self.estuMap[nm].assets]

            estu = [sid for sid in estu if sid in data.assetIdxMap]
            data.estimationUniverseIdx = [data.assetIdxMap[sid] for sid in estu]
                
        assert(len(estu) > 0)
        return estu

    def standardizeExposures(self, exposureMatrix, data, modelDate, modelDB, marketDB,
                             subIssueGroups=None, writeStats=True):
        """Standardize exposures using Median Absolute Deviation (MAD)
        Required inputs are an ExposureMatrix object, ESTU indices,
        and market caps.  No return value; the ExposureMatrix is updated
        accordingly with the standardized values.
        """
        self.log.debug('standardizeExposures: begin')
        if self.debuggingReporting and writeStats:
            exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                       modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                       subIssueGroups=subIssueGroups)

        # Set MAD bounds and then standardize
        self.exposureStandardization.mad_bound = 8.0

        # Set weighting scheme for de-meaning
        self.exposureStandardization.capMean = True

        self.exposureStandardization.standardize(
                exposureMatrix, data.estimationUniverseIdx, 
                data.marketCaps, modelDate, writeStats,
                eligibleEstu=data.eligibleUniverseIdx)

        if self.debuggingReporting and writeStats:
            exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=subIssueGroups)

        self.log.debug('standardizeExposures: end')

    def shrink_to_mean(self, modelDate, data, modelDB, marketDB,
            descriptorName, historyLength, values, useSector=True, useRegion=True):
        """ Code for shrinking exposure values towards a specified value
        in cases of short history due to e.g. IPO
        """
        self.log.debug('shrink_to_mean: begin')

        # Load up from dates and determine scaling factor
        fromDates = modelDB.loadIssueFromDates([modelDate],  data.universe)
        distance = ma.array([int((modelDate - dt).days) for dt in fromDates], int)
        distance = ma.masked_where(distance>=historyLength, distance)
        nVals = len(numpy.flatnonzero(ma.getmaskarray(distance)==0))
        scaleFactor = ma.filled(distance / float(historyLength), 1.0)

        # Get cap weights to be used for mean
        estuArray = numpy.zeros((len(data.universe)), float)
        numpy.put(estuArray, data.estimationUniverseIdx, 1.0)
        mcapsEstu = ma.filled(data.marketCaps, 0.0) * estuArray
        fillHist = ma.filled(ma.masked_where(scaleFactor<1.0, scaleFactor), 0.0)
        mcapsEstu *= fillHist

        # Get sector/industry group exposures
        if useSector:
            level = 'Sectors'
        else:
            level = 'Industry Groups'
        sectorExposures = Utilities.buildGICSExposures(
                data.universe, modelDate, modelDB, level=level, clsDate=self.gicsDate)
        sectorExposures = ma.masked_where(sectorExposures<1.0, sectorExposures)

        # Bucket assets into regions/countries
        regionIDMap = dict()
        regionAssetMap = dict()
        if (not useRegion) or self.SCM:
            for r in self.rmg:
                regionIDMap[r.rmg_id] = r.mnemonic
                regionAssetMap[r.rmg_id] = data.rmgAssetMap[r.rmg_id]
        else:
            for r in self.rmg:
                rmg_assets = data.rmgAssetMap[r.rmg_id]
            if r.region_id not in regionAssetMap:
                regionAssetMap[r.region_id] = list()
                regionIDMap[r.region_id] = 'Region %d' % r.region_id
            regionAssetMap[r.region_id].extend(rmg_assets)

        # Compute mean of entire estimation universe to be used if insufficient
        # values in any region/sector bucket
        globalMean = ma.average(values, axis=0, weights=mcapsEstu)
        meanValue = globalMean * numpy.ones((len(data.universe)), float)

        # Loop round countries/regions
        for regID in regionIDMap.keys():

            # Get relevant assets and data
            rmgAssets = [sid for sid in regionAssetMap[regID]]
            if len(rmgAssets) < 1:
                logging.warning('No assets for %s', regionIDMap[regID])
                continue
            rmgAssetsIdx = [data.assetIdxMap[sid] for sid in rmgAssets]
            exposuresRegion = ma.take(sectorExposures, rmgAssetsIdx, axis=0)
            capsRegion = numpy.take(mcapsEstu, rmgAssetsIdx, axis=0)
            valuesRegion = ma.take(values, rmgAssetsIdx, axis=0)

            # Now loop round sector/industry group
            for i_sec in range(sectorExposures.shape[1]):

                # Pick out assets exposed to sector
                sectorColumn = exposuresRegion[:, i_sec]
                sectorIdx = numpy.flatnonzero(ma.getmaskarray(sectorColumn)==0)
                sectorAssetsIdx = numpy.take(rmgAssetsIdx, sectorIdx, axis=0)
                capsSector = numpy.take(capsRegion, sectorIdx, axis=0)
                valuesSector = ma.take(valuesRegion, sectorIdx, axis=0)

                # Compute mean and populate larger array
                nonZeroWts = numpy.flatnonzero(capsSector)
                if (len(valuesSector) > 0) and (len(nonZeroWts) > 0):
                    meanSector = ma.average(valuesSector, axis=0, weights=capsSector)
                    ma.put(meanValue, sectorAssetsIdx, meanSector)
                    logging.debug('Mean for %d values of %s in %s Sector %d: %.3f',
                            len(sectorIdx), descriptorName, regionIDMap[regID], i_sec, meanSector)
                else:
                    logging.info('No values of %s for %s Sector %d',
                            descriptorName, regionIDMap[regID], i_sec)

        # Shrink relevant values
        logging.info('Shrinking %d values of %s', nVals, descriptorName)
        values = (values * scaleFactor) + ((1.0 - scaleFactor) * ma.filled(meanValue, 0.0))
        self.log.debug('shrink_to_mean: end')
        return values

    def proxy_missing_exposures(self, modelDate, data, modelDB, marketDB,
                                factorNames=['Value', 'Growth', 'Leverage'],
                                clip=True, sizeVec=None, kappa=5.0,
                                legacy=False):
        """Fill-in missing exposure values for the factors given in
        factorNames by cross-sectional regression.  For each region,
        estimation universe assets are taken, and their exposure values
        regressed against their Size and industry (GICS sector) exposures.
        Missing values are then extrapolated based on these regression
        coefficients, and trimmed to lie within [-1.5, 1.5] to prevent the
        proxies taking on extreme values.
        """
        self.log.debug('proxy_missing_exposures: begin')
        expM = data.exposureMatrix

        # Prepare sector exposures, will be used for missing data proxies
        roots = self.industryClassification.getClassificationRoots(modelDB)
        root = [r for r in roots if r.name=='Sectors'][0]
        sectorNames = [n.description for n in self.\
                industryClassification.getClassificationChildren(root, modelDB)]
        sectorExposures = Matrices.ExposureMatrix(data.universe)
        e = self.industryClassification.getExposures(modelDate,
                data.universe, sectorNames, modelDB, level=-(len(roots)-1))
        sectorExposures.addFactors(sectorNames, e, ExposureMatrix.IndustryFactor)

        # Set up regression to proxy raw exposures for assets missing data
        rp = {'dummyReturnType': 'market',
              'dummyThreshold': 10,
              'k_rlm': kappa}
        proxyParameters = ModelParameters.RegressionParameters2012(rp)
        returnCalculator = LegacyFactorReturns.RobustRegressionLegacy([proxyParameters])
        returnCalculator.regParameters = returnCalculator.allParameters[0]
        dp = returnCalculator.allParameters[0].getThinFactorInformation()

        mat = expM.getMatrix()
        for f in factorNames:
            if (not hasattr(self.styleParameters[f], 'fillMissing')) or \
                    (self.styleParameters[f].fillMissing is not True):
                continue
            # Determine which assets are missing data and require proxying
            values = mat[expM.getFactorIndex(f),:]
            missingIndices = numpy.flatnonzero(ma.getmaskarray(values))
            estuSet = set(data.estimationUniverseIdx)
            self.log.info('%d/%d assets missing %s fundamental data (%d/%d ESTU)',
                        len(missingIndices), len(values), f,
                        len(estuSet.intersection(missingIndices)), len(estuSet))
            if len(missingIndices)==0:
                continue

            # Loop around regions
            for (regionName, asset_indices) in self.exposureStandardization.\
                    factorScopes[0].getAssetIndices(expM, modelDate):
                missing_indices = list(set(asset_indices).intersection(missingIndices))
                good_indicesUniv = set(asset_indices).difference(missingIndices)
                good_indices = list(good_indicesUniv.intersection(estuSet))
                propGood = len(good_indices) / float(len(missing_indices))
                if propGood < 0.1:
                    good_indices = list(good_indicesUniv.intersection(set(\
                            data.eligibleUniverseIdx)))
                    propGood = len(good_indices) / float(len(missing_indices))
                    if propGood < 0.1:
                        if len(missing_indices) > 0:
                            self.log.warning('Too few assets (%d) in %s with %s data present',
                                        len(good_indices), regionName, f)
                        continue

                good_data = ma.take(values, good_indices, axis=0)
                good_data = ma.masked_where(good_data==0, good_data)
                nonMissingData = numpy.flatnonzero(ma.getmaskarray(good_data)==0)
                if len(nonMissingData) == 0:
                    self.log.warning('All non-missing values are zero, skipping')
                    continue

                # Assemble regressand, regressor matrix and weights
                weights = numpy.take(data.marketCaps, good_indices, axis=0)**0.5
                regressand = ma.take(values, good_indices, axis=0)
                regressor = ma.zeros((len(good_indices), len(sectorNames) + 1))
                if sizeVec is None:
                    regressor[:,0] = ma.take(mat[expM.getFactorIndex(
                                    'Size'),:], good_indices, axis=0)
                else:
                    regressor[:,0] = ma.take(sizeVec, good_indices, axis=0)
                regressor[:,1:] = ma.transpose(ma.take(
                        sectorExposures.getMatrix(), good_indices, axis=1).filled(0.0))
                regressor = ma.transpose(regressor)
                # Set up thin factor correction
                dp.factorIndices = list(range(regressor.shape[0]))
                dp.factorNames = ['Size'] + sectorNames
                if legacy:
                    dp.factorTypes = [ExposureMatrix.StyleFactor] * len(dp.factorIndices)
                else:
                    dp.factorTypes = [ExposureMatrix.StyleFactor] + ([ExposureMatrix.IndustryFactor] * (len(dp.factorIndices)-1))
                dp.nonzeroExposuresIdx = [0]
                returnCalculator.computeDummyReturns(\
                        dp, ma.array(regressand), list(range(len(weights))), weights,
                        [data.universe[j] for j in good_indices], modelDate)
                returnCalculator.thinFactorParameters = dp

                # Run regression to get proxy values
                self.log.info('Running %s proxy regression for %s (%d assets)',
                        f, regionName, len(good_indices))
                coefs = returnCalculator.calc_Factor_Specific_Returns(
                        self, modelDate, list(range(regressor.shape[1])),
                        regressand, regressor, ['Size'] + sectorNames, weights, None,
                        returnCalculator.allParameters[0].getFactorConstraints()).factorReturns

                # Substitute proxies for missing values
                self.log.info('Proxying %d %s exposures for %s',
                        len(missing_indices), f, regionName)
                for i in range(len(sectorNames)):
                    regSectorExposures = sectorExposures.getMatrix()[i,missing_indices]
                    reg_sec_indices = numpy.flatnonzero(ma.getmaskarray(regSectorExposures)==0)
                    if len(reg_sec_indices)==0:
                        continue
                    reg_sec_indices = [missing_indices[j] for j in reg_sec_indices]
                    sectorCoef = coefs[i+1]
                    if sizeVec is None:
                        sizeExposures = mat[expM.getFactorIndex(
                                        'Size'), reg_sec_indices]
                    else:
                        sizeExposures = ma.take(sizeVec, reg_sec_indices, axis=0)
                    proxies = sizeExposures * coefs[0] + sectorCoef
                    if clip:
                        proxies = ma.where(proxies < -2.0, -2.0, proxies)
                        proxies = ma.where(proxies > 2.0, 2.0, proxies)
                    mat[expM.getFactorIndex(f),reg_sec_indices] = proxies

        self.log.debug('proxy_missing_exposures: end')
        return expM

    def loadDescriptorData(self, descList, descDict, date, subIssues, modelDB,
                                currencyAssetMap, rollOver=True):

        # Load the descriptor data
        okDescriptorCoverageMap = dict()
        returnDict = dict()

        for ds in descList:
            if ds in descDict:
                descID = descDict[ds]
                logging.info('Loading %s data', ds)
                if rollOver:
                    values = modelDB.loadDescriptorData(date, subIssues,
                            self.numeraire.currency_code, descID, rollOverData=90)
                else:
                    values = modelDB.loadDescriptorData(date, subIssues,
                            self.numeraire.currency_code, descID)

                values = values.data[:,-1]
                if ma.count(values) / float(len(subIssues)) < 0.05:
                    self.log.warning('Descriptor %s has only %3.3f percent unmasked values on date %s',
                            ds, ma.count(values) / float(len(subIssues)) * 100, date.strftime('%Y%m%d'))
                    okDescriptorCoverageMap[ds] = 0
                else:
                    okDescriptorCoverageMap[ds] = 1
                returnDict[ds] = values
            else:
                raise Exception('Undefined descriptor %s!' % ds)
        return returnDict, okDescriptorCoverageMap

    def generate_exposures(self, modelDate, data, modelDB, marketDB):
        """Compute multiple-descriptor style exposures
        for assets in data.universe using descriptors from the relevant
        descriptor_exposure table(s)
        """
        self.log.debug('generate_exposures: begin')
        descriptorExposures = Matrices.ExposureMatrix(data.universe)
        metaDescriptorExposures = Matrices.ExposureMatrix(data.universe)

        # Check for descriptors of descriptors
        if hasattr(self, 'MetaDescriptorMap'):
            hasMDs = True
            MetaDescriptorMap = self.MetaDescriptorMap
        else:
            hasMDs = False
            MetaDescriptorMap = dict()

        # Get list of all descriptors needed
        descriptors = []
        for f in self.DescriptorMap.keys():
            dsList = [ds for ds in self.DescriptorMap[f] if ds[-3:] != '_md']
            descriptors.extend(dsList)
        for f in MetaDescriptorMap.keys():
            descriptors.extend(self.MetaDescriptorMap[f])
        descriptors = sorted(set(descriptors))

        # Map descriptor names to their IDs
        descDict = dict(modelDB.getAllDescriptors())

        # Pull out a proxy for size
        sizeVec = self.loadDescriptorData(['LnIssuerCap'], descDict, modelDate, data.universe,
                modelDB, data.currencyAssetMap, rollOver=False)[0]['LnIssuerCap']
        missingSize = numpy.flatnonzero(ma.getmaskarray(sizeVec))
        if len(missingSize) > 0:
            missingIds = numpy.take(data.universe, missingSize, axis=0)
            missingIds = ','.join([sid.getSubIdString() for sid in missingIds])
            if self.forceRun:
                logging.warning('%d missing LnIssuerCaps', len(missingSize))
                logging.warning('Missing asset IDs: %s', missingIds)
            else:
                raise ValueError('%d missing LnIssuerCaps' % len(missingSize))

        # Load the descriptor data
        descValueDict, okDescriptorCoverageMap = self.loadDescriptorData(
                descriptors, descDict, modelDate, data.universe,
                modelDB, data.currencyAssetMap, rollOver=True)
        for ds in descriptors:
            if ds in descValueDict:
                descriptorExposures.addFactor(ds, descValueDict[ds], ExposureMatrix.StyleFactor)

        # Check that each exposure has at least one descriptor with decent coverage
        for expos in self.DescriptorMap.keys():
            numD = [okDescriptorCoverageMap[d] for d in self.DescriptorMap[expos]]
            numD = numpy.sum(numD, axis=0)
            if numD < 1:
                raise Exception('Factor %s has no descriptors with adequate coverage' % expos)
            logging.info('Factor %s has %d/%d  descriptors with adequate coverage',
                    expos, numD, len(self.DescriptorMap[expos]))

        # Add country factors to ExposureMatrix, needed for regional-relative standardization
        country_indices = data.exposureMatrix.\
                            getFactorIndices(ExposureMatrix.CountryFactor)
        if len(country_indices) > 0:
            countryExposures = ma.take(data.exposureMatrix.getMatrix(),
                                country_indices, axis=0)
            countryNames = data.exposureMatrix.getFactorNames(ExposureMatrix.CountryFactor)
            descriptorExposures.addFactors(countryNames, countryExposures,
                                ExposureMatrix.CountryFactor)
            metaDescriptorExposures.addFactors(countryNames, countryExposures,
                                ExposureMatrix.CountryFactor)

        subIssueGroups = modelDB.getIssueCompanyGroups(modelDate, data.universe, marketDB)
        if self.debuggingReporting:
            descriptorExposures.dumpToFile('tmp/raw-Desc-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx, dp=self.dplace,
                    subIssueGroups=subIssueGroups, assetType=data.assetTypeDict)

        # Decide which descriptors will be proxied and/or standardised at this stage
        proxyDescExclude = []
        proxyDescOldValue = dict()
        proxyStyleExclude = []
        for cf in self.styles:
            params = self.styleParameters[cf.name]
            if hasattr(params, 'dontProxy') and params.dontProxy:
                proxyStyleExclude.append(cf)
                proxyDescExclude.extend(self.DescriptorMap[cf.description])

        # Clone DR and cross-listing exposures if required
        scores = self.load_ISC_Scores(modelDate, data, modelDB, marketDB)
        self.clone_linked_asset_descriptors(modelDate, data, descriptorExposures,
                modelDB, marketDB, scores, subIssueGroups=subIssueGroups,
                excludeList=proxyDescExclude)

        # Standardize raw descriptors for multi-descriptor factors
        self.descriptorStandardization.standardize(descriptorExposures,
                                                   data.estimationUniverseIdx,
                                                   data.marketCaps, modelDate, writeStats=True,
                                                   eligibleEstu=data.eligibleUniverseIdx)

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

        # Form meta descriptors and standardise these
        mdDict = dict()
        mat = descriptorExposures.getMatrix()
        for md in MetaDescriptorMap.keys():
            mdDescriptors = MetaDescriptorMap[md]
            self.log.info('Meta descriptor %s has %d descriptor(s): %s',
                    md, len(mdDescriptors), mdDescriptors)
            valueList = [mat[descriptorExposures.getFactorIndex(d),:] for d in mdDescriptors]
            valueList = ma.array(valueList)

            # Weight descriptors if appropriate
            if md in self.DescriptorWeights:
                weights = self.DescriptorWeights[md]
                for (idx, w) in enumerate(weights):
                    valueList[idx,:] *= w

            if len(valueList) > 1:
                e = ma.average(valueList, axis=0)
            else:
                e = valueList[0,:]
            metaDescriptorExposures.addFactor(md, e, ExposureMatrix.StyleFactor)

        if hasMDs:
            self.descriptorStandardization.standardize(metaDescriptorExposures,
                                                       data.estimationUniverseIdx,
                                                       data.marketCaps, modelDate, writeStats=True,
                                                       eligibleEstu=data.eligibleUniverseIdx)

        # Form multi-descriptor CompositeFactors and add to ExposureMatrix
        mat = descriptorExposures.getMatrix()
        mat_meta = metaDescriptorExposures.getMatrix()
        for cf in self.styles:
            params = self.styleParameters[cf.description]
            if cf.description not in self.DescriptorMap:
                self.log.warning('No descriptors for factor: %s', cf.description)
                continue
            cfDescriptors = self.DescriptorMap[cf.description]
            self.log.info('Factor %s has %d descriptor(s): %s',
                    cf.description, len(cfDescriptors), cfDescriptors)
            valueList = []
            for d in cfDescriptors:
                if d in MetaDescriptorMap.keys():
                    valueList.append(mat_meta[metaDescriptorExposures.getFactorIndex(d),:])
                else:
                    valueList.append(mat[descriptorExposures.getFactorIndex(d),:])
            valueList = ma.array(valueList)

            if len(valueList) > 1:
                if hasattr(params, 'descriptorWeights'):
                    weights = params.descriptorWeights
                else:
                    weights = [1/float(len(valueList))] * len(valueList)
                e = ma.average(valueList, axis=0, weights=weights)
            else:
                e = valueList[0,:]
            data.exposureMatrix.addFactor(cf.description, e, ExposureMatrix.StyleFactor)

        # Proxy raw style exposures for assets missing data
        self.proxy_missing_exposures(modelDate, data, modelDB, marketDB,
                         factorNames=[cf.description for cf in self.styles if cf not in proxyStyleExclude],
                         sizeVec=sizeVec)

        # Generate other, model-specific factor exposures
        data.exposureMatrix = self.generate_model_specific_exposures(
                modelDate, data, modelDB, marketDB)

        self.log.debug('generate_exposures: end')
        return descriptorData

    def generate_currency_exposures(self, modelDate, modelDB, marketDB, data):
        """Generate binary currency exposures.
        """
        logging.debug('generate_currency_exposures: begin')
        allCurrencies = self.currencies + self.hiddenCurrencies

        expMatrix = data.exposureMatrix
        # Set up currency exposures array
        currencyExposures = Matrices.allMasked(
                (len(allCurrencies), len(data.universe)))
        currencyIdxMap = dict([(c.name, i) for (i,c) in enumerate(allCurrencies)])

        # Populate currency factors, one country at a time
        for rmg in self.rmg:
            rmg_assets = list(data.rmcAssetMap[rmg.rmg_id])
            asset_indices = [data.assetIdxMap[n] for n in rmg_assets]
            pos = currencyIdxMap[rmg.currency_code]
            values = currencyExposures[pos,:]
            if len(asset_indices) > 0:
                ma.put(values, asset_indices, 1.0)
                currencyExposures[pos,:] = values

        # Insert currency exposures into exposureMatrix
        data.exposureMatrix.addFactors([c.name for c in allCurrencies],
                        currencyExposures, ExposureMatrix.CurrencyFactor)
        currencies = sorted(currencyIdxMap.keys())
        logging.info('Computed currency exposures for %d currencies: %s',
                    len(list(currencyIdxMap.keys())), ','.join(currencies))
        logging.debug('generate_currency_exposures: end')
        return

    def generate_binary_country_exposures(self, modelDate, modelDB, marketDB, data):
        """Assign unit exposure to the country of quotation.
        """
        logging.debug('generate_binary_country_exposures: begin')
        countryExposures = Matrices.allMasked((len(self.rmg), len(data.universe)))
        rmgIdxMap = dict([(j,i) for (i,j) in enumerate(self.rmg)])

        for rmg in self.rmg:
            # Determine list of assets for each market
            rmg_assets = data.rmgAssetMap[rmg.rmg_id]
            indices = [data.assetIdxMap[n] for n in rmg_assets]
            logging.debug('Computing market exposures %s, %d assets (%s)',
                        rmg.description, len(indices), rmg.currency_code)
            values = Matrices.allMasked(len(data.universe))
            if len(indices) > 0:
                ma.put(values, indices, 1.0)
                countryExposures[rmgIdxMap[rmg],:] = values
        data.exposureMatrix.addFactors([r.description for r in self.rmg],
                            countryExposures, ExposureMatrix.CountryFactor)

        logging.debug('generate_binary_country_exposures: end')
        return

    def runModelTests(self, modelDate, modelDB, marketDB):
        """Runs assorted model tests on data etc.
        To be filled in as we go
        Provides a neater way of keeping some of our tests in one
        place
        """
        self.log.debug('runModelTests: begin')

        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi, returnExtra=True)
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                numeraire_id=self.numeraire.currency_id,
                forceRun=self.forceRun,
                legacyDates=self.legacyMCapDates,
                nurseryRMGList=self.nurseryRMGs)
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)

        # Test IBES sales data
        StyleExposures.test_ibes_data(
                modelDate, data, self, modelDB, marketDB, 'sales',
                useQuarterlyData=self.quarterlyFundamentalData)
        # Test IBES EPS data
        StyleExposures.test_ibes_data(
                modelDate, data, self, modelDB, marketDB, 'earnings',
                useQuarterlyData=self.quarterlyFundamentalData)
        return

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
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                numeraire_id=self.numeraire.currency_id,
                forceRun=self.forceRun, legacyDates=self.legacyMCapDates,
                nurseryRMGList=self.nurseryRMGs)
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        data.eligibleUniverse = self.generate_eligible_universe(
                modelDate, data, modelDB, marketDB)

        if self.SCM and not hasattr(self, 'indexSelector'):
            self.indexSelector = MarketIndex.\
                    MarketIndexSelector(modelDB, marketDB)
            self.log.info('Index Selector: %s', self.indexSelector)

        # Generate eligible universe
        data.eligibleUniverse = self.generate_eligible_universe(
                modelDate, data, modelDB, marketDB, assetTypes=self.estuAssetTypes)

        # Fetch trading calendars for all risk model groups
        # Start-date should depend on how long a history is required
        # for exposures computation
        data.rmgCalendarMap = dict()
        startDate = modelDate - datetime.timedelta(365*2)
        for rmg in self.rmg:
            data.rmgCalendarMap[rmg.rmg_id] = \
                    modelDB.getDateRange(rmg, startDate, modelDate)
        
        # Compute issuer-level market caps if required
        AssetProcessor.computeTotalIssuerMarketCaps(
                data, modelDate, self.numeraire, modelDB, marketDB,
                debugReport=self.debuggingReporting)

        if not self.SCM:
            self.generate_binary_country_exposures(modelDate, modelDB, marketDB, data)
            self.generate_currency_exposures(modelDate, modelDB, marketDB, data)

        # Generate 0/1 industry exposures
        self.generate_industry_exposures(
            modelDate, modelDB, marketDB, data.exposureMatrix)
        
        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB, data)
        
        # Create intercept factor
        if self.intercept is not None:
            beta = numpy.ones((len(data.universe)), float)
            data.exposureMatrix.addFactor(self.intercept.name, beta, ExposureMatrix.InterceptFactor)

        # Build all style exposures
        descriptorData = self.generate_exposures(modelDate, data, modelDB, marketDB)

        # Shrink some values where there is insufficient history
        for st in self.styles:
            params = self.styleParameters.get(st.name, None)
            if (params is None) or (not  hasattr(params, 'shrinkValue')):
                continue
            fIdx = data.exposureMatrix.getFactorIndex(st.name)
            values = data.exposureMatrix.getMatrix()[fIdx]
            # Check and warn of missing values
            missingIdx = numpy.flatnonzero(ma.getmaskarray(values))
            if len(missingIdx) > 0:
                missingSIDs = numpy.take(data.universe, missingIdx, axis=0)
                self.log.warning('%d assets have missing %s data', len(missingIdx), st)
                self.log.debug('Subissues: %s', missingSIDs)
            shrunkValues = self.shrink_to_mean(modelDate, data, modelDB, marketDB,
                    st.name, params.daysBack, values)
            data.exposureMatrix.getMatrix()[fIdx] = shrunkValues

        # Clone DR and cross-listing exposures if required
        subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        scores = self.load_ISC_Scores(modelDate, data, modelDB, marketDB)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores,
                subIssueGroups=subIssueGroups)
        
        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/raw-expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=subIssueGroups, assetType=data.assetTypeDict, dp=self.dplace)

        tmpDebug = self.debuggingReporting
        self.debuggingReporting = False
        self.standardizeExposures(data.exposureMatrix, data, modelDate, modelDB, marketDB, subIssueGroups)

        # Orthogonalise where required
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
            Utilities.partial_orthogonalisation(modelDate, data, modelDB, marketDB, orthogDict)
            tmpExcNames = list(self.exposureStandardization.exceptionNames)
            self.exposureStandardization.exceptionNames = [st.name for st in self.styles if st.name not in orthogDict]
            self.standardizeExposures(data.exposureMatrix, data, modelDate,
                        modelDB, marketDB, subIssueGroups)
            self.exposureStandardization.exceptionNames = tmpExcNames
        self.debuggingReporting = tmpDebug

        expMatrix = data.exposureMatrix.getMatrix()
        for st in self.styles:
            params = self.styleParameters[st.name]
            if hasattr(params, 'fillWithZero'):
                if (params.fillWithZero is True) and st.name in data.exposureMatrix.meanDict:
                    fIdx = data.exposureMatrix.getFactorIndex(st.name)
                    values = expMatrix[fIdx,:]
                    nMissing = numpy.flatnonzero(ma.getmaskarray(values))
                    fillValue = (0.0 - data.exposureMatrix.meanDict[st.name] / \
                            data.exposureMatrix.stdDict[st.name])
                    expMatrix[fIdx,:] = ma.filled(values, fillValue)
                    logging.info('Filling %d missing values for %s with standardised zero: %.2f',
                            len(nMissing), st.name, fillValue)

        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[sid] for sid in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=subIssueGroups, assetType=data.assetTypeDict, dp=self.dplace)

        # Check for exposures with all missing values
        for st in self.styles:
            fIdx = data.exposureMatrix.getFactorIndex(st.name)
            values = Utilities.screen_data(expMatrix[fIdx,:])
            nonMissingIdx = numpy.flatnonzero(ma.getmaskarray(values)==0)
            if len(nonMissingIdx) < 1:
                self.log.error('All %s values are missing', st)
                if not self.forceRun:
                    assert(len(nonMissingIdx)>0)
            
        self.log.debug('generateExposureMatrix: end')
        return [data, descriptorData]
    
    def assetReturnLoader(self, data, modelDate, modelDB, marketDB, buildFMP=False):
        """ Function to load in returns for factor regression
        """
        daysBack = 1
        returnsProcessor = ProcessReturns.assetReturnsProcessor(
                self.rmg, data.universe, data.rmgAssetMap,
                data.tradingRmgAssetMap, data.assetTypeDict,
                numeraire_id=self.numeraire.currency_id,
                tradingCurrency_id=self.numeraire.currency_id,
                returnsTimingID=self.returnsTimingId,
                debuggingReporting=self.debuggingReporting,
                gicsDate=self.gicsDate)
        assetReturnMatrix = returnsProcessor.process_returns_history(
                modelDate, daysBack, modelDB, marketDB,
                drCurrMap=data.drCurrData, loadOnly=True,
                applyRT=(self.SCM==False), trimData=False,
                useAllRMGDates=(self.SCM==False))
        return assetReturnMatrix

    def generateFactorSpecificReturns(self, modelDB, marketDB, modelDate,
            buildFMPs=False, internalRun=False):
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
        keepNTDsInRegression = True
        if buildFMPs:
            prevDate = modelDate
            if testFMPs:
                futureDates = modelDB.getDateRange(self.rmg, modelDate,
                        modelDate+datetime.timedelta(20), excludeWeekend=True)
                nextTradDate = futureDates[1]

            if not hasattr(self, 'fmpCalculator'):
                logging.warning('No FMP parameters set up, skipping')
                return None
            rcClass = self.fmpCalculator
        elif internalRun:
            return
        else:
            dateList = modelDB.getDates(self.rmg, modelDate, 1, excludeWeekend=True)
            dateList = dateList[-2:]
            if len(dateList) < 2:
                raise LookupError(
                    'No previous trading day for %s' %  str(modelDate))
            prevDate = dateList[0]
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
                checkHomeCountry=(self.SCM==0), numeraire_id=self.numeraire.currency_id,
                forceRun=self.forceRun, legacyDates=self.legacyMCapDates, nurseryRMGList=self.nurseryRMGs)

        # Generate eligible universe
        data.eligibleUniverse = self.generate_eligible_universe(
                prevDate, data, modelDB, marketDB, assetTypes=self.estuAssetTypes)

        # Load previous day's exposure matrix
        expM = self.loadExposureMatrix(rmi, modelDB, addExtraCountries=True, assetList=data.universe)
        prevFactors = self.factors + self.nurseryCountries
        prevSubFactors = modelDB.getSubFactorsForDate(prevDate, prevFactors)
        nameSubIDMap = dict([(s.factor.name, s.subFactorID) for s in prevSubFactors]) 

        data.exposureMatrix = expM
        data.eligibleUniverse = self.generate_eligible_universe(
                prevDate, data, modelDB, marketDB)

        # Get map of current day's factor IDs
        self.setFactorsForDate(modelDate, modelDB)
        allFactors = self.factors + self.nurseryCountries
        subFactors = modelDB.getSubFactorsForDate(modelDate, allFactors)
        subFactorIDIdxMap = dict([(subFactors[i].subFactorID, i)
                                for i in range(len(subFactors))])
        deadFactorNames = [s.factor.name for s in prevSubFactors if s not in subFactors]
        deadFactorIdx = [expM.getFactorIndex(n) for n in deadFactorNames]
        if len(deadFactorIdx) > 0:
            self.log.warning('Dropped factors %s on %s', deadFactorNames, modelDate)
        
        # Get main estimation universe for today
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        estu = self.loadEstimationUniverse(rmi, modelDB, data)
        estuIdx = [data.assetIdxMap[sid] for sid in estu]
        data.estimationUniverseIdx = estuIdx
        if self.estuMap is not None:
            if 'nursery' in self.estuMap:
                estu = estu + self.estuMap['nursery'].assets
                self.estuMap['main'].assets = estu
            estuIdx = [data.assetIdxMap[sid] for sid in estu]
            data.estimationUniverseIdx = estuIdx
            self.estuMap['main'].assetIdx = estuIdx

        # Load asset returns
        assetReturnMatrix = self.assetReturnLoader(
                data, modelDate, modelDB, marketDB, buildFMP=buildFMPs)
        assetReturnMatrix.data = assetReturnMatrix.data[:,-1][:,numpy.newaxis]
        assetReturnMatrix.dates = [assetReturnMatrix.dates[-1]]
        missingReturns = ma.getmaskarray(assetReturnMatrix.data)[:,0]
        if keepNTDsInRegression:
            assetReturnMatrix.data = ma.filled(assetReturnMatrix.data, 0.0)

        # Compute excess returns
        badRets = ma.masked_where(abs(assetReturnMatrix.data)<1.0e-12, assetReturnMatrix.data)
        badRets = numpy.flatnonzero(ma.getmaskarray(ma.take(badRets, estuIdx, axis=0)))
        self.log.info('%.1f%% of main ESTU returns missing or zero',
                100.0 * len(badRets) / float(len(estuIdx)))
        (assetReturnMatrix, rfr) = self.computeExcessReturns(modelDate, 
                                assetReturnMatrix, modelDB, marketDB, data.drCurrData)

        for i in range(len(rfr.assets)):
            if rfr.data[i,0] is not ma.masked:
                self.log.debug('Using risk-free rate of %f%% for %s',
                        rfr.data[i,0] * 100.0, rfr.assets[i])
        excessReturns = assetReturnMatrix.data[:,0]
        
        # FMP testing stuff
        if nextTradDate is not None:
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

        # If fewer than 5% of a market's ESTU assets have returns
        # consider it non-trading and remove from regression
        nonTradingMarketsIdx = []
        for r in self.rmg:
            rmg_indices = [data.assetIdxMap[n] for n in \
                            data.rmgAssetMap[r.rmg_id].intersection(data.eligibleUniverse)]
            rmg_returns = ma.take(excessReturns, rmg_indices, axis=0)
            if self.debuggingReporting:
                medReturn = ma.masked_where(abs(rmg_returns) < 1.0e-12, rmg_returns)
                medReturn = ma.median(abs(medReturn))
                self.log.info('Date: %s, RMG: %s, Median Absolute Return: %s',
                        modelDate, r.mnemonic, ma.filled(medReturn, 0.0))
            noReturns = numpy.sum(ma.getmaskarray(rmg_returns))
            allZeroReturns = ma.masked_where(abs(rmg_returns) < 1.0e-12, rmg_returns)
            allZeroReturns = numpy.sum(ma.getmaskarray(allZeroReturns))
            rmgCalendarList = modelDB.getDateRange(r,
                    assetReturnMatrix.dates[0], assetReturnMatrix.dates[-1])
            if noReturns >= 0.95 * len(rmg_returns) or modelDate not in rmgCalendarList:
                self.log.info('Non-trading day for %s, %d/%d returns missing',
                              r.description, noReturns, len(rmg_returns))
                if not keepNTDsInRegression:
                    if expM.checkFactorType(r.description, ExposureMatrix.CountryFactor):
                        fIdx = expM.getFactorIndex(r.description)
                        nonTradingMarketsIdx.append(fIdx)
            elif allZeroReturns == len(rmg_returns):
                self.log.warning('All returns are zero for %s' \
                        % r.description)

        # Get industry asset buckets
        data.industryAssetMap = dict()
        for idx in expM.getFactorIndices(ExposureMatrix.IndustryFactor):
            assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
            data.industryAssetMap[idx] = numpy.take(data.universe, assetsIdx, axis=0)

        # Get indices of factors that we don't want in the regression
        if not self.SCM:
            currencyFactorsIdx = expM.getFactorIndices(ExposureMatrix.CurrencyFactor)
            excludeFactorsIdx = set(deadFactorIdx + currencyFactorsIdx + nonTradingMarketsIdx)
        else:
            excludeFactorsIdx = deadFactorIdx

        # Remove any remaining empty style factors
        for idx in expM.getFactorIndices(ExposureMatrix.StyleFactor):
            assetsIdx = numpy.flatnonzero(expM.getMatrix()[idx,:])
            if len(assetsIdx) == 0:
                self.log.warning('100%% empty factor, excluded from all regressions: %s', expM.getFactorNames()[idx])
                excludeFactorsIdx.append(idx)
            else:
                propn = len(assetsIdx) / float(len(data.universe))
                if propn < 0.01:
                    self.log.warning('%.1f%% exposures non-missing, excluded from all regressions: %s',
                            100*propn, expM.getFactorNames()[idx])
                    excludeFactorsIdx.append(idx)

        # Call nested regression routine
        returnData = rcClass.run_factor_regressions(
                self, rcClass, prevDate, excessReturns, expM, estu, data,
                excludeFactorsIdx, modelDB, marketDB, robustWeightList=[])

        # Map specific returns for cloned assets
        returnData.specificReturns = ma.masked_where(missingReturns, returnData.specificReturns)
        if len(data.hardCloneMap) > 0:
            cloneList = [n for n in data.universe if n in data.hardCloneMap]
            for sid in cloneList:
                if data.hardCloneMap[sid] in data.universe:
                    returnData.specificReturns[data.assetIdxMap[sid]] = returnData.specificReturns\
                            [data.assetIdxMap[data.hardCloneMap[sid]]]

        # This here as a reminder to do this eventually
        #specificReturns = self.adjustDRSpecificReturnsForTiming(
        #        date, returns, modelDB, marketDB,
        #        adjustments, ids_DR, rmgList)

        # Store regression results
        factorReturns = Matrices.allMasked((len(allFactors),))
        regressionStatistics = Matrices.allMasked((len(allFactors), 4))
        for (fName, ret) in returnData.factorReturnsMap.items():
            idx = subFactorIDIdxMap.get(nameSubIDMap[fName], None)
            if idx is not None:
                factorReturns[idx] = ret
                regressionStatistics[idx,:] = returnData.regStatsMap[fName]
        result = Utilities.Struct()
        result.universe = data.universe
        result.factorReturns = factorReturns
        result.specificReturns = returnData.specificReturns
        result.exposureMatrix = expM
        result.regressionStatistics = regressionStatistics
        result.adjRsquared = returnData.anova.calc_adj_rsquared()
        result.regression_ESTU = list(zip([result.universe[i] for i in returnData.anova.estU_],
                returnData.anova.weights_ / numpy.sum(returnData.anova.weights_)))
        result.VIF = self.VIF

        # Process robust weights
        newRWtMap = dict()
        sid2StringMap = dict([(sid.getSubIDString(), sid) for sid in data.universe])
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
        if len(nonTradingMarketsIdx) > 0:
            nonTradingMarketNames = ', '.join([allFactorNames[i] \
                    for i in nonTradingMarketsIdx])
            self.log.info('%d non-trading market(s): %s',
                    len(nonTradingMarketsIdx), nonTradingMarketNames)
            for i in nonTradingMarketsIdx:
                idx = subFactorIDIdxMap[nameSubIDMap[allFactorNames[i]]]
                result.factorReturns[idx] = 0.0

        # Pull in currency factor returns from currency model
        if not self.SCM:
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
            assert (crmi is not None)
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
                        self.log.warning('Missing currency factor return for %s',
                                        allFactors[j].name)
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
                            modelDate, buildFMPs=buildFMPs, constrComp=constrComp)
        else:
            self.regressionReporting(excessReturns, result, expM, nameSubIDMap, data.assetIdxMap,
                            modelDate, buildFMPs=buildFMPs, constrComp=constrComp,
                            specificRets=result.specificReturns)

        if self.debuggingReporting:
            for (i,sid) in enumerate(data.universe):
                if abs(returnData.specificReturns[i]) > 1.5:
                    self.log.warning('Large specific return for: %s, ret: %s',
                            sid, returnData.specificReturns[i])
        return result
    
    def adjustFactorReturnsForTiming(self, modelDate, factorReturns,
                                     adjustments, modelDB, marketDB):
        """Sets up an array of returns-timing adjustment factors that
        will be added to the factor returns time-series.
        Both factorReturns and adjustments should be TimeSeriesMatrix
        objects containing the factor returns and market adjustment
        factors time-series, respectively.  The adjustments array
        cannot contain masked values.
        Returns the factor returns TimeSeries matrix with an extra
        'adjust' attribute appended.
        """
        if not self.usesReturnsTimingAdjustment():
            return factorReturns
        self.log.debug('adjustFactorReturnsForTiming: begin')
        factorIdxMap = dict([(f.factor.name, i) for (i, f)
                             in enumerate(factorReturns.assets)])
        rmgList = [rmg for rmg in self.rmg if rmg not in self.nurseryRMGs]
        rmgIndices = [factorIdxMap[rmg.description] for rmg in rmgList]
        adjustArray = numpy.zeros(factorReturns.data.shape[0:2], float)
        # Only keep adjustments relevant to model geographies
        rmgIdxMap = dict([(j.rmg_id,i) for (i,j) in \
                            enumerate(adjustments.assets)])
        modelCountryAdjustments = numpy.take(adjustments.data,
                            [rmgIdxMap[r.rmg_id] for r in rmgList], axis=0)
        # Weighted country factor return should sum to zero
        # so take out the weighted average and add it to the global term
        dateList = factorReturns.dates
        dateLen = len(dateList)
        weights = modelDB.loadRMSFactorStatisticHistory(self.rms_id,
                    factorReturns.assets, dateList, 'regr_constraint_weight')
        weights = Utilities.screen_data(weights.data, fill=True)
        marketAdjTerm = numpy.sum(weights[rmgIndices,:] * \
                    modelCountryAdjustments[:,:dateLen], axis=0)
        adjustArray[factorIdxMap[self.intercept.description],:] = marketAdjTerm
        for (ii, idx) in enumerate(rmgIndices):
            adjustArray[idx,:] = modelCountryAdjustments[ii,:dateLen] - marketAdjTerm

        factorReturns.adjust = adjustArray
        if self.debuggingReporting:
            adjNames = [f.factor.name for f in factorReturns.assets]
            dateFieldNames = [str(d) for d in dateList]
            outData = numpy.transpose(adjustArray)
            adjOutFile = 'tmp/%s-adjHist.csv' % self.name
            Utilities.writeToCSV(outData, adjOutFile, columnNames=adjNames,
                                 rowNames=dateFieldNames)

        self.log.debug('adjustFactorReturnsForTiming: end')
        return factorReturns

    def adjustDRSpecificReturnsForTiming(self, modelDate, specificReturns,
                                         modelDB, marketDB, adjustments,
                                         ids_DR, rmgList, data):
        """If DR-like instruments are present in the universe, adjust their
        specific returns for returns-timing.  Both specificReturns and
        adjustments should be TimeSeriesMatrix objects containing the
        specific returns and market adjustment factors time-series,
        respectively.  The adjustments array cannot contain masked values.
        Returns the adjusted specific returns.
        """
        self.log.debug('adjustDRSpecificReturnsForTiming: begin')
        drSet = set(ids_DR)
        subIssues = specificReturns.assets

        # Determine DRs' home country versus country of quotation
        homeCountryMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                        data.rmgAssetMap.items() for sid in ids \
                        if sid in drSet])
        tradingCountryMap = dict([(sid, rmg.rmg_id) for (sid, rmg) in \
                        modelDB.getSubIssueRiskModelGroupPairs(
                        modelDate, restrict=list(ids_DR))])
        rmgIdxMap = dict([(j.rmg_id,i) for (i,j) in \
                            enumerate(adjustments.assets)])

        # Set up market time zones
        rmgZoneMap = dict((rmg.rmg_id, rmg.gmt_offset) for rmg in rmgList)

        # Remove adjustment from home country, add from trading country
        dateLen = len(specificReturns.dates)
        self.log.info('Adjusting specific returns for %s assets' % len(ids_DR))
        sameTZ = []
        for i in range(len(ids_DR)):
            sid = ids_DR[i]
            homeIdx = homeCountryMap.get(sid, None)
            tradingIdx = tradingCountryMap.get(sid, None)
            if homeIdx is None or tradingIdx is None:
                continue
            if abs(rmgZoneMap[homeIdx] - rmgZoneMap[tradingIdx]) <= 3:
                sameTZ.append(sid)
            else:
                idx = data.assetIdxMap[sid]
                srAdjust = adjustments.data[rmgIdxMap[homeIdx],:dateLen] \
                        - adjustments.data[rmgIdxMap[tradingIdx],:dateLen]
                specificReturns.data[idx,:] -= srAdjust
        self.log.info('%s assets in same time-zone, not adjusted' % len(sameTZ))

        self.log.debug('adjustDRSpecificReturnsForTiming: end')
        return specificReturns


class StatisticalModel(FactorRiskModelv3):
    """Statistical factor model
    """
    # ZZZZZZ
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        # No style or industry factors
        self.styles = []
        self.industries = []
        self.estuMap = None
        self.legacyISCSwitchDate = datetime.date(1960, 1, 1)
        
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
        if len(self.rmg) < 2:
            self.SCM = True
        else:
            self.SCM = False

        # Setup industry classification
        chngDate = list(self.industrySchemeDict.keys())[0]
        chngDates = sorted(d for d in self.industrySchemeDict.keys() if d <= date)
        self.industryClassification = self.industrySchemeDict[chngDates[-1]]
        self.log.debug('Using %s classification scheme', self.industryClassification.name)

        # Create currency factors
        self.currencies = []
        # Add additional currency factors (to allow numeraire changes) if necessary
        if self.allCurrencies:
            self.allRMG = modelDB.getAllRiskModelGroups(inModels=True)
            for rmg in self.allRMG:
                rmg.setRMGInfoForDate(date)
            additionalCurrencyFactors = [ModelFactor(f, None)
                    for f in set([r.currency_code for r in self.allRMG])]
            additionalCurrencyFactors.extend([ModelFactor('EUR', 'Euro')])
            self.additionalCurrencyFactors = [f for f in additionalCurrencyFactors
                    if f not in self.currencies and f.name in self.nameFactorMap]
            self.additionalCurrencyFactors = list(set(self.additionalCurrencyFactors))

        if not self.SCM:
            self.currencies = [ModelFactor(f, None)
                    for f in set([r.currency_code for r in self.rmg])]
            # Add additional currency factors
            self.currencies.extend([f for f in self.additionalCurrencyFactors
                                    if f not in self.currencies
                                    and f.isLive(date)])
        self.currencies = sorted(self.currencies)
                                
        allFactors = self.blind + self.currencies
        for f in allFactors:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factors = allFactors
        self.validateFactorStructure(date)
    
    def insertEstimationUniverse(self, rmi, modelDB):
        """Inserts the estimation universes into the database for the given
        risk model instance.
        estuMap is a dict of estu objects, mapped by name
        """
        for estuName in self.baseModel.estuMap.keys():
            if hasattr(self.baseModel.estuMap[estuName], 'assets'):
                modelDB.insertEstimationUniverseV3(rmi, self.baseModel.estuMap[estuName], estuName)

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
        for name in self.estuMap.keys():
            idx = self.estuMap[name].id
            logging.info('Loading %s estimation universe, ID: %s', name, idx)
            self.estuMap[name].assets = modelDB.getRiskModelInstanceESTU(rmi_id, estu_idx=idx)
        if len(self.estuMap['main'].assets) > 0:
            estu = self.estuMap['main'].assets
        else:
            self.log.warning('Main estimation universe empty')
            estu = []

        # Ensure we have total coverage
        if data is not None:
            for nm in self.estuMap.keys():
                self.estuMap[nm].assets = [sid for sid in self.estuMap[nm].assets if sid in data.assetIdxMap]
                self.estuMap[nm].assetIdx = [data.assetIdxMap[sid] for sid in self.estuMap[nm].assets]

            estu = [sid for sid in estu if sid in data.assetIdxMap]
            data.estimationUniverseIdx = [data.assetIdxMap[sid] for sid in estu]

        assert(len(estu) > 0)
        return estu

    def generate_model_universe(self, modelDate, modelDB, marketDB):
        """Generate risk model instance universe and estimation
        universe.  Return value is a Struct containing a universe
        attribute (list of SubIssues) containing the model universe
        and an estimationUniverse attribute (list of index positions
        corresponding to ESTU members).
        """
        self.log.debug('generate_model_universe: begin')
        # Get basic risk model instance universe
        assert(self.baseModel is not None)
        self.baseModel.setFactorsForDate(modelDate, modelDB)
         
        universe = AssetProcessor.getModelAssetMaster(
                self, modelDate, modelDB, marketDB)

        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                legacyDates=self.legacyMCapDates,
                numeraire_id=self.numeraire.currency_id,
                forceRun=self.forceRun)

        # Build a temporary exposure matrix to house things
        # like industry and country membership
        data.exposureMatrix = Matrices.ExposureMatrix(universe)

        # Populate industry membership if required
        if self.industryClassification is not None:
            self.generate_industry_exposures(
                    modelDate, modelDB, marketDB, data.exposureMatrix)

        # Populate country membership for regional models
        if len(self.rmg) > 1:
            self.generate_binary_country_exposures(
                    modelDate, modelDB, marketDB, data)

        # Generate universe of eligible assets
        if hasattr(self.baseModel, 'generate_eligible_universe'):
            data.eligibleUniverse = self.baseModel.generate_eligible_universe(
                    modelDate, data, modelDB, marketDB)
             
        # Call model-specific estu construction routine
        data.estimationUniverseIdx = self.baseModel.generate_estimation_universe(
                modelDate, data, modelDB, marketDB)

        # Prune basemodel estu dict
        for key in list(self.baseModel.estuMap.keys()):
            if key != 'main':
                del self.baseModel.estuMap[key]

        # Some reporting of stats
        mcap_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        if self.debuggingReporting:
            sidList = [universe[idx].getSubIDString() for idx in data.estimationUniverseIdx]
            outName = 'tmp/estu-%s-%s.csv' % (self.mnemonic, modelDate)
            Utilities.writeToCSV(mcap_ESTU, outName, rowNames=sidList)

        mcap_ESTU = ma.sum(mcap_ESTU)
        self.log.info('ESTU contains %d assets, %.2f tr %s market cap',
                      len(data.estimationUniverseIdx), mcap_ESTU / 1e12,
                      self.numeraire.currency_code)

        self.log.debug('generate_model_universe: end')
        return data

    def compute_EP_statistic(self, assetReturns, specificReturns, estu, de_mean=True):
        """Computes model EP statistic and 'averaged R-squared' as
        defined in Connor (1995)
        """
        assetReturns = ma.take(assetReturns, estu, axis=0)
        specificReturns = ma.take(specificReturns, estu, axis=0)
        if de_mean:
            assetReturns = ma.transpose(ma.transpose(assetReturns) - \
                            ma.average(assetReturns, axis=1))
        numerator = numpy.sum([ma.inner(e,e) for e in specificReturns], axis=0)
        denominator = numpy.sum([ma.inner(r,r) for r in assetReturns], axis=0)
        ep = 1.0 - numerator/denominator
        self.log.info('EP statistic: %f', ep)
        sse = [float(ma.inner(e,e)) for e in ma.transpose(specificReturns)]
        sst = [float(ma.inner(r,r)) for r in ma.transpose(assetReturns)]
        sst = ma.masked_where(sst==0.0, sst)
        sst = 1.0 / sst
        avg_r2 = 1.0 - ma.inner(sse, sst.filled(0.0)) / len(sse)
        self.log.info('Average R-Squared: %f', avg_r2)
        return (ep, avg_r2)
    
    def generateStatisticalModel(self, modelDate, modelDB, marketDB):
        """Compute statistical factor exposures and returns
        Then combine returns with currency returns and build the
        composite covariance matrix
        """

        # Set up covariance parameters
        (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
        (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
        (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
        (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        self.returnHistory = maxOmegaObs
 
        # Determine home country info and flag DR-like instruments
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi)
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                legacyDates=self.legacyMCapDates,
                numeraire_id=self.numeraire.currency_id,
                forceRun=self.forceRun)

        # Get list of assets not to proxy
        if hasattr(self, 'legacyModel'):
            noProxyType = []
        else:
            noProxyType = [typ.lower() for typ in self.noProxyType]
        data.noProxyList = [sid for sid in data.universe \
                if data.assetTypeDict[sid].lower() in noProxyType]
        if len(data.noProxyList) > 0:
            logging.info('%d assets of type %s excluded from proxying',
                    len(data.noProxyList), self.noProxyType)

        # Build exposure matrix with currency factors
        data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
        if len(self.currencies) > 0:
            self.addCurrencyFactorExposures(modelDate, data, modelDB, marketDB)

        # Load estimation universe
        estu = self.loadEstimationUniverse(rmi, modelDB, data)

        # Compute excess returns
        data = self.build_excess_return_history_v3(
                data, modelDate, modelDB, marketDB)

        # Trim estimation universe to include only assets with 95%
        # of returns existant
        okIdx = [idx for (idx, nret) in enumerate(data.numOkReturns) if nret > 0.95] 
        okIds = numpy.take(data.universe, okIdx, axis=0)
        newEstu = [sid for sid in estu if sid in okIds]
        diff = set(estu).difference(set(newEstu))
        if len(diff) > 0:
            logging.info('Dropping %d assets from estu with more than 5%% of returns missing',
                    len(diff))
            estu = newEstu
            data.estimationUniverseIdx = [data.assetIdxMap[sid] for sid in estu]

        # Remove dates on which lots of assets don't trade
        io = numpy.sum(ma.getmaskarray(data.returns.data), axis=0)
        goodDatesIdx = numpy.flatnonzero(io < 0.7 * len(data.returns.assets))
        badDatesIdx = [i for i in range(len(data.returns.dates)) if \
                i not in set(goodDatesIdx) and data.returns.dates[i].weekday() <= 4]
        if len(badDatesIdx) > 0:
            self.log.debug('Omitting weekday dates: %s',
                    ','.join([str(data.returns.dates[i])
                        for i in badDatesIdx]))
        keep = min(len(goodDatesIdx), self.returnHistory)
        goodDatesIdx = goodDatesIdx[-keep:]
        data.returns.dates = [data.returns.dates[i] for i in goodDatesIdx]
        data.returns.data = ma.take(data.returns.data, goodDatesIdx, axis=1)
        data.preIPOFlag = ma.take(data.preIPOFlag, goodDatesIdx, axis=1)
        data.maskedData = ma.take(data.maskedData, goodDatesIdx, axis=1)

        # Trim outliers along region/sector buckets
        if hasattr(self, 'legacyModel'):
            data.clippedReturns = Utilities.bucketedMAD(
                    self.rmg, modelDate, data.returns.data, data, modelDB,
                    nDev=[3.0, 3.0], MADaxis=0, gicsDate=self.gicsDate)
        else:
            opms = dict()
            opms['nBounds'] = [3.0, 3.0]
            outlierClass = Outliers.Outliers(opms, industryClassificationDate=self.gicsDate)
            data.clippedReturns = outlierClass.bucketedMAD(
                    self.rmg, modelDate, data.returns.data, data, modelDB, axis=0,
                    gicsDate=self.gicsDate)

        if self.debuggingReporting:
            dates = [str(d) for d in data.returns.dates]
            idList = [s.getSubIDString() for s in data.universe]
            retOutFile = 'tmp/%s-retHist.csv' % self.name
            Utilities.writeToCSV(data.returns.data, retOutFile, rowNames=idList,
                    columnNames=dates)
            retOutFile = 'tmp/%s-retHist-clipped.csv' % self.name
            Utilities.writeToCSV(data.clippedReturns, retOutFile, rowNames=idList,
                    columnNames=dates)

        # Downweight some countries if required
        for r in [r for r in self.rmg if r.downWeight < 1.0]:
            for sid in data.rmgAssetMap[r.rmg_id].intersection(estu):
                data.clippedReturns[data.assetIdxMap[sid],:] *= r.downWeight

        # Compute exposures, factor and specific returns
        self.log.debug('Computing exposures and factor returns: begin')
        originalData = ma.array(data.returns.data, copy=True)
        data.returns.data = ma.array(data.clippedReturns, copy=True)

        # Build Statistical model of returns
        (expMatrix, factorReturns, specificReturns0, regressANOVA, pctgVar) = \
            self.returnCalculator.calc_ExposuresAndReturns(
                data.returns, data.estimationUniverseIdx, T=self.pcaHistory)
        data.returns.data = originalData

        # Add statistical factor exposures to exposure matrix
        factorNames = [f.name for f in self.blind]
        subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        data.exposureMatrix.addFactors(factorNames, numpy.transpose(expMatrix),
                ExposureMatrix.StatisticalFactor)
        scores = self.load_ISC_Scores(modelDate, data, modelDB, marketDB)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores,
                subIssueGroups=data.subIssueGroups)
        
        if self.debuggingReporting:
            dr_indices = [data.assetIdxMap[n] for n in data.foreign]
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, estu=data.estimationUniverseIdx,
                    subIssueGroups=data.subIssueGroups, dp=self.dplace)
            dates = [str(d) for d in data.returns.dates]
            retOutFile = 'tmp/%s-facretHist-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(factorReturns, retOutFile, rowNames=factorNames,
                    columnNames=dates, dp=8)

        # Compute 'real' specific returns using non-clipped returns
        exposureIdx = [data.exposureMatrix.getFactorIndex(n) for n in factorNames]
        expMatrix = numpy.transpose(ma.take(data.exposureMatrix.getMatrix(),
                            exposureIdx, axis=0))
        specificReturns = data.returns.data - numpy.dot(ma.filled(expMatrix, 0.0), factorReturns)

        # Map specific returns for cloned assets
        if len(data.hardCloneMap) > 0:
            cloneList = [n for n in data.universe if n in data.hardCloneMap]
            for sid in cloneList:
                if data.hardCloneMap[sid] in data.universe:
                    specificReturns[data.assetIdxMap[sid],:] = specificReturns\
                            [data.assetIdxMap[data.hardCloneMap[sid]],:]
        
        # Compute various regression statistics
        data.regressionStatistics = Matrices.allMasked((len(self.factors), 4))
        regressStats = regressANOVA.calc_regression_statistics(
                            factorReturns[:,-1], ma.take(
                            expMatrix, regressANOVA.estU_, axis=0))
        data.regressionStatistics[:len(self.blind),:3] = regressStats
        data.regression_ESTU = list(zip([data.universe[i] for i in regressANOVA.estU_], 
                        regressANOVA.weights_ / numpy.sum(regressANOVA.weights_)))
        (ep, avg_r2) = self.compute_EP_statistic(
                        data.clippedReturns, specificReturns0, data.estimationUniverseIdx)
        
        # Compute root-cap-weighted R-squared to compare w/ cross-sectional model
        # Note that this is computed over the initial ESTU, not the 'real'
        # one going into the factor analysis
        weights = ReturnCalculator.calc_Weights(self.rmg, modelDB, marketDB,
                    modelDate, [data.universe[i] for i in data.estimationUniverseIdx],
                    self.numeraire.currency_id)
        
        regANOVA = ReturnCalculator.RegressionANOVA(data.clippedReturns[:,-1],
                specificReturns0[:,-1], self.numFactors, data.estimationUniverseIdx, weights)
        data.adjRsquared = regANOVA.calc_adj_rsquared()
        data.pctgVar = pctgVar
        self.log.debug('Computing exposures and factor returns: end')
        
        # Create TimeSeriesMatrix objects for factor and specific returns
        self.log.debug('Building time-series matrices: begin')
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
         
        # Check returns history lengths
        # Note that dates are switched to reverse chronological order
        dateList = data.returns.dates
        dateList.reverse()
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        omegaObs = min(len(dateList), maxOmegaObs)
        deltaObs = min(len(dateList), maxDeltaObs)
        self.log.info('Using %d of %d days of factor return history',
                      omegaObs, len(dateList))
        self.log.info('Using %d of %d days of specific return history',
                      deltaObs, len(dateList))
        
        # Set up specific returns history matrix
        data.srMatrix = Matrices.TimeSeriesMatrix(
                                data.returns.assets, dateList[:deltaObs])
        # Mask specific returns corresponding to missing returns
        data.srMatrix.data = ma.masked_where(
                numpy.fliplr(data.maskedData), numpy.fliplr(ma.filled(specificReturns, 0.0)))
        data.srMatrix.data = data.srMatrix.data[:,:deltaObs]
        self.log.debug('building time-series matrices: end')

        # Compute specific variances
        data.specificCov = dict()
        data.srMatrix.nameDict = modelDB.getIssueNames(
                data.srMatrix.dates[0], data.returns.assets, marketDB)

        # Recompute proportion of non-missing returns over specific return history
        if hasattr(self, 'legacyModel'):
            numOkReturns = data.numOkReturns
        else:
            nonMissingFlag = data.nonMissingFlag[:,-deltaObs:]
            numOkReturns = ma.sum(nonMissingFlag, axis=1)
            numOkReturns = numOkReturns / float(numpy.max(numOkReturns, axis=None))

        # Extra tweaking for linked assets
        if not self.SCM:
            entireUniverse = modelDB.getAllActiveSubIssues(modelDate)
            allSubIssueGroups = modelDB.getIssueCompanyGroups(
                    modelDate, entireUniverse, marketDB)
            for (groupId, subIssueList) in allSubIssueGroups.items():
                for sid in subIssueList:
                    if sid in data.assetIdxMap:
                        numOkReturns[data.assetIdxMap[sid]] = 1.0

        # Compute asset specific risks
        (data.specificVars, data.specificCov) = self.specificRiskCalculator.\
                computeSpecificRisks(data.srMatrix, data,
                        data.subIssueGroups, self.rmg, modelDB,
                        nOkRets=numOkReturns,
                        restrict=data.estimationUniverseIdx, scores=scores,
                        debuggingReporting=self.debuggingReporting,
                        hardCloneMap=data.hardCloneMap,
                        excludeTypes=noProxyType,
                        gicsDate=self.gicsDate)

        self.log.debug('computed specific variances')
        
        # More complex factor covariance part now
        # Set up statistical factor returns history and corresponding
        # information
        nonCurrencySubFactors = [f for f in subFactors \
                if f.factor not in self.currencies]
        nonCurrencyFactorReturns = Matrices.TimeSeriesMatrix(
                nonCurrencySubFactors, dateList[:omegaObs])
        nonCurrencyFactorReturns.data = ma.array(numpy.fliplr(factorReturns))[:,:omegaObs]
        
        if self.SCM:
            (shrinkType, shrinkFactor) = self.cp.getShrinkageParameters()
            if shrinkType is not None:
                self.compute_shrinkage_matrix(shrinkType, shrinkFactor,
                        nonCurrencySubFactors, modelDate, data.exposureMatrix, modelDB)

            data.factorCov = self.covarianceCalculator.\
                    computeFactorCovarianceMatrix(nonCurrencyFactorReturns)
            currencySubFactors = []
        else:
            # Load currency returns
            currencySubFactors = [f for f in subFactors \
                    if f.factor in self.currencies]
            currencyFactorReturns = self.currencyModel.loadCurrencyFactorReturnsHistory(
                    currencySubFactors, dateList[:maxOmegaObs], modelDB)
            # Copy currency factor returns, since they may get overwritten during
            # the cov matrix calculation
            tmpCFReturns = ma.array(currencyFactorReturns.data, copy=True)
        
            # Load currency risk model and compute factor cov
            crmi = modelDB.getRiskModelInstance(self.currencyModel.rms_id, modelDate)
            assert(crmi is not None)
            data.factorCov = self.build_regional_covariance_matrix(\
                        modelDate, dateList[:maxOmegaObs],
                        currencyFactorReturns, nonCurrencyFactorReturns,
                        crmi, modelDB, marketDB)

        if self.debuggingReporting:
            # Write correlation matrix to flatfile
            ncFactorNames = [f.factor.name for f in nonCurrencySubFactors]
            cFactorNames = [f.factor.name for f in currencySubFactors]
            factorNames = ncFactorNames + cFactorNames
            (d, corrMatrix) = Utilities.cov2corr(data.factorCov, fill=True)
            corroutfile = 'tmp/%s-corrFinal-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(corrMatrix, corroutfile, columnNames=factorNames,
                    rowNames=factorNames)

            # Output asset risks
            exposureIdx = [data.exposureMatrix.getFactorIndex(n) for n in factorNames]
            expMatrix = ma.filled(ma.take(data.exposureMatrix.getMatrix(), exposureIdx, axis=0), 0.0)
            assetFactorCov = numpy.dot(numpy.transpose(expMatrix), numpy.dot(data.factorCov, expMatrix))
            totalVar = numpy.diag(assetFactorCov) + data.specificVars
            sidList = [sid.getSubIDString() for sid in data.universe]
            fname = 'tmp/%s-totalRisks-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ma.sqrt(totalVar), fname, rowNames=sidList)
            fname = 'tmp/%s-specificRisks-%s.csv' % (self.name, dateList[0])
            Utilities.writeToCSV(ma.sqrt(data.specificVars), fname, rowNames=sidList)

        # Add covariance matrix to return object
        data.frMatrix = Matrices.TimeSeriesMatrix(
                subFactors, dateList[:omegaObs])
        data.frMatrix.data[:len(self.blind),:] = nonCurrencyFactorReturns.data
        if not self.SCM:
            data.frMatrix.data[len(self.blind):,:] = tmpCFReturns
        
        self.log.debug('computed factor covariances')
        return data

    def addCurrencyFactorExposures(self, modelDate, data, modelDB, marketDB):
        data.exposureMatrix = GlobalExposures.generate_currency_exposures(
                modelDate, self, modelDB, marketDB, data)

    def factorFlipFunction(self, modelDate, modelDB, marketDB):
        """Function that attempts to flip stat model exposures
        and factor returns by comparing the time series generated
        at consecutive points in time
        """

        # Determine home country info and flag DR-like instruments
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        universe = modelDB.getRiskModelInstanceUniverse(rmi)
        data = AssetProcessor.process_asset_information(
                modelDate, universe, self.rmg, modelDB, marketDB,
                checkHomeCountry=(self.SCM==0),
                legacyDates=self.legacyMCapDates,
                numeraire_id=self.numeraire.currency_id,
                forceRun=self.forceRun)

        # Pick up list of dates
        dateList = modelDB.getDates(self.rmg, modelDate, self.pcaHistory, fitNum=True, excludeWeekend=True)

        # Load cov matrix and get factor info
        (factorCov, factors) = self.loadFactorCovarianceMatrix(rmi, modelDB)
        factorIdxMap = dict(zip(factors, list(range(len(factors)))))
        blindFactors = [f.name for f in factors if f in self.blind]
        subFactors = modelDB.getSubFactorsForDate(modelDate, factors)
        P = numpy.zeros((factorCov.shape), float)

        # Load exposures
        exposureMatrix = self.loadExposureMatrix(rmi, modelDB)
        expMArray = exposureMatrix.getMatrix()
        expMCopy = ma.array(expMArray, copy=True)

        # Load factor return histories for current date
        factorT = modelDB.loadStatFactorReturnsHistory(self.rms_id, subFactors, dateList)
        tCount = len(Utilities.nonEmptyColumns(factorT.data))
        if tCount < factorT.data.shape[1]:
            diff = factorT.data.shape[1] - tCount
            logging.error('%d missing statistical factor returns for %s', diff, dateList[-1])
            assert(diff==0)
        # Load factor return histories for previous date
        factorTminus1 = modelDB.loadStatFactorReturnsHistory(self.rms_id, subFactors, dateList[:-1])
        tCount = len(Utilities.nonEmptyColumns(factorTminus1.data))
        if tCount < factorTminus1.data.shape[1]:
            diff = factorTminus1.data.shape[1] - tCount
            logging.warning('%d missing statistical factor returns for %s', diff, dateList[-2])
        # Line up the two series
        facretCopy = ma.array(factorT.data, copy=True)
        dataT = ma.filled(factorT.data[:,1:-1], 0.0)
        dataTm1 = ma.filled(factorTminus1.data[:,1:], 0.0)
        if dataT.shape != dataTm1.shape:
            logging.error('Current and previous returns histories are different lengths')
        nonMissingData = ma.masked_where(factorTminus1.data==0.0, factorTminus1.data)
        nonMissingData = numpy.flatnonzero(ma.getmaskarray(nonMissingData)==0)

        # Do the flipping
        flipData = False
        changeCount = 0
        runningList = list(range(len(subFactors)))
        runningList = [idx for idx in runningList \
                if subFactors[idx].factor.name in blindFactors]

        for (idx, fid) in enumerate(subFactors):

            if (fid.factor.name not in blindFactors) or (len(nonMissingData)==0):
                P[idx,idx] = 1.0
                continue

            firstTime = True
            mlt = 1.0
            for jdx in runningList:

                # Compute stats for comparison with another factor
                correl = numpy.corrcoef(dataT[jdx,:], dataTm1[idx,:], rowvar=False)[0,1]
                rreg = Utilities.robustLinearSolver(dataT[jdx,:], dataTm1[idx,:], robust=True)
                pval = rreg.pvals[0]
                if pval > 0.25:
                    w = (pval - 0.25) / 0.75
                    correl = (w * 0.0) + ((1.0 - w) * correl)
                logging.debug('%s, Index: %s, To: %s, correl: %s',
                        modelDate, idx, jdx, correl)

                # Score the factor data to see whether it's worth switching
                if firstTime:
                    maxCorr = correl
                    firstTime = False
                    kdx = jdx
                    mlt = 1.0
                    if abs(maxCorr) > 0.1:
                        mlt = numpy.sign(maxCorr)
                else:
                    if abs(correl) > abs(maxCorr):
                        maxCorr = correl
                        kdx = jdx
                        mlt = 1.0
                        if abs(maxCorr) > 0.1:
                            mlt = numpy.sign(maxCorr)
            runningList = [jdx for jdx in runningList if jdx != kdx]
            logging.info('%s, %s mapped to: %s', modelDate, idx, kdx)

            if (idx != kdx) or (mlt < 0):
                logging.info('Flipping factor %s for %s * factor %s',
                        fid.factor.name, mlt, subFactors[kdx].factor.name)
                flipData = True
                changeCount += 1
                factorT.data[idx,:] = mlt * facretCopy[kdx,:]
                expMArray[idx,:] = mlt * expMCopy[kdx,:]
                P[idx,kdx] = mlt
            else:
                P[idx,idx] = 1.0

        if self.debuggingReporting:
            exposureMatrix.dumpToFile('tmp/expM-flip-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate,
                    subIssueGroups=data.subIssueGroups, dp=self.dplace)
            dates = [str(d) for d in dateList]
            factorNames = [sf.factor.name for sf in subFactors]
            retOutFile = 'tmp/%s-facretHist-flip-%s.csv' % (self.name, modelDate)
            Utilities.writeToCSV(factorT.data, retOutFile, rowNames=factorNames, columnNames=dates)

        if flipData:
            results = Utilities.Struct()
            results.factorCov = numpy.dot(P, numpy.dot(factorCov, numpy.transpose(P)))
            results.exposureMatrix = exposureMatrix
            results.frMatrix = factorT
            results.count = changeCount
        else:
            results = None
        return results

class MacroeconomicModel(FactorRiskModelv3):
    """Macroeconmic factor model"""
    
    def __init__(self, primaryID, modelDB, marketDB):
        FactorRiskModel.__init__(self, primaryID, modelDB, marketDB)
        factors = modelDB.getRiskModelSerieFactors(self.rms_id)

        self.descFactorMap = dict([(i.description, i) for i in factors])
        self.nameFactorMap = dict([(i.name, i) for i in factors])
        # Add factor ID to factors, matching by description
        for f in (self.macro_core + self.macro_market_traded   
                  + self.macro_equity  + self.additionalCurrencyFactors + self.macro_sectors):
            dbFactor = self.descFactorMap[f.description]
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt
        if len(self.rmg) < 2:
            self.SCM = True
        else:
            self.SCM = False
        # No stat, style or industry factors
        self.blind = []
        self.styles = []
        self.industries = []
        self.estuMap = None

        #Subclasses may override these.
        self._factorHistoryPadding=70
        self.dateFloor=datetime.date(1985,1,1) 
        self.dateCeiling=datetime.date(2999,12,31)
        self.datesForMacroCache=[self.dateFloor,datetime.date.today(),datetime.date.today()]
        self._needsAssetReturns=False 
        self._needsMacroData=False 

    def getEstuUnion(self,dates,modelDB):
        dates=sorted(dates)
        rmi=modelDB.getRiskModelInstance(self.rms_id, dates[-1])
        universeInt=set(modelDB.getRiskModelInstanceUniverse(rmi))
        estuInt=set(modelDB.getRiskModelInstanceESTU(rmi))
        estus=set(estuInt)
        universes=set(universeInt)
        
        for date in dates[:-1]:
            rmi=modelDB.getRiskModelInstance(self.rms_id, date)
            if rmi is not None:
                univTmp=modelDB.getRiskModelInstanceUniverse(rmi)
                estuTmp=modelDB.getRiskModelInstanceESTU(rmi)
                universes.update(univTmp)
                estus.update(estuTmp)
                universeInt.intersection_update(univTmp)
                estuInt.intersection_update(estuTmp)
         
        return (list(estus),list(universes),list(estuInt),list(universeInt))

    def setDatesForMacroCache(self,dates,modelDB):
        allDates=modelDB.getDateRange(self.rmg,self.dateFloor,max(dates),excludeWeekend=True)
        shortDates=[d for d in allDates if d<=min(dates)][-(self.factorHistory+self._factorHistoryPadding):]
        self.datesForMacroCache=[min(shortDates),min(dates),max(dates)]

    def setFactorsForDate(self,date, modelDB=None):
        if not self.SCM:
            self.setBaseModelForDate(date)

        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)
        self.currencies = []

        # Create currency factors
        if not self.SCM:
            self.currencies = [ModelFactor(f, None)
                    for f in set([r.currency_code for r in self.rmg])]
            # Add additional currency factors (to allow numeraire changes)
            # if necessary
            self.currencies.extend([f for f in self.additionalCurrencyFactors
                                    if f not in self.currencies
                                    and f.isLive(date)])

            for f in self.currencies:
                dbFactor = self.nameFactorMap[f.name]
                f.description = dbFactor.description
                f.factorID = dbFactor.factorID
                f.from_dt = dbFactor.from_dt
                f.thru_dt = dbFactor.thru_dt
        self.currencies = sorted(self.currencies)

        allFactors = (self.macro_core + self.macro_market_traded 
             + self.macro_equity + self.currencies + self.macro_sectors)
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factors = allFactors
        self.validateFactorStructure(date, warnOnly=True)

        self.fundamentalModel.setFactorsForDate(date, modelDB)
        self.statisticalModel.setFactorsForDate(date, modelDB)
        
        # Fundamental model may have switched classification
        self.industryClassification = self.fundamentalModel.industryClassification

    def generate_model_universe(self,date, modelDB, marketDB): 
        return self.statisticalModel.generate_model_universe(date, modelDB, marketDB)

    def getMasterMacroData(self,modelDB,marketDB,force_reload=False):
        """ """
        self.log.debug('getMasterMacroData: begin')
        import pandas
        from riskmodels import MacroUtils
        from riskmodels.MacroUtils import MacroConfigDataLoader, getActiveSeries, getActiveSeriesAsOf
        if not force_reload and self._macroDataMaster is not None:
            return self._macroDataMaster

        self._macroDataMaster=MacroConfigDataLoader(self.macroMetaDataFile,self._needsAssetReturns,
                needs_sectors=self.factorReturnsCalculator.params.include_sectors)
        self._macroDataMaster.loadRawData(self,modelDB,marketDB,
                self.datesForMacroCache[0],self.datesForMacroCache[1],self.datesForMacroCache[2]) 
        
        self.log.debug('getMasterMacroData: end')
        return self._macroDataMaster

    def getBlocks(self):
        return [
                #[c.name for c in self.macro_core],
                #[c.name for c in self.macro_market_traded],
                #[c.name for c in self.macro_equity + self.macro_sectors],
                [c.name for c in self.macro_core+self.macro_market_traded+self.macro_equity + self.macro_sectors],
               ]

    def getMacroDataForDate(self,modelDate,modelDB,marketDB,
            load_assets=True,load_macro=True,history=None,initial=False):
        """ """
        self.log.debug('getMacroDataForDate: begin')
        from riskmodels import MacroUtils
        from riskmodels.MacroUtils import getActiveSeries, getActiveSeriesAsOf
        import pandas
        import numpy as np

        if history is None:
            history = self.factorHistory+70
        dataMaster=self.getMasterMacroData(modelDB,marketDB)
        dates=[d for d in self._macroDataMaster.dates_model if d<=modelDate][-history:]
        modelData=self._macroDataMaster.getDataForDates(self,min(dates),max(dates),modelDB,initial=initial)
        modelData.results=Utilities.Struct()
        if self.debugging:
            self.dbg_modelData=modelData

        estuData=self.getEstuUnion([modelDate],modelDB)
        modelData.estu=estuData[0]
        modelData.universe=estuData[1]

        #if load_assets or self._needsAssetReturns: #HACK:
        #if self._needsAssetReturns: #HACK:
        #    self.log.debug('Loading asset returns in factor return calculator.  Is this what you want?')
        #    modelData.assetReturnsEstu=self._loadAssetReturnsAsDF(modelDate,modelDB,marketDB,history=history-400,subIds=modelData.estu)
        #    modelData.spotRateDaily=self._loadRiskFreeRateHistoryAsDF(modelDB,marketDB,dates,currencies=['USD'],annualize=False)
        #    modelData.spotRateAnnual=self._loadRiskFreeRateHistoryAsDF(modelDB,marketDB,dates,currencies=['USD'],annualize=True)

        modelData.blocks=self.getBlocks()

        self.log.debug('getMacroDataForDate: end')
        return modelData

    def getExposureStepData(self,modelDate, modelDB,marketDB):
        """Load all the data required to run the exposure step
           and return it in a Struct
        """
        self.log.debug('getExposureStepData: begin')
        modelData = Utilities.Struct()
        all_dates = sorted((rmi for rmi in modelDB.getRiskModelInstances(self.rms_id) if rmi.date <= modelDate), key=lambda x: x.date)[-self.returnHistory:]
        if not all(rmi.has_returns for rmi in all_dates):
            raise Exception('Missing factor return estimates in last ' + str(self.returnHistory) + ' days for model date ' + str(modelDate))
        modelData.dates = [rmi.date for rmi in all_dates]
        modelData.date = modelDate
        estuData=self.getEstuUnion([modelDate],modelDB)
        modelData.estu=estuData[0]
        modelData.universe=estuData[1]
        subIds=modelData.universe

        asset_data=self._loadAssetReturnsAsDF(modelDate,modelDB,marketDB,
                history=self.returnHistory,subIds=subIds)
        modelData.assetReturnsHistory=asset_data.excessReturnsDF
        modelData.clippedReturnsHistory=asset_data.clippedReturnsDF
        modelData.marketCaps=asset_data.marketCaps
        modelData.factorReturnsHistory=self._loadFactorReturnsAsDF(modelData.dates[0],modelData.dates[-1],modelDB)
        modelData.fundamentalFactorReturns=self._loadFactorReturnsAsDF(modelData.dates[0],modelData.dates[-1],modelDB,self.fundamentalModel)

        cls = self.industryClassification
        modelData.sectors = [s.description for s in cls.getNodesAtLevel(modelDB, 'Sectors')]
        asset2sector = dict((subid,val.classification.description) for subid, val in cls.getAssetConstituents(modelDB, subIds, modelDate, level=1).items())
        asset2ind = dict((subid,val.classification.description) for subid, val in cls.getAssetConstituents(modelDB, subIds, modelDate).items())
        sector2asset = defaultdict(list)
        for asset, sector in asset2sector.items():
            sector2asset[sector].append(asset)
        modelData.sector2asset = dict((sector, assets) for sector, assets in sector2asset.items())
        modelData.clsdata = pandas.DataFrame({'industry': asset2ind, 'sector': asset2sector})

        modelData.blocks=self.getBlocks()

        self.log.debug('getExposureStepData: end')
        return modelData

    def packageResults(self,modelData):
        assert hasattr(modelData,'results')
        data=modelData.results
        data.universe=modelData.universe
        factors = [f.name for f in self.factors]

        if hasattr(data,'factorReturns') and data.factorReturns is not None:
            data.factorReturnsFinal=data.factorReturns.copy().reindex(factors)
        
        if hasattr(data,'factorReturnsDF') and data.factorReturnsDF is not None:
            data.factorReturnsDFFinal=data.factorReturnsDF.copy().reindex(columns=factors)
        
        if hasattr(data,'specificReturns') and data.specificReturns is not None:
            data.specificReturnsFinal=data.specificReturns.reindex(data.universe).dropna().copy() #HACK: why is this needed?

    def generateFactorReturns(self, modelDate, modelDB, marketDB,
            initial=False, modelDateLower=None, force=False, skipMacroQA=False):
        self.setFactorsForDate(modelDate, modelDB)
        if hasattr(self,'macroQAhelper') and not skipMacroQA:
            self.log.info('Verifying macro time series data.')
            self.macroQAhelper.verifyEssentialModelTimeSeries(modelDB,marketDB,modelDate,force)
            self.log.info('Done verifying macro time series data.')
        else:
            self.log.warning('Attribute macroQAhelpr not set.  No QA is being done (here) on the macro time series.')

        modelData=self.getMacroDataForDate(modelDate,modelDB,marketDB,initial=initial)
        modelData.modelDateLower=modelDateLower #ADAM: probably a more elegant way to do this
        self.factorReturnsCalculator.compute(modelData)
        
        self.packageResults(modelData)
        return modelData.results        

    def generateExposuresAndSpecificReturns(self,modelDate,modelDB,marketDB):
        self.setFactorsForDate(modelDate, modelDB)
        modelData=self.getExposureStepData(modelDate, modelDB, marketDB)
        modelData.results = Utilities.Struct()
        self.exposureCalculator.compute(modelData)

        exposures = modelData.results.exposures
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        expM = self.loadExposureMatrix(rmi, modelDB, skipAssets=True)
        assert (set(exposures.columns) == set(expM.getFactorNames()))
        expM.setAssets(exposures.index)
        for fname, b in exposures.items():
            ftype = expM.getFactorType(fname)
            b = ma.array(b.values.copy(), mask=pandas.isnull(b).values)
            expM.addFactor(fname, b, ftype)
        
        # Clone DR and cross-listing exposures if required
        data = Utilities.Struct()
        data.universe = modelData.universe
        data.exposureMatrix = expM
        data.assetIdxMap = dict((subid,idx) for idx, subid in enumerate(data.exposureMatrix.assets_))
        subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        scores = self.score_linked_assets(
                modelDate, data.universe, modelDB, marketDB,
                subIssueGroups=subIssueGroups)
        data.exposureMatrix = self.clone_linked_asset_exposures(
                modelDate, data, modelDB, marketDB, scores,
                subIssueGroups=subIssueGroups)
        if self.debuggingReporting:
            data.exposureMatrix.dumpToFile('tmp/expM-%s-%04d%02d%02d.csv'\
                    % (self.name, modelDate.year, modelDate.month, modelDate.day),
                    modelDB, marketDB, modelDate, subIssueGroups=subIssueGroups, dp=self.dplace)
        modelData.results.exposureMatrix = data.exposureMatrix
        self.packageResults(modelData)
        return modelData.results        

    def loadRawFactorReturnsAsDF(self,startDate,endDate,modelDB,riskModel=None):
        if riskModel is None:
           riskModel=self

        modelDates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, excludeWeekend=True)
        subFactors = modelDB.getSubFactorsForDate(max(modelDates), riskModel.factors)

        fr = modelDB.loadRawFactorReturnsHistory(riskModel.rms_id, subFactors, modelDates)
        facRetDF = pandas.DataFrame(fr.data, index=[f.factor.name for f in subFactors], columns=modelDates).T.copy()

        return facRetDF

    def insertRawFactorReturnsFromDF(self,fRetDF,modelDB,
                                       riskModel=None,write_db=True):
        if riskModel is None:
            riskModel=self
        
        dates=list(fRetDF.index)
        factors=list(fRetDF.columns)
        for date in dates:
            rmi=riskModel.getRiskModelInstance(date,modelDB)
            if rmi is None:
                continue
            subFactorSet=set(modelDB.getRiskModelInstanceSubFactors(rmi, riskModel.factors))
            subFactorMap=dict((s.factor.name,s) for s in subFactorSet)
            subFactors=[subFactorMap[i] for i in factors]

            if write_db:
                modelDB.deleteRawFactorReturns(rmi)
                modelDB.insertRawFactorReturns(riskModel.rms_id, date, subFactors, fRetDF.loc[date].copy().values)

    def insertRawFactorReturnsFromSeries(self,fRetSeries,modelDate,modelDB,
                                       riskModel=None,write_db=True):
        """ """ 
        fretDF = pandas.DataFrame({modelDate: fRetSeries}).T
        self.insertRawFactorReturnsFromDF(fretDF, modelDB,riskModel,write_db)

    def _loadFactorReturnsAsDF(self,startDate,endDate,modelDB,riskModel=None):
        if riskModel is None:
           riskModel=self

        modelDates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, excludeWeekend=True)
        subFactors = modelDB.getSubFactorsForDate(max(modelDates), riskModel.factors)

        fr = modelDB.loadFactorReturnsHistory(riskModel.rms_id, subFactors, modelDates)
        facRetDF = pandas.DataFrame(fr.data, index=[f.factor.name for f in subFactors], columns=modelDates).T.copy()

        return facRetDF

    def _loadExposureMatrixAsDF(self,date,modelDB,riskModel=None):
        if riskModel is None:
            riskModel=self
        rmi=riskModel.getRiskModelInstance(date,modelDB)
        expmatRaw=riskModel.loadExposureMatrix(rmi,modelDB)
        expmatDF=pandas.DataFrame(
            expmatRaw.data_.T.copy(),
            index=expmatRaw.assets_,columns=expmatRaw.factors_).fillna(0.).copy()
        return expmatDF,expmatRaw

    def _insertFactorCovAndSpecRisksFromDF(self,covmatDF,specificRisksSeries,
                                           date,modelDB,
                                           riskModel=None,write_db=True):
        """Warning: does not insert specific co-variances."""
        if riskModel is None:
            riskModel=self
        rmi=riskModel.getRiskModelInstance(date,modelDB)
        factors=list(covmatDF.index)
        subFactorSet=set(modelDB.getRiskModelInstanceSubFactors(rmi, riskModel.factors))
        subFactorMap=dict((s.factor.name,s) for s in subFactorSet)
        subFactors=[subFactorMap[i] for i in factors]

        srSeries=specificRisksSeries.copy()
        subIssues=list(srSeries.index)

        if write_db:
            modelDB.deleteRMIFactorSpecificRisk(rmi)
            modelDB.insertFactorCovariances(rmi,subFactors,covmatDF.copy().values)
            modelDB.insertSpecificRisks(rmi, subIssues, srSeries.copy().values )
            rmi.setHasRisks(True, modelDB)

    def _insertFactorReturnsFromDF(self,fRetDF,modelDB,
                                       riskModel=None,write_db=True):
        if riskModel is None:
            riskModel=self
        
        dates=list(fRetDF.index)
        factors=list(fRetDF.columns)
        for date in dates:
            rmi=riskModel.getRiskModelInstance(date,modelDB)
            if rmi is None:
                next
            subFactorSet=set(modelDB.getRiskModelInstanceSubFactors(rmi, riskModel.factors))
            subFactorMap=dict((s.factor.name,s) for s in subFactorSet)
            subFactors=[subFactorMap[i] for i in factors]

            if write_db:
                modelDB.deleteFactorReturns(rmi)
                modelDB.insertFactorReturns(riskModel.rms_id, date, subFactors, fRetDF.loc[date].copy().values)
                rmi.setHasReturns(True,modelDB)

    def _insertFactorReturnsFromSeries(self,fRetSeries,modelDate,modelDB,
                                       riskModel=None,write_db=True):
        """ """ 
        if riskModel is None:
            riskModel=self
        
        factors=list(fRetSeries.index)
        rmi=riskModel.getRiskModelInstance(modelDate,modelDB)
        subFactorSet=set(modelDB.getRiskModelInstanceSubFactors(rmi, riskModel.factors))
        subFactorMap=dict((s.factor.name,s) for s in subFactorSet)
        subFactors=[subFactorMap[i] for i in factors]

        if write_db:
            modelDB.deleteFactorReturns(rmi)
            modelDB.insertFactorReturns(riskModel.rms_id, modelDate, subFactors, fRetSeries.copy().values)
            rmi.setHasReturns(True,modelDB)

    def _insertSpecificReturnsFromSeries(self,specificReturnsSeries,modelDate,
            modelDB,riskModel=None,write_db=True):
        
        if riskModel is None:
            riskModel=self
        
        if write_db:
            rmi=riskModel.getRiskModelInstance(modelDate,modelDB)
            modelDB.deleteSpecificReturns(rmi)
            modelDB.insertSpecificReturns(riskModel.rms_id,modelDate,
                    list(specificReturnsSeries.index),specificReturnsSeries.values)

    def _loadFXReturnsAsDF(self,modelDB,marketDB,startDate,endDate,
            currencies=['XDR'],base='USD',riskModel=None):

        if riskModel is None:
            riskModel=self

        #history=(((endDate-startDate).days)) + 64 #
        history=(((endDate-startDate).days)*5)/7 + 128 #Approximation 
        cRetRaw=modelDB.loadCurrencyReturnsHistory(riskModel.rmg,endDate,history,currencies,base)
        cRetDF=pandas.DataFrame(cRetRaw.data.T.copy(),
                index=cRetRaw.dates,columns=cRetRaw.assets)
        return cRetDF

    def _loadRiskFreeRateHistoryAsDF(self,modelDB,marketDB,dateList,
            currencies=['USD'],annualize=False,riskModel=None):

        if riskModel is None:
            riskModel=self

        cRetRaw=modelDB.getRiskFreeRateHistory(currencies,dateList,marketDB,annualize=annualize)
        cRetDF=pandas.DataFrame(cRetRaw.data.T.copy(),
                index=cRetRaw.dates,columns=cRetRaw.assets)
        return cRetDF

    def _loadAssetReturnsAsDF(self,modelDate,modelDB,marketDB,history=None,subIds=None):
        logging.debug('_loadAssetReturnsAsDF: begin')
        import pandas
        tmpReturnHistory=self.returnHistory
        if history is None:
            history=self.returnHistory
        self.returnHistory=history
        rmi=modelDB.getRiskModelInstance(self.rms_id,modelDate)
        if subIds is not None:
            universe=subIds
        else:
            universe=modelDB.getRiskModelInstanceUniverse(rmi)

        results=Utilities.Struct()
        
        data=AssetProcessor.process_asset_information(
            modelDate, universe, self.rmg, modelDB, marketDB, False,
            legacyDates=self.legacyMCapDates)

        data = self.build_excess_return_history_v3(data, modelDate, modelDB, marketDB)
        
        data.clippedReturns = Utilities.bucketedMAD(
                self.rmg, modelDate, data.returns.data, data, modelDB,
                nDev=[6.0, 6.0], MADaxis=0,
                gicsDate=self.gicsDate)

        #TODO: check for missing/extra assets

        results.excessReturnsDF=data.returns.toDataFrame().T
        results.clippedReturnsDF=pandas.DataFrame(data.clippedReturns.T,
                index=results.excessReturnsDF.index,columns=results.excessReturnsDF.columns)

        results.marketCaps=pandas.Series(data.marketCaps, index=data.universe)
        
        self.returnHistory=tmpReturnHistory
        logging.debug('_loadAssetReturnsAsDF: end')
        return results       


# vim: set softtabstop=4 shiftwidth=4:

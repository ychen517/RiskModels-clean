import logging
import numpy.ma as ma
import numpy
import pandas
from collections import defaultdict
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities
from riskmodels import ModelDB
from riskmodels import ProcessReturns
from riskmodels import AssetProcessor_V4 as AssetProcessor

class ConstructEstimationUniverse:
    def __init__(self, date, assets, rmClass, modelDB, marketDB, debugOutput=False):
        """ Generic class for estU construction
        """
        self.rmClass = rmClass
        self.date = date
        self.assets = set(assets)
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.debugOutput = debugOutput

    def apply_exclusion_list(self, baseEstu=None, productExclude=True):
        """Exclude assets in the RMS_ESTU_EXCLUDED table.
        """
        logging.debug('apply_exclusion_list: begin')
        if baseEstu is None:
            baseEstu = self.assets

        # Get exclusion assets from table
        exclIssues = self.modelDB.getRMSESTUExcludedIssues(self.rmClass.rms_id, self.date)
        if len(exclIssues) > 0:
            mdlIdxMap = dict([(sid.getModelID(), sid) for sid in self.assets])
            exclIndices= set([mdlIdxMap[mdl] for mdl in exclIssues if mdl in mdlIdxMap])
            baseEstu = baseEstu.difference(exclIndices) 

        if productExclude:
            productExcludes = self.modelDB.getProductExcludedSubIssues(self.date)
            baseEstu = baseEstu.difference(set(productExcludes))

        logging.debug('apply_exclusion_list: end')
        return baseEstu

    def exclude_by_cap_ranking(\
            self, assetData, baseEstu=None, byFactorType=False, expM=None, excludeFactors=[],
            lower_pctile=25, upper_pctile=100, minFactorWidth=10):
        """ Exclude assets outside particular capitalisation bands by percentage
        """
        logging.debug('exclude_by_cap_ranking: begin')
        # initialise stuff
        estu = set()
        qualifyDict = dict()
        mcap = assetData.marketCaps.copy(deep=True)
        targetRatio = (upper_pctile - lower_pctile) / 100.0

        # Set up factor and exposure matrix info
        if not byFactorType:
            exposures = pandas.DataFrame(1.0, index=assetData.universe, columns=['Market'])
        else:
            assert(byFactorType in expM.factorTypes_)
            exposures = expM.toDataFrame().loc[:, expM.getFactorNames(byFactorType)]
        excludeFactorNames = [f.name for f in excludeFactors]

        # Extract eligible market caps
        if baseEstu is not None:
            mcap = mcap[baseEstu]
        mcap = mcap.reindex(assetData.universe)
        nonMissingCaps = set(mcap[numpy.isfinite(mcap)].index)

        # Loop round set of factors
        for factorName in exposures.columns:
            if factorName in excludeFactorNames:
                continue

            # Pick out eligible assets
            expos = exposures.loc[:, factorName].squeeze()
            qualified = set(expos[numpy.isfinite(expos)].index).intersection(nonMissingCaps)
            if len(qualified)==0:
                logging.warning('%s: No assets in %s!', self.date, factorName)
                qualifyDict[factorName] = set()
                continue

            # Get effective number of assets exposed to factor according to Herfindahl index
            factorMCap = mcap[qualified]
            effectiveWidth = Utilities.inverse_herfindahl(numpy.sqrt(factorMCap))

            # If factor is sufficiently populated, proceed
            if effectiveWidth > minFactorWidth:
                sortedCap = factorMCap.sort_values(ascending=False).cumsum(axis=0) / factorMCap.sum(axis=None)
                qualified = set(sortedCap[numpy.isfinite(sortedCap.mask(sortedCap>=targetRatio))].index)
                if len(qualified) < len(sortedCap):
                    qualified.add(sortedCap.index[len(qualified)])
                qualified.add(sortedCap.index[0])

            # Add IDs to tally
            estu = estu.union(qualified)
            qualifyDict[factorName] = qualified

        # Show list of industries/countries and their respective caps
        if self.debugOutput:
            totalFCaps = mcap[estu].sum(axis=None)
            for factorName in exposures.columns:
                if factorName in excludeFactorNames:
                    logging.info('Factor %s excluded', factorName)
                    continue
                if len(qualifyDict[factorName]) < 1:
                    continue
                fc = mcap[qualifyDict[factorName]].sort_values()
                fcSum = fc.sum(axis=None)
                logging.info('%s: %.2f bn %s (%d assets, %.2f%%, top 5: %.2f%%, top 1: %.2f%%)',
                    factorName, fcSum / 1e9, self.rmClass.numeraire.currency_code, len(qualifyDict[factorName]),
                    100.0 * fcSum / totalFCaps, 100.0 * fc[fc.index[-5:]].sum(axis=None) / fcSum,
                    100.0 * fc[fc.index[-1]] / fcSum)

        # Report on final state of estu
        logging.info('Subset contains %d assets, %.4f tr %s market cap',
                len(estu), mcap[estu].sum(axis=None) / 1.0e12, self.rmClass.numeraire.currency_code)
        logging.debug('exclude_by_cap_ranking: end')
        return estu

    def report_on_changes(self, n, estu):
        if n > len(estu):
            logging.info('... Eligible Universe down %d and currently stands at %d stocks', n-len(estu), len(estu))
        elif n < len(estu):
            logging.info('... Eligible Universe up %d and currently stands at %d stocks', len(estu)-n, len(estu))

        if hasattr(self.rmClass, 'trackList'):

            for sid in self.rmClass.trackList:
                if sid not in self.assets:
                    continue
                if sid in estu:
                    if sid in self.rmClass.dropList:
                        logging.info('Asset %s added back to estimation universe process', sid.getSubIDString())
                        self.rmClass.addList.append(sid)
                        self.rmClass.dropList.remove(sid)
                    elif sid not in self.rmClass.addList:
                        logging.info('Asset %s added to estimation universe process', sid.getSubIDString())
                        self.rmClass.addList.append(sid)
                else:
                    if sid in self.rmClass.addList:
                        logging.info('Asset %s dropped from estimation universe process', sid.getSubIDString())
                        self.rmClass.dropList.append(sid)
                        self.rmClass.addList.remove(sid)
                    elif sid not in self.rmClass.dropList:
                        logging.info('Asset %s dropped from estimation universe process', sid.getSubIDString())
                        self.rmClass.dropList.append(sid)
        return len(estu)

    def filter_by_user_score(self, scoreDict, baseEstu=None, lower_pctile=25, upper_pctile=100):
        """Filter out assets outside some percentile range of a user-specified list of scores
        """
        logging.debug('filter_by_user_score: begin')

        # Determine subset of eligible assets for comparison
        if baseEstu is None:
            baseEstu = set(self.assets)

        # Get score percentile values
        advScores = scoreDict.loc[:, 'ISC_ADV_Score'].fillna(0.0)
        cap_thresholds = Utilities.prctile(advScores[baseEstu].values, [lower_pctile, upper_pctile])
        validSids = advScores[advScores.between(cap_thresholds[0], cap_thresholds[1])].index
        estu = baseEstu.intersection(set(validSids))

        # Return list of asset indices
        logging.info('%d assets with ADV scores within percentiles %d to %d', len(estu), lower_pctile, upper_pctile)
        logging.debug('filter_by_user_score: end')
        return estu

    def filter_by_cap_and_volume(self, assetData, baseEstu=None, hiCapQuota=1000, loCapQuota=2000,
                                 volDays=250, bufferFactor=1.2, downWeights=None):
        """Filter out a fixed number of stocks by capitalisation and volume.
        Attempts to identify suitable assets using size and liquidity at the same time,
        without having to rank by one criteria then imposing the other as a penalty.
        Methodology is adapted from the Japanese TOPIX Index Series.
        """
        logging.debug('filter_by_cap_and_volume: begin')

        # Restrict universe to keep things manageable
        if baseEstu is None:
            baseEstu = set(self.assets)
        mcap_estu = assetData.marketCaps[baseEstu]

        if hiCapQuota + loCapQuota >= len(baseEstu):
            return baseEstu

        # Check volume cache
        if self.modelDB.volumeCache.maxDates < volDays:
            logging.warning('ModelDB volume cache too small (%d days), resizing to %d days',
                    self.modelDB.volumeCache.maxDates, volDays)
            self.modelDB.setVolumeCache(volDays)

        # Load volumes for required dates
        dateList = self.modelDB.getDates(self.rmClass.rmg, self.date, volDays-1)
        baseCurrencyID = self.rmClass.numeraire.currency_id
        volume = self.modelDB.loadVolumeHistory(dateList, baseEstu, baseCurrencyID).toDataFrame()
        totalVol = volume.fillna(0.0).sum(axis=1)

        # Extra user-downweights if required
        if downWeights is not None:
            mcap_estu = mcap_estu * downWeights
            totalVol = totalVol * downWeights

        # Sort from highest to lowest
        mcapRank = mcap_estu.sort_values(ascending=False)
        volRank = totalVol.sort_values(ascending=False)

        # Pick out large, liquid stocks
        upperBound = int(hiCapQuota * bufferFactor)
        mcap_estu = set(mcapRank[mcapRank.index[:upperBound]].index)
        vol_estu = set(volRank[volRank.index[:upperBound]].index)
        estu = mcap_estu.intersection(vol_estu)
        estu = set(mcapRank[estu].sort_values(ascending=False).index[:hiCapQuota])
        logging.debug('Using cap and volume to determine %d of %d large/mid-caps', len(estu), hiCapQuota)

        # Attempt to pad out if we fall short; ie, opt for more liquid stocks
        if len(estu) < hiCapQuota:
            missing = hiCapQuota - len(estu)
            logging.debug('Using only volume to determine remaining %d large/mid-caps', missing)
            diff = vol_estu.difference(estu)
            estu = estu.union(set(volRank[diff].sort_values(ascending=False).index[:missing]))

        # Determine set of smaller stocks
        upperBound = int((hiCapQuota + loCapQuota) * bufferFactor)
        mcap_estu = set(mcapRank[mcapRank.index[:upperBound]].index)
        vol_estu = set(volRank[volRank.index[:upperBound]].index)
        extra = (mcap_estu.intersection(vol_estu)).difference(estu)
        extra = set(mcapRank[extra].sort_values(ascending=False).index[:loCapQuota])

        # Add these to the total
        logging.debug('Using cap and volume to determine %d small-caps', len(extra))
        estu = estu.union(extra)

        # If we're still short after that, filter by volume alone
        if len(extra) < loCapQuota:
            missing = loCapQuota - len(extra)
            logging.debug('Using only volume to determine remaining %d small-caps', missing)
            extra = vol_estu.difference(estu)
            extra = set(volRank[extra].sort_values(ascending=False).index[:missing])
            estu = estu.union(extra)

        logging.debug('filter_by_cap_and_volume: end')
        return estu
    
    def pump_up_factors(self, assetData, exposureMatrix, currentEstu=None, baseEstu=None,
            byFactorType=[Matrices.ExposureMatrix.IndustryFactor],
            minFactorWidth=10, cutOff=0.01, excludeFactors=[], downWeights=None, quiet=False):
        """Attempts to inflate thin factors
        Routine will stop if increase in effective width of factor from adding another asset
        falls below value cutOff (default is 1%)
        Thus, to make the routine more stringent, increase this value
        """
        logging.debug('pump_up_factors: begin')
        hasThinFactors = False
        
        # Initialise
        mcap = assetData.marketCaps.copy(deep=True)
        exposures = exposureMatrix.toDataFrame()
        excludeFactorNames = []
        if excludeFactors is not None:
            excludeFactorNames = [f.name for f in excludeFactors]
        if baseEstu == None:
            baseEstu = set(self.assets)
        if currentEstu == None:
            currentEstu = set()
        else:
            currentEstu = set(currentEstu)

        # Get eligible assets and weights
        if downWeights is not None:
            mcap = mcap * downWeights
        eligibleSids = baseEstu.union(currentEstu)
        sqrtCap = numpy.sqrt(mcap)

        # Loop round factor types
        for fType in byFactorType:
            assert(fType in exposureMatrix.factorTypes_)

            # Loop round set of factors of each type
            for fname in exposureMatrix.getFactorNames(fType):
                if fname in excludeFactorNames:
                    continue

                # Get assets with exposure to factor
                factorExp = exposures.loc[:, fname]
                factorSids = set(factorExp[numpy.isfinite(factorExp)].index).intersection(eligibleSids)
                if len(factorSids) < 1:
                    if not quiet:
                        logging.warning('Factor %s is empty', fname)
                    hasThinFactors = True
                    continue

                # Calculate number of assets exposed to factor according to Herfindahl methodology
                estuFactorSids = factorSids.intersection(currentEstu)
                effectiveWidth = Utilities.inverse_herfindahl(sqrtCap[estuFactorSids])
                logging.debug('Factor %s has %.2f effective assets (%d)', fname, effectiveWidth, len(estuFactorSids))

                # Pad thin factors if possible
                if effectiveWidth < minFactorWidth:

                    # Get list of assets that can be used for padding and sort by size
                    spareFactorSids = factorSids.difference(currentEstu)
                    if len(spareFactorSids)< 1:
                        hasThinFactors = True
                        continue
                    sorted_idx = list(mcap[spareFactorSids].sort_values(ascending=True).index)

                    # Add next-largest asset to estu and recaculate factor width
                    n = 0
                    while effectiveWidth < minFactorWidth and len(sorted_idx) > 0:
                        n += 1
                        nextID = sorted_idx.pop()
                        estuFactorSids.add(nextID)
                        prevWidth = effectiveWidth
                        effectiveWidth = Utilities.inverse_herfindahl(sqrtCap[estuFactorSids])

                        # Check for convergence criterion agreement
                        if prevWidth != 0.0 and (effectiveWidth/prevWidth - 1.0) < cutOff:
                            sorted_idx = set()

                    currentEstu = currentEstu.union(estuFactorSids)
                    logging.debug('Padded %s (%.2f/%d) with %d additional assets',
                            fname, effectiveWidth, len(estuFactorSids), n)

                    # Check whether factor still thin
                    if effectiveWidth < minFactorWidth:
                        hasThinFactors = True

        logging.debug('pump_up_factors: end')
        return currentEstu, hasThinFactors

    def grandfather(self, estu, baseEstu=None, addDays=1, daysBack=61, estuInstance=None, grandfatherRMS_ID=None):
        """Apply grandfathering rules to improve stability.
        Assets must 'qualify' for ESTU membership for a certain number of days before they are genuinely added
        """
        logging.debug('grandfather: begin')

        # Set up universe
        all_indices = range(len(self.assets))

        # Create list of dates for consideration
        dateList = self.modelDB.getDates(self.rmClass.rmg, self.date, daysBack, excludeWeekend=True)
        dateList = dateList[:-1]
        addDays = min(addDays, len(dateList)+1)

        # Determine relevant model's rms ID
        if grandfatherRMS_ID is not None:
            rms_id = grandfatherRMS_ID
        else:
            rms_id = self.rmClass.rms_id

        # Pull up a history of assets' ESTU eligibility (1/0)
        logging.info('Getting grandfather history from RMS: %d', rms_id)
        qualifiedAssets = self.modelDB.loadESTUQualifyHistory(\
                rms_id, self.assets, dateList, estuInstance=estuInstance, returnDF=True)

        # Find assets that meet the criteria for qualification now
        qualifiedAssets = qualifiedAssets.fillna(0.0).sum(axis=1)
        addDays = min(addDays, qualifiedAssets.max()+1)
        idxQualified = set(qualifiedAssets[qualifiedAssets>=addDays].index)
        if baseEstu != None:
            idxQualified = idxQualified.intersection(baseEstu)

        # Combine sets of assets to get final estimation universe
        qualifiedToday = set(estu)
        nAdd = len(idxQualified.difference(estu))
        estu = estu.union(idxQualified)
        logging.debug('Added %d assets (qualified at least %d out of %d days)', nAdd, addDays, qualifiedAssets.max())
        logging.debug('ESTU now contains %d assets (%d qualified)', len(estu), len(qualifiedToday))

        logging.debug('grandfather: end')
        return estu, qualifiedToday

    def include_by_market_classification(self, clsMember, clsFamily, clsCodesList, baseEstu=None):
        """Include assets based on classification -- asset type, exchange of quotation, and so forth.
        Assets belonging to classifications with names clsCodeList are the ones kept.
        """
        logging.debug('exclude_by_market_classification: begin')

        # Initialise
        if baseEstu is not None:
            univ = set(baseEstu)
        else:
            univ = set(self.assets)

        # Load the relevant classification
        hcClass = AssetProcessor.get_home_country(univ, self.date, self.modelDB, self.marketDB, clsType='HomeCountry')

        # Find those assets that qualify
        dropList = set()
        for sid, hc in hcClass.items():
            hcCode = hc.classification.code
            if hcCode not in clsCodesList:
                dropList.add(sid)

        msg = '%d out of %d assets classified as %s or missing' \
                % (len(univ)-len(dropList), len(univ), ','.join(clsCodesList))
        logging.debug(msg)

        estu = univ.difference(dropList)
        logging.debug('exclude_by_market_classification: end')
        return estu

    def exclude_sparsely_traded_assets_legacy(\
            self, assetScoreDict, baseEstu=None, minNonMissing=0.95, minNonZero=0.5):
        """Exclude assets with a high proportion of missing (zero) returns
        """
        logging.debug('exclude_sparsely_traded_assets_legacy: begin')

        # Initialise
        if baseEstu is not None:
            estu = set(baseEstu)
        else:
            estu = set(self.assets)
        if not hasattr(self.modelDB, 'subidMapCache'):
            self.modelDB.subidMapCache = ModelDB.FromThruCache(useModelID=False)
         
        # Load returns - fill all masked values
        pctNonMissingRets = assetScoreDict.loc[estu, 'ISC_Ret_Score'].fillna(0.0)
        pctNonZeroRets = assetScoreDict.loc[estu, 'ISC_Zero_Score'].fillna(0.0)
        
        # Remove assets trading on fewer than a certain proportion of days
        n = len(estu)
        dropIdx = set(pctNonMissingRets[pctNonMissingRets<minNonMissing].index)
        logging.debug('Excluding %d assets with fewer than %d%% of returns trading',
                        len(dropIdx), int(100*minNonMissing))
        estu = estu.difference(dropIdx)
         
        # Remove assets with too many zero returns
        dropIdx2 = set(pctNonZeroRets[pctNonZeroRets<minNonZero].index).difference(dropIdx)
        logging.debug('Excluding %d assets with fewer than %d%% of returns non-zero',
                        len(dropIdx2), int(100*minNonZero))

        # Return final estimation universe
        estu = estu.difference(dropIdx2)
        if len(estu) < n:
            logging.debug('%d zero/illiquid assets dropped from estu (%d to %d)', n-len(estu), n, len(estu))
        return estu

    def exclude_sparsely_traded_assets(self, assetScoreDict, baseEstu=None, minGoodReturns=0.75):
        """Exclude assets with a high proportion of missing and zero returns
        """
        logging.debug('exclude_sparsely_traded_assets: begin')
        if baseEstu is not None:
            estu = set(baseEstu)
        else:
            estu = set(self.assets)

        # Load missing/zero returns scores
        pctNonMissingRets = assetScoreDict.loc[estu, 'ISC_Ret_Score'].fillna(0.0)
        pctNonZeroRets = assetScoreDict.loc[estu, 'ISC_Zero_Score'].fillna(0.0)
        pctOkReturns = pctNonMissingRets + pctNonZeroRets - 1.0
        pctOkReturns = pctOkReturns.clip(0.0, 1.0)

        # Remove assets trading on fewer than a certain proportion of days
        n = len(estu)
        dropIdx = set(pctOkReturns[pctOkReturns<minGoodReturns].index)
        logging.debug('Excluding %d assets with fewer than %d%% of returns trading', len(dropIdx), int(100*minGoodReturns))
        estu = estu.difference(dropIdx)

        # Return final estimation universe
        if len(estu) < n:
            logging.debug('%d zero/illiquid assets dropped from estu (%d to %d)', n-len(estu), n, len(estu))
        return estu

    def exclude_by_asset_type(self, assetData, includeFields=['all-com'], excludeFields=None, baseEstu=None):
        """"Include or exclude assets based on their Axioma asset type
        """
        logging.debug('exclude_by_asset_type: begin')
        type2Asset = assetData.getTypeToAsset()
        typeList = set(type2Asset.keys())
        if baseEstu is not None:
            univ = set(baseEstu)
        else:
            univ = set(self.assets)
        estu = set()
        excludeStocks = set()

        # First deal with what we wish to explicitly include
        if (includeFields is not None) and (len(includeFields) > 0):
            includeFields = set(includeFields)
            includeTypes = includeFields.intersection(typeList)
            if 'all-com' in includeFields:
                commonTypes = set([typ for typ in typeList is typ[:3].lower() == 'com'])
                includeTypes = includeTypes.union(commonTypes)
            if 'all-pref' in includeFields:
                prefTypes = set([typ for typ in typeList is typ[:4].lower() == 'pref'])
                includeTypes = includeTypes.union(prefTypes)
            for typ in includeTypes:
                estu = estu.union(set(type2Asset[typ]))
            estu = estu.intersection(univ)
            logging.debug('%d stocks marked for inclusion by type: %s', len(estu), includeTypes)
        else:
            estu = set(univ)
        
        # Next with that which we wish to explicitly exclude

        if (excludeFields is not None) and (len(excludeFields) > 0):
            excludeFields = set(excludeFields)
            excludeTypes = excludeFields.intersection(typeList)
            if 'all-pref' in excludeFields:
                prefTypes = set([typ for typ in typeList is typ[:4].lower() == 'pref'])
                excludeTypes = excludeTypes.union(prefTypes)
            for typ in excludeTypes:
                excludeStocks = excludeStocks.union(set(type2Asset[typ]))
            excludeStocks = excludeStocks.intersection(univ)
            logging.debug('%d stocks marked for exclusion by type: %s', len(excludeStocks), excludeTypes)

        # Return final estimation universe
        estu = estu.difference(excludeStocks)
        logging.debug('exclude_by_asset_type: end')
        return estu

    def exclude_by_market_type(self, assetData, includeFields=None, excludeFields=['OTC'], baseEstu=None):
        """"Exclude assets based on their market exchange type
        """
        logging.debug('exclude_by_market_type: begin')

        # Initialise
        if baseEstu is not None:
            univ = set(baseEstu)
        else:
            univ = set(self.assets)
        mkt2AssetMap = assetData.getMktTypeToAsset()
        estu = set()
        excludeStocks = set()

        # First deal with what we wish to explicitly include
        if includeFields is not None:
            for mkt in includeFields:
                estu = estu.union(set(mkt2AssetMap.get(mkt, set())))
            estu = estu.intersection(univ)
            logging.debug('%d stocks marked for inclusion by type: %s', len(estu), includeFields)
        else:
            estu = set(univ)

        # Next with that which we wish to explicitly exclude
        if excludeFields is not None:
            for mkt in excludeFields:
                excludeStocks = excludeStocks.union(set(mkt2AssetMap.get(mkt, set())))
            excludeStocks = excludeStocks.intersection(univ)
            logging.debug('%d stocks marked for exclusion by type: %s', len(excludeStocks), excludeFields)

        estu = estu.difference(excludeStocks)
        logging.debug('exclude_by_market_type: end')
        return estu

    def generate_eligible_universe(self, assetData, quiet=False):
        """Creates subset of eligible assets for consideration
        in estimation universes
        """
        logging.debug('generate_eligible_universe: begin')
        if quiet:
            origLoggingLevel = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)

        # Initialise
        universe = self.assets
        rmc = self.rmClass
        n = len(universe)
        stepNo = 0
        iscScores = None
        universalExcludeTypes = self.modelDB.getExcludeTypes(self.date, self.marketDB, excludeLogic='ESTU_EXCLUDE')
        logging.info('Eligible Universe currently stands at %d stocks', n)

        # Set some defaults if parameters not defined
        HomeCountry_List = rmc.elig_parameters.get('HomeCountry_List', [r.mnemonic for r in rmc.rmg])
        use_isin_country_Flag = rmc.elig_parameters['use_isin_country_Flag']
        assetTypes = rmc.elig_parameters['assetTypes']
        excludeTypes  = rmc.elig_parameters['excludeTypes']
        if assetTypes is None:
            assetTypes = []
        if excludeTypes is None:
            excludeTypes = rmc.etfAssetTypes + rmc.otherAssetTypes
        excludeTypes = [xt for xt in excludeTypes if xt not in universalExcludeTypes]

        # Remove assets from the exclusion table
        stepNo+=1
        logging.info('Step %d: Applying exclusion table', stepNo)
        estu = self.apply_exclusion_list(productExclude=rmc.productExcludeFlag)
        n = self.report_on_changes(n, estu)

        # Exclude assets with missing or unrecognised asset type
        stepNo+=1
        logging.info('Step %d: Checking for missing/unrecognised asset type', stepNo)
        invalidTypes = set(assetData.getTypeToAsset().keys()).difference(set(rmc.allAssetTypes))
        estu = self.exclude_by_asset_type(assetData, includeFields=None, excludeFields=invalidTypes, baseEstu=estu)
        n = self.report_on_changes(n, estu)

        # Universally excluded asset types
        stepNo+=1
        logging.info('Step %d: Exclude default asset types %s', stepNo, ','.join(universalExcludeTypes))
        estu = self.exclude_by_asset_type(assetData, includeFields=None, excludeFields=universalExcludeTypes, baseEstu=estu)
        n = self.report_on_changes(n, estu)

        # Pull out Chinese H shares and redchips to save for later
        stepNo+=1
        logging.info('Step %d: Getting list of Chinese assets: %s', stepNo, ','.join(rmc.intlChineseAssetTypes))
        chineseIntl = self.exclude_by_asset_type(\
                assetData, includeFields=rmc.intlChineseAssetTypes, excludeFields=None, baseEstu=estu)
        logging.info('... Found %d Chinese H-shares and Redchips', len(chineseIntl))

        # Remove assets classed as foreign via home market classification
        dr_indices = []
        if len(assetData.foreign) > 0:
            stepNo+=1
            logging.info('Step %d: Excluding foreign listings by home country mapping', stepNo)
            estu = estu.difference(assetData.foreign)
            n = self.report_on_changes(n, estu)

        # Remove assets classed as foreign via home market classification - part 2
        stepNo+=1
        if len(HomeCountry_List) < 4:
            logging.info('Step %d: Excluding foreign listings not in %s market(s) by market classification',
                    stepNo, ','.join(HomeCountry_List))
        else:
            logging.info('Step %d: Excluding foreign listings not in %d markets by market classification',
                    stepNo, len(HomeCountry_List))
        estu = self.include_by_market_classification('HomeCountry', 'REGIONS', HomeCountry_List, baseEstu=estu)
        n = self.report_on_changes(n, estu)

        # Add back selected foreign common stock with no listings on other markets
        if len(assetData.foreign) > 0:
            stepNo+=1

            # Find number of individual markets per company
            marketOverLap = set()
            assetRMGMap = Utilities.flip_dict_of_lists(assetData.rmgAssetMap)
            assetTradingRMGMap = Utilities.flip_dict_of_lists(assetData.tradingRmgAssetMap)
            for (groupID, sidList) in assetData.getSubIssueGroups().items():
                homeMarkets = set([assetRMGMap[sid] for sid in sidList])
                tradeMarkets = set([assetTradingRMGMap[sid] for sid in sidList])
                if len(homeMarkets.intersection(tradeMarkets)) > 0:
                    marketOverLap = marketOverLap.union(sidList)

            # Get list of foreign-traded issues linked to only one market
            allowedList = set()
            for (ky, sidList) in assetData.getTypeToAsset().items():
                if ky in rmc.commonStockTypes + rmc.intlChineseAssetTypes:
                    allowedList = allowedList.union(sidList.intersection(assetData.foreign))
            allowedList = allowedList.difference(marketOverLap)

            # Get the scores for assets eligible to be added back
            iscScores = rmc.load_ISC_Scores(self.date, assetData, self.modelDB, self.marketDB, returnDF=True)

            # Check for duplicates and take either the largest, or give automatic favour to H-shares where relevant
            for (groupID, sidList) in assetData.getSubIssueGroups().items():
                duplic = set(sidList).intersection(allowedList)
                if len(duplic) > 1:
                    allowedList = allowedList.difference(duplic)
                    chIntlDuplic = chineseIntl.intersection(duplic)
                    if len(chIntlDuplic) > 0:
                        keepSid = iscScores[chIntlDuplic].idxmax()
                    else:
                        keepSid = iscScores[duplic].idxmax()
                    allowedList.add(keepSid)

            logging.info('Step %d: Adding back %d foreign-traded assets with no listings on their home market',
                    stepNo, len(allowedList))
            estu = estu.union(allowedList)
            n = self.report_on_changes(n, estu)

        # Remove various types of DRs and foreign listings
        stepNo+=1
        logging.info('Step %d: Exclude DR asset types %s', stepNo, ','.join(rmc.drAssetTypes))
        estu = self.exclude_by_asset_type(assetData, includeFields=None, excludeFields=rmc.drAssetTypes, baseEstu=estu)
        n = self.report_on_changes(n, estu)

        # Remove cloned assets
        hardCloneMap = assetData.getCloneMap(cloneType='hard')
        if len(hardCloneMap) > 0:
            stepNo+=1
            logging.info('Step %d: Removing cloned assets', stepNo)
            estu = estu.difference(set(hardCloneMap.keys()))
            n = self.report_on_changes(n, estu)

        # Include by asset type field
        stepNo+=1
        logging.info('Step %d: Include by asset types %s', stepNo, ','.join(assetTypes))
        estu = self.exclude_by_asset_type(assetData, includeFields=assetTypes, excludeFields=None, baseEstu=estu)
        n = self.report_on_changes(n, estu)

        # Exclude by asset type field
        if len(excludeTypes) > 0:
            stepNo+=1
            logging.info('Step %d: Exclude model-specific asset types %s', stepNo, ','.join(excludeTypes))
            estu = self.exclude_by_asset_type(assetData, includeFields=None, excludeFields=excludeTypes, baseEstu=estu)
            n = self.report_on_changes(n, estu)

        # Remove trusts, funds and other odds and ends
        stepNo+=1
        fundTypes = [fType for fType in rmc.fundAssetTypes if fType not in assetTypes]
        logging.info('Step %d: Exclude fund/trust asset types %s', stepNo, ','.join(fundTypes))
        estu = self.exclude_by_asset_type(assetData, includeFields=None, excludeFields=fundTypes, baseEstu=estu)
        n = self.report_on_changes(n, estu)

        # Remove Chinese A and B shares
        if rmc.elig_parameters.get('remove_China_AB', True):
            stepNo+=1
            logging.info('Step %d: Exclude Chinese asset types %s', stepNo, ','.join(rmc.localChineseAssetTypes))
            estu = self.exclude_by_asset_type(assetData, includeFields=None, excludeFields=rmc.localChineseAssetTypes, baseEstu=estu)
            n = self.report_on_changes(n, estu)

        # Manually add H-shares and Red-Chips back into list of eligible assets
        if rmc.elig_parameters.get('addBack_H_DR', True):
            stepNo+=1
            logging.info('Step %d: Adding back %d H-Shares and Redchips', stepNo, len(chineseIntl))
            estu = estu.union(chineseIntl)
            n = self.report_on_changes(n, estu)

            # Get the scores for assets eligible to be added back
            if iscScores is None:
                iscScores = rmc.load_ISC_Scores(self.date, assetData, self.modelDB, self.marketDB, returnDF=True)

            # Find DRs with no underlying asset
            drNoUnderlying = set([dr for (dr, under) in assetData.getDr2UnderMap().items() if under is None])

            # Exclude singleton DRs where they are from a company with other valid types
            invalidTypes = rmc.drAssetTypes + rmc.otherAssetTypes + rmc.etfAssetTypes
            for (groupID, sidList) in assetData.getSubIssueGroups().items():
                if len(sidList) > 1:
                    sidListNoDR = [sid for sid in sidList if assetData.getAssetType()[sid] not in invalidTypes]
                    if len(sidListNoDR) > 0:
                        drNoUnderlying = drNoUnderlying.difference(sidList)

            # Keep highest scoring singleton DR
            for (groupID, sidList) in assetData.getSubIssueGroups().items():
                duplic = set(sidList).intersection(drNoUnderlying)
                if len(duplic) > 1:
                    drNoUnderlying = drNoUnderlying.difference(duplic)
                    keepSid = iscScores[duplic].idxmax()
                    drNoUnderlying.add(keepSid)

            # Add remaining singleton DRs back to eligible universe
            if len(drNoUnderlying) > 0:
                stepNo+=1
                logging.info('Step %d: Adding back %d DRs with no underlying', stepNo, len(drNoUnderlying))
                estu = estu.union(drNoUnderlying)
                n = self.report_on_changes(n, estu)

        # Logic to selectively filter RTS/MICEX Russian stocks
        rmgRU = self.modelDB.getRiskModelGroupByISO('RU')
        if rmgRU in rmc.rmg:
            stepNo+=1
            logging.info('Step %d: Cleaning up Russian assets', stepNo)

            # Get lists of RTS-qouted assets
            rtsIdx = self.exclude_by_market_type(assetData, includeFields='RTS', excludeFields=None, baseEstu=estu)
            if len(rtsIdx) > 0:

                # Find companies with multiple listings which include lines on both RTS and MICEX
                micIdx = self.exclude_by_market_type(assetData, includeFields='MIC', excludeFields=None, baseEstu=estu)
                rtsDropList = set()
                for (groupID, sidList) in assetData.getSubIssueGroups().items():
                    if len(sidList) > 1:
                        micOverLap = micIdx.intersection(sidList)
                        if len(micOverLap) > 0:
                            rtsDropList = rtsDropList.union(rtsIdx.intersection(sidList))
                estu = estu.difference(rtsDropList)
                n = self.report_on_changes(n, estu)

        # Remove preferred stock except for selected markets
        prefMarketList = ['BR','CO']
        prefAssetDict = dict()
        for rmg in rmc.rmg:
            # Find assets in our current estu that are in the relevant markets
            if rmg.mnemonic in prefMarketList:
                prefAssetDict[rmg] = assetData.rmgAssetMap[rmg].intersection(estu)

        stepNo+=1
        logging.info('Step %d: Dropping preferred stocks', stepNo)
        # Identify preferred stock in general
        prefIdx = self.exclude_by_asset_type(assetData, includeFields=rmc.preferredStockTypes, excludeFields=None, baseEstu=estu)

        if len(prefAssetDict) > 0:
            # Find which of our subset of assets are preferred stock
            for rmg, rmgAssets in prefAssetDict.items():
                logging.info('...Adding back preferred stocks on market: %s', rmg.mnemonic)
                okPref = self.exclude_by_asset_type(\
                        assetData, includeFields=rmc.preferredStockTypes, excludeFields=None, baseEstu=rmgAssets)
                prefIdx = prefIdx.difference(okPref)

        # Add back in allowed preferred stock
        estu = estu.difference(prefIdx)
        n = self.report_on_changes(n, estu)

        # Get rid of assets on exchanges we don't want
        exchangeCodes = ['REG','TKS-ETF'] + AssetProcessor.connectExchanges
        if self.date.year > 2009:
            exchangeCodes.extend(['XSQ','OFE','XLF']) # GB junk
        for xCode in exchangeCodes:
            stepNo+=1
            logging.info('Step %d: Removing stocks on exchange: %s', stepNo, xCode)
            estu = self.exclude_by_market_type(assetData, includeFields=None, excludeFields=[xCode], baseEstu=estu)
            n = self.report_on_changes(n, estu)

        # Limit stocks to certain exchanges in select countries
        mktExchangeMap = {'CA': ['TSE'],
                          'JP': ['TKS-S1','TKS-S2','OSE-S1','OSE-S2','TKS-MSC','TKS-M','JAS']}
        if self.date.year > 2009:
            mktExchangeMap['US'] = ['NAS','NYS','IEX','EXI']
        for mkt, exchgList in mktExchangeMap.items():
            stepNo+=1
            logging.info('Step %d: Dropping %s stocks NOT on exchanges: %s', stepNo, mkt, ','.join(exchgList))

            # Pull out subset of assets on relevant market
            rmgMkt = self.modelDB.getRiskModelGroupByISO(mkt)
            baseEstu = assetData.rmgAssetMap.get(rmgMkt, set()).intersection(estu)
            if len(baseEstu) < 1:
                continue

            # Remove assets not on approved list of markets
            nonMktEstu = self.exclude_by_market_type(assetData, includeFields=None, excludeFields=exchgList, baseEstu=baseEstu)
            estu = estu.difference(nonMktEstu)
            n = self.report_on_changes(n, estu)

        # Exclude where original home country not in HomeCountry_List
        if len(assetData.foreign) > 0:
            stepNo+=1
            logging.info('Step %d: Excluding assets with original home market not in the model', stepNo)

            # Load original home country classification objects
            hcClass = AssetProcessor.get_home_country(estu, self.date, self.modelDB, self.marketDB, clsType='HomeCountry')

            # Remove invalid assets from estu
            for sid, hc in hcClass.items():
                hcCode = hc.classification.code
                if hcCode not in HomeCountry_List:
                    estu.discard(sid)

            n = self.report_on_changes(n, estu)

        # Report on final state and return data
        assetData.eligibleUniverse = estu
        mcap_elig = assetData.marketCaps[estu].sum(axis=None)
        if quiet:
            logging.getLogger().setLevel(origLoggingLevel)
        logging.info('%d eligible assets out of %d total, %.2f tr %s market cap',
                len(estu), len(universe), mcap_elig / 1e12, rmc.numeraire.currency_code)
        logging.debug('generate_eligible_universe: end')

        # Back compatibility bit
        assetIDxMap = dict(zip(assetData.universe, range(len(assetData.universe))))
        assetData.eligibleUniverseIdx = [assetIDxMap[sid] for sid in assetData.eligibleUniverse]

        return estu

    def estimation_universe_reporting(self, repData, exposureMatrix):
        # Write various files for debugging and reporting

        if not self.debugOutput:
            return
        # Shorthand
        estU = repData.estimationUniverse
        estuStr = [sid.getSubIDString() for sid in estU]
        univ = repData.universe
        univStr = [sid.getSubIDString() for sid in univ]
        assetIdxMap = dict(zip(univ, range(len(univ))))
        estuIdx = [assetIdxMap[sid] for sid in estU]
        mcaps = repData.marketCaps

        if not hasattr(repData, 'mCapDF'):
            repData.mCapDF = AssetProcessor.computeTotalIssuerMarketCaps(
                    self.date, mcaps, self.rmClass.numeraire, self.modelDB, self.marketDB, debugReport=self.debugOutput)

        # Output weights
        mcap_ESTU = numpy.array(mcaps[estU].values*100, int) / 100.0
        outName = 'tmp/estuWt-%s-%s.csv' % (self.rmClass.mnemonic, self.date)
        Utilities.writeToCSV(mcap_ESTU, outName, rowNames=estuStr)

        # Output type of each asset
        estuTypeMap = pandas.Series(repData.getAssetType())[estU]
        AssetProcessor.dict_to_csv(
                    'tmp/estuTypes-%s-%s.csv' % (self.rmClass.mnemonic, self.date), estuTypeMap)

        # Output list of available types in the estu
        typeCount = estuTypeMap.value_counts()
        AssetProcessor.dict_to_csv(
                    'tmp/AssetTypes-ESTU-%s-%s.csv' % (self.rmClass.mnemonic, self.date), typeCount)

        # Output map to trading and home markets
        mkt_df = AssetProcessor.get_all_markets(\
                repData.universe, self.date, self.modelDB, self.marketDB, tradeDict=repData.tradingRmgAssetMap)
        mkt_df.index = [sid.getSubIDString() for sid in mkt_df.index]
        mkt_df.to_csv('tmp/marketMap-%s-%s.csv' % (self.rmClass.mnemonic, self.date))

        # Output DR currencies
        if repData.drCurrData is not None:
            outName = 'tmp/drCurr-%s-%s.csv' % (self.rmClass.mnemonic, self.date)
            AssetProcessor.dict_to_csv(outName, pandas.Series(repData.drCurrData))

        # Output identifier mapping file for Axioma ID types
        modelIDList = [sid.getModelID().getIDString() for sid in univ]
        publicIDList = [sid.getModelID().getPublicID() for sid in univ]
        mod2mktIDMap = self.modelDB.getIssueMapPairs(self.date)
        mod2mktIDMap = dict((i,j) for (i,j) in mod2mktIDMap)
        mktIDList = [mod2mktIDMap[sid.getModelID()] for sid in univ]
        mktIDStrList = [mid.getIDString() for mid in mktIDList]
        mktIDPubList = [mid.getPublicID() for mid in mktIDList]
        id_DF = pandas.DataFrame(list(zip(modelIDList, publicIDList, mktIDStrList, mktIDPubList)),
                index=univStr, columns=['modelID', 'publicID', 'mktID', 'mktPublicID'])
        id_DF.to_csv('tmp/axioma-ID-%s-%s.csv' % (self.rmClass.mnemonic, self.date))

        # Output estimation universe exposure matrix
        exposureMatrix.dumpToFile('tmp/estu-expM-%s-%04d%02d%02d.csv'\
                % (self.rmClass.name, self.date.year, self.date.month, self.date.day),
                self.modelDB, self.marketDB, self.date, estu=estuIdx, assetData=repData)

        # Output estu composition by country/industry (if relevant)
        for fType in [ExposureMatrix.CountryFactor, ExposureMatrix.IndustryFactor]:
            factorIdxList = exposureMatrix.getFactorIndices(fType)
            if len(factorIdxList) < 1:
                continue
            factorNameList = exposureMatrix.getFactorNames(fType)
            exp_ESTU = ma.take(exposureMatrix.getMatrix(), estuIdx, axis=1)
            mCapList = []
            numList = []
            # Loop round set of factors
            for (j,i) in enumerate(factorIdxList):
                assetIdx = numpy.flatnonzero(ma.getmaskarray(exp_ESTU[i])==0)
                factorMCap = numpy.take(mcap_ESTU, assetIdx, axis=0)
                factorMCapSum = ma.filled(ma.sum(factorMCap, axis=None), 0.0)
                mCapList.append(factorMCapSum)
                numList.append(len(factorMCap))
                logging.info('Factor %s has %d assets', factorNameList[j], len(assetIdx))
            outName = 'tmp/estuMCap-%s-%s.csv' % (fType.name, self.date)
            mCapList = mCapList / numpy.sum(mCapList, axis=None)
            Utilities.writeToCSV(numpy.transpose(numpy.array((mCapList, numList))),
                outName, rowNames=factorNameList, columnNames=['mcap','N'])

        # Look at the various levels of market cap
        capArray = repData.mCapDF.loc[univ, ['marketCap', 'totalCap', 'dlcCap']]
        outName = 'tmp/mCap-%s-%s.csv' % (self.rmClass.mnemonic, self.date)
        colNames = ['mcap', 'issuer', 'DLC']
        Utilities.writeToCSV(Utilities.df2ma(capArray), outName, rowNames=univStr, columnNames=colNames, dp=4)

        # Output list of clones
        hardCloneMap = repData.getCloneMap(cloneType='hard')
        if len(hardCloneMap) > 0:
            fileName = 'tmp/cloneList-%s.csv' % str(self.date)
            outFile = open(fileName, 'w')
            outFile.write('Slave,Master,\n')
            for slv in sorted(hardCloneMap.keys()):
                outFile.write('%s,%s,\n' % (slv.getSubIdString(), hardCloneMap[slv].getSubIdString()))
            outFile.close()

        # Output DR to Underlying map
        dr2Underlying = repData.getDr2UnderMap()
        assetTypeDict = repData.getAssetType()
        if len(dr2Underlying) > 0:
            ax2MktIDMap = self.modelDB.getIssueMapPairs(self.date)
            ax2MktIDMap = dict((i.getIDString(),j.getIDString()) for (i,j) in ax2MktIDMap)
            fileName = 'tmp/dr_2_under-%s-%s.csv' % (self.rmClass.mnemonic, str(self.date))
            outFile = open(fileName, 'w')
            outFile.write('DR,,Type,Underlying,,Type,N,\n')

            # Get count of DRs that each underlying is mapped to
            countDict = defaultdict(list)
            for (dr, und) in dr2Underlying.items():
                if und is not None:
                    countDict[und].append(dr)

            # Output DRs and underlyings to file
            for dr in sorted(dr2Underlying.keys()):
                und = dr2Underlying[dr]
                drMktId = ax2MktIDMap.get(dr.getModelID().getIDString(), None)
                if und is None:
                    outFile.write('%s,%s,%s,None,None,1,\n' % (dr.getSubIdString(), drMktId, assetTypeDict[dr]))
                else:
                    undMktId = ax2MktIDMap.get(und.getModelID().getIDString(), None)
                    outFile.write('%s,%s,%s,%s,%s,%s,%d,\n' % \
                            (dr.getSubIdString(), drMktId, assetTypeDict[dr],
                             und.getSubIdString(), undMktId, assetTypeDict.get(und, None), len(countDict[und])))
            outFile.close()

        # Get list of issuers with more than one line of stock in the estu
        multIssueDict = dict()
        for (groupID, sidList) in repData.getSubIssueGroups().items():
            intersectList = set(sidList).intersection(set(estU))
            if len(intersectList) > 1:
                multIssueDict[groupID] = intersectList
        if len(multIssueDict) > 0:
            logging.info('%d issuers with more than one line in the estu', len(multIssueDict))
        fileName = 'tmp/estu-multiples-%s.csv' % str(self.date)
        outFile = open(fileName, 'w')
        outFile.write(',CID,Name,Type,Market\n')
        for cid in sorted(multIssueDict.keys()):
            for sid in sorted(multIssueDict[cid]):
                outFile.write('%s,%s,%s,%s,%s,\n' % \
                    (sid.getSubIDString(), cid, repData.getNameMap().get(sid, None),
                        repData.getAssetType()[sid], repData.getMarketType().get(sid, None)))
        outFile.close()

        # Report on nursery universe
        if len(repData.nurseryUniverse) > 0:
            logging.info('Nursery universe: %d assets (%d before reassignment)',
                    len(repData.nurseryUniverse), len(repData.originalNurseryUniverse))
            fileName = 'tmp/nursery-%s.csv' % str(self.date)
            outFile = open(fileName, 'w')
            outFile.write(',CID,Name,Type,Market\n')
            for sid in sorted(repData.nurseryUniverse):
                outFile.write('%s,%s,%s,%s,%s,\n' % \
                    len(repData.nurseryUniverse), len(repData.originalNurseryUniverse))
            fileName = 'tmp/nursery-%s.csv' % str(self.date)
            outFile = open(fileName, 'w')
            outFile.write(',CID,Name,Type,Market\n')
            for sid in sorted(repData.nurseryUniverse):
                outFile.write('%s,%s,%s,%s,%s,\n' % \
                        (sid.getSubIDString(), repData.getSubIssue2CidMapping()[sid], repData.getNameMap().get(sid, None),
                         repData.getAssetType()[sid], repData.getMarketType().get(sid, None)))
            outFile.close()
            fileName = 'tmp/nursery-orig-%s.csv' % str(self.date)
            outFile = open(fileName, 'w')
            outFile.write(',CID,Name,Type,Market\n')
            for sid in sorted(repData.originalNurseryUniverse):
                outFile.write('%s,%s,%s,%s,%s,\n' % \
                        (sid.getSubIDString(), repData.getSubIssue2CidMapping()[sid], repData.getNameMap().get(sid, None),
                            repData.getAssetType()[sid], repData.getMarketType().get(sid, None)))

        # Report on ETFs
        statETFs = set([sid for sid in repData.universe if repData.getAssetType()[sid] in self.rmClass.etfAssetTypes])
        if len(statETFs) > 0:
            logging.info('Non-equity ETFs: %d assets', len(statETFs))

        # Report on Chinese A-Shares
        aShareUniv = set([sid for sid in repData.universe if repData.getAssetType()[sid] in self.rmClass.localChineseAssetTypes])
        if len(aShareUniv) > 0:
            logging.info('Chinese A and B shares: %d assets', len(aShareUniv))

        # Report on core universe
        coreUniv = set(repData.universe).difference(set(repData.nurseryUniverse))
        coreUniv = (coreUniv.difference(statETFs)).difference(aShareUniv)
        logging.info('Core universe: %d assets', len(coreUniv))
        return

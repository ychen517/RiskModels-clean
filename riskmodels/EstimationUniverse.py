import logging
import numpy.ma as ma
import numpy
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities
from riskmodels import ModelDB
from riskmodels import ProcessReturns

class ConstructEstimationUniverse:
    def __init__(self, assets, modelSelector, modelDB, marketDB):
        """ Generic class for estU construction
        """
        self.modelSelector = modelSelector
        self.assets = assets
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.assetIdxMap = dict(zip(assets, list(range(len(assets)))))
        logging.debug('%d assets in full universe', len(assets))

    def report_estu_content(self, mcap, estuIdx, stepName='unknown'):
        """Give a report on size of current estimation universe
        """
        mcap_estu = ma.take(mcap, estuIdx, axis=0)
        mcap_estu = ma.sum(mcap_estu, axis=None) / 1.0e9
        logging.debug('ESTU after %s step: n: %d, mcap: %.2f Bn',
                stepName, len(estuIdx), mcap_estu)

    def exclude_specific_assets(self, exclIndices, baseEstu=None):
        """Explicitly exclude asests corresponding to index
        positions specified by exclIndices.
        """
        logging.debug('exclude_specific_assets: begin')
        if baseEstu is None:
            baseEstu = list(range(len(self.assets)))
        estu = set(baseEstu).difference(exclIndices)
        logging.debug('%d excluded assets from list, %d in universe',
                len(exclIndices), len(baseEstu) - len(estu))
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_specific_assets: end')
        return (list(estu), list(nonest))

    def apply_exclusion_list(self, modelDate, baseEstu=None, productExclude=True):
        """Exclude assets in the RMS_ESTU_EXCLUDED table.
        """
        logging.debug('apply_exclusion_list: begin')
        if baseEstu is None:
            baseEstu = list(range(len(self.assets)))
        exclIssues = self.modelDB.getRMSESTUExcludedIssues(
                                self.modelSelector.rms_id, modelDate)
        if len(exclIssues) > 0:
            mdlIdxMap = dict([(j.getModelID(),idx) for (idx,j)
                in enumerate(self.assets)])
            exclIndices= [mdlIdxMap[mdl] for mdl in exclIssues
                if mdl in mdlIdxMap]
            logging.debug('%d excluded assets from RMS_ESTU_EXCLUDED, %d in universe',
                    len(exclIssues), len(self.assets))
            baseEstu = set(baseEstu).difference(exclIndices) 

        if productExclude:
            productExcludes = self.modelDB.getProductExcludedSubIssues(modelDate)
            excludeIdx = set([self.assetIdxMap[sid] for sid in productExcludes if sid in self.assetIdxMap])
            logging.info('%d excluded assets from PRODUCT_EXCLUDE, %d in universe',
                    len(excludeIdx), len(self.assets))
            baseEstu = set(baseEstu).difference(excludeIdx)

        nonest = set(range(len(self.assets))).difference(baseEstu)
        logging.debug('apply_exclusion_list: end')
        return (list(baseEstu), list(nonest))

    def exclude_missing_exposures(self, data, factorTypeList, modelDate, excludeFactorNames=None):
        """ Exclude assets with missing exposures
        """
        logging.debug('exclude_missing_exposures: begin')
        estu = range(len(self.assets))
        logging.debug('%d assets in universe', len(estu))

        expM = data.exposureMatrix
        sumExposures = ma.zeros((expM.getMatrix().shape[1]))
        assert(len(sumExposures)==len(estu))

        # Get Ids for excluded factors
        if not excludeFactorNames:
            excludeFactorNames = []
        excludeFactorsIdx = [expM.getFactorIndex(f) \
                for f in excludeFactorNames if f in expM.getFactorNames()]

        # Seek out dummy style factors and add to exclude list
        if ExposureMatrix.StyleFactor in factorTypeList:
            for i in expM.getFactorIndices(ExposureMatrix.StyleFactor):
                exposures = expM.getMatrix()[i]
                exposures = ma.take(exposures, numpy.flatnonzero(
                    ma.getmaskarray(exposures)==0), axis=0 )
                freq = numpy.unique(exposures.filled(0.0))
                if len(freq) < 3:
                    logging.debug('%s identified as dummy factor, excluding',
                            expM.getFactorNames()[i])
                    excludeFactorsIdx.append(i)

        # Check each factor type
        for fType in factorTypeList:
            assert(fType in expM.factorTypes_)
            factorsIdx = [i for i in expM.getFactorIndices(fType) \
                                if i not in excludeFactorsIdx]
            if fType == ExposureMatrix.StyleFactor:
                for i in factorsIdx:
                    sumExposures += expM.getMatrix()[i] 
            else:
                sumType = ma.sum(ma.take(expM.getMatrix(), factorsIdx, 
                            axis=0), axis=0)
                sumExposures += sumType

            # Extract only those assets with no missing exposures
            estu = numpy.flatnonzero(ma.getmaskarray(sumExposures)==0)
            logging.debug('%d out of %d assets with all %s exposures',
                    len(estu), len(self.assets), fType.name)

        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_missing_exposures: end')
        return (list(estu), list(nonest))

    def exclude_by_style_ranking(self, data, factorNames, baseEstu=None,
                             byFactorType=None, lower_pctile=5, upper_pctile=100):
        """ Exclude assets outside particular style-factor criteria
        """
        logging.debug('exclude_by_style_ranking: begin')

        # initialise stuff
        expM = data.exposureMatrix
        exposures = expM.getMatrix()
        factorIndices = [expM.getFactorIndex(f) for f in factorNames]
        if not byFactorType:
            byFactorIdxList = [0]
            byFactorNameList = ['Total']
            byExposures = numpy.ones((expM.getMatrix().shape[1]))[numpy.newaxis,:]
        else:
            assert(byFactorType in expM.factorTypes_)
            byFactorIdxList = expM.getFactorIndices(byFactorType)
            byFactorNameList = expM.getFactorNames(byFactorType)
            byExposures = expM.getMatrix()

        # Extract eligible assets
        if baseEstu == None:
            estu_filter = ma.ones(exposures.shape[1])
        else:
            estu_filter = Matrices.allMasked((exposures.shape[1]))
            ma.put(estu_filter, baseEstu, 1)

        running_filter = Matrices.allMasked((exposures.shape[1]))

        # Loop round by factors
        for (j,i) in enumerate(byFactorIdxList):
            # Loop round filter factors
            for k in factorIndices:
                filterExposures = ma.masked_where(
                                ma.getmaskarray(byExposures[i]), exposures[k])
                indices = numpy.flatnonzero(
                                ma.getmaskarray(filterExposures)==0)
                if len(indices) > 0:
                    # Keep only estu exposures
                    factorExp = ma.masked_where(
                                ma.getmaskarray(estu_filter), filterExposures)
                    # Mask exposures outside percentile bounds
                    exp_thresholds = Utilities.prctile(
                                factorExp, [lower_pctile, upper_pctile])
                    factorExp = ma.masked_where(
                                factorExp < exp_thresholds[0], factorExp)
                    factorExp = ma.masked_where(
                                factorExp > exp_thresholds[1], factorExp)

                    # Add IDs to tally
                    eligibleIdx = numpy.flatnonzero(
                                    ma.getmaskarray(factorExp)==0)
                    ma.put(running_filter, eligibleIdx, 1)
                    logging.debug('Ranking by %s factor for %s, %d assets qualify',
                            expM.getFactorNames()[k],byFactorNameList[j],
                            len(numpy.flatnonzero(ma.getmaskarray(factorExp)==0)))

        # Report on final state of estu
        estu_filter = ma.masked_where(
                    ma.getmaskarray(running_filter), estu_filter)
        estu = numpy.flatnonzero(ma.getmaskarray(estu_filter)==0)
        logging.debug('Ranked by %d factors, estu now consists of %d assets',
                    len(factorIndices), len(estu))
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_by_style_ranking: end')
        return (list(estu), list(nonest))

    def exclude_by_cap_ranking(self, data, date, baseEstu=None, byFactorType=False,
                               lower_pctile=25, upper_pctile=100, 
                               minFactorWidth=10, method='percentile',
                               excludeFactors=[], weight='mcap',
                               extraWeights=None):
        """ Exclude assets outside particular capitalisation bands
        """
        logging.debug('exclude_by_cap_ranking: begin')
        # initialise stuff
        expM = data.exposureMatrix
        mcap = ma.array(data.marketCaps, copy=True)
        if not byFactorType:
            factorIdxList = [0]
            factorNameList = ['Total']
            exposures = numpy.ones((len(mcap)))[numpy.newaxis,:]
        else:
            assert(byFactorType in expM.factorTypes_)
            factorIdxList = expM.getFactorIndices(byFactorType)
            factorNameList = expM.getFactorNames(byFactorType)
            exposures = expM.getMatrix()
        if extraWeights is not None:
            zeroMsk = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(extraWeights==0.0, extraWeights)))
            extraWeights = numpy.clip(extraWeights, 0.1, 1.0)
            numpy.put(extraWeights, zeroMsk, 0.000001)
            mcap *= extraWeights

        excludeFactorNames =[]
        if excludeFactors is not None:
            excludeFactorNames = [f.name for f in excludeFactors]

        # Extract eligible market caps
        if baseEstu is not None:
            filter = Matrices.allMasked((len(mcap)))
            ma.put(filter, baseEstu, 1)
            mcap = ma.masked_where(ma.getmaskarray(filter), mcap)
        if weight == 'rootCap':
            mcap = ma.sqrt(mcap)

        capRanking = []
        estu = []

        # Loop round set of factors
        for (j,i) in enumerate(factorIdxList):
            if factorNameList[j] in excludeFactorNames:
                capRanking.append(list())
                continue
            # Pick out eligible assets
            factorMCap = ma.masked_where(ma.getmaskarray(exposures[i]), mcap)
            eligibleIdx = numpy.flatnonzero(ma.getmaskarray(factorMCap)==0)
            if len(eligibleIdx)==0:
                logging.warning('%s: No assets in %s!', date, factorNameList[j])
                capRanking.append(list())
                continue
            sortedIdx = ma.argsort(-factorMCap)
            sortedIdx = sortedIdx[:len(eligibleIdx)]

            # Calculate number of assets exposed to factor according
            # to Herfindahl methodology
            factorWgt = ma.sqrt(factorMCap)
            if ma.sum(factorWgt) > 0.0:
                factorWgt = factorWgt / ma.sum(factorWgt)
                effectiveWidth = 1.0 / ma.inner(factorWgt, factorWgt)
            else:
                effectiveWidth = 0.0

            # If factor is sufficiently populated, proceed
            if effectiveWidth > minFactorWidth and len(sortedIdx) > 0:
                # If not using percentiles, exclude via proportion
                if method == 'percentage':
                    targetCapRatio = (upper_pctile - lower_pctile) / 100.0
                    runningCapRatio = numpy.cumsum(ma.take(factorMCap, sortedIdx), axis=0)
                    runningCapRatio /= ma.sum(ma.take(factorMCap, sortedIdx))
                    reachedTarget = list(runningCapRatio >= targetCapRatio)
                    m = min(reachedTarget.index(True)+1, len(sortedIdx))
                    eligibleIdx = sortedIdx[:m]
                    if m == len(sortedIdx):
                        loCap = 0.0
                    else:
                        loCap = factorMCap[sortedIdx[m]]
                    factorMCap = ma.masked_where(factorMCap < loCap, factorMCap)
                else:
                    # Mask capitalisations outside percentile bounds
                    cap_thresholds = Utilities.prctile(factorMCap,\
                            [lower_pctile, upper_pctile])
                    factorMCap = ma.masked_where(
                            factorMCap < cap_thresholds[0], factorMCap)
                    factorMCap = ma.masked_where(
                            factorMCap > cap_thresholds[1], factorMCap)
                    eligibleIdx = numpy.flatnonzero(ma.getmaskarray(factorMCap)==0)

            # Add IDs to tally
            estu.extend(eligibleIdx)
            capRanking.append(ma.take(factorMCap, eligibleIdx, axis=0))

        # Show list of industries/countries and their respective caps
        factorCaps = numpy.array([ma.sum(caps) for caps in capRanking])
        for j in numpy.argsort(-factorCaps):
            fc = ma.sum(capRanking[j])
            capRanking[j].sort()
            if factorNameList[j] in excludeFactorNames:
                continue
            elif len(capRanking[j]) > 0:
                logging.debug('%s: %.2f bn %s (%d assets, %.2f%%, top 5: %.2f%%, top 1: %.2f%%)',
                            factorNameList[j], fc / 1e9,
                            self.modelSelector.numeraire.currency_code,
                            len(capRanking[j]), fc / numpy.sum(factorCaps) * 100.0,
                            ma.sum(capRanking[j][-5:]) / fc * 100.0,
                            capRanking[j][-1] / fc * 100.0)
            else:
                logging.debug('%s: %.2f bn %s (%d assets, %.2f%%, top 5: %.2f%%, top 1: %.2f%%)',
                            factorNameList[j], fc / 1e9,
                            self.modelSelector.numeraire.currency_code,
                            len(capRanking[j]), fc / numpy.sum(factorCaps) * 100.0, 0.0, 0.0)

        # Report on final state of estu
        mcap_ESTU = ma.sum(ma.take(mcap, estu)) / 1e12
        logging.debug('Subset contains %d assets, %.4f tr %s market cap',
                len(estu), mcap_ESTU, self.modelSelector.numeraire.currency_code)
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_by_cap_ranking: end')
        return (list(estu), list(nonest))

    def filter_by_user_score(self, data, date, scoreDict, baseEstu=None, lower_pctile=25, upper_pctile=100):
        """Filter out assets outside some percentile range of a user-specified list of scores
        """
        logging.debug('filter_by_user_score: begin')

        # Determine subset of eligible assets for comparison
        if baseEstu is None:
            baseEstu = range(len(self.assets))
        assets = [self.assets[i] for i in baseEstu]

        # Set up array of scores
        if type(scoreDict) is dict:
            scores = numpy.array([scoreDict.get(sid, 0.0) for sid in self.assets], float)
        else:
            scores = numpy.array(scoreDict, copy=True)

        # Mask scores for ineligible assets
        filter = Matrices.allMasked((len(self.assets)))
        ma.put(filter, baseEstu, 1)
        scores = ma.masked_where(ma.getmaskarray(filter), scores)

        # Mask scores outside percentile bounds
        cap_thresholds = Utilities.prctile(scores, [lower_pctile, upper_pctile])
        scores = ma.masked_where(scores < cap_thresholds[0], scores)
        scores = ma.masked_where(scores > cap_thresholds[1], scores)

        # Return list of asset indices
        estu = numpy.flatnonzero(ma.getmaskarray(scores)==0)
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('filter_by_user_score: end')
        return (list(estu), list(nonest))

    def filter_by_cap_and_volume(self, data, date,
                                 baseEstu=None, hiCapQuota=1000, loCapQuota=2000, 
                                 volDays=250, bufferFactor=1.2, downWeights=None):
        """Filter out a fixed number of stocks by capitalisation and volume.
        Attempts to identify suitable assets using size and liquidity at
        the same time, without having to rank by one criteria then imposing 
        the other as a penalty.
        Methodology is adapted from the Japanese TOPIX Index Series.
        """
        logging.debug('filter_by_cap_and_volume: begin')

        # Restrict universe to keep things manageable
        if baseEstu is None:
            baseEstu = range(len(self.assets))
        assets = [self.assets[i] for i in baseEstu]
        mcap = ma.take(data.marketCaps, baseEstu, axis=0)

        if hiCapQuota + loCapQuota >= len(assets):
            nonest = set(range(len(self.assets))).difference(baseEstu)
            logging.debug('filter_by_cap_and_volume: end')
            return (list(baseEstu), list(nonest))

        # Load volume data
        if self.modelDB.volumeCache.maxDates < volDays:
            logging.warning('ModelDB volume cache too small (%d days), resizing to %d days',
                    self.modelDB.volumeCache.maxDates, volDays)
            self.modelDB.setVolumeCache(volDays)
        dateList = self.modelDB.getDates(self.modelSelector.rmg, date, volDays-1)
        assetCurrMap = self.modelDB.getTradingCurrency(date, assets, self.marketDB)
        currencies = set(assetCurrMap.values())
        if len(currencies) > 1:
            baseCurrencyID = self.modelSelector.numeraire.currency_id
        else:
            baseCurrencyID = None
        volume = self.modelDB.loadVolumeHistory(dateList, assets, baseCurrencyID).data.filled(0.0)
        totalVol = numpy.sum(volume, axis=1)

        # Rank cap and volume
        if downWeights is not None:
            mcap = mcap * downWeights
            totalVol = totalVol * downWeights
        capRank = [self.assetIdxMap[assets[i]] for i in ma.argsort(-mcap)]
        volRank = [self.assetIdxMap[assets[i]] for i in ma.argsort(-totalVol)]

        # Pick out large, liquid stocks
        upperBound = int(hiCapQuota * bufferFactor)
        estu = set([i for i in capRank[:upperBound]\
                    if i in set(volRank[:upperBound])][:hiCapQuota])
        logging.debug('Using cap and volume to determine %d of %d large/mid-caps',
                        len(estu), hiCapQuota)

        # Attempt to pad out if we fall short; ie, opt for more liquid stocks
        if len(estu) < hiCapQuota:
            missing = hiCapQuota - len(estu)
            logging.debug('Using only volume to determine remaining %d large/mid-caps',
                        missing)
            alt = [i for i in volRank[:upperBound] if i not in estu][:missing]
            estu.update(alt)

        # Add smaller stocks
        upperBound = int((hiCapQuota + loCapQuota) * bufferFactor)
        nextLot = set([i for i in capRank[:upperBound] if i in set(
                    volRank[:upperBound]) and i not in estu][:loCapQuota])
        logging.debug('Using cap and volume to determine %d small-caps',
                    len(nextLot))
        estu.update(nextLot)
        if len(nextLot) < loCapQuota:
            missing = loCapQuota - len(nextLot)
            logging.debug('Using only volume to determine remaining %d small-caps',
                        missing)
            alt = [i for i in volRank if i not in estu][:missing]
            estu.update(alt)

        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('filter_by_cap_and_volume: end')
        return (list(estu), list(nonest))
    
    def pump_up_factors(self, data, date, currentEstu=None, baseEstu=None,
            byFactorType=[Matrices.ExposureMatrix.IndustryFactor],
            minFactorWidth=10, cutOff=0.01, excludeFactors=[], downWeights=None,
            returnFlag=False):
        """Attempts to inflate thin factors
        Routine will stop if increase in effective width of factor
        from adding another asset falls below value cutOff (default is 1%)
        Thus, to make the routine more stringent, increase this value
        """
        logging.debug('pump_up_factors: begin')
        hasThinFactors = False
        
        # Initialise
        expM = data.exposureMatrix
        mcap = numpy.array(data.marketCaps, copy=True)
        excludeFactorNames =[]
        if excludeFactors is not None:
            excludeFactorNames = [f.name for f in excludeFactors]
        if baseEstu == None:
            baseEstu = range(len(self.assets))
        if currentEstu == None:
            currentEstu = set()
        else:
            currentEstu = set(currentEstu)
        if downWeights is None:
            downWeights = numpy.ones((len(mcap)), float)

        # Mask unwanted assets
        filter = Matrices.allMasked((len(self.assets)))
        ma.put(filter, baseEstu, 1)
        ma.put(filter, list(currentEstu), 1)
        mcap = ma.masked_where(ma.getmaskarray(filter), mcap)

        # Else we attempt to inflate thin factors
        exposures = expM.getMatrix()
        # Loop round factor types
        for fType in byFactorType:
            assert(fType in expM.factorTypes_)
            factorIdxList = expM.getFactorIndices(fType)
            factorNameList = expM.getFactorNames(fType)
            # Loop round set of factors
            for (j,k) in enumerate(factorIdxList):
                if factorNameList[j] in excludeFactorNames:
                    continue
                # Pick out eligible assets
                factorMCap = ma.masked_where(ma.getmaskarray(exposures[k]), mcap)
                factorDW = ma.masked_where(ma.getmaskarray(exposures[k]), downWeights)
                factorIdx = numpy.flatnonzero(ma.getmaskarray(factorMCap)==0)
                if len(factorIdx) < 1:
                    logging.warning('Factor %s is empty', factorNameList[j])
                    hasThinFactors = True
                # Sort from largest to smallest
                sortedIdx = ma.argsort(-factorMCap*factorDW)
                sortedIdx = sortedIdx[:len(factorIdx)]
                # Sort into those already in the estu and those not
                estuFactorIdx = [i for i in sortedIdx if i in currentEstu]
                spareFactorIdx = [i for i in sortedIdx if i not in currentEstu]
                # Get cap of everything in factor which is also in the estu
                filter = Matrices.allMasked((len(self.assets)))
                if len(estuFactorIdx) > 0:
                    ma.put(filter, estuFactorIdx, 1)
                factorMCap = ma.masked_where(ma.getmaskarray(filter), factorMCap)

                if len(spareFactorIdx) > 0:
                    # Calculate number of assets exposed to factor according
                    # to Herfindahl methodology
                    factorWgt = ma.sqrt(factorMCap)
                    if ma.sum(factorWgt.filled(0.0)) > 0.0:
                        factorWgt = factorWgt / ma.sum(factorWgt)
                        effectiveWidth = 1.0 / ma.inner(factorWgt, factorWgt)
                    else:
                        effectiveWidth = 0.0
                    logging.debug('Factor %s has %.2f effective assets (%d)',
                            factorNameList[j], effectiveWidth, len(estuFactorIdx))
                    # Pad thin factors if possible
                    if effectiveWidth < minFactorWidth:
                        # Get list of assets that can be used for padding
                        spareFactorIdx.reverse()
                        n = 0
                        # Add next-largest asset to estu and recaculate factor width
                        while effectiveWidth < minFactorWidth and len(spareFactorIdx) > 0:
                            nextID = spareFactorIdx.pop()
                            factorMCap[nextID] = mcap[nextID]
                            factorWgt = ma.sqrt(factorMCap)
                            factorWgt = factorWgt / ma.sum(factorWgt)
                            prevWidth = effectiveWidth
                            effectiveWidth = 1.0 / ma.inner(factorWgt, factorWgt)
                            currentEstu.add(nextID)
                            n += 1
                            if prevWidth != 0.0 and (effectiveWidth/prevWidth - 1.0) < cutOff:
                                spareFactorIdx = []
                        totalN = len(estuFactorIdx) + n
                        logging.debug('Padded %s (%.2f/%d) with %d additional assets',
                                    factorNameList[j], effectiveWidth, totalN, n)
                        if effectiveWidth < minFactorWidth:
                            hasThinFactors = True

        nonest = set(range(len(self.assets))).difference(currentEstu)
        logging.debug('pump_up_factors: end')
        if returnFlag:
            return list(currentEstu), list(nonest), hasThinFactors
        else:
            return list(currentEstu), list(nonest)

    def pump_up_factors2(self, data, date, currentEstu=None, baseEstu=None,
            byFactorType=[Matrices.ExposureMatrix.IndustryFactor],
            minFactorWidth=10, cutOff=0.01, excludeFactors=[], downWeights=None):
        """Attempts to inflate thin factors
        Routine will stop if increase in effective width of factor
        from adding another asset falls below value cutOff (default is 1%)
        Thus, to make the routine more stringent, increase this value
        """
        logging.debug('pump_up_factors: begin')
        
        # Initialise
        expM = data.exposureMatrix
        mcap = numpy.array(data.marketCaps, copy=True)
        excludeFactorNames = []
        if excludeFactors is not None:
            excludeFactorNames = [f.name for f in excludeFactors]
        if baseEstu == None:
            baseEstu = range(len(self.assets))
        if currentEstu == None:
            currentEstu = set()
        else:
            currentEstu = set(currentEstu)
        if downWeights is None:
            downWeights = numpy.ones((len(mcap)), float)

        # Mask unwanted assets
        filter = Matrices.allMasked((len(self.assets)))
        ma.put(filter, baseEstu, 1)
        mcap = ma.masked_where(ma.getmaskarray(filter), mcap)

        # Else we attempt to inflate thin factors
        exposures = expM.getMatrix()
        # Loop round factor types
        herf_num_list=[]
        for fType in byFactorType:
            assert(fType in expM.factorTypes_)
            factorIdxList = expM.getFactorIndices(fType)
            factorNameList = expM.getFactorNames(fType)
            # Loop round set of factors
            for (j,k) in enumerate(factorIdxList):
                if factorNameList[j] in excludeFactorNames:
                    continue
                # Pick out eligible assets
                factorMCap = ma.masked_where(ma.getmaskarray(exposures[k]), mcap)
                factorDW = ma.masked_where(ma.getmaskarray(exposures[k]), downWeights)
                factorIdx = numpy.flatnonzero(ma.getmaskarray(factorMCap)==0)
                if len(factorIdx) < 1:
                    logging.warning('Factor %s is empty', factorNameList[j])
                # Sort from largest to smallest
                sortedIdx = ma.argsort(-factorMCap*factorDW)
                sortedIdx = sortedIdx[:len(factorIdx)]
                # Sort into those already in the estu and those not
                estuFactorIdx = [i for i in sortedIdx if i in currentEstu]
                spareFactorIdx = [i for i in sortedIdx if i not in currentEstu]
                # Get cap of everything in factor which is also in the estu
                filter = Matrices.allMasked((len(self.assets)))
                if len(estuFactorIdx) > 0:
                    ma.put(filter, estuFactorIdx, 1)
                factorMCap = ma.masked_where(ma.getmaskarray(filter), factorMCap)

                # Calculate number of assets exposed to factor according
                # to Herfindahl methodology
                factorWgt = ma.sqrt(factorMCap)
                if ma.sum(factorWgt.filled(0.0)) > 0.0:
                    factorWgt = factorWgt / ma.sum(factorWgt)
                    effectiveWidth = 1.0 / ma.inner(factorWgt, factorWgt)
                else:
                    effectiveWidth = 0.0
                logging.debug('Factor %s has %.2f effective assets (%d)',
                        factorNameList[j], effectiveWidth, len(estuFactorIdx))
                if len(spareFactorIdx) > 0:
                    # Pad thin factors if possible
                    if effectiveWidth < minFactorWidth:
                        # Get list of assets that can be used for padding
                        spareFactorIdx.reverse()
                        n = 0
                        # Add next-largest asset to estu and recaculate factor width
                        while effectiveWidth < minFactorWidth and len(spareFactorIdx) > 0:
                            nextID = spareFactorIdx.pop()
                            factorMCap[nextID] = mcap[nextID]
                            factorWgt = ma.sqrt(factorMCap)
                            factorWgt = factorWgt / ma.sum(factorWgt)
                            prevWidth = effectiveWidth
                            effectiveWidth = 1.0 / ma.inner(factorWgt, factorWgt)
                            currentEstu.add(nextID)
                            n += 1
                            if prevWidth != 0.0 and (effectiveWidth/prevWidth - 1.0) < cutOff:
                                spareFactorIdx = []
                        # if effectiveWidth < minFactorWidth:
                        #     import ipdb;ipdb.set_trace()
                        logging.debug('Padded %s (%.2f) with %d additional assets',factorNameList[j], effectiveWidth, n)
                    # logging.info('Current %s has HerfNum: %.2f.',factorNameList[j], effectiveWidth)
                herf_num_list.append([factorNameList[j], effectiveWidth])
        nonest = set(range(len(self.assets))).difference(currentEstu)
        logging.debug('pump_up_factors: end')
        return (list(currentEstu), list(nonest),herf_num_list)

    def filter_by_bm_membership(self, data, benchmarkNameList,
                                date, baseEstu=None,
                                action='union'):
        """ Add assets to estu according to membership of benchmarks
        """
        logging.debug('filter_by_bm_membership: begin')
        
        # Set up universe
        if action not in ('union', 'intersect'):
            raise Exception('Invalid action: %s' % action)
        all_indices = range(len(self.assets))
        assetIdx = dict([(i.getModelID(), j) for (i,j) in \
                zip(self.assets, all_indices)])
        if baseEstu == None:
            estu = range(len(self.assets))
        else:
            estu = baseEstu
        
        # Loop round benchmarks
        for bmName in benchmarkNameList:
            logging.debug('Processing index %s', bmName)
            bm = self.modelDB.getIndexConstituents(bmName, date, self.marketDB, rollBack=90)
            if len(bm)==0:
                continue
            else:
                bmIds = set([assetIdx[a] for (a,w) in bm if a in assetIdx])
                # Either add the BM assets or make them mutually exclusive
                if action == 'union':
                    estu = bmIds.union(estu)
                elif action == 'intersect':
                    estu = bmIds.intersection(all_indices)
        nonest = set(all_indices).difference(estu)
        logging.debug('ESTU now contains %d assets', len(estu))
        logging.debug('filter_by_bm_membership: end')
        return (list(estu), list(nonest))


    def grandfather(self, date, estu, baseEstu=None,
                    addDays=1, remDays=0, daysBack=61, estuInstance=None,
                    grandfatherRMS_ID=None):
        """Apply grandfathering rules to improve stability.
        Assets must 'qualify' for ESTU membership for a certain
        number of days before they are genuinely added and must have
        been absent from the list of eligible assets for a protracted
        period of time before they are dropped.
        """
        logging.debug('grandfather: begin')

        # Set up universe
        all_indices = range(len(self.assets))

        # Create list of dates for consideration
        dateList = self.modelDB.getDates(self.modelSelector.rmg, date, daysBack, excludeWeekend=True)
        dateList = dateList[:-1]
        addDays = min(addDays, len(dateList)+1)
        remDays = min(remDays, len(dateList))

        # Pull up a history of assets' ESTU eligibility (1/0)
        if grandfatherRMS_ID is not None:
            rms_id = grandfatherRMS_ID
        else:
            rms_id = self.modelSelector.rms_id
        logging.info('Getting grandfather history from RMS: %d', rms_id)
        qualifiedAssets = self.modelDB.loadESTUQualifyHistory(
                rms_id, self.assets, dateList, estuInstance=estuInstance)
        qualifiedAssets = ma.sum(qualifiedAssets.data.filled(0.0), axis=1)
        addDays = min(addDays, numpy.max(qualifiedAssets)+1)
        remDays = min(remDays, numpy.max(qualifiedAssets))

        # Sum the number of times each has qualified
        assetsToAdd = ma.masked_where(qualifiedAssets < addDays, qualifiedAssets)
        assetsToRem = ma.masked_where(qualifiedAssets >= remDays, qualifiedAssets)

        # Get locations of those that qualified previously  
        idxQualified = numpy.flatnonzero(ma.getmaskarray(assetsToAdd)==0)
        idxUnqualified = numpy.flatnonzero(ma.getmaskarray(assetsToRem)==0)
        if len(idxQualified) > 0.9 * len(all_indices):
            idxQualified = list()
        if len(idxUnqualified) > 0.9 * len(all_indices):
            idxUnqualified = list()
        # Make sure anything being resurrected is bona fide
        if baseEstu != None:
            idxQualified = set(idxQualified).intersection(set(baseEstu))

        # Save list of assets that genuinely qualify today
        qualifiedToday = list(estu)

        # Pick out current non-qualifiers from qualifier list
        estu = set(estu)
        nAdd = len(set(idxQualified).difference(estu))
        estu = estu.union(idxQualified)
        nRem = len(estu.intersection(set(idxUnqualified)))
        estu = estu.difference(set(idxUnqualified))
        nonest = set(all_indices).difference(estu)

        logging.debug('Added %d assets (qualified at least %d out of %d days)',
                    nAdd, addDays, numpy.max(qualifiedAssets))
        logging.debug('Removed %d assets (qualified less than %d out of %d days)',
                    nRem, remDays,  numpy.max(qualifiedAssets))
        logging.debug('ESTU now contains %d assets (%d qualified)',
                    len(estu), len(qualifiedToday))

        logging.debug('grandfather: end')
        return (list(estu), qualifiedToday, list(nonest))

    def combine_estu(self, estuList, baseEstu=None, action='union'):
        """ Function to neatly combine estimation universes
            and do the necessary reporting
        """
        logging.debug('combine_estu: begin')

        # Set up the base estU
        all_indices = range(len(self.assets))
        if baseEstu == None:
            estu = set(all_indices)
        else:
            estu = set(baseEstu)

        # Loop round the set of other estus and combine accordingly
        for subEstu in estuList:
            if action == 'intersect':
                estu = estu.intersection(set(subEstu))
            elif action == 'difference':
                estu = estu.difference(set(subEstu))
            else:
                estu = estu.union(set(subEstu))

        # Form nonest
        nonest = set(all_indices).difference(estu)

        logging.debug('%d assets in estu after action \'%s\'', len(estu), action)
        logging.debug('combine_estu: end')
        return (list(estu), list(nonest))

    def exclude_by_market_classification(self, date,
                                clsMember, clsFamily, clsCodesList, 
                                baseEstu=None, keepMissing=True):
        """Exclude assets based on classification -- asset type,
        exchange of quotation, and so forth.  Assets belonging to
        classifications with names clsCodeList are the ones *kept*,
        not excluded.
        """
        logging.debug('exclude_by_market_classification: begin')
        clsFamily = self.marketDB.getClassificationFamily(clsFamily)
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                            getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get(clsMember, None)
        assert(clsMember is not None)
        clsRevision = self.marketDB.\
                getClassificationMemberRevision(clsMember, date)
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))
        clsData = self.modelDB.getMktAssetClassifications(
                        clsRevision, univ, date, self.marketDB)
        indices = set([self.assetIdxMap[n] for n in clsData.keys() \
                       if clsData[n].classification.code not in clsCodesList])
        if not keepMissing:
            missing = set(univ).difference(clsData.keys())
            missing = [self.assetIdxMap[n] for n in missing]
            indices.update(missing)
        msg = '%d out of %d assets classified as %s' \
                % (len(univ)-len(indices), len(univ), ','.join(clsCodesList))
        if keepMissing:
            msg += ' (or missing)'
        logging.debug(msg)
        estu = set(estu).difference(indices)
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('Excluded %d assets based on %s',
                len(univ)-len(estu), clsMember.name)

        logging.debug('exclude_by_market_classification: end')
        return (list(estu), list(nonest))

    def exclude_low_price_assets(self, date, baseEstu=None, minPrice=1.0):
        """Exclude assets with prices below a certain threshold.
        For regional models, prices are converted to the
        model numeraire currency for comparison.
        """
        logging.debug('exclude_low_price_assets: begin')
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))
        if len(self.modelSelector.rmg) > 1:
            numeraire = self.modelSelector.numeraire.currency_id
        else:
            numeraire = None
        priceData = self.modelDB.loadUCPHistory([date], univ, numeraire)
        prices = priceData.data[:,0]
        prices = ma.masked_where(prices <= minPrice, prices)
        indices = [self.assetIdxMap[priceData.assets[i]] \
                       for i in numpy.flatnonzero(ma.getmaskarray(prices))]
        logging.debug('Excluding %d assets with price below %f (%s)', 
                    len(indices), minPrice, self.modelSelector.numeraire.currency_code)
        estu = set(estu).difference(indices)
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_low_price_assets: end')
        return (list(estu), list(nonest))

    def exclude_thinly_traded_assets(
            self, date, data, baseEstu=None, daysBack=250,
                        minDays=21, minNonMissing=0.95, minNonZero=0.25,
                        maskZeroWithNoADV=True, legacy=True):
        """Exclude assets with a high proportion of missing (zero)
        returns
        """
        logging.debug('exclude_thinly_traded_assets: begin')
        if maskZeroWithNoADV:
            logging.debug('Zero returns with missing ADV treated as missing')
        else:
            logging.debug('Zero returns with missing ADV treated as zero')
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))
        if not hasattr(self.modelDB, 'subidMapCache'):
            self.modelDB.subidMapCache = ModelDB.FromThruCache(useModelID=False)
         
        # Load returns - fill all masked values
        if legacy:
            returnsProcessor = ProcessReturns.assetReturnsProcessor(
                    self.modelSelector.rmg, univ, data.rmgAssetMap,
                    data.tradingRmgAssetMap, None, None)
            returns = self.modelDB.loadTotalReturnsHistory(
                self.modelSelector.rmg, date, univ, daysBack, notTradedFlag=True)
            (noRetArray, ipoArray, ntdArray, zeroRetArray, newIPOList) = returnsProcessor.sort_trading_days(
                    returns, self.modelDB, self.marketDB, date, maskZeroWithNoADV=maskZeroWithNoADV)

            # Find proportion of assets with non-missing returns
            validDays = len(returns.dates) - ma.sum(ipoArray + ntdArray, axis=1)
            validRets = validDays - ma.sum(noRetArray, axis=1)
            validRets = ma.where(validDays < minDays, 0, validRets)
            pctNonMissingRets = validRets / numpy.array(validDays, float)
         
            # Find proportion of assets with non-zero returns
            validRets = validDays - ma.sum(zeroRetArray, axis=1)
            validRets = ma.where(validDays < minDays, 0, validRets)
            pctNonZeroRets = validRets / numpy.array(validDays, float)
        else:
            # Newer models have relevant data pre-computed
            pctNonMissingRets = numpy.array([data.retScoreDict[sid] for sid in univ], float)
            pctNonZeroRets = numpy.array([data.zeroScoreDict[sid] for sid in univ], float)
        
        # Remove assets trading on fewer than a certain proportion of days
        n = len(estu)
        dropIdx = [self.assetIdxMap[univ[i]]\
                for i in numpy.flatnonzero(ma.where(pctNonMissingRets<minNonMissing, 1, 0))]
        logging.debug('Excluding %d assets with fewer than %d%% of returns trading',
                len(dropIdx), int(100*minNonMissing))
        estu = set(estu).difference(dropIdx)
         
        # Remove assets with above a given number of zero returns
        dropIdx2 = [self.assetIdxMap[univ[i]]\
                for i in numpy.flatnonzero(ma.where(pctNonZeroRets<minNonZero, 1, 0))]
        dropIdx2 = set(dropIdx2).difference(set(dropIdx))
        logging.debug('Excluding %d assets with fewer than %d%% of returns non-zero',
                len(dropIdx2), int(100*minNonZero))

        estu = set(estu).difference(dropIdx2)
        if len(estu) < n:
            logging.debug('%d zero/illiquid assets dropped from estu (%d to %d)',
                    n-len(estu), n, len(estu))
        nonest = set(range(len(self.assets))).difference(estu)
        return (list(estu), list(nonest))

    def exclude_sparsely_traded_assets(
            self, date, data, baseEstu=None, minGoodReturns=0.75):
        """Exclude assets with a high proportion of missing and zero returns
        """
        logging.debug('exclude_sparsely_traded_assets: begin')
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))

        # Load missing/zero returns scores
        pctNonMissingRets = numpy.array([data.retScoreDict[sid] for sid in univ], float)
        pctNonZeroRets = numpy.array([data.zeroScoreDict[sid] for sid in univ], float)

        # Combine the scores to get all missing/zero returns
        extraWeights0 = numpy.array([data.retScoreDict[sid] for sid in univ], float)
        extraWeights1 = numpy.array([data.zeroScoreDict[sid] for sid in univ], float)
        extraWeights = extraWeights0 + extraWeights1 - 1.0
        pctNonMissingRets = numpy.clip(extraWeights, 0.0, 1.0)
        pctNonZeroRets = numpy.ones(len(univ), float)

        # Remove assets trading on fewer than a certain proportion of days
        n = len(estu)
        dropIdx = [self.assetIdxMap[univ[i]]\
                for i in numpy.flatnonzero(ma.where(pctNonMissingRets<minGoodReturns, 1, 0))]
        logging.debug('Excluding %d assets with fewer than %d%% of returns trading',
                len(dropIdx), int(100*minGoodReturns))
        estu = set(estu).difference(dropIdx)

        if len(estu) < n:
            logging.debug('%d zero/illiquid assets dropped from estu (%d to %d)',
                    n-len(estu), n, len(estu))
        nonest = set(range(len(self.assets))).difference(estu)
        return (list(estu), list(nonest))

    def exclude_by_isin_country(self, keepList, date,
                                baseEstu=None, keepBDI=True, keepMissing=True):
        """Exclude assets based on their ISIN country prefix.
        Assets with prefixes in keepList are retained.  Any assets
        with no ISIN mapping can be excluded if keepMissing=False.
        Assets belonging to common BDI countries are kept by defalt,
        but can be excluded if keepBDI=False.
        """
        logging.debug('exclude_by_isin_country: begin')
        if keepBDI:
            bdi_codes = ['AI','AG','BS','BB','BZ','BM',
                         'VG','KY','CK','FO','GI','IM',
                         'LR','MH','AN','PA','TC','JE','GG']
            keepList.extend(bdi_codes)
        keepList = set(keepList)
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))
        mdl2sub = dict([(n.getModelID(), n) for n in univ])
        isinMap = self.modelDB.getIssueISINs(date, 
                        [n.getModelID() for n in univ], self.marketDB)
        exclude_ids = [mid for (mid,isin) in isinMap.items() if \
                       isin[0:2] not in keepList]
        exclude_indices = [self.assetIdxMap[mdl2sub[mid]] for mid in exclude_ids]
        logging.debug('Found %d out of %d ISINs with foreign country prefixes',
                        len(exclude_indices), len(univ))
        estu = set(estu).difference(exclude_indices)
        if not keepMissing:
            estu = estu.intersection([self.assetIdxMap[mdl2sub[mid]] \
                        for mid in isinMap.keys()])
        logging.debug('Excluded %d assets based on ISIN country prefix',
                        len(univ)-len(estu))
        nonest = set(range(len(self.assets))).difference(estu)

        logging.debug('exclude_by_isin_country: end')
        return (list(estu), list(nonest))

    def exclude_by_asset_type(self, date, data,
            includeFields=['all-com'], excludeFields=None, baseEstu=None):
        """"Exclude assets based on their Axioma asset type
        """
        logging.debug('exclude_by_asset_type: begin')
        if (includeFields is not None) and (len(includeFields) == 0):
            includeFields = None
        if (excludeFields is not None) and (len(excludeFields) == 0):
            excludeFields = None
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))

        # First deal with what we wish to explicitly include
        if includeFields is not None:
            includeFields = [ic.lower() for ic in includeFields]
            includeStocks = set([sid for sid in univ \
                    if sid in data.assetTypeDict and \
                    data.assetTypeDict[sid].lower() in includeFields])
            if 'all-com' in includeFields:
                commonStock = set([sid for sid in univ \
                        if sid in data.assetTypeDict and \
                        data.assetTypeDict[sid][:3].lower() == 'com'])
                includeStocks = includeStocks.union(commonStock)
            if 'all-pref' in includeFields:
                prefStock = set([sid for sid in univ \
                        if sid in data.assetTypeDict and \
                        data.assetTypeDict[sid][:4].lower() == 'pref'])
                includeStocks = includeStocks.union(prefStock)
            logging.debug('%d stocks marked for inclusion by type: %s',
                    len(includeStocks), includeFields)
        else:
            includeStocks = set(univ)
        
        # Next with that which we wish to explicitly exclude
        if excludeFields is not None:
            excludeFields = [ic.lower() for ic in excludeFields]
            excludeStocks = set([sid for sid in univ \
                    if sid in data.assetTypeDict and \
                    data.assetTypeDict[sid].lower() in excludeFields])
            if 'all-pref' in excludeFields:
                prefStock = set([sid for sid in univ \
                        if sid in data.assetTypeDict and \
                        data.assetTypeDict[sid][:4].lower() == 'pref'])
                excludeStocks = excludeStocks.union(prefStock)
            logging.debug('%d stocks marked for exclusion by type: %s',
                    len(excludeStocks), excludeFields)
        else:
            excludeStocks = set()

        includeStocks = includeStocks.difference(excludeStocks)
        estu = set([self.assetIdxMap[sid] for sid in includeStocks])
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_by_asset_type: end')
        return (list(estu), list(nonest))

    def exclude_by_market_type(self, date, data,
            includeFields=None, excludeFields=['OTC'], baseEstu = None):
        """"Exclude assets based on their market exchange type
        """
        logging.debug('exclude_by_market_type: begin')
        if baseEstu is not None:
            univ = [self.assets[i] for i in baseEstu]
            estu = baseEstu
        else:
            univ = self.assets
            estu = range(len(univ))

        # First deal with what we wish to explicitly include
        if includeFields is not None:
            includeStocks = set([sid for sid in univ \
                    if sid in data.marketTypeDict and \
                    data.marketTypeDict[sid] in includeFields])
            logging.debug('%d stocks marked for inclusion by type: %s',
                    len(includeStocks), includeFields)
        else:
            includeStocks = set(univ)

        # Next with that which we wish to explicitly exclude
        if excludeFields is not None:
            excludeStocks = set([sid for sid in univ \
                    if sid in data.marketTypeDict and \
                    data.marketTypeDict[sid] in excludeFields])
            logging.debug('%d stocks marked for exclusion by type: %s',
                    len(excludeStocks), excludeFields)
        else:
            excludeStocks = set()

        includeStocks = includeStocks.difference(excludeStocks)
        estu = set([self.assetIdxMap[sid] for sid in includeStocks])
        nonest = set(range(len(self.assets))).difference(estu)
        logging.debug('exclude_by_market_type: end')
        return (list(estu), list(nonest))

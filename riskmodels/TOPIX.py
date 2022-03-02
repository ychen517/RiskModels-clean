
import numpy
import numpy.ma as ma
import logging
from riskmodels import EstimationUniverse
from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities

class TOPIXIndexSerie:
    """Class representing the various TOPIX
    marketcap segment based benchmarks.
    """
    def __init__(self, name, startIdx, endIdx, useFreeFloat=True):
        self.name = name
        self.startIdx = startIdx
        self.endIdx = endIdx
        self.useFreeFloat = useFreeFloat

class TOPIXReplicator:
    """Class to replicate TOPIX benchmark construction logic.
    """
    TOPIXCore30 = TOPIXIndexSerie('TOPIX Core 30', 0, 30)
    TOPIXLarge70 = TOPIXIndexSerie('TOPIX Large 70', 30, 100)
    TOPIX100 = TOPIXIndexSerie('TOPIX 100', 0, 100)
    TOPIXMid400 = TOPIXIndexSerie('TOPIX Mid 400', 100, 500)
    TOPIX500 = TOPIXIndexSerie('TOPIX 500', 0, 500)
    # Can't steal free-float weights for these because
    # the Russell/FTSE universes are too small
    TOPIX1000 = TOPIXIndexSerie('TOPIX 1000', 0, 1000, False)
    TOPIXSmall = TOPIXIndexSerie('TOPIX Small', 500, 1800, False)

    def __init__(self, modelSelector, modelDB, 
                 freeFloatSource=None, volumeDays=250):
        self.modelSelector = modelSelector
        self.freeFloatSource = freeFloatSource
        self.volumeDays = volumeDays
        self.returnDays = 60
        self.baseUniverse = None

        # Make sure the various TimeSeriesCaches are ready
        if modelDB.totalReturnCache is None:
            modelDB.setTotalReturnCache(self.returnDays)
        elif modelDB.totalReturnCache.maxDates < self.returnDays:
            modelDB.setTotalReturnCache(self.returnDays)
        if modelDB.volumeCache is None:
            modelDB.setVolumeCache(volumeDays)
        elif modelDB.volumeCache.maxDates < volumeDays:
            modelDB.setVolumeCache(volumeDays)

    def generate_topix_universe(self, univ, mcaps, date, 
                                modelDB, marketDB):
        """Attempts to create the basic TOPIX broad-market universe.
        Problematic areas include assets trading on multiple exchanges
        with non-TSE primary quotation, recently listed assets (or those
        changing boards/sections), and instruments other than common equity
        from non-Japan-domiciled issuers.  And trashy Thompson data.
        Returns a list of index positions corresponding to univ.
        """
        logging.debug('generate_topix_universe: begin')
        assetIdxMap = dict(zip(univ, range(len(univ))))

        # Weed out all the undesirable stuff first
        e = EstimationUniverse.ConstructEstimationUniverse(
                            univ, self.modelSelector, modelDB, marketDB)
        (indices0, ne) = e.apply_exclusion_list(date)
        # Keep only JP issuers (via ISIN prefix)
        (indices1, ne) = e.exclude_by_isin_country(['JP'], 
                            date, baseEstu=indices0, keepBDI=False)
        # Flag any open and closed ended funds for removal
        (funds1, ne) = e.exclude_by_market_classification(
                            date, 'TQA FTID Global Asset Type',
                            'ASSET TYPES', ['6C','6D'], 
                            baseEstu=indices1, keepMissing=False)
        # Ditto ETFs and mutual funds
        (funds2, ne) = e.exclude_by_market_classification(
                            date, 'DataStream2 Asset Type',
                            'ASSET TYPES', ['CF','CEF','ET','ETF'], 
                            baseEstu=indices1, keepMissing=False)
        indices2 = list(set(indices1).difference(funds1).difference(funds2))

        # Get rid of some other undesirables
        tmpData = Utilities.Struct()
        from riskmodels import AssetProcessor
        tmpData.assetTypeDict = AssetProcessor.get_asset_info(
                date, univ, modelDB, marketDB,
                'ASSET TYPES', 'Axioma Asset Type')
        (indices3, ne) = e.exclude_by_asset_type(
                date, tmpData, includeFields=None,
                excludeFields=['ComWI', 'UnCls', 'StatETF', 'LLC', 'LP', 'NonEqETF'],
                baseEstu=indices2)

        # Start with all assets with primary quotation on TSE-1
        (indices, ne) = e.exclude_by_market_classification(
                            date, 'Market', 'REGIONS', 
                            ['TKS-S1'], baseEstu=indices3, keepMissing=False)
        logging.info('Starting with %d assets from TSE Section 1', len(indices))

        # Remove recent IPOs in accordance with TOPIX monthly review logic
        dateList = modelDB.getDates(self.modelSelector.rmg, date, 90)
        eomDate = [prev for (prev,next) in zip(dateList[:-1], dateList[1:])
                if (next.month > prev.month or next.year > prev.year)][-2]
        if not hasattr(modelDB, 'subidMapCache'):
            modelDB.subidMapCache = ModelDB.FromThruCache(useModelID=False)
        issueData = modelDB.loadFromThruTable('sub_issue', 
                        [date], [univ[i] for i in indices], ['issue_id'], 
                        keyName='sub_id', cache=modelDB.subidMapCache)[0,:]
        newListingsIdx = [j for (i,j) in enumerate(indices) \
                            if issueData[i].fromDt > eomDate]
        logging.info('Removing %d suspected recent listings since %s', 
                        len(newListingsIdx), str(eomDate))
        indices = set(indices).difference(newListingsIdx)

        # Add in any large-caps primarily traded in Osaka (eg Nintendo)
        percentiles = Utilities.prctile(mcaps, [75.0, 90.0])
        (ose_idx, ne) = e.exclude_by_market_classification(
                            date, 'Market', 'REGIONS', 
                            ['OSE-S1'], baseEstu=indices1, keepMissing=False)
        ose_mcaps = ma.take(mcaps, ose_idx, axis=0)
        ose_mcaps = ma.masked_where(ose_mcaps < percentiles[0], ose_mcaps)
        goodOSEAssetsIdx = numpy.flatnonzero(ma.getmaskarray(ose_mcaps)==0)
        goodOSEAssetsIdx = [ose_idx[i] for i in goodOSEAssetsIdx]
        logging.info('Adding %d assets from OSE Section 1 above %.2f bn %s',
                len(goodOSEAssetsIdx), percentiles[0] / 1e9, 
                self.modelSelector.numeraire.currency_code)
        indices = indices.union(goodOSEAssetsIdx)

        # Remove any low-price assets filing for liquidation/bankruptcy
        returns = modelDB.loadTotalReturnsHistory(self.modelSelector.rmg,
                    date, [univ[i] for i in indices], self.returnDays-1, None)
        recentReturns = returns.data.filled(0.0)
        recentReturns = ma.masked_where(recentReturns <= -0.50, recentReturns)
        suspect = numpy.flatnonzero(ma.sum(ma.getmaskarray(recentReturns), axis=1))
        suspect = [univ[assetIdxMap[returns.assets[i]]] for i in suspect]
        logging.info('Found %d assets with big negative returns since %s', 
                        len(suspect), str(returns.dates[0]))
        if len(suspect) > 0:
            prices = modelDB.loadUCPHistory(
                [returns.dates[-1]], suspect, None).data[:,0]
            lowPriceIdx = numpy.flatnonzero(ma.getmaskarray(
                                    ma.masked_where(prices < 5.0, prices)))
            lowPriceIdx = [assetIdxMap[suspect[i]] for i in lowPriceIdx]
            logging.info('Removing %d possible bankruptcy/liquidation-pending assets', 
                        len(lowPriceIdx))
            logging.debug('Removed Axioma IDs: %s', ', '.join(
                    [univ[i].getSubIDString() for i in lowPriceIdx]))
            indices = indices.difference(lowPriceIdx)
        indices = list(indices)
        self.baseUniverse = [univ[i] for i in indices]

        logging.debug("generate_topix_universe: end")
        return indices

    def replicate_topix_subindex(self, indexClass, date, univ, mcaps, 
                              modelDB, marketDB, quick=False):
        """Attempts to replicate the list of constituents in
        the TOPIX sub-index given by indexClass.
        Works best if generate_topix_universe() is first run.
        If quick=True we will do it the cheapo way.
        Returns a list of index positions corresponding to univ.
        """
        logging.debug('replicate_topix_subindex: begin')
        assert(isinstance(indexClass, TOPIXIndexSerie))

        logging.info('Replicating %s', indexClass.name)
        # If so specified, just do it the quick and dirty way
        if quick:
            capRank = ma.argsort(-mcaps)
            indices = capRank[indexClass.startIdx:indexClass.endIdx]
        else:
            # Steal free-float weights, if available
            if self.freeFloatSource is not None and indexClass.useFreeFloat:
                ff = modelDB.getIndexConstituents(
                              self.freeFloatSource, date, marketDB, rollBack=90)
                if len(ff) > 0:
                    ffWeightsDict = dict(ff)
                    mcaps = [ffWeightsDict.get(n.getModelID(), 0.0) for n in univ]

            # Restrict to TOPIX broad market universe
            if self.baseUniverse is not None:
                assetIdxMap = dict(zip(univ, range(len(univ))))
                keepIdx = [assetIdxMap[n] for n in self.baseUniverse]
                tmp = numpy.zeros(len(univ))
                for i in keepIdx:
                    tmp[i] = mcaps[i]
                mcaps = tmp

            # Sort market caps
            capRank = numpy.argsort(-numpy.array(mcaps))

            # Get total historical volume
            volumeDays = modelDB.getDates(self.modelSelector.rmg, date,
                                          self.volumeDays-1)
            volumeHistory = modelDB.loadVolumeHistory(volumeDays, univ, None)
            totalVol = ma.sum(volumeHistory.data, axis=1)
            volRank = ma.argsort(-totalVol)

            logging.info('Constructing TOPIX Core 30')
            indices = set([i for i in capRank if i in set(volRank[:90])][:15])
            next15 = [i for i in capRank[:40] if i in set(volRank[:90]) \
                      and i not in indices][:15]
            indices.update(next15)
            if len(next15) < 15:
                missing = 15 - len(next15)
                logging.info('Using only volume ranking to determine %d CORE-30 assets', missing)
                alt = [i for i in volRank[:90] if i not in indices][:missing]
                indices.update(alt)
            if indexClass == self.TOPIXCore30:
                return list(indices)

            logging.info('Constructing TOPIX Large 70')
            large70 = set([i for i in capRank[:130] if i in set(volRank[:200]) \
                           and i not in indices][:70])
            if len(large70) < 70:
                missing = 70 - len(large70)
                logging.info('Using only volume ranking to determine %d LARGE-70 assets', missing)
                alt = [i for i in set(volRank[:200]) if i not in indices][:missing]
                large70.update(alt)
            if indexClass == self.TOPIXLarge70:
                return list(large70)
            logging.info('Constructing TOPIX 100')
            indices.update(large70)
            if indexClass == self.TOPIX100:
                return list(indices)

            logging.info('Constructing TOPIX Mid 400')
            mid400 = set([i for i in capRank[:600] if i in set(volRank[:1000]) \
                          and i not in indices][:400])
            if len(mid400) < 400:
                missing = 400 - len(mid400)
                logging.info('Using only volume ranking to determine %d MID-400 assets', missing)
                alt = [i for i in volRank[:1000] if i not in indices][:missing]
                mid400.update(alt)
            if indexClass == self.TOPIXMid400:
                return list(mid400)
            logging.info('Constructing TOPIX 500')
            indices.update(mid400)
            if indexClass == self.TOPIX500:
                return list(indices)

            logging.info('Constructing TOPIX 1000')
            remaining = set([i for i in capRank[:1200] if i in set(volRank[:1200]) \
                             and i not in indices][:500])
            if len(remaining) < 500:
                missing = 500 - len(remaining)
                logging.info('Using only volume ranking to determine %d TOPIX-1000 assets',
                                missing)
                alt = [i for i in volRank[:1200] if i not in indices][:missing]
                remaining.update(alt)
            if indexClass == self.TOPIXSmall and self.baseUniverse is not None:
                logging.info('Constructing TOPIX Small')
                small = set(keepIdx).difference(indices)
                return list(small)
            indices.update(remaining)
            if indexClass == self.TOPIX1000:
                return list(indices)

        logging.debug('replicate_topix_subindex: end')
        return list()

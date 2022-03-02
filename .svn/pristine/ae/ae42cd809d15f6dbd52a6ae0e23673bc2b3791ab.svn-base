import logging
import datetime
import numpy.ma as ma
import numpy.linalg as linalg
import numpy
import scipy.stats as stats
import nantools.utils as nanutils
import itertools
import pandas
import copy
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Utilities
from riskmodels import FactorReturns
from riskmodels import Outliers
from riskmodels import AssetProcessor_V4

class assetReturnsProcessor:
    """Class to perform various returns-related tasks, such as proxying
    and back-filling
    """
    def __init__(self, rmgList, universe, rmgAssetMap, tradingRmgAssetMap,
            assetTypeDict, dr2UnderDict=None,
            debuggingReporting=False, estu=None, gicsDate=datetime.date(2016,9,1),
            numeraire_id=1, tradingCurrency_id=None, returnsTimingID=1,
            boT=datetime.date(1980,1,1), returnLowerBound=-0.999999,
            simpleProxyRetTol=0.95):
        from riskmodels.CurrencyRisk import ModelCurrency
        self.tradingCurrency_id = tradingCurrency_id
        self.numeraire_id = numeraire_id
        self.returnsTimingID = returnsTimingID
        self.rmg = rmgList
        self.universe = universe
        self.rmgAssetMap = rmgAssetMap # asset/home-country mapping
        self.tradingRmgAssetMap = tradingRmgAssetMap # asset/trading-country mapping
        self.assetTypeDict = assetTypeDict
        self.dr2UnderDict = dr2UnderDict
        self.debuggingReporting = debuggingReporting
        self.gicsDate = gicsDate
        self.countryProxyList = ['AU','BR','CA','CN','FR','DE','HK','IN',\
                'IT','JP','KR','MY','SG','ZA','SE','CH','TW','TH','GB','US']
        if estu is None:
            self.estuIdx = list(range(len(universe)))
        else:
            self.estuIdx = list(estu)
        self.boT = boT
        self.returnLowerBound = returnLowerBound
        self.simpleProxyRetTol = simpleProxyRetTol

    def compound_currency_returns(self, returns, zeroMask):
        """ Compounds currency returns for missing asset returns into non-missing
        returns locations
        """
        logging.info('Compounding currency and returns-timing adjustments')
        data = Utilities.screen_data(returns)
        mask = numpy.array(ma.getmaskarray(data), copy=True, dtype=numpy.int8)
        zeroMask = zeroMask.view(dtype=numpy.int8)
        tmpreturns = ma.filled(data, numpy.nan)
        data = ma.filled(data, 0.0)
        nanutils.compound_currency_returns(tmpreturns, data, mask, zeroMask)
        returns = ma.array(data, mask=mask.view(dtype=bool))
        logging.debug('Finished compounding currency and returns-timing adjustments')
        return returns

    def sort_trading_days(self, returns, modelDB, marketDB, date, maskZeroWithNoADV=True):
        """ Function to sift through missing asset returns and
        decide what is missing due to illiquidity, what is due to
        not yet being live, and what is due to non-trading days
        Returns several Boolean arrays denoting a missing return as True
        """
        nT = returns.data.shape[0] * returns.data.shape[1]
        # Optional - treat as missing zero returns with zero or missing trading volume
        # Load trading volumes for all assets and dates
        vol = modelDB.loadVolumeHistory(returns.dates, returns.assets, None, loadAllDates=True).data
        vol = ma.masked_where(vol<=0.0, vol)

        # Look for zero returns with missing volumes
        tmpData = ma.filled(returns.data, 1.0)
        tmpData = ma.masked_where(tmpData==0.0, tmpData)
        missingVol = ma.getmaskarray(vol)
        zeroReturnFlag = ma.getmaskarray(tmpData)
        dubiousFlag = zeroReturnFlag & missingVol
        nDF = ma.sum(dubiousFlag, axis=None)
        nZero = ma.sum(zeroReturnFlag, axis=None)
        if maskZeroWithNoADV:
            if nZero != 0.0:
                logging.info('Sort returns: %d out of %d zero returns (%.1f%%) with no volume data treated as missing',
                        nDF, nZero, 100.0 * nDF/nZero)
            returns.data = ma.masked_where(dubiousFlag, returns.data)
            zeroReturnFlag = zeroReturnFlag & ~missingVol
        else:
            if nZero != 0.0:
                logging.info('Sort returns: %d out of %d zero returns (%.1f%%) with no volume data treated as zero',
                        nDF, nZero, 100.0 * nDF/nZero)

        # For comparison purposes, output proportion of non-zero, non-missing
        # returns with no trading volume
        tmpData = ma.masked_where(returns.data==0.0, returns.data)
        nonMissing = ~ma.getmaskarray(tmpData)
        dubiousFlag = nonMissing & missingVol
        nDF = ma.sum(dubiousFlag, axis=None)
        nOk = ma.sum(nonMissing, axis=None)
        if nOk != 0.0:
            logging.info('Sort returns: %d out of %d non-zero returns (%.1f%%) with no volume data',
                    nDF, nOk, 100.0 * nDF/nOk)

        # Get mask array for absolutely everything that's missing
        allMasked = ma.getmaskarray(returns.data)

        # Get the various data 'masks' - first the roll-over indicator
        # Denote as True where an asset has missing return
        rollOverFlag = ma.array(ma.filled(returns.notTradedFlag, 1.0), bool) & allMasked
        nNonTrade = ma.sum(rollOverFlag, axis=None)
        logging.info('Sort returns: %d out of %d returns (%.1f%%) flagged as missing',
                nNonTrade, nT, 100.0 * nNonTrade/nT)

        # Then the pre-IPO flag indicator
        # Marked as True wherever an asset is pre-IPO
        issueFromDates = Utilities.load_ipo_dates(
                date, returns.assets, modelDB, marketDB, exSpacAdjust=True, returnList=True)
        preIPOFlag = Matrices.fillAndMaskByDate(returns.data, issueFromDates, returns.dates)
        preIPOFlag = numpy.array(ma.getmaskarray(preIPOFlag), copy=True)

        # Pull out a list of IPOs for the latest date
        issueFromDates = pandas.Series(issueFromDates, index=returns.assets)
        newIPOList = [sid for sid in returns.assets if issueFromDates[sid]==returns.dates[-1]]

        # Thirdly, the non-trading day mask
        # Marked True on all non-trading days for each market
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(returns.dates)])
        ntdFlag = Matrices.allMasked(returns.data.shape)
        for rmg_id in self.tradingRmgAssetMap.keys():
            rmgAssets = list(set(self.tradingRmgAssetMap[rmg_id]).intersection(set(returns.assetIdxMap.keys())))
            asset_indices = [returns.assetIdxMap[sid] for sid in rmgAssets]
            if hasattr(rmg_id, 'rmg_id'):
                rmgCalendarSet = set(modelDB.getDateRange(rmg_id, returns.dates[0], returns.dates[-1]))
            else:
                rmgCalendarSet = set(modelDB.getDateRange(
                        modelDB.getRiskModelGroup(rmg_id), returns.dates[0], returns.dates[-1]))
            if len(asset_indices) > 0:
                ntd_dates = list(set(dateIdxMap.keys()).difference(set(rmgCalendarSet)))
                ntd_date_indices = [dateIdxMap[d] for d in ntd_dates]
                positions = [len(returns.dates) * sIdx + dIdx for \
                        sIdx in asset_indices for dIdx in ntd_date_indices]
                ma.put(ntdFlag, positions, 1)
        ntdFlag = ~ma.getmaskarray(ntdFlag)

        # Remove preIPO days that are also NTDs from NTD flag so
        # we don't double-count
        ntdFlag = ~preIPOFlag & ntdFlag
        totalNTDs = ma.sum(ntdFlag, axis=1) * 260.0 / float(ntdFlag.shape[1])
        logging.info('NTD stats: (min %d, mean %d, max %d)', 
                ma.min(totalNTDs, axis=None), ma.average(totalNTDs, axis=None), ma.max(totalNTDs, axis=None))

        # Determine days where assets genuinely haven't traded due to illiquidity
        # This is all non-trading days in rollOverFlag that aren't in either
        # of the other two arrays
        rollOverFlag = rollOverFlag & ~preIPOFlag & ~ntdFlag

        # Flag anything else that's masked that isn't accounted for by one
        # of the existing flags
        xFlag = allMasked & ~rollOverFlag & ~preIPOFlag & ~ntdFlag
        nXF = ma.sum(xFlag, axis=None)
        logging.info('Sort returns: %d out of %d returns (%.1f%%) flagged as unaccounted for missing',
                nXF, nT, 100.0 * nXF/nT)
        # Assume they're rolled-over returns that have been missed
        rollOverFlag = rollOverFlag | xFlag

        # Finally do some reporting on the stats
        nIlliquid = 100.0 * ma.sum(rollOverFlag, axis=None) / float(nT)
        nPreIPO =  100.0 * ma.sum(preIPOFlag, axis=None) / float(nT)
        nNTD = 100.0 * ma.sum(ntdFlag, axis=None) / float(nT)
        nZero = 100.0 * ma.sum(zeroReturnFlag, axis=None) / float(nT)
        logging.info('Sort returns: %d returns, %.1f%% illiquid, %.1f%% pre IPO, %.1f%% NTDs, %.1f%% zero',
                nT, nIlliquid, nPreIPO, nNTD, nZero)
        return (rollOverFlag, preIPOFlag, ntdFlag, zeroReturnFlag, newIPOList)

    def dump_return_history(self, fileName, returns, returnDates, data, modelDB, marketDB, outspac=False):
        if (data.shape[1] < 1):
            return
        extraOutput = outspac
        self.output_stats(data, fileName)
        if not extraOutput:
            return
        # For debugging
        exSpac = None
        if outspac:
            date = returnDates[-1]
            exSpac = AssetProcessor_V4.sort_spac_assets(date, returns.assets, modelDB, marketDB, returnExSpac=True)
        if self.debuggingReporting:
            nameDict = modelDB.getIssueNames(returnDates[-1], returns.assets, marketDB)
            dateList = [str(d) for d in returnDates]
            mask = ma.getmaskarray(data)
            cidList1 = sorted([cid for cid in self.subIssueGroups.keys() if type(cid) is str])
            cidList2 = sorted([cid for cid in self.subIssueGroups.keys() if type(cid) is not str])
            cidList = cidList1 + cidList2
            if len(self.rmg) > 1:
                filepath = 'tmp/%s' % fileName
            else:
                filepath = 'tmp/%s-%s' % (self.rmg[0].mnemonic, fileName)
            outfile = open(filepath, 'w')
            outfile.write('CompanyID,axid,name,type,ex-SPAC,')
            for d in dateList:
                outfile.write('%s,' % d)
            outfile.write('\n')
            for groupId in cidList:
                subIssueList = sorted(self.subIssueGroups[groupId])
                for sid in subIssueList:
                    aType = self.assetTypeDict.get(sid, None)
                    idx = returns.assetIdxMap[sid]
                    n = 'None'
                    if sid in nameDict:
                        n = nameDict[sid].replace(',','')
                    spacFlag = 0
                    if (exSpac is not None) and (sid in exSpac):
                        spacFlag = 1
                    outfile.write('%s,%s,%s,%s,%s,' % (groupId, sid if isinstance(sid, str) \
                            else sid.getModelID().getPublicID(), n, aType, spacFlag))
                    for (t,d) in enumerate(returnDates):
                        if not mask[idx,t]:
                            fld = Utilities.p2_round(ma.filled(data[idx,t], 0.0), 16)
                            outfile.write('%.8f,' % fld)
                        else:
                            outfile.write(',')
                    outfile.write('\n')
            outfile.close()
            #if outspac:
            #    from pandas.stats.ols import MovingOLS
            #    spacRets = pandas.DataFrame(data, index=returns.assets, columns=returns.dates).loc[exSpac, :]
            #    marketReturnHistory = modelDB.loadRMGMarketReturnHistory(returns.dates, self.rmg, useAMPs=False, returnDF=True)
            #    cumRet = pandas.DataFrame(numpy.nan, index=spacRets.index, columns=range(len(spacRets.columns)))
            #    betas = pandas.DataFrame(numpy.nan, index=spacRets.index, columns=range(len(spacRets.columns)))
            #    for sid in exSpac:
            #        xDate = exSpac[sid]
            #        y = spacRets.loc[sid, spacRets.columns<xDate].fillna(0.0)
            #        x = marketReturnHistory.loc[self.rmg[0], spacRets.columns<xDate].fillna(0.0)
            #        if len(x) > 0:
            #            rreg = MovingOLS(y=y, x=x, window_type='rolling', intercept=True, window=60)
            #            beta = numpy.flipud(rreg.beta.loc[:, 'x'].values)
            #            betas.loc[sid] = pandas.Series(beta, index=range(len(beta)))
            #            betas.loc[sid, 'xdate'] = xDate
            #    betas.to_csv('tmp/spacBeta.csv')

            # Output stats on agreement between returns
            corrList = []
            for (groupId, subIssueList) in self.subIssueGroups.items():
                idxList = [returns.assetIdxMap[sid] for sid in subIssueList]
                retSubSet = ma.filled(ma.take(data, idxList, axis=0), 0.0)
                corr = Utilities.compute_covariance(retSubSet, axis=1, corrOnly=True)
                for i in range(corr.shape[0]):
                    for j in range(i+1,corr.shape[0]):
                        if corr[i,j] != 0.0 and corr[i,j] < 1.0:
                            corrList.append(corr[i,j])
            if len(corrList) > 0:
                medianCorr = ma.median(corrList, axis=None)
            else:
                medianCorr = 0.0
            if len(self.rmg) > 1:
                logging.info('Run: %s, Date: %s, MEDIAN: %s',
                        fileName, returnDates[-1], medianCorr)
            else:
                logging.info('%s run: %s, Date: %s, MEDIAN: %s',
                        self.rmg[0].mnemonic, fileName, returnDates[-1], medianCorr)

    def process_returns_history(self, date, needDays, modelDB, marketDB,
            excludeWeekend=True, applyRT=False, trimData=False, drCurrMap=None,
            loadOnly=False, noProxyList=[], useAllRMGDates=False):
        """Does extensive processing of asset returns to alleviate the problem
        of illiquid data. Does currency conversion and returns-timing
        adjustment to cross-listings to make them "local"
        Then, compounds these returns when there is no asset return on
        a particular day, and scales the non-missing returns accordingly
        Finally, fills in missing returns from non-missing linked partners
        and, again, scales non-missing illiquid returns
        """
        # Determine whether list of return dates is pre-specified
        if type(date) is list:
            dateList = list(date)
            date = dateList[-1]
            needDays = len(dateList)
        else:
            dateList = None

        logging.debug('Back-filling %d days of %d asset returns', needDays, len(self.universe))
        self.subIssueGroups = modelDB.getIssueCompanyGroups(date, self.universe, marketDB)
        inSIGrps = list(itertools.chain.from_iterable(list(self.subIssueGroups.values())))
        newIDs = list(set(self.universe) - set(inSIGrps))
        self.subIssueGroups.update({x: [x] for x in newIDs})

        # Set cache info
        cacheSize = max(int(1.5*needDays+5), 61)
        if modelDB.currencyCache is not None:
            if not hasattr(modelDB, 'currencyCache'):
                modelDB.createCurrencyCache(marketDB, boT=datetime.date(1980,1,1))
        if modelDB.volumeCache is not None:
            if modelDB.volumeCache.maxDates < cacheSize:
                logging.warning('ModelDB volume cache too small (%d days), resizing to %d days',
                        modelDB.volumeCache.maxDates, cacheSize)
                modelDB.setVolumeCache(cacheSize)
        if modelDB.totalReturnCache is not None:
            if modelDB.totalReturnCache.maxDates < cacheSize:
                logging.warning('ModelDB totalReturn cache too small (%d days), resizing to %d days',
                        modelDB.totalReturnCache.maxDates, cacheSize)
                modelDB.setTotalReturnCache(cacheSize)

        # Decide whether to convert returns to home market currency
        if drCurrMap is not None:
            currencyConversion = drCurrMap
            if isinstance(drCurrMap, dict):
                nCurrs = len(set(drCurrMap.values()))
            else:
                nCurrs = 1
            logging.debug('Performing conversion to %d currencies', nCurrs)
        else:
            currencyConversion = self.tradingCurrency_id
            logging.debug('Performing conversion to currency ID: %s', str(currencyConversion))

        # Get daily returns for the past calendar days(s)
        # We convert everything to trading currency AND time zone for the back filling to ensure consistency,
        # then transform back to trading currency and trading time zone afterwards
        logging.info('Loading history of %d days by %d returns', int(needDays), len(self.universe))
        if dateList is not None:
            returns = modelDB.loadTotalReturnsHistoryV3(
                    self.rmg, date, self.universe, int(needDays),
                    assetConvMap=currencyConversion, excludeWeekend=excludeWeekend,
                    dateList=dateList, allRMGDates=useAllRMGDates)
        else:
            returns = modelDB.loadTotalReturnsHistoryV3(
                    self.rmg, date, self.universe, int(needDays),
                    assetConvMap=currencyConversion, excludeWeekend=excludeWeekend,
                    allRMGDates=useAllRMGDates, boT=self.boT)
        if returns.data is None:
            return returns
        returns.data = ma.array(returns.originalData, copy=True)
        # Uncomment for Python 3 testing
        #returns.data = Utilities.p2_round(returns.data, 16)
        self.dump_return_history('rets_s0_raw.csv', returns, returns.dates, returns.data, modelDB, marketDB, outspac=True)

        if self.debuggingReporting:
            # Output low frequency correlations for comparison
            weekDates = Utilities.change_date_frequency(returns.dates)
            if len(weekDates) > 0:
                weeklyReturns, weekDates = compute_compound_returns_v3(
                        returns.data, returns.dates, weekDates)
                self.dump_return_history('rets_s0_weekly.csv',
                        returns, weekDates, weeklyReturns, modelDB, marketDB)
            monthDates = Utilities.change_date_frequency(returns.dates, frequency='monthly')
            if len(monthDates) > 0:
                monthlyReturns, monthDates = compute_compound_returns_v3(
                        returns.data, returns.dates, monthDates)
                self.dump_return_history('rets_s0_monthly.csv',
                        returns, monthDates, monthlyReturns, modelDB, marketDB)

        # Get masked returns, corresponding to IPOs, rolled prices and NTDs
        (rollOverFlag, preIPOFlag, ntdFlag, zeroFlag, newIPOList) = self.sort_trading_days(
                returns,  modelDB, marketDB, date)

        # Combine roll-over flag with NTD flag - treat all as illiquid henceforth
        rollOverFlag = rollOverFlag | ntdFlag
        missingFlag = rollOverFlag | preIPOFlag

        # Set important flags:
        # All returns missing for any reason
        returns.missingFlag = missingFlag
        # Returns missing pre-IPO
        returns.preIPOFlag = preIPOFlag
        # Returns flagged as rolled-over
        returns.rollOverFlag = rollOverFlag
        # Returns missing because of non-trading day
        returns.ntdFlag = ntdFlag
        # Zero, but non-missing returns
        returns.zeroFlag = zeroFlag
        self.dump_return_history('flag_preIPO.csv', returns, returns.dates, returns.preIPOFlag, modelDB, marketDB)
        self.dump_return_history('flag_rollOver.csv', returns, returns.dates, returns.rollOverFlag, modelDB, marketDB)
        self.dump_return_history('flag_NTD.csv', returns, returns.dates, returns.ntdFlag, modelDB, marketDB)
        self.dump_return_history('flag_Zero.csv', returns, returns.dates, returns.zeroFlag, modelDB, marketDB)
        self.dump_return_history('flag_Missing.csv', returns, returns.dates, returns.missingFlag, modelDB, marketDB)

        # Convert to specified currencies
        returnsNumer = (ma.filled(returns.currencyMatrix, 1.0) * (1.0 + ma.filled(returns.data, 0.0))) - 1.0
        self.dump_return_history('rets_s3_cur.csv', returns, returns.dates, returnsNumer, modelDB, marketDB)

        # Convert "foreign" listings to home time zone
        if self.returnsTimingID is not None:
            logging.info('Applying returns timing adjustments to transform DRs to home market')
            rmgAll = modelDB.getAllRiskModelGroups()
            adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                    self.returnsTimingID, rmgAll, returns.dates, legacy=False)
            adjustments.data = ma.filled(adjustments.data, 0.0)
            returnsNumer = self.adjustReturnsForTiming(
                    date, returns, returnsNumer, adjustments, rmgAll, alignWithHome=True,
                    USAdjustMentLater=applyRT)

        # Compound currency returns for illiquid assets
        returns.data = ma.masked_where(preIPOFlag, returnsNumer)
        noCurrencyReturn = ma.masked_where(returns.currencyMatrix==1.0, returns.currencyMatrix)
        hasCurrencyReturn = ma.getmaskarray(noCurrencyReturn)==0
        hasChangeIdx = numpy.flatnonzero(ma.sum(hasCurrencyReturn, axis=1))
        diff = hasCurrencyReturn.shape[0] - len(hasChangeIdx)
        if diff > 0:
            logging.info('%d assets have no currency conversion necessary', diff)
        if len(hasChangeIdx) > 0:
            # Do currency compounding for assets that require it
            tmpReturns = ma.take(returns.data, hasChangeIdx, axis=0)
            tmpRollFlag = ma.take(rollOverFlag, hasChangeIdx, axis=0)
            tmpReturns = self.compound_currency_returns(tmpReturns, tmpRollFlag)
            for (ii, idx) in enumerate(hasChangeIdx):
                returns.data[idx, :] = tmpReturns[ii, :]
        nMasked0 = len(numpy.flatnonzero(preIPOFlag))
        nMasked1 = len(numpy.flatnonzero(rollOverFlag))
        returns.data = ma.masked_where(rollOverFlag, returns.data)
        self.dump_return_history('rets_s4_curc.csv', returns, returns.dates, returns.data, modelDB, marketDB)
        missingDataMask = numpy.array(ma.getmaskarray(returns.data), copy=True)
        nMiss = numpy.size(numpy.flatnonzero(missingDataMask))

        if nMiss < 1:
            logging.info('No missing returns, skipping proxy generation')
        elif not loadOnly:
            # Estimate proxies via PCA
            outlierClass1 = Outliers.Outliers(industryClassificationDate=self.gicsDate)
            estu = [self.universe[idx] for idx in self.estuIdx]
            # Loop round RMGs
            for rmg in self.rmg:

                # Get subset of assets for RMG
                rmgSidList = sorted(Utilities.readMap(rmg, self.rmgAssetMap, []))
                rmgIdxList = [returns.assetIdxMap[sid] for sid in rmgSidList]
                if len(rmgIdxList) < 1:
                    continue

                # Create temporary returns structure
                tmpReturns = Matrices.TimeSeriesMatrix(rmgSidList, returns.dates)
                tmpReturns.data = ma.array(ma.take(returns.data, rmgIdxList, axis=0), copy=True)
                initialReturns = Matrices.allMasked(tmpReturns.data.shape)

                # Create temporary data structure
                tmpData = Utilities.Struct()
                tmpData.universe = rmgSidList
                tmpData.rmgAssetMap = {rmg.rmg_id: rmgSidList}
                tmpData.assetIdxMap = dict(zip(rmgSidList, list(range(len(rmgSidList)))))
                marketCaps = modelDB.getAverageMarketCaps(
                        tmpReturns.dates[-21:], tmpData.universe, self.numeraire_id, loadAllDates=True)
                tmpData.marketCaps = ma.filled(marketCaps, 0.0)

                # Get subset of estu for relevant RMG
                rmgEstu = list(set(rmgSidList).intersection(set(estu)))
                rmgEstuIdx = [tmpData.assetIdxMap[sid] for sid in rmgEstu]

                # Compute cross-sectional proxy fill in for initial estimate
                if rmg.mnemonic in self.countryProxyList:
                    logging.info('Performing cross-sectional initial estimate for %s', rmg.mnemonic)
                    tmpReturns.data, dummy1 = Utilities.proxyMissingAssetData(
                            [rmg], date, tmpReturns.data, tmpData, modelDB,
                            outlierClass1, countryFactor=True, industryGroupFactor=False,
                            debugging=self.debuggingReporting, gicsDate=self.gicsDate,
                            pctGoodReturns=self.simpleProxyRetTol)
                    tmpReturns.data = ma.clip(tmpReturns.data, -0.75, 2.5)
                    tmpReturns.data = ma.masked_where(tmpReturns.data==0.0, tmpReturns.data)

                # Extra safety-net - fill in with beta*mkt return if anything has been missed
                marketCaps = modelDB.getAverageMarketCaps(returns.dates[-21:], returns.assets, self.numeraire_id, loadAllDates=True)
                marketCaps = ma.filled(marketCaps, 0.0)
                marketReturnHistory = modelDB.loadRMGMarketReturnHistory(\
                        tmpReturns.dates, [rmg], useAMPs=False, returns=returns, rmgAssetMap=self.rmgAssetMap, avgMcap=marketCaps)
                marketReturnHistory.data = Utilities.screen_data(marketReturnHistory.data)
                returns_idx = [returns.assetIdxMap.get(x) for x in tmpReturns.assets]
                missingIdx_bools = missingDataMask[returns_idx,:]
                missingIG_bools =  ma.getmaskarray(tmpReturns.data)
                okProxyIdx_bools = numpy.logical_and(missingIdx_bools,numpy.logical_not(missingIG_bools))
                noProxyIdx_bools = numpy.logical_and(missingIdx_bools,missingIG_bools)

                initialReturns = Matrices.allMasked(tmpReturns.data.shape)
                initialReturns[okProxyIdx_bools] = tmpReturns.data[okProxyIdx_bools]
                if noProxyIdx_bools.any():
                    logging.info('Loading betas and market return for simple proxy')
                    hBetaDict = modelDB.getHistoricBetaDataV3(date, tmpReturns.assets,
                            field='value', home=1, rollOverData=True)
                    pValueDict = modelDB.getHistoricBetaDataV3(date, tmpReturns.assets,
                            field='p_value', home=1, rollOverData=True)
                    hbeta_list = [hBetaDict.get(x, 1.0) for x in tmpReturns.assets]
                    hbeta_arr = numpy.array(hbeta_list)
                    # Shrink beta towards one based on p-value
                    pValue_list = [pValueDict.get(x, 1.0) for x in tmpReturns.assets]
                    pvalue_arr = numpy.array(pValue_list)
                    hbeta_arr = pvalue_arr + ((1.0-pvalue_arr) * hbeta_arr)
                    hbeta_arr = ma.clip(hbeta_arr, 0.25, 2.0)
                    # Compute simple proxy
                    hbeta_arr = hbeta_arr[:,numpy.newaxis]
                    proxy_values = hbeta_arr * marketReturnHistory.data # broadcasting
                    initialReturns[noProxyIdx_bools] = proxy_values[noProxyIdx_bools]

                # Finally estimate proxies
                rmgReturns = Matrices.TimeSeriesMatrix(rmgSidList, returns.dates)
                rmgReturns.data = ma.take(returns.data, rmgIdxList, axis=0)
                rmgReturns.preIPOFlag = ma.take(returns.preIPOFlag, rmgIdxList, axis=0)
                rmgReturns = self.computeProxyReturns(
                        rmg, rmgEstuIdx, rmgReturns, initialReturns, tmpData, modelDB, marketDB, noProxyList)
                for (idx, sid) in enumerate(rmgSidList):
                    returns.data[returns.assetIdxMap[sid], :] = rmgReturns.data[idx, :]

        # Redo masked data flags
        self.dump_return_history('rets_s5_bfil.csv', returns, returns.dates, returns.data, modelDB, marketDB)
        allMasked = numpy.array(ma.getmaskarray(returns.data), copy=True)

        # Get returns timing adjustments for conversion to USA time zone
        if (self.returnsTimingID is not None) and applyRT:
            returns.data = ma.filled(returns.data, 0.0)
            logging.info('Applying returns timing adjustments to adjust to US market')
            rmgAll = modelDB.getAllRiskModelGroups()
            adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                    self.returnsTimingID, rmgAll, returns.dates, legacy=False)
            adjustments.data = ma.filled(adjustments.data, 0.0)
            # If required, adjust all returns to US time-zone
            returns.data = self.adjustReturnsForTiming(
                    date, returns, returns.data, adjustments, rmgAll)

        returns.data = ma.masked_where(allMasked, returns.data)
        returns.data = ma.where(returns.data<=self.returnLowerBound, self.returnLowerBound, returns.data)
        if trimData:
            outlierClass0 = Outliers.Outliers()
            returns.data = outlierClass0.twodMAD(returns.data)

        self.dump_return_history('rets_s6_final.csv', returns, returns.dates, returns.data, modelDB, marketDB, outspac=True)
        return returns

    def outputDebugLine(self, ms, returnsData):
        logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX %s: Data bounds: (Min, Mean, Max) [%.3f, %.3f, %.3f]' % \
                (ms, ma.min(returnsData, axis=None), ma.average(returnsData, axis=None),
                    ma.max(returnsData, axis=None)))
        return

    def output_stats(self, dataIn, name):
        logger = logging.getLogger()
        if logger.isEnabledFor(logging.DEBUG):
            data = Utilities.screen_data(dataIn)
            mask = ma.getmaskarray(data)
            zeros = ma.where(ma.ravel(data)==0.0)[0]
            logging.debug('%s data: dim(%s) Bounds: [%.3f, %.3f, %.3f], Missing: %d, Zero: %d',
                    name, data.shape, ma.min(data, axis=None),
                    ma.average(data, axis=None), ma.max(data, axis=None),
                    len(numpy.flatnonzero(mask)), len(numpy.ravel(zeros)))

    def computeProxyReturns(self, rmg, estuIdx, returns, initialReturns, assetData, modelDB, marketDB, noProxyList):
        """Function to compute proxy returns for a market
        via an APCA model. These proxies are used to fill in
        missing returns, which are then fed back into the APCA
        model until convergence is achieved
        """
        logging.info('Computing proxy returns for missing values')
        # Set up initial parameters
        nFactors = min(20, int(len(estuIdx)/5))
        nFactors = max(nFactors, 1)
        pcaInstance = FactorReturns.AsymptoticPrincipalComponents2017(
                nFactors, trimExtremeExposures=False, replaceReturns=False, flexible=True, TOL=None)
        returnsCopy = Matrices.TimeSeriesMatrix(returns.assets, returns.dates)
        returnsCopy.data = Utilities.screen_data(returns.data)
        initialReturns = Utilities.screen_data(initialReturns)

        # Sort out what's missing and what's not
        maskedData = numpy.array(ma.getmaskarray(returnsCopy.data), copy=True)
        returnsCopy.missingFlag = numpy.array(ma.getmaskarray(returnsCopy.data), copy=True)
        returnsCopy.data = ma.filled(returnsCopy.data, 0.0)
        fillIndices = [idx for (idx, sid) in enumerate(assetData.universe) if sid not in noProxyList]

        # Fill in missing returns with initial estimate
        returnsCopy.data = fill_and_smooth_returns(returnsCopy.data, initialReturns, maskedData,
                indices=fillIndices, preIPOFlag=returns.preIPOFlag)[0]

        # Trim outliers along region/sector buckets
        opms = dict()
        opms['nBounds'] = [3.0, 3.0]
        outlierClass = Outliers.Outliers(opms, industryClassificationDate=self.gicsDate)
        returnsCopy.data = outlierClass.bucketedMAD(
                rmg, returns.dates[-1], returnsCopy.data, assetData, modelDB, axis=0)

        # Build Statistical model of returns
        if len(estuIdx) < 1:
            logging.warning('Empty estimation universe for %s', self.rmg)
            return returns
        if returnsCopy.data.shape[1] < 21:
            logging.warning('History length too short (%d) for proxy return step',
                    returnsCopy.data.shape[1])
            return returns
        (X, F, Delta, ANOVA, pctgVar) = pcaInstance.calc_ExposuresAndReturns(returnsCopy, estu=estuIdx)
        proxyReturns = ma.clip(numpy.dot(X, F), -0.75, 1.5)

        # Fill in missing returns with proxy values
        returnsCopy.data = fill_and_smooth_returns(returns.data, proxyReturns, maskedData,
                indices=fillIndices, preIPOFlag=returns.preIPOFlag)[0]

        returns.data = numpy.array(returnsCopy.data, copy=True)
        return returns

    def adjustReturnsForTiming(self, modelDate, returns, returnsData, adjustments, rmgList,
                                    alignWithHome=False, USAdjustMentLater=False):
        """"Adjust returns for returns-timing, aligning them either with the US market
        or with each asset's "home" market.
        Both returns and adjustments should be TimeSeriesMatrix objects containing the
        asset returns and market adjustment factors time-series,
        respectively.  The adjustments array cannot contain masked values.
        Returns the adjusted data.
        """
        logging.debug('adjustReturnsForTiming: begin')
        subIssues = returns.assets
        data = ma.array(returnsData, copy=True)
        rmgIdxMap = dict([(rmg.rmg_id,i) for (i,rmg) in enumerate(adjustments.assets)])

        # Determine asset's home and trading countries (country of quotation)
        # If DRs have already been adjusted to home, then the trading country is already the 
        # home country for everything
        homeCountryMap = dict((sid, rmg) for (rmg, idSet) in self.rmgAssetMap.items() for sid in idSet)
        if alignWithHome:
            tradingCountryMap = dict((sid, rmg) for (rmg, idSet) in self.tradingRmgAssetMap.items() for sid in idSet)
        else:
            tradingCountryMap = copy.deepcopy(homeCountryMap)

        # Set up market time zones
        rmgZoneMap = dict((rmg.rmg_id, rmg.gmt_offset) for rmg in rmgList)
        dateLen = len(returns.dates)

        # Compute adjustments to align with US market
        tradingRMGId_array = numpy.array([tradingCountryMap.get(sid, numpy.nan) for sid in subIssues])
        rmgIdx_array_US     = numpy.array([Utilities.readMap(rmg, rmgIdxMap, numpy.nan) for rmg in tradingRMGId_array])
        not_nan_idx_US = numpy.where(~numpy.isnan(rmgIdx_array_US))

        # Compute adjustments to convert to home country time-zone if required
        if alignWithHome:
            homeRMGId_array = numpy.array([homeCountryMap.get(sid, numpy.nan) for sid in subIssues])
            rmgIdx_array_Home = numpy.array([Utilities.readMap(rmg, rmgIdxMap, numpy.nan) for rmg in homeRMGId_array])
            not_nan_idx_Home = numpy.where(~numpy.isnan(rmgIdx_array_Home))

            # Compute difference in time zones between home and trading market
            # Note that we skip this bit if we're later converting to US returns, as we'd have
            # the wrong 'trading' market otherwise
            if not USAdjustMentLater:
                tzDiff = numpy.array([Utilities.readMap(rmg, rmgZoneMap, 0.0) for rmg in list(tradingRMGId_array)]) \
                        - numpy.array([Utilities.readMap(rmg, rmgZoneMap, 0.0) for rmg in list(homeRMGId_array)])
                tzDiffIdx = numpy.flatnonzero(tzDiff.astype(int))
                not_nan_idx_US = numpy.intersect1d(not_nan_idx_US, tzDiffIdx)
                not_nan_idx_Home = numpy.intersect1d(not_nan_idx_Home, tzDiffIdx)

            # First align with US market
            data[not_nan_idx_US,:] += adjustments.data[rmgIdx_array_US[not_nan_idx_US].astype(int),:dateLen]
            # Now adjust to home market time-zone
            data[not_nan_idx_Home,:] -= adjustments.data[rmgIdx_array_Home[not_nan_idx_Home].astype(int),:dateLen]
        else:
            # Adjust to align with US market
            data[not_nan_idx_US,:] += adjustments.data[rmgIdx_array_US[not_nan_idx_US].astype(int),:dateLen]

        logging.debug('adjustReturnsForTiming: end')
        return data

def compute_returns_timing_adjustments(rmgList, date, modelDB, synchMarkets, debugReporting=False):
    """ Uses a VAR-based technique to synchronise market returns
    for early-closing markets with those of later markets
    """
    logging.debug('synchronise_returns: begin')

    # Bucket markets into regions
    regionMarketMap = dict()
    for rmg in rmgList:
        if rmg.region_id not in regionMarketMap:
            regionMarketMap[rmg.region_id] = list()
        totalRMG = modelDB.getRiskModelGroup(rmg.rmg_id)
        totalRMG.setRMGInfoForDate(date)
        regionMarketMap[rmg.region_id].append(totalRMG)

    # Re-order list of RMGs by region (purely for convenience of debugging output)
    rmgList = []
    regionOrder = [11, 12, 13, 17, 16, 14, 15]
    for region_id in regionOrder:
        if region_id in regionMarketMap:
            rmgList.extend(regionMarketMap[region_id])

    # Load history of local market returns
    dates = modelDB.getAllRMGDateRange(date, 730, excludeWeekend=True)
    allDates = modelDB.getDateRange(None, dates[0], dates[-1], excludeWeekend=False)
    if dates[-1] != date:
        return dict(), dict()
    marketReturnHistory = modelDB.loadRMGMarketReturnHistory(allDates, rmgList, useAMPs=False)

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        outFile = 'tmp/mktretHist-%s.csv' % date
        Utilities.writeToCSV(ma.transpose(marketReturnHistory.data),
                outFile, columnNames=rmgNames, rowNames=allDates)
        proxy = modelDB.loadReturnsTimingAdjustmentsHistory(
                1, rmgList, allDates, loadProxy=True, legacy=False)
        adj = modelDB.loadReturnsTimingAdjustmentsHistory(
                1, rmgList, allDates, legacy=False)
        mktProxies = ma.filled(proxy.data, 0.0) + ma.filled(adj.data, 0.0)
        adjHistory = ma.filled(marketReturnHistory.data, 0.0) + mktProxies
        outFile = 'tmp/mktretHistAdj-%s.csv' % date
        Utilities.writeToCSV(ma.transpose(adjHistory),
                outFile, columnNames=rmgNames, rowNames=allDates)
        adj = modelDB.loadReturnsTimingAdjustmentsHistory(
                1, rmgList, allDates, legacy=True)
        adjHistory = ma.filled(marketReturnHistory.data, 0.0) + ma.filled(adj.data, 0.0)
        outFile = 'tmp/mktretHistAdj-Legacy-%s.csv' % date
        Utilities.writeToCSV(ma.transpose(adjHistory),
                outFile, columnNames=rmgNames, rowNames=allDates)

    # Align dates and trim outliers
    marketReturnHistoryData = compute_compound_returns_v3(
            marketReturnHistory.data, marketReturnHistory.dates, dates, keepFirst=True)[0]
    missingDataMask = numpy.array(ma.getmaskarray(marketReturnHistoryData), copy=True)
    outlierClass = Outliers.Outliers()
    nB = [8.0, 8.0, 8.0, 8.0]
    initialReturns = ma.filled(outlierClass.twodMAD(marketReturnHistoryData, nBounds=nB), 0.0)

    # First fill in missing values
    itr = 0
    maxIter = 25
    tol = 1.0e-4
    validMkts = [rmg for rmg in rmgList if rmg.developed]
    while itr < maxIter:
        # Compute pseudo returns history
        pseudoMarketReturn = fillMissingReturns(
                rmgList, date, modelDB, numpy.fliplr(initialReturns),
                validMkts, debugReporting=debugReporting)
        pseudoMarketReturn = numpy.array(numpy.fliplr(pseudoMarketReturn))

        # Create filled-in returns
        filledMarketReturn = fill_and_smooth_returns(
                marketReturnHistoryData, pseudoMarketReturn, mask=missingDataMask)[0]

        # Check for convergence
        if itr > 0:
            diff = pseudoMarketReturn - previousEstimate
            error = ma.sum(diff*diff, axis=None)
            logging.info('Iteration: %d, Residual norm: %.6f', itr, error)
            if error < tol:
                break

        # Reset
        itr += 1
        previousEstimate = numpy.array(pseudoMarketReturn, copy=True)
        initialReturns = outlierClass.twodMAD(filledMarketReturn, nBounds=nB, suppressOutput=True)

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        outFile = 'tmp/mktret-%s.csv' % date
        Utilities.writeToCSV(marketReturnHistoryData[:,-1][numpy.newaxis,:], outFile, columnNames=rmgNames, rowNames=[dates[-1]])
        outFile = 'tmp/mktretFilled-%s.csv' % date
        Utilities.writeToCSV(filledMarketReturn[:,-1][numpy.newaxis,:], outFile, columnNames=rmgNames, rowNames=[dates[-1]])

    # Compute adjusted market returns
    synchedMarketReturn = computeAdjustedReturns(rmgList, date, modelDB,
        numpy.fliplr(filledMarketReturn), synchMarkets, debugReporting=debugReporting)
    if len(synchedMarketReturn) == 0:
        return dict(), dict()
    synchedMarketReturn = numpy.fliplr(synchedMarketReturn)

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        outFile = 'tmp/mktretAdjusted-%s.csv' % date
        Utilities.writeToCSV(synchedMarketReturn[:,-1][numpy.newaxis,:], outFile, columnNames=rmgNames, rowNames=[dates[-1]])
    
    # Compute adjustments
    proxyAdjustments = filledMarketReturn[:,-1] - ma.filled(marketReturnHistoryData[:,-1], 0.0)
    timingAdjustments = synchedMarketReturn[:,-1] - filledMarketReturn[:,-1]
    for (rmg, adj, t_adj) in zip(rmgList, proxyAdjustments, timingAdjustments):
        logging.info('Date: %s, Market %s: %s, GMT-off: %d, Proxy Adjustment: %.4f, Timing Adjustment: %.8f',
                date, rmg.mnemonic, rmg.description, rmg.gmt_offset, adj, t_adj)

    # Set up dicts of adjustments
    proxyAdjustments = ma.masked_where(proxyAdjustments==0.0, proxyAdjustments)
    timingAdjustments = ma.masked_where(timingAdjustments==0.0, timingAdjustments)
    proxyAdjustDict = dict((rmg, adj) for (rmg, adj) in \
            zip(rmgList, proxyAdjustments) if adj is not ma.masked)
    timingAdjustDict = dict((rmg, adj) for (rmg, adj) in \
            zip(rmgList, timingAdjustments) if adj is not ma.masked)

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        outFile = 'tmp/RetTim-%s.csv' % date
        data = timingAdjustments[:, numpy.newaxis]
        Utilities.writeToCSV(data, outFile, rowNames=rmgNames, columnNames=['%s' % date])
        outFile = 'tmp/MktProxy-%s.csv' % date
        data = proxyAdjustments[:, numpy.newaxis]
        Utilities.writeToCSV(data, outFile, rowNames=rmgNames, columnNames=['%s' % date])

    return proxyAdjustDict, timingAdjustDict

def computeAdjustedReturns(rmgList, date, modelDB, marketReturnHistory,
                        synchMarkets, maxLags=1, debugReporting=False):
    # Uses a VAR technique to predict market returns for
    # early markets based on those of the later markets

    # Initialise variables
    n = len(rmgList)
    T = marketReturnHistory.shape[1]
    # Resize the number of lags if insufficient observations
    p = int((T-1)/(n+1))
    p = min(p, maxLags)
    p = max(p, 1)

    if T < p+1:
        # If too few observations to do anything at all, abort
        logging.warning('Not enough observations (%s) to perform VAR-%s', T, p)
        return []
    np = n*p
    logging.info('Using %s lags, %s Time Periods, iDim %s for %s variables', p, T, np, n)
    
    # Set up lagged return history matrix
    bigBlockReturnMatrix = numpy.zeros((np,T-p), float)
    for j in range(T-p):
        for iBlock in range(p):
            iLoc = iBlock*n
            bigBlockReturnMatrix[iLoc:iLoc+n,j] = marketReturnHistory[:,j+iBlock]
    currentReturnMatrix = marketReturnHistory[:,:T-p-1]
    
    # Loop round each market and compute the relevant row of weights M
    M = numpy.zeros((n,np), float)
    MStat = numpy.zeros((n,np+1), float)
    for (i, rmg) in enumerate(rmgList):

        # Ensure that only markets trading after particular market
        # has closed have non-zero values in M
        mktIdx = [idx for idx in range(n)
                if (rmgList[idx].gmt_offset+2) < rmg.gmt_offset
                and rmgList[idx] in synchMarkets]

        if len(mktIdx) > 0:

            # Pick out particular market's return
            mktRets = currentReturnMatrix[i,:][numpy.newaxis,:]
            # Pick out selection of lagged returns
            for ip in range(p-1):
                ids = [(1+ip)*m for m in mktIdx]
                mktIdx.extend(ids)
            subBlock = ma.take(bigBlockReturnMatrix, mktIdx, axis=0)
            lagRets = subBlock[:,1:]

            # Check to make sure that current day's reference market
            # returns are non-missing. Skip everything if they are
            curMktRet = subBlock[:,0]
            curMktRet = ma.masked_where(abs(curMktRet)<1.0e-12, curMktRet)
            missingCurMkt = numpy.flatnonzero(ma.sum(
                ma.getmaskarray(curMktRet), axis=0))
            if len(missingCurMkt) == len(synchMarkets):
                mktList = [r.description for r in synchMarkets]
                logging.warning('Missing current return for markets: %s', mktList)
                logging.warning('Aborting...')
                return []

            # Solve the matrix system via non-negative least squares
            X = numpy.transpose(numpy.array(ma.filled(lagRets, 0.0)))
            y = numpy.array(numpy.ravel(ma.filled(mktRets, 0.0)))
            m = Utilities.non_negative_least_square(X, y)
            resid = y - numpy.sum(X*m, axis=1)
             
            # Pick out non-zero coefficients
            nObs = len(y)
            maskCoeffs = ma.masked_where(m < 1.0e-12, m)
            nonZeroIdx = numpy.flatnonzero(ma.getmaskarray(maskCoeffs)==0)
            nPar = len(nonZeroIdx)
             
            # Compute t-stats
            tStats = numpy.zeros((len(m)), float)
            if nPar > 0:
                stdErr = Utilities.computeRegressionStdError(\
                        resid, ma.take(X, nonZeroIdx, axis=1))
                ts = ma.take(m, nonZeroIdx, axis=0) / stdErr
                numpy.put(tStats, nonZeroIdx, ts)
                
            # Compute R-Square
            sst = float(ma.inner(y, y))
            sse = float(ma.inner(resid, resid))
            adjr = 0.0
            if sst > 0.0:
                adjr = 1.0 - sse / sst
            if nObs > nPar:
                adjr = max(1.0 - (1.0-adjr)*(nObs-1)/(nObs-nPar), 0.0)
                
            # Store regression statistics
            for (j,idx) in enumerate(mktIdx):
                M[i,idx] = m[j]
                MStat[i,idx] = tStats[j] * tStats[j]
                MStat[i,-1] = adjr 
    
    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        syncNames = [r.description.replace(',','') for r in synchMarkets]
        syncNames = ['%s|%s' % (date, nm) for nm in syncNames]
        mktIdx = [idx for idx in range(n) if rmgList[idx] in synchMarkets]
        j0 = 0
        j1 = np
        for lag in range(p):
            outFile = 'tmp/M-Lag%d-%s.csv' % (lag, date)
            msub = numpy.transpose(M[:,j0:j1])
            msub = numpy.take(msub, mktIdx, axis=0)
            Utilities.writeToCSV(msub, outFile, rowNames=syncNames, columnNames=rmgNames)
            outFile = 'tmp/MStats-Lag%d-%s.csv' % (lag, date)
            msub = numpy.transpose(MStat[:,j0:j1])
            msub = numpy.take(msub, mktIdx, axis=0)
            Utilities.writeToCSV(msub, outFile, rowNames=syncNames, columnNames=rmgNames)
            j0+=np
            j1+=np

    # Compute synchronised returns
    residual = currentReturnMatrix - numpy.dot(M, bigBlockReturnMatrix[:,1:])
    synchedMarketReturn = numpy.dot(M, bigBlockReturnMatrix[:,:-1]) + residual
    return synchedMarketReturn

def fillMissingReturns(rmgList, date, modelDB, marketReturnHistory,
                        synchMarkets, debugReporting=False):
    # Computes sensitivity of each market to selection of other markets
    # in order to impute missing values

    # Initialise variables
    n = len(rmgList)
    T = marketReturnHistory.shape[1]
    M = numpy.zeros((n,n), float)
    MStat = numpy.zeros((n,n+1), float)

    if T < 6:
        # If too few observations to do anything at all, abort
        logging.warning('Not enough observations (%s) to compute betas', T)
        return []

    # Loop round each market and compute the relevant row of weights M
    for (i, rmg) in enumerate(rmgList):

        mktIdx = [idx for idx in range(n)
                if (rmgList[idx].gmt_offset <= rmg.gmt_offset+2)
                and (rmg.gmt_offset-2 <= rmgList[idx].gmt_offset)]
        mktIdx = [idx for idx in mktIdx if rmgList[idx] in synchMarkets \
                    and (rmg != rmgList[idx])]

        if len(mktIdx) > 0:

            # Pick out relevant return series
            lhs = marketReturnHistory[i,:]
            rhs = ma.take(marketReturnHistory, mktIdx, axis=0)

            # Solve the matrix system via non-negative least squares
            X = numpy.transpose(numpy.array(ma.filled(rhs, 0.0)))
            y = numpy.array(ma.filled(lhs, 0.0))
            m = Utilities.non_negative_least_square(X, y)
            resid = y - numpy.sum(X*m, axis=1)

            # Pick out non-zero coefficients
            nObs = len(y)
            maskCoeffs = ma.masked_where(m < 1.0e-12, m)
            nonZeroIdx = numpy.flatnonzero(ma.getmaskarray(maskCoeffs)==0)
            nPar = len(nonZeroIdx)

            # Compute t-stats
            tStats = numpy.zeros((len(m)), float)
            if nPar > 0:
                stdErr = Utilities.computeRegressionStdError(\
                        resid, ma.take(X, nonZeroIdx, axis=1))
                ts = ma.take(m, nonZeroIdx, axis=0) / stdErr
                numpy.put(tStats, nonZeroIdx, ts)

            # Compute R-Square
            sst = float(ma.inner(y, y))
            sse = float(ma.inner(resid, resid))
            adjr = 0.0
            if sst > 0.0:
                adjr = 1.0 - sse / sst
            if nObs > nPar:
                adjr = max(1.0 - (1.0-adjr)*(nObs-1)/(nObs-nPar), 0.0)

            # Store regression statistics
            for (j,idx) in enumerate(mktIdx):
                M[i,idx] = m[j]
                MStat[i,idx] = tStats[j] * tStats[j]
                MStat[i,-1] = adjr

    if debugReporting:
        rmgNames = [r.description.replace(',','') for r in rmgList]
        outFile = 'tmp/M-%s.csv' % date
        Utilities.writeToCSV(M, outFile, rowNames=rmgNames, columnNames=rmgNames)
        outFile = 'tmp/MStats-%s.csv' % date
        Utilities.writeToCSV(MStat, outFile, rowNames=rmgNames,
                columnNames=rmgNames+['Adj-RSquare'])

    # Compute proxy returns
    return numpy.dot(M, ma.filled(marketReturnHistory, 0.0))

def compute_compound_returns_v3(dataIn, highFreqDateListIn, lowFreqDateListIn,
                                keepFirst=False, matchDates=False,
                                sumVals=False, mean=False):
    """Converts a set of higher frequency returns (e.g. daily)
    into lower frequency (weekly, monthly)
    data is an n*t array where t is the number of dates in highFreqDateList
    """
    data = Utilities.screen_data(dataIn)
    highFreqDateList = list(highFreqDateListIn)
    lowFreqDateList = list(lowFreqDateListIn)
    maskArray = numpy.array(ma.getmaskarray(data)==0, copy=True)
    data = ma.filled(data, 0.0)

    # Dimensional rearranging if input is a vector
    vector = False
    if len(data.shape) == 1:
        data = data[numpy.newaxis,:]
        maskArray = maskArray[numpy.newaxis,:]
        vector = True

    logging.debug('Compounding %s returns from %s dates to %s',
            data.shape[0], len(highFreqDateList), len(lowFreqDateList))
    # Sift out dates with no returns that are not in the output date list
    sumNonMissing = ma.sum(maskArray, axis=0)
    sumNonMissing = ma.masked_where(sumNonMissing>0, sumNonMissing)
    allMissingIdx = numpy.flatnonzero(ma.getmaskarray(sumNonMissing)==0)
    allMissingIdx = [idx for idx in allMissingIdx \
            if highFreqDateList[idx] not in lowFreqDateList]
    if len(allMissingIdx) > 0:
        logging.debug('Removing %d dates where all returns missing',
                len(allMissingIdx))
        okIdx = [idx for idx in range(data.shape[1]) if idx not in allMissingIdx]
        highFreqDateList = [highFreqDateList[idx] for idx in okIdx]
        data = numpy.take(data, okIdx, axis=1)
        maskArray = numpy.take(maskArray, okIdx, axis=1)

    if sorted(highFreqDateList) == sorted(lowFreqDateList):
        logging.debug('Lists of dates are identical: skipping')
        data = ma.masked_where(maskArray==0, data)
        if vector:
            data = data[0,:]
        return (data, highFreqDateList)

    origLFDates = list(lowFreqDateList)
    if len(lowFreqDateList) < 1:
        logging.warning('Problem with dates: zero length. Bailing...')
        return (dataIn, highFreqDateList)

    # Set up date info
    lowFreqDateList = [d for d in lowFreqDateList \
            if d >= highFreqDateList[0] and d <= highFreqDateList[-1]]
    dateIdxMap = dict([(d,i) for (i,d) in enumerate(highFreqDateList)])
    dt = highFreqDateList[0]
    idx = None
    while dt < highFreqDateList[-1]:
        if dt not in dateIdxMap:
            dateIdxMap[dt] = idx
        idx = dateIdxMap[dt]
        dt += datetime.timedelta(1)
    dIds = [dateIdxMap[d] for d in lowFreqDateList]

    # Work out compound frequency
    freq = int(len(highFreqDateList) / float(len(lowFreqDateList)))
    if freq < 2:
        keepFirst = True
    else:
        if dateIdxMap[highFreqDateList[-1]] - dateIdxMap[lowFreqDateList[-1]] >= freq:
            dIds.append(dateIdxMap[highFreqDateList[-1]])

    # Compute compound returns
    freqList = []
    if not sumVals:
        lowFreqData = numpy.zeros((data.shape[0],len(dIds)-1), float)
        id0 = dIds[0]
        for (idx,id1) in enumerate(dIds[1:]):
            if id1 == id0:
                lowFreqData[:,idx] = data[:,id0]
                freqList.append(1.0)
            else:
                lowFreqData[:,idx] = numpy.product(data[:,id0+1:id1+1]+1.0, axis=1) - 1.0
                freqList.append(id1-id0)
            id0 = id1
    else:
        lowFreqData = numpy.zeros((data.shape[0],len(dIds)-1), float)
        id0 = dIds[0]
        for (idx,id1) in enumerate(dIds[1:]):
            if id1 == id0:
                lowFreqData[:,idx] = data[:,id0]
                freqList.append(1.0)
            else:
                lowFreqData[:,idx] = numpy.sum(data[:,id0+1:id1+1], axis=1)
                freqList.append(id1-id0)
            id0 = id1
    cumMask = numpy.cumsum(maskArray, axis=1)
    cumMaskSample = numpy.take(cumMask, dIds, axis=1)
    lowFreqMask = cumMaskSample[:,1:] - cumMaskSample[:,:-1]
    lowFreqMask = numpy.array(lowFreqMask, dtype='bool')
    if keepFirst:
        # Optional retaining of first return if we're merely removing
        # a few rogue dates rather than truly compounding
        lowFreqData = numpy.concatenate((
            data[:,dIds[0]][:,numpy.newaxis], lowFreqData), axis=1)
        lowFreqMask = numpy.concatenate((
            cumMaskSample[:,0][:,numpy.newaxis], lowFreqMask), axis=1)
        freqList = [1.0] + freqList
    dateList = lowFreqDateList[:lowFreqData.shape[1]]
    if mean:
        if not sumVals:
            for idx in range(lowFreqData.shape[1]):
                lowFreqData[:,idx] = lowFreqData[:,idx] ** (1.0/float(freqList[idx]))
        else:
            for idx in range(lowFreqData.shape[1]):
                lowFreqData[:,idx] = lowFreqData[:,idx] / float(freqList[idx])

    # Replace masks where appropriate
    lowFreqData = ma.masked_where(lowFreqMask==0, lowFreqData)

    # If required, map returns to array of original size/dates
    if matchDates:
        tmpData = Matrices.allMasked((data.shape[0], len(origLFDates)))
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(dateList)])
        for (ii, d) in enumerate(origLFDates):
            if d in dateIdxMap:
                idx = dateIdxMap[d]
                tmpData[:,ii] = lowFreqData[:,idx]
        lowFreqData = tmpData
        dateList = origLFDates

    if vector:
        lowFreqData = lowFreqData[0,:]
    return (lowFreqData, dateList)

def compute_compound_returns_v4(dataIn, highFreqDateListIn, lowFreqDateListIn, fillWithZeros=True):
    """Converts a set of higher frequency returns (e.g., daily)
    into lower frequency returns (e.g., weekly, monthly)
    -- data is an n*t array of returns corresponding to the t dates in
    highFreqDateList, where the t dates are the end-of-period dates for
    the t time periods
    -- note that the dates in lowFreqDateListIn are assumed to be the
    end-of-period dates for the periods to which returns are compounded
    -- the date list that is returned from this function corresponds to the
    end-of-period dates for the lower frequency returns
    -- if fillWithZeros is True, low frequency returns for which ALL
    corresponding high frequency returns are null will be set to zeros (and
    will not be masked), otherwise they will be set to np.nan (and masked)
    """

    # get mask identifying nans, infs, and masked values
    if isinstance(dataIn, ma.MaskedArray):
        if not isinstance(dataIn.mask, numpy.bool_):
            dataMask = numpy.isnan(dataIn.data) + numpy.isinf(dataIn.data) + dataIn.mask
        else:
            dataMask = numpy.isnan(dataIn.data) + numpy.isinf(dataIn.data)
    else:
        dataMask = numpy.isnan(dataIn) + numpy.isinf(dataIn)

    data = ma.array(dataIn, copy=True) # convert ndarray or ma to ma
    data = ma.filled(data, 0.0) # create ndarray from ma, filling masked values with zeros
    data[dataMask] = 0.0
    vector = False
    if len(data.shape) == 1:
        vector = True
        data = data[numpy.newaxis, :]
        dataMask = dataMask[numpy.newaxis, :]

    highFreqDateList = list(highFreqDateListIn)
    lowFreqDateList = list(lowFreqDateListIn)
    assert(data.shape[1] == len(highFreqDateListIn))
    if sorted(highFreqDateList) == sorted(lowFreqDateList):
        if fillWithZeros:
            return (data, highFreqDateList)
        else:
            return (dataIn, highFreqDateList)

    # compound returns
    lowFreqDateList = sorted([d for d in lowFreqDateList \
            if d >= highFreqDateList[0] and d <= highFreqDateList[-1]])
    dateIdxMap = dict([(d,i) for (i,d) in enumerate(highFreqDateList)])
    lowFreqData = numpy.empty((data.shape[0], len(lowFreqDateList)), float)
    if fillWithZeros:
        lowFreqData[:] = 0.
    else:
        lowFreqData[:] = numpy.nan
    mask = numpy.ones(lowFreqData.shape[0], dtype=numpy.bool)
    for idx, dt in enumerate(lowFreqDateList):
        if idx == 0:
           retIdx = [dateIdxMap[d] for d in dateIdxMap.keys() if d <= dt]
        else:
           prevDt = lowFreqDateList[idx-1]
           retIdx = [dateIdxMap[d] for d in dateIdxMap.keys() if d > prevDt and d <= dt]
        prets = numpy.take(data, retIdx, axis=1)
        cumrets = numpy.product(1. + prets, axis=1) - 1.
        if not fillWithZeros:
            if prets.shape[1] == 1:
                mask = ~numpy.take(dataMask, retIdx, axis=1)[:, 0]
            else:
                # only replace values if at least one pret is not masked/null
                mask = numpy.take(dataMask, retIdx, axis=1).sum(axis=1) < len(retIdx)
        lowFreqData[mask, idx]  = cumrets[mask]

    if vector:
        lowFreqData = lowFreqData.flatten()

    return (ma.masked_invalid(lowFreqData), lowFreqDateList)

def fill_and_smooth_returns(data, fillData, mask=None, indices=None, preIPOFlag=None):
    """ Given a returns matrix with missing values,
    together with an array of the same size containing proxies, this
    routine back-fills the history, and smooths out the
    non-missing values, to preserve the long-term return
    """
    logging.debug('fill_and_smooth_returns: begin')
    returns = Utilities.screen_data(data)
    if mask is None:
        maskArray = numpy.array(ma.getmaskarray(returns), copy=True)
    else:
        maskArray = mask.copy()
    if preIPOFlag is None:
        preIPOFlag = numpy.zeros((returns.shape), bool)

    returns = ma.filled(returns, 0.0)

    # Manipulation of axes in case of 1-D arrays
    vector = False
    if len(data.shape) < 2:
        logging.info('Data is 1-D: assuming row vector')
        returns = returns[numpy.newaxis,:]
        fillData = fillData[numpy.newaxis,:]
        maskArray = maskArray[numpy.newaxis,:]
        vector = True
    if indices is None:
        indices = list(range(data.shape[0]))

    try:
        import nantools.utils as nanutils
        fill_and_smooth_returns_fast(returns, fillData, maskArray, indices, preIPOFlag)
    except ImportError:
        raise
        fill_and_smooth_returns_slow(returns, fillData, maskArray, indices, preIPOFlag)

    returns = ma.array(returns, mask=maskArray)
    if vector:
        returns = ma.ravel(returns)
        maskArray = ma.ravel(maskArray)
    logging.debug('fill_and_smooth_returns: end')
    return (returns, maskArray)

def fill_and_smooth_returns_fast(returns, fillData, maskArray,
        indices, preIPOFlag):
    import nantools.utils as nanutils
    fillData = ma.filled(fillData, numpy.nan)
    preIPOFlag = preIPOFlag.view(dtype=numpy.int8)
    maskArray = maskArray.view(dtype=numpy.int8)
    indices = numpy.array(indices,dtype=numpy.intp)
    nanutils.fill_and_smooth(returns, fillData, maskArray, preIPOFlag, indices)

def fill_and_smooth_returns_slow(returns, fillData, maskArray, indices, preIPOFlag):
    proxyMask = ma.getmaskarray(fillData)

    # Loop round indices to be filled by the relevant proxy
    for idx in indices:
        maskedRetIdx = numpy.flatnonzero(maskArray[idx,:])
        if len(maskedRetIdx) > 0:
            t0 = maskedRetIdx[0]
            cumret = 1.0
            for t in maskedRetIdx:
                tmin1 = max(t-1,0)
                # Fill in particular masked return
                if not proxyMask[idx,t]:
                    returns[idx, t] = fillData[idx,t]
                    maskArray[idx, t] = False
                # If consecutive missing returns, compound replacement returns
                if t-t0 < 2:
                    if not maskArray[idx,t] and not preIPOFlag[idx,tmin1]:
                        cumret = (1.0 + returns[idx,t]) * cumret
                # Else scale non-missing return by previous compounded return
                else:
                    if not maskArray[idx,t0+1]:
                        returns[idx,t0+1] = -1.0 + (1.0 + returns[idx,t0+1]) / cumret
                    else:
                        returns[idx,t0+1] = -1.0 + (1.0 / cumret)
                        maskArray[idx,t0+1] = False
                    if not maskArray[idx,t] and not preIPOFlag[idx,tmin1]:
                        cumret = 1.0 + returns[idx,t]
                    else:
                        cumret = 1.0
                t0 = t

            # Deal with last in list
            if t < returns.shape[1]-1:
                if not maskArray[idx,t+1]:
                    returns[idx,t+1] = -1.0 + (1.0 + returns[idx,t+1]) / cumret
                else:
                    returns[idx,t+1] = -1.0 + (1.0 / cumret)
                    maskArray[idx,t+1] = False

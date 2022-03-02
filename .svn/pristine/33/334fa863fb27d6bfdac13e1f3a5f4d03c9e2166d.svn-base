
import logging
import datetime
import numpy.ma as ma
import numpy.linalg as linalg
import numpy
import scipy.stats as stats
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import LegacyUtilities as Utilities
from riskmodels import FactorReturns
from riskmodels import LegacyFactorReturns
from riskmodels import Outliers
try:
    import nantools.utils as nanutils
    oldCode = False
except:
    logging.debug('Package nanutils not installed: reverting to old backfill code')
    oldCode = True

class assetReturnsProcessor:
    """Class to perform various returns-related tasks, such as proxying
    and back-filling
    """
    def __init__(self, rmgList, universe, rmgAssetMap, tradingRmgAssetMap,
            assetTypeDict, marketTypeDict, dr2UnderDict=None,
            debuggingReporting=False, numeraire_id=1, returnsTimingID=1,
            estu=None, testOnly=False, dontWrite=False, saveIDs=[]):
        from riskmodels.CurrencyRisk import ModelCurrency
        self.numeraire_id = numeraire_id
        self.returnsTimingID = returnsTimingID
        self.rmg = rmgList
        self.universe = universe
        self.rmgAssetMap = rmgAssetMap # asset/home-country mapping
        self.tradingRmgAssetMap = tradingRmgAssetMap # asset/trading-country mapping
        self.assetTypeDict = assetTypeDict
        self.marketTypeDict = marketTypeDict
        self.dr2UnderDict = dr2UnderDict
        self.debuggingReporting = debuggingReporting
        self.testOnly = testOnly
        self.dontWrite = dontWrite
        self.saveIDs = saveIDs
        if estu is None:
            self.estu = list(range(len(universe)))
        else:
            self.estu = estu

    def back_fill_linked_assets(self, returns, scoreDict,
                        modelDB, marketDB):
        """Backfills missing returns for linked assets using their 
        more liquid partners
        Assumes currencies are already matched across linked assets
        dates are assumed to be ordered oldest to newest
        """
        logging.info('Back-filling linked asset returns')
        # Loop round sets of linked assets
        indexList = []
        assetIdxMap = returns.assetIdxMap
        proxyReturns = ma.array(returns.data, copy=True)
        assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                self.rmgAssetMap.items() for sid in ids])
        assetTradingRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                self.tradingRmgAssetMap.items() for sid in ids])
        for (groupId, subIssueList) in self.subIssueGroups.items():
            # Pull out linked assets and rank them
            scores = scoreDict[groupId]
            scores = numpy.array([scores[sid] for sid in subIssueList])
            indices  = [assetIdxMap[n] for n in subIssueList]
            if len(indices) > 1:
                sortedIndices = [indices[j] for j in numpy.argsort(-scores)]
                sortedSIDs = [returns.assets[j] for j in sortedIndices]
                # Populate proxy return matrix with best available returns
                for (idx, sid) in zip(sortedIndices[1:], sortedSIDs[1:]):
                    for (id0, sid0) in zip(sortedIndices[:idx], sortedSIDs[:idx]):
                        # Only allow back-filling where market matches
                        # If trading markets match, use the full history, otherwise skip
                        # the most recent date as it might not be ready in all markets
                        if assetRMGMap[sid0] == assetRMGMap[sid]:
                            if assetTradingRMGMap[sid0] == assetTradingRMGMap[sid]:
                                proxyReturns[idx,:] = returns.data[id0,:]
                            else:
                                proxyReturns[idx,:-1] = returns.data[id0,:-1]
                            break
                indexList.extend(sortedIndices[1:])

        # Back fill and smooth non-missing non-daily returns
        (data, maskArray) = Utilities.fill_and_smooth_returns(
                returns.data, proxyReturns, indices=indexList,
                preIPOFlag=returns.preIPOFlag)
        logging.debug('Done Back-filling asset returns')
        returns.data[:] = ma.array(data, mask=maskArray)
        return returns

    def dump_return_history(self, fileName, returns, data,
                            modelDB, marketDB, outFile=True):
        if data.shape[1] < 1:
            return
        Utilities.output_stats(data, fileName)
        # For debugging
        if self.debuggingReporting and outFile:
            nameDict = modelDB.getIssueNames(returns.dates[-1], returns.assets, marketDB)
            dateList = [str(d) for d in returns.dates]
            mask = ma.getmaskarray(data)
            if len(self.rmg) > 1:
                filepath = 'tmp/%s' % fileName
            else:
                filepath = 'tmp/%s-%s' % (self.rmg[0].mnemonic, fileName)
            outfile = open(filepath, 'w')
            outfile.write(',,,')
            for d in dateList:
                outfile.write('%s,' % d)
            outfile.write('\n')
            for (groupId, subIssueList) in self.subIssueGroups.items():
                for sid in subIssueList:
                    idx = returns.assetIdxMap[sid]
                    n = 'None'
                    if sid in nameDict:
                        n = nameDict[sid].replace(',','')
                    outfile.write('%s,%s,%s,' % (groupId, sid.getSubIDString(), n))
                    for (t,d) in enumerate(returns.dates):
                        if not mask[idx,t]:
                            outfile.write('%12.8f,' % ma.filled(data[idx,t], 0.0))
                        else:
                            outfile.write(',')
                    outfile.write('\n')
            outfile.close()
         
        if self.debuggingReporting:
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
                        fileName, returns.dates[-1], medianCorr)
            else:
                logging.info('%s run: %s, Date: %s, MEDIAN: %s',
                        self.rmg[0].mnemonic, fileName, returns.dates[-1], medianCorr)
        return

    def compound_currency_returns(self, returns, zeroMask):
        """ Compounds currency returns for missing asset returns into non-missing
        returns locations
        """
        logging.info('Compounding currency and returns-timing adjustments')
        data = Utilities.screen_data(returns.data)
        mask = numpy.array(ma.getmaskarray(data), copy=True, dtype=numpy.int8)
        zeroMask = zeroMask.view(dtype=numpy.int8)
        tmpreturns = ma.filled(data, numpy.nan)
        data = ma.filled(data, 0.0)
        nanutils.compound_currency_returns(tmpreturns, data, mask, zeroMask)
        returns.data[:] = ma.array(data, mask=mask.view(dtype=bool))
        logging.debug('Done Compounding currency and returns-timing adjustments')
        return returns

    def compound_currency_returns_old(self,  returns, zeroMask):
        """ Compounds currency returns for missing asset returns into non-missing
        returns locations
        """
        logging.info('Compounding currency and returns-timing adjustments')
        data = Utilities.screen_data(returns.data)
        mask = numpy.array(ma.getmaskarray(data), copy=True)
        data = ma.filled(data, 0.0)
        # Loop round each asset in turn
        for (i, sid) in enumerate(returns.assets):
            # Pick out days when asset returns are missing
            maskedRetIdx = numpy.flatnonzero(zeroMask[i,:])
            if len(maskedRetIdx) > 0:
                t0 = maskedRetIdx[0] - 1
                t1 = maskedRetIdx[0]
                for t in maskedRetIdx:
                    dataBlock = ma.filled(returns.data[i,t0+1:t1+1], 0.0)
                    if t-t1 > 1:
                        if not mask[i,t1+1]:
                            cumRet = numpy.prod(dataBlock + 1.0, axis=0)
                            data[i,t1+1] = -1.0 + ((1.0 + data[i,t1+1]) * cumRet)
                        t0 = t-1
                    data[i,t] = 0.0
                    mask[i,t] = False
                    t1 = t
                # Deal with last in list
                if t < returns.data.shape[1]-1:
                    dataBlock = ma.filled(returns.data[i,t0+1:t1+1], 0.0)
                    if not mask[i,t1+1]:
                        cumRet = numpy.cumproduct(dataBlock + 1.0, axis=0)[-1]
                        data[i,t+1] = -1.0 + ((1.0 + data[i,t1+1]) * cumRet)
                        mask[i,t+1] = False
        logging.debug('Done Compounding currency and returns-timing adjustments')
        returns.data[:] = ma.array(data, mask=mask)
        return returns

    def sort_trading_days(self, returns, modelDB, marketDB, date, maskZeroWithNoADV=False):
        """ Function to sift through missing asset returns and
        decide what is missing due to illiquidity, what is due to
        not yet being live, and what is due to non-trading days
        Returns several Boolean arrays denoting a missing return as True
        """
        nT = returns.data.shape[0] * returns.data.shape[1]
        # Optional - treat as missing zero returns with zero or missing trading volume
        # Load trading volumes for all assets and dates
        if modelDB.volumeCache.maxDates < len(returns.dates):
            logging.warning('ModelDB volume cache too small (%d days), resizing to %d days',
                    modelDB.volumeCache.maxDates, int(1.5*len(returns.dates)))
            modelDB.setVolumeCache(int(1.5*len(returns.dates)))
        vol = modelDB.loadVolumeHistory(
                returns.dates, returns.assets, self.numeraire_id, loadAllDates=True).data
        vol = ma.masked_where(vol<=0.0, vol)

        # Look for zero returns with missing volumes
        tmpData = ma.filled(returns.data, 1.0)
        tmpData = ma.masked_where(tmpData==0.0, tmpData)
        missingVol = ma.getmaskarray(vol)
        zeroReturn = ma.getmaskarray(tmpData)
        dubiousFlag = zeroReturn * missingVol
        nDF = ma.sum(dubiousFlag, axis=None)
        nZero = ma.sum(zeroReturn, axis=None)
        if nZero != 0.0:
            logging.info('%d out of %d zero returns (%.1f%%) with no volume data',
                    nDF, nZero, 100.0 * nDF/nZero)
        if maskZeroWithNoADV:
            returns.data = ma.masked_where(dubiousFlag, returns.data)
        zeroReturnFlag = zeroReturn & ~missingVol

        # For comparison purposes, output proportion of non-zero, non-missing
        # returns with no trading volume
        tmpData = ma.masked_where(returns.data==0.0, returns.data)
        nonMissing = ~ma.getmaskarray(tmpData)
        dubiousFlag = nonMissing & missingVol
        nDF = ma.sum(dubiousFlag, axis=None)
        nOk = ma.sum(nonMissing, axis=None)
        logging.info('%d out of %d non-zero returns (%.1f%%) with no volume data',
                nDF, nOk, 100.0 * nDF/nOk)
         
        # Get mask array for absolutely everything that's missing
        allMasked = ma.getmaskarray(returns.data)

        # Get the various data 'masks' - first the roll-over indicator
        # Denote as True where an asset has missing return
        rollOverFlag = ma.array(ma.filled(returns.notTradedFlag, 1.0), bool) & allMasked
        nNonTrade = ma.sum(rollOverFlag, axis=None)
        logging.info('%d out of %d returns (%.1f%%) flagged as missing',
                nNonTrade, nT, 100.0 * nNonTrade/nT)
         
        # Then the pre-IPO flag indicator
        # Marked as True wherever an asset is pre-IPO
        issueFromDates = modelDB.loadIssueFromDates(returns.dates, returns.assets)
        preIPOFlag = Matrices.fillAndMaskByDate(
                returns.data, issueFromDates, returns.dates)
        preIPOFlag = numpy.array(ma.getmaskarray(preIPOFlag), copy=True)

        # Pull out a list of IPOs for the latest date
        issueFromDateMap = dict(zip(returns.assets, issueFromDates))
        newIPOList = [sid for sid in returns.assets if \
                issueFromDateMap[sid]==returns.dates[-1]]
         
        # Thirdly, the non-trading day mask
        # Marked True on all non-trading days for each market
        dateIdxMap = dict([(d,i) for (i,d) in enumerate(returns.dates)])
        ntdFlag = Matrices.allMasked(returns.data.shape)
        for rmg in self.rmg:
            asset_indices = [returns.assetIdxMap[sid] for sid in \
                    self.rmgAssetMap[rmg.rmg_id] if sid in returns.assetIdxMap]
            rmgCalendarSet = set(modelDB.getDateRange(rmg, returns.dates[0], returns.dates[-1]))
            if len(asset_indices) > 0:
                ntd_date_indices = [dateIdxMap[d] for d in returns.dates
                        if d not in rmgCalendarSet and d in dateIdxMap]
                positions = [len(returns.dates) * sIdx + dIdx for \
                        sIdx in asset_indices for dIdx in ntd_date_indices]
                ma.put(ntdFlag, positions, 1)
        ntdFlag = ~ma.getmaskarray(ntdFlag)
        # Remove preIPO days that are also NTDs from NTD flag so
        # we don't double-count
        ntdFlag = ~preIPOFlag & ntdFlag
         
        # Determine days where assets genuinely haven't traded due to illiquidity
        # This is all non-trading days in rollOverFlag that aren't in either
        # of the other two arrays
        rollOverFlag = rollOverFlag & ~preIPOFlag & ~ntdFlag

        # Flag anything else that's masked that isn't accounted for by one
        # of the existing flags
        xFlag = allMasked & ~rollOverFlag & ~preIPOFlag & ~ntdFlag
        nXF = ma.sum(xFlag, axis=None)
        logging.info('%d out of %d returns (%.1f%%) flagged as unaccounted for missing',
                nXF, nT, 100.0 * nXF/nT)
        # Assume they're rolled-over returns that have been missed
        rollOverFlag = rollOverFlag | xFlag
         
        # Finally do some reporting on the stats
        nIlliquid = 100.0 * ma.sum(rollOverFlag, axis=None) / float(nT)
        nPreIPO =  100.0 * ma.sum(preIPOFlag, axis=None) / float(nT)
        nNTD = 100.0 * ma.sum(ntdFlag, axis=None) / float(nT)
        nZero = 100.0 * ma.sum(zeroReturnFlag, axis=None) / float(nT)
        logging.info('%d returns, %.1f%% illiquid, %.1f%% pre IPO, %.1f%% NTDs, %.1f%% zero',
                nT, nIlliquid, nPreIPO, nNTD, nZero)
        return (rollOverFlag, preIPOFlag, ntdFlag, zeroReturnFlag, newIPOList)

    def process_returns_history(self, date, needDays, modelDB, marketDB,
            subIssueGroups=None, drCurrMap=None, backFill=False,
            excludeWeekend=True, loadOnly=False, qadLoad=False,
            applyRT=False, applyProxy=True, backFilledIssues=None,
            trimData=False):
        """Does extensive processing of asset returns to alleviate the problem
        of illiquid data. Does currency conversion and returns-timing
        adjustment to cross-listings to make them "local"
        Then, compounds these returns when there is no asset return on
        a particular day, and scales the non-missing returns accordingly
        Finally, fills in missing returns from non-missing linked partners
        and, again, scales non-missing illiquid returns
        """
        logging.info('Back-filling %d days of %d asset returns',
                needDays, len(self.universe))
        deepDebug = False
        # Get groups of linked assets
        if subIssueGroups == None:
            if backFill:
                self.subIssueGroups = modelDB.getIssueCompanyGroups(
                        date, self.universe, marketDB)
            else:
                self.subIssueGroups = dict()
                self.subIssueGroups[0] = self.universe
        else:
            self.subIssueGroups = subIssueGroups

        # Decide whether to convert returns to home market currency
        if drCurrMap is not None:
            currencyConversion = drCurrMap
        else:
            currencyConversion = self.numeraire_id

        # Get daily returns for the past calendar days(s)
        # We convert everything to USD currency AND time zone for the
        # back filling to ensure consistency, then transform back to trading
        # currency and trading time zone afterwards
        logging.info('Loading history of %d days by %d returns', 
                int(needDays), len(self.universe))
        returns = modelDB.loadTotalReturnsHistoryV3(
                self.rmg, date, self.universe, int(needDays),
                assetConvMap=currencyConversion, excludeWeekend=excludeWeekend,
                allRMGDates=(loadOnly==False))
        if returns.data is None:
            logging.warning('No returns data found for %s', date)
            return None
        returns.data = ma.array(returns.originalData, copy=True)

        self.dump_return_history('rets_s0_raw.csv',
                returns, returns.data, modelDB, marketDB)

        if loadOnly and trimData:
            outlierClass0 = Outliers.Outliers()
            returns.data = outlierClass0.twodMAD(returns.data)
            self.dump_return_history('rets_s0_trimmed.csv',
                    returns, returns.data, modelDB, marketDB)
         
        if self.debuggingReporting:
            # Output low frequency correlations for comparison
            weekDates = Utilities.change_date_frequency(returns.dates)
            if len(weekDates) > 0:
                weeklyReturns = Utilities.compute_compound_returns_v3(
                        returns.data, returns.dates, weekDates)[0]
                self.dump_return_history('weekly returns correlations',
                        returns, weeklyReturns, modelDB, marketDB, outFile=False)
            monthDates = Utilities.change_date_frequency(returns.dates, frequency='monthly')
            if len(monthDates) > 0:
                monthlyReturns = Utilities.compute_compound_returns_v3(
                        returns.data, returns.dates, monthDates)[0]
                self.dump_return_history('monthly returns correlations',
                        returns, monthlyReturns, modelDB, marketDB, outFile=False)
         
        # Get masked returns, corresponding to IPOs, rolled prices and NTDs
        (rollOverFlag, preIPOFlag, ntdFlag, zeroFlag, newIPOList) = self.sort_trading_days(
                returns,  modelDB, marketDB, date, maskZeroWithNoADV=True)
        # Combine roll-over flag with NTD flag - treat all as illiquid henceforth
        rollOverFlag = rollOverFlag | ntdFlag
        missingFlag = rollOverFlag | preIPOFlag
        returns.missingFlag = missingFlag
        returns.preIPOFlag = preIPOFlag
        returns.rollOverFlag = rollOverFlag
        returns.ntdFlag = ntdFlag

        # Get returns timing adjustments for conversion to USA time zone
        if self.returnsTimingID is not None:
            returns.data = ma.filled(returns.data, 0.0)
            logging.info('Loading returns timing adjustments')
            rmgAll = modelDB.getAllRiskModelGroups()
            adjustments = modelDB.loadReturnsTimingAdjustmentsHistory(
                    self.returnsTimingID, rmgAll, returns.dates)
            adjustments.data = ma.filled(adjustments.data, 0.0)
            returnsTC = self.adjustReturnsForTiming(
                    date, returns, modelDB, marketDB, adjustments, rmgAll)
            adjustArray = returnsTC - returns.data
            returns.data = ma.masked_where(missingFlag, returns.data)
        else:
            adjustArray = numpy.zeros((returns.data.shape), float)

        if backFill:
            # Assign asset scores between linked assets
            scores = self.score_linked_assets(
                    date, modelDB, marketDB, returns, rollOverFlag)

        # Combine existing proxies into original returns
        (returns, proxyReturns, missingDataMask, missingProxies, noMissingFlag) =\
                self.load_proxy_returns(
                        modelDB, returns, preIPOFlag=returns.preIPOFlag, qadFlag=qadLoad,
                        applyProxy=applyProxy, backFilledIssues=backFilledIssues)
        rollOverFlag = rollOverFlag & missingDataMask
        preIPOFlag = preIPOFlag & missingDataMask
        returnsAfterProxy = ma.array(returns.data, copy=True)

        # Add returns timing adjustments
        returnsTZ = returns.data + adjustArray
        # Convert to USD (or whatever currencies are required)
        if loadOnly and not applyRT:
            returnsUSD = (ma.filled(returns.currencyMatrix, 1.0) * (1.0 + ma.filled(returns.data, 0.0))) - 1.0
        else:
            returnsUSD = (ma.filled(returns.currencyMatrix, 1.0) * (1.0 + ma.filled(returnsTZ, 0.0))) - 1.0

        if deepDebug and self.debuggingReporting:
            self.dump_return_history('rets_s1_proxy.csv', returns, returns.data, modelDB, marketDB)
            self.dump_return_history('rets_s2_rtim.csv', returns, returnsTZ, modelDB, marketDB)
            self.dump_return_history('rets_s3_cur.csv', returns, returnsUSD, modelDB, marketDB)

        # Compound currency and returns timing entries for illiquid assets
        returns.data = ma.masked_where(preIPOFlag, returnsUSD)
        if oldCode:
            returns = self.compound_currency_returns_old(returns, rollOverFlag)
        else:
            returns = self.compound_currency_returns(returns, rollOverFlag)
        nMasked0 = len(numpy.flatnonzero(preIPOFlag))
        nMasked1 = len(numpy.flatnonzero(rollOverFlag))
        if deepDebug and self.debuggingReporting:
            self.dump_return_history('rets_s4_curc.csv', returns, returns.data, modelDB, marketDB)

        # Mask missing returns once more
        returns.data = ma.masked_where(rollOverFlag, returns.data)
        if loadOnly:
            # Convert returns back to trading currency
            tmpReturns = ma.array(returns.data, copy=True)
            returns.data = ((1.0 + ma.filled(returns.data, 0.0)) / \
                    ma.filled(returns.currencyMatrix, 1.0)) - 1.0
            if oldCode:
                returns = self.compound_currency_returns_old(returns, rollOverFlag)
            else:
                returns = self.compound_currency_returns(returns, rollOverFlag)
            # Make sure any still-missing values are masked
            returns.tcData = ma.masked_where(rollOverFlag, returns.data)
            returns.tcData = ma.masked_where(preIPOFlag, returns.tcData)
            returns.data = ma.masked_where(rollOverFlag, tmpReturns)
            returns.data = ma.masked_where(preIPOFlag, returns.data)
            return returns

        # Back fill missing returns between linked assets
        missingDataMask = numpy.array(ma.getmaskarray(returns.data), copy=True)
        if noMissingFlag:
            logging.info('No missing proxies, skipping new proxy generation')
        elif backFill:
            # Back fill linked assets directly
            returns = self.back_fill_linked_assets(
                    returns, scores, modelDB, marketDB)
        else:
            # Alternatively, estimate proxies via PCA
            # Compute cross-sectional proxy fill in for initial estimate
            initialReturns = Matrices.allMasked(returns.data.shape)
            tmpData = Utilities.Struct()
            tmpData.universe = returns.assets
            tmpData.rmgAssetMap = self.rmgAssetMap
            tmpData.assetIdxMap = returns.assetIdxMap
            tmpReturns = Matrices.TimeSeriesMatrix(returns.assets, returns.dates)
            tmpReturns.data = ma.array(returns.data, copy=True)
            marketCaps = modelDB.loadMarketCapsHistory(
                    returns.dates[-20:], returns.assets, self.numeraire_id)
            tmpData.marketCaps = ma.filled(ma.median(marketCaps, axis=1), 0.0)
            outlierClass1 = Outliers.Outliers()
            tmpReturns.data, dummy1 = Utilities.proxyMissingAssetData(
                    self.rmg, date, tmpReturns.data, tmpData, modelDB,
                    outlierClass1, countryFactor=True, industryGroupFactor=False,
                    debugging=self.debuggingReporting,
                    gicsDate=datetime.date(2014,3,1),
                    minGoodAssets=0.01, pctGoodReturns=0.95)
            tmpReturns.data = ma.clip(tmpReturns.data, -0.75, 2.5)
            
            # Extra safety-net - fill in with beta*mkt return if anything has been missed
            marketReturnHistory = modelDB.loadRMGMarketReturnHistory(returns.dates, self.rmg, useAMPs=False)
            hBetaDict = modelDB.getHistoricBetaFixed(date, returns.assets)
            tmpReturns.data = ma.masked_where(tmpReturns.data==0.0, tmpReturns.data)

            # Fill in initial guess
            for (i_rmg, rmg) in enumerate(self.rmg):
                for sid in self.rmgAssetMap[rmg.rmg_id]:
                    idx = returns.assetIdxMap[sid]
                    missingIdx = set(numpy.flatnonzero(missingDataMask[idx,:]))
                    missingIG = set(numpy.flatnonzero(ma.getmaskarray(tmpReturns.data[idx,:])))
                    okProxyIdx = list(missingIdx.difference(missingIG))
                    noProxyIdx = list(missingIdx.intersection(missingIG))
                    hbeta = 1.0
                    if sid in hBetaDict:
                        hbeta = ma.filled(hBetaDict[sid], 1.0)
                    for jdx in okProxyIdx:
                        initialReturns[idx,jdx] = tmpReturns.data[idx, jdx]
                    for jdx in noProxyIdx:
                        initialReturns[idx,jdx] = hbeta * marketReturnHistory.data[i_rmg, jdx]

            # And estimate proxies
            returns = self.computeProxyReturns(returns, initialReturns, tmpData, modelDB)
         
        # Redo missing data flags
        if deepDebug and self.debuggingReporting:
            self.dump_return_history('rets_s5_bfil.csv', returns, returns.data, modelDB, marketDB)
        allMasked = numpy.array(ma.getmaskarray(returns.data), copy=True)
        rollOverFlag = rollOverFlag & allMasked
        preIPOFlag = preIPOFlag & allMasked
        filledDataFlag = missingDataMask & ~allMasked

        # Final reporting
        propMasked1 = 100.0 * float(nMasked0) / numpy.size(returns.data)
        propMasked2 = 100.0 * float(nMasked1) / numpy.size(returns.data)
        logging.info('BEFORE: (%4.1f%%/%4.1f%%) of returns missing due to IPOs/illiquidity',
                propMasked1, propMasked2)
        nMasked = len(numpy.flatnonzero(preIPOFlag))
        propMasked1 = 100.0 * float(nMasked) / numpy.size(returns.data)
        nMasked = len(numpy.flatnonzero(rollOverFlag))
        propMasked2 = 100.0 * float(nMasked) / numpy.size(returns.data)
        logging.info('AFTER: (%4.1f%%/%4.1f%%) of returns missing due to IPOs/illiquidity',
                propMasked1, propMasked2)

        # Convert returns back to trading currency for consistency
        returns.data = ((1.0 + ma.filled(returns.data, 0.0)) / ma.filled(returns.currencyMatrix, 1.0)) - 1.0
        returns.data = ma.masked_where(preIPOFlag, returns.data)
        if oldCode:
            returns = self.compound_currency_returns_old(returns, rollOverFlag)
        else:
            returns = self.compound_currency_returns(returns, rollOverFlag)

        # Convert returns back to trading country time zone
        returns.data = returns.data - adjustArray
        returns.data = ma.masked_where(allMasked, returns.data)

        # Pick out differences and return
        noProxyIdx = ma.getmaskarray(ma.masked_where(~filledDataFlag, returns.data))
        noChangeIdx = ma.getmaskarray(ma.masked_where(abs(ma.filled(
            returnsAfterProxy, 0.0) - ma.filled(returns.data,0.0)) < 1.0e-6, returns.data))
        noChangeIdx = noProxyIdx | noChangeIdx
        newProxies = ma.masked_where(noChangeIdx, returns.data)
        #newProxies = ma.masked_where(abs(newProxies) < 1.0e-12, newProxies)
        self.dump_return_history('rets_s6_final.csv', returns, returns.data, modelDB, marketDB)

        nProxies = len(numpy.flatnonzero(ma.getmaskarray(newProxies)==0))
        if nProxies < 1:
            return returns

        # If only updating one or more subissue, mask everything else
        idxList = None
        if backFill:
            qadLoad = False
        if qadLoad:
            idxList = [returns.assetIdxMap[sid] for sid in newIPOList]
        if len(self.saveIDs) > 0:
            emptyProxies = Matrices.allMasked(newProxies.shape)
            saveIDs = [sid for sid in self.saveIDs if sid in returns.assetIdxMap]
            if qadLoad:
                idxList = [returns.assetIdxMap[sid] for sid in newIPOList if sid in saveIDs]
            else:
                idxList = [returns.assetIdxMap[sid] for sid in saveIDs]
            for idx in idxList:
                emptyProxies[idx,:] = newProxies[idx,:]
            newProxies = emptyProxies

        # Write new proxy returns to DB
        self.save_proxy_returns(modelDB, marketDB, returns.dates, returns,
                newProxies, idxList=idxList)

        return returns

    def computeProxyReturns(self, returns, initialReturns, data, modelDB):
        """Function to compute proxy returns for a market
        via an APCA model. These proxies are used to fill in
        missing returns, which are then fed back into the APCA
        model until convergence is achieved
        """
        logging.info('Computing proxy returns for missing values')
        # Set up initial parameters
        nFactors = min(20, int(len(self.estu)/5))
        nFactors = max(nFactors, 1)
        pcaInstance = LegacyFactorReturns.AsymptoticPrincipalComponentsLegacy(
                nFactors, flexible=True)
        returnsCopy = Matrices.TimeSeriesMatrix(returns.assets, returns.dates)
        returnsCopy.data = Utilities.screen_data(returns.data)
        maskedData = numpy.array(ma.getmaskarray(returnsCopy.data), copy=True)
        returnsCopy.data = ma.filled(returnsCopy.data, 0.0)

        # Fill in missing returns with initial estimate
        returnsCopy.data = Utilities.fill_and_smooth_returns(
                returnsCopy.data, initialReturns, maskedData,
                preIPOFlag=returns.preIPOFlag)[0]

        # Trim outliers along region/sector buckets
        opms = dict()
        opms['nBounds'] = [3.0, 3.0]
        outlierClass = Outliers.Outliers(opms, industryClassificationDate=datetime.date(2014,3,1))
        returnsCopy.data = outlierClass.bucketedMAD(
                self.rmg, returns.dates[-1], returnsCopy.data, data, modelDB, axis=0)

        # Build Statistical model of returns
        if len(self.estu) < 1:
            logging.warning('Empty estimation universe for %s', self.rmg)
            return returns
        (X, F, Delta, ANOVA, pctgVar) = pcaInstance.calc_ExposuresAndReturns(
                returnsCopy, estu=self.estu)
        proxyReturns = ma.clip(numpy.dot(X, F), -0.75, 1.5)
        
        # Fill in missing returns with proxy values
        returnsCopy.data = Utilities.fill_and_smooth_returns(
                returns.data, proxyReturns, maskedData,
                preIPOFlag=returns.preIPOFlag)[0]
             
        returns.data = numpy.array(returnsCopy.data, copy=True)
        return returns

    def adjustReturnsForTiming(self, modelDate, returns,
            modelDB, marketDB, adjustments, rmgList):
        """If DR-like instruments are present in the universe, adjust their
        returns for returns-timing, aligning them with the US market
        Both returns and adjustments should be TimeSeriesMatrix objects containing the
        asset returns and market adjustment factors time-series,
        respectively.  The adjustments array cannot contain masked values.
        Returns the adjusted.
        """
        logging.debug('adjustReturnsForTiming: begin')
        subIssues = returns.assets
        data = ma.array(returns.data, copy=True)
         
        # Determine DR's trading country (country of quotation)
        tradingCountryMap = dict([(sid, rmg.rmg_id) for (sid, rmg) in \
                modelDB.getSubIssueRiskModelGroupPairs(
                    modelDate, restrict=list(subIssues))])
        rmgIdxMap = dict([(j.rmg_id,i) for (i,j) in \
                enumerate(adjustments.assets)])

        # Set up market time zones
        rmgZoneMap = dict((rmg.rmg_id, rmg.gmt_offset) for rmg in rmgList)

        # Add adjustment
        dateLen = len(returns.dates)
        for (idx, sid) in enumerate(subIssues):
            tradingIdx = tradingCountryMap.get(sid, None)
            if tradingIdx is not None and tradingIdx in rmgIdxMap:
                rmgIdx = rmgIdxMap[tradingIdx]
                data[idx,:] += adjustments.data[rmgIdx,:dateLen]
        logging.debug('adjustReturnsForTiming: end')
        return data

    def load_proxy_returns(self, modelDB, assetReturns, skipLatest=False,
            simple=False, preIPOFlag=None, qadFlag=False, applyProxy=True, backFilledIssues=None):
        """ Loads in the proxy returns and fills in missing returns
        in the input returns data
        """
        logging.info('Loading and merging existing proxies')
        assetIdxMap = assetReturns.assetIdxMap
        # Some date manipulation
        rowNames = [sid.getSubIDString() for sid in assetReturns.assets]
        allDates = modelDB.getDateRange(None, assetReturns.dates[0], assetReturns.dates[-1])
         
        # Load history of previous proxy returns
        if skipLatest:
            allDates = allDates[:-1]

        if applyProxy:
            proxyReturns = modelDB.loadProxyReturnsHistory(
                    assetReturns.assets, allDates).data
        else:
            proxyReturns = Matrices.allMasked((len(assetReturns.assets), len(allDates)))

        if qadFlag:
            # If doing proxy-lite, ignore past history
            tmpProxyRet = ma.array(proxyReturns, copy=True)
            proxyReturns = Matrices.allMasked((len(assetReturns.assets), len(allDates)))
            proxyReturns[:,-1] = tmpProxyRet[:,-1]
            if backFilledIssues is not None:
                for sid in backFilledIssues:
                    if sid in assetIdxMap:
                        proxyReturns[assetIdxMap[sid],:] = tmpProxyRet[assetIdxMap[sid],:]

        # Line up dates
        proxyReturns = Utilities.screen_data(proxyReturns)
        proxyReturns = Utilities.compute_compound_returns_v3(
                proxyReturns, allDates, assetReturns.dates,
                keepFirst=True, matchDates=True)[0]

        missingDataMask = numpy.array(ma.getmaskarray(assetReturns.data), copy=True)
        missingProxies = numpy.array(ma.getmaskarray(proxyReturns), copy=True)
        missingAnyReturn = missingProxies & missingDataMask
        
        # Use previously computed proxies to fill in missing returns
        nMiss = numpy.size(numpy.flatnonzero(missingDataMask))
        nRets = numpy.size(assetReturns.data)
        logging.info('Missing %d returns (%.2f%%) before proxy filling',
                nMiss, 100.0*float(nMiss)/nRets)
        nonMissingVals = numpy.flatnonzero(ma.getmaskarray(proxyReturns)==0)
        if len(nonMissingVals) > 0:
            if simple:
                # If required, simple pasting of proxy data into returns
                assetReturns.data = ma.masked_where(~missingProxies, assetReturns.data)
                assetReturns.data = ma.filled(assetReturns.data, 0.0)
                assetReturns.data += ma.filled(proxyReturns, 0.0)
            else:
                # Otherwise, more complex filling and scaling
                (assetReturns.data, dataMask) = Utilities.fill_and_smooth_returns(
                        assetReturns.data, proxyReturns, preIPOFlag=preIPOFlag)
        assetReturns.data = ma.masked_where(missingAnyReturn, assetReturns.data)
        missingDataMask = numpy.array(ma.getmaskarray(assetReturns.data), copy=True)
        nMiss = numpy.size(numpy.flatnonzero(missingDataMask))
        logging.info('Missing %d returns (%.2f%%) after proxy filling',
                nMiss, 100.0*float(nMiss)/nRets)
        if nMiss == 0:
            noMissingFlag = True
        else:
            noMissingFlag = False
        
        return (assetReturns, proxyReturns, missingDataMask, missingAnyReturn, noMissingFlag)

    def save_proxy_returns(self, modelDB, marketDB, dates, returns, proxyReturns, idxList=None):
        logging.info('Saving new proxy returns to DB')
        finalProxies = Matrices.allMasked((proxyReturns.shape))
        proxyMask = ma.getmaskarray(proxyReturns)
        for (j,d) in enumerate(dates):
            # Pick out latest proxies
            proxyCol = ma.filled(proxyReturns[:,j], 0.0)
            proxyIdx = numpy.flatnonzero(proxyMask[:,j]==0)
            if (idxList is not None) and (d != dates[-1]):
                proxyIdx = [idx for idx in proxyIdx if idx in idxList]
            proxyRets = numpy.take(proxyCol, proxyIdx, axis=0)
            subIssues = numpy.take(numpy.array(
                self.universe, dtype=object), proxyIdx, axis=0)
            if len(subIssues) > 0:
                for idx in proxyIdx:
                    finalProxies[idx, j] = proxyCol[idx]
                if not self.dontWrite:
                    logging.debug('Saving %d proxy returns for %s', len(subIssues), d)
                    modelDB.deleteProxyReturns(subIssues, d)
                    modelDB.insertProxyReturns(d, subIssues, proxyRets)
        self.dump_return_history('proxies_final.csv', returns, finalProxies, modelDB, marketDB)
        return

    def score_linked_assets(self, date, modelDB, marketDB, returns, rollOverFlag):
        """Assigns scores to linked assets based on their cap and
        liquidity, in order to assist cloning or other adjustment
        of exposures, specific risks, correlations etc.
        """
        logging.debug('Start score_linked_assets')
        daysBack = returns.data.shape[1]
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(returns.assets)])
        if self.subIssueGroups is None:
            self.subIssueGroups = modelDB.getIssueCompanyGroups(
                    date, returns.assets, marketDB)
        scoreDict = dict()

        # Determine how many calendar days of data we need
        volDates = modelDB.getDates(self.rmg, date, daysBack, excludeWeekend=True, fitNum=True)
        mcapDates = modelDB.getDates(self.rmg, date, daysBack, excludeWeekend=True, fitNum=True)
        if modelDB.volumeCache.maxDates < len(volDates):
            logging.warning('ModelDB volume cache too small (%d days), resizing to %d days',
                    modelDB.volumeCache.maxDates, int(1.5*len(volDates)))
            modelDB.setVolumeCache(int(1.5*len(volDates)))
        if modelDB.marketCapCache.maxDates < len(mcapDates):
            logging.warning('ModelDB market cap cache too small (%d days), resizing to %d days',
                    modelDB.marketCapCache.maxDates, int(1.5*len(mcapDates)))
            modelDB.setMarketCapCache(len(mcapDates))

        # Compute average trading volume of each asset
        vol = modelDB.loadVolumeHistory(volDates, returns.assets, self.numeraire_id, loadAllDates=True)
        volume = ma.filled(ma.median(vol.data, axis=1), 0.0)

        # Load in market caps
        marketCap = modelDB.loadMarketCapsHistory(mcapDates, returns.assets, self.numeraire_id, loadAllDates=True)
        avgMarketCap = ma.filled(ma.median(marketCap, axis=1), 0.0)

        # Get proportion of non-missing returns per asset
        propNonMissing = ma.filled(ma.average(~ma.getmaskarray(returns.data), axis=1), 0.0)

        # Convert returns to single currency
        returnsCopy = ma.array(returns.data, copy=True)
        returns.data = (ma.filled(returns.currencyMatrix, 1.0) * \
                (1.0 + ma.filled(returns.data, 0.0))) - 1.0
        if oldCode:
            returns = self.compound_currency_returns_old(returns, rollOverFlag)
        else:
            returns = self.compound_currency_returns(returns, rollOverFlag)
        returns.data = ma.masked_where(returns.preIPOFlag, returns.data)

        sidList = []
        scoreList = []
        # Loop round sets of linked assets and pull out exposures
        for (groupId, subIssueList) in self.subIssueGroups.items():
             
            indices  = [assetIdxMap[n] for n in subIssueList]
            sidList.extend(subIssueList)
             
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
            if max(nMissSubSet) > 0.0:
                nMissSubSet /= float(max(nMissSubSet))
            else:
                nMissSubSet = numpy.ones((len(indices)), float)
            # Now combine the cap, vol and trading day scores
            score = volumeSubSet * mcapSubSet * nMissSubSet
            score *= score
            scoreList.extend(score)
            scoreDict[groupId] = dict(zip(subIssueList, score))

        writeScores = False
        if writeScores:
            # Get cointegration results
            cointResults = Utilities.compute_cointegration_parameters(
                    returns.data, self.subIssueGroups, returns.assets, self.rmgAssetMap,
                    skipDifferentMarkets=False)

            # Output ISC scores
            outfile = 'tmp/scores-%s.csv' % date
            outfile = open(outfile, 'w')
            outfile.write('GID,SID1,Type,Score,SID2,Type,Score,\n')
             
            for (groupId, subIssueList) in self.subIssueGroups.items():
                scores = scoreDict[groupId]
                for (ii, sid1) in enumerate(subIssueList[:-1]):
                    idx1 = assetIdxMap[sid1]
                    type1 = self.assetTypeDict.get(sid1, None)
                    score1 = scores[sid1]
                    for sid2 in subIssueList[ii+1:]:
                        idx2 = assetIdxMap[sid2]
                        type2 = self.assetTypeDict.get(sid2, None)
                        score2 = scores[sid2]

                        outfile.write('%s,%s,%s,%s,%s,%s,%s,\n' % (groupId, \
                                sid1.getSubIDString(), type1, score1,
                                sid2.getSubIDString(), type2, score2))
            outfile.close()

            # Output cointegration results
            outfile = 'tmp/coint-%s.csv' % date
            outfile = open(outfile, 'w')
            outfile.write('GID,SID1,Type1,Mkt1,Parent1,SID2,Type2,Mkt2,Parent2,DF-Stat,C-Value,N,\n')
            ktDict = dict()

            for (groupId, subIssueList) in self.subIssueGroups.items():
                for (idx1, sid1) in enumerate(subIssueList):
                    type1 = self.assetTypeDict.get(sid1, None)
                    mkt1 = self.marketTypeDict.get(sid1, None)
                    for (idx2, sid2) in enumerate(subIssueList):
                        type2 = self.assetTypeDict.get(sid2, None)
                        mkt2 = self.marketTypeDict.get(sid2, None)
                        pnt1 = 0
                        pnt2 = 0
                        if (sid1 in self.dr2UnderDict) and self.dr2UnderDict[sid1] == sid2:
                            pnt2 = 1
                        if (sid2 in self.dr2UnderDict) and self.dr2UnderDict[sid2] == sid1:
                            pnt1 = 1
                        if sid1 == sid2:
                            continue
                        cKey1 = '%s-%s' % (sid1.getSubIDString(), sid2.getSubIDString())
                        cKey2 = '%s-%s' % (sid2.getSubIDString(), sid1.getSubIDString())
                        if (cKey1 in ktDict) or (cKey2 in ktDict):
                            continue
                        try:
                            adfStat = cointResults.dfStatDict[sid1][sid2]
                            pvalue = cointResults.dfCValueDict[sid1][sid2]
                            nobs = cointResults.nobsDict[sid1][sid2]
                            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%d,\n' % (groupId, \
                                    sid1.getSubIDString(), type1, mkt1, pnt1,
                                    sid2.getSubIDString(), type2, mkt2, pnt2,
                                    adfStat, pvalue, nobs))
                            ktDict[cKey1] = True
                            ktDict[cKey2] = True
                        except KeyError:
                            continue
            outfile.close()

        # Write info to the DB
        scores = numpy.array(scoreList)
        subIssues = numpy.array(sidList, dtype=object)
        if writeScores and len(subIssues) > 0:
            if not self.dontWrite:
                modelDB.deleteISCScores(subIssues, date)
                modelDB.insertISCScores(date, subIssues, scores)

        logging.debug('End score_linked_assets')
        returns.data = ma.array(returnsCopy, copy=True)
        return scoreDict

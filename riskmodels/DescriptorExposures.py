import datetime
import logging
import numpy.ma as ma
import numpy
import collections
import pandas as pd
import scipy.stats as sm
from collections import defaultdict
from marketdb import MarketDB
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels.DescriptorRatios import DataFrequency
from riskmodels import Outliers
from riskmodels import CurrencyRisk
from riskmodels import ProcessReturns
from riskmodels import AssetProcessor_V4

def generate_pace_volatility(
        returns, assets, daysBack, csvHistory=None, indices=None, weights=None, trackList=[]):
    """The cross-sectional volatility is the square-root of the average
    of the weighted absolute asset returns for each asset over the
    last 60 (default) days. The asset returns of each days are
    weighted by the standard deviation of all returns on that day.
    If indices is specified, the std dev is taken across that
    subset of assets only.
    """
    logging.debug('generate_pace_volatility: begin')
    # Set up parameters
    overlap = sorted(set(trackList).intersection(set(assets)))
    returns = ma.filled(returns.data[:, -daysBack:], 0.0)

    if csvHistory is None:
        logging.warning('No CSV history, generating on the fly')
        outlierClass = Outliers.Outliers()

        # Restrict universe for std dev
        if indices is None:
            indices = list(range(returns.shape[0]))
        goodReturns = ma.take(returns, indices, axis=0)

        # Remove assets with lots of missing/zero returns
        bad = ma.masked_where(goodReturns==0.0, goodReturns)
        io = numpy.sum(ma.getmaskarray(bad), axis=1)
        goodAssetsIdx = numpy.flatnonzero(io < 0.5 * daysBack)
        logging.info('%d out of %d assets have sufficient returns', len(goodAssetsIdx), goodReturns.shape[0])
        if len(goodAssetsIdx) > 0:
            goodReturns = ma.take(goodReturns, goodAssetsIdx, axis=0)
        goodReturns = outlierClass.twodMAD(goodReturns)

        # Compute the cross-sectional volatility
        crossSectionalStdDev = Utilities.mlab_std(goodReturns, axis=0)
        crossSectionalStdDev = ma.masked_where(crossSectionalStdDev==0.0, crossSectionalStdDev)

        # If tiny markets that trade on irregular days are involved,
        # replace std dev on weekends and other days with sparse trading
        # with previous value
        bad = ma.masked_where(goodReturns==0.0, goodReturns)
        io = numpy.sum(ma.getmaskarray(bad), axis=0)
        badDatesIdx = numpy.flatnonzero(io > 0.9 * goodReturns.shape[0])
        logging.info('%d out of %d bad dates (fewer than 90%% assets trading)', len(badDatesIdx), goodReturns.shape[1])
        if len(badDatesIdx) < goodReturns.shape[1]:
            for i in badDatesIdx:
                if i==0:
                    prevStdDev = ma.average(ma.take(crossSectionalStdDev,
                        [j for j in range(goodReturns.shape[1]) if j not in badDatesIdx]))
                else:
                    prevStdDev = crossSectionalStdDev[i-1]
                if prevStdDev is not ma.masked:
                    crossSectionalStdDev[i] = prevStdDev
    else:
        if len(csvHistory) > daysBack:
            crossSectionalStdDev = csvHistory[-daysBack:]
        else:
            crossSectionalStdDev = csvHistory
        prevStdDev = crossSectionalStdDev[0]
        for idx in range(len(crossSectionalStdDev)):
            if crossSectionalStdDev[idx] is ma.masked:
                crossSectionalStdDev[idx] = prevStdDev
            else:
                prevStdDev = crossSectionalStdDev[idx]

    # Combine the various parts
    crossSectionalStdDev = Utilities.screen_data(crossSectionalStdDev)
    crossSectionalStdDev = ma.masked_where(crossSectionalStdDev<0.0001, crossSectionalStdDev)
    crossSectionalStdDev = ma.filled(crossSectionalStdDev, ma.average(crossSectionalStdDev, axis=None))
    csv = ma.average(abs(returns) / crossSectionalStdDev, axis=1, weights=weights)
    csv = ma.sqrt(csv)
    if len(overlap) > 0:
        numer = pd.Series(ma.average(abs(returns), axis=1, weights=weights), index=assets)
        denom = ma.average(crossSectionalStdDev, axis=None, weights=weights)
        for sid in overlap:
            pvol = pd.Series(csv, index=assets)
            logging.info('Tracking %s, numer: %s, denom: %s, PACE Vol: %s', sid.getSubIDString(), numer[sid], denom, pvol[sid])

    logging.debug('generate_pace_volatility: end')
    return csv

def generate_historic_volatility(returns, daysBack):
    """The historic volatility is the square-root of the average
    of the squared asset returns for each asset over the
    last given number of days.
    """
    logging.info('generate_historic_volatility: begin')
    assert(daysBack > 0)
    assert(returns.data.shape[1] >= daysBack)
    myReturns = ma.filled(returns.data[:, -daysBack:], 0.0)
    myReturns = myReturns * myReturns
    hv = numpy.sqrt(numpy.average(myReturns, axis=1))
    logging.info('generate_historic_volatility: end')
    return hv

def generate_returns_skewness(returns, daysBack):
    """Generates returns skewness exposures. Corrects for sample bias
    """
    logging.info('generate_returns_skewness: begin')
    assert(daysBack > 0)
    assert(returns.data.shape[1] >= daysBack)
    myReturns = ma.filled(returns.data[:, -daysBack:], 0.0)
    return sm.skew(myReturns, axis=1)

def generate_momentum(returns, fromT, thruT=0, weights=None,
            peak=0, peak2=None, useLogRets=False):
    """Medium-term momentum is the return over the previous trading days
    from fromT (inclusive) to thruT (exclusive).
    """
    logging.info('generate_momentum: begin')
    assert(fromT > 0 and thruT >= 0 and thruT < fromT)
    if thruT == 0:
        returnsHistory = ma.filled(returns.data[:,-fromT:], 0.0)
    else:
        returnsHistory = ma.filled(returns.data[:,-fromT:-thruT], 0.0)

    if returnsHistory.shape[1] < 1:
        logging.info('Insufficient returns')
        return Matrices.allMasked(returnsHistory.shape[0])
    returnsHistory = Utilities.symmetric_clip(returnsHistory)

    # Get weights if any
    if weights == 'exponential':
        weights = Utilities.computeExponentialWeights(peak, returnsHistory.shape[1])
        weights.reverse
    elif weights == 'triangle':
        weights = Utilities.computeTriangleWeights(peak, returnsHistory.shape[1])
    elif weights == 'pyramid':
        weights = Utilities.computePyramidWeights(peak, peak2, returnsHistory.shape[1])

    if useLogRets:
        returnsHistory = numpy.log(returnsHistory + 1.0)
        if weights is not None:
            returnsHistory = returnsHistory * weights
    else:
        returnsHistory = returnsHistory + 1.0
        if weights is not None:
            returnsHistory = returnsHistory ** weights

    # Get cumulative return
    if useLogRets:
        momentum = numpy.sum(returnsHistory, axis=1)
    else:
        momentum = numpy.cumproduct(returnsHistory, axis=1)[:,-1]
        momentum = momentum - 1.0
         
    logging.info('generate_momentum: end')
    return momentum

def generate_size_exposures(data):
    """Generate the size exposures.
    They are the log() of the average issuer market cap.
    Size exposures always take into account the total market cap
    and not the free-float adjusted market cap
    """
    return ma.log(data.issuerTotalMarketCaps)

def generate_size_nl_exposures(data):
    """Generate the non-linear size exposures.
    They are the cube of the log() of the average issuer market cap.
    Size exposures always take into account the total market cap
    and not the free-float adjusted market cap
    """
    totalMCap = pd.Series(data.issuerTotalMarketCaps, index=data.universe)
    val = ma.log(totalMCap)
    # Quick and dirty standardisation
    val_estu = val[data.estimationUniverse]
    # Take out the mean
    mcap_estu = totalMCap[data.estimationUniverse]
    mean = ma.average(val_estu.values, weights=mcap_estu.values, axis=0)
    val = val - mean
    # Pseudo-standard deviation
    val_estu = val[data.estimationUniverse]
    stdev = ma.sqrt(ma.inner(val_estu.values, val_estu.values) / (len(val_estu) - 1.0))
    val = val / stdev
    val = val * val * val
    return val

def generate_trading_volume_exposures(
        modelDate, data, modelDB, marketDB, params, currencyID=None):
    """Returns the trading volume exposures for the assets in
    data.universe.
    The exposures are computed as the ratio of the average daily
    trading volume over the last daysBack trading days and the
    market capitalization as given by data.markCaps.
    """
 
    # Set up parameters
    simple = params.simple
    lnComb = params.lnComb
    median = params.median

    # Set up dates
    dateList =  modelDB.getAllRMGDateRange(modelDate, params.daysBack)
    allDates = modelDB.getDateRange(None, dateList[0], dateList[-1])

    # Load trading volumes
    vol = modelDB.loadVolumeHistory(allDates, data.universe, currencyID)
    sidList = [sid.getSubIdString() for sid in data.universe]
    (vol.data, vol.dates) = ProcessReturns.compute_compound_returns_v3(
            vol.data, allDates, dateList, keepFirst=True, matchDates=True, sumVals=True)

    # Compute ADV for new listings differently
    issueFromDates = Utilities.load_ipo_dates(\
            modelDate, data.universe, modelDB, marketDB, returnList=True, exSpacAdjust=True)
    vol.data = Matrices.fillAndMaskByDate(vol.data, issueFromDates, vol.dates)

    # Get weights
    if params.weights is None:
        weights = numpy.ones((vol.data.shape[1]), float)
    else:
        weights = Utilities.computePyramidWeights(params.peak, params.peak2, vol.data.shape[1])

    # Compute average dollar volume
    if median:
        volume = ma.median(vol.data, axis=1)
    else:
        volume = ma.average(vol.data, axis=1, weights=weights)
    newListings = [d for d in issueFromDates if d > vol.dates[0]]
    logging.info('Compute different ADV for %d new listings', len(newListings))

    # Mask out volumes or mcaps <= 0.0
    volume = ma.masked_where(volume <= 0.0, volume)
    missing = numpy.flatnonzero(ma.getmaskarray(volume))
    if len(missing) > 0:
        logging.info('%d assets are missing trading volume and/or mcap information', len(missing))
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([data.universe[i].getSubIDString() for i in missing]))
    mcap = ma.masked_where(data.marketCaps <= 0.0, data.marketCaps)
    volume = ma.masked_where(data.marketCaps <= 0.0, volume)

    # Scale by cap if required
    if simple:
        tra = volume
    elif lnComb:
        volume = ma.filled(volume, 0.0) + 1.0
        tra = ma.log(volume) - ma.log(mcap)
    else:
        tra = volume / mcap
 
    return ma.filled(tra, 0.0)

def generate_relative_strength(modelDate, data, returns, modelSelector, 
                                modelDB, smooth=10):
    logging.info('generate_relative_strength: begin')
    halfLife = -numpy.log(2.0) / numpy.log(1.0 - 2.0 / (smooth+1.0))
    wgt = Utilities.computeExponentialWeights(halfLife, returns.data.shape[1])
    rsi = Matrices.allMasked(len(returns.assets))
    for n in range(len(returns.assets)):
        ret = ma.filled(returns.data[n,:], 0.0)
        u = ma.where(ret > 0, ret, 0.0) * wgt
        d = ma.where(ret < 0, ret, 0.0) * wgt
        if numpy.sum(d, axis=0) == 0:
            rsi[n] = 1.0
        else:
            rs = numpy.cumproduct(u+1.0, axis=0)[-1] \
                    / numpy.cumproduct(d+1.0, axis=0)[-1]
            rsi[n] = 1.0 - 1.0 / (1.0 + rs)
    logging.info('generate_relative_strength: end')
    return rsi

def generate_variance_ratios(returns, daysBack=250, periodSize=10, 
                             finiteSampleAdj=True, takeDifference=False,
                             keepSigStd=0.0):
    """Computes variance ratios using daily returns and period
    returns of the specified periodSize.
    Roughly speaking, this statistic measures the extent to
    which an asset's returns depart from white noise, and
    the magnitude of serial dependence.  
    See Campbell, Lo, MacKindlay (1999) for details.
    """
    logging.info('generate_variance_ratios: begin')
    daysBack = min(returns.data.shape[1], daysBack)
    assetReturns = ma.filled(returns.data[:,-daysBack:], 0.0)

    # Compute (overlapping) cumulative period returns
    cumulativeReturns = numpy.cumproduct(assetReturns + 1.0, axis=1)
    sampleSize = daysBack - periodSize
    cumulativePeriodReturns = numpy.zeros((assetReturns.shape[0], sampleSize))
    prevPeriodReturns = numpy.ones(assetReturns.shape[0])
    for i in range(sampleSize):
        r = cumulativeReturns[:,i+periodSize] / prevPeriodReturns - 1.0
        cumulativePeriodReturns[:,i] = r
        prevPeriodReturns = cumulativeReturns[:,i]

    # Compute variance of single period returns
    var = Utilities.mlab_std(assetReturns, axis=1)**2.0

    # Compute variance of period cumulative returns
    var_q = Utilities.mlab_std(cumulativePeriodReturns, axis=1)**2.0
    var_q = ma.masked_where(var_q==0.0, var_q)

    # Finite-sample adjustment, if required
    if finiteSampleAdj:
        m = sampleSize * (1.0 - periodSize / (daysBack - 1.0))
        var_q *= (sampleSize - 1.0) / m
        var *= (daysBack - 1.0) / (daysBack - 2.0)

    s = 2.0 * (2.0 * periodSize - 1.0) * (periodSize - 1.0) / (3.0 * periodSize)
    if takeDifference:
        result = var_q / periodSize - var
        bounds = s / var**2.0
    else:
        result = var_q / (periodSize * var) - 1.0
        bounds = s / (daysBack - 1.0)
    if keepSigStd > 0.0:
        result = ma.masked_where(abs(result) / bounds**0.5 < keepSigStd, result)
        logging.debug('%d out of %d values less than %.1f std devs from 0.0',\
                numpy.sum(ma.getmaskarray(result)), len(result), keepSigStd)

    logging.info('generate_variance_ratios: end')
    return result

def generateAmihudLiquidityExposures(
        modelDate, returns, data, rmg, modelDB, marketDB, daysBack, currencyID=None, useFreeFloatMktCap=False):
    """Returns the liquidity exposures for the assets in universe.
    The liquidity exposure is based on the illiquidity measure defined
    in Amihud(2000): the average of the absolute daily returns divided
    by average daily volume over the past daysBack days.
    """
    # Set up parameters
    dateList = returns.dates
    returns = ma.array(returns.data, copy=True)

    # Set up necessary histories of returns and volumes
    assert returns.shape[1] > daysBack, "Need %d days, got %d" % (daysBack, returns.shape[1])
    returns = returns[:,-daysBack:]
    dateList = dateList[-daysBack:]
    assert returns.shape[1] == daysBack, "Need %d days, got %d" % (daysBack, returns.shape[1])
    allDates = modelDB.getDateRange(None, dateList[0], dateList[-1])
    volume = modelDB.loadVolumeHistory(
            allDates, data.universe, currencyID)
    (volume.data, volume.dates) = ProcessReturns.compute_compound_returns_v3(
            volume.data, allDates, dateList, keepFirst=True, matchDates=True, sumVals=True)

    assert(volume.data.shape[1] == daysBack)
    assert(volume.data.shape[0] == returns.shape[0])

    # Compute ADV for new listings differently
    issueFromDates = Utilities.load_ipo_dates(\
            modelDate, data.universe, modelDB, marketDB, returnList=True, exSpacAdjust=True)
    volume = Matrices.fillAndMaskByDate(volume.data, issueFromDates, volume.dates)
    nGoodDates = ma.sum(ma.getmaskarray(volume)==0, axis=1)

    # Mask out volumes <= 0.0
    volume = ma.where(volume <= 0.0, ma.masked, volume)

    # Scale volume by mcap
    if useFreeFloatMktCap:
        mcap = data.freeFloat_marketCaps[:, numpy.newaxis]
    else:
        mcap = data.marketCaps[:, numpy.newaxis]
    mcap = ma.masked_where(mcap <= 0.0, mcap)
    volume = volume / mcap

    # Compute descriptor
    ratio = ma.absolute(returns) / volume
    illiq = ma.average(ratio, axis=1)
    assert(illiq.shape[0] == len(data.universe))

    # Report on missing values
    nonMissing = ma.sum(ma.getmaskarray(ratio)==0, axis=1)
    missing = numpy.flatnonzero(ma.getmaskarray(illiq))
    if len(missing) > 0:
        logging.info('%d assets are missing Amihud liquidity information', len(missing))
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([data.universe[i].getSubIDString() for i in missing]))

    # Final conversion from iliquidty to liquidity measure
    illiq = 1.0 / illiq
    return illiq

def generate_trading_activity(modelSelector, modelDate, data, modelDB, 
                    currencyID=None, equalWeight=True, daysBack=20):
    """Computes 'volume betas' -- sensitivity of changes in 
    assets' daily volume traded to changes in the average
    volume traded in the overall market.
    By default the market's average 
    volume is computed using equal-weighted average but the
    root-cap weighted average is used if equalWeight=False.
    """
    logging.info('generate_trading_activity: begin')
    univ = data.universe
    dateList = modelDB.getDates(modelSelector.rmg, modelDate, daysBack)
    volume = modelDB.loadVolumeHistory(dateList, univ, currencyID).data
    volume = ma.masked_where(volume<=0.0, volume)
    volumeChanges = (volume[:,1:] - volume[:,:-1]) / volume[:,:-1]
    volumeChanges -= 1.0
    badAssetsIdx = numpy.flatnonzero(numpy.sum(
                ma.getmaskarray(volumeChanges), axis=1)==daysBack)
    marketVolumeChange = volumeChanges.filled(0.0)

    if equalWeight:
        marketVolumeChange = numpy.average(marketVolumeChange, axis=0)
    else:
        weights = data.marketCaps / ma.sum(data.marketCaps)
        marketVolumeChange = numpy.dot(weights, marketVolumeChange)

    regressor = numpy.transpose(numpy.array(
                        [numpy.ones(daysBack, float), marketVolumeChange]))
    (beta, e) = Utilities.ordinaryLeastSquares(
                        numpy.transpose(volumeChanges), regressor)
    beta = ma.array(beta[1,:])
    ma.put(beta, badAssetsIdx, ma.masked)
    logging.info('generate_trading_activity: end')
    return beta

def generate_price_variability(modelDate, modelSelector, assets, 
                               data, modelDB, daysBack=20, copyUSE3=True):
    """Computes the ratio of the maximum to minimum market cap
    attained over the past daysBack days.  Replicates Barra's HILO
    descriptor or Northfield's 'Price Volatility Index' factor.
    Market caps are in local currency, not converted to a numeraire.
    """
    logging.info('generate_price_variability: begin')
    dateList = modelDB.getDates(modelSelector.rmg, modelDate, daysBack)
    mcaps = data.marketCaps
    mcaps = ma.masked_where(mcaps <= 0.0, mcaps)
    high = numpy.max(mcaps, axis=1)
    low = numpy.min(mcaps, axis=1)
    if copyUSE3:
        values = numpy.log(high / low)          # Barra
    else:
        values = (high - low) / (high + low)    # NF
    logging.info('generate_price_variability: end')
    return values

def process_IBES_data_all(date, subIssues, sid2siblings, startDate, endDate, modelDB, marketDB, ibesField, currency_id,
                          realisedField, days=0, dropOldEstimates=False, scaleByTSO=False, useNewTable=False,
                          useFixedFrequency=None, trackList=[], splitAdjust=None):

    """ Logic to extract IBES data, retrieving all forecast values that fall
    between the startDate and endDate and that are for a filing date that is
    greater than the 'latest available realized filing date + days'.

    If dropOldEstimates==True, drop forecast values if the forecast's eff_dt
    precedes the model date (date) by more than one year (this is required to
    remove 'stale' forecasts that are no longer relevant).

    If scaleByTSO==True, the IBES data is scaled by the tso for the eff_dt
    associated with the IBES entry (this is required to convert eps values
    to earnings in currency units); it is then divided by 1e6.0, thereby returning
    ibes data in 'million currency units'. If scaleByTSO==False, the IBES data
    is not scaled at all, and returned in the units in which it
    is stored in the database (note that per-share fields are stored in 'currency
    units' and all other fields are stored in 'million currency units').

    Returns a list of lists of lists.

    Note that the call to modelDB.getIBESCurrencyItem() essentially invokes the
    following query:
        SELECT sub_issue_id, dt, value, currency_id, eff_dt
        FROM sub_issue_estimate_data_active a
        WHERE item_code=:someCode AND dt BETWEEN :startDate AND :endDate
        AND sub_issue_id IN (...)
        AND eff_dt=(SELECT MAX(eff_dt) FROM sub_issue_estimate_data_active b
                    WHERE a.sub_issue_id = b.sub_issue_id
                    AND a.item_code = b.item_code
                    AND a.dt = b.dt
                    AND b.eff_dt <= :date )
        ORDER BY dt, eff_dt
    That is, it retrieves forecast data where the dt is within some date range.
    For each (asset, item_code, dt) tuple, it then retrieves the estimate with
    the max eff_dt, provided the eff_dt <= the date (modelDate).

    """
    namedTuple = collections.namedtuple('ibesTuple', ['dt', 'val', 'ccy', 'effDate'])

    # Grab extra sibings
    extraAssets = set([sid for sl in sid2siblings.values() for sid in sl])
    allAssets = list(set(subIssues).union(extraAssets))

    # Pull out IBES data
    fcDataRaw = modelDB.getIBESCurrencyItem(ibesField, startDate, endDate, allAssets, date, marketDB,
                convertTo=currency_id, namedTuple=namedTuple, splitAdjust=splitAdjust)

    # Drop out-of-date estimates
    fcData = []
    for est in fcDataRaw:
        if (est is not None) and (est is not ma.masked) and len(est) > 0:
            eVals = [ev for ev in est if ev.dt >= startDate]
        else:
            eVals = []
        fcData.append(eVals)

    # Load realised data for comparison
    if useFixedFrequency is None:
        (rlDataRaw, frequencies) = modelDB.getMixFundamentalCurrencyItem(
                realisedField, startDate, date, allAssets, date, marketDB, currency_id)
    else:
        itemcode = '%s%s' % (realisedField, useFixedFrequency.suffix)
        if useNewTable:
            itemcode = itemcode.upper()
        rlDataRaw = modelDB.getFundamentalCurrencyItem(\
                itemcode, startDate, date, allAssets, date, marketDB, currency_id, useNewTable=useNewTable)
    rlData = Utilities.extractLatestValueAndDate(rlDataRaw)

    # Get list of realised data dates
    realDates = []
    for rl in rlData:
        if (rl is not None) and (rl is not ma.masked) and len(rl) > 0:
            realDates.append(rl[0])
        else:
            realDates.append(None)

    # Load TSO history for all effDates
    if scaleByTSO:
        allEffDates = set()
        for est in fcData:
            for e in est:
                allEffDates.add(e.effDate)
        allEffDates = sorted(list(allEffDates))

    # Initialise
    tmpEstData = dict()
    sid2EffDates = dict()

    # Loop round forecast data by asset
    for idx, (est, dt) in enumerate([list(a) for a in zip(fcData, realDates)]):
        sid = allAssets[idx]

        # Keep forecast values if the forecast dt is at least x days greater than
        # the latest date for which realized data is available
        if (dt is None):
            eVals = est
        else:
            eVals = [ev for ev in est if ev.dt > dt + datetime.timedelta(days)]

        # Keep forecast values if the eff_dt precedes the modelDate (date)
        # by less than 366 days, ensuring that forecast values are not 'stale'
        if dropOldEstimates:
            eVals = [ev for ev in eVals if (date - ev.effDate).days <= 366]
        tmpEstData[sid] = eVals

        if sid in trackList:
            printVals = True
            logging.info('TRACKING %s, IBES %s', sid.getSubIDString(), ibesField)
            for e in eVals:
                logging.info('...... %.2f, %s, %s, %s', e.val, e.dt, e.ccy, e.effDate)

        # Get effDates for each sub-issue in case we scale by TSO
        if scaleByTSO:
            sid2EffDates[sid] = []
            for e in eVals:
                if (e is not None) and (len(e) == 4):
                    sid2EffDates[sid].append(e.effDate)

    # If scaleByTSO, multiply forecast value by tso on forecast's eff_dt
    if scaleByTSO:
        sid2TSO = getCompanyTSO(
                date, allEffDates, sid2siblings, sid2EffDates, modelDB, marketDB)
        fcData = dict()

        for sid, eVals in tmpEstData.items():
            tmp_eVals = []
            if sid in trackList:
                logging.info('TRACKING %s, TSO', sid.getSubIDString())
                dtList = []

            for e in eVals:
                # scale by tso retrieved for the eff_dt associated with the forecast value
                if (sid in sid2TSO) and (e.effDate in sid2TSO[sid]):
                    newVal = (e.dt, e.val*sid2TSO[sid][e.effDate]) + e[2:]
                    tmp_eVals.append(namedTuple(*newVal))
                    if sid in trackList:
                        if e.effDate not in dtList:
                            logging.info('...... %s, %f M', e.effDate, sid2TSO[sid][e.effDate])
                            dtList.append(e.effDate)

            fcData[sid] = tmp_eVals
        return fcData

    return tmpEstData

def getCompanyTSO(date, allEffDates, sid2sib, sid2EffDates, modelDB, marketDB):

    subIssues = sorted(sid2EffDates.keys())
    allEffDates = sorted(set(allEffDates))

    # Load TSO history for all effDates
    tsoIssues = sorted(set(subIssues))
    tsoData = modelDB.loadTSOHistory(allEffDates, tsoIssues, expandForMissing=True).toDataFrame()
    tsoData = tsoData.fillna(method='ffill', axis=1).loc[:, allEffDates]

    # Initialise
    sid_to_TSO = defaultdict(dict)

    # Loop round forecast data by asset
    for sid, effDates in sid2EffDates.items():
        for effDt in effDates:
            tso = retrieveTSOValue(sid, tsoData, effDt, modelDB)

            # scale by tso retrieved for the eff_dt associated with the forecast value
            if (tso is not None) and (tso > 0.0):
                sid_to_TSO[sid][effDt] = tso

    return sid_to_TSO


    subIssues = sorted(sid2EffDates.keys())
    allEffDates = sorted(set(allEffDates))

    # Load TSO history for all effDates
    tsoIssues = sorted(set(subIssues))
    tsoData = modelDB.loadTSOHistory(allEffDates, tsoIssues, expandForMissing=True).toDataFrame()
    tsoData = tsoData.fillna(method='ffill', axis=1).loc[:, allEffDates]

    # Initialise
    sid_to_TSO = defaultdict(dict)

    # Loop round forecast data by asset
    for sid, effDates in sid2EffDates.items():
        for effDt in effDates:
            tso = retrieveTSOValue(sid, tsoData, effDt, modelDB)

            # scale by tso retrieved for the eff_dt associated with the forecast value
            if (tso is not None) and (tso > 0.0):
                sid_to_TSO[sid][effDt] = tso

    return sid_to_TSO


    subIssues = sorted(sid2EffDates.keys())
    allEffDates = sorted(set(allEffDates))

    # Load TSO history for all effDates
    tsoIssues = sorted(set(subIssues))
    tsoData = modelDB.loadTSOHistory(allEffDates, tsoIssues, expandForMissing=True).toDataFrame()
    tsoData = tsoData.fillna(method='ffill', axis=1).loc[:, allEffDates]

    # Initialise
    sid_to_TSO = defaultdict(dict)

    # Loop round forecast data by asset
    for sid, effDates in sid2EffDates.items():
        for effDt in effDates:
            tso = retrieveTSOValue(sid, tsoData, effDt, modelDB)

            # scale by tso retrieved for the eff_dt associated with the forecast value
            if (tso is not None) and (tso > 0.0):
                sid_to_TSO[sid][effDt] = tso

    return sid_to_TSO

def retrieveTSOValue(sid, tsoData, effDate, modelDB, defaultVal=0.0):
    # scale by tso retrieved for the eff_dt associated with the forecast value
    tsoRow = tsoData.loc[sid, :]
    if not numpy.isnan(tsoRow[effDate]):
        return tsoRow[effDate] / 1.0e6

    # TODO Later .....
    # Reminder - tso[2] can sometimes be None - what should we do in such situations
    tso = modelDB.loadTSOHistoryWithBackfill(effDate, sid)
    if (tso is not None) and (tso[2] is not None):
        return tso[2] / 1.0e6
    return defaultVal

def generate_est_earnings_to_price_12MFL(modelDate, assetData, modelDB, marketDB, params, currency_id, daysBack=(2*366),
            useFixedFrequency=None, negativeTreatment=None, trackList=[]):
    """Compute a 12-month forward-looking estimated earnings-to-price ratio for the
    assets in data.universe.

    For each asset, this routine:

    - Retrieves IBES earnings estimates, in million currency units, that fall between
    the modelDate and 'modelDate + 3 years'.

    - If IBES earnings estimates are available for the current and next fiscal years,
    it takes a weighted average of these two estimates, multiplies the weighted average
    by 1.0e6, converting it to currency units, and then divides by issuer market cap
    to obtain earnings_to_price estimates

    - If only one IBES earnings estimate is available over the current fiscal year,
    it is multiplied by 1.0e6, converting it to currency units, and then divided by
    issuer market cap

    Returns a masked array
    """
    logging.info('generate_est_earnings_to_price_12MFL: begin')

    # Set up important parameters
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(3*366)

    if hasattr(params, 'maskNegative'):
        maskNegative = params.maskNegative
    else:
        maskNegative = False

    # Load raw IBES data
    fcValues = process_IBES_data_all(modelDate, assetData.universe, assetData.sid2sib, startDate, endDate, modelDB, marketDB,
                            'eps_median_ann', currency_id, 'ibei', days=0, dropOldEstimates=True, scaleByTSO=True,
                            useFixedFrequency=useFixedFrequency, trackList=trackList)
    allEstEarnings = pd.Series(numpy.nan, index=fcValues.keys())

    # Get estimate as linear combination of next two estimates based on dates
    for sid in fcValues.keys():
        printVals = False
        currFYidx = None
        nextFYidx = None

        # Get relevant estimates for this asset
        est = fcValues[sid]
        eVals = [ev for ev in est if ev.dt > modelDate]
        if sid in trackList:
            printVals = True
            logging.info('TRACKING %s, IBES eps_median_ann', sid.getSubIDString())

        if len(eVals) >= 2:
            # Form linear combination of next two estimates
            if (eVals[0].dt - modelDate).days <= 366:
                currFYidx = 0
                dtDiff = numpy.array([(eVals[j].dt - eVals[currFYidx].dt).days for j in range(1, len(eVals))])
                index = numpy.where(numpy.logical_and(dtDiff >= 330, dtDiff <= 400))
                if (len(index) > 0) and (len(index[0]) > 0):
                    nextFYidx = index[0][0] + 1
                    if eVals[currFYidx].effDate <= eVals[nextFYidx].effDate:
                        tau = (eVals[currFYidx].dt - modelDate).days/365.0
                        allEstEarnings[sid] = (tau*eVals[currFYidx].val + (1-tau)*eVals[nextFYidx].val)*1.0e6

                        if printVals:
                            logging.info('......... (%.2fx%.2f B, %s), (%.2fx%.2f B, %s)',\
                                    tau, eVals[currFYidx].val/1.0e3, eVals[currFYidx].dt, 1.0-tau, eVals[nextFYidx].val/1.0e3, eVals[nextFYidx].dt)
                    else:
                        nextFYidx = None

        elif len(eVals) == 1:
            # If only one estimate available, use that if it's close enough
            if (eVals[0].dt - modelDate).days <= 366:
                currFYidx = 0

        if currFYidx is not None and nextFYidx is None:
            allEstEarnings[sid] = eVals[currFYidx].val*1.0e6
            if printVals:
                logging.info('......... (%.2f, %.2f)', eVals[currFYidx].val, eVals[currFYidx].dt)

    # First pull out issues with non-missing IBES estimates
    missingSids = set(allEstEarnings[allEstEarnings.isnull()].index)

    # Combine estimates across valid siblings
    estEarnings = pd.Series(numpy.nan, index=assetData.universe)
    for sid in set(assetData.universe).difference(missingSids):
        sumEarn = allEstEarnings[assetData.sid2sib[sid]].sum()
        if numpy.isnan(sumEarn):
            estEarnings[sid] = allEstEarnings[sid]
        else:
            estEarnings[sid] = sumEarn
        if sid in trackList:
            tsid = sid
            logging.info('Final combined EstEarnings for %s: %.2f B', sid.getSubIDString(), estEarnings[sid]/1.0e9)
            for sib in assetData.sid2sib[sid]:
                logging.info('......... (%s, %.2f B)', sib.getSubIDString(), allEstEarnings[sib]/1.0e9)

    # Convert to ETP
    mcaps = pd.Series(assetData.issuerTotalMarketCaps, index=assetData.universe)
    estETP = estEarnings / mcaps

    # Mask or set to zero negative earnings
    if maskNegative:
        estETP = estETP.mask(estETP < 0.0)
    if negativeTreatment == 'zero':
        estETP = estETP.mask(estETP < 0.0, 0.0)

    # Report on missing values
    missing = set(estETP[estETP.isnull()].index)
    if len(missing) > 0:
        logging.info('%d (%.2f%%) assets are missing estETPV2 information',
                     len(missing), 100*float(len(missing))/float(len(assetData.universe)))
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([sid.getSubIDString() for sid in missing]))

    logging.info('generate_est_earnings_to_price_12MFL: end')
    return estETP

def generate_short_interest(modelDate, data, model, modelDB, marketDB,
                            daysBack=(2*365), restrict=None, maskZero=False):
    """Compute the short interest for the assets in data.universe.
    Returns an array with the short interest, defined as: number of shares
    held short/total shares outstanding, for each asset.  
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    logging.info('generate_short_interest: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    siRaw= modelDB.getXpressFeedItem('SHORTINT', startDate, modelDate, subIssues, modelDate, marketDB, splitAdjust='divide')
    tso = modelDB.loadTSOHistory([modelDate], subIssues)
    si = Utilities.extractLatestValue(siRaw)/tso.data[:, 0]
    missing = numpy.flatnonzero(ma.getmaskarray(si))
    tmpData = ma.filled(si, 1.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpData==0.0, tmpData)))

    if len(missing) > 0:
        logging.info('%d (%.2f%%) assets are missing short interest information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([subIssues[i].getSubIDString() for i in missing]))
    if len(zeroVals) > 0:
        logging.info('%d (%.2f%%) assets have zero short interest', len(zeroVals), 100*float(len(zeroVals)))
        if maskZero:
            si = ma.masked_where(si==0.0, si)

    if restrict is not None:
        siRest = Matrices.allMasked(len(data.universe))
        ma.put(siRest, restrict, si)
        for (i,val) in enumerate(si):
            if val is ma.masked:
                siRest[restrict[i]] = ma.masked
        si = siRest
    logging.info('generate_short_interest: end')
    return si

def generate_net_equity_issuance(modelDate, data, modelDB, marketDB, currency_id, daysBack=(5*365)):
    """Computes the log of the ratio of split adjusted shares from one year to the next
    """
    logging.info('generate_net_equity_issuance: begin')
    lookBackStartDate = modelDate - datetime.timedelta(daysBack)
    adjStartDate = modelDate - datetime.timedelta(364)
    tsoStartDate = modelDate - datetime.timedelta(365)
    allDates = modelDB.getDateRange(None, lookBackStartDate, modelDate)
    dateIdxMap = dict(zip(allDates, list(range(len(allDates)))))

    # Load in 5 years history of TSO
    tso = modelDB.loadTSOHistory(allDates, data.universe).data

    # Roll forward TSO to fill in missing dates
    tso_df = pd.DataFrame(tso)
    tso_df = tso_df.fillna(method='ffill',axis=1)
    tso = ma.array(tso_df.values.copy(), mask=pd.isnull(tso_df).values)

    # Load in the adjustment data
    adjDict = modelDB.getShareAdjustmentFactors(adjStartDate, modelDate, data.universe, marketDB)

    # Compute the cumulative adjustment
    cumAdjList = []
    for sid in data.universe:
        cumAdj = 1.0
        if sid in adjDict:
            sidAdj = adjDict[sid]
            for adj in sidAdj:
                cumAdj *= adj[1]
        cumAdjList.append(cumAdj)

    # Finally, compute the EISS
    values = ma.log(tso[:,-1] / (tso[:,dateIdxMap[tsoStartDate]] * cumAdjList)) 
    return values

def generate_net_debt_issuance(modelDate, data, modelDB, marketDB, currency_id,
        daysBack=365, quarterly=False):
    """Computes the log of the ratio of total debt from one year to the next
    """
    logging.info('generate_net_debt_issuance: begin')

    # Load in current total debt
    startDate = modelDate - datetime.timedelta(daysBack)
    if quarterly:
        debt_t = modelDB.getQuarterlyTotalDebt(
                startDate, modelDate, data.universe, modelDate, marketDB, currency_id)
    else:
        debt_t = modelDB.getAnnualTotalDebt(
                startDate, modelDate, data.universe, modelDate, marketDB, currency_id)
    debt_t = Utilities.extractLatestValue(debt_t)

    # Load in previous total debt
    newStartDate = startDate - datetime.timedelta(daysBack)
    if quarterly:
        debt_tm1 = modelDB.getQuarterlyTotalDebt(
                newStartDate, startDate, data.universe, startDate, marketDB, currency_id)
    else:
        debt_tm1 = modelDB.getAnnualTotalDebt(
                newStartDate, startDate, data.universe, startDate, marketDB, currency_id)
    debt_tm1 = Utilities.extractLatestValue(debt_tm1)

    return ma.log(debt_t / debt_tm1)

def generate_net_payout_over_profits(modelDate, data, modelDB, marketDB, currency_id, quarterly=False):
    """Computes the ratio of net income minus change in book equity to total profits over 
    the last 5 years
    """

    from riskmodels import DescriptorRatios
    # Load net income
    newModelDate = modelDate
    totalIncome = numpy.zeros((len(data.universe)), float)
    totalSales = numpy.zeros((len(data.universe)), float)
    totalCOGS = numpy.zeros((len(data.universe)), float)
    for dt_iter in range(5):
        if quarterly:
            income, freqList = DescriptorRatios.EarningsAlone(modelDB, marketDB,
                    useFixedFrequency=None,
                    numeratorProcess='annualize',
                    numeratorNegativeTreatment=None,
                    sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)
            sales, freqList = DescriptorRatios.SalesAlone(modelDB, marketDB,
                    useFixedFrequency=None,
                    numeratorProcess='annualize',
                    sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)
            cogs, freqList = DescriptorRatios.CostOfGoodsAlone(modelDB, marketDB,
                    useFixedFrequency=None,
                    numeratorProcess='annualize',
                    sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)
        else:
            income, freqList = DescriptorRatios.EarningsAlone(modelDB, marketDB,
                    useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                    numeratorProcess='extractlatest',
                    numeratorNegativeTreatment=None,
                    sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)
            sales, freqList = DescriptorRatios.SalesAlone(modelDB, marketDB,
                    useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                    numeratorProcess='extractlatest',
                    sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)
            cogs, freqList = DescriptorRatios.CostOfGoodsAlone(modelDB, marketDB,
                    useFixedFrequency=DescriptorRatios.DescriptorRatio.AnnualFrequency,
                    numeratorProcess='extractlatest',
                    sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)

        newModelDate = modelDate - datetime.timedelta(365)
        totalIncome = totalIncome + ma.filled(income, 0.0)
        totalSales = totalSales + ma.filled(sales, 0.0)
        totalCOGS = totalCOGS + ma.filled(cogs, 0.0)

    # Load latest book equity
    if quarterly:
        bookLatest, freqList = DescriptorRatios.BookAlone(modelDB, marketDB,
                sidRanges=data.sidRanges).getValues(modelDate, data, currency_id)
    else:
        bookLatest, freqList = DescriptorRatios.BookAlone(modelDB, marketDB,
                DescriptorRatios.DescriptorRatio.AnnualFrequency,
                sidRanges=data.sidRanges).getValues(modelDate, data, currency_id)

    # Load previous book equity
    newModelDate = modelDate - datetime.timedelta(5*365)
    if quarterly:
        bookPrev, freqList = DescriptorRatios.BookAlone(modelDB, marketDB,
                sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)
    else:
        bookPrev, freqList = DescriptorRatios.BookAlone(modelDB, marketDB,
                DescriptorRatios.DescriptorRatio.AnnualFrequency,
                sidRanges=data.sidRanges).getValues(newModelDate, data, currency_id)

    # Now compute the beast
    nops = (totalIncome + bookLatest - bookPrev) / (totalSales - totalCOGS)
    return nops

def generate_share_buyback(modelDate, data, model, modelDB, marketDB,
                            daysBack=(2*365), restrict=None, maskZero=False):
    """Compute the shares buyback for the assets in data.universe.
    Returns an array with the shares buyback, defined as: sum of number of 
    shares boughtback over the previous 4 quarters/ total shares outstanding, 
    for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    logging.info('generate_shares_buyback: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
        restrict = list(range(len(data.universe)))
    else:
        subIssues = [data.universe[i] for i in restrict]
    buyBackRaw= modelDB.getXpressFeedItem('CSHOPQ',
                                     startDate, modelDate, subIssues, modelDate, 
                                     marketDB)
    buyBack = modelDB.annualizeQuarterlyValues(buyBackRaw, adjustShortHistory=True)
    tso = modelDB.loadTSOHistory([modelDate], subIssues)
    buyBack = buyBack * 1e6/ tso.data[:, 0]
    missing = numpy.flatnonzero(ma.getmaskarray(buyBack))
    tmpData = ma.filled(buyBack, 1.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpData==0.0, tmpData)))

    if len(missing) > 0:
        logging.info('%d (%.2f%%) assets are missing shares buy back information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([subIssues[i].getSubIDString() for i in missing]))

    if len(zeroVals) > 0:
        logging.info('%d (%.2f%%) assets have zero buy back', len(zeroVals), 100*float(len(zeroVals)))
        if maskZero:
            buyBack = ma.masked_where(buyBack==0.0, buyBack)

    if restrict is not None:
        buyBackRest = Matrices.allMasked(len(data.universe))
        ma.put(buyBackRest, restrict, buyBack)
        for (i,val) in enumerate(buyBack):
            if val is ma.masked:
                buyBackRest[restrict[i]] = ma.masked
        buyBack = buyBackRest
    logging.info('generate_shares_buyback: end')
    return buyBack

def generate_growth_rate_annual(item, assetData, modelDate, currency_id, modelDB, marketDB,
            daysBack=int((5+10./12)*366), forecastItem=None, forecastItemScaleByTSO=False, trackList=[],
            computeVar=False, splitAdjust=None):
    """Generic code to generate growth rate for particular fundamental data item.
    item - fundamental data item code
    subids - the universe you want to generate growth rate for,
    Returns an array of regression coefficients divided by the average absolute item value
    """
    logging.info('generate_growth_rate_annual for %s: begin', item)
    # Initialise
    subids = assetData.universe
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(366)
    slopes = pd.Series(numpy.nan, index=assetData.universe)
    minObs = 2
    if computeVar:
        minObs = 4

    # Load realized fundamental data
    values = modelDB.getFundamentalCurrencyItem(item + '_ann' , startDate, modelDate, assetData.universe,
            modelDate, marketDB, convertTo=currency_id, splitAdjust=splitAdjust)

    # Load forecast data if required
    if forecastItem is not None:
        fcValues = process_IBES_data_all(modelDate, assetData.universe, assetData.sid2sib, startDate, endDate,
                modelDB, marketDB, forecastItem, currency_id, item, days=90, dropOldEstimates=False,
                scaleByTSO=forecastItemScaleByTSO, useFixedFrequency=DataFrequency('_ann'),
                trackList=trackList, splitAdjust=splitAdjust)

    # Loop round each subissue in turn
    for (idx, sid) in enumerate(assetData.universe):
        realVals = values[idx]

        # Pick out eligible data values
        valueDict = dict([vl[:2] for vl in realVals])
        if len(valueDict) < 1:
            continue
        latestRealDt = realVals[-1][0]
        valueDates = [vl[0] for vl in realVals if (latestRealDt - vl[0]).days/365. <= 4.05]
        valueArray = numpy.array([valueDict[dt] for dt in valueDates])
        if len(numpy.unique(valueArray)) <= minObs:
            continue

        if sid in trackList:
            logging.info('Using Realised values for %s:', sid.getSubIDString())
            for (itemdt, itemvl) in zip(valueDates, valueArray):
                logging.info('..... (%s, %s)', itemvl, itemdt)

        # Add forecast data if any
        if (forecastItem is not None):

            # Find closest forecast for asset
            fcVals = fcValues[sid]
            if len(fcVals) > 0:

                # Get fcValue that is closest to the date that should follow the sequence of realized dates
                targetDate = latestRealDt + datetime.timedelta(366)
                fcIdx = numpy.array([abs(fcv.dt - targetDate) for fcv in fcVals]).argmin()
                fcDate = fcVals[fcIdx].dt
                fcVal = fcVals[fcIdx].val

                # If forecast date is not too hot or too cold, but just right, keep it
                if (fcDate > latestRealDt) and (fcDate < (latestRealDt+datetime.timedelta(400))):

                    if sid in trackList:
                        logging.info('Using IBES values for %s: (%s, %s)', sid.getSubIDString(), fcVal, fcDate)

                    if forecastItemScaleByTSO:
                        # Now check siblings for date matches
                        sibVal = 0.0
                        validSiblings = False
                        for sibl in assetData.sid2sib[sid]:
                            sibVals = fcValues[sibl]

                            if len(sibVals) > 0:
                                fcIdx = numpy.array([abs(fcv.dt - fcDate) for fcv in sibVals]).argmin()
                                sibDate = sibVals[fcIdx].dt
                                validSiblings = True

                                # Keep sibling forecast if it's within 1 month
                                if (fcDate-datetime.timedelta(30)<sibDate<fcDate+datetime.timedelta(30)):
                                    sibVal += sibVals[fcIdx].val
                                    if sid in trackList:
                                        logging.info('.... Using IBES values for %s sibling %s: (%s, %s)',
                                            sid.getSubIDString(), sibl.getSubIDString(), sibVals[fcIdx].val, sibDate)
                        if validSiblings:
                            fcVal = sibVal

                    if sid in trackList:
                        logging.info('Final combined IBES values for %s: %.2f B', sid.getSubIDString(), fcVal/1.0e3)

                    # Append to realised arrays
                    valueArray = numpy.append(valueArray, fcVal)
                    valueDates = numpy.append(valueDates, fcDate)

        # Perform the regression
        if computeVar:
            #valueArray = -1.0 + (valueArray[1:] / valueArray[:-1])
            coef = [numpy.std(valueArray, ddof=1)]
        else:
            x_axis = numpy.cumsum([(valueDates[n]-valueDates[n-1]).days
                    if n >= 1 else 365 for n in range(0, len(valueDates))])/365.0
            (coef, resid) = Utilities.ordinaryLeastSquares(
                    valueArray, numpy.transpose(numpy.array([x_axis, numpy.ones(len(valueArray))])))

        # Normalise the computed value
        slopes[sid] = coef[0] / numpy.average(abs(valueArray))

    # Report on missing values
    missing = set(slopes[slopes.isnull()].index)
    if len(missing) > 0:
        logging.info('%d assets are missing %s information', len(missing), item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([sid.getSubIDString() for sid in missing]))

    # Report on zero values, if more than 1% are zero
    tmpArray = slopes.fillna(-999.0).mask(slopes==0.0)
    zeroVals = set(tmpArray[tmpArray.isnull()].index)
    if len(zeroVals) > 0.01*len(slopes):
        logging.info('%d assets have zero values for %s growth', len(zeroVals), item)

    logging.info('generate_growth_rate_annual for %s: end', item)
    return slopes

def combine_forecasts(estVals, latestRealDate, targetDate, track):
    val = None
    if len(estVals) >= 2:
        if 355 <= (estVals[0].dt - latestRealDate).days <= 366:
            val = estVals[0].val
            if track:
                logging.info('... %s, %.2f', estVals[0].dt, estVals[0].val)
        elif ( (estVals[0].dt > latestRealDate) and ((estVals[0].dt - latestRealDate).days < 355) and
               (estVals[1].dt > targetDate) and (355 <= (estVals[1].dt - estVals[0].dt).days <= 366) ):
            tau = (estVals[0].dt - latestRealDate).days/365.0
            val = tau * estVals[0].val + (1 - tau) * estVals[1].val
            if track:
                logging.info('... %s, %.2f*%.2f, %s, %.2f*%.2f',\
                        estVals[0].dt, tau, estVals[0].val, estVals[1].dt, 1.0-tau, estVals[1].val)
    elif len(estVals) == 1:
        if 300 <= (estVals[0].dt - latestRealDate).days <= 400:
            val = estVals[0].val
            if track:
                logging.info('... %s, %.2f', estVals[0].dt, estVals[0].val)
    return val

def generate_growth_rateAFQ_mix_version(item, assetData, modelDate, currency_id, modelDB, marketDB,
        daysBack=(5*366)+94, forecastItem=None, forecastItemScaleByTSO=False, trackList=[], requireConsecQtrData=None):
    '''
    This function retrieves a time series of quarterly or annual realized fundamental data
    for the item code specified, where quarterly data is returned if and only if the
    quarterly data series is consecutive (i.e., with no missing quarters) and has at
    least 16 observations.

    Quarterly data is then converted to annual data, indexed by the most recent realized
    end-of-period date in the quarterly series.

    One forecast element, if available, is appended to the series of realized data. This
    forecast element is comprised of the element (or elements) that most closely correspond
    to the next date in the realized time series (latest realized date + 365 days).

    It then regresses the time series values against time, obtaining a "slope" coefficient,
    which is standardized by the average absolute value in the time series.
    '''
    useFixedFrequency = None
    logging.info('generate_growth_rateAFQ_LEGACY for %s, useFixedFreq-%s: begin', item, useFixedFrequency)
    botDate = datetime.date(1981, 12, 31)
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(2*366)
    slopes = pd.Series(numpy.nan, index=assetData.universe)

    if modelDate >= botDate:
        # Load realized fundamental data
        values, valueFreq = modelDB.getMixFundamentalCurrencyItem(
                item, startDate, modelDate, assetData.universe, modelDate, marketDB,
                convertTo=currency_id, requireConsecQtrData=requireConsecQtrData)

        # Convert quarterly data to annual
        for (idx, sid) in enumerate(assetData.universe):
            vals = values[idx]
            if (valueFreq[idx] == DataFrequency('_qtr')) and (len(vals) > 0):
                tmp = vals[len(vals)%4:len(vals)]
                newValues = []
                for j in numpy.arange(3, len(tmp), 4):
                    newValues.append((tmp[j][0], tmp[j][1] + tmp[j-1][1] + tmp[j-2][1] + tmp[j-3][1], tmp[j][2]))
                values[idx] = newValues
    else:
        # Load realized fundamental data without requiring 16 obs of consec qtr data
        values, valueFreq = modelDB.getMixFundamentalCurrencyItem(
                item, startDate, modelDate, assetData.universe, modelDate, marketDB, convertTo=currency_id)

    # Load forecast data if required
    if forecastItem is not None:
        fcValues = process_IBES_data_all(
                    modelDate, assetData.universe, assetData.sid2sib, startDate, endDate, modelDB, marketDB,
                    forecastItem, currency_id, item, dropOldEstimates=True, scaleByTSO=forecastItemScaleByTSO,
                    useFixedFrequency=useFixedFrequency, trackList=trackList)

    # Loop round each subissue in turn
    for (idx, sid) in enumerate(assetData.universe):

        # Set up arrays of dates and values for asset
        realVals = values[idx]
        valueArray = ma.filled(ma.array([val[1] for val in realVals]), 0.0)
        valueDates = ma.array([val[0] for val in realVals])
        # ESTHER - what if a date field is empty? Probably impossible, but need to  check

        # Deal with assets whose histories are too short
        if (len(valueArray) < 1) or (len(numpy.unique(valueArray)) <= 2):
            continue

        track = sid in trackList
        if track:
            logging.info('Using Realised values for %s:', sid.getSubIDString())
            for (itemdt, itemvl) in zip(valueDates, valueArray):
                logging.info('..... (%s, %s)', itemvl, itemdt)

        # Add forecast data if any
        if forecastItem is not None:
            latestRealDate = realVals[-1][0]
            targetDate = latestRealDate + datetime.timedelta(366)

            # Get combination of forecasts closest to the date that should follow the sequence of realized dates
            if track:
                logging.info('Combining raw IBES value for %s ...', sid.getSubIDString())
            estVal = combine_forecasts(fcValues[sid], latestRealDate, targetDate, track)

            if estVal is not None:
                if track:
                    logging.info('Composite IBES value for %s: %s, %.2f', sid.getSubIDString(), targetDate, estVal)

                if forecastItemScaleByTSO:
                    # Now check siblings for date matches
                    totalSibVal = 0.0
                    validSiblings = False
                    for sibl in assetData.sid2sib[sid]:
                        sibVal = combine_forecasts(fcValues[sibl], latestRealDate, targetDate, track)

                        # Combine sibling forecasts
                        if sibVal is not None:
                            validSiblings = True
                            totalSibVal += sibVal
                            if track:
                                logging.info('.... Using Composite IBES values for %s sibling %s: (%s, %s)',
                                    sid.getSubIDString(), sibl.getSubIDString(), targetDate, sibVal)
                    if validSiblings:
                        estVal = totalSibVal

                # Append forecast to realised data
                valueArray = numpy.append(valueArray, [estVal])
                valueDates = numpy.append(valueDates, [targetDate])
                if track:
                    logging.info('Final combined IBES values for %s: %s, %.2f B', sid.getSubIDString(), targetDate, estVal/1.0e3)

        # Perform the regression - robust or not
        x_axis = numpy.cumsum([(valueDates[n] - valueDates[n-1]).days if n >= 1 else 365 for n in range(0, len(valueDates))])/365.0
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        valueArray, numpy.transpose(numpy.array( [x_axis, numpy.ones(len(valueArray))])))
        slopes[sid] = coef[0] / numpy.average(abs(valueArray))

    # Report on missing values
    missing = set(slopes[slopes.isnull()].index)
    if len(missing) > 0:
        logging.info('%d assets are missing %s information', len(missing), item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([sid.getSubIDString() for sid in missing]))

    # Report on zero values, if more than 1% are zero
    tmpArray = slopes.fillna(-999.0).mask(slopes==0.0)
    zeroVals = set(tmpArray[tmpArray.isnull()].index)
    if len(zeroVals) > 0.01 * len(slopes):
        logging.info('%d assets have zero values for %s growth', len(zeroVals), item)

    logging.info('generate_growth_rateAFQ_LEGACY for %s, useFixedFreq-%s: begin', item, useFixedFrequency)
    return slopes

def generate_growth_rateAFQ(item, assetData, modelDate, currency_id, modelDB, marketDB, daysBack=(5*366)+94,
        forecastItem=None, forecastItemScaleByTSO=False, trackList=[], requireConsecQtrData=None,
        computeVar=False, splitAdjust=None):
    '''
    This function retrieves a time series of quarterly realized fundamental data
    for the item code specified, where data is returned if and only if the
    data series is consecutive (i.e., with no missing quarters) and has at
    least 16 observations.

    Quarterly data is then converted to annual data, indexed by the most recent realized
    end-of-period date in the quarterly series.

    One forecast element, if available, is appended to the series of realized data. This
    forecast element is comprised of the element (or elements) that most closely correspond
    to the next date in the realized time series (latest realized date + 365 days).

    It then regresses the time series values against time, obtaining a "slope" coefficient,
    which is standardized by the average absolute value in the time series.
    '''
    dataFrequency = DataFrequency('_qtr')
    logging.info('generate_growth_rateAFQ for %s, StDev: %s, dataFreq-%s: begin', item, computeVar, dataFrequency.suffix)
    minObs = 2
    botDate = datetime.date(1981, 12, 31)
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(2*366)
    slopes = pd.Series(numpy.nan, index=assetData.universe)
    if computeVar:
        botDate = datetime.date(1982, 12, 31)
        minObs = 4

    if modelDate >= botDate:
        values = modelDB.getFundamentalCurrencyItem(
                item + dataFrequency.suffix, startDate, modelDate, assetData.universe,
                modelDate, marketDB, currency_id, splitAdjust=splitAdjust,
                requireConsecQtrData=requireConsecQtrData)

        # Convert quarterly data to annual
        if not computeVar:
            tmpVals = []
            for vals in values:
                newValues = []
                if len(vals) > 0:
                    tmp = vals[len(vals)%4:len(vals)]
                    for j in numpy.arange(3, len(tmp), 4):
                        newValues.append((tmp[j][0], tmp[j][1] + tmp[j-1][1] + tmp[j-2][1] + tmp[j-3][1], tmp[j][2]))
                tmpVals.append(newValues)
            values = tmpVals
    else:
        values = modelDB.getFundamentalCurrencyItem(
                item + dataFrequency.suffix, startDate, modelDate, assetData.universe,
                modelDate, marketDB, currency_id, splitAdjust=splitAdjust, requireConsecQtrData=None)

    # Load forecast data if required
    if forecastItem is not None:
        fcValues = process_IBES_data_all(
                    modelDate, assetData.universe, assetData.sid2sib, startDate, endDate, modelDB, marketDB,
                    forecastItem, currency_id, item, dropOldEstimates=True, scaleByTSO=forecastItemScaleByTSO,
                    useFixedFrequency=dataFrequency, trackList=trackList, splitAdjust=splitAdjust)

    # Loop round each subissue in turn
    for (idx, sid) in enumerate(assetData.universe):

        # Set up arrays of dates and values for asset
        realVals = values[idx]
        valueArray = ma.filled(ma.array([val[1] for val in realVals]), 0.0)
        # ESTHER - what if a date field is empty? Probably impossible, but need to check
        valueDates = ma.array([val[0] for val in realVals])

        # Deal with assets whose histories are too short
        if (len(valueArray) < 1) or (len(numpy.unique(valueArray)) <= minObs):
            continue

        track = sid in trackList
        if track:
            logging.info('Using Realised values for %s:', sid.getSubIDString())
            for (itemdt, itemvl) in zip(valueDates, valueArray):
                logging.info('..... (%s, %s)', itemvl, itemdt)

        # Add forecast data if any
        if forecastItem is not None:
            latestRealDate = realVals[-1][0]
            targetDate = latestRealDate + datetime.timedelta(366)

            # Get combination of forecasts closest to the date that should follow the sequence of realized dates
            if track:
                logging.info('Combining raw IBES value for %s ...', sid.getSubIDString())
            estVal = combine_forecasts(fcValues[sid], latestRealDate, targetDate, track)

            if estVal is not None:
                if track:
                    logging.info('Composite IBES value for %s: %s, %.2f', sid.getSubIDString(), targetDate, estVal)
                    for j in numpy.arange(3, len(tmp), 4):
                        newValues.append((tmp[j][0], tmp[j][1] + tmp[j-1][1] + tmp[j-2][1] + tmp[j-3][1], tmp[j][2]))
                tmpVals.append(newValues)
            values = tmpVals

    # Load forecast data if required
    if forecastItem is not None:
        fcValues = process_IBES_data_all(
                    modelDate, assetData.universe, assetData.sid2sib, startDate, endDate, modelDB, marketDB,
                    forecastItem, currency_id, item, dropOldEstimates=True, scaleByTSO=forecastItemScaleByTSO,
                    useFixedFrequency=dataFrequency, trackList=trackList, splitAdjust=splitAdjust)

    # Loop round each subissue in turn
    for (idx, sid) in enumerate(assetData.universe):

        # Set up arrays of dates and values for asset
        realVals = values[idx]
        valueArray = ma.filled(ma.array([val[1] for val in realVals]), 0.0)
        # ESTHER - what if a date field is empty? Probably impossible, but need to check
        valueDates = ma.array([val[0] for val in realVals])

        # Deal with assets whose histories are too short
        if (len(valueArray) < 1) or (len(numpy.unique(valueArray)) <= minObs):
            continue

        track = sid in trackList
        if track:
            logging.info('Using Realised values for %s:', sid.getSubIDString())
            for (itemdt, itemvl) in zip(valueDates, valueArray):
                logging.info('..... (%s, %s)', itemvl, itemdt)

        # Add forecast data if any
        if forecastItem is not None:
            latestRealDate = realVals[-1][0]
            targetDate = latestRealDate + datetime.timedelta(366)

            # Get combination of forecasts closest to the date that should follow the sequence of realized dates
            if track:
                logging.info('Combining raw IBES value for %s ...', sid.getSubIDString())
            estVal = combine_forecasts(fcValues[sid], latestRealDate, targetDate, track)

            if estVal is not None:
                if track:
                    logging.info('Composite IBES value for %s: %s, %.2f', sid.getSubIDString(), targetDate, estVal)

                if forecastItemScaleByTSO:
                    # Now check siblings for date matches
                    totalSibVal = 0.0
                    validSiblings = False
                    for sibl in assetData.sid2sib[sid]:
                        sibVal = combine_forecasts(fcValues[sibl], latestRealDate, targetDate, track)

                        # Combine sibling forecasts
                        if sibVal is not None:
                            validSiblings = True
                            totalSibVal += sibVal
                            if track:
                                logging.info('.... Using Composite IBES values for %s sibling %s: (%s, %s)',
                                    sid.getSubIDString(), sibl.getSubIDString(), targetDate, sibVal)
                    if validSiblings:
                        estVal = totalSibVal

                # Append forecast to realised data
                if computeVar:
                    estVal = estVal / 4.0
                valueArray = numpy.append(valueArray, [estVal])
                valueDates = numpy.append(valueDates, [targetDate])
                if track:
                    logging.info('Final combined IBES data for %s: %s, %.2f B', sid.getSubIDString(), targetDate, estVal/1.0e3)

        # Perform the regression - robust or not
        if computeVar:
            seasonDecomp = tsa.seasonal.seasonal_decompose(valueArray, model='additive', freq=4)
            valueArray = valueArray - seasonDecomp.seasonal
            #valueArray = -1.0 + (valueArray[1:] / valueArray[:-1])
            coef = [numpy.std(valueArray, ddof=1)]
        else:
            x_axis = numpy.cumsum([(valueDates[n] - valueDates[n-1]).days if n >= 1 \
                    else 365 for n in range(0, len(valueDates))])/365.0
            (coef, resid) = Utilities.ordinaryLeastSquares(
                        valueArray, numpy.transpose(numpy.array( [x_axis, numpy.ones(len(valueArray))])))
        slopes[sid] = coef[0] / numpy.average(abs(valueArray))

    # Report on missing values
    missing = set(slopes[slopes.isnull()].index)
    if len(missing) > 0:
        logging.info('%d assets are missing %s information', len(missing), item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([sid.getSubIDString() for sid in missing]))

    # Report on zero values, if more than 1% are zero
    tmpArray = slopes.fillna(-999.0).mask(slopes==0.0)
    zeroVals = set(tmpArray[tmpArray.isnull()].index)
    if len(zeroVals) > 0.01 * len(slopes):
        logging.info('%d assets have zero values for %s growth', len(zeroVals), item)

    logging.info('generate_growth_rateAFQ for %s, StDev: %s, dataFreq-%s: begin', item, computeVar, dataFrequency.suffix)
    return slopes

def generate_linked_asset_ADV_score(date, universe, numeraire_id, modelDB, daysBack=365):
    """Computes the ADV score for each asset for exposure cloning and ISC
    """
    logging.debug('Start: generate_linked_asset_ADV_score')
    # Set up dates
    dateList = modelDB.getAllRMGDateRange(date, daysBack)
    return modelDB.getAverageTradingVolume(dateList, universe, numeraire_id, loadAllDates=True) / 1.0e6

def generate_linked_asset_ret_score(returns, daysBack, minDays, zeroReturns, countPreIPO=False):
    """Computes the % of non-missing returns score for each asset for exposure cloning and ISC
    """
    logging.debug('Start: generate_linked_asset_ret_score')
    if returns.data.shape[1] <= daysBack:
        daysBack = returns.data.shape[1]

    # Get number of genuine trading days per asset
    if countPreIPO:
        validDays = len(returns.dates[-daysBack:]) - ma.sum(returns.ntdFlag[:,-daysBack:], axis=1)
    else:
        validDays = len(returns.dates[-daysBack:]) - ma.sum(\
                returns.preIPOFlag[:,-daysBack:].astype(int) + returns.ntdFlag[:,-daysBack:].astype(int), axis=1)

    if zeroReturns:
        # Get number of non-zero returns over history
        validRets = validDays - ma.sum(returns.zeroFlag[:,-daysBack:], axis=1)
    elif countPreIPO:
        # Get number of non-missing returns, counting pre-IPO dates as missing
        validRets = validDays - ma.sum(returns.missingFlag[:,-daysBack:].astype(int) \
                - returns.ntdFlag[:,-daysBack:].astype(int), axis=1)
    else:
        # Get number of non-missing returns over history
        validRets = validDays - ma.sum(returns.rollOverFlag[:,-daysBack:].astype(int) \
                - returns.ntdFlag[:,-daysBack:].astype(int), axis=1)

    # Find proportion of valid returns
    validRets = ma.where(validDays < minDays, 0, validRets)
    pcntValidRets = validRets / numpy.array(validDays, float)
    
    return pcntValidRets

def generate_linked_asset_IPO_score(date, universe, modelDB, marketDB, maxAge=1250):
    """Computes the IPO score for each asset for exposure cloning and ISC
    """
    logging.debug('Start: generate_linked_asset_IPO_score')
    # Load from dates and compute "age" of asset
    issueFromDates = Utilities.load_ipo_dates(\
            date, universe, modelDB, marketDB, returnList=True, exSpacAdjust=True)
    age = [min(int((date-dt).days), maxAge) for dt in issueFromDates]
    age = numpy.array(age, float)
    maxAge = max(age)
    if maxAge > 0.0:
        age = age / maxAge
    return age

def generate_linked_asset_scores_legacy(
            date, rmgList, universe, modelDB, marketDB, subIssueGroups,
            currency_id, daysBack=125, scale=15.0):
    """Assigns scores to linked assets based on their cap and
    liquidity, in order to assist cloning or other adjustment
    of exposures, specific risks, correlations etc.
    """
    logging.debug('Start score_linked_assets')
    assetIdxMap = dict([(j,i) for (i,j) in enumerate(universe)])
    scoreDict = dict()
    scores = Matrices.allMasked(len(universe))

    # Determine how many calendar days of data we need
    dateList = modelDB.getDates(rmgList, date, daysBack-1)

    # Compute average trading volume of each asset
    vol = modelDB.loadVolumeHistory(dateList, universe, currency_id)
    volume = numpy.average(ma.filled(vol.data, 0.0), axis=1)

    # Load in market caps
    mcapDates = modelDB.getDates(rmgList, date, 19)
    avgMarketCap = modelDB.getAverageMarketCaps(mcapDates, universe, currency_id, marketDB)

    # Load in returns and get proportion of days traded
    dateList = modelDB.getDates(rmgList, date, 250)
    returns = modelDB.loadTotalReturnsHistory(rmgList, dateList[-1], universe, 250, None)
    propNonMissing = ma.filled(ma.average(~ma.getmaskarray(returns.data), axis=1), 0.0)

    # And do something similar with from dates
    issueFromDates = Utilities.load_ipo_dates(\
            date, universe, modelDB, marketDB, returnList=True, exSpacAdjust=True)
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
                nMissSubSet[ii] = nMissSubSet[ii] / maxNMiss
        else:
            nMissSubSet = numpy.ones((len(indices)), float)

        # Score each asset by its age
        ageSubSet = numpy.take(age, indices, axis=0)
        maxAge = max(ageSubSet)
        if maxAge > 0.0:
            for (ii, idx) in enumerate(indices):
                ageSubSet[ii] = ageSubSet[ii] / maxAge
        else:
            ageSubSet = numpy.ones((len(indices)), float)

        # Now combine the scores and exponentially scale them
        scoreSubSet = volumeSubSet * mcapSubSet * nMissSubSet * ageSubSet
        scoreSubSet = numpy.exp(scale * (scoreSubSet - 1.0))
        scoreDict[groupId] = scoreSubSet
        for (idx, scr) in zip(indices, scoreSubSet):
            scores[idx] = scr

    debuggingReporting = False
    if debuggingReporting:
        idList = []
        scoreList = []
        for (groupId, subIssueList) in subIssueGroups.items():
            sidList = [groupId + ':' + sid.getSubIDString() for sid in subIssueList]
            idList.extend(sidList)
            scoreList.extend(scoreDict[groupId])
        Utilities.writeToCSV(numpy.array(scoreList), 'tmp/legacy_scores-%s.csv' % date, rowNames=idList)

    logging.debug('End score_linked_assets')
    return scores

# Run an example when called as main.
if __name__ == '__main__':
    import optparse
    import pickle

    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    logging.getLogger().setLevel(logging.DEBUG)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)

    riskModel = modelClass(modelDB, marketDB)
    modelDate = datetime.date(2012,7,19)
    data = Utilities.Struct()
    data.universe = [ModelDB.SubIssue('DMVWA65M8311')]
    (mcapDates, goodRatio) = riskModel.getRMDates(
                            modelDate, modelDB, 20, ceiling=False)
    data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
    data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
    data.marketCaps = modelDB.getAverageMarketCaps(
            mcapDates, data.universe, riskModel.numeraire.currency_id, marketDB)

    modelDB.revertChanges()
    marketDB.finalize()
    modelDB.finalize()


import pandas
import numpy.ma as ma
import numpy
import datetime
import logging
import scipy.stats as sm
from marketdb import MarketDB
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities

_log = None
def myLog():
    global _log
    if _log == None:
        _log = logging.getLogger('StyleExposures')
    return _log

def generate_book_to_price(modelDate, data, model, modelDB, marketDB,
                           daysBack=(2*365), restrict=None, 
                           useQuarterlyData=True):
    """Compute the book-to-price ratio for the assets in data.universe.
    The value is computed based on common equity and current market
    capitalization.
    Common equity is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the book-to-price value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_book_to_price: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'ce_qtr'
    else:
        fieldName = 'ce_ann'
    if hasattr(data, 'DLCMarketCap'):
        marketCaps = data.DLCMarketCap
    else:
        marketCaps = data.issuerMarketCaps
    if restrict is None:
        subIssues = data.universe
        divisor = marketCaps
    else:
        subIssues = [data.universe[i] for i in restrict]
        divisor = ma.take(marketCaps, restrict, axis=0)
    ce = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    ceArray = Utilities.extractLatestValue(ce)
    bookToPrice = 1e6 * ceArray / divisor
    missing = numpy.flatnonzero(ma.getmaskarray(bookToPrice))
    if len(missing) > 0:
        myLog().info('%d assets are missing bookToPrice information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        btp = Matrices.allMasked(len(data.universe))
        ma.put(btp, restrict, bookToPrice)
        for (i,val) in enumerate(bookToPrice):
            if val is ma.masked:
                btp[restrict[i]] = ma.masked
        bookToPrice = btp
    myLog().info('generate_book_to_price: end')
    return bookToPrice

def generate_sales_to_price(modelDate, data, model, modelDB, marketDB,
                            daysBack=(2*365), restrict=None, 
                            useQuarterlyData=True):
    """Compute the sales-to-price ratio for the assets in data.universe.
    The value is computed based on sales and market capitalization.
    Sales is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the sales-to-price value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_sales_to_price: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'sale_qtr'
    else:
        fieldName = 'sale_ann'
    if restrict is None:
        subIssues = data.universe
        divisor = data.marketCaps
    else:
        subIssues = [data.universe[i] for i in restrict]
        divisor = ma.take(data.marketCaps, restrict, axis=0)
    sale = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    saleArray = Utilities.extractLatestValue(sale)
    salesToPrice = 1e6 * saleArray / divisor
    missing = numpy.flatnonzero(ma.getmaskarray(salesToPrice))
    if len(missing) > 0:
        myLog().info('%d assets are missing salesToPrice information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        stp = Matrices.allMasked(len(data.universe))
        ma.put(stp, restrict, salesToPrice)
        for (i,val) in enumerate(salesToPrice):
            if val is ma.masked:
                stp[restrict[i]] = ma.masked
        salesToPrice = stp
    myLog().info('generate_sales_to_price: end')
    return salesToPrice

def generate_debt_to_marketcap(modelDate, data, model, modelDB, marketDB,
                               daysBack=(2*365), restrict=None,
                               useQuarterlyData=True, useTotalAssets=False):
    """Compute the debt-to-marketcap ratio for the given assets.
    The value is computed as the ratio of debt and the market capitalization.
    Debt is computed as the sum of long-term debt and debt in current
    liabilities.  If useTotalAssets=True, debt-to-assets is computed instead.
    Both debt values are taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the debt-to-marketcap value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_debt_to_marketcap: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    targetCurrency = model.numeraire.currency_id
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    if useQuarterlyData:
        totalDebt = modelDB.getQuarterlyTotalDebt(
            startDate, modelDate, subIssues, modelDate, marketDB,
            convertTo=targetCurrency)
        fieldName = 'at_qtr'
    else:
        totalDebt = modelDB.getAnnualTotalDebt(
            startDate, modelDate, subIssues, modelDate, marketDB,
            convertTo=targetCurrency)
        fieldName = 'at_ann'
    debtArray = Utilities.extractLatestValue(totalDebt)
    marketCaps = data.issuerMarketCaps
    if useTotalAssets:
        at = modelDB.getFundamentalCurrencyItem(fieldName, 
                startDate, modelDate, subIssues, modelDate, marketDB,
                convertTo=model.numeraire.currency_id)
        totalAssets = Utilities.extractLatestValue(at) * 1e6
        divisor = ma.masked_where(totalAssets < 0.0, totalAssets)
        divisorName = 'Assets'
    else:
        if restrict is None:
            divisor = marketCaps
        else:
            divisor = ma.take(marketCaps, restrict, axis=0)
        divisorName = 'MarketCap'
    debtToMarketCap = 1e6 * debtArray / divisor
    missing = numpy.flatnonzero(ma.getmaskarray(debtToMarketCap))
    if len(missing) > 0:
        myLog().info('%d assets are missing debtTo%s information',
                     len(missing), divisorName)
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, debtToMarketCap)
        for (i,j) in enumerate(debtToMarketCap):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        debtToMarketCap = val
    myLog().info('generate_debt_to_marketcap: end')
    return debtToMarketCap

def generate_earnings_to_price(modelDate, data, model, modelDB, marketDB,
                               daysBack=(2*365), restrict=None,
                               useQuarterlyData=True, legacy=False,
                               maskNegative=True):
    """Compute the earnings-to-price ratio for the assets in data.universe.
    The value is computed based on EBITDA and current market cap.
    Earnings data is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the earnings-to-price value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_earnings_to_price: begin')
    useQuarterlyData = False        # We don't have quarterly data at the moment
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        if legacy:
            fieldName = 'ebitda_qtr'
        else:
            fieldName = 'ibei_qtr'
    else:
        if legacy:
            fieldName = 'ebitda_ann'
        else:
            fieldName = 'ibei_ann'
    if hasattr(data, 'DLCMarketCap'):
        marketCaps = data.DLCMarketCap
    else:
        marketCaps = data.issuerMarketCaps
    if restrict is None:
        subIssues = data.universe
        divisor = marketCaps
    else:
        subIssues = [data.universe[i] for i in restrict]
        divisor = ma.take(marketCaps, restrict, axis=0)
    earnings= modelDB.getFundamentalCurrencyItem(fieldName, 
                startDate, modelDate, subIssues, modelDate, 
                marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        earningsArray = modelDB.annualizeQuarterlyValues(earnings)
    else:
        earningsArray = Utilities.extractLatestValue(earnings)
    earningsToPrice = 1e6 * earningsArray / divisor
    missing = numpy.flatnonzero(ma.getmaskarray(earningsToPrice))
    if len(missing) > 0:
        myLog().info('%d assets are missing earningsToPrice information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if maskNegative:
        earningsToPrice = ma.masked_where(earningsToPrice < 0.0, earningsToPrice)
    if restrict is not None:
        etp = Matrices.allMasked(len(data.universe))
        ma.put(etp, restrict, earningsToPrice)
        for (i,val) in enumerate(earningsToPrice):
            if val is ma.masked:
                etp[restrict[i]] = ma.masked
        earningsToPrice = etp
    myLog().info('generate_earnings_to_price: end')
    return earningsToPrice

def generate_dividend_yield(modelDate, data, model, modelDB, marketDB,
        restrict=None, params=None):
    """Compute the dividend yield for the assets in data.universe.
    The value is computed based on current market capitalization
    and dividends paid out over the past daysBack.
    Returns an array with the dividend yield value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    """
    myLog().info('generate_dividend_yield: begin')
    if restrict is None:
        subIssues = data.universe
        divisor = data.marketCaps
    else:
        subIssues = [data.universe[i] for i in restrict]
        divisor = ma.take(data.marketCaps, restrict, axis=0)

    includeSpecial = False
    includeStock = False
    if params is not None:
        if hasattr(params, 'includeSpecial'):
            includeSpecial = params.includeSpecial
        if hasattr(params, 'includeStock'):
            includeStock = params.includeStock

    divs = computeAnnualDividends(
                modelDate, subIssues, model, modelDB, marketDB, maskMissing=True,
                includeSpecial=includeSpecial, includeStock=includeStock)
    divYield = divs / divisor
    missing = numpy.flatnonzero(ma.getmaskarray(divYield))
    if len(missing) > 0:
        myLog().info('%d assets are missing dividend information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        dy = Matrices.allMasked(len(data.universe))
        ma.put(dy, restrict, divYield)
        for (i,val) in enumerate(divYield):
            if val is ma.masked:
                dy[restrict[i]] = ma.masked
        divYield = dy
    myLog().info('generate_dividend_yield: end')
    return divYield

def computeAnnualDividends(modelDate, subIssues, model, 
                             modelDB, marketDB, maskMissing=False,
                             includeSpecial=True, includeStock=False):
    """Computes total dividends paid for the past year, defined as 
    reported dividends times the shares outstanding on the pay date.
    Fuzzy logic employed to account for periodic (annual, quarterly, etc)
    dividends that are not spaced apart in exactly equal time intervals.
    """
    prevYear = modelDate - datetime.timedelta(380)
    paidDividends = modelDB.getPaidDividends(
        prevYear, modelDate, subIssues, marketDB,
        convertTo=model.numeraire.currency_id,
        includeSpecial=includeSpecial, includeStock=includeStock)
    for (sid, divData) in paidDividends.items():
        if len(divData)==0:
            continue
        highFreq = False
        if len(divData) >= 11:
            highFreq = True
        lastExDate = divData[-1][0]
        toRemove = list()
        if lastExDate >= modelDate - datetime.timedelta(30):
            for (dt, div) in divData:
                if not highFreq and dt < prevYear + datetime.timedelta(28):
                    toRemove.append((dt, div))
                elif highFreq:
                    tolerancePeriod = [lastExDate - datetime.timedelta(365) + \
                                       datetime.timedelta(t) for t in [-4,4]]
                    if dt >= tolerancePeriod[0] and dt <= tolerancePeriod[1]:
                        toRemove.append((dt, div))
        for n in toRemove:
            divData.remove(n)
    if maskMissing:
        paidDividendSum = Matrices.allMasked(len(subIssues))
        for (idx, sid) in enumerate(subIssues):
            divData = paidDividends.get(sid, list())
            if len(divData) > 0:
                paidDividendSum[idx] = ma.sum([dv[1] for dv in divData])
    else:
        paidDividendSum = numpy.array([ma.sum([i[1] for i in
            paidDividends.get(sid, list())])
            for sid in subIssues])
    
    return paidDividendSum

def computeAnnualDividendsNew(modelDate, subIssues, model, 
                              modelDB, marketDB, maskMissing=False,
                              includeSpecial=True, includeStock=False):
    """Computes total dividends paid for the past year, defined as 
    reported dividends times the shares outstanding on the pay date.
    Fuzzy logic employed to account for periodic (annual, quarterly, etc)
    dividends that are not spaced apart in exactly equal time intervals.
    """
    prevYear = modelDate - datetime.timedelta(365 + 75)
    annSumFromDtDefault = modelDate - datetime.timedelta(365 + 15)
    recentDivCutoff = modelDate - datetime.timedelta(60)
    paidDividends = modelDB.getPaidDividends(
        prevYear, modelDate, subIssues, marketDB,
        convertTo=model.numeraire.currency_id,
        includeSpecial=includeSpecial, includeStock=includeStock)
    for (sid, divData) in paidDividends.items():
        if len(divData)==0:
            continue
        lastExDate = divData[-1][0]
        annSumFromDt = annSumFromDtDefault
        if lastExDate >= recentDivCutoff:
            prevDivPeriodicity = len([d for d in divData
                                      if (d[0] >= lastExDate - datetime.timedelta(365 + 15)
                                          and d[0] <= lastExDate - datetime.timedelta(365 - 15))
                                      or (d[0] >= lastExDate - datetime.timedelta(365 - 45)
                                          and d[0] <= lastExDate - datetime.timedelta(365 - 75))])
            if prevDivPeriodicity >= 1:
                annSumFromDt = max(annSumFromDt, lastExDate - datetime.timedelta(365 - 15))
            else:
                annSumFromDt = lastExDate - datetime.timedelta(365 - 45)
        toRemove = [d for d in divData if d[0] < annSumFromDt]
        for d in toRemove:
            divData.remove(d)
    if maskMissing:
        paidDividendSum = Matrices.allMasked(len(subIssues))
        for (idx, sid) in enumerate(subIssues):
            divData = paidDividends.get(sid, list())
            if len(divData) > 0:
                paidDividendSum[idx] = ma.sum([dv[1] for dv in divData])
    else:
        paidDividendSum = numpy.array([ma.sum([i[1] for i in
            paidDividends.get(sid, list())])
            for sid in subIssues])
    
    return paidDividendSum

def generate_proxied_dividend_payout(modelDate, data, model, 
                            modelDB, marketDB, daysBack=(2*365),
                            restrict=None, useQuarterlyData=True,
                            maskZero=False, includeStock=False):
    """Compute the dividend payout for the given universe.
    The value is computed based on paid dividends for the last year,
    annual income. Paid dividends are computed as reported dividends
    times the shares outstanding on the pay date.
    Annual income is taken as the most recent filing in the 'daysBack'
    period. If useQuarterlyData is true, then the annualized value taken
    from the quarterly filings is used. Otherwise the most recent annual
    filing is used.
    Returns an array with the dividend payout value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    """
    myLog().info('generate_proxied_dividend_payout: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'ibei_qtr'
    else:
        fieldName = 'ibei_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    income = modelDB.getFundamentalCurrencyItem(
        fieldName, startDate, modelDate, subIssues, modelDate,
        marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        income = modelDB.annualizeQuarterlyValues(income)
    else:
        income = Utilities.extractLatestValue(income)
    paidDividends = Utilities.screen_data(computeAnnualDividends(
                        modelDate, subIssues, model, modelDB, marketDB,
                        includeStock=includeStock))
    divPayout = paidDividends / (ma.masked_where(income==0.0, income) * 1e6)
    divPayout = ma.where(divPayout > 1.0, 1.0, divPayout)
    divPayout = ma.where(divPayout < 0.0, 0.0, divPayout)
    divPayout = ma.where((paidDividends < 0.0) * (income == 0.0), 0.0,
                         divPayout)
    divPayout = ma.where((paidDividends > 0.0) * (income == 0.0), 1.0,
                         divPayout)
    missing = numpy.flatnonzero(ma.getmaskarray(divPayout))
    tmpData = ma.filled(divPayout, 1.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpData==0.0, tmpData)))
    if len(missing) > 0:
        myLog().info('%d assets are missing dividend payout information' % (len(missing)))
        myLog().debug('missing Axioma IDs: ' + ', '.join([subIssues[i].getSubIDString() for i in missing]))
    if len(zeroVals) > 0:
        myLog().info('%d assets have zero dividend payout information' % (len(zeroVals)))
        if maskZero:
            divPayout = ma.masked_where(divPayout==0.0, divPayout)
    if restrict is not None:
        div = Matrices.allMasked(len(data.universe))
        ma.put(div, restrict, divPayout)
        for (i,val) in enumerate(divPayout):
            if val is ma.masked:
                div[restrict[i]] = ma.masked
        divPayout = div
    myLog().info('generate_proxied_dividend_payout: end')
    return divPayout

def generate_dividend_payout(modelDate, data, model, modelDB, marketDB,
                             daysBack=(2*365), useQuarterlyData=True,
                             maskZero=False):
    """Compute the dividend payout for the given assets.
    The value is computed based on dividends per share (dps)
    and earnings per share (eps): dps/eps
    Both per share values are taken from the quarterly filings if
    useQuarterlyData is true, otherwise from the annual filings.
    Returns an array with the dividend payout value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_dividend_payout: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        dps = modelDB.getFundamentalCurrencyItem(
            'dps_qtr', startDate, modelDate, data.universe, modelDate,
            marketDB, convertTo=model.numeraire.currency_id,
            splitAdjust='divide')
        dps = modelDB.annualizeQuarterlyValues(dps)
        eps = modelDB.getFundamentalCurrencyItem(
            'epsxei_qtr', startDate, modelDate, data.universe, modelDate,
            marketDB, convertTo=model.numeraire.currency_id,
            splitAdjust='divide')
        eps = modelDB.annualizeQuarterlyValues(eps)
    else:
        dps = modelDB.getFundamentalCurrencyItem(
            'dps_ann', startDate, modelDate, data.universe, modelDate,
            marketDB, convertTo=model.numeraire.currency_id,
            splitAdjust='divide')
        dps = Utilities.extractLatestValue(dps)
        eps = modelDB.getFundamentalCurrencyItem(
            'epsxei_ann', startDate, modelDate, data.universe, modelDate,
            marketDB, convertTo=model.numeraire.currency_id,
            splitAdjust='divide')
        eps = Utilities.extractLatestValue(eps)
        
    divPayout = dps / eps
    divPayout = ma.where(divPayout > 1.0, 1.0, divPayout)
    divPayout = ma.where(divPayout < 0.0, 0.0, divPayout)
    divPayout = ma.where((dps < 0.0) * (eps == 0.0), 0.0, divPayout)
    divPayout = ma.where((dps > 0.0) * (eps == 0.0), 1.0, divPayout)
    missing = numpy.flatnonzero(ma.getmaskarray(divPayout))
    tmpData = ma.filled(divPayout, 1.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpData==0.0, tmpData)))
    if len(missing) > 0:
        myLog().info('%d assets are missing dividend payout information' % (len(missing)))
        myLog().debug('missing Axioma IDs: ' + ', '.join([data.universe[i].getSubIDString() for i in missing]))
    if len(zeroVals) > 0:
        myLog().info('%d assets have zero dividend payout information' % (len(zeroVals)))
        if maskZero:
            divPayout = ma.masked_where(divPayout==0.0, divPayout)

    myLog().info('generate_dividend_payout: end')
    return divPayout

def generate_return_on_equity(modelDate, data, model, modelDB, marketDB,
                              daysBackCE=(3*365), daysBackInc=(2*365),
                              restrict=None, useQuarterlyData=True, legacy=False):
    """Compute the return-on-equity for the given assets.
    The value is computed as the ratio of income and common equity.
    Common equity and income are taken from the quarterly filings
    if useQuarterlyData is true, otherwise from the annual filings.
    Returns an array with the return-on-equity value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past three years (common 
    equity) and two years (income) of data are used.
    """
    myLog().info('generate_return_on_equity: begin')
    startDateCE = modelDate - datetime.timedelta(daysBackCE)
    endDateCE = modelDate - datetime.timedelta(365)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    if useQuarterlyData:
        fieldName0 = 'ce_qtr'
        fieldName1 = 'ibei_qtr'
    else:
        fieldName0 = 'ce_ann'
        fieldName1 = 'ibei_ann'
    ce = modelDB.getFundamentalCurrencyItem(fieldName0, 
            startDateCE, endDateCE, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    ce = Utilities.extractLatestValue(ce)
    startDate = modelDate - datetime.timedelta(daysBackInc)
    income = modelDB.getFundamentalCurrencyItem(fieldName1, 
            startDate, modelDate, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        income = modelDB.annualizeQuarterlyValues(income)
    else:
        income = Utilities.extractLatestValue(income)
    roe = income / ce
    if not legacy:
        roe = ma.where(ce <= 0.0, 0.0, roe)
    missing = numpy.flatnonzero(ma.getmaskarray(roe))
    if len(missing) > 0:
        myLog().info('%d assets are missing return-on-equity information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([data.universe[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, roe)
        for (i,j) in enumerate(roe):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        roe = val
    myLog().info('generate_return_on_equity: end')
    return roe

def generate_return_on_equity_v2(modelDate, data, model, modelDB, marketDB,
                              daysBackCE=(3*365), daysBackInc=(2*365),
                              restrict=None, useQuarterlyData=True, legacy=False):
    """Compute the return-on-equity for the given assets.
    The value is computed as the ratio of income and common equity.
    Common equity and income are taken from the quarterly filings
    if useQuarterlyData is true, otherwise from the annual filings.
    Returns an array with the return-on-equity value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past three years (common 
    equity) and two years (income) of data are used.
    """
    myLog().info('generate_return_on_equity_v2: begin')
    startDateCE = modelDate - datetime.timedelta(daysBackCE)
    endDateCE = modelDate - datetime.timedelta(365)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    if useQuarterlyData:
        fieldName0 = 'ce_qtr'
        fieldName1 = 'ibei_qtr'
    else:
        fieldName0 = 'ce_ann'
        fieldName1 = 'ibei_ann'
    ce = modelDB.getFundamentalCurrencyItem(fieldName0, 
            startDateCE, endDateCE, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    startDate = modelDate - datetime.timedelta(daysBackInc)
    income = modelDB.getFundamentalCurrencyItem(fieldName1, 
            startDate, modelDate, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        income = modelDB.annualizeQuarterlyValues(income)
        ce = Utilities.extractLatestValue(ce)
    else:
        income = Utilities.extractLatestValueAndDate(income)
        incomevals = ma.masked_all(len(subIssues), float)
        cevals = ma.masked_all(len(subIssues), float)
        ceDates = ma.masked_all(len(subIssues), datetime.date)
        for i, vals in enumerate(ce):
            if len(vals) == 0 or ma.is_masked(income[i]):
                continue
            incomevals[i] = income[i][1]
            absdist = numpy.array([abs((income[i][0]-v[0]).days-365) 
                                   for v in vals])
            idx = absdist.argmin() if -40 < absdist.min() < 40 else None
            if idx is not None:
                cevals[i] = vals[idx][1]
                ceDates[i] = vals[idx][0]
        income = incomevals
        ce = cevals

    roe = income / ce
    if not legacy:
        roe = ma.where(ce <= 0.0, 0.0, roe)
    missing = numpy.flatnonzero(ma.getmaskarray(roe))
    if len(missing) > 0:
        myLog().info('%d assets are missing return-on-equity information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([data.universe[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, roe)
        for (i,j) in enumerate(roe):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        roe = val
    myLog().info('generate_return_on_equity_v2: end')
    return roe

def generate_return_on_assets(modelDate, data, model, modelDB, marketDB,
                              daysBackTA=(2*365), daysBackInc=(2*365),
                              restrict=None, useQuarterlyData=True, legacy=False):
    """Compute the return-on-assets for the given assets.
    The value is computed as the ratio of income and total assets.
    Total assets and income are taken from the quarterly filings
    if useQuarterlyData is true, otherwise from the annual filings.
    Returns an array with the return-on-assets value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years are used.
    """
    myLog().info('generate_return_on_assets: begin')
    startDateTA = modelDate - datetime.timedelta(daysBackTA)
    endDateTA = modelDate - datetime.timedelta(365)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    if useQuarterlyData:
        fieldName0 = 'at_qtr'
        fieldName1 = 'ibei_qtr'
    else:
        fieldName0 = 'at_ann'
        fieldName1 = 'ibei_ann'
    ta = modelDB.getFundamentalCurrencyItem(fieldName0, 
            startDateTA, endDateTA, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    totalAssets = Utilities.extractLatestValue(ta)
    totalAssets = ma.masked_where(totalAssets < 0.0, totalAssets)
    startDate = modelDate - datetime.timedelta(daysBackInc)
    income = modelDB.getFundamentalCurrencyItem(fieldName1, 
            startDate, modelDate, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        income = modelDB.annualizeQuarterlyValues(income)
    else:
        income = Utilities.extractLatestValue(income)
    roa = income / totalAssets
    if not legacy:
        roa = ma.where(totalAssets <= 0.0, 0.0, roa)
    missing = numpy.flatnonzero(ma.getmaskarray(roa))
    if len(missing) > 0:
        myLog().info('%d assets are missing return-on-assets information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([data.universe[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, roa)
        for (i,j) in enumerate(roa):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        roa = val
    myLog().info('generate_return_on_assets: end')
    return roa

def generate_sales_growth(modelDate, data, model, modelDB, marketDB,
                          daysBack=(5*365), restrict=None, 
                          useQuarterlyData=True):
    """Compute growth in sales for the assets in data.universe by regressing
    sales against time plus an intercept term.  Growth is then computed as 
    regression slope coefficient divided by the average absolute sales.
    Assets with fewer than 2 observations in the past 5 years are excluded.
    Returns an array with the regression coefficients for each asset.
    By default, the latest values from the past 5 years of data are used.
    """
    myLog().info('generate_sales_growth: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'sale_qtr'
    else:
        fieldName = 'sale_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    sale = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    salesGrowth = Matrices.allMasked(len(subIssues))
    for i in range(len(subIssues)):
        saleArray = numpy.array([n[1] for n in sale[i]])
        if len(saleArray) == 0:
            continue
        elif len(numpy.unique(saleArray)) <= 2:
            salesGrowth[i] = 0.0
            continue
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        saleArray, numpy.transpose(numpy.array(
                        [list(range(1, len(saleArray)+1)), numpy.ones(len(saleArray))])))
        salesGrowth[i] = coef[0] / numpy.average(abs(saleArray))
    missing = numpy.flatnonzero(ma.getmaskarray(salesGrowth))
    if len(missing) > 0:
        myLog().info('%d assets are missing sales information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, salesGrowth)
        for (i,j) in enumerate(salesGrowth):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        salesGrowth = val
    myLog().info('generate_sales_growth: end')
    return salesGrowth

def generate_sales_growth_v2(modelDate, data, model, modelDB, marketDB,
                          daysBack=int((5+10./12)*366), restrict=None, 
                          useQuarterlyData=True, debug=False,
                          legacy=False):
    """Compute growth in sales for the assets in data.universe by regressing
    sales against time plus an intercept term.  Growth is then computed as 
    regression slope coefficient divided by the average absolute sales.
    Assets with fewer than 2 observations in the past 5 years are excluded.
    Returns an array with the regression coefficients for each asset.
    By default, the latest values from the past 5 years of data are used.
    WARNING: THIS FUNCTION SHOULD NOT BE USED WITH QUARTERLY DATA
    SINCE QUARTERLY DATA OFTEN EXHIBITS SEASONALITY
    """
    myLog().info('generate_sales_growth_v2: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'sale_qtr'
    else:
        fieldName = 'sale_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    sale = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    salesGrowth = Matrices.allMasked(len(subIssues))

    if debug:
        regData = {}
    for i in range(len(subIssues)):
        saleDict = dict([n[:2] for n in sale[i]])
        if len(saleDict) == 0:
            continue
        # backwards compatibility - if there are <=2 unique values
        # in the last 5 years, continue
        tmpDateArray = [n[0] for n in sale[i] if (modelDate - n[0]).days/365. <= 5.]
        tmpSaleArray = numpy.array([saleDict[dt] for dt in tmpDateArray])
        if len(numpy.unique(tmpSaleArray)) <= 2:
            if legacy:
                salesGrowth[i] = 0.0
            continue
        if useQuarterlyData:
            dateArray = [n[0] for n in sale[i]
                         if (sale[i][-1][0] - n[0]).days/365. <= 4.8]
        else:
            dateArray = [n[0] for n in sale[i] 
                         if (sale[i][-1][0] - n[0]).days/365. <= 4.05]
        saleArray = numpy.array([saleDict[dt] for dt in dateArray])
        if len(numpy.unique(saleArray)) <= 2:
            if legacy:
                salesGrowth[i] = 0.0
            continue
        x_axis = numpy.cumsum([(dateArray[n] - dateArray[n-1]).days 
                               if n >= 1 else 365 
                               for n in range(0, len(dateArray))])/365.0
        if legacy:
            x_axis = numpy.array(list(range(1, len(saleArray)+1)))
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        saleArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(saleArray))])))
        salesGrowth[i] = coef[0] / numpy.average(abs(saleArray))
        if debug:
            regData[subIssues[i].getSubIdString()] = \
                    {'numObs': len(saleArray),
                     'maxDt': max(dateArray),
                     'minDt': min(dateArray)}
    if debug:
        regDataDF = pandas.DataFrame.from_dict(regData, orient='index')
        regDataDF.to_csv('tmp/regData-generate_sales_growth-%s.csv' % \
                modelDate.isoformat())
        targetN = 20 if useQuarterlyData else 5
        myLog().debug('%.3f of sales growth values estimated from %.0f obs' % \
                ((regDataDF.numObs == targetN).sum()/float(len(regDataDF))*100., targetN))
        myLog().debug('%.3f of sales growth values estimated from <%.0f obs' % \
                ((regDataDF.numObs < targetN).sum()/float(len(regDataDF))*100., targetN))
        myLog().debug('%.3f of sales growth values estimated from >%.0f obs' % \
                ((regDataDF.numObs > targetN).sum()/float(len(regDataDF))*100., targetN))
    missing = numpy.flatnonzero(ma.getmaskarray(salesGrowth))
    if len(missing) > 0:
        myLog().info('%d assets are missing sales information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, salesGrowth)
        for (i,j) in enumerate(salesGrowth):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        salesGrowth = val
    myLog().info('generate_sales_growth_v2: end')
    return salesGrowth

def generate_est_sales_growth(modelDate, data, model, modelDB, marketDB,
                          daysBack=(5*365), restrict=None, 
                          useQuarterlyData=True, winsoriseRaw=False,
                          legacy=False):
    """Compute growth in sales for the assets in data.universe by regressing
    sales against time plus an intercept term.  Growth is then computed as 
    regression slope coefficient divided by the average absolute sales.
    Assets with fewer than 2 observations in the past 5 years are excluded.
    Returns an array with the regression coefficients for each asset.
    If estimated sales numbers are available they are included as 
    additional datapoints in the regression.  
    By default, the latest values from the past 5 years of data are used.
    """
    myLog().info('generate_est_sales_growth: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'sale_qtr'
    else:
        fieldName = 'sale_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    
    # Load sales
    sale = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    
    # Load estimated sales
    estSaleRaw = modelDB.getIBESCurrencyItemLegacy('rev_median_ann',
            modelDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    estSale = Utilities.extractLatestValueAndDate(estSaleRaw)
    estSaleMissing = numpy.flatnonzero(ma.getmaskarray(estSale))
    if len(estSaleMissing) > 0:
        pct = 100.0 * len(estSaleMissing) / float(len(subIssues))
        logging.info('%d out of %d (%.2f%%) assets are missing forecast sales',
                len(estSaleMissing), len(subIssues), pct)
    salesGrowth = Matrices.allMasked(len(subIssues))
    
    for i in range(len(subIssues)):
        missing = True
        saleArray = ma.filled(ma.array([n[1] for n in sale[i]]), 0.0)
        if len(saleArray) == 0:
            continue
        elif len(numpy.unique(saleArray)) <= 2:
            salesGrowth[i] = 0.0
            continue
        elif i not in estSaleMissing and sale[i] != []:
            missing = False
            estSaleDate = estSale[i][0]
            latestSaleDate = sale[i][-1][0]
            if estSaleDate > latestSaleDate: 
                if useQuarterlyData:
                    if estSaleDate - latestSaleDate <= datetime.timedelta(91):
                        val = estSale[i][1]/4.0
                        saleArray = numpy.append(saleArray, val)
                    elif datetime.timedelta(92) <= estSaleDate - latestSaleDate <= datetime.timedelta(182):
                        val = estSale[i][1]/4.0
                        saleArray = numpy.append(saleArray,[val]*2)
                    elif datetime.timedelta(183) <= estSaleDate - latestSaleDate <= datetime.timedelta(266):
                        val = estSale[i][1]/4.0
                        saleArray = numpy.append(saleArray,[val]*3)
                    elif datetime.timedelta(267) <= estSaleDate - latestSaleDate <= datetime.timedelta(365):
                        val = estSale[i][1]/4.0
                        saleArray = numpy.append(saleArray,[val]*4)
                else:
                    saleArray = numpy.append(saleArray, estSale[i][1]) 
        if legacy and missing:
            continue
        if (not legacy) and winsoriseRaw:
            saleArray = Utilities.twodMAD(saleArray, nDev=[3.0, 3.0], axis=0,
                    suppressOutput=True)
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        saleArray, numpy.transpose(numpy.array(
                        [list(range(1, len(saleArray)+1)), numpy.ones(len(saleArray))])))
        denom = ma.filled(ma.average(abs(saleArray),axis=None),0.0)
        if denom != 0.0:
            salesGrowth[i] = coef[0] / denom
    missing = numpy.flatnonzero(ma.getmaskarray(salesGrowth))
    if len(missing) > 0:
        myLog().info('%d assets are missing sales information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, salesGrowth)
        for (i,j) in enumerate(salesGrowth):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        salesGrowth = val
    myLog().info('generate_est_sales_growth: end')
    return salesGrowth

def generate_earnings_growth(modelDate, data, model, modelDB, marketDB,
                             daysBack=(5*365), restrict=None, 
                             useQuarterlyData=True):
    """Compute growth in earnings for the assets in data.universe by regressing
    earnings against time plus an intercept term.  Growth is then computed as 
    regression slope coefficient divided by the average absolute earnings.
    Assets with fewer than 2 observations in the past 5 years are excluded.
    Returns an array with the regression coefficients for each asset.
    By default, the latest values from the past 5 years of data are used.
    """
    myLog().info('generate_earnings_growth: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'ibei_qtr'
    else:
        fieldName = 'ibei_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    earn = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    earningsGrowth = Matrices.allMasked(len(subIssues))
    for i in range(len(subIssues)):
        earnArray = numpy.array([n[1] for n in earn[i]])
        if len(earnArray) == 0:
            continue
        elif len(numpy.unique(earnArray)) <= 2:
            earningsGrowth[i] = 0.0
            continue
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        earnArray, numpy.transpose(numpy.array(
                        [list(range(1, len(earnArray)+1)), numpy.ones(len(earnArray))])))
        earningsGrowth[i] = coef[0] / numpy.average(abs(earnArray))
    missing = numpy.flatnonzero(ma.getmaskarray(earningsGrowth))
    if len(missing) > 0:
        myLog().info('%d assets are missing earnings information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, earningsGrowth)
        for (i,j) in enumerate(earningsGrowth):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        earningsGrowth = val
    myLog().info('generate_earnings_growth: end')
    return earningsGrowth

def generate_earnings_growth_v2(modelDate, data, model, modelDB, marketDB,
                             daysBack=int((5+10./12)*366), restrict=None, 
                             useQuarterlyData=True, debug=False,
                             legacy=False):
    """Compute growth in earnings for the assets in data.universe by regressing
    earnings against time plus an intercept term.  Growth is then computed as 
    regression slope coefficient divided by the average absolute earnings.
    Assets with fewer than 2 observations in the past 5 years are excluded.
    Returns an array with the regression coefficients for each asset.
    By default, the latest values from the past 5 years of data are used.
    WARNING: THIS FUNCTION SHOULD NOT BE USED WITH QUARTERLY DATA
    SINCE QUARTERLY DATA OFTEN EXHIBITS SEASONALITY
    """
    myLog().info('generate_earnings_growth_v2: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'ibei_qtr'
    else:
        fieldName = 'ibei_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    earn = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    earningsGrowth = Matrices.allMasked(len(subIssues))
    if debug:
        regData = {}
    for i in range(len(subIssues)):
        earnDict = dict([n[:2] for n in earn[i]])
        if len(earnDict) == 0:
            continue
        # backwards compatibility - if there are <=2 unique values
        # in the last 5 years, continue
        tmpDateArray = [n[0] for n in earn[i] if (modelDate - n[0]).days/365. <= 5.]
        tmpEarnArray = numpy.array([earnDict[dt] for dt in tmpDateArray])
        if len(numpy.unique(tmpEarnArray)) <= 2:
            if legacy:
                earningsGrowth[i] = 0.
            continue
        if useQuarterlyData:
            dateArray = [n[0] for n in earn[i]
                         if (earn[i][-1][0] - n[0]).days/365. <= 4.8]
        else:
            dateArray = [n[0] for n in earn[i]
                         if (earn[i][-1][0] - n[0]).days/365. <= 4.05]
        earnArray = numpy.array([earnDict[dt] for dt in dateArray])
        if len(numpy.unique(earnArray)) <= 2:
            if legacy:
                earningsGrowth[i] = 0.0
            continue
        x_axis = numpy.cumsum([(dateArray[n] - dateArray[n-1]).days
                               if n >= 1 else 365
                               for n in range(0, len(dateArray))])/365.0
        if legacy:
            x_axis = numpy.array(list(range(1, len(earnArray)+1)))
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        earnArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(earnArray))])))
        earningsGrowth[i] = coef[0] / numpy.average(abs(earnArray))
        if debug:
            regData[subIssues[i].getSubIdString()] = \
                    {'numObs': len(earnArray),
                     'maxDt': max(dateArray),
                     'minDt': min(dateArray)}
    if debug:
        regDataDF = pandas.DataFrame.from_dict(regData, orient='index')
        regDataDF.to_csv('tmp/regData-generate_earnings_growth-%s.csv' % \
                modelDate.isoformat())
        targetN = 20 if useQuarterlyData else 5
        myLog().debug('%.3f of earn growth values estimated from %.0f obs' % \
                ((regDataDF.numObs == targetN).sum()/float(len(regDataDF))*100., targetN))
        myLog().debug('%.3f of earn growth values estimated from <%.0f obs' % \
                ((regDataDF.numObs < targetN).sum()/float(len(regDataDF))*100., targetN))
        myLog().debug('%.3f of earn growth values estimated from >%.0f obs' % \
                ((regDataDF.numObs > targetN).sum()/float(len(regDataDF))*100., targetN))
    missing = numpy.flatnonzero(ma.getmaskarray(earningsGrowth))
    if len(missing) > 0:
        myLog().info('%d assets are missing earnings information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, earningsGrowth)
        for (i,j) in enumerate(earningsGrowth):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        earningsGrowth = val
    myLog().info('generate_earnings_growth_v2: end')
    return earningsGrowth

def generate_est_earnings_growth(modelDate, data, model, modelDB, marketDB,
                             daysBack=(5*365), restrict=None, 
                             useQuarterlyData=True, winsoriseRaw=False,
                             legacy=False):
    """Compute growth in earnings for the assets in data.universe by regressing
    earnings against time plus an intercept term.  Growth is then computed as 
    regression slope coefficient divided by the average absolute earnings.
    Assets with fewer than 2 observations in the past 5 years are excluded.
    Returns an array with the regression coefficients for each asset.
    If estimated sales numbers are available they are included as 
    additional datapoints in the regression.  
    By default, the latest values from the past 5 years of data are used.
    """
    myLog().info('generate_est_earnings_growth: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if useQuarterlyData:
        fieldName = 'ibei_qtr'
    else:
        fieldName = 'ibei_ann'
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]

    # Load earnings
    earn = modelDB.getFundamentalCurrencyItem(fieldName, 
            startDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    # Load estimated earnings
    estEPSRaw = modelDB.getIBESCurrencyItemLegacy('eps_median_ann',
            modelDate, modelDate, subIssues, modelDate, 
            marketDB, convertTo=model.numeraire.currency_id)
    tso = modelDB.loadTSOHistory([modelDate], subIssues)
    tso.data = ma.filled(tso.data, 0.0)
    estEPS = Utilities.extractLatestValueAndDate(estEPSRaw)
    estEPSMissing = numpy.flatnonzero(ma.getmaskarray(estEPS))
    if len(estEPSMissing) > 0:
        pct = 100.0 * len(estEPSMissing) / float(len(subIssues))
        logging.info('%d out of %d (%.2f%%) assets are missing forecast EPS',
                len(estEPSMissing), len(subIssues), pct)
    earningsGrowth = Matrices.allMasked(len(subIssues))
     
    for i in range(len(subIssues)):
        earnArray = ma.filled(ma.array([n[1] for n in earn[i]]), 0.0)
        missing = True
        if len(earnArray) == 0:
            continue
        elif len(numpy.unique(earnArray)) <= 2:
            earningsGrowth[i] = 0.0
            continue
        elif i not in estEPSMissing and earn[i] != []:
            missing = False
            estEarnDate = estEPS[i][0]
            latestEarnDate = earn[i][-1][0]
            if estEarnDate > latestEarnDate: 
                if useQuarterlyData:
                    if estEarnDate - latestEarnDate <= datetime.timedelta(91):
                        estEarn = estEPS[i][1]*tso.data[i, 0]/(4.0*1.0e6)
                        earnArray = numpy.append(earnArray, estEarn)
                    elif datetime.timedelta(92) <= estEarnDate - latestEarnDate <= datetime.timedelta(182):
                        estEarn = estEPS[i][1]*tso.data[i, 0]/(4.0*1.0e6)
                        earnArray = numpy.append(earnArray,[estEarn]*2)
                    elif datetime.timedelta(183) <= estEarnDate - latestEarnDate <= datetime.timedelta(266):
                        estEarn = estEPS[i][1]*tso.data[i, 0]/(4.0*1.0e6)
                        earnArray = numpy.append(earnArray,[estEarn]*3)
                    elif datetime.timedelta(267) <= estEarnDate - latestEarnDate <= datetime.timedelta(365):
                        estEarn = estEPS[i][1]*tso.data[i, 0]/(4.0*1.0e6)
                        earnArray = numpy.append(earnArray,[estEarn]*4)
                else:
                    estEarn = estEPS[i][1]*tso.data[i,0] / 1.0e6
                    earnArray = numpy.append(earnArray, estEarn) 

        if legacy and missing:
            continue
        if winsoriseRaw:
            earnArray = Utilities.twodMAD(earnArray, nDev=[3.0, 3.0], axis=0,
                    suppressOutput=True)
        (coef, resid) = Utilities.ordinaryLeastSquares(
                        earnArray, numpy.transpose(numpy.array(
                        [list(range(1, len(earnArray)+1)), numpy.ones(len(earnArray))])))
        denom = ma.filled(ma.average(abs(earnArray), axis=None), 0.0)
        if denom != 0.0:
            earningsGrowth[i] = coef[0] / denom
    missing = numpy.flatnonzero(ma.getmaskarray(earningsGrowth))
    if len(missing) > 0:
        myLog().info('%d assets are missing earnings information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, earningsGrowth)
        for (i,j) in enumerate(earningsGrowth):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        earningsGrowth = val
    myLog().info('generate_est_earnings_growth: end')
    return earningsGrowth    

def test_ibes_data(modelDate, data, model, modelDB, marketDB,
        field, daysBack=365, useQuarterlyData=True):
    """Short piece of code to test the IBES data - see how the forecasts
    measure up to the reality
    """
    myLog().info('test_ibes_data (%s): begin', field)
    endDate = datetime.date(modelDate.year, 12, 31)
    effDate = endDate + datetime.timedelta(daysBack)
    subIssues = data.universe

    if field == 'earnings':
        if useQuarterlyData:
            fieldName = 'ibei_qtr'
        else:
            fieldName = 'ibei_ann'
        estFieldName = 'eps_median_ann'
    else:
        if useQuarterlyData:
            fieldName = 'sale_qtr'
        else:
            fieldName = 'sale_ann'
        estFieldName = 'rev_median_ann'

    # Load estimated earnings
    startDate = datetime.date(modelDate.year, 1, 1)
    estEPSRaw = modelDB.getIBESCurrencyItemLegacy(estFieldName,
            startDate, endDate, subIssues, effDate,
            marketDB, convertTo=model.numeraire.currency_id,
            getEarliest=True, maxEndDate=endDate)

    # Take latest estimate
    tmpEst = []
    for estE in estEPSRaw:
        if len(estE) > 0:
            eVals = [ev for ev in estE if ev[0] >= startDate]
        else:
            eVals = estE
        tmpEst.append(eVals)
    estEPSRaw = tmpEst
    estEPSRaw = Utilities.extractLatestValuesAndDate(estEPSRaw)
    estEPSMissing = numpy.flatnonzero(ma.getmaskarray(estEPSRaw))

    # Report on coverage
    if len(estEPSMissing) > 0:
        pct = 100.0 * len(estEPSMissing) / float(len(subIssues))
        pctCap = ma.take(data.marketCaps, estEPSMissing, axis=0)
        pctCap = 100.0 * ma.sum(pctCap, axis=None) / ma.sum(data.marketCaps, axis=None)
        logging.info('%s, IBES Coverage: missing %s, (N: %.2f%%, MCap: %.2f%%)',
                modelDate, estFieldName, pct, pctCap)
        #estuMissingIdx = list(set(estEPSMissing).intersection(set(data.estimationUniverseIdx)))
        #if len(estuMissingIdx) > 0:
        #    pct = 100.0 * len(estuMissingIdx) / float(len(data.estimationUniverseIdx))
        #    pctCap = ma.take(data.marketCaps, estuMissingIdx, axis=0)
        #    estuCap = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        #    pctCap = 100.0 * ma.sum(pctCap, axis=None) / ma.sum(estuCap, axis=None)
        #    logging.info('%s, IBES Coverage ESTU: missing %s, (N: %.2f%%, MCap: %.2f%%)',
        #            modelDate, estFieldName, pct, pctCap)
    # Load earnings
    startDate = datetime.date(modelDate.year-1, 1, 1)
    earn = modelDB.getFundamentalCurrencyItem(fieldName,
            startDate, endDate, subIssues, effDate,
            marketDB, convertTo=model.numeraire.currency_id)

    # Throw out realised earnings later than forecast date
    tmpEarn = []
    for (sid, estE, realE) in zip(subIssues, estEPSRaw, earn):
        #logging.info('XXX: %s, %s, %s', sid, estE, realE)
        if realE is ma.masked or len(realE) < 1:
            eVals = []
        elif estE is ma.masked or len(estE) < 1:
            eVals = realE
        else:
            eVals = [ev for ev in realE if ev[0] <= estE[0]]
        tmpEarn.append(eVals)
    earn = tmpEarn

    # Extract latest available annual value
    if useQuarterlyData:
        earnAnn = modelDB.annualizeQuarterlyValues(earn)
    else:
        earnAnn = Utilities.extractLatestValue(earn)

    earnDates = []
    estDates = []
    estEPSLatest = []
    earnLatest = []
    sids = []
    estu = []

    # Collect together all "good" values - i.e. where dates sufficiently
    # close together and nothing missing
    for i in range(len(subIssues)):
        missing = False
        if len(earn[i]) < 1:
            missing = True
        if (i in estEPSMissing) or (len(estEPSRaw[i]) < 1):
            missing = True

        if not missing:
            estEarnDate = estEPSRaw[i][0]
            latestEarnDate = earn[i][-1][0]
            if (estEarnDate >= latestEarnDate) and (estEarnDate - latestEarnDate <= datetime.timedelta(30)):
                eu = 1
            else:
                eu = 0
            if field == 'earnings':
                tso = modelDB.loadTSOHistory([estEPSRaw[i][-1]], [subIssues[i]])
                tso = ma.filled(tso.data, 0.0) / 1.0e6
                estEPS = estEPSRaw[i][1] * tso[0,0]
            else:
                estEPS = estEPSRaw[i][1]
            if estEPS != 0.0:
                estEPSLatest.append(estEPS)
                sids.append(subIssues[i])
                estDates.append(estEarnDate)
                earnDates.append(latestEarnDate)
                earnLatest.append(earnAnn[i])
                estu.append(eu)

    if len(earnDates) < 1:
        logging.warning('Not enough valid data for IBES %s test for %d',
                field, modelDate.year)
        return

    # Output underlying data 
    mp = int(len(earnDates)/2)
    rn = [sid.getSubIDString() + ':' + str(dt1) + ':' + str(dt2) \
            for (sid, dt1, dt2) in zip(sids, earnDates, estDates)]
    cn = ['Realised','Forecast','eligible','RMG_ID']
    assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
            data.rmgAssetMap.items() for sid in ids])
    rmgIDList = [assetRMGMap[sid] for sid in sids]
    Utilities.writeToCSV(numpy.transpose(numpy.array([earnLatest, estEPSLatest, estu, rmgIDList], float)),
            'tmp/ibes_%s_%d.csv' % (field, earnDates[mp].year), rowNames=rn, columnNames=cn)

    estuIdx = numpy.flatnonzero(numpy.array(estu, int))
    earnLatest = numpy.array(earnLatest, float)
    earnLatest = numpy.take(earnLatest, estuIdx, axis=0)
    estEPSLatest = numpy.array(estEPSLatest, float)
    estEPSLatest = numpy.take(estEPSLatest, estuIdx, axis=0)
    # Perform regression
    if len(earnLatest) > 0:
        res = Utilities.robustLinearSolver(earnLatest, estEPSLatest, robust=False, k=5.0)
        logging.info('IBES TEST, %s, %s, n, %d, BETA, %s, R-Square, %s',
                earnDates[mp], field, len(earnLatest), res.params[0], res.rsquare[0])
    myLog().info('test_ibes_data (%s): begin', field)
    return 

def generate_cross_sectional_volatility(returns, indices=None, daysBack=60):
    """The cross-sectional volatility is the square-root of the average
    of the weighted absolute asset returns for each asset over the
    last 60 (default) days. The asset returns of each days are 
    weighted by the standard deviation of all returns on that day.
    If indices is specified, the std dev is taken across that 
    subset of assets only.
    """
    myLog().info('generate_cross_sectional_volatility: begin')
    # Restrict universe for std dev
    if not indices:
        indices = list(range(len(returns.assets)))
    returns = returns.data[:, -daysBack:]
    goodReturns = ma.take(returns, indices, axis=0)

    # Remove assets with lots of missing/zero returns
    bad = ma.masked_where(goodReturns==0.0, goodReturns)
    io = numpy.sum(ma.getmaskarray(bad), axis=1)
    goodAssetsIdx = numpy.flatnonzero(io < 0.5 * daysBack)
    logging.info('%d out of %d assets have sufficient returns',
            len(goodAssetsIdx), goodReturns.shape[0])
    goodReturns = ma.take(goodReturns, goodAssetsIdx, axis=0)
    crossSectionalStdDev = Utilities.mlab_std(goodReturns, axis=0)
    crossSectionalStdDev = ma.masked_where(crossSectionalStdDev==0.0, crossSectionalStdDev)

    # If tiny markets that trade on irregular days are involved,
    # replace std dev on weekends and other days with sparse trading
    # with previous value
    bad = ma.masked_where(goodReturns==0.0, goodReturns)
    io = numpy.sum(ma.getmaskarray(bad), axis=0)
    badDatesIdx = numpy.flatnonzero(io > 0.9 * goodReturns.shape[0])
    logging.info('%d out of %d bad dates (fewer than 90%% assets trading)',
            len(badDatesIdx), goodReturns.shape[1])
    if len(badDatesIdx) < goodReturns.shape[1]:
        for i in badDatesIdx:
            if i==0:
                prevStdDev = ma.average(ma.take(crossSectionalStdDev, 
                    [j for j in range(goodReturns.shape[1]) if j not in badDatesIdx]))
            else:
                prevStdDev = crossSectionalStdDev[i-1]
            if prevStdDev is not ma.masked:
                crossSectionalStdDev[i] = prevStdDev
    crossSectionalStdDev = Utilities.screen_data(crossSectionalStdDev)
    crossSectionalStdDev = ma.masked_where(crossSectionalStdDev==0.0, crossSectionalStdDev)
    csv = ma.average(abs(returns) / crossSectionalStdDev, axis=1)
    csv = ma.sqrt(csv)
    myLog().info('generate_cross_sectional_volatility: end')
    return csv

def generate_cross_sectional_volatility_v3(
                returns, params, indices=None, clippedReturns=None):
    """The cross-sectional volatility is the square-root of the average
    of the weighted absolute asset returns for each asset over the
    last 60 (default) days. The asset returns of each days are
    weighted by the standard deviation of all returns on that day.
    If indices is specified, the std dev is taken across that
    subset of assets only.
    """
    myLog().info('generate_cross_sectional_volatility: begin')
    # Set up parameters
    daysBack = params.daysBack

    # Restrict universe for std dev
    if not indices:
        indices = list(range(len(returns.assets)))
    returns = returns.data[:, -daysBack:]
    if clippedReturns is None:
        goodReturns = ma.take(returns, indices, axis=0)
    else:
        goodReturns = ma.take(clippedReturns[:, -daysBack:], indices, axis=0)

    # Remove assets with lots of missing/zero returns
    bad = ma.masked_where(goodReturns==0.0, goodReturns)
    io = numpy.sum(ma.getmaskarray(bad), axis=1)
    goodAssetsIdx = numpy.flatnonzero(io < 0.5 * daysBack)
    logging.info('%d out of %d assets have sufficient returns',
            len(goodAssetsIdx), goodReturns.shape[0])
    goodReturns = ma.take(goodReturns, goodAssetsIdx, axis=0)
    crossSectionalStdDev = Utilities.mlab_std(goodReturns, axis=0)
    crossSectionalStdDev = ma.masked_where(crossSectionalStdDev==0.0, crossSectionalStdDev)

    # If tiny markets that trade on irregular days are involved,
    # replace std dev on weekends and other days with sparse trading
    # with previous value
    bad = ma.masked_where(goodReturns==0.0, goodReturns)
    io = numpy.sum(ma.getmaskarray(bad), axis=0)
    badDatesIdx = numpy.flatnonzero(io > 0.9 * goodReturns.shape[0])
    logging.info('%d out of %d bad dates (fewer than 90%% assets trading)',
            len(badDatesIdx), goodReturns.shape[1])
    if len(badDatesIdx) < goodReturns.shape[1]:
        for i in badDatesIdx:
            if i==0:
                prevStdDev = ma.average(ma.take(crossSectionalStdDev,
                    [j for j in range(goodReturns.shape[1]) if j not in badDatesIdx]))
            else:
                prevStdDev = crossSectionalStdDev[i-1]
            if prevStdDev is not ma.masked:
                crossSectionalStdDev[i] = prevStdDev

    # Combine the various parts
    crossSectionalStdDev = Utilities.screen_data(crossSectionalStdDev)
    crossSectionalStdDev = ma.masked_where(crossSectionalStdDev==0.0, crossSectionalStdDev)
    csv = ma.average(abs(returns) / crossSectionalStdDev, axis=1)
    csv = ma.sqrt(csv)
    myLog().info('generate_cross_sectional_volatility: end')
    return csv

def generate_historic_volatility(returns, params):
    """The historic volatility is the square-root of the average
    of the squared asset returns for each asset over the
    last given number of days.
    """
    daysBack = params.daysBack
    myLog().info('generate_historic_volatility: begin')
    assert(daysBack > 0)
    assert(returns.data.shape[1] >= daysBack)
    myReturns = returns.data[:, -daysBack:]
    myReturns = myReturns * myReturns
    hv = numpy.sqrt(numpy.average(myReturns, axis=1))
    myLog().info('generate_historic_volatility: end')
    return hv

def generate_returns_skewness(returns, params):
    """Generates returns skewness exposures. Corrects for sample bias
    """
    myLog().info('generate_returns_skewness: begin')
    daysBack = params.daysBack
    assert(daysBack > 0)
    assert(returns.data.shape[1] >= daysBack)
    myReturns = returns.data[:, -daysBack:]
    return sm.skew(myReturns, axis=1)

def generate_medium_term_momentum(returns, fromT=250, thruT=20):
    """Medium-term momentum is the return over the previous trading days
    from fromT (inclusive) to thruT (exclusive).
    """
    myLog().debug('generate_medium_term_momentum: begin')
    assert(fromT > 0 and thruT >= 0 and thruT < fromT)
    mediumTermReturns = returns.data[:,-fromT:-thruT]
    mediumTermMomentum = numpy.cumproduct(
        mediumTermReturns+1.0, axis=1)[:,-1] - 1.0
    myLog().debug('generate_medium_term_momentum: end')
    return mediumTermMomentum

def generate_short_term_momentum(returns, numDays=20):
    """Short-term momentum is the return over the last 
    numDays trading days.  Default is 20 days.
    """
    myLog().debug('generate_short_term_momentum: begin')
    assert(numDays > 0)
    shortTermReturns = returns.data[:,-numDays:]
    shortTermMomentum = numpy.cumproduct(
        shortTermReturns+1.0, axis=1)[:,-1] - 1.0
    myLog().debug('generate_short_term_momentum: end')
    return shortTermMomentum

def generate_short_term_momentum_tm2(returns, numDays=20):
    """Short-term momentum is the return over the last 
    numDays trading days.  Default is 20 days.
    Differs from generate_short_term_momentum by considering
    returns from t-20 to t-2, instead of from t-20 to t-1.
    """
    myLog().info('generate_short_term_momentum_tm2: begin')
    assert(numDays > 0)
    shortTermReturns = returns.data[:,-numDays:]
    shortTermMomentum = numpy.cumproduct(
        shortTermReturns+1.0, axis=1)[:,-2] - 1.0
    myLog().info('generate_short_term_momentum_tm2: end')
    return shortTermMomentum

def generate_momentum(returns, params):
    """Medium-term momentum is the return over the previous trading days
    from fromT (inclusive) to thruT (exclusive).
    """
    myLog().info('generate_momentum: begin')
    fromT = params.fromT
    thruT = params.thruT
    assert(fromT > 0 and thruT >= 0 and thruT < fromT)
    if thruT == 0:
        returnsHistory = returns.data[:,-fromT:]
    else:
        returnsHistory = returns.data[:,-fromT:-thruT]
    if params.name=='Short-Term Momentum Tm2':
        end = -2
    else: 
        end = -1
    #returnsHistory = Utilities.clip_extrema(returnsHistory)
    momentum = numpy.cumproduct(
        returnsHistory+1.0, axis=1)[:,end] - 1.0
    myLog().info('generate_momentum: end')
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
    val = ma.log(data.issuerTotalMarketCaps)
    # Quick and dirty standardisation
    val_estu = ma.take(val, data.estimationUniverseIdx, axis=0)
    # Take out the mean
    mcap_estu = ma.take(data.issuerTotalMarketCaps, data.estimationUniverseIdx, axis=0)
    mean = ma.average(val_estu, weights=mcap_estu, axis=0)
    val = val - mean
    # Pseudo-standard deviation
    val_estu = ma.take(val, data.estimationUniverseIdx, axis=0)
    stdev = ma.sqrt(ma.inner(val_estu, val_estu) / (len(val_estu) - 1.0))
    val = val / stdev
    val = val * val * val
    return val

def generate_trading_volume_exposures(modelDate, data, rmgList,
                            modelDB, currencyID=None, daysBack=20,
                            legacy=False, median=False, simple=False):
    """Returns the trading volume exposures for the assets in
    data.universe.
    The exposures are computed as the ratio of the log average daily
    trading volume over the last daysBack trading days and the 
    log of the market capitalization as given by data.markCaps.
    If legacy=False, then the exposure is computed as simply
    volume/cap
    """
    dateList = modelDB.getDates(rmgList, modelDate, daysBack)
    vol = modelDB.loadVolumeHistory(
                    dateList, data.universe, currencyID)
    # Compute ADV for new listings differently
    if not hasattr(modelDB, 'subidMapCache'):
        modelDB.subidMapCache = ModelDB.FromThruCache(useModelID=False)
    issueData = modelDB.loadFromThruTable('sub_issue', 
                    [modelDate], data.universe, ['issue_id'], 
                    keyName='sub_id', cache=modelDB.subidMapCache)[0,:]
    issueFromDates = [n.fromDt for n in issueData]
    vol.data = Matrices.fillAndMaskByDate(
                    vol.data, issueFromDates, vol.dates)
    if median:
        volume = ma.median(vol.data, axis=1)
    else:
        volume = ma.average(vol.data, axis=1)
    newListings = [d for d in issueFromDates if d > vol.dates[0]]
    myLog().debug('Compute different ADV for %d new listings',
                    len(newListings))
    
    # Mask out volumes <= 0.0
    volume = ma.masked_where(volume <= 0.0, volume)
    if legacy == True:
        volume = ma.log(volume)
        mcap = ma.log(data.marketCaps)
    else:
        mcap = ma.masked_where(data.marketCaps <= 0.0, data.marketCaps)
    if simple:
        tra = volume
    else:
        tra = volume / mcap
    missing = numpy.flatnonzero(ma.getmaskarray(tra))
    if len(missing) > 0:
        myLog().debug('%d assets are missing trading volume information', len(missing))
        myLog().debug('missing Axioma IDs: ' + ', '.join([data.universe[i].getSubIDString() for i in missing]))
    return ma.filled(tra, 0.0)

def generate_trading_volume_exposures_v3(
            modelDate, data, rmgList, modelDB, params, currencyID=None):
    """Returns the trading volume exposures for the assets in
    data.universe.
    The exposures are computed as the ratio of the average daily
    trading volume over the last daysBack trading days and the
    market capitalization as given by data.markCaps.
    """
 
    # Set up parameters
    median = params.median
    simple = params.simple
    lnComb = params.lnComb
    if hasattr(params, 'legacy'):
        legacy = params.legacy
    else:
        legacy = False

    # Set up dates
    dateList =  modelDB.getDates(rmgList, modelDate, params.daysBack, excludeWeekend=True, fitNum=True)
    allDates = modelDB.getDateRange(None, dateList[0], dateList[-1])
    vol = modelDB.loadVolumeHistory(allDates, data.universe, currencyID)
    sidList = [sid.getSubIdString() for sid in data.universe]
    (vol.data, vol.dates) = Utilities.compute_compound_returns_v3(
            vol.data, allDates, dateList, keepFirst=True, matchDates=True, sumVals=True)

    # Compute ADV for new listings differently
    issueFromDates = modelDB.loadIssueFromDates([modelDate], data.universe)
    vol.data = Matrices.fillAndMaskByDate(
                    vol.data, issueFromDates, vol.dates)

    # Compute average or median dollar volume
    if median:
        volume = ma.median(vol.data, axis=1)
    else:
        volume = ma.average(vol.data, axis=1)
    newListings = [d for d in issueFromDates if d > vol.dates[0]]
    myLog().debug('Compute different ADV for %d new listings',
                    len(newListings))

    # Mask out volumes <= 0.0
    volume = ma.masked_where(volume <= 0.0, volume)
    mcap = ma.masked_where(data.marketCaps <= 0.0, data.marketCaps)
    missing = numpy.flatnonzero(ma.getmaskarray(volume))
    if len(missing) > 0:
        myLog().debug('%d assets are missing trading volume information', len(missing))
        myLog().debug('missing Axioma IDs: ' + ', '.join([data.universe[i].getSubIDString() for i in missing]))
    if not legacy:
        volume = ma.filled(volume, 0.0) + 1.0

    # Scale by cap if required
    if simple:
        tra = volume
    elif lnComb:
        tra = ma.log(volume) - ma.log(data.marketCaps)
    else:
        tra = volume / mcap
 
    return ma.filled(tra, 0.0)

def generate_relative_strength(modelDate, data, returns, modelSelector, 
                                modelDB, smooth=10):
    myLog().info('generate_relative_strength: begin')
    halfLife = -numpy.log(2.0) / numpy.log(1.0 - 2.0 / (smooth+1.0))
    wgt = Utilities.computeExponentialWeights(halfLife, returns.data.shape[1])
    rsi = Matrices.allMasked(len(returns.assets))
    for n in range(len(returns.assets)):
        ret = returns.data[n,:]
        u = ma.where(ret > 0, ret, 0.0) * wgt
        d = ma.where(ret < 0, ret, 0.0) * wgt
        if numpy.sum(d, axis=0) == 0:
            rsi[n] = 1.0
        else:
            rs = numpy.cumproduct(u+1.0, axis=0)[-1] \
                    / numpy.cumproduct(d+1.0, axis=0)[-1]
            rsi[n] = 1.0 - 1.0 / (1.0 + rs)
    myLog().info('generate_relative_strength: end')
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
    myLog().info('generate_variance_ratios: begin')
    daysBack = min(returns.data.shape[1], daysBack)
    assetReturns = returns.data[:,-daysBack:]

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
        myLog().debug('%d out of %d values less than %.1f std devs from 0.0' % (numpy.sum(ma.getmaskarray(result)), len(result), keepSigStd))

    myLog().info('generate_variance_ratios: end')
    return result

def generateAmihudLiquidityExposures(
        modelDate, returns, data, rmg, modelDB, params, currencyID=None,
        scaleByTO=False, originalReturns=None):
    """Returns the liquidity exposures for the assets in universe.
    The liquidity exposure is based on the illiquidity measure defined
    in Amihud(2000): the average of the absolute daily returns divided
    by average daily volume over the past daysBack days.
    For the exposure the illiquidity measure is normalized by its
    cross-sectional standard deviation.
    """
    # Set up parameters
    daysBack = params.daysBack
    dateList = returns.dates
    if originalReturns is None:
        returns = returns.data
    else:
        returns = originalReturns
    legacy = params.legacy

    # Set up necessary histories of returns and volumes
    assert(returns.shape[1] > daysBack)
    returns = returns[:,-daysBack:]
    dateList = dateList[-daysBack:]
    assert(returns.shape[1] == daysBack)
    allDates = modelDB.getDateRange(None, dateList[0], dateList[-1])
    volume = modelDB.loadVolumeHistory(
            allDates, data.universe, currencyID)
    (volume.data, volume.dates) = Utilities.compute_compound_returns_v3(
            volume.data, allDates, dateList, keepFirst=True, matchDates=True, sumVals=True)

    assert(volume.data.shape[1] == daysBack)
    assert(volume.data.shape[0] == returns.shape[0])

    # Compute ADV for new listings differently
    issueFromDates = modelDB.loadIssueFromDates([modelDate], data.universe)
    volume = Matrices.fillAndMaskByDate(
            volume.data, issueFromDates, volume.dates)
    nGoodDates = ma.sum(ma.getmaskarray(volume)==0, axis=1)

    # Mask out volumes <= 0.0
    volume = ma.where(volume <= 0.0, ma.masked, volume)
    if scaleByTO:
        mcap = modelDB.loadMarketCapsHistory(dateList, data.universe, currencyID)
        mcap = ma.masked_where(mcap <= 0.0, mcap)
        volume = volume / mcap
    ratio = ma.absolute(returns) / volume
    nonMissing = ma.sum(ma.getmaskarray(ratio)==0, axis=1)
    illiq = ma.average(ratio, axis=1)
    assert(illiq.shape[0] == len(data.universe))
    missing = numpy.flatnonzero(ma.getmaskarray(illiq))
    if len(missing) > 0:
        myLog().info('%d assets are missing Amihud liquidity information' % (len(missing)))
        myLog().debug('missing Axioma IDs: ' + ', '.join([data.universe[i].getSubIDString() for i in missing]))

    # Fill missing values with median for want of anything better
    if legacy:
        medianVal = ma.median(illiq, axis=0)
        illiq = ma.filled(illiq, medianVal)
        illiq = ma.log(1.0 + illiq)
    else:
        illiq = -1.0 * illiq
    return illiq

def generate_proportion_non_missing_returns(returns, data, modelDB, marketDB, daysBack=60):
    """Computes the proportion of non-missing returns over a given
    period as a proxy for the liquidity of an asset
    """
    # Get boolean array of actual valid live dates for each asset
    minDays = 5
    validDays = len(returns.dates) - ma.sum(returns.preIPOFlag + returns.ntdFlag, axis=1)
    validRets = validDays - ma.sum(returns.rollOverFlag, axis=1)
    validRets = ma.where(validDays < minDays, 0, validRets)
    pcntValidRets = ma.filled(validRets / numpy.array(validDays, float), 0.0)

    return pcntValidRets

def generate_trading_activity(modelSelector, modelDate, data, modelDB, 
                    currencyID=None, restrict=None, equalWeight=True, daysBack=20):
    """Computes 'volume betas' -- sensitivity of changes in 
    assets' daily volume traded to changes in the average
    volume traded in the overall market.  The market is defined
    as all the assets, or those corresponding to array 
    positions in restrict.  By default the market's average 
    volume is computed using equal-weighted average but the
    root-cap weighted average is used if equalWeight=False.
    """
    myLog().info('generate_trading_activity: begin')
    univ = data.universe
    dateList = modelDB.getDates(modelSelector.rmg, modelDate, daysBack)
    volume = modelDB.loadVolumeHistory(
        dateList, univ, currencyID).data
    volume = ma.masked_where(volume<=0.0, volume)
    volumeChanges = (volume[:,1:] - volume[:,:-1]) / volume[:,:-1]
    volumeChanges -= 1.0
    badAssetsIdx = numpy.flatnonzero(numpy.sum(
                ma.getmaskarray(volumeChanges), axis=1)==daysBack)
    volumeChanges = volumeChanges.filled(0.0)

    if restrict is None:
        restrict = list(range(len(univ)))
    marketVolumeChange = numpy.take(volumeChanges, restrict, axis=0)
    if equalWeight:
        marketVolumeChange = numpy.average(marketVolumeChange, axis=0)
    else:
        mcaps = ma.take(data.marketCaps, restrict, axis=0)**0.5
        weights = mcaps / ma.sum(mcaps)
        marketVolumeChange = numpy.dot(weights, marketVolumeChange)

    regressor = numpy.transpose(numpy.array(
                        [numpy.ones(daysBack, float), marketVolumeChange]))
    (beta, e) = Utilities.ordinaryLeastSquares(
                        numpy.transpose(volumeChanges), regressor)
    beta = ma.array(beta[1,:])
    ma.put(beta, badAssetsIdx, ma.masked)
    myLog().info('generate_trading_activity: end')
    return beta

def generate_world_sensitivity(returns, mcaps, periodsBack=120, 
                               indices=None, simple=False):
    """Computes sensitivity of asset returns to the 
    'world market', represented by either all the assets
    in returns.assets, or those corresponding to index positions
    if indices is given.  If simple=True, returns a vector
    of ones (to be used as an intercept term).
    """
    myLog().info('generate_world_sensitivity: begin')
    if simple:
        myLog().info('generate_world_sensitivity: end')
        return ma.ones(len(mcaps))
    # Restrict history and assets
    assert(len(returns.dates) >= periodsBack)
    assetReturns = returns.data[:,-periodsBack:]
    if indices is None:
        indices = range(len(returns.assets))
    coreReturns = ma.take(assetReturns, indices, axis=0)

    # Remove days where many returns are zero or missing
    io = numpy.sum(ma.getmaskarray(ma.masked_where(
                   coreReturns==0.0, coreReturns)), axis=0)
    goodDatesIdx = numpy.flatnonzero(io < 0.5 * len(returns.assets))
    coreReturns = ma.take(coreReturns, goodDatesIdx, axis=1)
    coreReturns = coreReturns.filled(0.0)

    # Construct naive 'market' portfolio
    goodCaps = ma.take(mcaps, indices, axis=0)**0.5
    weights = goodCaps / ma.sum(goodCaps)
    marketReturns = ma.dot(weights, coreReturns).filled(0.0)

    # Remove assets with lots of zero/missing returns
    assetReturns = ma.take(assetReturns, goodDatesIdx, axis=1)
    bad = ma.masked_where(assetReturns==0.0, assetReturns)
    io = numpy.sum(ma.getmaskarray(bad), axis=1)
    goodAssetsIdx = numpy.flatnonzero(io < 0.5 * len(goodDatesIdx))
    assetReturns = ma.take(assetReturns, goodAssetsIdx, axis=0)

    # Regress asset returns on market returns
    x = numpy.transpose(ma.array(
                [ma.ones(len(goodDatesIdx), float), marketReturns]))
    y = numpy.transpose(assetReturns.filled(0.0))
    (coef, resid) = Utilities.ordinaryLeastSquares(y, x)
    betas = Matrices.allMasked(len(returns.assets)) 
    ma.put(betas, goodAssetsIdx, coef[1,:])

    # Tidy things up
    betas = betas.filled(1.0)
    betas = numpy.clip(betas, -1.0, 4.0)
    betas = numpy.where(betas==0.0, 1.0, betas)
    
    myLog().info('generate_world_sensitivity: end')
    return betas

def generate_forex_sensitivity(returns, modelSelector, modelDB,
                               periodsBack, numeraire='USD', 
                               lag=0, swAdj=False):
    """Compute assets' sensitivity to currency returns.
    The lag parameter can be used to regress local returns
    against lagged currency returns, particularly useful for 
    Asian and European markets, where local market movements 
    are often heavily influenced by events in the US in the
    previous day.  swAdj=True will apply the Scholes-Williams
    adjustment to account for asynchronous trading.
    """
    myLog().info('generate_forex_sensitivity: begin')

    # Restrict history
    assert(returns.data.shape[1] >= periodsBack)
    assetReturns = returns.data[:,-periodsBack:]
    assetIdxMap = dict(zip(returns.assets, range(returns.data.shape[0])))
    dateList = returns.dates[-periodsBack:]

    beta = Matrices.allMasked(assetReturns.shape[0])

    # These currencies we need not worry about
    ignoreList = getPeggedCurrencies(numeraire,\
            dateList[0], dateList[-1])

    # Load the risk model group's currency returns
    # versus the given numeraire currency
    rmgList = modelSelector.rmg
    currencies = list(set([r.currency_code for r in rmgList if \
                            r.currency_code not in ignoreList]))
    currencyIdxMap = dict([(code, idx) for (idx, code) \
                            in enumerate(currencies)])
    fxReturnMatrix = modelDB.loadCurrencyReturnsHistory(
                            rmgList, dateList[-(1+lag)], 
                            periodsBack-1, currencies, numeraire)
    # Back-compatibility point
    if modelSelector.modelHack.fxSensReturnsClip:
        (fxReturnMatrix.data, bounds) = Utilities.mad_dataset(
                fxReturnMatrix.data, modelSelector.xrBnds[0], modelSelector.xrBnds[1],
                axis=1, treat='clip')

    for (i,r) in enumerate(rmgList):
        # Disregard pegged currencies
        if r.currency_code in ignoreList:
            continue

        # Determine list of assets for each market
        if len(rmgList) > 1:
            rmg_assets = modelSelector.rmgAssetMap[r.rmg_id]
            indices = [assetIdxMap[n] for n in rmg_assets]
        else:
            indices = list(range(len(returns.assets)))

        # Fetch corresponding currency and asset returns
        if len(indices) > 0:
            threshold = 0.7
            rmgReturns = ma.take(assetReturns, indices, axis=0)
            # Remove days on which majority of assets don't trade
            io = numpy.sum(ma.getmaskarray(ma.masked_where(
                    rmgReturns==0.0, rmgReturns)), axis=0)
            tradingDatesIdx = numpy.flatnonzero(io < threshold * len(indices))
            if len(tradingDatesIdx) <= max(2.0, 0.05 * len(io)):
                myLog().info('Skipping %s, %d out of %d days with %d%%+ of stocks trading', 
                        r.description, len(tradingDatesIdx), len(io), threshold * 100.0)
                continue
            rmgReturns = ma.take(rmgReturns, tradingDatesIdx, axis=1)
            currencyReturns = fxReturnMatrix.data[currencyIdxMap[r.currency_code]]
            currencyReturns = ma.take(currencyReturns, tradingDatesIdx, axis=0)

            # TODO: net out the risk-free rate
            t = rmgReturns.shape[1]
            n = rmgReturns.shape[0]
            assert(currencyReturns.shape[0] == t)

            # Compute beta
            x = numpy.transpose(ma.array(
                        [ma.ones(t, float), currencyReturns.filled(0.0)]))
            y = numpy.transpose(ma.filled(rmgReturns, 0.0))
            (b0, e0) = Utilities.ordinaryLeastSquares(y, x)

            values = b0[1,:]
            if swAdj:
                # Market lagging asset returns...
                x_lag = x[:-1,:]
                y_lag = y[1:,:]
                (b_lag, e_lag) = Utilities.ordinaryLeastSquares(y_lag, x_lag)
                # Market leading asset returns...
                x_lead = x[1:,:]
                y_lead = y[:-1,:]
                (b_lead, e_lead) = Utilities.ordinaryLeastSquares(y_lead, x_lead)

                # Put it all together...
                xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
                corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
                k = 1.0 + 2.0 * corr
                logging.info('KAPPA REPORT: %s', k)
                k = abs(k)
                if k != 0.0:
                    values = (b_lag[1,:] + b0[1,:] + b_lead[1,:]) / k

            ma.put(beta, indices, values)

    myLog().info('generate_forex_sensitivity: end')
    return beta

def generate_forex_sensitivity_v2(returns, modelSelector, modelDB,
        daysBack, numeraire='XDR', frequency='weekly', lag=0,
        marketFactor=True, swAdj=False, fixDate=False):
    """Compute assets' sensitivity to currency returns.
    Data are lagged by an interval given by the lag parameter
    swAdj=True will apply the Scholes-Williams adjustment to
    account for asynchronous trading.
    """
    myLog().info('generate_forex_sensitivity_v2: begin')

    # Restrict history
    assert(returns.data.shape[1] >= daysBack)
    assetReturns = returns.data[:,-daysBack:]
    assetIdxMap = dict(zip(returns.assets, range(returns.data.shape[0])))
    dateList = returns.dates[-daysBack:]
    beta = Matrices.allMasked(assetReturns.shape[0])

    # These currencies we need not worry about
    ignoreList = getPeggedCurrencies(numeraire,\
            dateList[0], dateList[-1])

    # Load the risk model group's currency returns
    # versus the given numeraire currency
    rmgList = modelSelector.rmg
    currencies = list(set([r.currency_code for r in rmgList if \
            r.currency_code not in ignoreList]))
    currencyIdxMap = dict([(code, idx) for (idx, code) \
            in enumerate(currencies)])
    fxReturns = modelDB.loadCurrencyReturnsHistory(
            rmgList, dateList[-(1+lag)],
            daysBack-1, currencies, numeraire).data

    if marketFactor:
        marketReturns = modelDB.loadRMGMarketReturnHistory(dateList, rmgList, useAMPs=False).data
    # Compound returns if necessary
    if frequency == 'weekly':
        if fixDate:
            periodDateList = [prv for (prv, nxt) in \
                    zip(dateList[:-1], dateList[1:])
                    if nxt.weekday() < prv.weekday()]
        else:
            periodDateList = [nxt for (nxt, prv) in \
                    zip(dateList[:-1], dateList[1:])
                    if nxt.weekday() < prv.weekday()]
    elif frequency == 'monthly':
        if fixDate:
            periodDateList = [prv for (prv, nxt) in \
                    zip(dateList[:-1], dateList[1:])
                    if nxt.month > prv.month or nxt.year > prv.year]
        else:
            periodDateList = [nxt for (nxt, prv) in \
                    zip(dateList[:-1], dateList[1:])
                    if nxt.month > prv.month or nxt.year > prv.year]
    else:
        periodDateList = list(dateList)

    if len(dateList) != len(periodDateList):
        assetReturns = Utilities.compute_compound_returns(
                        assetReturns, dateList, periodDateList)
        fxReturns = Utilities.compute_compound_returns(
                        fxReturns, dateList, periodDateList)
        if marketFactor:
            marketReturns = Utilities.compute_compound_returns(
                    marketReturns, dateList, periodDateList)

    for (i,r) in enumerate(rmgList):
        # Disregard pegged currencies
        if r.currency_code in ignoreList:
            continue

        # Determine list of assets for each market
        if len(rmgList) > 1:
            rmg_assets = modelSelector.rmcAssetMap[r.rmg_id]
            indices = [assetIdxMap[n] for n in rmg_assets]
        else:
            indices = list(range(len(returns.assets)))

        # Fetch corresponding currency and asset returns
        properFix = False
        if len(indices) > 0 and not (len(indices) <= 1 and marketFactor):
            rmgReturns = ma.take(assetReturns, indices, axis=0)
            currencyReturns = fxReturns[currencyIdxMap[r.currency_code]]
            if properFix:
                # Compound returns for non-trading days into the following trading-day
                rmgCalendarSet = set(modelDB.getDateRange(r,
                    periodDateList[0], periodDateList[-1]))
                tradingDates = [d for d in periodDateList if d in rmgCalendarSet]
                rmgReturns = Utilities.compute_compound_returns(
                        rmgReturns, periodDateList, tradingDates)
                currencyReturns = Utilities.compute_compound_returns(
                        currencyReturns, periodDateList, tradingDates)
            else:
                # Old unsatisfactory logic
                threshold = 0.7
                # Remove days on which majority of assets don't trade
                io = numpy.sum(ma.getmaskarray(ma.masked_where(
                    rmgReturns==0.0, rmgReturns)), axis=0)
                tradingDatesIdx = numpy.flatnonzero(io < threshold * len(indices))
                rmgReturns = ma.take(rmgReturns, tradingDatesIdx, axis=1)
                currencyReturns = ma.take(currencyReturns, tradingDatesIdx, axis=0)
                 
            # Warn if all currency returns are missing/zero
            tmpArray = ma.masked_where(currencyReturns == 0.0, currencyReturns)
            if numpy.sum(ma.getmaskarray(tmpArray)) == len(currencyReturns):
                myLog().info('Skipping %s, zero %s currency returns', r.description, r.currency_code)
                continue
            # Add market factor to the mix if required
            if marketFactor:
                if properFix:
                    marketReturn = Utilities.compute_compound_returns(
                            marketReturns[i], periodDateList, tradingDates)
                else:
                    marketReturn = ma.take(marketReturns[i], tradingDatesIdx, axis=0)
                # Warn if all market returns are missing/zero
                tmpArray = ma.masked_where(marketReturn == 0.0, marketReturn)
                if numpy.sum(ma.getmaskarray(tmpArray)) == len(marketReturn):
                    myLog().info('Skipping %s, zero market returns', r.description)
                    continue
                marketReturn = ma.filled(marketReturn, 0.0)

            # TODO: net out the risk-free rate
            t = rmgReturns.shape[1]
            n = rmgReturns.shape[0]
            assert(currencyReturns.shape[0] == t)

            # Compute beta
            if marketFactor:
                x = numpy.transpose(ma.array(
                    [ma.ones(t, float), marketReturn, ma.filled(currencyReturns, 0.0)]))
                offset = 1
            else:
                x = numpy.transpose(ma.array(
                    [ma.ones(t, float), ma.filled(currencyReturns, 0.0)]))
                offset = 0
            y = numpy.transpose(ma.filled(rmgReturns, 0.0))
            (b0, e0) = Utilities.ordinaryLeastSquares(y, x, implicit=True)

            values = b0[1+offset,:]
            if swAdj:
                # Market lagging asset returns...
                x_lag = x[:-1,:]
                y_lag = y[1:,:]
                (b_lag, e_lag) = Utilities.ordinaryLeastSquares(y_lag, x_lag, implicit=True)
                # Market leading asset returns...
                x_lead = x[1:,:]
                y_lead = y[:-1,:]
                (b_lead, e_lead) = Utilities.ordinaryLeastSquares(y_lead, x_lead, implicit=True)

                # Put it all together...
                xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
                corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
                k = 1.0 + 2.0 * corr
                logging.info('KAPPA REPORT: %s', k)
                k = abs(k)
                if k != 0.0:
                    values = (b_lag[1+offset,:] + b0[1+offset,:] + b_lead[1+offset,:]) / k

            ma.put(beta, indices, values)
        else:
            myLog().info('Skipping %s, too few assets (%d)', r.description, len(indices))

    myLog().info('generate_forex_sensitivity_v2: end')
    return beta

def generate_forex_sensitivity_v3(data, modelSelector, modelDB, pm, 
        clippedReturns=None, marketRegion=None):
    """Compute assets' sensitivity to currency returns.
    Data are lagged by an interval given by the lag parameter
    swAdj=True will apply the Scholes-Williams adjustment to
    account for asynchronous trading.
    """
    myLog().info('generate_forex_sensitivity_v3: begin')

    # Set up parameters
    daysBack = pm.daysBack
    numeraire = pm.numeraire
    frequency = pm.frequency
    if hasattr(pm, 'lag'):
        lag = pm.lag
    else:
        lag = 0
    if hasattr(pm, 'marketFactor'):
        marketFactor = pm.marketFactor
    else:
        marketFactor = True
    if hasattr(pm, 'swAdj'):
        swAdj = pm.swAdj
    else:
        swAdj = False
    if hasattr(pm, 'robust'):
        robustBeta = pm.robust
    else:
        robustBeta = False
    if hasattr(pm, 'k_val'):
        k = pm.k_val
    else:
        k = 1.345

    # Restrict history
    assert(data.returns.data.shape[1] >= daysBack)
    if robustBeta or (clippedReturns is None):
        assetReturns = data.returns.data[:,-daysBack:]
    else:
        assetReturns = clippedReturns[:,-daysBack:]
    assetIdxMap = dict(zip(data.returns.assets, range(data.returns.data.shape[0])))
    dateList = data.returns.dates[-daysBack:]
    beta = Matrices.allMasked(assetReturns.shape[0])

    # These currencies we need not worry about
    ignoreList = getPeggedCurrencies(numeraire,\
            dateList[0], dateList[-1])

    # Load the risk model group's currency returns
    # versus the given numeraire currency
    rmgList = modelSelector.rmg
    currencies = list(set([r.currency_code for r in rmgList if \
            r.currency_code not in ignoreList]))
    currencyIdxMap = dict([(code, idx) for (idx, code) \
            in enumerate(currencies)])
    fxReturns = modelDB.loadCurrencyReturnsHistory(
            rmgList, dateList[-(1+lag)],
            daysBack-1, currencies, numeraire).data

    if marketFactor:
        mktDateList = modelDB.getDateRange(None, min(dateList), max(dateList))
        marketReturns = modelDB.loadRMGMarketReturnHistory(mktDateList, rmgList, useAMPs=False).data
        if mktDateList != dateList:
            marketReturns = Utilities.compute_compound_returns_v3(
                    marketReturns, mktDateList, dateList, keepFirst=True)[0]

    # Compound returns if necessary
    if frequency == 'weekly':
        periodDateList = [prv for (prv, nxt) in \
                zip(dateList[:-1], dateList[1:])
                if nxt.weekday() < prv.weekday()]
    elif frequency == 'monthly':
        periodDateList = [prv for (prv, nxt) in \
                zip(dateList[:-1], dateList[1:])
                if nxt.month > prv.month or nxt.year > prv.year]
    else:
        periodDateList = list(dateList)

    if dateList != periodDateList:
        (assetReturns, pdList) = Utilities.compute_compound_returns_v3(
                        assetReturns, dateList, periodDateList, matchDates=True)
        (fxReturns, pdList) = Utilities.compute_compound_returns_v3(
                        fxReturns, dateList, periodDateList, matchDates=True)
        if marketFactor:
            (marketReturns, pdList) = Utilities.compute_compound_returns_v3(
                    marketReturns, dateList, periodDateList, matchDates=True)
        periodDateList = pdList

    for (i,r) in enumerate(rmgList):
        # Disregard pegged currencies
        if r.currency_code in ignoreList:
            continue

        # Determine list of assets for each market
        if len(rmgList) > 1:
            rmg_assets = data.rmgAssetMap[r.rmg_id]
            indices = [assetIdxMap[n] for n in rmg_assets]
        else:
            indices = list(range(len(data.returns.assets)))

        # Fetch corresponding currency and asset returns
        if len(indices) > 0 and not (len(indices) <= 1 and marketFactor):
            rmgReturns = ma.take(assetReturns, indices, axis=0)
            currencyReturns = fxReturns[currencyIdxMap[r.currency_code]]
             
            # Compound returns for non-trading days into the following trading-day
            rmgCalendarSet = set(modelDB.getDateRange(r,
                periodDateList[0], periodDateList[-1]))
            tradingDates = [d for d in periodDateList if d in rmgCalendarSet]
            rmgReturns = Utilities.compute_compound_returns_v3(
                    rmgReturns, periodDateList, tradingDates, keepFirst=True, matchDates=True)[0]
            currencyReturns = Utilities.compute_compound_returns_v3(
                    currencyReturns, periodDateList, tradingDates, keepFirst=True, matchDates=True)[0]
                 
            # Warn if all currency returns are missing/zero
            tmpArray = ma.masked_where(currencyReturns == 0.0, currencyReturns)
            if numpy.sum(ma.getmaskarray(tmpArray)) == len(currencyReturns):
                myLog().info('Skipping %s, zero %s currency returns', r.description, r.currency_code)
                continue
             
            # Add market factor to the mix if required
            if marketFactor:
                marketReturn = Utilities.compute_compound_returns_v3(
                        marketReturns[i], periodDateList, tradingDates, keepFirst=True, matchDates=True)[0]
                # Warn if all market returns are missing/zero
                tmpArray = ma.masked_where(marketReturn == 0.0, marketReturn)
                if numpy.sum(ma.getmaskarray(tmpArray)) == len(marketReturn):
                    myLog().info('Skipping %s, zero market returns', r.description)
                    continue
                marketReturn = ma.filled(marketReturn, 0.0)
                # Fix for assets with deficient histories by filling missing values with the market return
                maskedReturns = numpy.array(ma.getmaskarray(rmgReturns), dtype='float')
                for ii in range(len(marketReturn)):
                    maskedReturns[:,ii] *= marketReturn[ii]
                rmgReturns += maskedReturns

            # TODO: net out the risk-free rate
            t = rmgReturns.shape[1]
            n = rmgReturns.shape[0]
            assert(currencyReturns.shape[0] == t)

            # Compute beta
            if marketFactor:
                x = numpy.transpose(ma.array(
                    [ma.ones(t, float), marketReturn, ma.filled(currencyReturns, 0.0)]))
                offset = 1
            else:
                x = numpy.transpose(ma.array(
                    [ma.ones(t, float), ma.filled(currencyReturns, 0.0)]))
                offset = 0
            y = numpy.transpose(ma.filled(rmgReturns, 0.0))
            res = Utilities.robustLinearSolver(y, x, robust=robustBeta, k=k)
            b0 = res.params
            tvalues = res.tstat
            values = b0[1+offset,:]
             
            if swAdj:
                # Market lagging asset returns...
                x_lag = x[:-1,:]
                y_lag = y[1:,:]
                res = Utilities.robustLinearSolver(y_lag, x_lag, robust=robustBeta, k=k)
                b_lag = res_lag.params
                t_lag = res_lag.tstat
                # Market leading asset returns...
                x_lead = x[1:,:]
                y_lead = y[:-1,:]
                res_lead = Utilities.robustLinearSolver(y_lead, x_lead, robust=robustBeta, k=k)
                b_lead = res_lead.params
                t_lead = res_lead.tstat

                # Put it all together...
                xCombined = ma.transpose(ma.array([x_lag[:,1], x_lead[:,1]]))
                corr = numpy.corrcoef(xCombined, rowvar=False)[0,1]
                k = 1.0 + 2.0 * corr
                logging.info('KAPPA REPORT: %s', k)
                k = abs(k)
                if k != 0.0:
                    values = (b_lag[1+offset,:] + b0[1+offset,:] + b_lead[1+offset,:]) / k
                    tvalues = (t_lag[1+offset,:] + tvalues[1+offset,:] + t_lead[1+offset,:]) / k
            else:
                values = b0[1+offset,:]
                tvalues = tvalues[1+offset,:]

            ma.put(beta, indices, values)
        else:
            myLog().info('Skipping %s, too few assets (%s)', r.description, len(indices))

    myLog().info('generate_forex_sensitivity_v3: end')
    return beta
 
def generate_interest_rate_sensitivity(returns, modelDB, marketDB,
                    periodsBack, currencyCode='USD', lagDays=0):
    """Compute assets' sensitivity to interest rates in the 
    given currency.
    The lagDays parameter can be used to regress local returns
    against lagged currency returns, particularly useful for 
    Asian and European markets, where local market movements 
    are often heavily influenced by events in the US in the
    previous day.
    """
    myLog().info('generate_interest_rate_sensitivity: begin')

    # Restrict history
    assert(returns.data.shape[1] >= periodsBack)
    assetReturns = returns.data[:,-periodsBack:]
    dateList = returns.dates[-periodsBack:]

    # Remove days where many returns are zero
    io = numpy.sum(ma.getmaskarray(ma.masked_where(
            assetReturns==0.0, assetReturns)), axis=0)
    goodDatesIdx = numpy.flatnonzero(io < 0.9 * len(returns.assets))
    goodDates = numpy.take(dateList, goodDatesIdx, axis=0)
    assetReturns = ma.take(assetReturns, goodDatesIdx, axis=1)

    # Load interest rates for the given currency
    interestRatesMatrix = modelDB.getRiskFreeRateHistory(
                            [currencyCode], goodDates, marketDB)
    interestRates = interestRatesMatrix.data[0,:].filled(0.0)
    interestRateReturns = interestRates[1:] - interestRates[:-1]
    assetReturns = assetReturns[:,1:]
    if lagDays != 0:
        interestRateReturns = interestRateReturns[:-lagDays]
        assetReturns = assetReturns[:,lagDays:]

    t = assetReturns.shape[1]
    n = assetReturns.shape[0]
    assert(interestRateReturns.shape[0] == t)
    beta = ma.zeros(n)

    # Compute sensitivities
    x = numpy.transpose(ma.array(
                [ma.ones(t, float), interestRateReturns]))
    y = numpy.transpose(assetReturns)
    (b0, e0) = Utilities.ordinaryLeastSquares(y, x)
    beta = b0[1,:]

    myLog().info('generate_interest_rate_sensitivity: end')
    return beta

def generate_price_variability(modelDate, modelSelector, assets, 
                               modelDB, daysBack=20, copyUSE3=True):
    """Computes the ratio of the maximum to minimum market cap
    attained over the past daysBack days.  Replicates Barra's HILO
    descriptor or Northfield's 'Price Volatility Index' factor.
    Market caps are in local currency, not convereted to a numeraire.
    """
    myLog().info('generate_price_variability: begin')
    dateList = modelDB.getDates(modelSelector.rmg, modelDate, daysBack)
    mcaps = modelDB.loadMarketCapsHistory(dateList, assets, None)
    mcaps = ma.masked_where(mcaps <= 0.0, mcaps)
    high = numpy.max(mcaps, axis=1)
    low = numpy.min(mcaps, axis=1)
    if copyUSE3:
        values = numpy.log(high / low)          # Barra
    else:
        values = (high - low) / (high + low)    # NF
    myLog().info('generate_price_variability: end')
    return values

def getPeggedCurrencies(numeraire, startDate, endDate=None):
    """ List of currencies with fixed exchange rates, including
    pegs, currency boards, but not managed floats, and
    the start/end dates for the fixed rate regime.
    Currencies trading within very tight bands (eg. DKK/EUR)
    are also included.
    """
    if endDate == None:
        endDate = startDate

    fixedRates = {'USD': {
            'HKD': [None, None],
            'MYR': ['1998-09-01','2005-07-21'],
            'CNY': [None, '2005-07-21'],
            'THB': ['1985-01-01', '1997-07-02'],
            'ARS': [None, '2002-01-01'],
            'BHD': ['1980-12-01', None],
            'JOD': ['1995-10-23', None],
            'OMR': [None, None],
            'KWD': ['2003-01-05','2007-05-20'],
            'AED': ['1978-01-28', None],
            'VEF': [None, None],
            'VEB': [None, None],
            'LTL': [None, '2005-02-02'],
            'SAR': ['1986-06-01', None],
            'LBP': [None, None],
            'TTD': [None, None],
            'QAR': ['1975-03-01', None],
            'EGP': [None, '2003-01-29'],
            'IDR': [None, '1997-07-02'],
            'PHP': [None, '1997-07-02']
        },
        'EUR': {
            'DKK': [None, None],
            'EEK': [None, None],
            'LVL': ['2005-01-01', None],
            'BGN': [None, None],
            'LTL': ['2002-02-02', None]
        },
        'XDR': {
            'LVL': [None, '2005-01-01'],
            'BHD': ['1980-12-01', None],
            'JOD': ['1995-10-23', None],
            'AED': ['1978-01-28', None],
            'SAR': ['1986-06-01', None],
            'QAR': ['1975-03-01', None]
            }
        }
    for base in fixedRates.keys():
        for curr in fixedRates[base].keys():
            value = fixedRates[base][curr]
            if not value[0]:
                value[0] = '1950-01-01'
            if not value[1]:
                value[1] = '2999-12-31'
            fixedRates[base][curr]= [Utilities.parseISODate(value[0]),
                    Utilities.parseISODate(value[1])]

    ignoreList = [numeraire]
    if numeraire in fixedRates:
        for code in fixedRates[numeraire].keys():
            lifeTime = fixedRates[numeraire][code]
            if (not startDate >= lifeTime[1]) \
                    and (not endDate <= lifeTime[0]):
                ignoreList.append(code)
    logging.info('Pegged currencies: %s', ignoreList)
    return ignoreList

def generate_estimate_EBITDA(modelDate, data, model, modelDB, marketDB,
                             daysBack=365, restrict=None,
                             useQuarterlyData=False):

    """Compute the estimated EBITDA for the assets in data.universe.
    Earnings data is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the EBITDA value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_estimate_EBITDA: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    
    ebitdaRaw = modelDB.getIBESCurrencyItemLegacy('ebitda_median_ann',
                                         startDate, modelDate, subIssues, modelDate, 
                                         marketDB,
                                         convertTo=model.numeraire.currency_id)

    #measure
    #consensus options: 'median', 'mean', 'stdev', 'high', 'low','numest','numup','numdown'
    ebitda = Utilities.extractLatestValue(ebitdaRaw)
    missing = numpy.flatnonzero(ma.getmaskarray(ebitda))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estimateEBITA information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))

    if restrict is not None:
        ebitdaRest = Matrices.allMasked(len(data.universe))
        ma.put(ebitdaRest, restrict, ebitda)
        for (i,val) in enumerate(ebitda):
            if val is ma.masked:
                ebitdaRest[restrict[i]] = ma.masked
        ebitda = ebitdaRest
    myLog().info('generate_estimate_EBITDA: end')
    return ebitda

def generate_est_earnings_to_price(modelDate, data, model, modelDB, marketDB, params,
                               daysBack=(2*365), restrict=None,
                               useQuarterlyData=False, legacy=False):
    """Compute an earnings-to-price ratio for the assets in data.universe.
    If estimated earnings-per-share information is available the value is computed
    based on realised and estimated earnings-per-share information, otherwise the
    value is computed using just realised earnings-per-share information. In either
    case the earnings-per-share information is combined with current TSO.
    Returns an array with the estimated earnings-to-price value for each asset.
    If data is not available for an asset the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_est_earnings_to_price: begin')
    if hasattr(params, 'maskNegative'):
        maskNegative = params.maskNegative
    else:
        maskNegative = False
    if hasattr(params, 'estOnly'):
        estOnly = params.estOnly
    else:
        estOnly = False
    if hasattr(params, 'winsoriseRaw'):
        winsoriseRaw = params.winsoriseRaw
    else:
        winsoriseRaw = False

    startDate = modelDate - datetime.timedelta(daysBack)
    if hasattr(data, 'DLCMarketCap'):
        marketCaps = data.DLCMarketCap
    else:
        marketCaps = data.issuerMarketCaps
    if restrict is None:
        subIssues = data.universe
        divisor = marketCaps
    else:
        subIssues = [data.universe[i] for i in restrict]
        divisor = ma.take(marketCaps, restrict, axis=0)
    useQuarterlyData = False        # We don't have quarterly data at the moment
    if useQuarterlyData:
        if legacy:
            fieldName = 'ebitda_qtr'
        else:
            fieldName = 'ibei_qtr'
    else:
        if legacy:
            fieldName = 'ebitda_ann'
        else:
            fieldName = 'ibei_ann'
    earnings= modelDB.getFundamentalCurrencyItem(fieldName, 
                startDate, modelDate, subIssues, modelDate, 
                marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        earningsArray = modelDB.annualizeQuarterlyValues(earnings)
    else:
        earningsArray = Utilities.extractLatestValue(earnings)
    estEPSRaw = modelDB.getIBESCurrencyItemLegacy('eps_median_ann',
                                         startDate, modelDate, subIssues, modelDate, marketDB,
                                         convertTo=model.numeraire.currency_id)

    tso = modelDB.loadTSOHistory([modelDate], subIssues)
    estEPS = Utilities.extractLatestValue(estEPSRaw)
    estEarningsArray = estEPS*tso.data[:, 0]/1e6
    earningsToPrice = 1e6 * earningsArray / divisor
    estEarningsToPrice = 1e6 * estEarningsArray / divisor

    # Simple variant - merely mask negative earnings of any sort
    if maskNegative:
        earningsToPrice = ma.masked_where(earningsToPrice < 0.0, earningsToPrice)
        estEarningsToPrice = ma.masked_where(estEarningsToPrice < 0.0, estEarningsToPrice)

    if winsoriseRaw:
        earningsToPrice = Utilities.twodMAD(earningsToPrice, nDev=[3.0, 3.0], estu=data.estimationUniverseIdx)
        estEarningsToPrice = Utilities.twodMAD(estEarningsToPrice, nDev=[3.0, 3.0], estu=data.estimationUniverseIdx)

    # Estimate only, or combination
    if estOnly:
        estE2P = ma.array(estEarningsToPrice)
    else:
        e2pList = [earningsToPrice, estEarningsToPrice]
        estE2P = ma.average(ma.array(e2pList), axis=0)

    missing = numpy.flatnonzero(ma.getmaskarray(estE2P))
    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estEarningsToPrice information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        estE2PRest = Matrices.allMasked(len(data.universe))
        ma.put(estE2PRest, restrict, estE2P)
        for (i,val) in enumerate(estE2P):
            if val is ma.masked:
                estE2PRest[restrict[i]] = ma.masked
        estE2P = estE2PRest
    myLog().info('generate_est_earnings_to_price: end')
    return estE2P

def generate_est_earnings_to_price_nu(modelDate, data, model, modelDB, 
                marketDB, daysBack=(2*365), useQuarterlyData=False):
    """Compute an earnings-to-price ratio for the assets in data.universe.
    If estimated earnings-per-share information is available the value is computed
    based on realised and estimated earnings-per-share information, otherwise the
    value is computed using just realised earnings-per-share information. In either
    case the earnings-per-share information is combined with current TSO.
    Returns an array with the estimated earnings-to-price value for each asset.
    If data is not available for an asset the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_est_earnings_to_price: begin')

    # Initialise things
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(daysBack)
    if hasattr(data, 'DLCMarketCap'):
        marketCaps = data.DLCMarketCap
    else:
        marketCaps = data.issuerMarketCaps
    subIssues = data.universe
    divisor = marketCaps

    # Load forecast EPS
    estEPSRaw = modelDB.getIBESCurrencyItemLegacy(
                    'eps_median_ann', startDate, endDate, subIssues, modelDate,
                    marketDB, convertTo=model.numeraire.currency_id)

    # Take latest estimate
    tmpEst = []
    for estE in estEPSRaw:
        if len(estE) > 0:
            eVals = [ev for ev in estE if ev[0] >= startDate]
        else:
            eVals = estE
        tmpEst.append(eVals)
    latestEstEPS = Utilities.extractLatestValuesAndDate(tmpEst)

    # Load realised earnings for comparison
    if useQuarterlyData:
        fieldName = 'ibei_qtr'
    else:
        fieldName = 'ibei_ann'
    earn = modelDB.getFundamentalCurrencyItem(fieldName,
            startDate, modelDate, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    latestEarn = Utilities.extractLatestValueAndDate(earn)

    # Throw out forecast earnings earlier than latest realised date
    tmpEstEarn = []
    for (estE, realE) in zip(latestEstEPS, latestEarn):
        if realE is ma.masked or len(realE) < 1:
            eVals = estE
        elif (estE is ma.masked) or (len(estE) < 1) or (estE[0] <= realE[0]):
            eVals = None
        else:
            eVals = estE
        tmpEstEarn.append(eVals)
    latestEstEPS = tmpEstEarn

    # Scale by TSO and convert to ETP
    estETP = Matrices.allMasked(len(subIssues))
    for (idx, estE) in enumerate(latestEstEPS):
        sid = subIssues[idx]
        if (estE is ma.masked) or (estE is None) or (len(estE) < 1):
            continue
        tso = modelDB.loadTSOHistory([estE[-1]], [sid])
        estETP[idx] = estE[1] * tso.data[0,0] / divisor[idx]

    estETP = ma.array(Utilities.twodMAD(
                    estETP, nDev=[3.0, 3.0], estu=data.estimationUniverseIdx))

    # Report on missing values
    missing = numpy.flatnonzero(ma.getmaskarray(estETP))
    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estEarningsToPrice information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    myLog().info('generate_est_earnings_to_price: end')
    return estETP

def generate_estimate_enterprise_value(modelDate, data, model, modelDB, marketDB,
                                       daysBack=365, restrict=None,
                                       useQuarterlyData=False):
    """Compute the estimated enterprise value for the assets in data.universe.
    Data is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the enterprise value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_estimate_enterprise_value: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
        
    entValRaw = modelDB.getIBESCurrencyItemLegacy('ent_median_ann',
                                            startDate, modelDate, subIssues, modelDate, 
                                            marketDB, 
                                            convertTo=model.numeraire.currency_id)
    entVal = Utilities.extractLatestValue(entValRaw)
    missing = numpy.flatnonzero(ma.getmaskarray(entVal))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estimateEnterpriseValue information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))

    if restrict is not None:
        entValRest = Matrices.allMasked(len(data.universe))
        ma.put(entValRest, restrict, entVal)
        for (i,val) in enumerate(entVal):
            if val is ma.masked:
                entValRest[restrict[i]] = ma.masked
        entVal = entValRest
    myLog().info('generate_estimate_enterprise_value: end')
    return entVal

def generate_estimate_cash_flow_per_share(modelDate, data, model, modelDB, marketDB,
                                       daysBack=365, restrict=None,
                                       useQuarterlyData=False):
    """Compute the estimated cash flow per share for the assets in data.universe.
    Data is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the cash flow per share value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_estimate_cash_flow_per_share: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
        
    cfpsRaw = modelDB.getIBESCurrencyItemLegacy('cps_median_ann',
                                          startDate, modelDate, subIssues, modelDate, 
                                          marketDB, splitAdjust='divide',
                                          convertTo=model.numeraire.currency_id)
    cfps = Utilities.extractLatestValue(cfpsRaw)
    missing = numpy.flatnonzero(ma.getmaskarray(cfps))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estimateCashFlowPerShare information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))

    if restrict is not None:
        cfpsRest = Matrices.allMasked(len(data.universe))
        ma.put(cfpsRest, restrict, cfps)
        for (i,val) in enumerate(cfps):
            if val is ma.masked:
                cfpsRest[restrict[i]] = ma.masked
        cfps = cfpsRest
    myLog().info('generate_estimate_cash_flow_per_share: end')
    return cfps

def generate_est_return_on_equity(modelDate, data, model, modelDB, marketDB,
                              daysBackCE=(3*365), daysBackInc=(2*365),
                              restrict=None, useQuarterlyData=True, legacy=False):
    """Compute the return-on-equity for the given assets.
    The value is computed as the ratio of income and common equity.
    If estimated return-on-equity information is available the value is computed
    based on realised and estimated return-on-equity information, otherwise the
    value is computed using just realised return-on-equity information.
    Common equity and income are taken from the quarterly filings
    if useQuarterlyData is true, otherwise from the annual filings.
    Returns an array with the return-on-equity value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past three years (common 
    equity) and two years (income) of data are used.
    """
    myLog().info('generate_est_return_on_equity: begin')
    startDateCE = modelDate - datetime.timedelta(daysBackCE)
    endDateCE = modelDate - datetime.timedelta(365)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    if useQuarterlyData:
        fieldName0 = 'ce_qtr'
        fieldName1 = 'ibei_qtr'
    else:
        fieldName0 = 'ce_ann'
        fieldName1 = 'ibei_ann'
    ce = modelDB.getFundamentalCurrencyItem(fieldName0, 
            startDateCE, endDateCE, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    ce = Utilities.extractLatestValue(ce)
    startDate = modelDate - datetime.timedelta(daysBackInc)
    income = modelDB.getFundamentalCurrencyItem(fieldName1, 
            startDate, modelDate, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        income = modelDB.annualizeQuarterlyValues(income)
    else:
        income = Utilities.extractLatestValue(income)
    roe = (income / ce)*100
    if not legacy:
        roe = ma.where(ce <= 0.0, 0.0, roe)
    
    estROERaw = modelDB.getIBESCurrencyItemLegacy('roe_median_ann',
                                     startDate, modelDate, subIssues, modelDate, 
                                     marketDB, 
                                     convertTo=model.numeraire.currency_id)

    estROE = Utilities.extractLatestValue(estROERaw)
    roeList = [roe, estROE]
    estAveROE = ma.average(ma.array(roeList), axis=0)
    missing = numpy.flatnonzero(ma.getmaskarray(estAveROE))
    if len(missing) > 0:
        myLog().info('%d assets are missing return-on-equity information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([data.universe[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, estAveROE)
        for (i,j) in enumerate(estAveROE):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        estAveROE = val
    myLog().info('generate_est_return_on_equity: end')
    return estAveROE

def generate_est_return_on_assets(modelDate, data, model, modelDB, marketDB,
                              daysBackTA=(2*365), daysBackInc=(2*365),
                              restrict=None, useQuarterlyData=True, legacy=False):
    """Compute the return-on-assets for the given assets.
    The value is computed as the ratio of income and total assets.
    If estimated return-on-assets information is available the value is computed
    based on realised and estimated return-on-assets information, otherwise the
    value is computed using just realised return-on-assets information.
    Total assets and income are taken from the quarterly filings
    if useQuarterlyData is true, otherwise from the annual filings.
    Returns an array with the return-on-assets value for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years are used.
    """
    myLog().info('generate_est_return_on_assets: begin')
    startDateTA = modelDate - datetime.timedelta(daysBackTA)
    endDateTA = modelDate - datetime.timedelta(365)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    if useQuarterlyData:
        fieldName0 = 'at_qtr'
        fieldName1 = 'ibei_qtr'
    else:
        fieldName0 = 'at_ann'
        fieldName1 = 'ibei_ann'
    ta = modelDB.getFundamentalCurrencyItem(fieldName0, 
            startDateTA, endDateTA, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    totalAssets = Utilities.extractLatestValue(ta)
    totalAssets = ma.masked_where(totalAssets < 0.0, totalAssets)
    startDate = modelDate - datetime.timedelta(daysBackInc)
    income = modelDB.getFundamentalCurrencyItem(fieldName1, 
            startDate, modelDate, subIssues, modelDate,
            marketDB, convertTo=model.numeraire.currency_id)
    if useQuarterlyData:
        income = modelDB.annualizeQuarterlyValues(income)
    else:
        income = Utilities.extractLatestValue(income)
    roa = (income / totalAssets)*100
    if not legacy:
        roa = ma.where(totalAssets <= 0.0, 0.0, roa)
    
    estROARaw = modelDB.getIBESCurrencyItemLegacy('roa_median_ann',
                                 startDate, modelDate, subIssues, modelDate, 
                                 marketDB, 
                                 convertTo=model.numeraire.currency_id)

    estROA = Utilities.extractLatestValue(estROARaw)
    roaList = [roa, estROA]
    estAveROA = ma.average(ma.array(roaList), axis=0)
    missing = numpy.flatnonzero(ma.getmaskarray(estAveROA))
    if len(missing) > 0:
        myLog().info('%d assets are missing return-on-assets information',
                     len(missing))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([data.universe[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        val = Matrices.allMasked(len(data.universe))
        ma.put(val, restrict, estAveROA)
        for (i,j) in enumerate(estAveROA):
            if j is ma.masked:
                val[restrict[i]] = ma.masked
        estAveROA = val
    myLog().info('generate_est_return_on_assets: end')
    return estAveROA

def generate_estimate_revenue(modelDate, data, model, modelDB, marketDB,
                              daysBack=(2*365), restrict=None,
                              useQuarterlyData=False):
    """Compute the estimated revenue for the assets in data.universe.
    Data is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the estimated revenue for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_estimate_revenue: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]

    revenueRaw = modelDB.getIBESCurrencyItemLegacy('rev_median_ann',
                                         startDate, modelDate, subIssues, modelDate, 
                                         marketDB, 
                                         convertTo=model.numeraire.currency_id)
    revenue = Utilities.extractLatestValue(revenueRaw)
    missing = numpy.flatnonzero(ma.getmaskarray(revenue))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estimateRevenue information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        revenueRest = Matrices.allMasked(len(data.universe))
        ma.put(revenueRest, restrict, revenue)
        for (i,val) in enumerate(revenue):
            if val is ma.masked:
                revenueRest[restrict[i]] = ma.masked
        revenue = revenueRest
    myLog().info('generate_estimate_estimate_revenue: end')
    return revenue

def generate_estimate_net_income(modelDate, data, model, modelDB, marketDB,
                                 daysBack=365, restrict=None,
                                 useQuarterlyData=False):
    """Compute the estimated net income for the assets in data.universe.
    Data is taken from the quarterly filings if useQuarterlyData
    is true, otherwise from the annual filings.
    Returns an array with the estimated net income for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_estimate_net_income: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    netIncomeRaw = modelDB.getIBESCurrencyItemLegacy('net_median_ann',
                                               startDate, modelDate, subIssues, modelDate, 
                                               marketDB, 
                                               convertTo=model.numeraire.currency_id)
    netIncome = Utilities.extractLatestValue(netIncomeRaw)
    missing = numpy.flatnonzero(ma.getmaskarray(netIncome))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing estimateRevenue information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if restrict is not None:
        netIncomeRest = Matrices.allMasked(len(data.universe))
        ma.put(netIncomeRest, restrict, netIncome)
        for (i,val) in enumerate(netIncome):
            if val is ma.masked:
                netIncomeRest[restrict[i]] = ma.masked
        netIncome = netIncomeRest
    myLog().info('generate_estimate_net_income: end')
    return netIncome

def clone_linked_asset_exposures_test(rm, date, data, modelDB, marketDB,
                exposureNames=None, subIssueGroups=None):
    """Clones exposures for cross-listings/DRs etc.
    based on those of the most liquid/largest of each set
    and their degree of cointegration
    exposureNames is a list of factors to be adjusted. If set to None,
    all styles are treated.
    """
    myLog().info('clone_linked_asset_exposures: begin')
    expM = data.exposureMatrix
    if subIssueGroups is None:
        subIssueGroups = modelDB.getIssueCompanyGroups(
                date, data.universe, marketDB)
         
    # Get asset scores and cointegration stats from the DB
    scores = modelDB.loadISCScoreHistory(data.universe, [date])[:,-1]
    pvalueDict = modelDB.getISCPValues(date)

    # Pick out exposures to be cloned
    if exposureNames == None:
        exposureIdx = expM.getFactorIndices(ExposureMatrix.StyleFactor)
        exposureNames = expM.getFactorNames(ExposureMatrix.StyleFactor)
    else:
        exposureIdx = [expM.getFactorIndex(n) for n in exposureNames]
     
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
     
    if self.debuggingReporting:
        expos_ESTU = ma.take(ma.filled(expM.getMatrix(), 0.0), exposureIdx, axis=0)
        expos_ESTU = ma.take(expos_ESTU, data.estimationUniverseIdx, axis=1)
        wt_ESTU = ma.take(data.marketCaps, data.estimationUniverseIdx, axis=0)
        averageExposureBefore = ma.average(expos_ESTU, axis=1, weights=wt_ESTU)
     
    # Loop round sets of linked assets and pull out exposures
    for (groupId, subIssueList) in subIssueGroups.items():
        scores = [scoreDict[groupId][sid] for sid in subIssueList]
        outerIndices  = [data.assetIdxMap[n] for n in subIssueList]
        sortedSidList = [subIssueList[idx] for idx in numpy.argsort(-scores)]
         
        runningSet = list(subIssueList)
        subSetList = []
        # Divide set of linked assets into one or more subsets depending on 
        # closeness of their relationship
        while len(runningSet) > 0:
            rootSid = sortedSidList[0]
            subSet = [sid for sid in runningSet if pvalueDict[rootSid][sid] < TOL]
            runningSet = [sid for sid in runningSet if sid not in subSet]
            sortedSidList = [sid for sid in sortedSidList if sid not in subSet]
            subSetList.append(subSet)

        # Compute an average exposure for each subset
        exposDict = dict()
        for subset in subSetList:
            rootID = subset[0]
            weights = [pvalueDict[rootID][sid] * scores[sid] \
                    for sid in subIssueList]

            # Compute weighted average exposure for subgroup
            exposList = []
            for fIdx in exposureIdx:
                expos = ma.take(expM.getMatrix()[fIdx], outerIndices, axis=0)
                expos = ma.average(expos, weights=weights)
                exposList.append(expos)
     
    if self.debuggingReporting:
        expos_ESTU = ma.take(ma.filled(expM.getMatrix(), 0.0), exposureIdx, axis=0)
        expos_ESTU = ma.take(expos_ESTU, data.estimationUniverseIdx, axis=1)
        averageExposureAfter = ma.average(expos_ESTU, axis=1, weights=wt_ESTU)
        for (idx, n) in enumerate(exposureNames):
            self.log.info('Date: %s, Factor: %s, Mean Before Cloning: %8.6f, After: %8.6f',
                    date, n, averageExposureBefore[idx], averageExposureAfter[idx])
             
    self.log.info('clone_linked_asset_exposures: end')
    return data.exposureMatrix

def generate_short_interest(modelDate, data, model, modelDB, marketDB,
                            daysBack=(2*365), restrict=None, maskZero=False):
    """Compute the short interest for the assets in data.universe.
    Returns an array with the short interest, defined as: number of shares
    held short/total shares outstanding, for each asset.  
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_short_interest: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
    else:
        subIssues = [data.universe[i] for i in restrict]
    siRaw= modelDB.getXpressFeedItem('SHORTINT',
                                     startDate, modelDate, subIssues, modelDate, 
                                     marketDB, splitAdjust='divide')
    tso = modelDB.loadTSOHistory([modelDate], subIssues)
    si = Utilities.extractLatestValue(siRaw)/tso.data[:, 0]
    missing = numpy.flatnonzero(ma.getmaskarray(si))
    tmpData = ma.filled(si, 1.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpData==0.0, tmpData)))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing short interest information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))
    if len(zeroVals) > 0:
        myLog().info('%d (%.2f%%) assets have zero short interest',
                len(zeroVals), 100*float(len(zeroVals)))
        if maskZero:
            si = ma.masked_where(si==0.0, si)

    if restrict is not None:
        siRest = Matrices.allMasked(len(data.universe))
        ma.put(siRest, restrict, si)
        for (i,val) in enumerate(si):
            if val is ma.masked:
                siRest[restrict[i]] = ma.masked
        si = siRest
    myLog().info('generate_short_interest: end')
    return si

def generate_share_buyback(modelDate, data, model, modelDB, marketDB,
                            daysBack=(2*365), restrict=None, maskZero=False):
    """Compute the shares buyback for the assets in data.universe.
    Returns an array with the shares buyback, defined as: sum of number of 
    shares boughtback over the previous 4 quarters/ total shares outstanding, 
    for each asset.
    If data is not available for an asset, the corresponding value is masked.
    By default, the latest values from the past two years of data are used.
    """
    myLog().info('generate_shares_buyback: begin')
    startDate = modelDate - datetime.timedelta(daysBack)
    if restrict is None:
        subIssues = data.universe
        restrict = list(range(len(data.universe)))
    else:
        subIssues = [data.universe[i] for i in restrict]
    buyBackRaw= modelDB.getXpressFeedItem('CSHOPQ',
                                     startDate, modelDate, subIssues, modelDate, 
                                     marketDB)
    buyBack = modelDB.annualizeQuarterlyValues(buyBackRaw)
    tso = modelDB.loadTSOHistory([modelDate], subIssues)
    buyBack = buyBack * 1e6/ tso.data[:, 0]
    missing = numpy.flatnonzero(ma.getmaskarray(buyBack))
    tmpData = ma.filled(buyBack, 1.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpData==0.0, tmpData)))

    if len(missing) > 0:
        myLog().info('%d (%.2f%%) assets are missing shares buy back information',
                     len(missing), 100*float(len(missing))/float(len(data.universe)))
        myLog().debug('missing Axioma IDs: %s',
                      ', '.join([subIssues[i].getSubIDString()
                                 for i in missing]))

    if len(zeroVals) > 0:
        myLog().info('%d (%.2f%%) assets have zero buy back',
                len(zeroVals), 100*float(len(zeroVals)))
        if maskZero:
            buyBack = ma.masked_where(buyBack==0.0, buyBack)

    if restrict is not None:
        buyBackRest = Matrices.allMasked(len(data.universe))
        ma.put(buyBackRest, restrict, buyBack)
        for (i,val) in enumerate(buyBack):
            if val is ma.masked:
                buyBackRest[restrict[i]] = ma.masked
        buyBack = buyBackRest
    myLog().info('generate_shares_buyback: end')
    return buyBack

def generate_growth_rate(item, subids, modelDate, currency_id, 
                         modelDB, marketDB, 
                         daysBack =(5*365), useQuarterlyData=None,
                         robust=False):
    """Generic code to generate growth rate for particular fundamental data item.
    item - fundamental data item code
    subids - the universe you want to generate growth rate for,
    useQuarterlyData - None/True/False. If none, then it will use the getMixFundamentalCurrencyItem()
                       which will first look for quarterly data and if absent, find annual
    robust - True/False. If true, it will use robust regression to find the growth rate. OLS for false
    The return value should an array of regression coefficient for each sub_issue_id in subids
    """
    logging.info('generate_growht_rate for %s, robust- %s: begin',
                 item, robust)
    startDate = modelDate - datetime.timedelta(daysBack)

    if useQuarterlyData is not None:
        if useQuarterlyData:
            itemName = item + "_qtr"
        else:
            itemName = item + "_ann"
        values = modelDB.getFundamentalCurrencyItem(
            itemName, startDate, modelDate, subids, modelDate, 
            marketDB, convertTo=currency_id)
    else:
        values, valueFreq = modelDB.getMixFundamentalCurrencyItem(
            item, startDate, modelDate, subids, modelDate, 
            marketDB, convertTo=currency_id)

    slopes = Matrices.allMasked(len(subids))
    for i in range(len(subids)):
        if useQuarterlyData is not None:
            valueArray = numpy.array([n[1] for n in values[i]])
        else:
            valueArray = numpy.array([n[1] for n in values[i]])

        if len(valueArray) == 0:
            continue
        elif len(numpy.unique(valueArray)) <= 2:
            slopes[i] = 0.0
            continue
        if robust:
            coef = Utilities.robustLinearSolver(valueArray, numpy.transpose(numpy.array(
                        [list(range(1, len(valueArray)+1)), numpy.ones(len(valueArray))])), robust=True).params
        else:
            (coef, resid) = Utilities.ordinaryLeastSquares(
                valueArray, numpy.transpose(numpy.array(
                        [list(range(1, len(valueArray)+1)), numpy.ones(len(valueArray))])))
        slopes[i] = coef[0] / numpy.average(abs(valueArray))
    missing = numpy.flatnonzero(ma.getmaskarray(slopes))
    if len(missing) > 0:
        logging.info('%d assets are missin %s information',
                len(missing),item)
        logging.debug('missing Axioma IDs: %s',
                      ', '.join([subids[i].getSubIDString()
                                 for i in missing]))
    logging.info('generate_growht_rate for %s, robust- %s: end',
                 item, robust)
    return slopes

#####################################
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
    myLog().setLevel(logging.DEBUG)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    modelDate = datetime.date(2012,7,19)
    data = Utilities.Struct()
#    rmi = modelDB.getRiskModelInstance(riskModel.rms_id, modelDate)
#    data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
    data.universe = [ModelDB.SubIssue('DMVWA65M8311')]
    (mcapDates, goodRatio) = riskModel.getRMDates(
                            modelDate, modelDB, 20, ceiling=False)
    data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
    data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
    data.marketCaps = modelDB.getAverageMarketCaps(
            mcapDates, data.universe, riskModel.numeraire.currency_id, marketDB)
    from riskmodels import GlobalExposures
    data.issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
        data, modelDate, riskModel.numeraire, modelDB, marketDB)
    data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()


#    picklefileName='EstimatedEPS-'+riskModel.mnemonic+modelDate.isoformat()+'.csv'
#    picklefile = open(picklefileName,'wb')
#    pickle.dump(estEPS, picklefile)
#    picklefile.close()

#    picklefileIn = open(picklefileName,'rb')
#    estEPS = pickle.load(picklefileIn)
#    picklefileIn.close()
    modelDB.revertChanges()
    marketDB.finalize()
    modelDB.finalize()

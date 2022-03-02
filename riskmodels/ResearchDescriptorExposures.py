
import pandas as pd
import numpy
import numpy.ma as ma
import logging
import datetime
import bisect
from riskmodels import LegacyUtilities as Utilities
from riskmodels import Matrices
from riskmodels import Outliers
from riskmodels.DescriptorRatios import DataFrequency
from riskmodels import DescriptorExposures
from riskmodels import CurrencyRisk
from riskmodels import ProcessReturns

# candidate for deletion
def generate_growth_rateSD(item, subids, modelDate, currency_id, 
                         modelDB, marketDB, daysBack =(6*366),
                         robust=False, winsoriseRaw=False,
                         forecastItem=None, forecastItemScaleByTSO=False):
    """Generates growth rates from seasonally differenced data 
    for a particular fundamental data item code by:
    (1) retrieving 6 years of quarterly data for fundamental data item code (item); 
    (2) retrieving relevant forecast data item (forecastItem) for dates that are at
    least 90 days greater than the latest date for which fundamental data is available; 
    (3) appending one forecast observation to time series of fundamental data, if forecast
    exists within appropriate date range;
    (4) if time series of quarterly observations are complete (with no missing data), takes
    quarterly differences and regresses series of quarterly differences on time; computes 
    growth estimate as the estimated time coefficient divided by the average absolute quarterly
    difference

    Returns an array of growth estimates for each sub_issue_id in subids
    """
    logging.info('generate_growth_rateSD for %s, robust- %s: begin', item, robust)
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(366)
    slopes = Matrices.allMasked(len(subids))

    # Retrieve quarterly data
    values = modelDB.getFundamentalCurrencyItem(item + '_qtr' ,
                                                startDate, modelDate,
                                                subids, modelDate, marketDB,
                                                convertTo=currency_id)

    # Load forecast data if required
    fcArray = []
    fcArrayDates = []
    if forecastItem is not None:
        fcValues = DescriptorExposures.process_IBES_data_all(modelDate, 
                subids, startDate, endDate, modelDB, marketDB, 
                forecastItem, currency_id, item, days=90, 
                dropOldEstimates=True, scaleByTSO=forecastItemScaleByTSO)

        # Append forecast data element if it exists within appropriate timeframe
        for i in range(len(subids)):
            appendFc = False
            if (len(fcValues[i]) > 0) and (len(values[i]) > 0):
                # Get fcValue that is closest to the date that should follow the sequence of realized dates 
                latestDate = values[i][-1][0]
                targetDate = latestDate + datetime.timedelta(91) 

                fcIdx = bisect.bisect_left([n[0] for n in fcValues[i]], targetDate)
                if fcIdx == len(fcValues[i]):
                    fcIdx -= 1
                fcDate = fcValues[i][fcIdx][0]
                val = fcValues[i][fcIdx][1]
                if (fcDate > latestDate) and (fcDate < (latestDate + datetime.timedelta(400))):
                    appendFc = True
            if appendFc:
                # Append pseudo-quarterly forecast data
                fcArray.append([val/4.0])
                fcArrayDates.append([latestDate + datetime.timedelta(91)])
            else:
                fcArray.append([])
                fcArrayDates.append([])

    # Loop round each subissue in turn
    for i in range(len(subids)):
        valueArray = ma.filled(ma.array([n[1] for n in values[i]]), 0.0)
        valueDates = ma.array([n[0] for n in values[i]]) # ESTHER - what if a date field is empty? Probably impossible, but need to check

        # If history is too short, continue
        if len(valueArray) < 6:
            continue

        # Add forecast data if it exists
        if (forecastItem is not None) and (len(fcArray[i]) > 0):
            valueArray = numpy.append(valueArray, fcArray[i])
            valueDates = numpy.append(valueDates, fcArrayDates[i])

        qtrIntervals = [True if 85 <= (valueDates[j] - valueDates[j-1]).days <= 95 else False for j in range(1, len(valueDates))]
        if not all(qtrIntervals):
            continue

        # Create seasonally lagged time series 
        L = 4
        valueArray = valueArray[L::] - valueArray[0:-L]
        valueDates = valueDates[L::]
        if len(numpy.unique(valueArray)) < 2:
            slopes[i] = 0.0
            continue

        # Perform the regression - robust or not
        x_axis = list(range(1,len(valueArray)+1))

        if robust:
            coef = Utilities.robustLinearSolver(valueArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(valueArray))])), robust=True).params
        else:
            if winsoriseRaw:
                valueArray = Utilities.twodMAD(valueArray, nDev=[3.0, 3.0], axis=0,
                        suppressOutput=True)
            (coef, resid) = Utilities.ordinaryLeastSquares(
                valueArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(valueArray))])))
        slopes[i] = coef[0] / numpy.average(abs(valueArray))

    # Report on missing values
    missing = numpy.flatnonzero(ma.getmaskarray(slopes))
    if len(missing) > 0:
        logging.info('%d assets are missing %s information',
                len(missing),item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([subids[i].getSubIDString() for i in missing]))

    logging.info('generate_growth_rateSD for %s, robust- %s: end', item, robust)
    return slopes

# candidate for deletion
def generate_growth_rateOLD(item, subids, modelDate, currency_id, 
                         modelDB, marketDB, daysBack =(5*365),
                         robust=False, winsoriseRaw=True, forecastItem=None):
    """Generic code to generate growth rate for particular fundamental data item.
    item - fundamental data item code
    subids - the universe you want to generate growth rate for,
    useQuarterlyData - None/True/False. If none, then it will use the getMixFundamentalCurrencyItem()
                       which will first look for quarterly data and if absent, find annual
    robust - True/False. If true, it will use robust regression to find the growth rate. OLS for false
    The return value should an array of regression coefficient for each sub_issue_id in subids
    """
    logging.info('generate_growth_rateOLD for %s, robust- %s: begin', item, robust)
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(365)
    slopes = Matrices.allMasked(len(subids))

    # Load fundamental data: quarterly, annual or a mix
    values, valueFreq = modelDB.getMixFundamentalCurrencyItem(
        item, startDate, modelDate, subids, modelDate, 
        marketDB, convertTo=currency_id)

    isMixed = True
    if len(numpy.unique(valueFreq)) < 2:
        isMixed = False

    # Load forecast data if required
    fcArray = []
    if forecastItem is not None:
        fcValues, fcDates = process_IBES_data_legacy(modelDate, subids, startDate,
                endDate, modelDB, marketDB, forecastItem, currency_id,
                item, scaleData=False)

        # Report on missing values
        fcMissingIdx = numpy.flatnonzero(ma.getmaskarray(fcValues))
        if len(fcMissingIdx) > 0:
            pct = 100.0 * len(fcMissingIdx) / float(len(subids))
            logging.info('%d out of %d (%.2f%%) assets are missing forecast %s',
                    len(fcMissingIdx), len(subids), pct, forecastItem)

        # Create list of forecast values
        for i in range(len(subids)):
            if (i not in fcMissingIdx) and len(values[i]) > 0:
                fcDate = fcDates[i]
                latestDate = values[i][-1][0]
                val = fcValues[i]
                if fcDate > latestDate:
                    if valueFreq[i] == DataFrequency('_qtr'):
                        # Generate pseudo-quarterly forecast data
                        if fcDate - latestDate <= datetime.timedelta(91):
                            fcArray.append([val/4.0])
                        elif datetime.timedelta(92) <= fcDate - latestDate <= datetime.timedelta(182):
                            fcArray.append([val/4.0]*2)
                        elif datetime.timedelta(183) <= fcDate - latestDate <= datetime.timedelta(266):
                            fcArray.append([val/4.0]*3)
                        elif datetime.timedelta(267) <= fcDate - latestDate:
                            fcArray.append([val/4.0]*4)
                    else:
                        # Or simple annual data
                        fcArray.append([val])
                else:
                    fcArray.append([])
            else:
                fcArray.append([])

    # Loop round each subissue in turn
    for i in range(len(subids)):
        valueArray = ma.filled(ma.array([n[1] for n in values[i]]), 0.0)

        # Deal with assets whose histories are too short
        if len(valueArray) == 0:
            continue
        elif len(numpy.unique(valueArray)) <= 2:
            slopes[i] = 0.0
            continue

        # Add forecast data if any
        if (forecastItem is not None) and (len(fcArray[i]) > 0):
            valueArray = numpy.append(valueArray, fcArray[i])

        # Perform the regression - robust or not
        x_axis = list(range(1, len(valueArray)+1))
        if robust:
            coef = Utilities.robustLinearSolver(valueArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(valueArray))])), robust=True).params
        else:
            if winsoriseRaw:
                valueArray = Utilities.twodMAD(valueArray, nDev=[3.0, 3.0], axis=0,
                        suppressOutput=True)
            (coef, resid) = Utilities.ordinaryLeastSquares(
                valueArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(valueArray))])))
        slopes[i] = coef[0] / numpy.average(abs(valueArray))
        if isMixed and (valueFreq[i] == DataFrequency('_qtr')):
            slopes[i] *= 4.0

    # Report on missing values
    missing = numpy.flatnonzero(ma.getmaskarray(slopes))
    if len(missing) > 0:
        logging.info('%d assets are missing %s information',
                len(missing),item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s',
                      ', '.join([subids[i].getSubIDString()
                                 for i in missing]))

    logging.info('generate_growth_rateOLD for %s, robust- %s: end',
                 item, robust)
    return slopes

def process_IBES_data_legacy(date, subIssues, startDate, endDate, modelDB, marketDB,
                      ibesField, currency_id, realisedField, scaleData=False):
    """Logic to process IBES data and give us best available
    estimate, neither too distant in the past or far in the future
    """
    # Pull out IBES data
    fcDataRaw = modelDB.getIBESCurrencyItem(ibesField,
                startDate, endDate, subIssues, date, marketDB,
                convertTo=currency_id)

    # Drop out-of-date estimates
    fcData = []
    for est in fcDataRaw:
        if (est is not None) and (est is not ma.masked) and len(est) > 0:
            eVals = [ev for ev in est if ev[0] >= startDate]
        else:
            eVals = []
        fcData.append(eVals)

    # Load realised data for comparison
    (rlDataRaw, frequencies) = modelDB.getMixFundamentalCurrencyItem(
                               realisedField, startDate, date, subIssues,
                               date, marketDB, currency_id)
    rlData = Utilities.extractLatestValueAndDate(rlDataRaw)
    realDates = []
    for rl in rlData:
        if (rl is not None) and (rl is not ma.masked) and len(rl) > 0:
            realDates.append(rl[0])
        else:
            realDates.append(None)

    # Throw out forecast earnings less than 90 days after latest realised date
    tmpEstData = []
    for (est, dt) in zip(fcData, realDates):
        if (dt is None):
            eVals = est
        else:
            eVals = [ev for ev in est if ev[0] > dt + datetime.timedelta(90)]
        tmpEstData.append(eVals)
    fcData = tmpEstData

    # Take latest value and compute optional scaling
    fcData = Utilities.extractLatestValuesAndDate(fcData)
    scale = []
    if scaleData:
        for (est, dt) in zip(fcData, realDates):
            if (dt is None) or (est is None) or (est is ma.masked) or len(est) < 1:
                scale.append(1.0)
            elif (est[0] - dt) < datetime.timedelta(270):
                scl = (est[0] - dt).days / 365.0
                scale.append(scl)
            else:
                scale.append(1.0)

    # Get list of dates
    dateList = []
    for est in fcData:
        if (est is ma.masked) or (est is None) or (len(est) < 1):
            dateList.append(None)
        else:
            dateList.append(est[0])

    # Scale by TSO
    useLatestTSO = False
    if not useLatestTSO:
        values = Matrices.allMasked(len(subIssues))
        for (idx, est) in enumerate(fcData):
            sid = subIssues[idx]
            if (est is ma.masked) or (est is None) or (len(est) < 1):
                continue
            tso = modelDB.loadTSOHistory([est[-1]], [sid])
            values[idx] = est[1] * tso.data[0,0]
    else:
        tso = modelDB.loadTSOHistory([date], subIssues)
        fcData = [est[1] for est in fcData]
        values = ma.array(fcData) * tso.data[:, 0]

    if len(scale) > 0:
        values *= scale
    return values, dateList

# candidate for deletion
def generate_growth_rate(item, subids, modelDate, currency_id, 
                         modelDB, marketDB, daysBack=int(5.833*366),
                         robust=False, winsoriseRaw=False, 
                         useQuarterlyData=None, forecastItem=None, 
                         forecastItemScaleByTSO=False, getVar=False, 
                         debug=False, legacy=False):
    """Generic code to generate growth rate for particular fundamental 
    data item.
    item - fundamental data item code
    subids - the universe you want to generate growth rate for,
    useQuarterlyData - None/True/False. If none, then it will use the 
        getMixFundamentalCurrencyItem(), which will first look for 
        quarterly data and if absent, it will retrieve annual data
    robust - True/False. If true, it will use robust regression to 
        find the growth rate, otherwise OLS is used
    Returns an array of regression coefficients divided by the average
    absolute item value
    """
    logging.info('generate_growth_rate for %s, robust- %s: begin', 
            item, robust)
    startDate = modelDate - datetime.timedelta(daysBack)
    endDate = modelDate + datetime.timedelta(366)
    slopes = Matrices.allMasked(len(subids))
    outlierClass = Outliers.Outliers()

    # Load realized fundamental data 
    if useQuarterlyData is None: # retrieve mix of annual and quarterly data 
        values, valueFreq = modelDB.getMixFundamentalCurrencyItem(item, 
                startDate, modelDate, subids, modelDate, marketDB, 
                convertTo=currency_id)

    else: # retrieve either annual or quarterly data
        freq = DataFrequency('_qtr') if useQuarterlyData==True \
                else DataFrequency('_ann')
        freqStr = '_qtr' if useQuarterlyData==True else '_ann'
        valuesTmp = modelDB.getFundamentalCurrencyItem(item + freqStr,
                startDate, modelDate, subids, modelDate, marketDB,
                convertTo=currency_id)
        values = Matrices.allMasked(len(subids), dtype=object)
        valueFreq = Matrices.allMasked(len(subids), dtype=object)
        for idx in range(len(subids)):
            values[idx] = valuesTmp[idx]
            valueFreq[idx] = freq

    # Load forecast data if required
    fcArray = []
    fcArrayDates = []
    if forecastItem is not None:
        fcValues = DescriptorExposures.process_IBES_data_all(modelDate, subids, startDate,
                endDate, modelDB, marketDB, forecastItem, currency_id,
                item, days=90, dropOldEstimates=True,
                scaleByTSO=forecastItemScaleByTSO,
                useAnnualRealisedData=(useQuarterlyData==False))

        # Create list of forecast values
        for i in range(len(subids)):
            if (len(fcValues[i]) > 0) and (len(values[i]) > 0):
                # Get fcValue that is closest to the date that should 
                # follow the sequence of realized dates 
                latestDate = values[i][-1][0]
                targetDate = latestDate + \
                        datetime.timedelta(91) \
                        if valueFreq[i]==DataFrequency('_qtr') \
                        else latestDate+datetime.timedelta(366)

                fcIdx = bisect.bisect_left([n[0] for n in fcValues[i]], 
                                           targetDate)
                if fcIdx == len(fcValues[i]):
                    fcIdx -= 1
                fcDate = fcValues[i][fcIdx][0]
                val = fcValues[i][fcIdx][1]

                if (fcDate > latestDate) and \
                        (fcDate < (latestDate+datetime.timedelta(400))):
                    if valueFreq[i] == DataFrequency('_qtr'):
                        # Append pseudo-quarterly forecast data
                        fcArray.append([val/4.0])
                        fcArrayDates.append([latestDate+datetime.timedelta(91)])
                    else:
                        # Append annual data
                        fcArray.append([val])
                        fcArrayDates.append([fcDate])
                else:
                    fcArray.append([])
                    fcArrayDates.append([])
            else:
                fcArray.append([])
                fcArrayDates.append([])

    if getVar:
        allDates = modelDB.getDateRange(None, startDate, endDate)
        allTSO = modelDB.loadTSOHistory(allDates, subids).data
        dateIdxMap = dict(zip(allDates, list(range(len(allDates)))))

    if debug:
        regData = {}
    # Loop round each subissue in turn
    for i in range(len(subids)):
        valueDict = dict([n[:2] for n in values[i]])
        if len(valueDict) == 0:
            continue
        # backwards compatibility - if there are <=2 unique values
        # in the last 5 years, continue
        tmpDateArray = [n[0] for n in values[i] 
                       if (modelDate-n[0]).days/365.<=5.]
        tmpValueArray = numpy.array([valueDict[dt] for dt in tmpDateArray])
        if len(tmpValueArray) == 0:
            continue
        elif len(numpy.unique(tmpValueArray)) <= 2:
            if legacy:
                slopes[i] = 0.0
            continue
        if useQuarterlyData:
            dateArray = [n[0] for n in values[i]
                         if (values[i][-1][0] - n[0]).days/365. <= 4.8]
        else:
            dateArray = [n[0] for n in values[i]
                         if (values[i][-1][0] - n[0]).days/365. <= 4.05]
        valueArray = numpy.array([valueDict[dt] for dt in dateArray])
        valueDates = dateArray
        if len(numpy.unique(valueArray)) <= 2:
            if legacy:
                slopes[i] = 0.0
            continue
        if debug:
            numRealObs = len(valueArray)
            maxRealDt = max(valueDates)
            minRealDt = min(valueDates)

        # Add forecast data if any
        if (forecastItem is not None) and (len(fcArray[i]) > 0):
            valueArray = numpy.append(valueArray, fcArray[i])
            valueDates = numpy.append(valueDates, fcArrayDates[i])
            if debug:
                numFcstObs = len(fcArray[i])
        else:
            if debug:
                numFcstObs = 0
        # Perform the regression - robust or not
        x_axis = numpy.cumsum([(valueDates[n]-valueDates[n-1]).days 
                if n >= 1 else 365 for n in range(0, len(valueDates))])/365.0

        if getVar:
            tsoDates = sorted(list(valueDates))
            tsoDateIdx = [dateIdxMap[d] for d in tsoDates]
            tsoData = ma.take(allTSO[i], tsoDateIdx, axis=0)
            valueArray = 1.0e6 * valueArray / tsoData
            slopes[i] = ma.sqrt(ma.sum(valueArray*valueArray, axis=None) / \
                    float(len(valueArray)-1))
        else:
            if robust:
                coef = Utilities.robustLinearSolver(valueArray, 
                        numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(valueArray))])), 
                        robust=True).params
            else:
                if winsoriseRaw:
                    valueArray = outlierClass.twodMAD(valueArray, 
                            axis=0, suppressOutput=True)
                (coef, resid) = Utilities.ordinaryLeastSquares(
                        valueArray, numpy.transpose(numpy.array(
                        [x_axis, numpy.ones(len(valueArray))])))
            slopes[i] = coef[0] / numpy.average(abs(valueArray))
        if debug:
            regData[subids[i].getSubIdString()] = \
                    {'numRealObs': numRealObs,
                     'numFcstObs': numFcstObs,
                     'maxRealDt': maxRealDt,
                     'minRealDt': minRealDt}
    if debug:
        regDataDF = pd.DataFrame.from_dict(regData, orient='index')
        regDataDF.to_csv('tmp/regData-generate_growth_rate-%s-%s.csv' % \
                (item, modelDate.isoformat()))
        targetN = 20 if useQuarterlyData else 5
        logging.debug('%.3f of %s growth rates estimated from %.0f obs', \
                (regDataDF.numRealObs==targetN).sum()/float(len(regDataDF))*100., item, targetN)
        logging.debug('%.3f of %s growth rates estimated from <%.0f obs',\
                (regDataDF.numRealObs<targetN).sum()/float(len(regDataDF))*100., item, targetN)
        logging.debug('%.3f of %s growth rates estimated from >%.0f obs', \
                (regDataDF.numRealObs>targetN).sum()/float(len(regDataDF))*100., item, targetN)
    # Report on missing values
    missing = numpy.flatnonzero(ma.getmaskarray(slopes))
    if len(missing) > 0:
        logging.info('%d assets are missing %s information', len(missing),item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s',
                      ', '.join([subids[i].getSubIDString() for i in missing]))

    tmpArray = ma.filled(slopes, -999.0)
    zeroVals = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(tmpArray==0.0, 
            tmpArray)))
    if len(zeroVals) > 0.01 * len(slopes):
        logging.info('%d assets have zero values for %s growth', 
                len(zeroVals), item)

    if getVar:
        logging.info('Getting variability coefficient')
    logging.info('generate_growth_rate for %s, robust- %s: end', item, robust)
    return slopes

def generate_long_term_growth_rate(forecastItem, subids, modelDate, currency_id,
                                   modelDB, marketDB, daysBack=(2*366), trim=False):
    """Generic code to retrieve long term growth rates

    Note that long-term growth rates represent an expected annual increase
    in operating earnings over the company's next full business cycle.
    These forecasts refer to a period of betweeh three and five years, and
    are expressed as a percentage.

    If trim=True, trim data cross-sectionally at -300/+300.

    TO DO: for regional models, ensure that long term growth percentages are
    converted to appropriate numeraire currency.

    Returns a masked array of long term growth rates

    """
    logging.info('generate_long_term_growth_rate: begin')

    startDate = modelDate - datetime.timedelta(daysBack)
    endDate   = modelDate

    # Pull out IBES data
    fcDataRaw = modelDB.getIBESCurrencyItem(forecastItem,
                                            startDate, endDate, subids, modelDate, marketDB,
                                            convertTo=None)


    # Drop out-of-date estimates
    fcData = []
    for est in fcDataRaw:
        if (est is not None) and (est is not ma.masked) and len(est) > 0:
            eVals = [ev for ev in est if ev[0] >= startDate]
        else:
            eVals = []
        fcData.append(eVals)

    # Extract latest estimate
    fcData = Utilities.extractLatestValuesAndDate(fcData)
    values = ma.array([est[1] if est is not ma.masked else ma.masked for est in fcData])
    if trim:
        values = values.clip(-300, 300)

    logging.info('generate_long_term_growth_rate: end')
    return values

def generate_evol(subids, modelDate, currency_id, modelDB, marketDB,
                    daysBack =(5*366), useQuarterlyData=None,
                    item='evol', numItem='ibei', denomItem='ce'):
    """Computes a series of ROE values over a period of 5 years and computest the standard deviation
    item - fundamental data item code
    subids - the universe you want to generate growth rate for,
    useQuarterlyData - None/True/False. If none, then it will use the getMixFundamentalCurrencyItem()
                       which will first look for quarterly data and if absent, find annual
    The return value should an array of regression coefficient for each sub_issue_id in subids
    """
    logging.info('generate_%s: begin', item)
    startDate = modelDate - datetime.timedelta(daysBack)
    denomStartDate = startDate - datetime.timedelta(366)
    slopes = Matrices.allMasked(len(subids))
    outlierClass = Outliers.Outliers()

    # Load realized fundamental data
    if useQuarterlyData is None: # retrieve mix of annual and quarterly data
        numer, numerFreq = modelDB.getMixFundamentalCurrencyItem(
                        numItem, startDate, modelDate, subids, modelDate, marketDB,
                        convertTo=currency_id)
        denom, denomFreq = modelDB.getMixFundamentalCurrencyItem(
                        denomItem, denomStartDate, modelDate, subids, modelDate, marketDB,
                        convertTo=currency_id)

    else: # retrieve either annual or quarterly data
        if useQuarterlyData==True:
            freq = DataFrequency('_qtr')
            freqStr = '_qtr'
        else:
            freq = DataFrequency('_ann')
            freqStr = '_ann'

        values = modelDB.getFundamentalCurrencyItem(
                    numItem + freqStr, startDate, modelDate,
                    subids, modelDate, marketDB, convertTo=currency_id)
        denVals = modelDB.getFundamentalCurrencyItem(
                    denomItem + freqStr, denomStartDate, modelDate,
                    subids, modelDate, marketDB, convertTo=currency_id)

        numer = Matrices.allMasked(len(subids), dtype=object)
        denom = Matrices.allMasked(len(subids), dtype=object)
        numerFreq = Matrices.allMasked(len(subids), dtype=object)

        for idx in range(len(subids)):
            numer[idx] = values[idx]
            denom[idx] = denVals[idx]
            numerFreq[idx] = freq

    # Loop round each subissue in turn
    for i in range(len(subids)):

        # Pick out most relevant denominator per numerator value
        numerArray = [n[1] for n in numer[i]]
        denomArray = ma.array([n[1] for n in denom[i]])
        denomArray = ma.masked_where(denomArray<=0.0, denomArray)
        numerDates = [n[0] for n in numer[i]]
        denomDates = [n[0] for n in denom[i]]
        dnDateIdx = dict(zip(denomDates, list(range(len(denomDates)))))
        roeList = []
        for (nDt, num) in zip(numerDates, numerArray):
            dnDts = sorted(d for d in denomDates if (d<nDt) and (nDt-d)<datetime.timedelta(366))
            if len(dnDts) > 0:
                roe = num / denomArray[dnDateIdx[dnDts[-1]]]
                if roe is not ma.masked:
                    roeList.append(roe)

        # Deal with assets whose histories are too short
        if len(roeList) == 0:
            continue
        elif numerFreq[i] == DataFrequency('_ann') and (len(roeList) < 5):
            continue
        elif numerFreq[i] == DataFrequency('_qtr') and (len(roeList) < 12):
            continue

        # Compute the standard deviation of whatever's left
        slopes[i] = numpy.std(roeList)

    # Report on missing values
    missing = numpy.flatnonzero(ma.getmaskarray(slopes))
    if len(missing) > 0:
        logging.info('%d assets are missing %s information', len(missing),item)
        if len(missing) < 10:
            logging.debug('missing Axioma IDs: %s', ', '.join([subids[i].getSubIDString() for i in missing]))

    logging.info('generate_%s: end', item)
    return slopes

def generate_forex_sensitivity(data, returns, modelSelector, modelDB, marketDB, pm, debugOutput=False):
    """Compute assets' sensitivity to currency returns.
    Data are lagged by an interval given by the lag parameter
    """
    logging.info('generate_forex_sensitivity: begin')
    debugOutput = False

    # Set up parameters
    logging.info('Computing forex sensitivities vs. %s for %d days %s frequency',
            pm.numeraire, pm.daysBack, pm.frequency)
    daysBack = pm.daysBack
    numeraire = pm.numeraire
    frequency = pm.frequency
    if hasattr(pm, 'marketFactor'):
        marketFactor = pm.marketFactor
    else:
        marketFactor = True
    if hasattr(pm, 'robust'):
        robustBeta = pm.robust
    else:
        robustBeta = False
    if hasattr(pm, 'k_val'):
        k = pm.k_val
    else:
        k = 1.345

    # Restrict history
    assert(returns.data.shape[1] >= daysBack)
    assetReturns = returns.data[:,-daysBack:]
    assetIdxMap = dict(zip(returns.assets, range(returns.data.shape[0])))
    dateList = returns.dates[-daysBack:]
    allDateList = modelDB.getDateRange(None, min(dateList), max(dateList))
    beta = Matrices.allMasked(assetReturns.shape[0])

    # These currencies we need not worry about
    ignoreList = CurrencyRisk.getPeggedCurrencies(numeraire, dateList[0], dateList[-1])

    # Load the risk model group's currency returns versus the given numeraire currency
    rmgList = modelSelector.rmg
    currencies = list(set([r.currency_code for r in rmgList if r.currency_code not in ignoreList]))
    currencyIdxMap = dict([(code, idx) for (idx, code) in enumerate(currencies)])
    fxReturns = modelDB.loadCurrencyReturnsHistory(rmgList, dateList[-1], daysBack-1,
            currencies, numeraire, dateList=allDateList)
    if debugOutput:
        Utilities.writeToCSV(fxReturns.data, 'tmp/curRetDaily0-%s.csv' % dateList[-1], columnNames=fxReturns.dates)
    if fxReturns.dates != dateList:
        (fxReturns, pdList) = ProcessReturns.compute_compound_returns_v3(fxReturns.data, fxReturns.dates, dateList, matchDates=True)
    else:
        fxReturns = fxReturns.data
    if debugOutput:
        Utilities.writeToCSV(fxReturns.data, 'tmp/curRetDaily1-%s.csv' % dateList[-1], columnNames=dateList)

    # Load in the market returns if required
    if marketFactor:
        mktDateList = modelDB.getDateRange(None, min(dateList), max(dateList))
        marketReturns = modelDB.loadRMGMarketReturnHistory(allDateList, rmgList).data
        if debugOutput:
            Utilities.writeToCSV(marketReturns, 'tmp/mktRetDaily0-%s.csv' % dateList[-1], columnNames=allDateList)
        if allDateList != dateList:
            marketReturns = ProcessReturns.compute_compound_returns_v3(
                    marketReturns, mktDateList, dateList, keepFirst=True)[0]
            if debugOutput:
                Utilities.writeToCSV(marketReturns, 'tmp/mktRetDaily1-%s.csv' % dateList[-1], columnNames=dateList)

    # Compound returns if necessary
    if frequency == 'weekly':
        periodDateList = [prv for (prv, nxt) in zip(dateList[:-1], dateList[1:])
                if nxt.weekday() < prv.weekday()]
    elif frequency == 'monthly':
        periodDateList = [prv for (prv, nxt) in zip(dateList[:-1], dateList[1:])
                if nxt.month > prv.month or nxt.year > prv.year]
    else:
        periodDateList = list(dateList)

    if dateList != periodDateList:
        (assetReturns, pdList) = ProcessReturns.compute_compound_returns_v3(
                        assetReturns, dateList, periodDateList, matchDates=True)
        (fxReturns, pdList) = ProcessReturns.compute_compound_returns_v3(
                        fxReturns, dateList, periodDateList, matchDates=True)
        if marketFactor:
            (marketReturns, pdList) = ProcessReturns.compute_compound_returns_v3(
                    marketReturns, dateList, periodDateList, matchDates=True)
        periodDateList = pdList

    for (i,r) in enumerate(rmgList):
        hasMarket = marketFactor
        # Disregard pegged currencies
        if r.currency_code in ignoreList:
            continue

        # Determine list of assets for each market
        if len(rmgList) > 1:
            rmg_assets = Utilities.readMap(r, data.rmgAssetMap)
            indices = [assetIdxMap[n] for n in rmg_assets]
        else:
            indices = list(range(len(returns.assets)))

        # Fetch corresponding currency and asset returns
        if len(indices) > 0 and not (len(indices) <= 1 and marketFactor):
            rmgReturns = ma.take(assetReturns, indices, axis=0)
            currencyReturns = fxReturns[currencyIdxMap[r.currency_code]]

            # Compound returns for non-trading days into the following trading-day
            rmgCalendarSet = set(modelDB.getDateRange(r, periodDateList[0], periodDateList[-1]))
            tradingDates = [d for d in periodDateList if d in rmgCalendarSet]
            if len(tradingDates) < 1:
                logging.info('Warning %s, no valid trading dates', r.description)
            rmgReturns = ProcessReturns.compute_compound_returns_v3(
                    rmgReturns, periodDateList, tradingDates, keepFirst=True, matchDates=True)[0]
            currencyReturns = ProcessReturns.compute_compound_returns_v3(
                    currencyReturns, periodDateList, tradingDates, keepFirst=True, matchDates=True)[0]
            if debugOutput:
                Utilities.writeToCSV(currencyReturns, 'tmp/curRetPeriod-%s.csv' % tradingDates[-1], rowNames=tradingDates)

            # Warn if all currency returns are missing/zero
            tmpArray = ma.masked_where(currencyReturns == 0.0, currencyReturns)
            if numpy.sum(ma.getmaskarray(tmpArray)) == len(currencyReturns):
                ma.put(beta, indices, 0.0)
                logging.info('Skipping %s, zero %s currency returns', r.description, r.currency_code)
                continue

            # Add market factor to the mix if required
            if marketFactor:
                marketReturn = ProcessReturns.compute_compound_returns_v3(
                        marketReturns[i], periodDateList, tradingDates, keepFirst=True, matchDates=True)[0]
                if debugOutput:
                    Utilities.writeToCSV(marketReturn, 'tmp/mktRetPeriod-%s.csv' % tradingDates[-1], rowNames=tradingDates)
                # Warn if all market returns are missing/zero
                tmpArray = ma.masked_where(marketReturn == 0.0, marketReturn)
                if numpy.sum(ma.getmaskarray(tmpArray)) == len(marketReturn):
                    logging.info('Warning %s, all zero market returns', r.description)
                    hasMarket = False

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

            if pm.weights is None:
                weights = numpy.ones((t), float)
            elif pm.weights.lower() == 'exponential':
                weights = Utilities.computeExponentialWeights(pm.peak, t)
                weights.reverse
            elif pm.weights.lower() == 'pyramid':
                weights = Utilities.computePyramidWeights(pm.peak, pm.peak2, t)

            # Compute beta
            if hasMarket:
                x = numpy.transpose(ma.array(
                    [ma.ones(t, float), marketReturn, ma.filled(currencyReturns, 0.0)]))
                offset = 1
            else:
                x = numpy.transpose(ma.array(
                    [ma.ones(t, float), ma.filled(currencyReturns, 0.0)]))
                offset = 0
            y = numpy.transpose(ma.filled(rmgReturns, 0.0))
            res = Utilities.robustLinearSolver(y, x, robust=robustBeta, k=k, weights=weights, computeStats=False)
            b0 = res.params
            values = b0[1+offset,:]
            ma.put(beta, indices, values)
        else:
            if len(indices) == 1:
                ma.put(beta, indices, 0.0)
            logging.info('Skipping %s, too few assets (%s)', r.description, len(indices))

    logging.info('generate_forex_sensitivity: end')
    return beta

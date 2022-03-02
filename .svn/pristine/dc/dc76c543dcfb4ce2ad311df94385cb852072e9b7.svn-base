import copy
import datetime
import logging
import numpy
from math import sqrt
from numpy import ma as ma

from riskmodels import ModelDB, Utilities


def getSubFactorForDateRange(dateList, factor, modelDB):
    """
    Returns the sub-factors of the given factors which are active
    at some point during the dateList.
    The return value is a SubFactorRange object describing the subfactor
    and its start and end dates
    """
    query = """SELECT sub_id,from_dt,thru_dt FROM sub_factor
    WHERE from_dt <= :max_dt AND thru_dt > :min_dt
    AND factor_id = :factor_arg"""
    sf = None
    from_dt = None
    thru_dt = None
    min_dt = min(dateList)
    max_dt = max(dateList)
    modelDB.dbCursor.execute(query, factor_arg=factor.factorID,
                             min_dt=min_dt, max_dt=max_dt)
    r = modelDB.dbCursor.fetchall()
    if not r:
        logging.warning('No active subfactor for factor %s', factor)
    else:
        sf = ModelDB.ModelSubFactor(factor, r[0][0], min_dt)
        from_dt = r[0][1].date()
        thru_dt = r[0][2].date()
    return sf, from_dt, thru_dt


def writeStatFactorMatrix(riskModel, dt, modelDB, factorFile, delimiter='|',standardHeader=True, dateformat='%Y-%m-%d'):
    """
        Writes all N (=250 now) days worth of factor returns for the specified stat model for the given date
        to factorfile
    """
    rms = riskModel.rms_id
    modelDB.dbCursor.execute("""select from_dt, thru_dt from risk_model_serie rms
                                where serial_id=:rms_id""", rms_id=rms)
    r = modelDB.dbCursor.fetchall()
    assert len(r) == 1, "No factors for model %s found!" % riskModel.name
    revision = r[0]

    # Get a list of factors that are active within the risk model serie's lifespan
    factors = sorted(modelDB.getRiskModelSerieFactors(rms), key=lambda x: x.name)
    factors = [f for f in factors if f.from_dt<=dt and dt<f.thru_dt]
    if not factors:
        logging.warning("No factors for %s between %s and %s", riskModel.name, f.from_dt, f.thru_dt)
        return

    # cheat and get the data that are present in this exp_dt.  Ugly, could be done better
    query="""select ret.exp_dt, ret.dt, f.name, ret.value
                    from rms_%(rms)d_stat_factor_return ret,  rms_factor rf, sub_factor sf, factor f
                    where ret.exp_dt = :dt
                    and ret.rms_id = %(rms)d
                    and ret.sub_factor_id = sf.sub_id     and sf.from_dt <= :dt and :dt < sf.thru_dt
                    and sf.factor_id      = rf.factor_id  and rf.from_dt <= :dt and :dt < rf.thru_dt
                    and  f.factor_id      = rf.factor_id
              order by ret.dt """ % {'rms':rms}

    modelDB.dbCursor.execute(query, dt=dt)

    results = modelDB.dbCursor.fetchall()
    # set up a result dictionary that is indexed by ret_dt and factorName
    resDict = {}
    for exp_dt, ret_dt, factorName, value in results:
        if ret_dt not in resDict:
            resDict[ret_dt] = {}
        resDict[ret_dt][factorName] = '%.12f' % (100.0 * value) if value else '0.0'

    if standardHeader:
        factorFile.write('#Columns: Date|%s\n' % (
            "|".join([f.name for f in factors])))
        for ret_dt in sorted(resDict.keys()):
            factorFile.write('%s|%s\n' % (str(ret_dt)[:10], '|'.join([resDict[ret_dt][f.name] for f in factors])))
    else:
        factorFile.write(delimiter.join(['Date'] + ['"%s"' % f.name for f in factors]))
        factorFile.write('\n')

    return


def getRMCalendar(riskModel, modelDB, fromDt, thruDt, freq='daily'):
    """
    Returns a list of period-end calendar for the risk model, frequency is monthly or weekly.
    The return value is a sorted list of datetime object.
    """
    RMCalendar = set()
    for rmg in riskModel.rmg:
        if hasattr(rmg, 'from_dt'):
            myFromDt = rmg.from_dt
        else:
            myFromDt = min([d['from_dt'] for d in rmg.timeVariantDicts])
        if myFromDt <= fromDt:
            tradingDt = [i for i in modelDB.getDateRange(rmg, fromDt, thruDt)
                         if i.weekday() not in (5, 6)]
            RMCalendar.update(set(tradingDt))

    RMCalendar = sorted(RMCalendar)
    if freq != 'daily':
        RMCalendar = Utilities.change_date_frequency(RMCalendar, frequency=freq)

    return RMCalendar


def writeFactorMatrix(riskModel, startDate, endDate, modelDB, factorFile,
                      volatilityFile = None, returnfreq='daily', delimiter=',', dateformat='%m/%d/%Y',
                      standardHeader=False, version=3.2):
    """
    Writes all factor returns for the specified model between the two dates passed to factorFile.
    Writes to volatilityFile (if any) the square root (stddev) of the factor covariances for the same
    factors and dates.
    Dates are inclusive.
    """
    rms = riskModel.rms_id

    # Get start, end date and full return history
    modelDB.dbCursor.execute("""select distinct dt from rms_factor_return
                                where RMS_ID=:rms_id order by dt""", rms_id=rms)
    r = modelDB.dbCursor.fetchall()
    assert len(r) != 0, "No factor returns for model %s found!" % riskModel.name
    returnFullDates = [modelDB.oracleToDate(i[0]) for i in r]
    returnStartDate = modelDB.oracleToDate(r[0][0])
    returnEnddate = modelDB.oracleToDate(r[-1][0])

    # Get a date list for factor returns
    modelDB.dbCursor.execute("""select from_dt, thru_dt from risk_model_serie rms
                                where serial_id=:rms_id""", rms_id=rms)
    r = modelDB.dbCursor.fetchall()
    assert len(r) == 1, "No factors for model %s found!" % riskModel.name
    revision = r[0]
    rmsFromDt = revision[0].date()
    rmsThruDt = revision[1].date()
    seriesStartDate = max(startDate, rmsFromDt)
    seriesEndDate = min(endDate, rmsThruDt)
    logging.info('Processing %s from %s through %s', riskModel.name, seriesStartDate, seriesEndDate)
    modelDB.dbCursor.execute("""select distinct dt from rms_factor_return
        where dt >= :startdate and dt <= :enddate and rms_id = :rms_id
        order by dt""", startdate=seriesStartDate, enddate=seriesEndDate, rms_id=rms)
    r = modelDB.dbCursor.fetchall()
    dates = [modelDB.oracleToDate(a[0]) for a in r]
    if not dates:
        logging.warning("No dates for %s between %s and %s", riskModel.name, startDate, endDate)
        return

    # Get a list of factors that are active within the risk model serie's lifespan
    factors = sorted(modelDB.getRiskModelSerieFactors(rms), key=lambda x: x.name)
    factors = [f for f in factors
               if f.from_dt<=seriesEndDate and seriesStartDate<=f.thru_dt]
    if not factors:
        logging.warning("No factors for %s between %s and %s", riskModel.name, startDate, endDate)
        return
    logging.debug('getting subfactors for %d dates', len(dates))
    subfactors = set()
    sfIDMap = dict()
    factorcopy = copy.copy(factors)
    datefactors = []
    for f in factorcopy:
        subfactor, from_dt, thru_dt = getSubFactorForDateRange(dates, f, modelDB)
        if subfactor:
            subfactors.add(subfactor)
            sfIDMap[f] = subfactor.subFactorID
            datefactors.append([f, from_dt, thru_dt])
        else:
            factors.remove(f)
    logging.debug('got %s subfactors', len(subfactors))

    # Insert more dates that is prior to seriesStartDate to datelist
    # when frequency != daily and return is available for the date, so that the first period return
    # can be generated. eom-to-eom/eow-to-eow returns are needed to compute period return
    returnDatesIdxMap = dict([(d,i) for (i,d) in enumerate(returnFullDates)])
    FullRMCalendar = getRMCalendar(riskModel, modelDB, seriesStartDate,
                                       datetime.date(2999, 12, 31), returnfreq)

    if (returnfreq != 'daily') and (returnStartDate < seriesStartDate) and \
                    seriesStartDate not in FullRMCalendar:
        if seriesStartDate in returnFullDates:
            curIdx = returnDatesIdxMap[seriesStartDate]
        else:
            closestDt = min(returnFullDates, key=lambda dt: abs(dt - seriesStartDate))
            if closestDt < seriesStartDate:
                curIdx = returnDatesIdxMap[closestDt] + 1
            else:
                curIdx = returnDatesIdxMap[closestDt]
        if curIdx != 0 and returnfreq == 'weekly':
            while curIdx >= 0:
                dates.insert(0, returnFullDates[curIdx - 1])
                curIdx = curIdx - 1
                if dates[0] < seriesStartDate - datetime.timedelta(7):
                    break
        elif curIdx != 0 and returnfreq == 'monthly':
            while curIdx >= 0:
                dates.insert(0, returnFullDates[curIdx - 1])
                curIdx = curIdx - 1
                if dates[0] < seriesStartDate - datetime.timedelta(29):
                    break

    # Load up factor returns and prepare queries for extracting volatilities
    modelDB.setFactorReturnCache(len(dates))
    allReturns = modelDB.loadFactorReturnsHistory(rms, subfactors, dates)

    query = """SELECT sub_factor1_id, value FROM rmi_covariance WHERE
        rms_id = :rms_arg AND dt = :date_arg AND sub_factor1_id in (%s)
        AND sub_factor1_id = sub_factor2_id"""
    idList = ','.join([':sf%d' % n for n in range(len(subfactors))])
    idDict = dict(zip(['sf%d' % n for n in range(len(subfactors))],
                      [s.subFactorID for s in subfactors]))
    idDict['rms_arg'] = rms
    subIdxMap = dict(zip([s.subFactorID for s in subfactors],
                         list(range(len(subfactors)))))
    if standardHeader:
        factorFile.write('#Columns: Date|%s\n' % (
            "|".join([f.name for f in factors])))
    else:
        factorFile.write(delimiter.join(['Date'] + ['"%s"' % f.name for f in factors]))
        factorFile.write('\n')

    # Change frequency of date list if necessary
    if returnfreq == 'daily':
        perioddates = dates
        periodReturns = allReturns.data
    elif returnfreq == 'weekly':
        perioddates = Utilities.change_date_frequency(dates, frequency='weekly')
    else:
        perioddates = Utilities.change_date_frequency(dates, frequency='monthly')

    # Cross-check perioddates with RMCalendar and append the final period-end dates that is missing
    # in perioddates but exist in daily return list
    if returnfreq != 'daily':
        today = datetime.datetime.today().date()
        if seriesEndDate > today:
            RMCalendar = [i for i in FullRMCalendar if i >= seriesStartDate and i <= today]
        else:
            RMCalendar = [i for i in FullRMCalendar if i >= seriesStartDate and i <= seriesEndDate]
        if len(RMCalendar) > 0:
            finalDt = RMCalendar[-1]
            if finalDt in dates and returnfreq == 'weekly':
                if finalDt.isocalendar()[1] > perioddates[-1].isocalendar()[1] or \
                    finalDt.year > perioddates[-1].year:
                    perioddates.append(finalDt)
            elif finalDt in dates and returnfreq == 'monthly':
                if finalDt.month > perioddates[-1].month or finalDt.year > perioddates[-1].year:
                    perioddates.append(finalDt)

    # Change factor return periodicity if necessary
    if returnfreq != 'daily':
        datesIdxMap = dict([(d,i) for (i,d) in enumerate(dates)])
        cumReturns = numpy.cumprod(allReturns.data + 1.0, axis=1)
        cumReturns = numpy.take(cumReturns, [datesIdxMap[i] for i in perioddates], axis=1)
        periodReturns = cumReturns[:, 1:] / cumReturns[:, :-1] - 1.0
        perioddates = perioddates[1:]

    # Write factor return into file
    for (i, d) in enumerate(perioddates):
        data=[d.strftime(dateformat)]
        for factor in datefactors:
            # blank if subfactor->factor association not active on this day
            # from_dt > date or thru_dt <= date
            if factor[1] > d or factor[2] <= d:
                data.append('')
                continue
            # look up subfactor index by subfactor ID by factor
            subIdx = subIdxMap[sfIDMap[factor[0]]]
            rtn = periodReturns[subIdx][i]
            if rtn is not ma.masked:
                if version >= 4.0:
                    data.append('%.12f' % (100*rtn))
                else:
                    data.append(str(100*rtn))
            else:
                data.append('')
        factorFile.write(delimiter.join(data))
        factorFile.write('\n')

    # Write factor volatility into file if requested
    if volatilityFile:
        volatilityFile.write(delimiter.join(['Date'] + ['"%s"' % f.name for f in factors]))
        volatilityFile.write('\n')
        for (i, d) in enumerate(dates):
            coVars = ma.array(numpy.zeros(len(subfactors), float),
                                          mask=numpy.array([True]))
            idDict['date_arg'] = d
            modelDB.dbCursor.execute(query % idList, idDict)
            r =  modelDB.dbCursor.fetchall()
            for (sf, v) in r:
                i1 = subIdxMap[sf]
                coVars[i1] = v

            data1=[d.strftime(dateformat)]
            for factor in datefactors:
                if factor[1] > d or factor[2] <= d:
                    data1.append('')
                    continue

                subIdx = subIdxMap[sfIDMap[factor[0]]]
                if coVars[subIdx] is not ma.masked:
                    data1.append(str(sqrt(coVars[subIdx])))
                else:
                    data1.append('')
            volatilityFile.write(delimiter.join(data1))
            volatilityFile.write('\n')

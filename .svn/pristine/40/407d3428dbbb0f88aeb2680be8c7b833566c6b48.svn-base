

import datetime
import logging


def syncCurrencyReturns(sourceDB, targetDB, date, config):
    """Make the cumulative values in currency_risk_free_rate in targetDB
    the same as in sourceDB on the given date."""
    MY_SECTION = 'cumulative-risk-free-rate'
    TOLERANCE = 1e-12
    currencies = config.get(MY_SECTION, 'currencies').split(',')
    if  config.has_option(MY_SECTION, 'src-table'):
        srcTable = config.get(MY_SECTION, 'src-table').strip()
    else:
        srcTable = 'currency_risk_free_rate'
    for currency in currencies:
        logging.info('Updating cumulative risk-free rate for %s'
                     ' to %s', currency, srcTable)
        sourceDB.dbCursor.execute("""SELECT cumulative
        FROM %(srcTable)s WHERE dt=:dt_arg AND currency_code=:ccy""" % {
                'srcTable': srcTable}, dt_arg=date, ccy=currency)
        sourceValue = float(sourceDB.dbCursor.fetchone()[0])
        targetDB.dbCursor.execute("""SELECT cumulative
        FROM currency_risk_free_rate 
        WHERE dt=:dt_arg AND currency_code=:ccy""", dt_arg=date, ccy=currency)
        targetValue = float(targetDB.dbCursor.fetchone()[0])
        if abs(sourceValue - targetValue) < TOLERANCE:
            logging.debug('No update for %s, cumulative values'
                          ' already match', currency)
        else:
            scale = sourceValue / targetValue
            logging.info('Scaling cumulative returns for %s on target'
                         ' by %.8f up to %s', currency, scale, date)
            query = """UPDATE currency_risk_free_rate
            SET cumulative = cumulative * :scale_arg
            WHERE currency_code = :currency_arg and dt <= :dt_arg"""
            targetDB.dbCursor.execute(query, scale_arg=scale,
                                      currency_arg=currency, dt_arg=date)


def syncAssetReturns(sourceDB, targetDB, date, config):
    """Make the cumulative values in sub_issue_cumulative_return in targetDB
    the same as in sourceDB on the given date. The values in sourceDB
    are taken as active on the specified revision date.
    The 'sub-issues' option contains a list of sub-issue IDs and/or RMG IDs
    which defines the sub-issues that will be updated.
    """
    MY_SECTION = 'sub-issue-cumulative-return'
    TOLERANCE = 1e-9
    srcRev = config.get(MY_SECTION, 'sync-to-rev')
    srcRev = datetime.datetime.strptime(srcRev, '%Y-%m-%d %H:%M:%S')
    logging.info('Using source revision date %s', srcRev)
    sidList = config.get(MY_SECTION, 'sub-issues').split(',')
    sidSet = set()
    for sidSpec in sidList:
        sidSpec = sidSpec.strip()
        if len(sidSpec) == 12:
            sidSet.add(sidSpec)
        elif len(sidSpec) == 2:
            logging.info('Load all %s sub-issue IDs' % sidSpec)
            targetDB.dbCursor.execute("""SELECT distinct sub_id FROM sub_issue si JOIN
            risk_model_group rmg ON si.rmg_id=rmg.rmg_id
            WHERE rmg.mnemonic=:rmg_mnemonic""", rmg_mnemonic=sidSpec)
            sidSet.update([i[0] for i in targetDB.dbCursor.fetchall()])

        else:
            rmgID = int(sidSpec)
            targetDB.dbCursor.execute("""SELECT sub_id FROM sub_issue
              WHERE rmg_id=:rmg AND from_dt <= :dt AND thru_dt > :dt""",
                                      rmg=rmgID, dt=date)
            sidSet.update([i[0] for i in targetDB.dbCursor.fetchall()])
    logging.info('Updating %d sub-issues', len(sidSet))
    sidSet = sorted(sidSet)

    # build dictionary with source values
    srcDict = dict()
    for sid in sidSet:
        sourceDB.dbCursor.execute("""SELECT sub_issue_id, value
        FROM sub_issue_cumulative_return t1
        WHERE sub_issue_id=:sid AND dt=:dt_arg AND rev_dt = (
            SELECT MAX(rev_dt) FROM sub_issue_cumulative_return t2
            WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.dt=t2.dt
              AND rev_dt <= :rev_arg)
          AND rev_del_flag='N'""",
                                  dt_arg=date, rev_arg=srcRev,
                                  sid=sid)
        r = sourceDB.dbCursor.fetchone()
        if r is not None:
            srcDict[sid] = float(r[1])
    # Check current value and update if necessary
    tgtRev = datetime.datetime.now()
    for sid in sidSet:
        if sid not in srcDict:
            logging.debug('Skipping %s which has no source cumulative return',
                         sid)
            continue
        targetDB.dbCursor.execute("""SELECT sub_issue_id, value
        FROM sub_issue_cumulative_return t1
        WHERE sub_issue_id=:sid AND dt=:dt_arg AND rev_dt = (
            SELECT MAX(rev_dt) FROM sub_issue_cumulative_return t2
            WHERE t1.sub_issue_id=t2.sub_issue_id AND t1.dt=t2.dt
              AND rev_dt <= :rev_arg)
          AND rev_del_flag='N'""",
                                  dt_arg=date, rev_arg=tgtRev,
                                  sid=sid)
        r = targetDB.dbCursor.fetchone()
        if r is not None:
            tgtVal = float(r[1])
            srcVal = srcDict[sid]
            if abs(srcVal - tgtVal) / abs(srcVal) < TOLERANCE:
                logging.debug('No update for %s, cumulative values'
                              ' already match', sid)
            else:
                scale = srcVal / tgtVal
                logging.info('Scaling cumulative returns for %s'
                             ' on target by %.8f up to %s',
                             sid, scale, date)
                query = """UPDATE sub_issue_cumulative_return t1
                SET value = value * :scale_arg
                WHERE sub_issue_id = :subid_arg AND dt <= :maxdt
                AND rev_dt = (SELECT MAX(rev_dt) 
                   FROM sub_issue_cumulative_return t2
                   WHERE t2.sub_issue_id=t1.sub_issue_id
                   AND t2.dt=t1.dt)"""
                targetDB.dbCursor.execute(query, scale_arg=scale,
                                          subid_arg=sid, maxdt=date)
        else:
            logging.debug('Skipping %s which has no target cumulative return',
                          sid)


def syncFactorReturns(sourceDB, targetDB, date, config):
    """Make the cumulative values in rms_factor_return for the given
    risk model series in targetDB the same as in sourceDB on the given date."""
    MY_SECTION = 'factor-cumulative-return'
    TOLERANCE = 1e-11
    rmsPairs = config.get(MY_SECTION, 'rms-id-pairs').split(',')
    if  config.has_option(MY_SECTION, 'src-table'):
        srcTable = config.get(MY_SECTION, 'src-table').strip()
    else:
        srcTable = 'rms_factor_return'
    for rmsPair in rmsPairs:
        rmsPair = rmsPair.split(':')
        srcRms = int(rmsPair[0])
        tgtRms = int(rmsPair[1])
        logging.info('Updating cumulative factor returns for rms_id %d'
                     ' to match %d in %s', tgtRms, srcRms, srcTable)
        sourceDB.dbCursor.execute("""SELECT sub_factor_id, cumulative
        FROM %s WHERE dt=:dt_arg AND rms_id=:rms_arg""" % (srcTable),
                                  dt_arg=date, rms_arg=srcRms)
        sourceValues = dict(sourceDB.dbCursor.fetchall())
        targetDB.dbCursor.execute("""SELECT sub_factor_id, cumulative
        FROM rms_factor_return WHERE dt=:dt_arg AND rms_id=:rms_arg""",
                                  dt_arg=date, rms_arg=tgtRms)
        targetValues = dict(targetDB.dbCursor.fetchall())
        sourceFactors = set(sourceValues.keys())
        targetFactors = set(targetValues.keys())
        if len(targetFactors - sourceFactors) > 0:
            logging.warning('The following sub-factors will not be updated: '
                         '%s', ','.join([str(i) for i
                                         in targetFactors - sourceFactors]))
        for subid in (sourceFactors & targetFactors):
            sourceValue = sourceValues[subid]
            targetValue = targetValues[subid]
            if abs(sourceValue - targetValue) < TOLERANCE:
                logging.debug('No update for rms %d, factor %d'
                              ', cumulative values already match',
                              tgtRms, subid)
            else:
                scale = sourceValue / targetValue
                logging.info('Scaling cumulative returns for rms %d, factor %d'
                             ' on target by %.8f up to %s', tgtRms, subid,
                             scale, date)
                query = """UPDATE rms_factor_return
                SET cumulative = cumulative * :scale_arg
                WHERE sub_factor_id = :subid_arg AND rms_id=:rms_arg
                   AND dt <= :dt_arg"""
                targetDB.dbCursor.execute(query, scale_arg=scale,
                                          subid_arg=subid, rms_arg=tgtRms,
                                          dt_arg=date)
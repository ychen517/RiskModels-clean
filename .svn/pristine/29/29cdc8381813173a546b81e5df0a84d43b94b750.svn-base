import configparser
import datetime
import itertools
import logging
import optparse
from marketdb import MarketID
from riskmodels import Utilities
from riskmodels import Connections

def toDate(dt):
    if dt is not None:
        return dt.date()
    return dt

def buildAxiomaIDList(specList, mktDB, mdlDB):
    axiomaIDs = set()
    for spec in specList:
        spec = spec.strip()
        if len(spec) == 10:
            # assume Axioma ID
            axiomaIDs.add(spec)
        elif len(spec) == 2:
            # risk model group mnemonic
            assert(not 'Not implemented yet')
        elif len(spec) == len('add_dt-2009-01-01') \
                and spec[:len('add_dt')] == 'add_dt':
            # all Axioma IDs with the specified add_dt
            addDt = Utilities.parseISODate(spec[len('add_dt-'):])
            mktDB.dbCursor.execute("""SELECT axioma_id FROM asset_ref
               WHERE add_dt=:dt""", dt=addDt)
            axids = [i[0] for i in mktDB.dbCursor.fetchall()]
            logging.info('Adding %d Axioma IDs with add_dt=%s',
                         len(axids), addDt)
            axiomaIDs.update(axids)
        else:
            raise ValueError('Unsupport specifier %s' % spec)
    return axiomaIDs

def getHistory(mkt, tableName, axid, minMdlDt, maxMdlDt):
    mkt.dbCursor.execute("""SELECT id, change_dt,
          change_del_flag FROM %(table)s
          WHERE axioma_id=:axid
          ORDER BY change_dt ASC""" % {'table': tableName}, axid=axid)
    history = mkt.dbCursor.fetchall()
    history.append((None, datetime.datetime(2999,12,31), 'Y'))
    history.insert(0, (None, datetime.datetime(1950,1,1), 'Y'))
    idIntervals = [(id1, flag1, dt1.date(), dt2.date())
                   for ((id1,dt1,flag1), (id2,dt2,flag2))
                   in zip(history[:-1], history[1:])]
    idIntervals = [(id, flag, max(fromDt, minMdlDt), min(thruDt, maxMdlDt))
                   for (id, flag, fromDt, thruDt) in idIntervals
                   if fromDt < maxMdlDt and thruDt > minMdlDt]
    return idIntervals
    
def getClsHistory(mkt, tableName, revision, axid, minMdlDt, maxMdlDt):
    mkt.dbCursor.execute("""SELECT code, change_dt,
          change_del_flag FROM %(table)s it JOIN classification_ref cr
          ON cr.id=it.classification_id
          WHERE axioma_id=:axid AND it.revision_id=:rev
          ORDER BY change_dt ASC""" % {'table': tableName},
                         axid=axid, rev=revision.id)
    history = mkt.dbCursor.fetchall()
    history.append((None, datetime.datetime(2999,12,31), 'Y'))
    history.insert(0, (None, datetime.datetime(1950,1,1), 'Y'))
    idIntervals = [(id1, flag1, dt1.date(), dt2.date())
                   for ((id1,dt1,flag1), (id2,dt2,flag2))
                   in zip(history[:-1], history[1:])]
    idIntervals = [(id, flag, max(fromDt, minMdlDt), min(thruDt, maxMdlDt))
                   for (id, flag, fromDt, thruDt) in idIntervals
                   if fromDt < maxMdlDt and thruDt > minMdlDt]
    return idIntervals
    
def findMissingIDs(mkt, mdl, idName, ucpMinMaxTable, options):
    """Report axioma IDs that are missing identifiers in tableName during
    their life-time as given by the min/max date in ucpMinMaxTable.
    """
    (tableName, otherName, otherTableName) = {
        'SEDOL': ('asset_dim_sedol_active', 'ISIN', 'asset_dim_isin_active'),
        'ISIN': ('asset_dim_isin_active', 'SEDOL', 'asset_dim_sedol_active'),
        'Name': ('asset_dim_name_active', 'SEDOL', 'asset_dim_sedol_active'),
        'Currency': ('asset_dim_trading_curr_active', 'SEDOL',
                     'asset_dim_sedol_active'),
        'Country': ('asset_dim_trading_curr_active', 'SEDOL',
                     'asset_dim_sedol_active')
        }[idName]
    marketFamily = mkt.getClassificationFamily('REGIONS')
    marketMember = [i for i in mkt.getClassificationFamilyMembers(marketFamily)
                    if i.name == 'Market'][0]
    marketRev = mkt.getClassificationMemberRevision(
        marketMember, datetime.date.today())
    mkt.dbCursor.execute("""SELECT uc.axioma_id FROM %(ucpDates)s uc
       WHERE EXISTS(SELECT * FROM %(table)s it
                    WHERE it.axioma_id=uc.axioma_id and it.change_del_flag='Y'
                    AND change_dt BETWEEN uc.min_dt AND uc.max_dt)
       OR NVL((SELECT change_del_flag  FROM %(table)s it2
           WHERE it2.axioma_id=uc.axioma_id
             AND change_dt=(SELECT MAX(change_dt) FROM %(table)s it3
                            WHERE it2.axioma_id=it3.axioma_id
                            AND it3.change_dt <= uc.min_dt)), 'Y') = 'Y'
       """ % { 'table': tableName, 'ucpDates': ucpMinMaxTable })
    candidates = [i[0] for i in mkt.dbCursor.fetchall()]
    logging.debug('%d potential %s duplicates', len(candidates), idName)
    if options.restrictList:
        candidates = [i for i in candidates if i in options.restrictList]
        logging.debug('%d potential %s duplicates after applying restrictions',
                      len(candidates), idName)
    if options.excludeList:
        candidates = [i for i in candidates if i not in options.excludeList]
        logging.debug('%d potential %s duplicates after applying exclusions',
                      len(candidates), idName)
    print('Axioma ID|ID Type|From Dt|Thru Dt|Recent Name' \
        '|Pre Value|Pre From Dt|Post Value|Post Thru Dt'\
        '|Last Value|%(other)s while missing|Last %(other)s'\
        '|Country|Exchange|Min UCP Dt|Max UCP Dt|'\
        % {'other': otherName})
    for axid in sorted(candidates):
        (minMdlDt, maxMdlDt) = getModelDate(axid, mdl, mkt)
        if minMdlDt is None or minMdlDt > maxMdlDt:
            continue
        idIntervals = getHistory(mkt, tableName, axid, minMdlDt, maxMdlDt)
        hasMissing = len([i for i in idIntervals if i[1] == 'Y']) > 0
        if hasMissing:
            otherHistory = getHistory(mkt, otherTableName, axid,
                                      minMdlDt, maxMdlDt)
            nameHistory = getHistory(mkt, 'asset_dim_name_active', axid,
                                     minMdlDt, maxMdlDt)
            mkt.dbCursor.execute("""SELECT min(dt), max(dt)
               FROM asset_dim_ucp_active WHERE axioma_id=:axid
               AND price_marker <> 3""", axid=axid)
            (minTraded, maxTraded) = mkt.dbCursor.fetchone()
            minTraded = toDate(minTraded)
            maxTraded = toDate(maxTraded)
            axidObj = MarketID.MarketID(string=axid)
            marketCls = mkt.getAssetClassifications(
                marketRev, [axidObj], maxMdlDt)
            if axidObj not in marketCls:
                marketCls = mkt.getAssetClassifications(
                    marketRev, [axidObj], minMdlDt)
            if axidObj not in marketCls:
                country = ''
                exchange = ''
            else:
                marketCls = marketCls[axidObj].classification
                country = marketCls.levelParent[1].name
                if marketCls.level == 2:
                    exchange = marketCls.name
                else:
                    exchange = marketCls.levelParent[2].name
            values = [i for i in nameHistory if i[1] == 'N']
            if len(values) > 0:
                recentName = values[-1][0]
            else:
                recentName = ''
            values = [i for i in otherHistory if i[1] == 'N']
            if len(values) > 0:
                lastOther = values[-1][0]
            else:
                lastOther = ''
            values = [i for i in idIntervals if i[1] == 'N']
            if len(values) > 0:
                lastVal = values[-1][0]
            else:
                lastVal = ''
            for (idx, (id, flag, fromDt, thruDt)) in enumerate(idIntervals):
                if flag == 'N':
                    continue
                if idx == 0:
                    prevVal = ''
                    prevDt = ''
                else:
                    prevVal = idIntervals[idx-1][0]
                    prevDt = idIntervals[idx-1][2]
                if idx < len(idIntervals) - 1:
                    postVal = idIntervals[idx+1][0]
                    postDt = idIntervals[idx+1][3]
                else:
                    postVal = ''
                    postDt = ''
                values = [i for i in otherHistory
                         if i[2] < thruDt and i[3] > fromDt and i[1] == 'N']
                if len(values) > 0:
                    otherDuring = values[-1][0]
                else:
                    otherDuring = ''
                print('%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' % (
                    axid, idName, fromDt, thruDt, recentName, prevVal, prevDt,
                    postVal, postDt, lastVal, otherDuring, lastOther,
                    country, exchange, minTraded, maxTraded))

def findMissingCurrency(mkt, mdl, ucpMinMaxTable, options):
    """Report axioma IDs that are missing identifiers in tableName during
    their life-time as given by the min/max date in ucpMinMaxTable.
    """
    idName = 'Trading Currency'
    (tableName, otherName, otherTableName) = (
        'asset_dim_trading_curr_active', 'SEDOL', 'asset_dim_sedol_active')
    marketFamily = mkt.getClassificationFamily('REGIONS')
    marketMember = [i for i in mkt.getClassificationFamilyMembers(marketFamily)
                    if i.name == 'Market'][0]
    marketRev = mkt.getClassificationMemberRevision(
        marketMember, datetime.date.today())
    mkt.dbCursor.execute("""SELECT uc.axioma_id FROM %(ucpDates)s uc
       WHERE EXISTS(SELECT * FROM %(table)s it
                    WHERE it.axioma_id=uc.axioma_id and it.change_del_flag='Y'
                    AND change_dt BETWEEN uc.min_dt AND uc.max_dt)
       OR NVL((SELECT change_del_flag  FROM %(table)s it2
           WHERE it2.axioma_id=uc.axioma_id
             AND change_dt=(SELECT MAX(change_dt) FROM %(table)s it3
                            WHERE it2.axioma_id=it3.axioma_id
                            AND it3.change_dt <= uc.min_dt)), 'Y') = 'Y'
       """ % { 'table': tableName, 'ucpDates': ucpMinMaxTable })
    candidates = [i[0] for i in mkt.dbCursor.fetchall()]
    logging.debug('%d potentially missing %s', len(candidates), idName)
    if options.restrictList:
        candidates = [i for i in candidates if i in options.restrictList]
        logging.debug('%d potentially missing %s after applying restrictions',
                      len(candidates), idName)
    if options.excludeList:
        candidates = [i for i in candidates if i not in options.excludeList]
        logging.debug('%d potentially missing %s after applying exclusions',
                      len(candidates), idName)
    MODEL_START_DATE = datetime.date(1999, 1, 1)
    mkt.dbCursor.execute("""SELECT id, code FROM currency_ref""")
    currencyCodeDict = dict(mkt.dbCursor.fetchall())
    print('Axioma ID|ID Type|From Dt|Thru Dt|Recent Name' \
        '|Pre Value|Pre From Dt|Post Value|Post Thru Dt'\
        '|Last Value|%(other)s while missing|Last %(other)s'\
        '|Country|Exchange|UCP Currencies|Min UCP Dt|Max UCP Dt|'\
        % {'other': otherName})
    for axid in sorted(candidates):
        (minMdlDt, maxMdlDt) = getModelDate(axid, mdl, mkt)
        if minMdlDt is None or minMdlDt > maxMdlDt:
            continue
        idIntervals = getHistory(mkt, tableName, axid, minMdlDt, maxMdlDt)
        hasMissing = len([i for i in idIntervals if i[1] == 'Y']) > 0
        if hasMissing:
            otherHistory = getHistory(mkt, otherTableName, axid,
                                      minMdlDt, maxMdlDt)
            nameHistory = getHistory(mkt, 'asset_dim_name_active', axid,
                                     minMdlDt, maxMdlDt)
            mkt.dbCursor.execute("""SELECT min(dt), max(dt)
               FROM asset_dim_ucp_active WHERE axioma_id=:axid
               AND price_marker <> 3""", axid=axid)
            (minTraded, maxTraded) = mkt.dbCursor.fetchone()
            minTraded = toDate(minTraded)
            maxTraded = toDate(maxTraded)
            mkt.dbCursor.execute("""SELECT distinct currency_id
               FROM asset_dim_ucp_active WHERE axioma_id=:axid""", axid=axid)
            ucpCurrencies = ','.join([currencyCodeDict[i[0]] for i
                                      in mkt.dbCursor.fetchall()])
            axidObj = MarketID.MarketID(string=axid)
            marketCls = mkt.getAssetClassifications(
                marketRev, [axidObj], maxMdlDt)
            if axidObj not in marketCls:
                marketCls = mkt.getAssetClassifications(
                    marketRev, [axidObj], minMdlDt)
            if axidObj not in marketCls:
                country = ''
                exchange = ''
            else:
                marketCls = marketCls[axidObj].classification
                country = marketCls.levelParent[1].name
                if marketCls.level == 2:
                    exchange = marketCls.name
                else:
                    exchange = marketCls.levelParent[2].name
            values = [i for i in nameHistory if i[1] == 'N']
            if len(values) > 0:
                recentName = values[-1][0]
            else:
                recentName = ''
            values = [i for i in otherHistory if i[1] == 'N']
            if len(values) > 0:
                lastOther = values[-1][0]
            else:
                lastOther = ''
            values = [i for i in idIntervals if i[1] == 'N']
            if len(values) > 0:
                lastVal = currencyCodeDict[values[-1][0]]
            else:
                lastVal = ''
            for (idx, (id, flag, fromDt, thruDt)) in enumerate(idIntervals):
                if flag == 'N':
                    continue
                if idx == 0:
                    prevVal = ''
                    prevDt = ''
                else:
                    prevVal = currencyCodeDict[idIntervals[idx-1][0]]
                    prevDt = idIntervals[idx-1][2]
                if idx < len(idIntervals) - 1:
                    postVal = currencyCodeDict[idIntervals[idx+1][0]]
                    postDt = idIntervals[idx+1][3]
                else:
                    postVal = ''
                    postDt = ''
                values = [i for i in otherHistory
                         if i[2] < thruDt and i[3] > fromDt and i[1] == 'N']
                if len(values) > 0:
                    otherDuring = values[-1][0]
                else:
                    otherDuring = ''
                print('%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' % (
                    axid, idName, fromDt, thruDt, recentName, prevVal, prevDt,
                    postVal, postDt, lastVal, otherDuring, lastOther,
                    country, exchange, ucpCurrencies, minTraded, maxTraded))

def findMissingCountry(mkt, mdl, ucpMinMaxTable, options):
    """Report axioma IDs that are missing country classification during
    their life-time as given by the min/max date in ucpMinMaxTable.
    """
    idName = 'Country'
    (tableName, otherName, otherTableName) = (
        'classification_const_active', 'SEDOL', 'asset_dim_sedol_active')
    marketFamily = mkt.getClassificationFamily('REGIONS')
    marketMember = [i for i in mkt.getClassificationFamilyMembers(marketFamily)
                    if i.name == 'Market'][0]
    marketRev = mkt.getClassificationMemberRevision(
        marketMember, datetime.date.today())
    mkt.dbCursor.execute("""SELECT uc.axioma_id FROM %(ucpDates)s uc
       WHERE uc.axioma_id not like 'CSH_%%' AND (
       EXISTS(SELECT * FROM %(table)s it
                    WHERE it.axioma_id=uc.axioma_id AND it.change_del_flag='Y'
                    AND it.revision_id=:rev
                    AND change_dt BETWEEN uc.min_dt AND uc.max_dt)
       OR NVL((SELECT change_del_flag  FROM %(table)s it2
           WHERE it2.axioma_id=uc.axioma_id AND revision_id=:rev
             AND change_dt=(SELECT MAX(change_dt) FROM %(table)s it3
                            WHERE it2.axioma_id=it3.axioma_id
                            AND revision_id=:rev
                            AND it3.change_dt <= uc.min_dt)), 'Y') = 'Y')
       """ % { 'table': tableName, 'ucpDates': ucpMinMaxTable },
                         rev=marketRev.id)
    candidates = [i[0] for i in mkt.dbCursor.fetchall()]
    logging.debug('%d potentially missing %s', len(candidates), idName)
    if options.restrictList:
        candidates = [i for i in candidates if i in options.restrictList]
        logging.debug('%d potentially missing  %s after applying restrictions',
                      len(candidates), idName)
    if options.excludeList:
        candidates = [i for i in candidates if i not in options.excludeList]
        logging.debug('%d potentially missing %s after applying exclusions',
                      len(candidates), idName)
    print('Axioma ID|ID Type|From Dt|Thru Dt|Recent Name' \
        '|Pre Value|Pre From Dt|Post Value|Post Thru Dt'\
        '|Last Value|%(other)s while missing|Last %(other)s'\
        '|Min UCP Dt|Max UCP Dt|'\
        % {'other': otherName})
    for axid in sorted(candidates):
        (minMdlDt, maxMdlDt) = getModelDate(axid, mdl, mkt)
        if minMdlDt is None or minMdlDt > maxMdlDt:
            continue
        idIntervals = getClsHistory(mkt, tableName, marketRev,
                                    axid, minMdlDt, maxMdlDt)
        hasMissing = len([i for i in idIntervals if i[1] == 'Y']) > 0
        if hasMissing:
            otherHistory = getHistory(mkt, otherTableName, axid,
                                      minMdlDt, maxMdlDt)
            nameHistory = getHistory(mkt, 'asset_dim_name_active', axid,
                                     minMdlDt, maxMdlDt)
            mkt.dbCursor.execute("""SELECT min(dt), max(dt)
               FROM asset_dim_ucp_active WHERE axioma_id=:axid
               AND price_marker <> 3""", axid=axid)
            (minTraded, maxTraded) = mkt.dbCursor.fetchone()
            minTraded = toDate(minTraded)
            maxTraded = toDate(maxTraded)
            axidObj = MarketID.MarketID(string=axid)
            values = [i for i in nameHistory if i[1] == 'N']
            if len(values) > 0:
                recentName = values[-1][0]
            else:
                recentName = ''
            values = [i for i in otherHistory if i[1] == 'N']
            if len(values) > 0:
                lastOther = values[-1][0]
            else:
                lastOther = ''
            values = [i for i in idIntervals if i[1] == 'N']
            if len(values) > 0:
                lastVal = values[-1][0]
            else:
                lastVal = ''
            for (idx, (id, flag, fromDt, thruDt)) in enumerate(idIntervals):
                if flag == 'N':
                    continue
                if idx == 0:
                    prevVal = ''
                    prevDt = ''
                else:
                    prevVal = idIntervals[idx-1][0]
                    prevDt = idIntervals[idx-1][2]
                if idx < len(idIntervals) - 1:
                    postVal = idIntervals[idx+1][0]
                    postDt = idIntervals[idx+1][3]
                else:
                    postVal = ''
                    postDt = ''
                values = [i for i in otherHistory
                         if i[2] < thruDt and i[3] > fromDt and i[1] == 'N']
                if len(values) > 0:
                    otherDuring = values[-1][0]
                else:
                    otherDuring = ''
                print('%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s' % (
                    axid, idName, fromDt, thruDt, recentName, prevVal, prevDt,
                    postVal, postDt, lastVal, otherDuring, lastOther,
                    minTraded, maxTraded))

def findDuplicateIDs(mkt, table, options):
    mkt.dbCursor.execute("""SELECT distinct t1.axioma_id, t2.axioma_id
      FROM %(table)s t1, %(table)s t2
      WHERE t1.id = t2.id and t1.axioma_id < t2.axioma_id
      AND t1.change_del_flag = 'N' and t2.change_del_flag='N'""" % {
            'table': table})
    candidates = mkt.dbCursor.fetchall()
    logging.debug('%d potential %s duplicates', len(candidates), table)
    if options.restrictList:
        candidates = [(i,j) for (i,j) in candidates
                      if i in options.restrictList
                      or j in options.restrictList]
        logging.debug('%d potential %s duplictes after applying restrictions',
                      len(candidates), table)
    if options.excludeList:
        candidates = [(i,j) for (i,j) in candidates
                      if i not in options.excludeList
                      and j not in options.excludeList]
        logging.debug('%d potential %s duplictes after applying exclusions',
                      len(candidates), table)
        
    # retrieve detailed history and check
    realConflicts = list()
    for (axid1, axid2) in candidates:
        mkt.dbCursor.execute("""SELECT axioma_id, id, change_dt,
          change_del_flag FROM %(table)s
          WHERE axioma_id IN (:axid1, :axid2)
          ORDER BY change_dt ASC, change_del_flag DESC""" % {'table': table},
                             axid1=axid1, axid2=axid2)
        histories = mkt.dbCursor.fetchall()
        activeIDAxid1 = None
        activeIDAxid2 = None
        conflictHistory = list()
        isConflict = False
        for (axid, idVal, changeDt, changeFlag) in histories:
            if axid == axid1:
                if changeFlag == 'Y':
                    activeIDAxid1 = None
                else:
                    activeIDAxid1 = idVal
            if axid == axid2:
                if changeFlag == 'Y':
                    activeIDAxid2 = None
                else:
                    activeIDAxid2 = idVal
            if isConflict:
                if activeIDAxid1 != activeIDAxid2:
                    isConflict = False
                    conflictHistory.append((conflictHistory[-1][0],
                                            changeDt.date(), 'Y'))
            elif activeIDAxid1 == activeIDAxid2 \
                    and activeIDAxid1 is not None:
                isConflict = True
                conflictHistory.append((activeIDAxid1, changeDt.date(), 'N'))
        if isConflict:
            conflictHistory.append((conflictHistory[-1][0],
                                    datetime.date(2999,12,31), 'Y'))
        if len(conflictHistory) > 0:
            realConflicts.append((axid1, axid2, conflictHistory))
    realConflicts.sort()
    logging.info('%d conflicting assets pairs for %s',
                 len(realConflicts), table)
    print('Axioma ID1|Axioma ID2|ID|From Date|Through Date|')
    for (axid1, axid2, conflictHistory) in realConflicts:
        for (fromVal, thruVal) in \
                zip(itertools.islice(conflictHistory, 0, None, 2),
                               itertools.islice(conflictHistory, 1, None, 2)):
            assert(fromVal[2] == 'N')
            assert(thruVal[2] == 'Y')
            assert(fromVal[0] == thruVal[0])
            print('%s|%s|%s|%s|%s|' % (axid1, axid2, fromVal[0], fromVal[1],
                                       thruVal[1]))
    

def getModelDate(axid, mdl, mkt):
    MODEL_START_DATE = datetime.date(1999, 1, 1)
    mdl.dbCursor.execute("""SELECT min(from_dt), max(thru_dt)
      FROM issue_map where marketdb_id=:axid""", axid=axid)
    (minMdlDt, maxMdlDt) = mdl.dbCursor.fetchone()
    if minMdlDt is None or maxMdlDt is None:
        logging.debug('Missing Model min/max date info for %s,'
                      ' using price info', axid)
        mkt.dbCursor.execute("""SELECT min(dt), max(dt)
           FROM asset_dim_ucp_active WHERE axioma_id=:axid""", axid=axid)
        (minMdlDt, maxMdlDt) = mkt.dbCursor.fetchone()
        minMdlDt = toDate(minMdlDt)
        maxMdlDt = toDate(maxMdlDt)
    else:
        minMdlDt = max(minMdlDt.date(), MODEL_START_DATE)
        maxMdlDt = maxMdlDt.date()
    return (minMdlDt, maxMdlDt)

def main():
    usage = "usage: %prog [options] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option(
        "--exclude", action="store", default='', dest="excludeList",
        help="comma-separated list of Axioma IDs to exclude")
    cmdlineParser.add_option(
        "--restrict", action="store", default='', dest="restrictList",
        help="comma-separated list of Axioma IDs to process")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args_) = cmdlineParser.parse_args()
    if len(args_) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    
    connections = Connections.createConnections(config_)
    mkt = connections.marketDB
    mdl = connections.modelDB
    if len(options.excludeList) > 0:
        options.excludeList = options.excludeList.split(',')
        options.excludeList = buildAxiomaIDList(options.excludeList,
                                                mkt, mdl)
        logging.info('Excluding %d Axioma IDs from consideration',
                     len(options.excludeList))
    else:
        options.excludeList = None
    
    if len(options.restrictList) > 0:
        options.restrictList = options.restrictList.split(',')
        options.restrictList = buildAxiomaIDList(options.restrictList,
                                                 mkt, mdl)
        logging.info('Restricting to %d Axioma IDs',
                     len(options.restrictList))
    else:
        options.restrictList = None
    
    ucpMinMaxTable = 'tmp_ucp_min_max_dt'
    findDuplicateIDs(mkt, 'asset_dim_sedol_active', options)
    findMissingIDs(mkt, mdl, 'Name', ucpMinMaxTable, options)
    findMissingIDs(mkt, mdl, 'ISIN', ucpMinMaxTable, options)
    findMissingIDs(mkt, mdl, 'SEDOL', ucpMinMaxTable, options)
    findMissingCountry(mkt, mdl, ucpMinMaxTable, options)
    findMissingCurrency(mkt, mdl, ucpMinMaxTable, options)
    Connections.finalizeConnections(connections)

if __name__ == '__main__':
    main()

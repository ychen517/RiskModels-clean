
import datetime
import logging
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities

# ALL_IDS contains the IDs for the Master file, along with the length of each.
ALL_IDS = dict([('CUSIP',(2,9, 'asset_dim_cusip', False)),
                ('TICKER',(3,0, 'asset_dim_ticker', False)),
                ('SEDOL',(4,7, 'asset_dim_sedol', True)),
                ('ISIN',(6,12, 'asset_dim_isin', False)),
                ('NAME',(7,0, 'asset_dim_name', False)),
                ('COUNTRY',(9,2,None, False))
                ])
NUM_IDS = len(ALL_IDS)
assert(len(set([idnum for (idnum, length, table, pad)
                in ALL_IDS.values()])) == NUM_IDS)

def encodeDate(date):
    return '%04d%02d%02d' % (date.year, date.month, date.day)

def writeIDHistory(modelDB, date, marketDB, allIssues, issueRMGHist, options):
    BOT = datetime.datetime(1950,1,1)
    EOT = datetime.datetime(9999,12,31)
    logging.info('%d assets in master' % len(list(allIssues.keys())))
    out = open('%s/AX%s.%04d%02d%02d.idh' % (options.targetDir, options.country or 'GL',
                                           date.year, date.month, date.day), 'w')
    regionFamily = marketDB.getClassificationFamily('REGIONS')
    if regionFamily:
        legacy = False
        regionMembers = marketDB.getClassificationFamilyMembers(regionFamily)
        marketMember = [i for i in regionMembers if i.name=='Market'][0]
        marketRev = marketDB.getClassificationMemberRevision(marketMember, date)
        countryMap = modelDB.getMktAssetClassifications(marketRev, allIssues, date, marketDB, level=1)
    else:
        # legacy database
        assert options.country, "Must supply country code for legacy database"
        legacy = True
        countryMap = dict()
        for mid in allIssues.keys():
            c = Utilities.Struct()
            c.classification = Utilities.Struct()
            c.classification.code = options.country
            countryMap[mid] = c
    modelMarketMap = dict([(i,modelDB.getIdentifierHistory(i.getIDString())) \
                               for i in allIssues.keys()])
    out.write(str(date)+'\n')
    out.write('AxiomaID|AssetIDType|AssetID|StartDate|EndDate\n')
    # write out all issues
    allMarketIDs = set()
    for marketIDs in modelMarketMap.values():
        allMarketIDs.update([i[0] for i in marketIDs])
    allIDHistories = dict()
    for (name, (idnum, length, table, pad)) in ALL_IDS.items():
        if table is None:
            continue
        if legacy:
            tbl = table.split("_")[-1]
            if tbl == 'name':
                tbl = 'issuer'
            idHistories = [marketDB.getIdentifierHistory(mid, tbl, BOT, EOT) for mid in allMarketIDs]
        else:
            idHistories = marketDB.getIdentifierHistory(allMarketIDs, table)
        allIDHistories[table] = dict(zip(allMarketIDs, idHistories))

    rmgIDDict = dict()
    modelDB.dbCursor.execute("SELECT rmg_id FROM risk_model_group")
    for r in modelDB.dbCursor.fetchall():
        rmgIDDict[r[0]] = modelDB.getRiskModelGroup(r[0])
    for i in allIssues.keys():
        n = 0
        assetFromDt = allIssues[i][0]
        assetThruDt = allIssues[i][1]
        if i.isCashAsset():
            idRMG = None
        elif i not in issueRMGHist:
            logging.error('Asset %s not in sub_issue table', i)
            continue
        else:
            issueRMGID = issueRMGHist[i]
            rmg_id = None
            for r in issueRMGID:
                if rmg_id and rmg_id != r[0]:
                    raise ValueError('Differing rmg IDs for model ID %s' % r)
                rmg_id = r[0]
            issueRMGID = rmg_id
            if issueRMGID not in rmgIDDict:
                logging.error("rmgID %d not in risk_model_group table", issueRMGID)
                rmg = modelDB.getRiskModelGroup(issueRMGID)
                rmg.setRMGInfoForDate(date)
                rmgIDDict[rmg.rmg_id] = rmg
            idRMG = rmgIDDict[issueRMGID]

        # Assume the asset has always been in this country
        if i.isCashAsset():
            countryCode = ''
        elif i in countryMap:
            countryCode = countryMap[i].classification.code.encode('ascii')
        else:
            logging.error('No country classification for %s,'
                          ' using rmg instead.', i)
            countryCode = idRMG.mnemonic
        if not i.isCashAsset():
            out.write(i.getPublicID())
            out.write('|COUNTRY|%s' % countryCode)
            out.write('|%s|%s\n' % (encodeDate(assetFromDt), encodeDate(assetThruDt)))
            n += 1
        # find and count all IDs for this issue's market IDs throughout time
        for (marketID, map_from_dt, map_thru_dt) in modelMarketMap[i]:
            for (name, (idnum, length, table, pad)) in ALL_IDS.items():
                if table is None:
                    continue
                idList = allIDHistories[table][marketID]
                idList = [(value, max(from_dt, map_from_dt),
                           min(thru_dt, map_thru_dt))
                          for (value, from_dt, thru_dt) in idList
                          if from_dt < map_thru_dt and thru_dt > map_from_dt]
                prevThru = None
                for (value, from_dt, thru_dt) in idList:
                    assert prevThru is None or from_dt >= prevThru, \
                        "Asset %s has overlapping %ss!" % (i.getPublicID(), name)
                    out.write(i.getPublicID())
                    out.write('|%s|%s' %(name, value))
                    out.write('|%s|%s\n' % (encodeDate(from_dt), encodeDate(thru_dt)))
                    prevThru = thru_dt
                    n += 1
        if n > 255:
            logging.fatal('More than 255 IDs for AXID %s' % marketID)
            raise ValueError('More than 255 IDs for AXID %s' % marketID)
    out.close()

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             dest="targetDir", default=".",
                             help="directory for output files")
    cmdlineParser.add_option("-c", "--country", action="store",
                             dest="country", default=None,
                             help="country code for legacy database extraction")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    date = Utilities.parseISODate(args[0]) 
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, 
                                 passwd=options.marketDBPasswd)
    allIssues = dict([(id, (from_dt, thru_dt)) for (id, from_dt, thru_dt)
                      in modelDB.getAllIssues()])
    # map issues to their RMG
    modelDB.dbCursor.execute("""SELECT issue_id, rmg_id, from_dt, thru_dt FROM sub_issue""")
    issueRMGHist = dict()
    for (mid, rmg_id, from_dt, thru_dt) in modelDB.dbCursor.fetchall():
        issueRMGHist.setdefault(ModelID.ModelID(string=mid), list()).append((rmg_id, from_dt, thru_dt))
    
    writeIDHistory(modelDB, date, marketDB, allIssues, issueRMGHist, options)

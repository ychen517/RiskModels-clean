

import datetime
import logging

from marketdb import constants
from riskmodels.ModelID import ModelID


def getCountryMinMaxDate(cur, assetid, assetsrc, ignorePrices=False):
    warnings=[]
    country = None
    pricesStr =  ' and (NAV >0 or [close]>0) '
    if ignorePrices:
        pricesStr = ''
    if assetsrc == 'NA':
        query = """select min([Pricing Date]), max([Pricing Date])
                    from IDC.dbo.NAPrices 
                    where cusip in (
                            select cusip 
                            from IDC.dbo.NAAssets 
                            where AssetId=%(assetid)d)  
                    and [Exchange Code] in (
                            select [Exchange Code] from IDC.dbo.NAAssets where AssetId=%(assetid)d)
                    %(pricesStr)s
                            """
        logging.debug(query)
        cur.execute(query % {'assetid': assetid,
                             'pricesStr':pricesStr})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            mindt, maxdt = r[0][0], r[0][1]
            mindt, maxdt = datetime.datetime.strptime(mindt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxdt, "%Y-%m-%d").date()
        else:
            if len(r) > 1:
                msg = "ERROR: Too many price records found for %d/%s" % (assetid, assetsrc)
                warnings.append(msg)
                logging.error(msg)
            elif len(r) == 0:
                msg = "ERROR: No price records found for %d/%s" % (assetid, assetsrc)
                warnings.append(msg)
                logging.error(msg)
            mindt = datetime.date(1950, 1, 1)
            maxdt = datetime.date(9999, 12, 31)

        query = """select country, min(FromDate), max(ToDate) 
                    from IDC.dbo.NAAssets 
                    where AssetId=%(assetid)d group by country"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            country, miniddt, maxiddt = r[0][0], r[0][1], r[0][2]
            miniddt, maxiddt = datetime.datetime.strptime(miniddt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxiddt, "%Y-%m-%d").date()
        else:
            if len(r) > 1:
                msg = "ERROR: Too many price records found for %d/%s" % (assetid, assetsrc)
                warnings.append(msg)
                logging.error(msg)
            elif len(r) == 0:
                msg = "ERROR: No price records found for %d/%s" % (assetid, assetsrc)
                warnings.append(msg)
                logging.error(msg)
            miniddt = datetime.date(1950, 1, 1)
            maxiddt = datetime.date(9999, 12, 31)
    else:
        query = """select c.[ISO_CountryCode], min([Pricing Date]), max([Pricing Date]) maxPriceDate
                   from IDC.dbo.NonNAPrices a, IDC.dbo.NonNAExchangeCodes b, IDC.dbo.Country c 
                   where sedol in (
                     select sedol from IDC.dbo.NonNAAssets where assetid=%(assetid)d) 
                   and substring(a.[Exchange Code], 1, 2)=b.[Country Code] 
                   and c.[Code]=b.[Country Code]
                   %(pricesStr)s
                   group by c.[ISO_CountryCode]
                   order by maxPriceDate desc"""
        logging.debug(query)
        cur.execute(query % {'assetid': assetid,
                             'pricesStr':pricesStr})
        r = cur.fetchall()
        if len(r) > 0 and r[0][0]: #use the latest record https://jira.axiomainc.com:8443/browse/MAC-17135
            country, mindt, maxdt = r[0][0], r[0][1], r[0][2]
            mindt, maxdt = datetime.datetime.strptime(mindt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxdt, "%Y-%m-%d").date()
        else:
#             if len(r) > 1:
#                 msg = "ERROR: %d countries found for %d/%s: %s" % (len(r), assetid, assetsrc,r)
#                 warnings.append(msg)
#                 logging.error(msg)
#             elif len(r) == 0:
            msg = "ERROR: No price/country/exchange records found for %d/%s" % (assetid, assetsrc)
            warnings.append(msg)
            logging.error(msg)
            mindt = datetime.date(1950, 1, 1)
            maxdt = datetime.date(9999, 12, 31)

            return (country, mindt, maxdt, warnings)

        query = """select min(FromDate), max(ToDate) from IDC.dbo.NonNAAssets where AssetId=%(assetid)d"""
        #logging.debug(query)
        cur.execute(query % {'assetid': assetid})
        r = cur.fetchall()
        if len(r) == 1 and r[0][0]:
            miniddt, maxiddt = r[0][0], r[0][1]
            miniddt, maxiddt = datetime.datetime.strptime(miniddt, "%Y-%m-%d").date(), datetime.datetime.strptime(maxiddt, "%Y-%m-%d").date()
        else:
            miniddt = datetime.date(1950, 1, 1)
            maxiddt = datetime.date(9999, 12, 31)

    if miniddt > mindt:
        mindt = miniddt
    if maxdt < maxiddt:
        maxdt = maxiddt

    if maxdt == datetime.date(2099, 12, 31):
        maxdt = datetime.date(9999, 12, 31)

    return (country, mindt, maxdt, warnings)


def getIdcMappings(idccur, assetDict, idType, historical=False, ignorePrices=False):
    logging.debug("in getIdcMappings() %s/%s, historical %s" % (idType, assetDict, historical))
    mappings = []

    if historical:
        historicalStr=''
    else:
        historicalStr = " and a.[Pricing Date]>='1-Jan-2016'"

    if ignorePrices:
        pricesStr=''
    else:
        pricesStr = "  and (NAV >0 or [close]>0)  "

    queryna = """select distinct assetid, a.cusip, a.%s, a.[Exchange Code] 
            from IDC.dbo.NAPrices a, IDC.dbo.NAAssets b 
            where a.%s='%s' and a.cusip=b.cusip and a.[Exchange Code] = b.[Exchange Code] 
            """+pricesStr+historicalStr
    querynonna = """select distinct assetid, a.sedol, a.%s 
            from IDC.dbo.NonNAPrices a, IDC.dbo.NonNAAssets b 
            where a.%s='%s' and a.sedol=b.sedol
           """+pricesStr+historicalStr

    exchQuery = """select code, country from IDC.dbo.NAExchangeCodes"""
    idccur.execute(exchQuery)
    exchDict={}
    for r in idccur.fetchall():
        exchDict[r[0]] = r[1]

    idsFound = []
    for key, asset in assetDict.items():
        if asset.duplicateMarketDbMapping is not None:
            logging.debug('asset %s is already mapped in marketDB to %s', key, asset.duplicateMarketDbMapping )
#             continue
        idval = asset.idValue
        logging.debug("query to run: "+queryna % (idType, idType, idval))
        idccur.execute(queryna % (idType, idType, idval))
        r = idccur.fetchall()
        if len(r) == 0 or (len(r)==1 and asset.country !=None and asset.country != exchDict.get(str(r[0][3]))):
            msg = "Pricing not found for %s/%s/historical=%s in NA" % (idType, idval, historical)
            #  asset.warnings.append(msg)
            logging.debug(msg)
        else:
            found = False

            if len(r) > 1 and asset.country is not None:
                for row in r:
                    country = row[3]
                    print(exchDict.get(str(row[3]), None), asset.country)
                    if exchDict.get(str(row[3]), None) == asset.country:
                        assetid = row[0]
                        found = True
                        break
            elif len(r) > 0 and r[0][0]:
                assetid = r[0][0]
                found = True
#             elif len(r) > 1:
#                 msg = "Too many records for %s/%s in NA" % (idType, idval)
#                 asset.warnings.append(msg)
#                 logging.error(msg)

            if found:
                mappings.append((assetid, 'NA', None,idval, key))
                idsFound.append(idval)
                logging.info("Found mapping for %s/%s/%s in NA" % (idType, idval, assetid))

    if idType.lower() != 'cusip':
        for key, asset in assetDict.items():
            idval = asset.idValue
            logging.debug("query to run: "+querynonna % (idType, idType, idval))
            idccur.execute(querynonna % (idType, idType, idval))
            r = idccur.fetchall()
            if len(r) == 0:
                msg = "Pricing not found for %s/%s/historical=%s in NonNA" % (idType, idval, historical)
                #   asset.warnings.append(msg)
                logging.debug(msg)
            elif len(r) == 1 and r[0][0]:
                assetid = r[0][0]
                mappings.append((assetid, 'NonNA', None, idval, key))
                idsFound.append(idval)
                logging.info("Found mapping for %s/%s/%s in NonNA" % (idType, idval, assetid))
            elif len(r) > 1:
                msg = "Too many records for %s/%s in NonNA" % (idType, idval)
                asset.warnings.append(msg)
                logging.error(msg)

    for key, asset in assetDict.items():
        idval = asset.idValue
        if idval not in idsFound:
            if historical:
                s1=''
            else:
                s1 = " after Jan 01 2016 "
            if ignorePrices:
                s2=''
            else:
                s2 = " with positive prices "
            msg = "%s/%s not found in either NA or NonNA%s%s" % (idType, idval, s2, s1)
            asset.warnings.append(msg)
            logging.error(msg)

    mappingsnopricing = []
    if ignorePrices:
        for assetid, assetsrc, axiomadataid, idval, key in mappings:
            (country, mindt, maxdt, warnings) = getCountryMinMaxDate(idccur, assetid, assetsrc, ignorePrices=ignorePrices)
            if country is not None:
            #logging.debug('%d, %s, %s, %s, %s' % (assetid, assetsrc, mindt, maxdt, country))
                mappingsnopricing.append((assetid, assetsrc, mindt, maxdt, country, axiomadataid, idval, key))
                assetDict[key].warnings.extend(warnings)
        return mappingsnopricing

    mappingswithpricedts = []
    for assetid, assetsrc, axiomadataid, idval, key in mappings:
        (country, mindt, maxdt, warnings) = getCountryMinMaxDate(idccur, assetid, assetsrc)
        if country is not None:
            #logging.debug('%d, %s, %s, %s, %s' % (assetid, assetsrc, mindt, maxdt, country))
            mappingswithpricedts.append((assetid, assetsrc, mindt, maxdt, country, axiomadataid, idval, key))
        assetDict[key].warnings.extend(warnings)
    return mappingswithpricedts


def checkForMapping(mappingType, marketDb, assetDict):

    if mappingType == 'marketDb':
        query = """select distinct t1.axioma_id,
                       (select axioma_id from ID_Only_funds t2 where t2.axioma_id=t1.axioma_id),
                       (select from_dt from asset_ref t3 where t3.axioma_id = t1.axioma_id)   
                   from asset_dim_idc_funds_active t1
                   where asset_id=:asset_id and asset_src=:asset_src"""
    elif mappingType == 'modelDb':
        query = """select distinct modeldb_id 
                   from asset_dim_idc_funds_active a, modeldb_global.issue_map b 
                   where asset_id=:asset_id and asset_src=:asset_src and a.axioma_id=b.marketdb_id"""
    else:
        raise Exception('unknown mapping type: %s'%mappingType)

    marketDb.dbCursor.prepare(query)
    newAssetsList=[]
    i=0
    for idValue, asset in assetDict.items():
        i=i+1
        if not "assetId" in asset.idcAsset:
            logging.debug("skipping asset #%d %s because of empty idcAsset %s ",i, asset, asset.idcAsset)
            continue

        logging.debug("i %d asset: %s",i,idValue)
        logging.debug("asset: %s: %s",idValue, str(asset))
        marketDb.dbCursor.execute(None, {'asset_id': asset.idcAsset.get("assetId", None), 'asset_src':asset.idcAsset.get("region",None)})
        r = marketDb.dbCursor.fetchall()
        if len(r) == 0 or r[0][0] is None:
            newAssetsList.append(asset)
            logging.info("%s/%s (%s %s) will be added to %s",asset.idcAsset.get("assetId", None), asset.idcAsset.get("region",None), asset.idType, asset.idValue, mappingType)
        else:
            if mappingType == 'marketDb':
                asset.marketDbMapping=r[0][0]
                asset.idOnlyFunds=r[0][1]
                asset.axiomaIdFromDt=str(r[0][2])[:10]
            elif mappingType == 'modelDb':
                asset.modelDbMapping=r[0][0]
            logging.info("%s/%s already in %s(%s). Will not add",asset.idcAsset.get("assetId", None), asset.idcAsset.get("region",None), mappingType, r[0][0])

    return newAssetsList


def createNewModelDBRecords(marketDb, modelDB, newModelDbAssets, axiomaDataIdMap, newModelDbIds):
    assetIdMap = {}
    query = """select distinct axioma_id, asset_src 
                from marketdb_global.asset_dim_idc_funds 
                where asset_id=:asset_id and asset_src=:asset_src"""
    marketDb.dbCursor.prepare(query)
    for asset in newModelDbAssets:
        if "assetId" not in asset.idcAsset:
            logging.error("no idcAsset for asset %s", str(asset))
            continue
        marketDb.dbCursor.execute(None, {'asset_id': asset.idcAsset["assetId"], 'asset_src': asset.idcAsset["region"]})
        r = marketDb.dbCursor.fetchall()
        if len(r) > 0 and r[0][0] is not None:
            assetIdMap[(asset.idcAsset["assetId"], asset.idcAsset["region"])] = r[0][0]

    query = """select rmg_id, mnemonic from risk_model_group"""
    modelDB.dbCursor.execute(query)
    r = modelDB.dbCursor.fetchall()
    rmgMap = dict([(row[1], row[0]) for row in r])

    issueMapQuery = """insert into issue_map (marketdb_id, modeldb_id, from_dt, thru_dt)
                 values (:marketdb_id, :modeldb_id, :from_dt, :thru_dt)"""
    issueQuery = """insert into issue (issue_id, from_dt, thru_dt)
                    values (:issue_id, :from_dt, :thru_dt)"""
    subIssueQuery = """insert into sub_issue (issue_id, from_dt, thru_dt, sub_id, rmg_id)
                       values (:issue_id, :from_dt, :thru_dt, :sub_id, :rmg_id)"""

    issueMapParamsList = []
    issueParamsList = []
    subIssueParamsList = []
    counter = 0
#     msg=[]

    for asset in newModelDbAssets:
        if "assetId" not in asset.idcAsset:
            logging.error("no idcAsset for asset %s", str(asset))
            continue
#         assetid, assetsrc, mindt, maxdt, country, axiomadataid, idValue
        if asset.idcAsset["maxDt"] == datetime.date(9999, 12, 31):
            asset.idcAsset["maxDt"] = datetime.date(2999, 12, 31)

        marketDbId = assetIdMap.get(( asset.idcAsset["assetId"], asset.idcAsset["region"]), None)
        modelDbId = axiomaDataIdMap.get(asset.idcAsset["axiomaDataId"], None)
        if marketDbId is None or rmgMap.get(asset.idcAsset["tradingCountry"], None) is None:
            s1="Could not find marketDbId for %d/%s in marketdb_global.asset_dim_idc_funds or rmg for %s in risk_model_group" % (asset.idcAsset["assetId"], asset.idcAsset["region"], asset.idcAsset["tradingCountry"])
            asset.warnings.append(s1)
            logging.error(s1)
            raise Exception(s1)
        else:
            if modelDbId is None:
                modelDbId = newModelDbIds[counter]
                counter = counter + 1
            issueMapParams = {'marketdb_id': marketDbId,
                      'modeldb_id': modelDbId,
                      'from_dt': asset.idcAsset["minDt"],
                      'thru_dt': asset.idcAsset["maxDt"]}
            issueMapParamsList.append(issueMapParams)
            asset.issueMapRecord = issueMapParams

            issueParams = {'issue_id': modelDbId,
                      'from_dt': asset.idcAsset["minDt"],
                      'thru_dt': asset.idcAsset["maxDt"]}
            issueParamsList.append(issueParams)
            asset.issueRecord = issueParams

            subIssueParams = {'issue_id': modelDbId,
                          'from_dt': asset.idcAsset["minDt"],
                          'thru_dt': asset.idcAsset["maxDt"],
                          'sub_id': modelDbId+'11',
                          'rmg_id': rmgMap[asset.idcAsset["tradingCountry"]]}
            subIssueParamsList.append(subIssueParams)
            asset.subIssueRecord = subIssueParams

    if len(issueMapParamsList) > 0:
        #print imdictList
        modelDB.dbCursor.executemany(issueMapQuery, issueMapParamsList)
    if len(issueParamsList) > 0:
        #print issdictList
        modelDB.dbCursor.executemany(issueQuery, issueParamsList)
    if len(subIssueParamsList) > 0:
        #print subissdictList
        modelDB.dbCursor.executemany(subIssueQuery, subIssueParamsList)


class Asset():
    def __init__(self, idType, idValue, country=None):
        self.idType = idType
        self.country = country
        self.idValue = str(idValue)
        self.idcAsset = {}
        self.marketDbMapping = None
        self.modelDbMapping = None
        self.assetRefRecord = None
        self.idcFundRecord = None
        self.issueMapRecord = None
        self.issueRecord = None
        self.subIssueRecord = None
        self.warnings = []
        self.idOnlyFunds = None
        self.axiomaIdFromDt = None
        self.duplicateMarketDbMapping = None

    def __str__(self):
#         return 'asset: marketDbMapping %s, modelDbMapping %s,  assetId %s, assetsrc %s, mindt %s, maxdt %s, country %s, axiomadataid %s' % (self.marketDbMapping, self.modelDbMapping,  self.idcAsset.assetId, self.idcAsset.region, self.idcAsset.minDt, self.idcAsset.maxDt, self.idcAsset.tradingCountry, self.idcAsset.axiomaDataId)
        return 'asset: marketDbMapping %s, modelDbMapping %s,    idcAsset %s' % (self.marketDbMapping, self.modelDbMapping,  str(self.idcAsset))


def createUseCase6281(connections,useCaseParamsList, updateExistingUsage=False):
    
    query0 = """
        insert into classification_constituent
        (classification_id,axioma_id,weight,change_dt,change_del_flag,src_id,ref,rev_dt,rev_del_flag)
        select :usage_classification_id,:axioma_id,1, :change_dt, 'N',:src_id,:ref,:rev_dt, 'N' 
        from dual 
        where not exists(select 1 from classification_active_int 
                            where axioma_id=:axioma_id and revision_id=24 %s)
    """
    query1 = query0%''
    query2 = query0%' and classification_id = :usage_classification_id '

    if updateExistingUsage:
        query = query2
    else:
        query = query1
    logging.debug('query %s, useCaseParamsList %s',query, useCaseParamsList)
    connections.axiomaDB.dbCursor.executemany(query, useCaseParamsList)
    query3="select axioma_id from axiomadb.classification_constituent where rev_dt=:rev_dt"
    param1={'rev_dt':connections.marketDB.revDateTime}
    logging.debug('query3 %s, rev_dt %s',query3, param1)
    connections.axiomaDB.dbCursor.execute(query3, param1)
    r = connections.axiomaDB.dbCursor.fetchall()
    return r


def createNewMarketDBRecords(connections, newMarketDbAssets, newIds, srcId, ref= 'Created from IDC Funds data directly'):
    marketDB = connections.marketDB
    assetRefInsertQuery = """insert into asset_ref 
                             (axioma_id, from_dt, thru_dt, src_id, ref, add_dt, trading_country)
                             values (:axioma_id, :from_dt, :thru_dt, :src_id, :ref, :add_dt, :trading_country)"""


    idcFundsInsertQuery = """insert into asset_dim_idc_funds
        (axioma_id, asset_id, asset_src, change_dt, change_del_flag, rev_dt, rev_del_flag, src_id, ref)
        values (:axioma_id, :asset_id, :asset_src, :change_dt, :change_del_flag, :rev_dt, :rev_del_flag,
        :src_id, :ref)"""



    i = 0
    changedt = datetime.date(1950, 1, 1)
    revdt = marketDB.revDateTime

    assetRefParamsList = []
    idcFundsParamsList = []
    useCaseParamsList = []
    useCase=6281


    for asset in newMarketDbAssets:
#         for assetid, assetsrc, mindt, maxdt, country, axiomadataid, idVal in currMappings:
        logging.info("New asset to be added:%s", str(asset.idcAsset))
        if asset.idcAsset.get("tradingCountry", None) is None:
            msg = "ERROR: trading country is missing %s"%str(asset.idcAsset)
            asset.warnings.append(msg)
            logging.error(msg)
        else:
            assetRefParams = {'axioma_id': newIds[i],
                           'from_dt': asset.idcAsset["minDt"],
                           'thru_dt': asset.idcAsset["maxDt"],
                           'src_id': srcId,
                           'ref': ref,
                           'add_dt': revdt,
                           'trading_country': asset.idcAsset["tradingCountry"]
                           }
            assetRefParamsList.append(assetRefParams)
            asset.assetRefRecord = assetRefParams

            idcFundsParams = {'axioma_id': newIds[i],
                         'asset_id': asset.idcAsset["assetId"],
                         'asset_src': asset.idcAsset["region"],
                         'change_dt': changedt,
                         'change_del_flag': 'N',
                         'rev_dt': revdt,
                         'rev_del_flag': 'N',
                         'src_id': srcId,
                         'ref': ref}
            idcFundsParamsList.append(idcFundsParams)
            asset.idcFundRecord = idcFundsParams

            useCaseParams = {'usage_classification_id':useCase,
                             'axioma_id': newIds[i],
                                'change_dt': asset.idcAsset["minDt"],
                                'src_id': srcId,
                                'ref': ref,
                                'rev_dt': revdt}
            useCaseParamsList.append(useCaseParams)
            asset.useCaseRecord = useCaseParams

        i = i + 1

    marketDB.dbCursor.executemany(assetRefInsertQuery, assetRefParamsList)
    marketDB.dbCursor.executemany(idcFundsInsertQuery, idcFundsParamsList)
    createUseCase6281(connections, useCaseParamsList)
#         marketDB.dbCursor.executemany(USE_CASE_INSERT, useCaseParamsList)


def checkMarketDbMappings(marketDB,idList, idType):
    logging.debug("in checkMarketDbMappings idList %s idType %s)",idList, idType)
    assetDict = {}

    for idValue in idList:
        idValue=str(idValue).strip()
        country = None
        if idType == constants.CUSIP_CODE and len(idValue)==12:
            country = idValue[10:12]
            v=idValue[:9]
        else:
            v=idValue

        assetDict[idValue]=Asset(idType, v , country=country)


    query = """select distinct axioma_id, id
                    from asset_dim_%(idType)s_active
                    where id in ('%(idValuesStr)s')  
            """
    if idType == constants.CUSIP_CODE:
        idValuesStr="','".join([s[:9] for s in idList])
    else:
        idValuesStr = "','".join(idList)

    logging.debug(query)
    marketDB.dbCursor.execute(query % {'idType': idType, 'idValuesStr': idValuesStr})
    r = marketDB.dbCursor.fetchall()

    duplicateMarketDbMapping = {}
    for row in r:
        if row[1] in duplicateMarketDbMapping:
            duplicateMarketDbMapping[row[1]]=duplicateMarketDbMapping[row[1]]+','+row[0]
        else:
            duplicateMarketDbMapping[row[1]]=row[0]

    for idValue in idList:
#         print idValue, assetDict[idValue].idValue, assetDict[idValue].idValue in duplicateMarketDbMapping
        if assetDict[idValue].idValue in duplicateMarketDbMapping:
#             print idValue, assetDict[idValue].idValue, assetDict[idValue].idValue in duplicateMarketDbMapping, duplicateMarketDbMapping[assetDict[idValue].idValue]
            assetDict[idValue].duplicateMarketDbMapping = duplicateMarketDbMapping[assetDict[idValue].idValue]

    logging.debug("in checkMarketDbMappings(), retruning assetDict %s)",assetDict)
    return assetDict


def createMappings(connections,idList, idType, srcId, addModelDb, ref = 'from createMappings', useAxiomaDataId=False, historical=False, ignorePrices=False):
    if len(idList) <1:
        return {}
    marketDB = connections.marketDB
    modelDB = connections.modelDB
    qaDirect = connections.qaDirect
    qaDirect.setAutocommit(True)

    assetDict = checkMarketDbMappings(marketDB,idList, idType)
    logging.debug('assetDict %s, %s: %s',idType, idList,assetDict)

    idcMappings = getIdcMappings(qaDirect.dbCursor, assetDict, idType, historical, ignorePrices)
    logging.debug('mappings for %s: %s', idType,idcMappings)

    if len(idcMappings) <1:
        return assetDict

        #assetid, assetsrc, mindt, maxdt, country, axiomadataid, idval
    for (assetId, region, minDt, maxDt, tradingCountry, axiomaDataId, idValue, key) in idcMappings:
        assetDict[key].idcAsset={"assetId": assetId,"region":region, "minDt":minDt,"maxDt":maxDt,"tradingCountry":tradingCountry,"axiomaDataId":axiomaDataId }




    newMarketDbAssets = checkForMapping('marketDb', marketDB, assetDict)
    newModelDbAssets = checkForMapping('modelDb',marketDB, assetDict)

    logging.debug('newMarketDbAssets: %d', len(newMarketDbAssets))
    logging.debug('newModelDbAssets: %s', len(newModelDbAssets))

    logging.debug('newMarketDbAssets: %s', newMarketDbAssets)
    logging.debug('newModelDbAssets: %s', newModelDbAssets)

    if len(newMarketDbAssets) > 0:
        ret=marketDB.createNewMarketIDs(len(newMarketDbAssets))
        newIds = ['O'+r.getIDString()[1:] for r in ret]
        logging.debug('newMarketDbIds created(%d): %s', len(newIds), newIds)

        createNewMarketDBRecords(connections, newMarketDbAssets, newIds, srcId, ref)

    if len(newModelDbAssets) > 0 and addModelDb:
        new = 0
        mdlidindexes = []
        if useAxiomaDataId:
            for asset in newModelDbAssets:
#                 (assetid, assetsrc, mindt, maxdt, country, axiomadataid, idValue)
                if asset.idcAsset["axiomaDataId"] is not None:
                    mdlidindexes.append(asset.idcAsset["axiomaDataId"])
                else:
                    new = new + 1
        else:
            new = len(newModelDbAssets)

        axiomaDataIdMap = {}
        if len(mdlidindexes) > 0:
            for i in mdlidindexes:
                mdldbid = ModelID.ModelID(index=i).getIDString()
                axiomaDataIdMap[i] = mdldbid

        newIds=modelDB.createNewModelIDs(new)
        newIds = [r.getIDString() for r in newIds]
        logging.debug("New modeldbids(%d): %s",len(newIds),newIds)

        createNewModelDBRecords(marketDB, modelDB, newModelDbAssets, axiomaDataIdMap, newIds)
    logging.debug("assetDict: %s", assetDict)
    return assetDict
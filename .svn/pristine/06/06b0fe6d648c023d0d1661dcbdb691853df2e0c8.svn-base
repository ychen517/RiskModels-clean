import datetime
import logging
import sys

from marketdb.Utilities import listChunkIterator
from riskmodels import ModelID, ModelDB, Utilities, getModelByVersion
from riskmodels.ExcludedAssetProcessor import ExcludedAssetProcessor

logger = logging.getLogger('createRMSID')

SRC_ID=900
REF_VAL='createRMSID.py'

ModelTypes = type('Enum', (), {'FUNDAMENTAL': 1, 'STATISTICAL': 2, 'MACRO':3})

def getModelType(mnemonic):
    if mnemonic[-2:] == '-S':
        return ModelTypes.STATISTICAL
    elif mnemonic[-2:] == '-M':
        return ModelTypes.MACRO
    return ModelTypes.FUNDAMENTAL


def createRmgRmsMapping(connections, targetRM):
    """Create a full RMG-->RMS-->FROM/THRU_DT Mapping. We have to consider two tables -
    risk_model_map (rmm) and risk_model_serie (rms) tables. The from_dt should be the max of from_dt from
    rmm and rms, while the thru_dt should always be the min of the same two"""
    modelDB = connections.modelDB
    if targetRM is not None:
        addStr = ' AND serial_id in (%s)'%targetRM.rms_id
    else:
        addStr = ' AND rm.model_id >= 9'
    query = """SELECT serial_id, from_dt, thru_dt, rm.model_region, rm.mnemonic, r1.distribute 
               FROM risk_model_serie r1 JOIN risk_model rm ON rm.model_id=r1.rm_id
               WHERE model_region is not null""" + addStr
    tempRmsDateMap = dict()
    modelDB.dbCursor.execute(query)
    for (rmsId, fDt, tDt, region, mnem, distribute) in modelDB.dbCursor.fetchall():
        #convert to date
        tempRmsDateMap[rmsId] = (fDt, tDt, region, mnem, distribute)

    allRmg = modelDB.getAllRiskModelGroups(inModels=True)
    rmgRmsMapping = dict()
    for rmg in allRmg:
        query = """select rms_id, from_dt, thru_dt from rmg_model_map where rmg_id=:rmg_arg"""
        rmsDateDict = dict()
        modelDB.dbCursor.execute(query, rmg_arg = rmg.rmg_id)
        for (rms_id, rmmFromDt, rmmThruDt) in modelDB.dbCursor.fetchall():
            (rmsFromDt, rmsThruDt, region, mnem, dist) = tempRmsDateMap.get(rms_id,(None,None,None,None,None))
            if rmsFromDt is None or rmsThruDt is None:
                continue
            if rmsFromDt>rmmThruDt or rmsThruDt<rmmFromDt:
                #no overlapping
                continue
            # for fundamental models, use from_dt from rmg_model_map
            # for statistical and macro models, use from_dt from risk_model_serie or rmg_model_map, whichever is later
            if getModelType(mnem) == ModelTypes.FUNDAMENTAL:
                fromDt = rmmFromDt
            else:
                fromDt = max(rmsFromDt,rmmFromDt)
            thruDt = min(rmmThruDt, rmsThruDt)
            if fromDt == thruDt:
                continue
            rmsDateDict.setdefault((fromDt,thruDt), list()).append((rms_id,region,mnem))
        rmgRmsMapping[rmg.rmg_id] = rmsDateDict

    query = """SELECT rmg_id, max(from_dt) FROM rmg_model_map WHERE rms_id > 0 GROUP BY rmg_id"""
    modelDB.dbCursor.execute(query)
    rmgMinDate = dict([(modelDB.getRiskModelGroup(i[0]), i[1]) for i in \
                           modelDB.dbCursor.fetchall()])

    return rmgRmsMapping, tempRmsDateMap, rmgMinDate


class CreateRMSID:
    def __init__(self, connections, riskModel=None, ctry=None):
        self.modelDB = connections.modelDB
        self.mdlCursor = self.modelDB.dbCursor
        self.rmgRmsMapping, self.rmsDateCache, self.rmgMinDate =\
            createRmgRmsMapping(connections, riskModel)
        allRMG = self.modelDB.getAllRiskModelGroups(inModels=True)
        self.idRmgDict = dict((r.rmg_id, r) for r in allRMG)
        self.mnemRmgDict = dict((r.mnemonic, r) for r in allRMG)
        if ctry is not None:
            assert (len(ctry)==2)
            self.targetRMG = self.mnemRmgDict.get(ctry)
            if self.targetRMG is None:
                logging.error('Cannot find the specified country in the risk_model_group table. Not continuing')
                sys.exit(1)
        else:
            self.targetRMG = None
        self.INCR = 200
        self.argList = [('iid%d' % i) for i in range(self.INCR)]
        self.defaultDict = dict([(arg, None) for arg in self.argList])

        self.marketDB = connections.marketDB
        clsFamily = self.marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                               getClassificationFamilyMembers(clsFamily)])
        clsHc1 = clsMembers.get('HomeCountry', None)
        assert(clsHc1 is not None)
        self.clsHc1Rev = self.marketDB.\
            getClassificationMemberRevision(clsHc1, datetime.date(2010, 1, 1))
        assert(self.clsHc1Rev is not None)

        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                               getClassificationFamilyMembers(clsFamily)])
        clsHc2 = clsMembers.get('HomeCountry2', None)
        assert(clsHc2 is not None)
        self.clsHc2Rev = self.marketDB.\
            getClassificationMemberRevision(clsHc2, datetime.date(2010, 1, 1))
        assert(self.clsHc2Rev is not None)

        self.SGMCtry = set(['Australia','Canada','China','Japan','Taiwan','United States','United Kingdom'])
        self.legacyRms = set([11,12,17,18,20,21,29,30,31,32])
        self.modelsExcludingFrontier = set(['21','22'])

        self.etfSet = self.getETFs()
        self.compositeSet= self.getEquityComposites()
        self.mutualFunds = self.getMutualFunds()

    def setHomeCountryCache(self, subIssues, date):
        self.modelDB.getMktAssetClassifications(self.clsHc1Rev,
                                                subIssues, date, self.marketDB)
        self.hc1Cache = self.modelDB.marketClassificationCaches[self.clsHc1Rev.id]
        self.modelDB.getMktAssetClassifications(self.clsHc2Rev,
                                                subIssues, date, self.marketDB)
        self.hc2Cache = self.modelDB.marketClassificationCaches[self.clsHc2Rev.id]

    def getSubIssueInfo(self, issues):
        subIssueInfoDict = dict()
        duplicateCheckDict = dict()
        idStrs = [i.getIDString() for i in issues]
        query = """SELECT si.issue_id, sub_id, si.rmg_id, si.from_dt, si.thru_dt, im.marketdb_id FROM
                   sub_issue si JOIN issue_map im ON si.issue_id=im.modeldb_id
                   WHERE issue_id IN (%(ids)s)
                   AND si.issue_id NOT LIKE 'DCSH_%%'
                   AND si.from_dt<si.thru_dt"""%{
            'ids':','.join([':%s'%i for i in self.argList])}
        self.marketDB.dbCursor.execute("SELECT axioma_id, from_dt FROM asset_ref")
        market_ids = dict(self.marketDB.dbCursor.fetchall())
        for idChunk in listChunkIterator(idStrs, self.INCR):
            myDict = dict(self.defaultDict)
            myDict.update(dict(zip(self.argList, idChunk)))
            self.mdlCursor.execute(query, myDict)
            for (iid, sid, rmgId, fromDt, thruDt, market_id) in self.mdlCursor.fetchall():
                if market_id not in market_ids:
                    logging.debug("Can't find market ID for issue %s", iid)
                    continue
                iid = ModelID.ModelID(string=iid)
                sid = ModelDB.SubIssue(string=sid)
                rmg = self.idRmgDict.get(rmgId)
                subIssueInfoDict[sid] = (rmg, fromDt, thruDt, iid)
                duplicateCheckDict.setdefault(iid, set()).add(sid)

        #if there are two sids associated with one issue-id, don't put it into the subIssueInfoDict
        for (iid, sidList) in duplicateCheckDict.items():
            if len(sidList) > 1:
                for sid in sidList:
                    subIssueInfoDict.pop(sid)
                logging.error('Not processing %s because it got two sub-issue ids associated'%iid.getIDString())

        return subIssueInfoDict

    def mapMarketToModel(self, axids, dateAware=False):
        """Map marketdb Axioma IDs to their modeldb counterparts
        and returns them as a set.
        If an Axioma ID is mapped to multiple model IDs through time
        then all of them are included.
        """
        compMdlIDs = set()
        for axid in axids:
            if dateAware == False:
                self.mdlCursor.execute("""SELECT modeldb_id FROM issue_map
                                   WHERE marketdb_id=:axid""", axid=axid)

            else:
                (axidStr, fromDt, thruDt) = axid
                self.mdlCursor.execute("""SELECT distinct modeldb_id FROM issue_map
                                   WHERE marketdb_id=:axid AND
                                   from_dt<:thru_arg AND thru_dt>=:from_arg""",
                                       axid=axidStr, from_arg=fromDt, thru_arg= thruDt)
            compMdlIDs.update(i[0] for i in self.mdlCursor.fetchall())
        return compMdlIDs

    def getAllClassifiedAssets(self,
                               clsMember, clsFamily, clsDate, clsCodesList):
        clsFamily = self.marketDB.getClassificationFamily(clsFamily)
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                               getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get(clsMember, None)
        assert(clsMember is not None)
        clsRevision = self.marketDB.\
            getClassificationMemberRevision(clsMember, clsDate)
        allClassificationIDs = set()
        for code in clsCodesList:
            crefs = self.marketDB.getClassificationsByCode(datetime.date.today(), code)
            allClassificationIDs.update(
                cr.id for cr in crefs if cr.revision_id == clsRevision.id)
        self.marketDB.dbCursor.execute("""SELECT DISTINCT 
                                  cca.axioma_id, cai.from_dt, cai.thru_dt
                                  FROM classification_const_active cca, 
                                  classification_active_int cai
                                  WHERE cca.classification_id in (%(cids)s)
                                  AND cai.axioma_id=cca.axioma_id
                                  AND cca.classification_id=cai.classification_id
                                  AND change_del_flag='N'""" % {
                'cids': ','.join('%d' % i for i in allClassificationIDs)})

        axids = [(i[0],i[1].date(),i[2].date()) for i in self.marketDB.dbCursor.fetchall()]
        return self.mapMarketToModel(axids, True)

    def getEquityComposites(self):
        """Returns the set of model IDs (strings) that corresponds to
        equity composites that we track as actual composites based
        on marketdb_global.composite_member.
        Such model IDs will be excluded from all models.
        """
        # compFamilies = self.marketDB.getETFFamilies()
        # compAxids = set()
        # for fam in compFamilies:
        #     members = self.marketDB.getAllETFFamilyMembers(fam)
        #     compAxids.update(m.axioma_id for m in members
        #                      if m.axioma_id is not None)
        # return self.mapMarketToModel([i.getIDString() for i in compAxids])
        compETF = self.getAllClassifiedAssets('Axioma Asset Type',
            'ASSET TYPES', datetime.date.today(), ['ComETF'])
        return compETF

    def getETFs(self):
        """Returns the set of model IDs (strings) that corresponds
        to all model IDs tagged as ETFs by DataStream or FTID.
        Such model IDs are not allowed in the fundamental models
        but are okay to include in statistical models.
        """
        etf2 = self.getAllClassifiedAssets('Axioma Asset Type',
            'ASSET TYPES', datetime.date.today(), ['ComETF','NonEqETF'])
        logging.info("Found %s assets classified as ETF by AxiomaSecType"%len(etf2))

        return etf2

    def getMutualFunds(self):
        """REturns the set of model IDs that corresponds to
        all model IDs tagged as Mutual Funds. Such model IDs are
        excluded from all models.
        """
        self.modelDB.dbCursor.execute("""SELECT modeldb_id FROM issue_map WHERE marketdb_id LIKE 'O%'""")
        funds = [m[0] for m in self.modelDB.dbCursor.fetchall()]
        logging.info("Found %s assets classified as mutual fund by axioma ID"%len(funds))

        return funds

    def checkOverlap(self, rmsFDt, rmsTDt, siFDt, siTDt):
        if siTDt<rmsFDt:
            return False
        if siFDt>rmsTDt:
            return False
        return True

    def getRMSInfo(self, issues, version3=True, includeNonEquity=False):
        """Returns dictionary mapping modelIDs in issues to dictionary
        mapping rms_id to (from_dt, thru_dt)
        """
        rmsInfo = dict()
        subIssueInfo = self.getSubIssueInfo(issues)
        self.setHomeCountryCache(list(subIssueInfo.keys()), datetime.date(2999,12,31))
        logging.info('Working on %s sub_issue_ids'%len(list(subIssueInfo.keys())))
        modelRmgSet = set([self.idRmgDict.get(i,i) for i in self.rmgRmsMapping.keys()])
        count = len(list(subIssueInfo.keys()))
        for sid in subIssueInfo.keys():
            count -= 1
            if count%10000 == 0 and count !=0:
                logging.info('%d sub_issue_ids remain to be processed', count)
            if sid.getSubIDString()[0:10] in self.mutualFunds:
                # Do not assign rms_ids to mutual funds
                continue
            if sid.getSubIDString()[0:10] in self.compositeSet:
                #Do not assign rms_ids to composite ETFs
                continue
            # Flag non-equity ETFs so they can be excluded from fundamental models
            isNonEquity = sid.getSubIDString()[0:10] in self.etfSet
            hc1rmsDict = dict()
            hc2rmsDict = dict()
            rmsDict =  dict()
            (hc1List, hc2List, rmg) = (None, None, None)
            (rmg, subIssueFromDt, subIssueThruDt, iid) = subIssueInfo.get(sid, (None, None, None, None))
            if rmg is None:
                logging.debug('Missing rmg_id in sub_issue table/rmg does not belong to any model for sid %s'%sid.getSubIDString())
                continue

            hc1List = [i for i in self.hc1Cache.historyDict.get(sid.getModelID()) if i is not None]
            hc2List = [i for i in self.hc2Cache.historyDict.get(sid.getModelID()) if i is not None]

            hc1ClsList = [self.hc1Cache.classificationDict.get(hc1Cls.classification_id) for (hc1fDt, f, hc1Cls) in hc1List if hc1Cls is not None]
            hc2ClsList = [self.hc2Cache.classificationDict.get(hc2Cls.classification_id) for (hc2fDt, f, hc2Cls) in hc2List if hc2Cls is not None]

            hc1CodeList = [i.code for i in hc1ClsList if i is not None]
            hc1RmgSet = set([self.mnemRmgDict.get(i,i) for i in hc1CodeList if i is not None])
            hc2CodeList = [i.code for i in hc2ClsList if i is not None]
            hc2RmgSet = set([self.mnemRmgDict.get(i,i) for i in hc2CodeList if i is not None])

            assetRmgSet = set(hc1RmgSet).union(hc2RmgSet, set([rmg]))
            if self.targetRMG is not None and self.targetRMG not in assetRmgSet:
                #If we have specified to process exact country exposure, skip those not fitted
                continue

            if len(modelRmgSet.intersection(set(assetRmgSet))) == 0:
                #there is no overlapping between the model rmg and the asset hc1/hc2/rmg, skip
                continue

            hc1 = None
            if hc1List != []:
                for hc1Pos, hc1Tuple in enumerate(hc1List):
                    (hc1FromDt, delFlag, hc1Cls) = hc1Tuple
                    if delFlag == 'Y':
                        continue
                    if hc1Pos != len(hc1List) -1 :
                        hc1ThruDt = hc1List[hc1Pos+1][0]
                    else:
                        hc1ThruDt = subIssueThruDt.date()
                    mkthc1 = self.hc1Cache.classificationDict.get(hc1Cls.classification_id)
                    if mkthc1 is None:
                        logging.debug('Cannot identify classification_id %s in marketdb',hc1Cls.classification_id)
                        continue
                    hc1 = self.mnemRmgDict.get(mkthc1.code)
                    if hc1 is None:
                        logging.debug('Please specify country %s in the risk_model_group table. Sub-issue-id:%s',mkthc1.code,sid.getSubIDString())
                        continue
                    rmsMap = self.rmgRmsMapping.get(hc1.rmg_id)
                    if rmsMap is not None:
                        for (rmsFromDate,rmsThruDate) in sorted(rmsMap.keys()):
                            if self.rmgMinDate[rmg].date() >= rmsThruDate.date():
                                # if the rms thruDate is earlier than the rmgMinDate
                                # ignore rms_id since the trading country is not eligible to the model
                                continue
                            if self.checkOverlap(rmsFromDate.date(), rmsThruDate.date(), hc1FromDt, hc1ThruDt):
                                fDt = max(hc1FromDt, rmsFromDate.date())
                                tDt = min(hc1ThruDt, rmsThruDate.date())
                                for (rms, region, mnem) in rmsMap[(rmsFromDate,rmsThruDate)]:
                                    #alau: move the below into a method
                                    if region in self.SGMCtry and rmg.description != region and 'CN4' not in mnem:
                                        continue
                                    if rms in self.legacyRms and rmg != hc1:
                                    #EM1/EU1 is the only legacy models that
                                    #we dont't want to assign rms_ids base on HC1
                                    #recall v1 models have rms_ids base on RMG only
                                        if rmg.mnemonic != 'HK' or hc1.mnemonic != 'CN':
                                        #Except HK traded, hc1=CN assets
                                            continue
                                    # Exclude non-equities from fundamental models
                                    if isNonEquity and ((getModelType(mnem) == ModelTypes.FUNDAMENTAL) and not includeNonEquity):
                                        continue
                                    myfDt = fDt
                                    if (getModelType(mnem) != ModelTypes.FUNDAMENTAL or self.rmsDateCache[rms][0] < self.rmgMinDate[rmg]) \
                                        and fDt<self.rmgMinDate[rmg].date():
                                        # special case to allow LK in AP models before 2003
                                        if not version3 and not ('AP' in mnem and rmg.mnemonic == 'LK'):
                                            # if stat model or series is published before
                                            # minimum date for the trading country
                                            # then move the fromDt to align with the trading country fromDate
                                            myfDt = self.rmgMinDate[rmg].date()
                                    if rms not in hc1rmsDict:
                                        hc1rmsDict[rms] = (myfDt, tDt)
                                    else:
                                    #That means the new hc will have the same model membership
                                    #as the old hc and hence extend its rms thruDt
                                        (pFDt, ptDt) = hc1rmsDict[rms]
                                        hc1rmsDict[rms] = (pFDt, tDt)

            if len(hc2List) != 0:
                for hc2Pos, hc2Tuple in enumerate(hc2List):
                    (hc2FromDt, delFlag, hc2Cls) = hc2Tuple
                    if delFlag == 'Y':
                        continue
                    if hc2Pos != len(hc2List) -1 :
                        hc2ThruDt = hc2List[hc2Pos+1][0]
                    else:
                        hc2ThruDt = subIssueThruDt.date()
                    mkthc2 = self.hc2Cache.classificationDict.get(hc2Cls.classification_id)
                    if mkthc2 is None:
                        logging.debug('Cannot identify classification_id %s in marketdb'%hc2Cls.classification_id)
                        continue
                    hc2 = self.mnemRmgDict.get(mkthc2.code)
                    if hc2 is None:
                        logging.debug('Please specify country %s in the risk_model_group table. Sub-issue-id:%s'%(mkthc2.code,sid.getSubIDString()))
                        continue
                    rmsMap = self.rmgRmsMapping.get(hc2.rmg_id)
                    if rmsMap is not None:
                        for (rmsFromDate,rmsThruDate) in sorted(rmsMap.keys()):
                            if self.rmgMinDate[rmg].date() >= rmsThruDate.date():
                                # if the rms thruDate is earlier than the rmgMinDate
                                # ignore rms_id since the trading country is not eligible to the model
                                continue
                            if self.checkOverlap(rmsFromDate.date(), rmsThruDate.date(), hc2FromDt, hc2ThruDt):
                                fDt = max(hc2FromDt, rmsFromDate.date())
                                tDt = min(hc2ThruDt, rmsThruDate.date())
                                for (rms, region, mnem) in rmsMap[(rmsFromDate,rmsThruDate)]:
                                    if region in self.SGMCtry and rmg.description != region and 'CN4' not in mnem:
                                        continue
                                    if rms in self.legacyRms and rmg != hc2:
                                        continue
                                    if isNonEquity and ((getModelType(mnem) == ModelTypes.FUNDAMENTAL) and not includeNonEquity):
                                        continue
                                    myfDt = fDt
                                    if (getModelType(mnem) != ModelTypes.FUNDAMENTAL or self.rmsDateCache[rms][0] < self.rmgMinDate[rmg]) \
                                        and fDt<self.rmgMinDate[rmg].date():
                                        # special case to allow LK in AP models before 2003
                                        if not version3 and not ('AP' in mnem and rmg.mnemonic == 'LK'):
                                            # if stat model or series is published before
                                            # minimum date for the trading country
                                            # then move the fromDt to align with the trading country fromDate
                                            myfDt = self.rmgMinDate[rmg].date()
                                    if rms not in hc2rmsDict:
                                        hc2rmsDict[rms] = (myfDt, tDt)
                                    else:
                                        (pFDt, ptDt) = hc2rmsDict[rms]
                                        hc2rmsDict[rms] = (pFDt, tDt)
                rmsDict = self.processHc1Hc2Dict(hc1rmsDict, hc2rmsDict)
            else:
                rmsDict = hc1rmsDict
            rmsMap = self.rmgRmsMapping.get(rmg.rmg_id)
            if rmsMap is not None:
                for (rmsFromDate,rmsThruDate) in sorted(rmsMap.keys()):
                    if self.rmgMinDate[rmg].date() >= rmsThruDate.date():
                        # if the rms thruDate is earlier than the rmgMinDate
                        # ignore rms_id since the trading country is not eligible to the model
                        continue
                    if self.checkOverlap(rmsFromDate.date(), rmsThruDate.date(), \
                                             subIssueFromDt.date(), subIssueThruDt.date()):
                        fDt = max(subIssueFromDt.date(), rmsFromDate.date())
                        tDt = min(subIssueThruDt.date(), rmsThruDate.date())
                        for (rms, region, mnem) in rmsMap[(rmsFromDate,rmsThruDate)]:
                            if region in self.SGMCtry and rmg.description != region and 'CN4' not in mnem:
                                continue
                            if isNonEquity and ((getModelType(mnem) == ModelTypes.FUNDAMENTAL) and not includeNonEquity):
                                continue
                            if rms in rmsDict:
                                (oldFDt, oldTDt) = rmsDict[rms]
                                #Hypothetic case 1: HC1=IL, RMG=BR. ThruDt for EM should be 2999/12/31
                                #Case 2: HC1=QA, RMG=CN. FromDt for EM should be 2003/01/01
                                finalfDt = min(oldFDt, fDt)
                                # if stat model or series is published before
                                # minimum date for the trading country
                                # then move the fromDt to align with the trading country fromDate
                                if (getModelType(mnem) != ModelTypes.FUNDAMENTAL or self.rmsDateCache[rms][0] < self.rmgMinDate[rmg]) \
                                    and finalfDt<self.rmgMinDate[rmg].date():
                                    # special case to allow LK in AP models before 2003
                                    if not version3 and not ('AP' in mnem and rmg.mnemonic == 'LK'):
                                        finalfDt = self.rmgMinDate[rmg].date()
                                finaltDt = max(oldTDt, tDt)
                                rmsDict[rms] = (finalfDt, finaltDt)
                            else:
                                rmsDict[rms] = (fDt, tDt)
            # last check - fix dates that are outside of issue dates
            for rms in rmsDict.keys():
                if subIssueFromDt.date() > rmsDict[rms][0]:
                    # sub_issue starts after country assignment
                    rmsDict[rms] = (subIssueFromDt.date(), rmsDict[rms][1])
                if subIssueThruDt.date() < rmsDict[rms][1]:
                    # sub_issue ends before country assignment does
                    rmsDict[rms] = (rmsDict[rms][0], subIssueThruDt.date())
                if rmg.rmg_id >=75 and not version3:
                    allowFrontier = False
                    for e in self.modelsExcludingFrontier:
                        if e in self.rmsDateCache[rms][3]:
                        #Dirty code to get rid of frontier markets from
                        #joining models prior to gen 2.1
                            allowFrontier = True
                    if not allowFrontier:
                        rmsDict.pop(rms)
            rmsInfo[iid] = rmsDict

        logger.debug('rmsInfo before exclusions: %s', rmsInfo)
        ex = ExcludedAssetProcessor(self.marketDB, self.modelDB)
        ex.processAssets(rmsInfo)
        logger.debug('rmsInfo after exclusions: %s', rmsInfo)

        displayCorrectInfo = False
        if displayCorrectInfo:
            for iid in issues:
                rmsDict = rmsInfo.get(iid, dict())
                print('nonEquity?', isNonEquity)
                for serialId in sorted(rmsDict.keys()):
                    (trueFromDt, trueThruDt) = rmsDict[serialId]
                    print(serialId, iid.getIDString(), trueFromDt.isoformat()[0:10], trueThruDt.isoformat()[0:10])
        return rmsInfo

    def processHc1Hc2Dict(self, hc1rmsDict, hc2rmsDict):
        rmsDict = dict()
        fullRms = set(hc1rmsDict.keys()).union(set(hc2rmsDict.keys()))
        for rms in fullRms:
            (hc1FromDt, hc1ThruDt) = hc1rmsDict.get(rms, (None,None))
            (hc2FromDt, hc2ThruDt) = hc2rmsDict.get(rms, (None,None))
            if hc2FromDt == None:
                #Rms only associated with hc1
                rmsDict[rms] = (hc1FromDt, hc1ThruDt)
            elif hc1FromDt == None:
                rmsDict[rms] = (hc2FromDt, hc2ThruDt)
            else:
                fDt = min(hc1FromDt, hc2FromDt)
                tDt = max(hc1ThruDt, hc2ThruDt)
                rmsDict[rms] = (fDt, tDt)
        return rmsDict


class InsertUpdateRMSID:
    def __init__(self, connections, options):
        self.options= options
        self.modelDB = connections.modelDB
        self.mdlCursor = connections.modelDB.dbCursor
        self.allowAdd = options.addRecords
        self.allowDel = options.removeRecords
        self.allowEstuDel = options.removeESTURecords
        self.allowUp = options.updateRecords
        self.allowEstuUp = options.updateESTURecords

        self.INCR = 200
        self.argList = [('iid%d' % i) for i in range(self.INCR)]
        self.defaultDict = dict([(arg, None) for arg in self.argList])

    def executeMany(self, cursor, query, values, args):
        """
        Execute a query which only takes a subset of the values found
        in the dictionaries in values

        Arguments:

        cursor - the cursor used to execute the query
        query - the query string
        values - list of dictionaries containing the values to be passed to the query,
        plus extra arguments
        args - list of strings which are the arguments required by the query
        """
        newvals = []
        for v in values:
            d = dict()
            for s in args:
                d[s] = v[s]
            newvals.append(d)
        cursor.executemany(query, newvals)

    def process(self, rmsInfo):
        if list(rmsInfo.keys()) == []:
            logging.info('No rms_id can be assigned')
            return None

        curRMSrecords = self.getCurrentRecords(rmsInfo, 'rms_issue')
        curEsturecords = self.getCurrentRecords(rmsInfo, 'rms_estu_excl_active_int')

        if self.options.targetRM is not None:
            if self.options.otherESTU !=None:
                otherEsturecords = self.getCurrentRecords(\
                    rmsInfo, 'rms_estu_excl_active_int',self.options.otherESTU, False)
            else:
                otherEsturecords = self.getCurrentRecords(\
                    rmsInfo, 'rms_estu_excl_active_int', None, True)
        else:
            #if we work on all rms_ids, do not bother with the estu records
            otherEsturecords = dict()

        addDicts = list()
        estuAddDicts = list()
        removeDicts = list()
        estuRemoveDicts = list()
        updateDicts = list()
        estuUpdateDicts = list()

        count = len(rmsInfo)
        logging.info('Now process insert/update dict for %s issue_ids'%count)
        for iid, correctRMSDict in rmsInfo.items():
            count -= 1
            if count%10000 == 0 and count !=0:
                logging.info('%s issue_ids remain to be processed'%count)
#            correctRMSDict = rmsInfo[iid]
            currentRMSDict = curRMSrecords.get(iid, dict())
            currentEstuDict = curEsturecords.get(iid, dict())
            currentOtherEstuDict = otherEsturecords.get(iid, dict())

            for rmsID in set(correctRMSDict.keys()).union(set(currentRMSDict.keys())):
                (newFromDt, newThruDt) = correctRMSDict.get(rmsID, (None, None))
                (curFromDt, curThruDt) = currentRMSDict.get(rmsID, (None, None))
                (estuFromDt, estuThruDt) = currentEstuDict.get(rmsID, (None, None))
                if rmsID not in currentEstuDict and rmsID in currentOtherEstuDict:
                    (otherEstuFromDt, otherEstuThruDt)=currentOtherEstuDict[rmsID]
                    #Found in derived rms_ids but not in current one
                    if not otherEstuThruDt<newFromDt and not otherEstuFromDt>newThruDt:
                    #No overlapping
                        if otherEstuThruDt>newThruDt:
                        #if the estu thru date is later than the model thru date, align it with the latter
                            otherEstuThruDt = newThruDt
                        if otherEstuFromDt<newFromDt:
                        #if the estu from date is earlier than the model start date, align it with the latter
                            otherEstuFromdt = newFromDt
                        #if we're left with a valid interval
                        if otherEstuFromDt < otherEstuThruDt:
                            estuAddDicts.append({'issue':iid.getIDString(), 'fromDt':otherEstuFromDt,
                                                 'thruDt': otherEstuThruDt, 'rms_id':rmsID})
                if rmsID not in currentRMSDict:
                    #New record
                    addDicts.append({'issue':iid.getIDString(), 'fromDt':newFromDt,
                                     'thruDt': newThruDt, 'rms_id':rmsID})
                elif rmsID not in correctRMSDict:
                    #Remove existing records
                    removeDicts.append({'issue': iid.getIDString(), 'fromDt': curFromDt,
                                        'thruDt': curThruDt, 'rms_id': rmsID})
                    if rmsID in currentEstuDict:

                        estuRemoveDicts.append(
                            {'issue': iid.getIDString(), 'rms_id': rmsID,
                             'fromDt': estuFromDt, 'thruDt': estuThruDt})
                else:
                    #Overlap
                    if newFromDt != curFromDt or newThruDt != curThruDt:
                        updateDicts.append({'issue': iid.getIDString(), 'rms_id': rmsID,
                                            'oldFromDt': curFromDt,
                                            'oldThruDt': curThruDt,
                                            'newFromDt': newFromDt,
                                            'newThruDt': newThruDt})
                    if rmsID in currentEstuDict:
                        #Do the same date filter logic for insert
                        changed = False
                        if estuThruDt<newFromDt or estuFromDt>newThruDt:
                            continue
                        #alau: change to use min/max, instead of changed
                        if estuThruDt>newThruDt:
                            newEstuThruDt = newThruDt
                            changed = True
                        else:
                            newEstuThruDt = estuThruDt

                        if estuFromDt<newFromDt:
                            newEstuFromDt = newFromDt
                            changed = True
                        else:
                            newEstuFromDt = estuFromDt

                        if changed:
                            estuUpdateDicts.append(
                                {'issue': iid.getIDString(), 'rms_id': rmsID,
                                 'oldFromDt': estuFromDt,
                                 'oldThruDt': estuThruDt,
                                 'newFromDt': newEstuFromDt,
                                 'newThruDt': newEstuThruDt})

        if hasattr(self.options, 'writeOutFile') and self.options.writeOutFile:
            try:
                outFile = open('rms_check_%s.csv'%self.options.targetRM.rms_id,'w')
            except:
                outFile = open('rms_check.csv','w')
            if len(removeDicts)>0:
                outFile.write('RECORDS TO BE REMOVED FROM RMS_ISSUE TABLE\n')
                outFile.write('-'*50)
                outFile.write('\n')
                outFile.write('ISSUE_ID,OLD_FROM_DT,OLD_THRU_DT,RMS_ID\n')
                for d in sorted(removeDicts, key=lambda x: x['issue']):
                    result = [d['issue'],d['fromDt'],d['thruDt'],d['rms_id']]
                    outFile.write(','.join('%s'%i for i in result))
                    outFile.write('\n')
                outFile.write('\n')
            if len(estuRemoveDicts)>0:
                outFile.write('RECORDS TO BE REMOVED FROM RMS_ESTU_EXCLUDED TABLE\n')
                outFile.write('-'*50)
                outFile.write('\n')
                outFile.write('ISSUE_ID,OLD_FROM_DT,OLD_THRU_DT,RMS_ID\n')
                for d in sorted(estuRemoveDicts, key=lambda x: x['issue']):
                    result = [d['issue'],d['fromDt'],d['thruDt'],d['rms_id']]
                    outFile.write(','.join('%s'%i for i in result))
                    outFile.write('\n')
                outFile.write('\n')
            if len(addDicts)>0:
                outFile.write('RECORDS TO BE WRITTEN INTO RMS_ISSUE TABLE\n')
                outFile.write('-'*50)
                outFile.write('\n')
                outFile.write('ISSUE_ID,NEW_FROM_DT,NEW_THRU_DT,RMS_ID\n')
                for d in sorted(addDicts, key=lambda x: x['issue']):
                    result = [d['issue'],d['fromDt'],d['thruDt'],d['rms_id']]
                    outFile.write(','.join('%s'%i for i in result))
                    outFile.write('\n')
                outFile.write('\n')
            if len(estuAddDicts)>0:
                outFile.write('RECORDS TO BE WRITTEN INTO RMS_ESTU_EXCLUDED TABLE\n')
                outFile.write('-'*50)
                outFile.write('\n')
                outFile.write('ISSUE_ID,NEW_FROM_DT,NEW_THRU_DT,RMS_ID\n')
                for d in sorted(estuAddDicts, key=lambda x: x['issue']):
                    result = [d['issue'],d['fromDt'],d['thruDt'],d['rms_id']]
                    outFile.write(','.join('%s'%i for i in result))
                    outFile.write('\n')
                outFile.write('\n')
            if len(updateDicts)>0:
                outFile.write('RECORDS TO BE UPDATED TO RMS_ISSUE TABLE\n')
                outFile.write('-'*50)
                outFile.write('\n')
                outFile.write('ISSUE_ID,OLD_FROM_DT,NEW_FROM_DT,OLD_THRU_DT,NEW_THRU_DT,RMS_ID\n')
                for d in sorted(updateDicts, key=lambda x: x['issue']):
                    result = [d['issue'],d['oldFromDt'],d['newFromDt'],d['oldThruDt'],d['newThruDt'],d['rms_id']]
                    outFile.write(','.join('%s'%i for i in result))
                    outFile.write('\n')
                outFile.write('\n')
            if len(estuUpdateDicts)>0:
                outFile.write('RECORDS TO BE UPDATED TO RMS_ESTU_EXCLUDED TABLE\n')
                outFile.write('-'*50)
                outFile.write('\n')
                outFile.write('ISSUE_ID,OLD_FROM_DT,NEW_FROM_DT,OLD_THRU_DT,NEW_THRU_DT,RMS_ID\n')
                for d in sorted(estuUpdateDicts, key=lambda x: x['issue']):
                    result = [d['issue'],d['oldFromDt'],d['newFromDt'],d['oldThruDt'],d['newThruDt'],d['rms_id']]
                    outFile.write(','.join('%s'%i for i in result))
                    outFile.write('\n')
                outFile.write('\n')

        # print removeDicts, 'remove\n'
        # print estuRemoveDicts, 'removeEstu\n'
        # print addDicts, 'add\n'
        # print estuAddDicts, 'addEstu\n'
        # print updateDicts, 'update\n'
        # print estuUpdateDicts, 'updateEstu\n'

        self.insertUpdate(removeDicts, estuRemoveDicts, addDicts, estuAddDicts, updateDicts, estuUpdateDicts)

    def insertUpdate(self, removeDicts, estuRemoveDicts, addDicts, estuAddDicts, updateDicts, estuUpdateDicts):

        if len(removeDicts)>0 and self.allowDel:
            logging.info('Deleting %d records', len(removeDicts))
            self.mdlCursor.executemany("""DELETE FROM rms_issue
                            WHERE rms_id=:rms_id AND issue_id=:issue AND from_dt=:fromDt
                            AND thru_dt=:thruDt""", removeDicts)

        if len(estuRemoveDicts) >0 and self.allowEstuDel:
            self.mdlCursor.executemany("""INSERT INTO rms_estu_excluded
                            SELECT rms_id, issue_id, change_dt, change_del_flag, %(src)d, '%(ref)s', sysdate, 'Y'
                            FROM rms_estu_excl_active WHERE rms_id=:rms_id AND issue_id=:issue AND
                            change_dt in (:fromDt, :thruDt)""" % dict(src=SRC_ID, ref=REF_VAL),
                            removeDicts)

        if len(addDicts) > 0 and self.allowAdd:
            logging.info('Adding %d new records', len(addDicts))
            self.mdlCursor.executemany("""INSERT INTO rms_issue
                            (rms_id, issue_id, from_dt, thru_dt)
                            VALUES(:rms_id, :issue, :fromDt, :thruDt)""", addDicts)

        if len(estuAddDicts) >0 and self.allowAdd:
            self.executeMany(self.mdlCursor, """INSERT INTO rms_estu_excluded
                             (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
                              VALUES(:rms_id, :issue, :fromDt, 'N', %(src)d, '%(ref)s', sysdate, 'N')""" % \
                            dict(src=SRC_ID, ref=REF_VAL), estuAddDicts, ['rms_id','issue','fromDt'])
            self.executeMany(self.mdlCursor, """INSERT INTO rms_estu_excluded
                             (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
                             VALUES(:rms_id, :issue, :thruDt, 'Y', %(src)d, '%(ref)s', sysdate, 'N')""" % \
                            dict(src=SRC_ID, ref=REF_VAL),
                        [a for a in estuAddDicts if a['thruDt'] < datetime.date(2999,12,31)],
                        ['rms_id','issue','thruDt'])

        if len(updateDicts) > 0 and self.allowUp:
            logging.info('Updating %d records', len(updateDicts))
            self.mdlCursor.executemany("""UPDATE rms_issue
                            SET from_dt=:newFromDt, thru_dt=:newThruDt
                            WHERE rms_id=:rms_id AND issue_id=:issue AND from_dt=:oldFromDt
                            AND thru_dt=:oldThruDt""", updateDicts)

        if len(estuUpdateDicts)>0 and self.allowEstuUp:
            self.updateESTU(estuUpdateDicts, self.mdlCursor)

    def updateESTU(self, estuUpdateDicts, cur):
        self.executeMany(cur, """INSERT INTO rms_estu_excluded
                         SELECT rms_id, issue_id, change_dt, change_del_flag, %(src)d, '%(ref)s', sysdate, 'Y'
                         FROM rms_estu_excl_active WHERE rms_id=:rms_id AND issue_id=:issue AND
                         change_dt = :oldFromDt""" % dict(src=SRC_ID, ref=REF_VAL),
                    [d for d in estuUpdateDicts if d['oldFromDt'] != d['newFromDt']], ['rms_id','issue','oldFromDt'])
        self.executeMany(cur, """INSERT INTO rms_estu_excluded
                         SELECT rms_id, issue_id, change_dt, change_del_flag, %(src)d, '%(ref)s', sysdate, 'Y'
                         FROM rms_estu_excl_active WHERE rms_id=:rms_id AND issue_id=:issue AND
                         change_dt = :oldThruDt""" % dict(src=SRC_ID, ref=REF_VAL),
                    [d for d in estuUpdateDicts if d['oldThruDt'] != d['newThruDt']], ['rms_id','issue','oldThruDt'])
        self.executeMany(cur, """INSERT INTO rms_estu_excluded
                         (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
                         VALUES(:rms_id, :issue, :newFromDt, 'N', %(src)d, '%(ref)s', sysdate, 'N')""" % \
                        dict(src=SRC_ID, ref=REF_VAL),
                    [d for d in estuUpdateDicts if d['oldFromDt'] != d['newFromDt']], ['rms_id','issue','newFromDt'])
        self.executeMany(cur, """INSERT INTO rms_estu_excluded
                         (rms_id, issue_id, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag)
                         VALUES(:rms_id, :issue, :newThruDt, 'Y', %(src)d, '%(ref)s', sysdate, 'N')""" % \
                        dict(src=SRC_ID, ref=REF_VAL),
                    [a for a in estuUpdateDicts if a['newThruDt'] != a['oldThruDt']
                     and a['newThruDt'] < datetime.date(2999,12,31)],
                    ['rms_id','issue','newThruDt'])

    def getCurrentRecords(self, rmsInfo, tableName, otherEstu=None, fromOtherRevision=False):
        if tableName not in ('rms_estu_excl_active_int','rms_issue'):
            logging.error('Unknown table name, not proceeding')
            return dict()

        resultDict = dict()
        iidStrs = [i.getIDString() for i in rmsInfo.keys()]
        query = """SELECT DISTINCT issue_id, rms_id, from_dt, thru_dt FROM
                   %(tableName)s WHERE issue_id IN (%(iids)s)"""%{
            'iids':','.join([':%s'%i for i in self.argList]),
            'tableName':tableName}

        if self.options.targetRM is not None:
            if tableName == 'rms_estu_excl_active_int':
                if otherEstu != None:
                    if otherEstu.isdigit():
                        # rms_id rather than mnemonic
                        query += " AND rms_id=%d"%int(otherEstu)
                    else:
                        query += """ AND rms_id in (SELECT serial_id FROM risk_model_serie rms
                          JOIN risk_model rm ON rms.rm_id=rm.model_id
                          WHERE rm.mnemonic='%s' AND distribute=1)"""%otherEstu
                elif fromOtherRevision is True:
                #Get ESTU exclusions for other revisoins of the model
                    query += ' AND rms_id in (SELECT serial_id FROM risk_model_serie WHERE\
                           rm_id=%s and serial_id <> %s)'%(\
                        self.options.targetRM.rm_id,self.options.targetRM.rms_id)
                else:
                    query += ' AND rms_id in (%s)'%self.options.targetRM.rms_id
            else:
                #Case when tableName = 'rms_issue' and got a specified RM to assign
                query += ' AND rms_id in (%s)'%self.options.targetRM.rms_id

        for iidChunk in listChunkIterator(iidStrs, self.INCR):
            myDict = dict(self.defaultDict)
            myDict.update(dict(zip(self.argList, iidChunk)))
            self.mdlCursor.execute(query, myDict)
            for (iid, rms_id, fromDt, thruDt) in self.mdlCursor.fetchall():
                iid = ModelID.ModelID(string=iid)
                if iid in resultDict:
                    rmsDict = resultDict[iid]
                else:
                    rmsDict = dict()

                if tableName == 'rms_estu_excl_active_int':
                    #replace the derived rms_id by the targeting rms_id
                    if len(rmsInfo[iid]) == 1:
                        rms_id = list(rmsInfo[iid].keys())[0]
                        if rms_id in rmsDict:
                            oldFrom, oldThru = rmsDict[rms_id]
                            newFrom = min(fromDt.date(), oldFrom)
                            newThru = max(thruDt.date(), oldThru)
                            rmsDict[rms_id] = (newFrom, newThru)
                        else:
                            rmsDict[rms_id] = (fromDt.date(), thruDt.date())
                    elif len(list(rmsInfo[iid].keys()))>1:
                        logging.debug('Will not process the estu_exclusion table with mulitple rms_ids: %s'%
                                      ','.join('%s'%i for i in rmsInfo[iid].keys()))
                        continue
                    elif len(rmsInfo[iid]) == 0:
                        #These are those should get removed. No derived rms_ids should be found
                        continue
                else:
                    if rms_id not in rmsDict:
                        rmsDict[rms_id] = (fromDt.date(), thruDt.date())
                    else:
                        logging.error('Same rms_id with mutliple records for id %s in %s table.Skipping'%(iid,tableName))
                        continue
                resultDict[iid] = rmsDict
        return resultDict


def createRmsIssueRecords(modelDb, marketDb, issueId, modelIdList=None):
    logger.debug('in createRmsIssueRecords()')
    options = Utilities.Struct()
    options.restrictList = [issueId]
    options.excludeList = None
    options.addRecords = True
    options.removeRecords = True
    options.updateRecords = True
    options.removeESTURecords = True
    options.updateESTURecords = True
    options.otherESTUMnemonic = None
    options.targetRM = None
    options.writeOutFile = False
#    badModelIds = []
#    goodModelIds = []
#
    if not modelIdList:
        connections = Utilities.Struct()
        connections.marketDB = marketDb
        connections.modelDB = modelDb
        creator = CreateRMSID(connections)
        rmsInfo = creator.getRMSInfo([ModelID.ModelID(string=issueId)])
        rmsInsertUpdate = InsertUpdateRMSID(connections, options)
        rmsInsertUpdate.process(rmsInfo)
        return True

    if modelIdList is not None and len(modelIdList)>0:
        modelIdList=[str(m) for m in modelIdList]
        addClause=" and rm_id in (%s)"%(', '.join(modelIdList))
    else:
        addClause=""
    modelDb.dbCursor.execute("""select serial_id, rm_id, revision 
        from RISK_MODEL_SERIE  
        where distribute=1 
        """+addClause+"order by serial_id")

    r = modelDb.dbCursor.fetchall()
    logger.debug('serial_id, rm_id, revision: %s %s', len(r), r)

    for row in r:
        rmsID = int(row[0])
        rmID = int(row[1])
        modelRev = int(row[2])
        logger.debug('Using rms_id %d, rmID %d, modelRev %d', rmsID, rmID, modelRev)
        try:
            riskModelObj = getModelByVersion(rmID, modelRev)
            logger.debug('riskModelObj: %s', riskModelObj)
            options.targetRM = riskModelObj(modelDb, marketDb)
            logger.debug('options.targetRM: %s', options.targetRM)

#            goodModelIds.append([rmsID,rmID,modelRev])
#            ctryMap, assetRestrictions, assetExclusions =
            connections = Utilities.Struct()
            connections.marketDB = marketDb
            connections.modelDB = modelDb
            creator = CreateRMSID(connections, riskModel=riskModelObj)
            rmsInfo = creator.getRMSInfo([ModelID.ModelID(string=issueId)])
            rmsInsertUpdate = InsertUpdateRMSID(connections, options)
            rmsInsertUpdate.process(rmsInfo)

        except Exception as e:
            logger.exception('Error: %s',e)
            msg = 'ERROR: an error occured creating RMSIssue for rms_id %d, rmID %d, modelRev %d: %s', rmsID, rmID, modelRev, e
            logger.error(msg)
            raise Exception(msg)



#            print 'ctryMap:',ctryMap
#            print 'assetRestrictions:',assetRestrictions
#            print 'assetExclusions:',assetExclusions
#    logger.info('badModelIds: %s', badModelIds)
#    logger.info('goodModelIds: %s', goodModelIds)
#
#    if len(badModelIds)>0:
#        logger.error('some models ids generated errors: %s', badModelIds)
#        return False
    logger.debug('in createRmsIssueRecords(): done')
    return True

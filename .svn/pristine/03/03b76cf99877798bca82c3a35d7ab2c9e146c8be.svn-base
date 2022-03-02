'''
Created on Dec 18, 2017

@author: rgoodman
'''

import datetime

class ExcludedAssetProcessor(object):
    '''
    Processes information from classification_exclude table in marketDB when adding assets to models
    '''

    def createExcludeCache(self):
        '''
        Creates cache of excluded asset types with dates
        '''
        self.excludedAssetTypes = dict()
        self.marketDB.dbCursor.execute("""SELECT classification_id, from_dt, thru_dt
         FROM classification_exclude exc JOIN exclude_logic lg ON exc.exclude_logic_id=lg.id
         WHERE lg.exclude_logic='RMS_ISSUE_EXCLUDE'""")
        for r in self.marketDB.dbCursor.fetchall():
            self.excludedAssetTypes[r[0]] = (r[1].date(), r[2].date())
    
    def __init__(self, marketDB, modelDB):
        '''
        Sets up reference data
        
        @param MarketDB: MarketDB object
        @param ModelDB: ModelDB object
        '''
        self.marketDB = marketDB
        self.modelDB = modelDB
        self.createExcludeCache()

    def setAssetTypeCache(self, subIssues, date):
        clsFamily = self.marketDB.getClassificationFamily('ASSET TYPES')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in self.marketDB.\
                               getClassificationFamilyMembers(clsFamily)])
        clsType = clsMembers.get('Axioma Asset Type', None)
        assert(clsType is not None)
        assetTypeRev = self.marketDB.\
            getClassificationMemberRevision(clsType, datetime.date(2010, 1, 1))
        assert(assetTypeRev is not None)
        self.modelDB.getMktAssetClassifications(assetTypeRev, 
                                                subIssues, date, self.marketDB)
        self.assetTypeCache = self.modelDB.marketClassificationCaches[assetTypeRev.id]
        
    def getThruDt(self, iid):
        self.modelDB.dbCursor.execute("SELECT MAX(thru_dt) FROM sub_issue WHERE issue_id=:iid", iid=iid.getIDString())
        r = self.modelDB.dbCursor.fetchall()
        assert len(r) == 1
        return r[0][0].date()
        
    def processAssets(self, issue_info, date=datetime.date.today()):
        '''
        Adjusts dates in issue_info based on exclusions in marketDB (if any)
        
        @param issue_info: dictionary of ModelID=>(dictionary of rmg_id=>(from_dt, thru_dt))
        @param date: optional date for cache setup; defaults to today
        '''
        self.setAssetTypeCache(list(issue_info.keys()), date)
        for (iid, rmsDict) in issue_info.items():
            assetTypes = [i for i in self.assetTypeCache.historyDict.get(iid) if i != None]
            for assetPos, aTuple in enumerate(assetTypes):
                (aFromDt, delFlag, aCls) = aTuple
                if delFlag == 'Y':
                    continue
                if assetPos != len(assetTypes) -1 :
                    aThruDt = assetTypes[assetPos+1][0]
                else:
                    aThruDt = self.getThruDt(iid)
                if aCls.classification_id in self.excludedAssetTypes:
                    # get dates from classification_exclude
                    for rms in list(rmsDict.keys()):
                        if aFromDt <= self.excludedAssetTypes[aCls.classification_id][1] and \
                            aThruDt >= self.excludedAssetTypes[aCls.classification_id][0]:
                            effFromDt = max(aFromDt, self.excludedAssetTypes[aCls.classification_id][0])
                            effThruDt = min(aThruDt, self.excludedAssetTypes[aCls.classification_id][1])
                            # check for overlap
                            if rmsDict[rms][0] <= effThruDt and rmsDict[rms][1] >= effFromDt:
                                if effFromDt > rmsDict[rms][0] and effFromDt < rmsDict[rms][1]:
                                    # exclude begins during RMS tenure; set rms end_dt to beginning of exclude
                                    rmsDict[rms] = (rmsDict[rms][0], effFromDt)
                                if effThruDt > rmsDict[rms][0] and effThruDt < rmsDict[rms][1]:
                                    # exclude ends during RMS tenure; set rms from_dt to end of exclude
                                    rmsDict[rms] = (effThruDt, rmsDict[rms][1])
                                if effFromDt > rmsDict[rms][0] and effThruDt < rmsDict[rms][1]:
                                    # exclude is entirely within tenure; can we create two tenures?
                                    # for now just have tenure start when exclude ends
                                    rmsDict[rms] = (effThruDt, rmsDict[rms][1])
                                if effFromDt <= rmsDict[rms][0] and rmsDict[rms][1] <= effThruDt:
                                    # exclude includes all dates of tenure; remove asset
                                    rmsDict.pop(rms)


if __name__ == '__main__':
    from riskmodels import ModelDB
    from marketdb import MarketDB
    from riskmodels import ModelID
    mdl = ModelDB.ModelDB(user='modeldb_global', passwd='modeldb_global', sid='glsdg')
    mkt = MarketDB.MarketDB(user='marketdb_global', passwd='marketdb_global', sid='glsdg')
    ex = ExcludedAssetProcessor(mkt, mdl)
    rmgDict = dict()
    rmgDict[180] = (datetime.date(2015,1,8), datetime.date(2999,12,31))
    rmgDict2 = dict()
    rmgDict2[180] = (datetime.date(2017,3,24), datetime.date(2999,12,31))
    rmgDict3 = dict()
    rmgDict3[180] = (datetime.date(2017,5,2), datetime.date(2999,12,31))
    ex.processAssets(dict([(ModelID.ModelID(string='DSCB64C9U4'), rmgDict),
                           (ModelID.ModelID(string='DVYMHNV479'), rmgDict2),
                           (ModelID.ModelID(string='DHGYG3QD19'), rmgDict3)]))
    print('DSCB64C9U4',rmgDict)
    print('DVYMHNV479',rmgDict2)
    print('DHGYG3QD19',rmgDict3)
    

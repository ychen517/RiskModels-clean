

import configparser
import datetime
import logging
import optparse

from riskmodels import Connections, getModelByName
from riskmodels import MFM
from riskmodels import ModelID
from riskmodels import Utilities

MODEL_INDEX_MAP = {
                   'AXAP': [
                       'FTSE Asia Pacific',
                       'MSCI AC ASIA',
                       'MSCI AC ASIA PACIFIC',
                       'STOXX Asia/Pacific 600',
                   ],
                   'AXEU': [
                       'FTSE Europe',
                       'MSCI EUROPE',
                       'STOXX Europe 50',
                       'STOXX Europe 600',
                       'STOXX Europe Total Market',
                   ],
                   'AXEM': [
                       'FTSE Emerging',
                       'GS EM',
                       'MSCI EM (EMERGING MARKETS)',
                       'Russell Emerging',
                   ],
                   'AXUS': [
                       'RUSSELL 1000',
                       'RUSSELL 2000',
                       'RUSSELL 3000',
                       'RUSSELL MIDCAP',
                       'RUSSELL MICROCAP',
                       'S and P 400',
                       'S and P 500',
                       'S and P 1500',
                   ],
                   }

def writeResult(iidDateDict, outfile, dIdxMap, modelDB, marketDB, uc):
    for iid, missingDates in iidDateDict.items():
        missingDates = sorted(missingDates)
        for dIdx, date in enumerate(missingDates):
            if dIdx == 0:
                fromDt = date
                thruDt = None

            if len(missingDates) == 1:
                thruDt = date

            if thruDt is not None:
                name = modelDB.getIssueNames(date, [iid], marketDB).get(iid)
                if name is None:
                    name = ''
                imp = dict(modelDB.getIssueMapPairs(date))
                mktId = imp.get(iid)
                if mktId is not None:
                    mktIdStr = mktId.getIDString()
                elif uc.allmktImp.get(iid, None) is not None:
                    mktIdStr = uc.allmktImp[iid]
                else:
                    mktIdStr= ''
                outfile.write('%s,%s,%s,%s,%s\n'%(iid.getIDString(), mktIdStr, 
                                                  name, fromDt, thruDt))
            elif dIdx == 0:
                continue
            else:
                dPos = dIdxMap[date]
                prevDPos = dIdxMap[missingDates[dIdx - 1]]
                if dIdx == len(missingDates) - 1:
                    thruDt = date
                    name = modelDB.getIssueNames(date, [iid], marketDB).get(iid)
                    if name is None:
                        name = ''
                    imp = dict(modelDB.getIssueMapPairs(date))
                    mktId = imp.get(iid)
                    if mktId is not None:
                        mktIdStr = mktId.getIDString()
                    elif uc.allmktImp.get(iid, None) is not None:
                        mktIdStr = uc.allmktImp[iid]
                    else:
                        mktIdStr= ''
                    outfile.write('%s,%s,%s,%s,%s\n'%(iid.getIDString(), mktIdStr, 
                                                      name, fromDt, thruDt))
                    continue
                if dPos - prevDPos > 1:
                    thruDt = missingDates[dIdx - 1]
                    name = modelDB.getIssueNames(date, [iid], marketDB).get(iid)
                    if name is None:
                        name = ''
                    imp = dict(modelDB.getIssueMapPairs(date))
                    mktId = imp.get(iid)
                    if mktId is not None:
                        mktIdStr = mktId.getIDString()
                    else:
                        mktIdStr= ''
                    outfile.write('%s,%s,%s,%s,%s\n'%(iid.getIDString(), mktIdStr, 
                                                      name, fromDt, thruDt))
                    #Reset
                    fromDt = date
                    thruDt = None

    outfile.close()

def getallissuemktmap(modelDB):
    allimpdict = dict()
    allmktImpdict = dict()
    query = 'SELECT modeldb_id, marketdb_id FROM issue_map'
    modelDB.dbCursor.execute(query)
    allimp = [(ModelID.ModelID(string=i), j)
              for (i,j) in modelDB.dbCursor.fetchall()]
    allimpdict = dict([(j,i) for (i,j) in allimp])
    for (modelid, marketid) in allimp:
        if allmktImpdict.get(modelid, None) is None:
            allmktImpdict[modelid] = marketid
        else:
            allmktImpdict[modelid] += '/' + marketid
    return (allimpdict, allmktImpdict)

class UniverseChecker:
    def __init__(self, riskModel_, modelDB, marketDB, date, univ, allimp, allmktImp):
        self.riskModel_ = riskModel_
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.date = date
        self.riskModel_.setFactorsForDate(date,modelDB)
        self.allimp = allimp
        self.allmktImp = allmktImp

        query = """SELECT DISTINCT issue_id, from_dt, thru_dt
                   FROM sub_issue WHERE rmg_id in (%(rmg_arg)s)"""%{\
            'rmg_arg':','.join([str(k.rmg_id) for k in self.riskModel_.rmg])}
        modelDB.dbCursor.execute(query)
        iidInfo = [(ModelID.ModelID(string=k[0]), k[1].date(), k[2].date()) for k \
                       in modelDB.dbCursor.fetchall()]
        # marketDB.dbCursor.execute("""SELECT DISTINCT axioma_id FROM asset_dim_ctry_exch_int c
        #                              WHERE country in (:ctry_arg)""",
        #                           ctry_arg = ','.join('%s'%k.mnemonic for k in riskModel_.rmg))
        # tradingCtryAxiomaIDs = set([k[0] for k in marketDB.dbCursor.fetchall()])

        clsFamily = marketDB.getClassificationFamily('REGIONS')
        assert(clsFamily is not None)
        clsMembers = dict([(i.name, i) for i in marketDB.getClassificationFamilyMembers(clsFamily)])
        clsMember = clsMembers.get('HomeCountry')
        assert(clsMember is not None)
        self.clsRevision = marketDB.getClassificationMemberRevision(clsMember, datetime.date(2008, 1, 1))
        allAxiomaIDs = marketDB.getAllAxiomaIDs()

        self.univ = [sid.getModelID() for sid in univ]
        self.imp = dict([(j.getIDString(),i) for (i,j) in modelDB.getIssueMapPairs(date)])
        self.mktImp = dict(modelDB.getIssueMapPairs(date))
        hcData = marketDB.getAssetClassifications(self.clsRevision, allAxiomaIDs, date)
        hcIIDMap = dict()
        for k in list(hcData.keys()):
            if k.getIDString() in self.imp:
                hcIIDMap.setdefault(hcData[k].classification.code, list()).append(self.imp[k.getIDString()])

        #Now we have valid iid based on trading country and a hcIIDMap
        #So we use these two to create the valid model universe
        #For SCM, the valid model universe should base on the trading country
        #For regional models, universe should be the union of trading country and home country

        if len(self.riskModel_.rmg) == 1:
            finalIIDList = [j[0] for j in iidInfo if j[1] <= date and j[2] > date]
        else:
            finalIIDList = list()
            modelCtryMnemonic = set([r.mnemonic for r in self.riskModel_.rmg])
            for ctry in hcIIDMap:
                if ctry in modelCtryMnemonic:
                    #That home country belongs to the model, append
                    finalIIDList.extend(hcIIDMap[ctry])
            #Adding trading country assets
            finalIIDList.extend([j[0] for j in iidInfo if j[1] <= date and j[2] > date])
        #Remove duplicate
        self.finalIIDSet = set(finalIIDList)
        etf2 = self.getAllClassifiedAssets('Axioma Asset Type', 
                                           'ASSET TYPES', self.date,
                                           ['ComETF','NonEqETF'])
        etfSet = set()
        for axid, fromDt, thruDt in etf2:
            if axid in self.imp:
                etfSet.add(self.imp[axid])
                
        if not isinstance(self.riskModel_, MFM.StatisticalFactorModel) and \
                not isinstance(self.riskModel_, MFM.RegionalStatisticalFactorModel) and \
                not isinstance(self.riskModel_, MFM.StatisticalModel):
            #For fundamental model, we do not want to check ETFs
            self.finalIIDSet = self.finalIIDSet - etfSet
                
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
        return axids

    def getMissingComposite(self):
        self.marketDB.dbCursor.execute("""SELECT DISTINCT cce.axioma_id 
                                         FROM composite_constituent_easy cce, composite_member cm,
                                         COMPOSITE_MEMBER_FAMILY_MAP cmfm,
                                         modeldb_global.composite_family_model_map cfmm
                                         WHERE model_id = :model_arg AND dt = :date_arg
                                         AND cmfm.family_id = cfmm.composite_family_id
                                         AND cmfm.member_id = cm.id
                                         AND cce.etf_axioma_id = cm.axioma_id
                                         AND cce.axioma_id not like 'CSH_%'""", 
                                       model_arg = self.riskModel_.rm_id,
                                       date_arg = self.date)
        etfList = list()
        for aidStr in self.marketDB.dbCursor.fetchall():
            if aidStr[0] in self.imp:
                etfList.append(self.imp[aidStr[0]])

        diff = set(etfList).intersection(self.finalIIDSet) - set(self.univ)
        if len(diff) > 0:
            logging.error('Found %s missing from composite constituent'%len(diff))
        return diff

    def getMissingBenchmark(self,showAllTerminated=False,IncluTerminated=False):
        self.marketDB.dbCursor.execute("""SELECT distinct axioma_id FROM index_constituent ice
                                          WHERE revision_id in (SELECT id FROM index_revision_active ice
                                          WHERE dt = :dt_arg)""", dt_arg = self.date)

        bmkList = list()
        terminated = list()
        for aidStr in self.marketDB.dbCursor.fetchall():
            if aidStr[0] in self.imp:
                bmkList.append(self.imp[aidStr[0]])
            else: 
                terminated.append(aidStr[0])
            ####
            #Not to display any index constituents with invalid lifespan
            #the premise is that we are fully aware when we move the asset's from/thru date
            ####

            # elif aidStr[0] in tradingCtryAxiomaIDs:
            #     #Not exist in the issueMap. add it to iidDateDict
            #     logging.error('Missing index constituents, MakretID: %s'%(aidStr[0]))
            #     bmkDateDict.setdefault(ModelID.ModelID(string=aidStr[0]), list()).append(date)
       
        ctryBmkList = set(bmkList).intersection(self.finalIIDSet)
        if self.riskModel_.mnemonic[4:5] != 'x':
            modelBmk = MODEL_INDEX_MAP.get(self.riskModel_.mnemonic[:4])
        else:
            modelBmk = MODEL_INDEX_MAP.get(self.riskModel_.mnemonic[:7])
        if modelBmk:
            self.marketDB.dbCursor.execute("""SELECT distinct axioma_id FROM index_constituent_easy
                                              WHERE index_id in (SELECT id FROM index_member WHERE
                                              NAME IN (%s)) 
                                  AND dt = :dt_arg"""%','.join("'%s'"%b for b in modelBmk),
                                                                                  dt_arg =self.date)

            bmk = set(self.marketDB.dbCursor.fetchall())
            if not IncluTerminated:
                for aidStr in bmk:
                    if aidStr[0] in self.imp:
                        ctryBmkList.add(self.imp[aidStr[0]])
            else:
                for aidStr in bmk:
                    if aidStr[0] in self.imp:
                        ctryBmkList.add(self.imp[aidStr[0]])
                    else:
                        ctryBmkList.add(self.allimp[aidStr[0]])

        logging.debug('Found %s index_constituents belong to model rmgs: %s'%(\
                len(ctryBmkList), ','.join('%s'%k.mnemonic for k in self.riskModel_.rmg)))
        diff = ctryBmkList - set(self.univ)
        if len(diff) > 0:
            logging.error('Found %s missing index_consitutents'%(len(diff)))
            logging.error('Mid: %s'%(','.join('%s'%k.getIDString()\
                                                  for k in diff)))
        if not showAllTerminated: 
            return diff
        else: 
            return (diff,terminated)

    def getMissingDataAssets(self):
        self.modelDB.dbCursor.execute("""SELECT DISTINCT issue_id FROM rms_issue ri
                                         WHERE rms_id = :rms_arg AND from_dt <= :date_arg
                                         AND thru_Dt > :date_arg""", rms_arg = self.riskModel_.rms_id,
                                      date_arg = self.date)
        rmsUniv = [ModelID.ModelID(string=k[0]) for k in self.modelDB.dbCursor.fetchall()]
        diff = set(rmsUniv) - set(self.univ)
        if len(diff) > 0:
            logging.error('Found %s missing from rms_issue universe'%len(diff))
        return diff

    def getMissingFromOtherModels(self, modelList, date):
        result = set()
        for otherRiskModel_ in modelList:
            otherRmi = self.modelDB.getRiskModelInstance(otherRiskModel_.rms_id, date)
            if otherRmi is None:
                logging.error('Cannot not find universe for %s on %s. Skip.'%(\
                        otherRiskModel_.mnemonic, date))
                continue
            otherUniv = self.modelDB.getRiskModelInstanceUniverse(otherRmi)
            if len(otherUniv) > 0:
                #Ignore those exist in old models but doesn't not exist in sub_issue
                # universe. Probably because of content fix
                otherUniv = [sid.getModelID() for sid in otherUniv]
                diff = set(set(otherUniv) - set(self.univ)).intersection(self.finalIIDSet)
                if len(diff) > 0:
                    logging.error('Found %s missing from %s universe'%(\
                            len(diff), otherRiskModel_.mnemonic))
                    result = result.union(diff)
        return result

if  __name__ == '__main__':
    usage = "usage: %prog [options] config-file startDate <endDate>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--mock", "-o",action="store_true",
                             default=False,dest="mock",
                             help="generate the model universe on-the-fly")
    cmdlineParser.add_option("--modelList, -l",action="store",
                             default='',dest="modelList",
                             help="model names you want to compare against")
    cmdlineParser.add_option("--IncluTerminatedBmk",action="store_true",
                             default=False,dest="IncluTerminatedBmk",
                             help="include inactive index constituents in bmk check")

    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processModelAndDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)
    marketDB = connections.marketDB
    modelDB = connections.modelDB

    # prodModelDB = ModelDB.ModelDB(user='modeldb_global',passwd='modeldb_global',sid='glprod')
    # prodMarketDB = MarketDB.MarketDB(user='marketdb_global',passwd='marketdb_global',sid='glprod')

    startDate = Utilities.parseISODate(args_[1])
    if len(args_) < 3:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args_[2])
    
    riskModelClass = Utilities.processModelAndDefaultCommandLine(options_, cmdlineParser)
    riskModel_ = riskModelClass(connections.modelDB, connections.marketDB)
    if options_.modelList != '':
        rmList = [getModelByName(modelName) \
                         for modelName in options_.modelList.split(',')]
        modelList = [rm(modelDB, marketDB) for rm in rmList]
    else:
        modelList = []

    dates = modelDB.getDateRange(riskModel_.rmg, startDate, endDate, True)
    
    # dates = [prev for (prev, next) in zip(dates[:-1], dates[1:]) \
    #              if (next.month > prev.month or next.year > prev.year)]
    iidDateDict = dict()
    bmkDateDict = dict()
    modelDateDict = dict()

    (allimp, allmktImp) = getallissuemktmap(modelDB)
    for dIdx, date in enumerate(dates):
        logging.info('Processing %s for %s'%(date, riskModel_.mnemonic))
        result = list()
        if options_.mock:
            riskModel_.setFactorsForDate(date, modelDB)
            data = riskModel_.generate_model_universe(date, modelDB, marketDB)
            univ = data.universe
        else:
            rmi = modelDB.getRiskModelInstance(riskModel_.rms_id, date)
            if rmi is None:
                logging.error('Skipping %s because risk model %s has not been generated'%(date, riskModel_.mnemonic))
                continue

            data = Utilities.Struct()
            univ = modelDB.getRiskModelInstanceUniverse(rmi)

        logging.debug('%s universe contains %s assets'%(riskModel_.mnemonic, len(univ)))

        uc = UniverseChecker(riskModel_, modelDB, marketDB, date, univ, allimp, allmktImp)
        #First check benchmark
        if options_.IncluTerminatedBmk:
            diff = uc.getMissingBenchmark(IncluTerminated = True)
        else:
            diff = uc.getMissingBenchmark()
        for iid in diff:
            val = Utilities.Struct()
            if iid in uc.mktImp:
                val.axiomaid = uc.mktImp[iid].getIDString()
            else:
                val.axiomaid = uc.allmktImp[iid]
            val.date = date
            val.model = riskModel_.mnemonic
            val.sedol = ''
            val.cusip = ''
            val.isin = ''
            val.ticker = ''
            val.country = ''

            rmsIssue = 'modeldb_global.rms_issue'
            # idVal, reason = checkCoverage.checkCoverage(marketDB, modelDB, val, rmsIssue)
            # print reason

            iidDateDict.setdefault(iid, list()).append(date)
            bmkDateDict.setdefault(iid, list()).append(date)

        #Check composite constituent as well
        diff = uc.getMissingComposite()
        for iid in diff:
            iidDateDict.setdefault(iid, list()).append(date)
            bmkDateDict.setdefault(iid, list()).append(date)

        #Check rms_issue
        diff = uc.getMissingDataAssets()
        for iid in diff:
            iidDateDict.setdefault(iid, list()).append(date)

        #Check against other models
        diff = uc.getMissingFromOtherModels(modelList, date)
        for iid in diff:
            modelDateDict.setdefault(iid, set()).add(date)

    dIdxMap = dict([(j,i) for (i,j) in enumerate(dates)])
    outfile = open('checkUniverse_rmsIssue_%s_%04d-%04d.csv'%(riskModel_.mnemonic, dates[0].year,
                                                         dates[-1].year), 'w')
    writeResult(iidDateDict, outfile, dIdxMap, modelDB, marketDB, uc)

    outfile2 = open('checkUniverse_benchmark_%s_%04d-%04d.csv'%(riskModel_.mnemonic, dates[0].year,
                                                         dates[-1].year), 'w')
    writeResult(bmkDateDict, outfile2, dIdxMap, modelDB, marketDB, uc)

    outfile3 = open('checkUniverse_otherModels_%s_%04d-%04d.csv'%(riskModel_.mnemonic, dates[0].year,
                                                         dates[-1].year), 'w')
    writeResult(modelDateDict, outfile3, dIdxMap, modelDB, marketDB, uc)
    logging.info('Finished universe check for %s '%riskModel_.mnemonic)
    exit(1)

import datetime
import optparse
import configparser
import logging
import sys
try:
    import numpy.ma as ma
    HAS_OLD_MA = False
except:
    import numpy.core.ma as ma
    HAS_OLD_MA = True
import numpy

import riskmodels
from riskmodels import Connections
from riskmodels import Utilities
from marketdb import MarketDB
from riskmodels.ModelDB import SubIssue
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels.Matrices import ExposureMatrix, TimeSeriesMatrix
from riskmodels import Matrices

class mapEIFTool:
    def __init__(self, connections):
        self.marketDB = connections.marketDB
        self.modelDB = connections.modelDB
        self.trackingErrorThreshold = 0.005 
        #Previous one month median
        self.medianLength = 30
        self.timeBuffer = 10
        self.ETFRollover = 5 
        self.rmi = None
        
    def getAllIndexes(self):
        index_ids = list()
        query = """SELECT DISTINCT im.id, im.name , im.short_names FROM  \
                   FUTURE_FAMILY_ATTRIBUTE_ACTIVE a, index_member im
                   WHERE index_member_id IS NOT NULL and a.index_member_id = im.id"""
            
        self.marketDB.dbCursor.execute(query)
        for index_id, index_name, index_short_name in self.marketDB.dbCursor.fetchall():
            indexval = Utilities.Struct()
            indexval.index_id = index_id
            indexval.index_name = index_name
            indexval.index_short_name = index_short_name
            index_ids.append(indexval)
        return index_ids    

    def getIndex2ETFmapping(self, indexVals):
        index2ETF = dict()
        distinctETFs = set()
        indexInfoDict = dict([(i.index_id, i ) for i in indexVals])
        idArgList = [('id%d'%i) for i in range(len([i.index_id for i in indexVals]))]
        query = """select a.*, c.short_names  from (SELECT DISTINCT a.index_id, a.etf_axioma_id,
                   min(b.dt) as min_dt, max(b.dt)  as max_dt 
                   FROM composite_index_map a, composite_constituent_easy b 
                   WHERE a.etf_axioma_id= b.etf_axioma_id and a.index_id  IN(%(ids)s) group by  
                   a.index_id, a.etf_axioma_id) a left outer join composite_member c   on c.thru_dt>=a.max_dt and c.axioma_id = a.etf_axioma_id  """\
            %{'ids':','.join([(':%s'%i ) for i in idArgList])}
        valueDict = dict(zip(idArgList,[i.index_id for i in indexVals]))
        self.marketDB.dbCursor.execute(query, valueDict)
        result = []
    
        for index_id, etf_axioma_id, from_dt, thru_dt, latest_etf_short_name in self.marketDB.dbCursor.fetchall():
            val = Utilities.Struct()
            val.etf_axioma_id = etf_axioma_id
            val.etf_name = latest_etf_short_name
            val.from_dt = from_dt.date()
            val.thru_dt = thru_dt.date()+ datetime.timedelta(self.ETFRollover)
            index_val = indexInfoDict[int(index_id)]
            index2ETF.setdefault(index_val, list()).append(val)
            distinctETFs.add(etf_axioma_id)
        return index2ETF, list(distinctETFs)

    def getCompositeSubIssue(self, etfAids, rmgs):
        sidRmgDict= dict()
        rmgDict = dict([(rmg.rmg_id, rmg) for rmg in rmgs])
        self.modelDB.dbCursor.execute("""SELECT distinct axioma_id, SUB_ID, RMG_ID 
                                          FROM marketdb_global.composite_member a, issue_map b, sub_issue c 
                                          WHERE a.axioma_id in (%s) AND a.axioma_id = b.marketdb_id 
                                          AND b.modeldb_id = c.issue_id """ % ','.join(["'%s'" %k for k in etfAids]))
        for (etf_axioma_id, sid, rmg_id) in self.modelDB.dbCursor.fetchall():
            sidRmgDict[etf_axioma_id]=(SubIssue(string=sid), rmgDict[rmg_id])
        return sidRmgDict

    def getDates(self, inputDates):
        if ':' in inputDates:
            (minDt, maxDt) = inputDates.split(':')
        else:
            minDt = inputDates
            maxDt = inputDates
        #Look back extra ten days to make sure we have sufficient history everytime
        startDate = Utilities.parseISODate(minDt) - datetime.timedelta(self.medianLength) - \
        datetime.timedelta(self.timeBuffer)
        if maxDt == 'now':
            endDate = datetime.date.today()
        else:
            endDate = Utilities.parseISODate(maxDt)
        d = startDate
        dates = []
        dayInc = datetime.timedelta(1)
        while d<=endDate:
            if d.isoweekday() in [6,7]:
                d += dayInc
                continue
            elif d.month == 1 and d.day == 1:
                d += dayInc
                #every new year day is a holiday for Axioma
                continue
            else:
                dates.append(d)
                d += dayInc
                
        trueStartDate = [d for d in dates if d >= Utilities.parseISODate(minDt)][0]
        return sorted(dates), trueStartDate

    def setRiskModelMat(self, rm, rmi):
        #Exposure Matrix
        self.expM = rm.loadExposureMatrix(rmi, self.modelDB)
        logging.info('Loaded exposure Matrix for %s'%rmi.date.isoformat())

        #Factor Covariance
        (cov, factors) = rm.loadFactorCovarianceMatrix(rmi, self.modelDB)
        self.factorCov = cov
        logging.info('Loaded covariance Matrix for %s'%rmi.date.isoformat())
        #Specific risks
        (self.srDict, self.scDict) = rm.loadSpecificRisks(rmi, self.modelDB)
        logging.info('Loaded specific risk Matrix for %s'%rmi.date.isoformat())
        self.mdl2sub = dict([(n.getModelID(),n) for n in self.expM.getAssets()])

    def processPort(self, port):
        iids, wgts = zip(*port)
        assets = []
        weights = []
        for (a,w) in port:
             if a in self.mdl2sub:
                assets.append(self.mdl2sub[a])
                weights.append(w)
       # assets, weights = zip(*[(self.mdl2sub[a], w) for (a,w) in port if a in self.mdl2sub])
        newPort = zip(assets, weights)
        return newPort

    def compute_total_risk_portfolio(self, portfolio, expMatrix, factorCov, 
                                     srDict, scDict=None, factorTypes=None):
        """
        Copied from Utilities.py
        """

        expM_Map = dict([(expMatrix.getAssets()[i], i) \
                             for i in range(len(expMatrix.getAssets()))])
        # Discard any portfolio assets not covered in model
        indices = [i for (i,j) in enumerate(portfolio) \
                       if j[0] in expM_Map and j[0] in srDict]
        port = [portfolio[i] for i in indices]
        (assets, weights) = zip(*port)
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        weightMatrix = numpy.zeros((len(assets), 1))
        # Populate matrix of portfolio weights
        for (a,w) in port:
            pos = assetIdxMap.get(a, None)
            if pos is None:
                continue
            weightMatrix[pos, 0] = w

        expM = expMatrix.getMatrix().filled(0.0)
        expM = ma.take(expM, [expM_Map[a] for (a,w) in port], axis=1)
        if factorTypes is not None:
            f_indices = []
            for fType in factorTypes:
                fIdx = expMatrix.getFactorIndices(fType)
                f_indices.extend(fIdx)
            expM = ma.take(expM, f_indices, axis=0)
            factorCov = numpy.take(factorCov, f_indices, axis=0)
            factorCov = numpy.take(factorCov, f_indices, axis=1)

        # Compute total risk
        assetExp = numpy.dot(expM, weights)
        totalVar = numpy.dot(assetExp, numpy.dot(factorCov, assetExp))
        if factorTypes is None:
            totalVar += numpy.sum([(w * srDict[a])**2 for (a,w) in port])
        if scDict is None:
            return totalVar**0.5

        # Incorporate linked specific risk
        assetSet = set(assets)
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assets)])
        for sid0 in assets:
            if sid0 in scDict:
                for (sid1, cov) in scDict[sid0].items():
                    if sid1 not in assetSet:
                        continue
                    weight0 = weightMatrix[assetIdxMap[sid0], 0] 
                    weight1 = weightMatrix[assetIdxMap[sid1], 0] 
                    totalVar += 2.0 * weight0 * weight1 * cov
        return totalVar**0.5

    def insertQuery( self, from_dt , thru_dt, future_family_id, composite_id):
        query = """insert into TMP_EIF_TO_BEST_COMPOSITE_MAP
        (future_family_id, composite_id, ref, from_dt, thru_dt)
        values(:future_family_id,:composite_id, :ref
        ,:from_dt, :thru_dt) """
        valueDict = {'future_family_id': future_family_id,
                     'composite_id':composite_id,
                     'ref':'mapETF by src_id:908',
                     'from_dt': from_dt,
                     'thru_dt': thru_dt }
        self.marketDB.dbCursor.execute(query, valueDict)
        self.marketDB.commitChanges()
    def deleteQuery(self, from_dt, thru_dt, future_family_id, composite_id):
        query = """delete from TMP_EIF_TO_BEST_COMPOSITE_MAP where future_family_id = :future_family_id and composite_id = :composite_id and from_dt = :from_dt and thru_dt = :thru_dt"""
        valueDict = {'future_family_id': future_family_id,
                     'composite_id':composite_id,
                     'from_dt': from_dt,
                     'thru_dt': thru_dt }
        self.marketDB.dbCursor.execute(query, valueDict)
   
    def updateQuery(self, from_dt, thru_dt, future_family_id, composite_id, new_thru_dt):
        query = """update TMP_EIF_TO_BEST_COMPOSITE_MAP set thru_dt = :new_thru_dt \
        where future_family_id = :future_family_id and composite_id =:composite_id \
        and from_dt =:from_dt and thru_dt=:thru_dt"""
        valueDict = {'future_family_id': future_family_id,
                     'composite_id':composite_id,
                     'from_dt': from_dt,
                     'thru_dt': thru_dt,
                     'new_thru_dt': new_thru_dt}
        self.marketDB.dbCursor.execute(query, valueDict)

    def processResults(self, results, resultFile):
        wfile = open('tmp/%s.csv'%(resultFile),'w')
        wfile.write('Index_Name|Index_ID|Composite_aid|From_dt|Thru_dt')
        wfile.write('\n')
  
        for index in sorted(results.keys()):
            bestETFDict = results[index]
            proxyETF = list()
            for dIdx, d in enumerate(sorted(bestETFDict.keys())):
                etf, te, ref = bestETFDict.get(d)
                if len(proxyETF) == 0:  
                    proxyETF.append((index.index_name, index.index_id,etf.etf_axioma_id\
                                         ,d,d+ datetime.timedelta(1), dIdx ))
                elif len(proxyETF) > 0 :
                    iname, iid,p_etf_axioma_id, p_from_dt, p_thru_dt, p_dIdx = proxyETF[-1]
                    if dIdx-p_dIdx <= self.ETFRollover:
                        if p_etf_axioma_id == etf.etf_axioma_id :
                            proxyETF[-1] = iname, iid ,p_etf_axioma_id,p_from_dt,\
                                d+ datetime.timedelta(1),dIdx
                        else:    
                            proxyETF[-1] = iname, iid ,p_etf_axioma_id,p_from_dt,\
                                d,dIdx
                            proxyETF.append((iname, iid, etf.etf_axioma_id \
                                             , d, d+ datetime.timedelta(1),dIdx))

                    else:    
                        proxyETF.append((iname, iid, etf.etf_axioma_id \
                                             , d, d+ datetime.timedelta(1),dIdx))
            for (index_name, index_id ,etf_aid, from_dt, thru_dt, dIdx) in proxyETF:
                wfile.write('%s|%s|%s|%s|%s'%(index_name, index_id,etf_aid,from_dt, thru_dt))
                wfile.write('\n')
        wfile.close()
        return proxyETF

    def plotTe(self, teMat, mcapMat, bestETFDict, indexval, date):
        ofile = open('tmp/TePlot_%s_%04d%02d%02d.csv'%(indexval.index_short_name.replace('/', '_'),
                                                       date.year, date.month, date.day),'w')
        ofile.write('date|')
        ofile.write('|'.join('%s_%s|Mcap'%(k.etf_axioma_id, k.etf_name) for k in teMat.assets))
        ofile.write('|bestETF_aid|ref')
        ofile.write('\n')

        for dIdx, d  in enumerate(sorted(teMat.dates)):
            ofile.write('%s'%d)
            for etfIdx, etf in enumerate(teMat.assets):
                te = teMat.data[etfIdx, dIdx]
                mcap = mcapMat.data[etfIdx, dIdx]
                ofile.write('|%s|%s'%(te, mcap))
            bestETF = bestETFDict.get(d,[Utilities.Struct(),'',''])
            try:
                etf_aid = bestETF[0].etf_axioma_id
                ref = bestETF[2]
            except:
                etfName = ''
                etf_aid = ''
                ref = ''
            ofile.write('|%s|%s\n'%(etf_aid, ref))
        ofile.close()

    def getCompositeConstituents(self, compositeName, date, marketDB, 
                                 rollBack=0, marketMap=None, compositeAxiomaID=None):
        """Returns the assets and their weight in the composite for the
        given date. If there is no data for the specified date, the most
        recent assets up to rollBack days ago will be returned.
        The return value is a pair of date and list of (ModelID, weight)
        pairs.
        The returned date is the date from which the constituents were
        taken which will be different from the requested date if
        rollBack > 0 and there is no active revision for the requested date.
        marketMap: an optional dictionary mapping Axioma ID strings (marketdb)
        to their corresponding ModelID object.
        """
        self.log.debug('getCompositeConstituents: begin')
        # if axioma_id is supplied look it up by that
        if compositeAxiomaID:
            composite = marketDB.getETFbyAxiomaID(compositeAxiomaID.getIDString(), date)
        else:
            composite = marketDB.getETFByName(compositeName, date)
        if composite is None:
            return (None, list())
        compositeRevInfo = marketDB.getETFRevision(composite, date, rollBack)
        if compositeRevInfo is None:
            return (None, list())
        assetWeightsMap = marketDB.getETFConstituents(compositeRevInfo[0])
        if marketMap is None:
            issueMapPairs = self.getIssueMapPairs(date)
            marketMap = dict([(i[1].getIDString(), i[0]) for i in issueMapPairs])
        constituents = [(marketMap[i], assetWeightsMap[i]) for i
                        in list(assetWeightsMap.keys()) if i in marketMap]
        notMatched = [i for i in assetWeightsMap.keys()
                      if i not in marketMap]
        if len(notMatched) > 0:
            self.log.info("%d unmapped assets in composite", len(notMatched))
            self.log.debug("Can't match assets in %s to ModelDB on %s: %s",
                           composite.name, date, ','.join(notMatched))
        if len(constituents) > 0:
            self.log.debug('%d assets in %s composite', len(constituents),
                          composite.name)
        else:
            self.log.warning('no assets in %s composite', composite.name)
        self.log.debug('getCompositeConstituents: end')
        return (compositeRevInfo[1], constituents)
      
    def verifyAgainstMarketDB(self, result,dateList):
        ofile = open('ETFReport_%s.csv'%(dateList[-1].isoformat(),'w'))
        reportString=''
        reportString+='lastday|index|deducedETF_aid|deducedETF_name\
                       |storedETF_aid|storedETF_name\
                       |champion_tracking_error\
                       |ref'
        reportString+='\n'
        lastDay = dateList[-1]
        dbRecordDict=dict()
        query = """select a.index_member_id, a.etf_axioma_id, b.short_names
                   from index_best_composite_map a, composite_member b 
                   where a.from_dt <:dt_arg and a.etf_axioma_id = b.axioma_id
                   and a.thru_dt > :dt_arg"""
        self.marketDB.dbCursor.execute(query, dt_arg=lastDay)
        r = self.marketDB.dbCursor.fetchall()
        for index_id, etf_axioma_id, etf_short_name in r :
            dbRecordDict[index_id] = etf_axioma_id, etf_short_name
        for index in result.keys():
            bestETFDict = result[index]
            if lastDay not in bestETFDict:
                continue
            else:
                deductedETF, champTe, ref  = bestETFDict[lastDay]
            storedETF_aid, storedETF_name = dbRecordDict.get(index.index_id, (None, None))
            if storedETF_aid != deductedETF.etf_axioma_id:
                reportString+='%s|%s|%s|%s|%s|%s|%s|%s'%(lastDay.isoformat(),index.index_name,\
                                                    deductedETF.etf_axioma_id, \
                                                    deductedETF.etf_name,\
                                                    storedETF_aid, \
                                                    storedETF_name,
                                                    champTe,
                                                    ref)
                reportString+='\n'
                
        ofile.write(reportString)
        ofile.close()


    def computeTrackingError(self, d, etf, port, compositeSidDict):
        etfPort = self.modelDB.getCompositeConstituents('dummy', d, self.marketDB, rollBack=7,\
                                                            compositeAxiomaID=ModelID.ModelID(string=etf.etf_axioma_id))[1]
        if len(etfPort) == 0:
            logging.info('Missing etf constituent data %s on %s'%(etf.etf_name, d.isoformat()))
            return ma.masked
        else:
            etfNewPort = self.processPort(etfPort)
            logging.info('%s assets in %s'%(len(etfNewPort), etf.etf_name))
            longShortPort = []
            etfPortDict = dict(etfNewPort)
            bmkPortDict = dict(port)
            etfAssetsSet = set(etfPortDict.keys())
            bmkAssetsSet = set(bmkPortDict.keys()) 
            diffAssets = etfAssetsSet^bmkAssetsSet
            commonAssets = etfAssetsSet.intersection(bmkAssetsSet)

            for asset in diffAssets:
                if asset in etfPortDict.keys():
                    weight = etfPortDict[asset]
                else:
                    weight = -1 * bmkPortDict[asset]
                longShortPort.append((asset, weight))    
            for asset in commonAssets:
                weight = etfPortDict[asset] - bmkPortDict[asset]
                longShortPort.append((asset, weight))

            sid, rmg_id = compositeSidDict[etf.etf_axioma_id]
            te = mapTool.compute_total_risk_portfolio(longShortPort, self.expM,
                                                      self.factorCov, self.srDict,
                                                      self.scDict)
            return te
        
    def processMatrices(self, matDict, startDate):
        #eIdxMat indicates the available ETF position. Those not there are dead
        result = dict()
        for index, (teMat, mcapMat) in matDict.items():
            bestETF = dict()
            assert(teMat.dates == mcapMat.dates)

            dIdxMap = dict([(j, i) for (i,j) in enumerate(teMat.dates)])
            startIdx = dIdxMap[startDate]
            
            for dIdx, d  in enumerate(sorted(teMat.dates)):
                if dIdx < startIdx:
                    #These are dates for calculation only. Skip.
                    continue
                logging.debug('Process matrices...%04d%02d%02d, index: %s'%
                              (d.year, d.month, d.day, index.index_short_name))

                lookBack = dIdx - self.medianLength
                if lookBack < 0:
                    lookBack = 0 
                inactiveIdx = [i for (i,j) in enumerate(teMat.assets) \
                                   if j.thru_dt <= d or j.from_dt> d]
                if len(inactiveIdx) == len(teMat.assets):
                ## all mapped ETFs are either dead or not yet started trading##
                    logging.debug('All mapped ETFs are either dead or not yet started')
                    continue
                targetTe = teMat.data[:, lookBack: dIdx + 1]
                medianTe = ma.masked_all((teMat.data.shape[0], 1), float)

                for j in range(teMat.data.shape[0]):
                    if j in inactiveIdx:
                    ##filter all the dead or not yet started trading ETFs##    
                        continue
                    #Therefore, take out all the non 0.0 value and calculate Median
                    maskedFlg = numpy.flatnonzero(ma.getmaskarray(targetTe[j, :]))
                    if len(maskedFlg) == targetTe.shape[1]:
                        #All dates are missing Te, either no Index or ETF hasn't started trading
                        logging.debug('No te is found for the date range, either no index or ETF has\
                                       not started trading')

                        continue
                    else:
                        nonZeroFlg = [k for k in range(targetTe.shape[1]) if k not in maskedFlg]
                        median = numpy.median(ma.take(ma.getdata(targetTe[j, :]), nonZeroFlg), axis = 0)
                        numpy.put(medianTe, j, median)
                #filter out those do not pass threshold
                missingTe = numpy.flatnonzero(ma.getmaskarray(teMat.data[:, dIdx]))
                bad = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                            medianTe > self.trackingErrorThreshold, medianTe)))
                bad = list(set(missingTe).union(set(bad)))
                missingMcap = numpy.flatnonzero(ma.getmaskarray(mcapMat.data[:, dIdx])) 
                if len(bad) == len(teMat.assets):
                    #Check mcap first
                    noCapCount = numpy.flatnonzero(ma.getmaskarray(mcapMat.data[:, dIdx]))
                    if len(noCapCount) == len(teMat.assets):
                        missingTe = numpy.flatnonzero(ma.getmaskarray(ma.masked_where(
                                    medianTe == 999, medianTe)))
                        #no cap, no te. skip for that date
                        if len(missingTe) == len(teMat.assets):
                            if len(teMat.assets) == 1 :
                                rank = [0]
                                logging.info(' No cap, No Te for indexx_id: %s, date %s'%(index.index_short_name, d.isoformat()))
                                ref = 'case 1 - No cap, No Te found .Pick the sole avilable %.4f Mcap: %.4f'
                            else:
                                fromDates = [ i.from_dt for i in teMat.assets]
                                fromDatesOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(fromDates))])
                                fromDatesDataMap = dict([(i, fromDatesOrderMap[j]) for (i,j) in enumerate(fromDates)])
                                rank = [fromDatesDataMap[i] for (i,j) in enumerate(fromDates)]
                                logging.info(' No cap, No Te for indexx_id: %s, date %s'%(index.index_short_name, d.isoformat()))
                                ref = 'case 8- No cap, no te. with more than 1 live ETFs, pick the one with earliest from_dt. %.4f Mcap: %.4f'
                                    

                        else:        
                        #all cap absent, pick the smallest trcking error, even thought not qualified
                            teOnly = [j for j in range(len(teMat.assets)) if j not in missingTe]
                            teTrueData = ma.take(medianTe.data[:,0]+1, teOnly, axis = 0)
                            teTrueAsset =  ma.take(teMat.assets, teOnly, axis= 0)
                            teTrueOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(teTrueData))])
                            teTrueDataMap = dict([(i, teTrueOrderMap[j]) for (i,j) in enumerate(teTrueData)])
                            teTrueRank = [teTrueDataMap[i] for (i,j) in enumerate(teTrueData)]
                            teIdxMap = dict([(j,i) for (i,j) in enumerate(teMat.assets)])
                            tePrelimChampETF = [teTrueAsset[k] for (k, j) \
                                                       in enumerate(teTrueRank) if j == 0]
                            assert(len(tePrelimChampETF)> 0)
                            winner =  tePrelimChampETF[0]
                            rank = [1 for k in range(len(teMat.assets))]
                            rank[teIdxMap[winner]] = 0
                            ref =  'case 2 - No cap. No qualified Te. ' + \
                                'Pick smallest Te available: %.4f Mcap:  %.4f'
                    else:
                        #No qualified Te, only look at mcap and  pick largest cap available
                        McapOnly = [j for j in range(len(mcapMat.assets)) if j not in missingMcap]
                        McapTrueData = ma.take(mcapMat.data[:, dIdx], McapOnly, axis = 0)
                        McapTrueAsset =  ma.take(mcapMat.assets, McapOnly, axis= 0)
                        McapTrueRank = numpy.argsort(-ma.getdata(McapTrueData))
                        McapIdxMap = dict([(j,i) for (i,j) in enumerate(mcapMat.assets)])
                        McapPrelimChampETF = [McapTrueAsset[k] for (k, j) in \
                                                  enumerate(McapTrueRank) if j == 0]
                        assert(len(McapPrelimChampETF)> 0)
                        winner =  McapPrelimChampETF[0]
                        rank = [1 for k in range(len(mcapMat.assets))]
                        rank[McapIdxMap[winner]] = 0
                        ref = 'case 3 - Cap available. No qualified Te. ' + \
                            'Pick biggest cap available: %.4f Mcap:  %.4f'
                else:
                    #There are more than 0 qualified. Pick the largest cap 

                    if len(missingMcap) == len(teMat.assets):
                        #all assets are missing Mcap. Pick the lowest TE
                        teOnly = [j for j in range(len(teMat.assets)) if j not in missingTe]
                        teTrueData = ma.take(medianTe.data[:,0]+1, teOnly, axis = 0)
                        teTrueAsset =  ma.take(teMat.assets, teOnly, axis= 0)
                        teTrueOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(teTrueData))])
                        teTrueDataMap = dict([(i, teTrueOrderMap[j]) for (i,j) in enumerate(teTrueData)])
                        teTrueRank = [teTrueDataMap[i] for (i,j) in enumerate(teTrueData)]
                        teIdxMap = dict([(j,i) for (i,j) in enumerate(teMat.assets)])
                        tePrelimChampETF = [teTrueAsset[k] for (k, j) \
                                                   in enumerate(teTrueRank) if j == 0]
                        assert(len(tePrelimChampETF)> 0)
                        winner =  tePrelimChampETF[0]
                        rank = [1 for k in range(len(teMat.assets))]
                        rank[teIdxMap[winner]] = 0
                        ref = 'case 4 - No cap. %s qualified. ' %(len(teMat.assets) - len(bad)) + \
                            'Pick smallest Tracking error: %.4f Mcap:  %.4f'
                    else:
                        noMcapAndNoTe = list(set(missingMcap).union(missingTe))
                        McapAndTe = [j for j in range(len(teMat.assets)) if j not in noMcapAndNoTe]
                        if len(McapAndTe) > 0:
                            #chopping off the missing Te or missing Cap
                            trueData = ma.take(mcapMat.data[:, dIdx], McapAndTe, axis = 0)
                            trueAsset = ma.take(teMat.assets, McapAndTe, axis= 0)
                            trueRank = numpy.argsort(-ma.getdata(trueData))
                            #Now we know which asset is the champ, put a zero there
                            idxMap = dict([(j,i) for (i,j) in enumerate(teMat.assets)])
                            prelimChampETF = [trueAsset[k] for (k, j) in enumerate(trueRank) if j == 0]
                            assert(len(prelimChampETF) > 0)
                            theone = prelimChampETF[0]
                            rank = [1 for k in range(len(teMat.assets))]
                            rank[idxMap[theone]] = 0
                            ref = 'case 5 - Cap And Te available. %s qualified. '%(len(McapAndTe)) + \
                                'Pick biggest cap thats qualified: %.4f Mcap:  %.4f'
                        else:
                            if len(missingTe) < len(teMat.assets):
                                #Got Te but all missing Cap
                                teOnly = [j for j in range(len(teMat.assets)) if j not in missingTe]
                                teTrueData = ma.take(medianTe.data[:,0]+1, teOnly, axis = 0)
                                teTrueAsset =  ma.take(teMat.assets, teOnly, axis= 0)
                                teTrueOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(teTrueData))])
                                teTrueDataMap = dict([(i, teTrueOrderMap[j]) for (i,j) in enumerate(teTrueData)])
                                teTrueRank = [teTrueDataMap[i] for (i,j) in enumerate(teTrueData)]
                                teIdxMap = dict([(j,i) for (i,j) in enumerate(teMat.assets)])
                                tePrelimChampETF = [teTrueAsset[k] for (k, j) \
                                                   in enumerate(teTrueRank) if j == 0]
                                assert(len(tePrelimChampETF)> 0)
                                winner =  tePrelimChampETF[0]
                                rank = [1 for k in range(len(teMat.assets))]
                                rank[teIdxMap[winner]] = 0
                                ref = 'case 6 - No Cap but Te available. %s qualified. '%(len(teMat.assets) - len(missingTe)) + \
                                    'Pick smallest te thats qualified: %.4f Mcap:  %.4f'
                            else:
                                #Got mcap only, no Te
                                McapOnly = [j for j in range(len(mcapMat.assets)) if j not in missingMcap]
                                McapTrueData = ma.take(mcapMat.data[:, dIdx], McapOnly, axis = 0)
                                McapTrueAsset =  ma.take(mcapMat.assets, McapOnly, axis= 0)
                                McapTrueRank = numpy.argsort(-ma.getdata(McapTrueData))
                                McapIdxMap = dict([(j,i) for (i,j) in enumerate(mcapMat.assets)])
                                McapPrelimChampETF = [McapTrueAsset[k] for (k, j) in enumerate(McapTrueRank) if j == 0]
                                assert(len(McapPrelimChampETF)> 0)
                                winner =  McapPrelimChampETF[0]
                                rank = [1 for k in range(len(mcapMat.assets))]
                                rank[McapIdxMap[winner]] = 0
                                ref = 'case 7 - Missing Te but Cap available. %s qualified. '%(len(mcapMat.assets)- len(missingMcap)) + \
                                'Pick biggest cap thats qualified: Te: %.4f Mcap:  %.4f'
                champion = [teMat.assets[i] for (i,j) in enumerate(rank) if j == 0][0]
                champTe = [medianTe[i] for (i,j) in enumerate(rank) if j == 0][0]
                champMcap = [mcapMat.data[:,dIdx][i] for (i,j) in enumerate(rank) if j == 0][0]
                ref = ref %(champTe, champMcap)
                logging.info('Processing matrices...%04d%02d%02d, champ: %s, median_te: %.4f \
mcap: %.4f'%(\
                        d.year, d.month, d.day, champion.etf_name, champTe, champMcap))
                bestETF[d] = (champion, champTe, ref)
            result[index] = bestETF
        return result

if __name__=="__main__":
    usage = "usage: %prog [options] configfile index_id"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--dates",action="store",
                             default='2009-01-01:now',dest="dates",
                             help="dates to do analysis. In the format of minDate:maxDate")
    cmdlineParser.add_option("-m", "--models", action="store",
                             default='WWAxioma2011MH',dest="modelName",
                             help="models to be used")
    cmdlineParser.add_option("--plotTe", action="store_true",
                             default=False, dest="plotTe")
    cmdlineParser.add_option("--resultFile", action="store",
                             default='result', dest="resultFile")
    cmdlineParser.add_option("--verifyAgainstMarketDB", action="store",
                             default=False, dest="verifyAgainstMarketDB")


                             
    (options_, args_) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    if len(args_) != 2:
        cmdlineParser.error("Incorrect no. of input. %s"%usage)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)

    try:
        rmClass = riskmodels.getModelByName(options_.modelName)
    except KeyError:
        print('Unknown risk model "%s"' % options_.modelName)
        names = sorted(riskmodels.modelNameMap.keys())
        print('Possible values:', ', '.join(names))
        sys.exit(1)

    mapTool = mapEIFTool(connections)
    rm = rmClass(connections.modelDB, connections.marketDB)
    logging.info('RiskModel used: %s'%rm.mnemonic)

    input_index_ids = args_[1].split(',')
    indexList =  mapTool.getAllIndexes()
    
    if input_index_ids[0].upper()!=  'ALL':
        input_index_ids = [int(k) for k in input_index_ids]
        indexList = [i for i in indexList if i.index_id in input_index_ids]
    dates, trueStartDate = mapTool.getDates(options_.dates)
    index2ETF, distinctETFAids = mapTool.getIndex2ETFmapping(indexList)

    etfAidSet = set()
    matDict = dict()
    for index in sorted(index2ETF.keys()):
        etfList =  index2ETF[index]
        #matDict store two matrices, first one for te, second one for Mcap
        teMat = TimeSeriesMatrix(etfList, dates)
        #Force teMat.data to be a all-masked matrix
        teMat.data = Matrices.allMasked(teMat.data.shape)
        mcapMat = TimeSeriesMatrix(etfList, dates)
        mcapMat.data = Matrices.allMasked(mcapMat.data.shape)
        matDict[index] = [teMat, mcapMat]
        
    if len(list(index2ETF.values())) == 0 :
        logging.info('No Composite mapped to the index_id %s'%(','.join (k.index_short_name for k in indexList)))
        sys.exit()
    rmgs = mapTool.modelDB.getAllRiskModelGroups()
    compositeSidDict = mapTool.getCompositeSubIssue(distinctETFAids , rmgs) 
    teDayDict= dict()
    eIdxDict = dict()

    for dIdx, d  in enumerate(sorted(dates, reverse=False)):
        indexDayDict = dict()
        logging.info('Processing %s'%d.isoformat())
        imp =  mapTool.modelDB.getIssueMapPairs(d)
        rmi =  mapTool.modelDB.getRiskModelInstance(rm.rms_id, d)
        rollover_d = d 
        rolloverCount = 0
        if rmi is not None and (rmi.has_exposures or rmi.has_risks):
            mapTool.rmi = rmi
            rm.setFactorsForDate(d, mapTool.modelDB)
            mapTool.setRiskModelMat(rm, rmi)
        while (rmi is None or not rmi.has_exposures or not rmi.has_risks) and (rolloverCount < 31):
            rollover_d +=datetime.timedelta(-1)
            rmi =  mapTool.modelDB.getRiskModelInstance(rm.rms_id, rollover_d)
            rolloverCount += 1
            if rmi is not None:
                logging.info('Using rollover %s model on %s for %s'%(options_.modelName, 
                                                            rollover_d.isoformat(),
                                                            d.isoformat()))
                if  mapTool.rmi is None or rmi.date != mapTool.rmi.date:
                    #check self.expM contains the right data. If not, reload!
                    mapTool.rmi = rmi
                    rm.setFactorsForDate(rollover_d, mapTool.modelDB)
                    mapTool.setRiskModelMat(rm, rmi)
                    break
                
        if rolloverCount < 31 and (rmi is None or not rmi.has_exposures or not rmi.has_risks):
            logging.error('Missing risk model data for more than 30 days. Date to check: %s'%d.isoformat())
            continue
        for index in index2ETF.keys():
            sids = list()
            etfList = []
            rmgSet = set()
            portWeight =  mapTool.modelDB.getIndexConstituents(index.index_name,\
                                                                   d,  mapTool.marketDB, 
                                                      rollBack=35, issueMapPairs=imp)
            indexDayDict[index] = portWeight
            fullHistEtfList = index2ETF[index]
            if len(fullHistEtfList) == 0:
                logging.debug('No composite available to pick for index: %s'%index)
                continue
            for eIdx, etf in enumerate(fullHistEtfList):
                (sid, rmg) = compositeSidDict[etf.etf_axioma_id]
                sids.append(sid)
                rmgSet.add(rmg)
                etfList.append(etf)
            if len(etfList) == 0:
                logging.error('No suitable from/thru date ETF for index %s on %s'%\
                                  (index, d.isoformat()))
                continue
            teMat, mcapMat = matDict[index]
            mcapDates =  mapTool.modelDB.getDates(list(rmgSet), d, 19)
            mcaps =  mapTool.modelDB.getAverageMarketCaps(mcapDates, sids, currencyID=1)
            for etfIndex, etf in enumerate(etfList):
                if etf.from_dt > d or etf.thru_dt < d:
                    continue
                mcapMat.data[etfIndex, dIdx] = mcaps[etfIndex]
                
            matDict[index] = teMat, mcapMat
            if index not in indexDayDict:
                continue
            else:
                if len(indexDayDict[index]) == 0 :
                    continue
                else:
                    port = mapTool.processPort(indexDayDict[index])
                    subIssueIds , weights = zip(*port)
                #Get returns for all sids. If all are missing returns, assume it's holiday. Skip
                    rets =  mapTool.modelDB.loadSubIssueData([d], subIssueIds, 'sub_issue_return', 'tr',
                                                cache= mapTool.modelDB.totalReturnCache, withCurrency=False)
                    if len(numpy.flatnonzero(ma.getmaskarray(rets.data))) == len(subIssueIds):
                    #All masked. Skyp
                        logging.info('No return for index %s on %s.'%(index, d.isoformat()))
                        continue
            for etfIndex, etf in enumerate(etfList):
                if etf.from_dt > d or etf.thru_dt < d:
                    continue
                te = mapTool.computeTrackingError(d, etf, port, compositeSidDict)
                teMat.data[etfIndex, dIdx] = te
                
            matDict[index] = teMat, mcapMat

    result = mapTool.processMatrices(matDict, trueStartDate)
    mapTool.processResults(result, options_.resultFile)
    if options_.plotTe is not None:
        for index in matDict.keys():
            etfList = index2ETF.get(index, [])
            teMat, mcapMat = matDict[index]
            bestETFDict = result[index]
            mapTool.plotTe(teMat, mcapMat, bestETFDict, index, 
                           trueStartDate)
                                  
    if options_.verifyAgainstMarketDB:
         mapTool.verifyAgainstMarketDB(result, dates)
        
        

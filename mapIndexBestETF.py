import datetime
import optparse
import configparser
import logging
try:
    import numpy.ma as ma
    HAS_OLD_MA = False
except:
    import numpy.core.ma as ma
    HAS_OLD_MA = True
import numpy
import sys
import os
os.environ["NLS_LANG"] = ".AL32UTF8"

from marketdb import MarketDB
import riskmodels
from riskmodels import Connections
from riskmodels import Utilities
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
        self.ETFRollover = 15
        self.verifyAgainstMarketDBThreshold = 1
        self.rmi = None
        self.regionCtryDict = \
        {'AP':['AP','AU','CN','HK','ID','IN','JP','KR','LK','MY','NZ','PH','PK','SG','TH','TW'],\
         'EU':['EU','AT','BE','CH','DE','DK','ES','FI','FR','GB','IE','IT','LU','NL','NO','PT','SE','IS'\
                  ,'BG','CY','CZ','EE','GR','HR','HU','PL','RO','RU','SI','SK','TR','LV','LT'],
         'NA':['NA','CA','US'],
         'WW':['WW','AE','AR','BD','BH','BR','BW','CI','CL','CO','EC','EG','GH','IL','JM','JO',\
               'KE','KW','KZ','LB','MA','MT','MU','MX','NA','NG','OM','PE','QA','RS','SA','TN','TT',
               'UA','VE','VN','ZA','ZM']}
  
        self.regionModelDict = \
            {'AP':'APAxioma2011MH',\
             'EU':'EUAxioma2011MH',\
             'NA':'NAAxioma2011MH',\
             'WW':'WWAxioma2011MH',
        }    
    def getFileVersion(self, resultFile):
       dirName=os.path.dirname(resultFile)
       if dirName and not os.path.isdir(dirName):
           os.makedirs(dirName)
       i=0
       while os.path.isfile(resultFile + ".v" +str(i)):
           i=i+1
       resultFile=resultFile+'.v'+str(i)

       return resultFile



    def getAllEIFIndexes(self):
        index_ids = list()
        query = """SELECT DISTINCT im.id, im.name,im.short_names, a.underlying_geography, if.id
                   FROM  \
                   FUTURE_FAMILY_ATTRIBUTE_ACTIVE a, index_member im, index_family if
                   WHERE index_member_id IS NOT NULL and a.index_member_id = im.id \
                   and if.id=im.family_id and a.distribute='Y'"""
            
        self.marketDB.dbCursor.execute(query)
        for index_id, index_name, index_short_name, index_region, index_family_id in self.marketDB.dbCursor.fetchall():
            indexval = Utilities.Struct()
            indexval.index_id = index_id
            indexval.index_name = index_name
            indexval.index_short_name = index_short_name
            indexval.index_region = index_region
            indexval.index_family_id = index_family_id
            index_ids.append(indexval)
        return index_ids

    def getStoredTe(self, dateList, indexList):
        idArgList = [('id%d'%i) for i in range(len([i.index_id for i in indexList]))]
        query = """select index_member_id, etf_axioma_id, dt, tracking_error \
                   from index_composite_te where dt between :from_dt_arg and :thru_dt_arg
                   and index_member_id IN(%(ids)s) order by \
                   index_member_id, dt, etf_axioma_id asc""" %\
                                  {'ids':','.join([(':%s'%i ) for i in idArgList])}
        valueDict = dict(zip(idArgList,[i.index_id for i in indexList]))
        valueDict['from_dt_arg'] = dateList[0]
        valueDict['thru_dt_arg'] = dateList[-1]                      
                  
        self.marketDB.dbCursor.execute(query,valueDict) 
        r =  self.marketDB.dbCursor.fetchall()
        resultDict = dict()
        indexesDtDict = dict()
        etfsDtDict = dict()
        if len(r) > 0 :
            for index_member_id, etf_axioma_id, dt , te in r :
                indexesDtDict.setdefault(dt.date(), set()).add(index_member_id)
                etfsDtDict.setdefault(dt.date(),set()).add(etf_axioma_id)
                if (index_member_id, dt.date()) not in resultDict:
                    etfDict = dict([(etf_axioma_id, te)])
                else:
                    etfDict = resultDict[(index_member_id, dt.date())]
                    etfDict[etf_axioma_id] = te
                resultDict[(index_member_id, dt.date())] = etfDict
        return resultDict, indexesDtDict, etfsDtDict

    def getrmClass(self,modelName):
        try:
            rmClass = riskmodels.getModelByName(modelName)
        except KeyError:
            print('Unknown risk model "%s"' % modelName)
            names = sorted(riskmodels.modelNameMap.keys())
            print('Possible values:', ', '.join(names))
            return 
        return rmClass

    def insertTeQuery(self, d, index_member_id, etf_axioma_id, te):
        insertQuery = """ insert into \
                      index_composite_te values(:index_member_id, :etf_axioma_id,:date_arg,:te)"""
        self.marketDB.dbCursor.execute(insertQuery,index_member_id = index_member_id ,\
                                           etf_axioma_id= etf_axioma_id,\
                                           date_arg = d,\
                                           te = te)
        self.marketDB.dbConnection.commit()
    def getRMI(self, rm, d, modelName ):
        rmi =  self.modelDB.getRiskModelInstance(rm.rms_id, d)
        rollover_d = d 
        rolloverCount = 0
        if rmi is not None and (rmi.has_exposures or rmi.has_risks):
            self.rmi = rmi
            rm.setFactorsForDate(d, self.modelDB)
            self.setRiskModelMat(rm, rmi)
        while (rmi is None or not rmi.has_exposures or not rmi.has_risks) and (rolloverCount < 31):
            rollover_d +=datetime.timedelta(-1)
            rmi =  self.modelDB.getRiskModelInstance(rm.rms_id, rollover_d)
            rolloverCount += 1
            if rmi is not None:
                logging.info('Using rollover %s model on %s for %s'%(modelName, 
                                                            rollover_d.isoformat(),
                                                            d.isoformat()))
                if  self.rmi is None or rmi.date != self.rmi.date:
                    #check self.expM contains the right data. If not, reload!
                    self.rmi = rmi
                    rm.setFactorsForDate(rollover_d, self.modelDB)
                    self.setRiskModelMat(rm, rmi)
                    break

        if rolloverCount < 31 and (rmi is None or not rmi.has_exposures or not rmi.has_risks):
            logging.error('Missing risk model data for more than 30 days. \
                           Date to check: %s'%d.isoformat())
    def getallETFMap(self, date):
        query = """SELECT DISTINCT b.axioma_id,b.short_names, x.index_id, y.description  FROM  \
                composite_member b  left join composite_index_map x on \
                x.etf_axioma_id = b.axioma_id 
                left join index_member y on x.index_id=y.id  WHERE \
                b.from_dt <=:dt_arg and b.thru_dt > :dt_arg"""
        self.marketDB.dbCursor.execute(query, dt_arg = date)
        result = list()
        for etf_axioma_id, short_names, index_id, index_name in self.marketDB.dbCursor.fetchall():
            val = Utilities.Struct()
            val.etf_axioma_id = etf_axioma_id
            val.etf_name = short_names
            val.index_id = index_id
            val.index_name = index_name
            result.append(val)
        return result


    def getETFFullName(self, date):
        query = """SELECT distinct a.axioma_id, b.id from composite_member a,\
                   asset_dim_name_active_int b \
                   where a.axioma_id = b.axioma_id  \
                   and :date_arg >=b.from_dt
                   and :date_arg <b.thru_dt"""
        self.marketDB.dbCursor.execute(query, date_arg = date)
        return dict([(i,j) for (i,j) in self.marketDB.dbCursor.fetchall()])


    def getPortExpM(self, portfolio, expMatrix, srDict):
        expM_Map = dict([(expMatrix.getAssets()[i], i) \
                             for i in range(len(expMatrix.getAssets()))])
        # Discard any portfolio assets not covered in model
        indices = [i for (i,j) in enumerate(portfolio) \
                       if j[0] in expM_Map and j[0] in srDict]
        port = [portfolio[i] for i in indices]
        if len(port) == 0:
            return expMatrix.getMatrix().filled(0.0)
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
        return expM

    
    def mapETF2Index(self, modelName, date, etf_axioma_ids, index_ids, etf2IndexFile):
        """ This method is trying to associate an ETF to its corresponding Index by comparing the
          tracking error for the ETF, Index long short portfolio , asset count difference between
          bmk and etf, and the alignment on etf country and bmk country exposure """
        rmClass = self.getrmClass(modelName)
        rm = rmClass(self.modelDB, self.marketDB)
        logging.info('RiskModel used: %s'%rm.mnemonic)
        etfAllMap = self.getallETFMap(date)
        allIndexList = self.getAllEIFIndexes()
        indexFamNameMap = dict([i.id,i] for i in  self.marketDB.getIndexFamilies())
        if index_ids.upper() == 'ALL':
            indexList = allIndexList
        elif type(index_ids) == list() and len(index_ids) < 1:
            logging.error('No Index candidate to assign to')
            return
        else:
            index_ids = [int(i) for i in index_ids.split(',')]
            indexList = [ i for i in allIndexList if i.index_id in index_ids]
            EIFIndexFamilies = [i.index_id for i in allIndexList]
            for i in index_ids:
                if i not in EIFIndexFamilies:
                      logging.info('Index Families %s not tracked by any EIF'%(indexFamNameMap[i]))
        if len(etf_axioma_ids) < 1:
            logging.error('No ETF Input')
            return
        else:
            ### filter out ETF which has already been present in composite index map ###
            mappedETFList = [i for i in etfAllMap if i.etf_axioma_id in etf_axioma_ids and i.index_id is not None]
            mappedETFs = []
            if len(mappedETFList) > 0 :
                for i in mappedETFList:
                    logging.info('%s is mapped to %s in the composite_index_map table'\
                                     %(i.etf_name,i.index_name))
                    mappedETFs = [i.etf_axioma_id for i in mappedETFList]
            ### remains only with ETF which is unassigned with Index ###        
            etfList = [k for k in etfAllMap if k.etf_axioma_id in etf_axioma_ids and k.etf_axioma_id not in  mappedETFs]
            if len(etfList) == 0 :
                logging.info('All input etfs with existing index mapping')
                return
        etfNameDict = self.getETFFullName(date)
        self.getRMI(rm,date,modelName)

        ofile = open('%s'%(etf2IndexFile),'w')
        ofile.write('ETF_Id|ETF_FullName|ETF_ShortName|Mapped_Index_Id|MappedIndex_Name|te|')
        ofile.write('\n')

        
        self.ctryIdx = self.expM.getFactorIndices(ExposureMatrix.CountryFactor)
        indexRiskDict = dict()
        for index in indexList:
            portWeight =  self.modelDB.getIndexConstituents(index.index_name, date,  self.marketDB, 
                                                          rollBack=35)
            if len(portWeight) == 0 :
                logging.error('No index constituent for %s for last 35 days on %s'%(index.index_name,date.isoformat()))
                continue
                  
            else:
                port = self.processPort(portWeight)
                bmkExpM = self.getPortExpM(port, self.expM, self.srDict)
                bmkCtry = self.getCtryExpMList(bmkExpM)
                indexRiskDict[index] = (port, bmkCtry)
        for ETF in etfList:
            ofile.write('%s|%s|%s'%(ETF.etf_axioma_id, etfNameDict.get(ETF.etf_axioma_id, None), ETF.etf_name))
            bestIndex = list()
            indexNames = ['MSCI','RUSSELL','FTSE','STOXX','S&P']
            for index in indexRiskDict.keys():
                bmkPort, bmkCtry = indexRiskDict[index]
                te,  etfCtry = self.computeTrackingError(date,ETF,bmkPort,etf2Index=True)
                ### filter out index whose tracking error against ETF is greater than threshold ### 
                #To pick best index ETF, it has to be tracking error < 0.5%,  
                if round(te,4)>self.trackingErrorThreshold:
                    logging.info('%s ETF and %s Index Tracking error :%s exceeds threshold %s'\
                                     %(ETF.etf_name,index.index_name, \
                                           round(te, 2),self.trackingErrorThreshold)) 
                    continue
                else:
                    #best ETF has to be exposed to the same set of country factors #
                    if len(set(bmkCtry) ^ set(etfCtry)) == 0:
                        logging.info('BMK and ETF country exposure aligned')
                        #If there are 'MSCI' or 'FTSE' in the names, be careful with the name check!
                        indexCheck = [index.index_name.upper().find(k)+1 for k in indexNames]
                        etfCheck = [etfNameDict.get(ETF.etf_name,'').upper().find(k)+1 
                                    for k in indexNames]
                        finalCheck = [indexCheck[idx] + etfCheck[idx] 
                                      for idx in range(len(indexNames))
                                      if indexCheck[idx] + etfCheck[idx] > 0]

                        if len(finalCheck) >= 2:
                            logging.info('final Check : Index Name Mismatch '%(finalCheck))
                            continue

                        else:
                            bestIndex.append((index.index_name, index.index_id, float(te), float(abs(te))))
                    else:
                        logging.info('%s ctry exp. deviate from %s ctry exp. %s'%(index.index_name, ETF.etf_name,set(bmkCtry) ^ set(etfCtry)))
                        

            if len(bestIndex) == 0 :
                    logging.info('No Index mapped to %s'%ETF.etf_name)
                    ofile.write('\n')
                    continue
            else:
                ### see the best matched Index ###
                index_name, index_id, te, abs_te = \
                    sorted(bestIndex, key = lambda tup:tup[3])[0]
                ofile.write('|%s|%s|%s'%(index_id,
                                                index_name,
                                                te))

        ofile.close()
    def mainProcessor(self,input_index_ids, modelName, dates, indexBestETFMapFile=False,
                      plotTeFile=False,\
                      verifyAgainstMarketDBFile=False):
        """ This is a main processor which loop through the dateList, gather the index constituent,
            construct the ETF-Index Long Short Portfolio and compute the tracking error """
        allIndexList =  self.getAllEIFIndexes()
        if input_index_ids.upper()==  'ALL':
            indexList = allIndexList
        elif input_index_ids.upper() in self.regionCtryDict.keys():
            indexList = [i for i in allIndexList if i.index_region in self.regionCtryDict[input_index_ids.upper()]]
            modelName = self.regionModelDict[input_index_ids.upper()]
        else:
            input_index_ids_list = [int(k) for k in input_index_ids.split(',')]
            indexList = [i for i in allIndexList if i.index_id in input_index_ids_list]    

        rmClass = self.getrmClass(modelName)
        rm = rmClass(self.modelDB, self.marketDB)
        logging.info('RiskModel used: %s'%rm.mnemonic)
        dates, trueStartDate = self.getDates(dates)
        index2ETF, distinctETFAids = self.getIndex2ETFmapping(indexList)
        etfAidSet = set()
        matDict = dict()
        for index in index2ETF.keys():
            etfList =  index2ETF[index]
            etfList = [etf for etf in etfList if etf.thru_dt > max(dates)] # Filter out inactive ETFs
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
        rmgs = self.modelDB.getAllRiskModelGroups()
        compositeSidDict = self.getCompositeSubIssue([i.etf_axioma_id for i in distinctETFAids] , rmgs) 
        teDayDict= dict()
        eIdxDict = dict()
        indexStoredTeDict, indexesDtDict, etfsDtDict  = self.getStoredTe(dates, indexList)
        for dIdx, d  in enumerate(sorted(dates, reverse=False)):
            logging.info('Processing %s'%d.isoformat())
            noUnderlyingIndexETFs,noCompositeIndexes = self.getETFwithoutUnderlyingIndex(d)
            storedETFs = etfsDtDict.get(d, set())
            allETFs = set([i.etf_axioma_id for i in  distinctETFAids \
                               if i.from_dt<=d and i.thru_dt >d])
            etfsWithUnderlyingIndex = allETFs-noUnderlyingIndexETFs
            unLoadedETFs = etfsWithUnderlyingIndex ^ storedETFs
            if len(unLoadedETFs) > 0 :
                self.getRMI(rm,d,modelName)

            for index in index2ETF.keys():
                sids = list()
                etfList = []
                rmgSet = set()
                storedTeDict = dict()
                fullHistEtfList = index2ETF[index]
                for eIdx, etf in enumerate(fullHistEtfList):
                    if etf.from_dt <= max(dates) and etf.thru_dt > max(dates):
                        (sid, rmg) = compositeSidDict[etf.etf_axioma_id]
                        sids.append(sid)
                        rmgSet.add(rmg)
                        etfList.append(etf)
                if len(etfList) == 0:
                    logging.error('No suitable from/thru date ETF for index %s on %s'%\
                                      (index, d.isoformat()))
                    continue

                teMat, mcapMat = matDict[index]
                mcapDates =  self.modelDB.getDates(list(rmgSet), d, 19)
                mcaps =  self.modelDB.getAverageMarketCaps(mcapDates, sids, currencyID=1)
                etfPosDict = dict([(j.etf_axioma_id, i) for (i,j) in enumerate(etfList)])
                for etfIndex, etf in enumerate(etfList):
                    if etf.etf_axioma_id in [i.etf_axioma_id for i in fullHistEtfList]:
                        try:
                            mcapMat.data[etfIndex, dIdx] = mcaps[etfPosDict[etf.etf_axioma_id]]
                        except:
                            continue
                matDict[index] = teMat, mcapMat
                            
                if index.index_id in noCompositeIndexes:
                    continue
                ## try to look into index_composite_te to see if te has already been computed ##
                elif (index.index_id, d) in indexStoredTeDict.keys():
                    storedETFs = set([i for i in indexStoredTeDict.get((index.index_id, d), list())])
                    ## check if all the ETFs and their corresponding index tracking error is present or not ##
                    if len(storedETFs^set([i for i in etfList])) > 0 :                  
                        storedTeDict = indexStoredTeDict[(index.index_id, d)]
                        for etfIndex, etf in enumerate(etfList):
                            if etf.etf_axioma_id in storedTeDict.keys():
                                te = storedTeDict[etf.etf_axioma_id]
                                teMat.data[etfIndex, dIdx] = te
                else:
                    ## if missing index te on a particular date, or a missing ETF te is found under an existed Index, we then calcuate its tracking error now ##
                    portWeight =  self.modelDB.getIndexConstituents(index.index_name,\
                                                                       d,  self.marketDB, 
                                                          rollBack=35)
                    if len(portWeight) == 0 :
                        logging.error('No index constituent for %s for last 35 days on %s'%(index.index_name,d.isoformat()))
                        continue
                    else:
                        try:
                            port = self.processPort(portWeight)
                        except AttributeError:
                            logging.warning('mdl12s not set. Running getRMI.')
                            logging.warning('This should only happen when we have not already loaded the universe and have an ETF with two indexes mapped in COMPOSITE_INDEX_MAP.')
                            logging.warning('Expect an info message with "No return"')
                            self.getRMI(rm,d,modelName)
                            port = self.processPort(portWeight)
                        subIssueIds , weights = zip(*port)
                        #Get returns for all sids. If all are missing returns, assume it's holiday. Skip
                        rets =  self.modelDB.loadSubIssueData([d], subIssueIds,\
                                                           'sub_issue_return', 'tr',
                                                        cache= self.modelDB.totalReturnCache,\
                                                                  withCurrency=False)
                        if len(numpy.flatnonzero(ma.getmaskarray(rets.data))) \
                                == len(subIssueIds):
                        #All masked. Skip ##
                            logging.info('No return for index %s on %s.'%(index,\
                                                                              d.isoformat()))
                            continue
                        ## if the index and its etf te existed in index_composite_te,retrieve them from the table now ##
                        if (index.index_id, d) in indexStoredTeDict.keys():
                            storedTeDict = indexStoredTeDict[(index.index_id, d)]
                            for etfIndex, etf in enumerate(etfList):
                                if etf.etf_axioma_id in storedTeDict.keys():
                                     te = storedTeDict[etf.etf_axioma_id]
                                     teMat.data[etfIndex, dIdx] = te
                                else:     
                                    te, etfCtry  = self.computeTrackingError(d, etf, port, compositeSidDict)
                                    teMat.data[etfIndex, dIdx] = te

                                    try:
                                        self.insertTeQuery(d, index.index_id, \
                                                               etf.etf_axioma_id,\
                                                               te)
                                    except:
                                         #something must have gone wrong!!!!!!
                                        logging.error('Index: %s, ETF: %s'%(index, etf))

                        else:
                            for etfIndex, etf in enumerate(etfList):
                                te,  etfCtry = self.computeTrackingError(d, etf, port)
                                teMat.data[etfIndex, dIdx] = te
                                try:
                                    self.insertTeQuery(d, index.index_id, \
                                                           etf.etf_axioma_id,\
                                                           te)
                                except:
                                         #something must have gone wrong!!!!!!
                                    logging.error('Index: %s, ETF: %s'%(index, etf))
                                   
            matDict[index] = teMat, mcapMat
            
        result = self.processMatrices(matDict, trueStartDate)
        if plotTeFile is not None:
            for index in matDict.keys():
                etfList = index2ETF.get(index, [])
                teMat, mcapMat = matDict[index]
                bestETFDict = result[index]
                self.plotTe(teMat, mcapMat, bestETFDict, index, 
                               trueStartDate, plotTeFile)

        if verifyAgainstMarketDBFile is not None:
            self.verifyAgainstMarketDB(result, dates,verifyAgainstMarketDBFile)
        if indexBestETFMapFile is not None:    
            self.processResults(result, indexBestETFMapFile)
    def getETFwithoutUnderlyingIndex(self, date):
        noUnderlyingIndexETFs = set()
        noCompositeIndexes = set()
        check_query = """ select distinct etf_axioma_id, index_id \
                         from composite_index_map x,  index_member a where \
                         x.index_id=a.id and
                         not exists (select * from \
                         index_constituent_easy b where b.index_id=a.id and dt 
                         between :from_dt_arg and :thru_dt_arg)"""
        self.marketDB.dbCursor.execute(check_query, from_dt_arg =date-datetime.timedelta(45),\
                                           thru_dt_arg = date)
        for (i,j) in self.marketDB.dbCursor.fetchall():
            noUnderlyingIndexETFs.add(i)
            noCompositeIndexes.add(int(j))
        return noUnderlyingIndexETFs, noCompositeIndexes
        


    def getIndex2ETFmapping(self, indexVals):
        index2ETF = dict()
        distinctETFs = set()
        indexInfoDict = dict([(i.index_id, i ) for i in indexVals])
        idArgList = [('id%d'%i) for i in range(len([i.index_id for i in indexVals]))]
        query = """select a.*, c.short_names  from (SELECT DISTINCT a.index_id, a.etf_axioma_id,
                   min(b.dt) as min_dt, max(b.dt) as max_dt
                   FROM composite_index_map a, composite_revision_active b, composite_member c
                   WHERE a.etf_axioma_id= c.axioma_id and a.index_id  IN (%(ids)s) and c.axioma_id = a.etf_axioma_id and b.composite_id=c.id
                   group by a.index_id, a.etf_axioma_id) a left outer join composite_member c   on c.thru_dt>a.max_dt and c.axioma_id = a.etf_axioma_id
                """ % {'ids':','.join([(':%s'%i ) for i in idArgList])}
        valueDict = dict(zip(idArgList,[i.index_id for i in indexVals]))
        self.marketDB.dbCursor.execute(query, valueDict)
        for index_id, etf_axioma_id, from_dt, thru_dt, latest_etf_short_name in self.marketDB.dbCursor.fetchall():
            val = Utilities.Struct()
            val.etf_axioma_id = etf_axioma_id
            val.etf_name = latest_etf_short_name
            val.from_dt = from_dt.date()
            val.thru_dt = thru_dt.date()+ datetime.timedelta(self.ETFRollover)
            index_val = indexInfoDict[int(index_id)]
            index2ETF.setdefault(index_val, list()).append(val)
            distinctETFs.add(val)
            
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
        newPort = list(zip(assets, weights))
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

    def processResults(self, results, resultFile):
        wfile = open('%s'%(resultFile),'w')
        wfile.write('Index_Name|Index_ID|Composite_aid|From_dt|Thru_dt')
        wfile.write('\n')
  
        for index in sorted(results.keys()):
            bestETFDict = results[index]
            proxyETF = list()
            for dIdx, d in enumerate(sorted(bestETFDict.keys())):
                etf, te,mcap, ref, mcapMat, medianTe = bestETFDict.get(d)
                if len(proxyETF) == 0:  
                    ## first record on the proxyETF ##
                    proxyETF.append((index.index_name, index.index_id,etf.etf_axioma_id\
                                         ,d,d+ datetime.timedelta(1), dIdx ))
                elif len(proxyETF) > 0 :
                    iname, iid,p_etf_axioma_id, p_from_dt, p_thru_dt, p_dIdx=proxyETF[-1]
                    if dIdx-p_dIdx <= self.ETFRollover:
                        ## skip the holes between weekend / etf rollover period ##
                        if p_etf_axioma_id == etf.etf_axioma_id :
                            proxyETF[-1] = iname, iid ,p_etf_axioma_id,p_from_dt,\
                                d+ datetime.timedelta(1),dIdx
                        else:    
                            proxyETF[-1] = iname, iid ,p_etf_axioma_id,p_from_dt,\
                                d,dIdx
                            proxyETF.append((iname, iid, etf.etf_axioma_id \
                                             , d, d+ datetime.timedelta(1),dIdx))
                    ## possible holes which could be seen by clients since the missing period is greater than ETF rollover day ###
                    else:    
                        proxyETF.append((iname, iid, etf.etf_axioma_id \
                                             , d, d+ datetime.timedelta(1),dIdx))
            for (index_name, index_id ,etf_aid, from_dt, thru_dt, dIdx) in proxyETF:
                wfile.write('%s|%s|%s|%s|%s'%(index_name, index_id,etf_aid,from_dt, thru_dt))
                wfile.write('\n')
        wfile.close()

    def plotTe(self, teMat, mcapMat, bestETFDict, indexval, date, plotTeFile):
        """ Returns a report displaying the daily tracking error between an index and its corresponding ETF. 
            Each index will be stored under an individual csv file"""
        ofile = open('%s_%s_%04d%02d%02d.csv'%(plotTeFile,indexval.index_short_name.replace('/', '_'),
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
            bestETF = bestETFDict.get(d,[Utilities.Struct(),'','','','',''])
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
    
    def getCtryExpMList(self, expM):
        resultList = list()
        ctryMat = ma.take(expM, self.ctryIdx, axis=0)   
        for idx, i in enumerate(range(len(self.ctryIdx))):
            if numpy.sum(ctryMat[i,:], axis = 0) > 0:
                idxCtryMap = dict([(j,i) for (i,j) in
                                   self.expM.factorIdxMap_
                                   [ExposureMatrix.CountryFactor].items()])
                resultList.append(idxCtryMap[self.ctryIdx[idx]])
        return resultList      

    def getStoredChampionETF(self, date):
        dbRecordDict=dict() 
        query = """select a.index_member_id, a.etf_axioma_id, b.short_names
                   from index_best_composite_map a, composite_member b 
                   where a.from_dt <=:dt_arg and a.etf_axioma_id = b.axioma_id
                   and a.thru_dt > :dt_arg"""
        self.marketDB.dbCursor.execute(query, dt_arg=date)
        r = self.marketDB.dbCursor.fetchall()
        for index_id, etf_axioma_id, etf_short_name in r :
            dbRecordDict[index_id] = etf_axioma_id, etf_short_name
      
        return dbRecordDict    
    
      
    def verifyAgainstMarketDB(self, result,dateList, verifyAgainstMarketDBFile):
        reportDate = dateList[-1]
        ofile = open('%s'%(verifyAgainstMarketDBFile),'w')
        ffile = open('%s.fmt'%(verifyAgainstMarketDBFile),'w')
        dbRecordDict = self.getStoredChampionETF(reportDate)
        QAList = list()
        notQAList = list()
        for index in result.keys():
            bestETFDict = result[index]
            if reportDate not in bestETFDict:
                continue
            else:
                deductedETF, champTe, champMcap,  ref, mcapMat, medianTe  = bestETFDict[reportDate]
            storedETF_aid, storedETF_name = dbRecordDict.get(index.index_id, (None, None))
            if storedETF_aid != deductedETF.etf_axioma_id:
                posMap = dict([j.etf_axioma_id,i] for (i,j) in enumerate(mcapMat.assets))
                teDiff = 999.99
                champTe_mask= ma.getmaskarray(champTe)
                champMcap_mask = ma.getmaskarray(champMcap)
                mcapMat.mask = ma.getmaskarray(mcapMat.data[:,-1])
                medianTe.mask = ma.getmaskarray(medianTe)
                if champTe_mask == False:
                    champTe = champTe*100
                else:
                    champTe = 999999999.99
                if champMcap_mask == False:
                    champMcap = champMcap
                else:
                    champMcap = 999999999.99
                storedETF_mcap = 999999999.99;storedETF_medianTe = 999999999.99
                if storedETF_aid is not None: 
                    if mcapMat.mask[posMap[storedETF_aid]] == False: 
                        storedETF_mcap =  mcapMat.data[posMap[storedETF_aid],-1]
                    if medianTe.mask[posMap[storedETF_aid]] == False:
                        storedETF_medianTe =  medianTe[posMap[storedETF_aid]]*100
                try:    
                    teDiff = abs(storedETF_medianTe -champTe)
                except:
                    teDiff = 999.99
                if teDiff < self.verifyAgainstMarketDBThreshold:
                    notQAList.append((reportDate.isoformat(),index.index_name,
                                      deductedETF.etf_axioma_id, deductedETF.etf_name,
                                      champTe, champMcap,
                                      storedETF_aid, storedETF_name,
                                      storedETF_medianTe,storedETF_mcap,
                                      teDiff, ref))
                else:
                    QAList.append((reportDate.isoformat(),index.index_name,
                                   deductedETF.etf_axioma_id, deductedETF.etf_name,
                                   champTe, champMcap,
                                   storedETF_aid, storedETF_name,
                                   storedETF_medianTe,storedETF_mcap,
                                   teDiff, ref))

        ffile.write('\n\n')
        ofile.write('MISMATCH BEST PROXY ETF ON %s with TE_DIFF > %s (QA required): \n'%\
                        (reportDate.isoformat(),self.verifyAgainstMarketDBThreshold))
        ffile.write('MISMATCH BEST PROXY ETF ON %s with TE_DIFF > %s (QA required): \n'%\
                        (reportDate.isoformat(),self.verifyAgainstMarketDBThreshold))
        ofile.write('%-10s | %-10s | %-10s | %-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-15s \
                    |%-5s|%-100s\n' %\
                    ('DATE','INDEX','PROPOSED_ETF','ETF_NAME', 'MEDIAN_TE(inBasisPt)','MARKET_CAP',\
                     'DB_ETF','ETF_NAME','MEDIAN_TE(inBasisPt)','MARKET_CAP','TE_DIFF','REF'))
        ffile.write('%-10s | %-10s | %-10s | %-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-15s \
                    |%-5s|%-100s\n' %\
                    ('DATE','INDEX','PROPOSED_ETF','ETF_NAME', 'MEDIAN_TE(inBasisPt)','MARKET_CAP',\
                         'DB_ETF','ETF_NAME','MEDIAN_TE(inBasisPt)','MARKET_CAP','TE_DIFF','REF'))

        for reportDate,index_name,etf_axioma_id, etf_name, champTe, champMcap, storedETF_aid, storedETF_name, storedETF_medianTe,storedETF_mcap, teDiff, ref in QAList:
                     
            ofile.write('%-10s | %-10s | %-10s | %-15s | %-10.6f | %-15.6f|\
%-10s | %-10s | %-10.2f | %-15.2f|%-10.2f|%-100s\n' %\
                            (reportDate,index_name,\
                     etf_axioma_id, etf_name,\
                     champTe, champMcap,\
                     storedETF_aid, storedETF_name,\
                     storedETF_medianTe,storedETF_mcap,\
                     teDiff, ref))
            ffile.write('%-10s | %-10s | %-10s | %-15s | %-10.2f| %-15.2f|\
%-10s | %-10s | %-10.2f| %-15.2f|%-10.2f|%-100s\n' %\
                    (reportDate,index_name,\
                     etf_axioma_id, etf_name,\
                     champTe, champMcap,\
                     storedETF_aid, storedETF_name,\
                     storedETF_medianTe,storedETF_mcap,teDiff, ref))
        ofile.write('%s\n'%('-'*200))

        ffile.write('\n\n')
        ofile.write('\n\n')

        ofile.write('MISMATCH BEST PROXY ETF ON %s with TE_DIFF < %s (QA not required): \n'%\
                        (reportDate,self.verifyAgainstMarketDBThreshold))
        ffile.write('MISMATCH BEST PROXY ETF ON %s with TE_DIFF < %s (QA not required): \n'%\
                        (reportDate,self.verifyAgainstMarketDBThreshold))
        ofile.write('%-10s | %-10s | %-10s | %-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-15s \
                    |%-5s|%-100s\n' %\
                    ('DATE','INDEX','PROPOSED_ETF','ETF_NAME', 'MEDIAN_TE(inBasisPt)','MARKET_CAP',\
                     'DB_ETF','ETF_NAME','MEDIAN_TE(inBasisPt)','MARKET_CAP','TE_DIFF','REF'))
        ffile.write('%-10s | %-10s | %-10s | %-15s | %-10s | %-10s | %-10s | %-10s | %-10s | %-15s \
                    |%-5s|%-100s\n' %\
                    ('DATE','INDEX','PROPOSED_ETF','ETF_NAME', 'MEDIAN_TE(inBasisPt)','MARKET_CAP',\
                         'DB_ETF','ETF_NAME','MEDIAN_TE(inBasisPt)','MARKET_CAP','TE_DIFF','REF')) 
        for reportDate,index_name,etf_axioma_id, etf_name,champTe, champMcap, storedETF_aid, storedETF_name, storedETF_medianTe,storedETF_mcap, teDiff, ref in notQAList:
                     
            ofile.write('%-10s | %-10s | %-10s | %-15s | %-10.6f | %-15.6f|\
%-10s | %-10s | %-10.2f | %-15.2f|%-10.2f|%-100s\n' %\
                            (reportDate,index.index_name,\
                                 etf_axioma_id, etf_name,\
                                 champTe, champMcap,\
                                 storedETF_aid, storedETF_name,\
                                 storedETF_medianTe,storedETF_mcap,\
                                 teDiff, ref))
            ffile.write('%-10s | %-10s | %-10s | %-15s | %-10.2f| %-15.2f|\
%-10s | %-10s | %-10.2f| %-15.2f|%-10.2f|%-100s\n' %\
                    (reportDate,index_name,\
                     etf_axioma_id, etf_name,\
                     champTe, champMcap,\
                     storedETF_aid, storedETF_name,\
                     storedETF_medianTe,storedETF_mcap,teDiff, ref))
        ofile.write('%s\n'%('-'*200))

        ffile.close()                    
        ofile.close()
    def computeTrackingError(self, d, etf, port, etf2Index=False):
        etfPort = self.modelDB.getCompositeConstituents('dummy', d, self.marketDB, rollBack=15,\
                                                            compositeAxiomaID=ModelID.ModelID(string=etf.etf_axioma_id))[1]
        if len(etfPort) == 0:
            logging.info('Missing etf constituent data %s on %s'%(etf.etf_name, d.isoformat()))
            return ma.masked, list()
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
            if etf2Index:
                etfExpM = self.getPortExpM(etfNewPort, self.expM, self.srDict)
                etfCtry=self.getCtryExpMList(etfExpM)
            else:
                etfCtry =list()

            for asset in diffAssets:
                if asset in etfPortDict.keys():
                    weight = etfPortDict[asset]
                else:
                    weight = -1 * bmkPortDict[asset]
                longShortPort.append((asset, weight))    
            for asset in commonAssets:
                weight = etfPortDict[asset] - bmkPortDict[asset]
                longShortPort.append((asset, weight))

            te = self.compute_total_risk_portfolio(longShortPort, self.expM,
                                                      self.factorCov, self.srDict,
                                                      self.scDict)
            return te,  etfCtry 
        
    def processMatrices(self, matDict, startDate):

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
                                ref = 'case 8 - No cap, no te. with more than 1 live ETFs, pick the one with earliest from_dt. %.4f Mcap: %.4f'
                                    

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
                        McapTrueOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(McapTrueData, reverse=True))])
                        McapTrueDataMap = dict([(i, McapTrueOrderMap[j]) for (i,j) in enumerate(McapTrueData)])
                        McapTrueRank = [McapTrueDataMap[i] for (i,j) in enumerate(McapTrueData)]
                        
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
                            trueOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(trueData, reverse=True))])
                            trueDataMap = dict([(i, trueOrderMap[j]) for (i,j) in enumerate(trueData)])
                            trueRank=[trueDataMap[i] for (i,j) in enumerate(trueData)]
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
                                McapTrueOrderMap = dict([(j,i) for (i, j) in enumerate(sorted(McapTrueData, reverse=True))])
                                McapTrueDataMap = dict([(i, McapTrueOrderMap[j]) for (i,j) in enumerate(McapTrueData)])
                                McapTrueRank = [McapTrueDataMap[i] for (i,j) in enumerate(McapTrueData)]
                                McapIdxMap = dict([(j,i) for (i,j) in enumerate(mcapMat.assets)])
                                McapPrelimChampETF = [McapTrueAsset[k] for (k, j) in enumerate(McapTrueRank) if j == 0]
                                assert(len(McapPrelimChampETF)> 0)
                                winner =  McapPrelimChampETF[0]
                                rank = [1 for k in range(len(mcapMat.assets))]
                                rank[McapIdxMap[winner]] = 0
                                ref = 'case 7 - Missing Te but Cap available. %s qualified. '%(len(mcapMat.assets)- len(missingMcap)) + \
                                'Pick biggest cap thats qualified: Te: %.4f Mcap:  %.4f'
                champion = [teMat.assets[i] for (i,j) in enumerate(rank) if j == 0][0]
                champTe = [medianTe[i][0] for (i,j) in enumerate(rank) if j == 0][0]
                champMcap = [mcapMat.data[:,dIdx][i] for (i,j) in enumerate(rank) if j == 0][0]
                ref = ref %(champTe, champMcap)
                logging.info('Processing matrices...%04d%02d%02d, champ: %s, median_te: %.4f \
mcap: %.4f'%(\
                        d.year, d.month, d.day, champion.etf_name, champTe, champMcap))
                bestETF[d] = (champion, champTe,champMcap, ref, mcapMat, medianTe)
                
            result[index] = bestETF
        return result

if __name__=="__main__":
    usage = "usage: %prog [options] configfile"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--dates",action="store",
                             default='2009-01-01:now',dest="dates",
                             help="dates to do analysis. In the format of minDate:maxDate")
    cmdlineParser.add_option("-m", "--models", action="store",
                             default='WWAxioma2011MH',dest="modelName",
                             help="models to be used")
    cmdlineParser.add_option("--plotTe", action="store",\
                             default=None, dest="plotTe")
    cmdlineParser.add_option("--indexBestETFMap", action="store",
                             default=None, dest="indexBestETFMap")
    cmdlineParser.add_option("-v", "--verifyAgainstMarketDB", action="store",
                             default=None, dest="verifyAgainstMarketDB")
    cmdlineParser.add_option("--etf2Index", action="store",\
                             default=None, dest="etf2Index")
    cmdlineParser.add_option("--index-ids",action="store",
                             default='ALL', dest="input_index_ids")
    cmdlineParser.add_option("--etf_aid", action="store",
                             default='', dest="etfAids")

    (options_, args_) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    if len(args_) != 1:
        cmdlineParser.error("Incorrect no. of input. %s"%usage)
        
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)

    mapTool = mapEIFTool(connections)
    input_index_ids = options_.input_index_ids
    modelName = options_.modelName
    dates = options_.dates
    indexBestETFMapFile = options_.indexBestETFMap
    verifyAgainstMarketDBFile = options_.verifyAgainstMarketDB
    etf2IndexFile = options_.etf2Index
    plotTeFile = options_.plotTe
    etfAids= options_.etfAids.split(',')

    optionCheck = 0 
    if indexBestETFMapFile is not None:
        indexBestETFMapFile = mapTool.getFileVersion(indexBestETFMapFile)
        optionCheck+=1
    if verifyAgainstMarketDBFile is not None:
        verifyAgainstMarketDBFile = mapTool.getFileVersion(verifyAgainstMarketDBFile)
        optionCheck+=1
    if plotTeFile is not None:
        plotTeFile = mapTool.getFileVersion(plotTeFile)
        optionCheck+=1
    if etf2IndexFile is not None:
        etf2IndexFile =  mapTool.getFileVersion(etf2IndexFile)
    if optionCheck > 0:
        result = mapTool.mainProcessor(input_index_ids, modelName, dates, 
                                   indexBestETFMapFile, 
                                   plotTeFile, \
                                   verifyAgainstMarketDBFile)    
    if options_.etf2Index:
        mapTool.mapETF2Index(modelName, Utilities.parseISODate(dates), etfAids, input_index_ids,
                             etf2IndexFile)

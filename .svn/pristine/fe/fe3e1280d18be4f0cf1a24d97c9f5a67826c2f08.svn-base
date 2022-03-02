import configparser
import optparse
import logging
import numpy
import numpy.ma as ma
from marketdb.qa import TimeSeriesChart
from riskmodels import Connections
from riskmodels import Utilities

class CompareESTU:
    def __init__(self, marketDB, modelDB, dtList, baseRM, newRM):
        self.marketDB = marketDB
        self.modelDB = modelDB
        self.dtList = dtList
        self.baseRM = baseRM
        self.newRM = newRM
        self.baseRMInfo = self.modelDB.getRiskModelInfo(baseRM.rm_id, baseRM.revision)
        self.newRMInfo = self.modelDB.getRiskModelInfo(newRM.rm_id, newRM.revision)
        self.modelDB.createCurrencyCache(self.marketDB, 1, days=6000)
        
    def loadEstimationUniverse(self, rm, dt, compareStat):
        rmi_id = self.modelDB.getRiskModelInstance(rm.rms_id, dt)
        estuMap = self.modelDB.getEstuMappingTable(rm.rms_id)
        fullestu =[]
        if estuMap is None:
            fullestu = self.modelDB.getRiskModelInstanceESTU(rmi_id)
        else:
            for name in estuMap.keys():
                idx = estuMap[name].id
                logging.info('Loading %s estimation universe, ID: %s', name, idx)
                estuMap[name].assets = self.modelDB.getRiskModelInstanceESTU(rmi_id, estu_idx=idx)
                fullestu += estuMap[name].assets
            if len(estuMap['main'].assets) > 0:
                estu = estuMap['main'].assets
            else:
                self.log.warning('Main estimation universe empty')
                estu = []
        if not compareStat:
            assert(len(fullestu) > 0)
            return fullestu
        else:
            assert(len(estu) > 0)
            return estu

    def compare(self, dt, compareStat, percentile=30):
        if compareStat:
            percentile = 0
        baseEstu = self.loadEstimationUniverse(self.baseRM, dt, compareStat)
        newEstu = self.loadEstimationUniverse(self.newRM, dt, compareStat)
        # only look for those in base model but not in new models
        mcapDates = self.modelDB.getAllRMDates(self.baseRMInfo.rmgTimeLine, dt, 20, 
                                               excludeWeekend=True, fitNum=True)[1]
        marketCaps = self.modelDB.getAverageMarketCaps(mcapDates, baseEstu, 1, self.marketDB)
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(baseEstu)])
        eligibleIdx = numpy.flatnonzero(ma.getmaskarray(marketCaps)==0)
        sortedIdx = ma.argsort(-marketCaps)
        sortedIdx = sortedIdx[:len(eligibleIdx)]
        targetCapRatio = (100 - percentile) / 100.0
        runningCapRatio = numpy.cumsum(ma.take(marketCaps, sortedIdx), axis=0)
        runningCapRatio /= ma.sum(ma.take(marketCaps, sortedIdx))
        reachedTarget = list(runningCapRatio >= targetCapRatio)
        m = min(reachedTarget.index(True)+1, len(sortedIdx))
        eligibleIdx = sortedIdx[:m]
        eligibleAssets = [baseEstu[i] for i in eligibleIdx]
        if not compareStat:
            # only look for those in base model but not in new model
            diff = list((set(baseEstu)-set(newEstu))&set(eligibleAssets))
        else:
            diff = list((set(baseEstu)^set(newEstu))&set(eligibleAssets))
        result = [(i, marketCaps[assetIdxMap[i]]) for i in diff]
        return result

    def writeResult(self, sidDtDict, dIdxMap, outfile):
        for sid, missingDtAndCap in sidDtDict.items():
            missingDates, mcaps = zip(*missingDtAndCap)
            missingDtIdxMap = dict([(j,i) for (i,j) in enumerate(missingDates)])
            dtMcapsMap = dict([(dt,mcaps[idx]) for idx, dt in enumerate(missingDates)])
            for dIdx, date in enumerate(missingDates):
                if dIdx == 0:
                    fromDt = date
                    thruDt = None
                if len(missingDates) == 1:
                    thruDt = date 
                if thruDt is not None:
                    outfile.write('%s,%s,%s,%.2f\n'%(
                            sid.getSubIDString(), fromDt, thruDt, dtMcapsMap[fromDt]))
                elif dIdx == 0:
                    continue
                else:
                    dPos = dIdxMap[date]
                    prevDPos = dIdxMap[missingDates[dIdx - 1]]
                    if dIdx == len(missingDates) - 1:
                        thruDt = date
                        missingMcaps = mcaps[missingDtIdxMap[fromDt]:missingDtIdxMap[thruDt]+1]
                        avgMcap = sum(missingMcaps)/len(missingMcaps)
                        outfile.write('%s,%s,%s,%.2f\n'%(
                                sid.getSubIDString(), fromDt, thruDt, avgMcap))
                        continue
                    if dPos - prevDPos > 1:
                        thruDt = missingDates[dIdx - 1]
                        missingMcaps = mcaps[missingDtIdxMap[fromDt]:missingDtIdxMap[thruDt]+1]
                        avgMcap = sum(missingMcaps)/len(missingMcaps)
                        outfile.write('%s,%s,%s,%.2f\n'%(
                                sid.getSubIDString(), fromDt, thruDt, avgMcap))
                        fromDt = date
                        thruDt = None
        outfile.close()

    def output(self, outfile, compareStat=False):
        sidDtDict = dict()
        dIdxMap = dict([(j,i) for (i,j) in enumerate(self.dtList)])
        for dt in self.dtList:
            result = self.compare(dt, compareStat)
            for (sid, mcap) in result:
                sidDtDict.setdefault(sid, list()).append((dt, mcap))
        self.writeResult(sidDtDict, dIdxMap, outfile)

    def plotMcap(self):
        mcapList0 = list()
        mcapList1 = list()
        estumcapList0 = list()
        estumcapList1 = list()
        for dt in self.dtList:
            logging.info('Processing %s', str(dt))
            rmi0 = self.modelDB.getRiskModelInstance(self.baseRM.rms_id, dt)
            rmi1 = self.modelDB.getRiskModelInstance(self.newRM.rms_id, dt)
            baseunivere = self.modelDB.getRiskModelInstanceUniverse(rmi0, restrictDates=False,
                                                                    returnExtra=True)
            newunivere = self.modelDB.getRiskModelInstanceUniverse(rmi1, restrictDates=False,
                                                                   returnExtra=True)
            baseEstu = self.loadEstimationUniverse(self.baseRM, dt, False)
            newEstu = self.loadEstimationUniverse(self.newRM, dt, False)
            mcapDates = self.modelDB.getAllRMDates(self.baseRMInfo.rmgTimeLine, dt, 20,
                                                   excludeWeekend=True, fitNum=True)[1]
            mcap0 = self.modelDB.getAverageMarketCaps(mcapDates, baseunivere, 1, self.marketDB)
            mcap1 = self.modelDB.getAverageMarketCaps(mcapDates, newunivere, 1, self.marketDB)
            mcap0 = mcap0.filled(0)
            mcap1 = mcap1.filled(0)
            assetIdxMap0 = dict([(j,i) for (i,j) in enumerate(baseunivere)])
            assetIdxMap1 = dict([(j,i) for (i,j) in enumerate(newunivere)])
            estuMcap0 = 0.0
            estuMcap1 = 0.0
            for sid in baseEstu:
                estuMcap0 += mcap0[assetIdxMap0[sid]]
            for sid in newEstu:
                estuMcap1 += mcap1[assetIdxMap1[sid]]
            mcapList0.append(sum(mcap0))
            mcapList1.append(sum(mcap1))
            estumcapList0.append(estuMcap0)
            estumcapList1.append(estuMcap1)
        name = self.baseRMInfo.mnemonic+'_mcap_vs_'+self.newRMInfo.mnemonic+'_mcap'
        chart = TimeSeriesChart.TimeSeriesChart(name, name)
        chart.addTimeSeries(self.dtList, mcapList0, self.baseRMInfo.mnemonic+' universe')
        chart.addTimeSeries(self.dtList, mcapList1, self.newRMInfo.mnemonic+' universe',
                            formatStr='r-')
        chart.addTimeSeries(self.dtList, estumcapList0,
                            self.baseRMInfo.mnemonic+' estu', formatStr='b--')
        chart.addTimeSeries(self.dtList, estumcapList1,
                            self.newRMInfo.mnemonic+' estu', formatStr='r--')
        chart.create()
        chart.close()
    
            
def getModelRevision(modelDB, rms_id):
    modelDB.dbCursor.execute("""select rm_id, revision from risk_model_serie 
                             where serial_id=:rmsid_arg""",rmsid_arg=rms_id)
    r = modelDB.dbCursor.fetchall()
    result = Utilities.Struct()
    result.rm_id = r[0][0]
    result.revision = r[0][1]
    result.rms_id = rms_id
    return result

if __name__=="__main__":
    usage = "usage: %prog [options] configfile startDt [endDt] rms_id1 rms_id2"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--filename",action="store",
                             default='',dest="fname",
                             help="output file name")
    cmdlineParser.add_option("--compareStat",action="store_true",
                             default=False,dest="compareStat",
                             help="compare Fundamental with Stat model")
    cmdlineParser.add_option("--plot",action="store_true",
                             default=False,dest="plotMcap",
                             help="plot model universe and estu mcap")
    (options_, args_) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    if len(args_) < 4 or len(args_) > 5:
        logging.error('Incorrect numbers of arguments')

    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)

    modelDB = connections.modelDB
    marketDB = connections.marketDB

    startDt = Utilities.parseISODate(args_[1])
    if len(args_) == 4:
        endDt = startDt
        baseRMSID = args_[2]
        newRMSID = args_[3]
    else:
        endDt = Utilities.parseISODate(args_[2])
        baseRMSID = args_[3]
        newRMSID = args_[4]

    baseRMrevision = getModelRevision(modelDB, baseRMSID)
    newRMrevision = getModelRevision(modelDB, newRMSID)
    baseRMInfo = modelDB.getRiskModelInfo(baseRMrevision.rm_id, baseRMrevision.revision)
    newRMInfo = modelDB.getRiskModelInfo(newRMrevision.rm_id, newRMrevision.revision)
    modelDates = modelDB.getDateRange(baseRMInfo.rmgTimeLine, startDt, endDt, excludeWeekend=True)
    compareEstu = CompareESTU(marketDB, modelDB, modelDates, baseRMrevision, newRMrevision)
    
    if not options_.plotMcap:
        if options_.fname == '':
            fname = 'Missing_Estu_%s_vs_%s.csv'%(baseRMInfo.name, newRMInfo.name)
        else:
            fname = options_.fname+'.csv'
        logging.info('Processing Estu comparison bewteen %s and %s', baseRMInfo.name, newRMInfo.name)
        outfile = open(fname, 'w')
        compareEstu.output(outfile, options_.compareStat)
        logging.info('Write to %s'%fname)
    else:
        compareEstu.plotMcap()

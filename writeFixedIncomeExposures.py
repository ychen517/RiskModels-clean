
import numpy
import logging
import datetime
import optparse
import pymssql
import pprint as pp
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels.PhoenixModels import UNIAxioma2013MH
from riskmodels.ModelDB import SubIssue

class Asset():

    def __init__(self, id):
        self.id = id
        self.modelId = self.id
        self.subIssueId = self.modelId+"11"

#        retvals = numpy.empty((len(dateList), len(axidList)), dtype=object)
        self.exposures = {}    
    def __str__(self):
        return ' Asset(id:%s, modelId:%s, subIssueId:%s, exposures:%s)' % (
            self.id, self.modelId, self.subIssueId, self.exposures)
    def __repr__(self):
        return self.__str__()
    
class ExposureWriter():

    def __init__(self, options_, modelDB, marketDB):
        self.options = options_
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.dateStr = self.options.date
        self.date = Utilities.parseISODate(self.dateStr)
        self.log = logging.getLogger('ExposureWriter')
        self.connect()
        self.connectMarketData()
        self.assets = {}
        self.subissues = {}
        self.factors = {}
        self.exposures = {}
#        self.a = Asset('DRW78PHYJ5') # for now

    def connect(self):
        self.dbConnection = pymssql.connect(
        user=self.options.dbUser,
        password=self.options.dbPasswd,
        database='RiskResults_QA',
        host=self.options.dbHost)
        self.dbConnection.autocommit(True)
        self.dbCursor = self.dbConnection.cursor()

    def connectMarketData(self):
        self.dbConnectionMarketData = pymssql.connect(
        user=self.options.dbUserMd,
        password=self.options.dbPasswdMd,
        database='marketData',
        host=self.options.dbHostMd)
        self.dbConnectionMarketData.autocommit(True)
        self.dbCursorMarketData = self.dbConnectionMarketData.cursor()

    def getModelId(self, asset):
        self.log.info('getting modelId')
        q = """
            select securityidentifier 
            from instrumentxref where SecurityIdentifierType = 'MODELDB_ID' 
            and AxiomaDataId in (select axiomadataid from instrumentxref 
            where SecurityIdentifierType = 'InstrCode' and SecurityIdentifier = '%s')
            and '%s' between FromDate and ToDate 
        """ % (asset.id, self.date)
        self.dbCursorMarketData.execute(q)
        res = self.dbCursorMarketData.fetchall()
        for each in res:
            asset.modelId = each[0]
            asset.subIssueId = asset.modelId+"11"

    def getExposures(self):
        self.log.info('getting exposures')
        q = """
        select clientId, factorName, factorValue from RiskFactors where Date = '%s'
        """ % (self.dateStr)
        self.dbCursor.execute(q)
        res = self.dbCursor.fetchall()
        for each in res:
            if each[0] not in self.assets:
                a = Asset(each[0])
#                self.getModelId(a)
                self.assets[each[0]] = a
            self.assets[each[0]].exposures[each[1]] = each[2]
#        pp.pprint(self.assets)
        self.populateFactorsAndExposures()
        
    def populateFactorsAndExposures(self):
        for a in self.assets.values():
            print(a.subIssueId)
            if a.subIssueId is None:
                continue
            subissue = SubIssue(a.subIssueId)
            self.subissues[subissue] = 1
            for f, e in a.exposures.items():
                if e.strip() == "":
                    print('empty exposure')
                    continue
                if f not in self.factors:
                    self.factors[f] = []
                    self.exposures[f] = []
                self.factors[f].append(subissue)
                self.exposures[f].append(float(e))
#        self.factors['USD'] = []
#        self.exposures['USD'] = []
        
        pp.pprint(self.factors)
        pp.pprint(self.exposures)
        
    def insertRmsFactors(self):
        unimodel = UNIAxioma2013MH(self.modelDB, self.marketDB, list(self.factors.keys()))
        
    def writeExposures(self):
        self.log.info('writing exposures')
#        print '!!!'
#        UNIAxioma2013MH.GetFactorObjects(self.modelDB) 
#        print '!!!' 
        unimodel = UNIAxioma2013MH(self.modelDB, self.marketDB)
        unimodel.setRiskModelGroupsForDate(self.date)
        unimodel.setFactorsForDate(self.date, self.modelDB)
#        unimodel.deleteInstance(self.date, self.modelDB)
#        rmi = unimodel.createInstance(self.date, self.modelDB)
        rmi = modelDB.getRiskModelInstance(unimodel.rms_id, self.date)
        if not rmi:
            rmi = modelDB.createRiskModelInstance(unimodel.rms_id, self.date)
            rmi.setIsFinal(True, modelDB)

        self.log.info('deleting estu in modeldb')        
        modelDB.deleteEstimationUniverse(rmi)
        self.log.info('deleting rmi_universe in modeldb')        
        modelDB.deleteRiskModelUniverse(rmi)
#        rmi = modelDB.getRiskModelInstance(unimodel.rms_id, self.date)
        self.log.info('deleting exposures in modeldb')
        modelDB.deleteRMIExposureMatrix(rmi)
        print(unimodel.rms_id)
        print(self.date)
        self.subFactors = modelDB.getSubFactorsForDate(self.date, unimodel.factors)
#        self.subFactors = modelDB.getSerieSubFactorsForDate(unimodel.rms_id, self.date) 
        print(self.subFactors)      
        self.log.info('populating estu in modeldb')
        modelDB.insertEstimationUniverse(rmi, list(self.subissues.keys()))
        self.log.info('updating estu weights in modeldb')  
        modelDB.insertEstimationUniverseWeights(rmi, list(self.subissues.items()))  
        self.log.info('populating rmi_universe in modeldb')
        modelDB.insertExposureUniverse(rmi, list(self.subissues.keys()))
        self.log.info('populating sub_issue in modeldb')
        activeSubIssues = modelDB.getAllActiveSubIssues(self.date)
        toAdd = []
        for each in self.subissues.keys():
            if each not in activeSubIssues:
                toAdd.append(each)

        rmg = modelDB.getRiskModelGroup(1)
        modelDB.createNewSubIssue(toAdd, rmg, self.date)
            
        self.log.info('storing exposures to modeldb')

        for sf in self.subFactors:
            f = sf.factor.name.upper()
            if f in self.factors:
                print('found', f)
                modelDB.insertFactorExposures(rmi, sf, self.factors[f], numpy.array(self.exposures[f]), f)
                    
    def showUnresolvedFactors(self):
        modelList = []
        resolved = []
        unresolved = []
        for sf in self.subFactors:
                modelList.append(sf.factor.name.upper())
        for f in self.factors.keys():
            if f in modelList:
                resolved.append(f)
            else:
                unresolved.append(f)
        print('Unresolved factors: %s' % unresolved)
        print('Resolved factors: %s' % resolved)

            
        
if __name__=='__main__':
    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--db-user", action="store",
                             default='risk_user', dest="dbUser",
                             help="SQL Server username")
    cmdlineParser.add_option("--db-passwd", action="store",
                             default='ru1234', dest="dbPasswd",
                             help="SQL Server password")
    cmdlineParser.add_option("--db-host", action="store",
                             default='dev-mac-db', dest="dbHost",
                             help="SQL Server host")
    cmdlineParser.add_option("--db-user-md", action="store",
                             default='MarketDataLoader', dest="dbUserMd",
                             help="SQL Server md username")
    cmdlineParser.add_option("--db-passwd-md", action="store",
                             default='mdl1234', dest="dbPasswdMd",
                             help="SQL Server md password")
    cmdlineParser.add_option("--db-host-md", action="store",
                             default='prod-mac-mkt-db', dest="dbHostMd",
                             help="SQL Server md host")
    cmdlineParser.add_option("--update-database",action="store_false",
                             default=False,dest="testOnly",
                             help="change the database")
    cmdlineParser.add_option("--date", action="store",
                         default='2013-12-04', dest="date",
                         help="date")
    (options_, args_) = cmdlineParser.parse_args()
    logging.config.fileConfig('log.config')
    modelDB=ModelDB.ModelDB(user=options_.modelDBUser,passwd=options_.modelDBPasswd,sid=options_.modelDBSID)
    marketDB=MarketDB.MarketDB(user=options_.marketDBUser,passwd=options_.marketDBPasswd,sid=options_.marketDBSID)
    writer = ExposureWriter(options_, modelDB, marketDB)
    writer.getExposures()
#    writer.insertRmsFactors()
    writer.writeExposures()
    writer.showUnresolvedFactors()
                
    if not options_.testOnly:
        logging.info("Committing changes")
        modelDB.commitChanges()
    else:
        logging.info("Reverting changes")
        modelDB.revertChanges()
    marketDB.revertChanges()
    modelDB.finalize()
    marketDB.finalize()
 
 
 

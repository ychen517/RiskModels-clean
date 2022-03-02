
import numpy
import logging
import datetime
import optparse
import pymssql
import pprint as pp
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels.PhoenixModels import UNIAxioma2013MH
from riskmodels.ModelDB import SubIssue
from riskmodels import Utilities

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
    
class VarCovarWriter():

    def __init__(self, options_, modelDB, marketDB):
        self.options = options_
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.dateStr = self.options.date
        self.date = Utilities.parseISODate(self.dateStr)
        self.log = logging.getLogger('CovarWriter')
        self.connect()
        self.connectMarketData()
        self.varCovar = {}
        self.factors = {}
#        self.a = Asset('DRW78PHYJ5') # for now

    def connect(self):
        self.dbConnection = pymssql.connect(
        user=self.options.dbUser,
        password=self.options.dbPasswd,
        database='validation',
        host=self.options.dbHost)
        self.dbConnection.autocommit(True)
        self.dbCursor = self.dbConnection.cursor()

    def connectMarketData(self):
        self.dbConnectionMarketData = pymssql.connect(
        user=self.options.dbUserMd,
        password=self.options.dbPasswdMd,
        database='marketData',
        host=self.options.dbHostMde)
        self.dbConnectionMarketData.autocommit(True)
        self.dbCursorMarketData = self.dbConnectionMarketData.cursor()

    def getVarCovar(self):
        self.log.info('getting varCovar')
        q = """
        select FactorName1, FactorName2, Value from Riskfactorsvarcovar where Date = '%s'
        """ % (self.dateStr)
        self.dbCursor.execute(q)
        res = self.dbCursor.fetchall()
        for each in res:
            f1 = each[0].upper()
            f2 = each[1].upper()
            v = float(each[2])
            if f1 not in self.varCovar:
                self.varCovar[f1] = {}
            self.varCovar[f1][f2] = v
        pp.pprint(self.varCovar)
        
    def populateFactorsAndExposures(self):
        for a in self.assets.values():
            if a.subIssueId is None:
                continue
            for f, e in a.exposures.items():
                if e.strip() == "":
                    print('empty exposure')
                    continue
                if f not in self.factors:
                    self.factors[f] = []
                    self.exposures[f] = []
                self.factors[f].append(SubIssue(a.subIssueId))
                self.exposures[f].append(float(e))
        pp.pprint(self.factors)
        pp.pprint(self.exposures)
        
    def writeVarCovar(self):
        self.log.info('writing varCovar')
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

#        rmi = modelDB.getRiskModelInstance(unimodel.rms_id, self.date)
        self.log.info('deleting covMatrix in modeldb')
        modelDB. deleteRMIFactorCovMatrix (rmi)
        print(unimodel.rms_id)
        print(self.date)
        self.subFactors = modelDB.getSubFactorsForDate(self.date, unimodel.factors)
        self.log.info('storing covMatrix to modeldb')
        sfDict = {}
        i = 0
        for sf in self.subFactors:
            fields = sf.factor.description.split('|')
            if len(fields) == 1:
                f = fields[0].strip().upper()
                if f in self.varCovar:
                    self.factors[f] = sf
                    sfDict[sf] = i
                    i += 1
        print(self.factors)
        print(sfDict)
        mat = numpy.zeros((len(self.factors), len(self.factors)), float)
        for f1 in self.factors:
            for f2, v in self.varCovar[f1].items():
                if not f2 in self.factors:
                    continue
                mat[sfDict[self.factors[f1]], sfDict[self.factors[f2]]] = v
                mat[sfDict[self.factors[f2]], sfDict[self.factors[f1]]] = v# 
        unimodel.insertFactorCovariances(rmi,
                        mat, list(sfDict.keys()), self.modelDB)

                    
    def showUnresolvedFactors(self):
        modelList = []
        resolved = []
        unresolved = []
        for sf in self.subFactors:
            fields = sf.factor.description.split('|')
            if len(fields) == 1:
                f = fields[0].strip().upper()
                modelList.append(f)
            else:
                print('!!!', fields)
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
                             default='tqa_user', dest="dbUser",
                             help="SQL Server username")
    cmdlineParser.add_option("--db-passwd", action="store",
                             default='tqa_user', dest="dbPasswd",
                             help="SQL Server password")
    cmdlineParser.add_option("--db-host", action="store",
                             default='prod-vndr-db', dest="dbHost",
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
    writer = VarCovarWriter(options_, modelDB, marketDB)
    writer.getVarCovar()
    writer.writeVarCovar()
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
 
 
 

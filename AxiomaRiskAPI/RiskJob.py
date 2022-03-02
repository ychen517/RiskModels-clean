
import json
import os
from suds.client import Client
from suds.wsse import *
import dateutil.parser
import datetime
import time
import logging
from suds.sax.element import Element
import logging.config
from subprocess import Popen, PIPE
import csv

class Timer(object):
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start

class Result():
    def __init__(self, success, errorMsg = '', body = None): 
        self.success = success
        self.errorMsg = errorMsg
        self.body = body
                    
class Position():
    def __init__(self, riskJob, qty):
        self.riskJob = riskJob
        self.qty = qty
        self.covered = False
        self.measures = {}
        self.factors = {}
        self.id = None
        self.idType = None
        self.ccy = None
        self.qtyScale = None
        self.errors = []
        
    def __str__(self):
        return ' Position(id:%s, qty:%s, covered:%s, measures:%s, factors:%s)' % (
            self.id, self.qty, self.covered, self.measures, self.factors)
    def __repr__(self):
        return self.__str__()
    
class RiskJob():
    
    def __init__(self, mgr):
        self.mgr = mgr
        self.logger = logging.getLogger()
        self.configData = json.load(open(mgr.configName))
        self.clearStats()
        with Timer() as t:
            self.client = self.getClient()
        self.stats['getClient'] = '%.2f' % t.secs
        self.reportFormat = self.client.factory.create('ReportFileRequirements')
        self.reportFormat.CSV = True
        self.logger.info('connected')

    def clearStats(self):
        self.jobId = None
        self.stats = {}
        self.status = []
        self.results = []
        self.errors = []
        self.pollCount = 0
        self.posDict = {}
        self.initStore()
        self.url = self.configData["connect"]["url"]
        self.user = self.configData["connect"]["user"]
        self.password = self.configData["connect"]["password"]
        
    def initStore(self):
        self.store = {}
        self.store["base"] = self.configData["store"]["base"]
        if not os.path.exists(self.store["base"]):
            os.makedirs(self.store["base"])        
        if self.jobId is not None:
            self.store["base"] = os.path.join(self.store["base"], str(self.jobId))
        if not os.path.exists(self.store["base"]):
            os.makedirs(self.store["base"])
        self.store["stats"] = os.path.join(self.store["base"], "stats.txt")
        self.store["errors"] = os.path.join(self.store["base"], "errors.txt")
        self.store["results"] = os.path.join(self.store["base"], "results.txt")
        self.store["reportFile"] = os.path.join(self.store["base"], "reportFile.txt")    
                
    def getClient(self):
        soap_url = self.url
        wsdl_url = '%s?singleWsdl' % soap_url
        self.logger.info('%s', wsdl_url)
        client =  Client(wsdl_url, timeout=3000)
        ns = ('ns', "http://axioma.com")
        userid = Element('axUsername', ns=ns).setText(self.user)
        password = Element('axPassword', ns=ns).setText(self.password)
        client.set_options(soapheaders=(userid,password))
#        print client
        return client

    def validateParams(self, params):
        result = Result(True, 'some error message')
        return result
        
    def submit(self, params):
        self.logger.info(params)
        result = self.validateParams(params)
        if not result.success:
            return result
        self.identifiers = params['identifiers']
        self.portfolio = params['portfolio']
        self.template = params['template']
        self.date = datetime.datetime.strptime(params['date'], '%Y-%m-%d')
        self.stats['startTime'] = str(datetime.datetime.now())
        self.removePositions = params['removePositions']
        if self.removePositions.lower() == 'yes':
            self.logger.info('removing existing positions on %s', self.date)
            with Timer() as t:
                self.removeExistingPositions()
            self.stats['removeExistingPositions'] = '%.2f' % t.secs
        self.logger.info('importing new positions on %s', self.date)
        with Timer() as t:
            self.importPositions()
        self.stats['importPositions'] = '%.2f' % t.secs
        self.logger.info('submitting report job on %s', self.date)
        with Timer() as t:
            self.submitReportJob()
        self.stats['submitReportJob'] = '%.2f' % t.secs
        self.stats['submitTime'] = str(datetime.datetime.now())
        result.body = self.jobId
        return result

    def removeExistingPositions(self):
        result = self.client.service.RemovePositions(accountId = self.portfolio, date = self.date)
        self.logger.info('result of removing positions is  ' + str(result))
    
    def importPositions(self):
        self.positions = self.client.factory.create('Positions')
        self.addPosiitons()
        result = self.client.service.ImportPositions(accountId = self.portfolio, date = self.date, 
                    positions = self.positions, replacePostions = True)
        self.logger.info('result of importing positions is  ' + str(result))

    def addPosiitons(self):
        for id in self.identifiers:
            qty = 100
            p = Position(self, qty)
            p.id = id
            p.idType = 'modeldb_id'
            p.ccy = 'USD' # revise
            p.qtyScale = 'NotionalValue' #revise
            self.addPosition(p)

    def addPosition(self, p):
        pos = self.client.factory.create('Position')
        pos.DataId = p.id
        pos.InstrumentLocator = self.client.factory.create('SecurityInstrumentLocator')
        pos.InstrumentLocator.SecurityRef.LookupItem.append('%s=%s' % (p.idType, p.id))
        pos.InstrumentQuantity = self.client.factory.create('InstrumentQuantity')
        pos.InstrumentQuantity.Quantity = p.qty
        pos.InstrumentQuantity.QuantityScale = p.qtyScale
        pos.InstrumentQuantity.QuantityCurrency.LookupItem.append(p.ccy)
        self.logger.info('importing %s' % p)
        self.positions.Position.append(pos)
        self.posDict[pos.DataId] = p

    def submitReportJob(self):
        reportJob = self.client.factory.create('ReportJob')
        reportJob.AnalysisDate = self.date
        reportJob.AccountId = self.portfolio
        reportJob.RecomputeResultsOnUnmodifiedPositions = True                
        reportJob.ReportTemplateId = self.template
        self.jobId = str(self.client.service.QueueReportJob(reportJob))
        self.logger.info('jobId is  ' + self.jobId)

    def collectErrors(self, result):
        for each in  result.LogFile.LogEntry:
            self.logger.info('logEntry %s' % each)
            if each.__class__.__name__ in ('Error', 'Warning'):
                if each.Originator:
                    id = each.Originator.Id
                    if id in self.posDict:
                        p = self.posDict[id]
                        p.errors.append(each.Message)
                        self.errors.append(each.Message)
                else:
                    self.errors.append(each.Message)

    def collectResults(self, result): 
            fileName = self.store["reportFile"]
            result =  result.ZippedReportFiles.decode('base64')
            f = open('%s.gz' % fileName,'w')
            f.write(result)
            f.close()
            
            cmd = 'rm %s;gunzip %s.gz' % (fileName, fileName)
            p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
            out, err = p.communicate() 
            
            f = open(fileName, "r")
            reader = csv.reader(f)
            for row in reader:
                self.results.append(row)
       
    def reportStatus(self):
        self.logger.info('GetReportJobStatus for %s' % self.jobId)
        try:
            result = self.client.service.GetReportJobResults(reportJobId = self.jobId, reportFormats = self.reportFormat)
            res = {}
            if result.ComputationRiskJobIds is None:
                res['job'] = None
            else:
                res['job'] = str(result.ComputationRiskJobIds[0][0])
            res['time'] = str(datetime.datetime.now())
            res['progress'] = result.ComputationProgressPercent
            res['state'] = result.FinalState
            self.status.append(res)
            self.pollCount += 1
            self.stats["pollCount"] = str(self.pollCount)
            self.writeStats()  
            if res['state'] is not None:                
                self.collectErrors(result)                
                self.writeErrors() 
                self.collectResults(result)                   
                self.writeResults()  
            return res
        except Exception as e:
            self.logger.warn('reportStatus exception %s', e)
            time.sleep(5)
            return self.reportStatus()   
        
    def reportCompletion(self): 
        res = self.reportStatus()
        while res['state'] is None:
            self.logger.info('poll result: %s', res)
            self.logger.info('sleeping for 5 sec')
            time.sleep(5)
            res = self.reportStatus()
                    
    def writeStats(self):
        self.initStore()
        with open(self.store["stats"], 'w') as outfile:
          json.dump(self.stats, outfile, indent=4, sort_keys=True) 

    def writeErrors(self):
        self.initStore()
        with open(self.store["errors"], 'w') as outfile:
          json.dump(self.errors, outfile, indent=4, sort_keys=True)                 

    def writeResults(self):
        self.initStore()
        with open(self.store["results"], 'w') as outfile:
          json.dump(self.results, outfile, indent=4)                 

    def reportStats(self): 
        with open(self.store["stats"]) as data:
            return json.load(data)

    def reportErrors(self):     
        with open(self.store["errors"]) as data:
            return json.load(data)
        
    def reportResults(self):     
        with open(self.store["results"]) as data:
            return json.load(data)

if __name__=='__main__':
    logging.config.fileConfig('log.config')   

    #logging.basicConfig(level=logging.INFO)
    #logging.getLogger('suds.client').setLevel(logging.DEBUG)
    #logging.getLogger('suds.wsdl').setLevel(logging.DEBUG)
    #logging.getLogger('suds.wsse').setLevel(logging.DEBUG)

    rj = RiskJob('config.json')

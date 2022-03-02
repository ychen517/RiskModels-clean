
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

# logging.config.fileConfig('log.config')   

#logging.basicConfig(level=logging.INFO)
#logging.getLogger('suds.client').setLevel(logging.DEBUG)
#logging.getLogger('suds.wsdl').setLevel(logging.DEBUG)
#logging.getLogger('suds.wsse').setLevel(logging.DEBUG)

def toDateTime(date):
    if isinstance(date, datetime.date):
        date = datetime.datetime.combine(date, datetime.time())
    return date

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
    
    def __init__(self, configuration):
        self.logger = logging.getLogger()
        self.configData = configuration
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

    def submit(self, portfolio, date, template, identifiers, removePositions=False, recomputeResults=False):
        self.logger.info('Processing portfolio=%s, dateTime=%s, template=%s, removePositions=%s',
                         portfolio, date, template, removePositions)
        self.logger.debug('Requested identifiers=%s', identifiers)
        self.stats['startTime'] = str(datetime.datetime.now())
        if removePositions:
            self.logger.info('removing existing positions on %s', date)
            with Timer() as t:
                self.removeExistingPositions(portfolio, date)
            self.stats['removeExistingPositions'] = '%.2f' % t.secs
        self.logger.info('importing new positions on %s', date)
        with Timer() as t:
            self.importPositions(portfolio, date, identifiers)
        self.stats['importPositions'] = '%.2f' % t.secs
        self.logger.info('submitting report job on %s', date)
        with Timer() as t:
            result = self.submitReportJob(portfolio, date, template, recomputeResults)
        self.stats['submitReportJob'] = '%.2f' % t.secs
        self.stats['submitTime'] = str(datetime.datetime.now())
        return result

    def removeExistingPositions(self, portfolio, date):
        result = self.client.service.RemovePositions(accountId = portfolio, date = toDateTime(date))
        self.logger.info('removing positions complete')
        self.logger.debug('result of removing positions is  ' + str(result))
    
    def importPositions(self, portfolio, date, identifiers):
        positions = self.client.factory.create('Positions')
        self.addPositions(positions, identifiers)
        self.logger.info('created position list')
        result = self.client.service.ImportPositions(accountId = portfolio, date = toDateTime(date),
                    positions = positions, replacePostions = True)
        self.logger.info('importing positions complete')
        self.logger.debug('result of importing positions is  ' + str(result))

    def addPositions(self, positions, identifiers):
        self.posDict = {}
        for (idType, id) in identifiers:
            qty = 1
            p = Position(self, qty)
            p.id = id
            p.idType = idType
            p.ccy = 'USD' # revise
            p.qtyScale = 'MarketValue' #revise
            pos = self.createPosition(p)
            #self.logger.info('importing %s' % p)
            positions.Position.append(pos)
            self.posDict[pos.DataId] = p

    def createPosition(self, p):
        pos = self.client.factory.create('Position')
        pos.DataId = p.id
        pos.InstrumentLocator = self.client.factory.create('SecurityInstrumentLocator')
        pos.InstrumentLocator.SecurityRef.LookupItem.append('%s=%s' % (p.idType, p.id))
        pos.InstrumentQuantity = self.client.factory.create('InstrumentQuantity')
        pos.InstrumentQuantity.Quantity = p.qty
        pos.InstrumentQuantity.QuantityScale = p.qtyScale
        pos.InstrumentQuantity.QuantityCurrency.LookupItem.append(p.ccy)
        return pos

    def submitReportJob(self, portfolio, date, template, recomputeResults=False):
        reportJob = self.client.factory.create('ReportJob')
        reportJob.AnalysisDate = toDateTime(date)
        reportJob.AccountId = portfolio
        reportJob.RecomputeResultsOnUnmodifiedPositions = recomputeResults
        self.logger.info('Requesting job with RecomputeResultsOnUnmodifiedPositions=%s', recomputeResults)
        reportJob.DataPartition = None                  
        reportJob.ReportTemplateId = template 
        self.clearStats()
        self.jobId = str(self.client.service.QueueReportJob(reportJob))
        self.logger.info('jobId is  ' + self.jobId)
        result = Result(True, 'complete')
        result.body = self.jobId
        return result
    
    def collectErrors(self, result):
        for each in  result.LogFile.LogEntry:
            #self.logger.info('logEntry %s' % each)
            if each.__class__.__name__ in ('Error', 'Warning'):
                if each.Originator:
                    id = each.Originator.Id
                    if id in self.posDict:
                        p = self.posDict[id]
                        p.errors.append(each.Message)
                        self.errors.append(each.Originator.Id + ': ' + each.Message)
                else:
                    self.errors.append(each.__class__.__name__ + ": " + each.Message)

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
            if res['state'] is not None and res['state'] != 'Failed':
                self.collectResults(result)                   
                self.writeResults()  
            return res
        except Exception as e:
            self.logger.warning('reportStatus exception %s', e, exc_info=True)
            time.sleep(5)
            return self.reportStatus()   
        
    def reportCompletion(self): 
        res = self.reportStatus()
        delay = 10
        while res['state'] is None:
            self.logger.info('poll result: %s', res)
            self.logger.info('sleeping for %d sec', delay)
            time.sleep(delay)
            res = self.reportStatus()
        return res
                    
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
        
class RiskJobRunner():
    
    def __init__(self):
        self.logger = logging.getLogger()
        self.rj = RiskJob(json.load(open('config.json')))
        
    def run(self, date, dataIds):
        return self.rj.reportResults()
        pf = 'SHS_ModelGeneration'
        view = 'SHS_Exposures'
        idLookups = [('AxiomaDataId', dataId) for dataId in dataIds]
        result = self.rj.submit(pf, date, view, idLookups,
                        removePositions=True, recomputeResults=True)
        jobId = result.body
        logging.info('submitted exposure request, job id [%s]', jobId)
        res = self.rj.reportCompletion()
        if res['state'] is not None and res['state'] != 'Failed':
            return self.rj.reportResults()
        else:
            return None


if __name__=='__main__':
    runner = RiskJobRunner()
#     dataIds = ['D7KBAUV7M9']
#     dataIds = ['30333']
    dataIds = ['9']
    runner.run(datetime.date(2014, 5, 30), dataIds)



import logging
from RiskJob import *

class RiskJobManager:
    def __init__(self, configName):
        self.configName = configName
        self.pool = []
        self.submitted = {}
        self.completed = {}
        self.logger = logging.getLogger()    
        configData = json.load(open(self.configName))
                      
    def fromPool(self):
        if len(self.pool) > 0:
            riskJob = self.pool.pop()
            riskJob.clearStats()
            return riskJob
        else:
            return RiskJob(self)

    def submittedJob(self, jobId):
        if jobId not in self.submitted.keys():
            riskJob = self.fromPool()
            riskJob.jobId = jobId
            self.submitted[jobId] = riskJob
        return self.submitted[jobId]           


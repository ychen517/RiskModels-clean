
import requests
import json
import pprint as pp
import logging
import logging.config
import time

class RiskClient():
    
    def __init__(self):
        self.hostPort = 'http://vm2-criollo:9090'
#        self.hostPort = 'http://moon:9090'
        self.headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        
    def submitJob(self):
        params = {'identifiers': ['D12GMXD568'], \
#        params = {'identifiers': ['D12GMXD568', 'D13LHCDX71', 'DQ8L3JXGU1', 'DH2CMWN4M0'], \
#        params = {'identifiers': [], \
          'portfolio' : 'Regression Tool Portfolio', \
          'template' : 'Regression Tool View', \
#          'template' : 'CovarianceMatrix', \
          'removePositions' : 'no', \
          'date' : '2014-01-08'
          }
        print(json.dumps(params))
        r = requests.post('%s/submitJob' % self.hostPort, data=json.dumps(params), headers=self.headers)
        logger.info(json.loads(r.text))
        return json.loads(r.text)['jobId']
    
    def checkStatus(self, jobId):
        r = requests.get('%s/%s/jobStatus' % (self.hostPort, jobId), headers=self.headers)
        print(r.text)
        return json.loads(r.text)

    def checkCompletion(self, jobId):
        r = requests.get('%s/%s/jobCompletion' % (self.hostPort, jobId), headers=self.headers)
        print(r.text)
        return json.loads(r.text)

    def checkResults(self, jobId):
        r = requests.get('%s/%s/jobResults' % (self.hostPort, jobId), headers=self.headers)
        return json.loads(r.text)
    

    def checkStats(self, jobId):
        
        r = requests.get('%s/%s/jobStats' % (self.hostPort, jobId), headers=self.headers)
        return json.loads(r.text)

    def checkErrors(self, jobId):
        r = requests.get('%s/%s/jobErrors' % (self.hostPort, jobId), headers=self.headers)
        return json.loads(r.text)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger('RiskClient')   

    c = RiskClient()
    jobId = c.submitJob()
    print('jobId: ', jobId)

    res = c.checkCompletion(jobId)
    pp.pprint(res)

    #res = c.checkStatus(jobId)
    #pp.pprint(res)
    #
    #while res['state'] is None:
    #    print 'poll result:'
    #    pp.pprint(res)
    #    print 'sleeping for 5 sec'
    #    time.sleep(5)
    #    res = c.checkStatus(jobId)
    #print 'poll result:'
    #pp.pprint(res)

    res = c.checkResults(jobId)
    print('job results:')
    pp.pprint(res)

    res = c.checkStats(jobId)
    print('job stats:')
    pp.pprint(res)

    res = c.checkErrors(jobId)
    print('job errors:')
    pp.pprint(res)



    #
    #ids = {'aaa' : 'bbb'}
    #
    #headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    #r = requests.put('%sids' % hostPort, data=json.dumps(ids), headers=headers)
    ##print(json.dumps(r.text, sort_keys=True, indent=4 * ' '))
    #print r.text
    #
    #id = '11111'
    #r = requests.get('%sjobResult/%s' % (hostPort, id))
    #res = json.loads(r.text)
    #print json.dumps(res, indent = 4)

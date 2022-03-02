
import json
import os
from bottle import route, run, static_file, request, put
from RiskJob import *
from RiskJobManager import *
import logging
import socket

#class Result(object):
#    def __init__(self, errors, report):
#        self.errors = errors
#        self.report = report
    
class riskAPI:
    jobManager = RiskJobManager('config.json')
 
    @route('/submitJob', method='POST')
    def submitJob():
        params = json.loads(request.body.readline())
        logging.info(params)
        riskJob = riskAPI.jobManager.fromPool()
        result = riskJob.submit(params)
        riskAPI.jobManager.submitted[result.body] = riskJob
        return {'jobId' : result.body}

    @route('/<id>/jobStatus', method='GET')
    def reportStatus(id):
        riskJob = riskAPI.jobManager.submittedJob(id)
        res =  riskJob.reportStatus()
        return json.dumps(res)

    @route('/<id>/jobCompletion', method='GET')
    def reportCompletion(id):
        riskJob = riskAPI.jobManager.submittedJob(id)
        res =  riskJob.reportCompletion()
        return json.dumps(res)

    @route('/<id>/jobResults', method='GET')
    def reportResults(id):
        riskJob = riskAPI.jobManager.submittedJob(id)
        return  json.dumps(riskJob.reportResults())

    @route('/<id>/jobStats', method='GET')
    def reportStats(id):
        riskJob = riskAPI.jobManager.submittedJob(id)
        return  json.dumps(riskJob.reportStats())

    @route('/<id>/jobErrors', method='GET')
    def reportErrors(id):
        riskJob = riskAPI.jobManager.submittedJob(id)
        return  json.dumps(riskJob.reportErrors())
    
    @route('/test', method='GET')
    def test():
        f = open('/tmp/test.html')
        return  f.read()

if __name__ == '__main__':
    run(server='paste', host=socket.gethostname(), port=9090, debug=True, reloader=True)
    #run(host='vm2-criollo', port=9090, debug=True, reloader=True)

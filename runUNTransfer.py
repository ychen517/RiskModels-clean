
import optparse
import configparser
import logging
import logging.config
import sys

import riskmodels
from riskmodels import Utilities
from riskmodels import ModelDB
from riskmodels.Connections import createConnections, finalizeConnections


if __name__ == '__main__':
    usage = "usage: %prog [options] config-file axiomaIDlist fromdt thrudt section"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()

    connections = createConnections(config_)
    axiomaIdList=args_[1].split(',')
    startDate=Utilities.parseISODate(args_[2])
    endDate=Utilities.parseISODate(args_[3])
    endDate=args_[3]
    dates='%s:%s' % (startDate, endDate)
    section=args_[4]
    if options_.testOnly:
        commitValue=False
    else:
        commitValue=True
    
    if section=='UNModelReturn':
        # hard code the call to deal with the FI models and UN models
        startDate=Utilities.parseISODate(dates.split(':')[0])
        for modelName in ('FIAxioma2014MH', 'FIAxioma2014MH1', 'Univ10AxiomaMH'):
            riskModelClass = riskmodels.getModelByName(modelName)
            riskModel = riskModelClass(connections.modelDB, connections.marketDB)
            riskModel.rmg = connections.modelDB.getAllRiskModelGroups()
            dateList = connections.modelDB.getDateRange(riskModel.rmg, startDate, endDate, True)
            for date in dateList:
                rmi = connections.modelDB.getRiskModelInstance(riskModel.rms_id, date)
                if not rmi:
                    rmi=ModelDB.RiskModelInstance(date=date, rms_id=riskModel.rms_id)
                logging.info('Transfer returns for %s %s on %s', modelName, axiomaIdList,date)
                riskModel.transferReturnsforAxIDList(date, rmi, connections.modelDB, connections.marketDB, axiomaIdList)

    if options_.testOnly:
        logging.info("Reverting changes")
        connections.modelDB.revertChanges()
    else:
        logging.info("Committing changes")
        connections.modelDB.commitChanges()
    finalizeConnections(connections)
    sys.exit(0)

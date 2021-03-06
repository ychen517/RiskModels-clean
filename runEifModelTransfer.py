# $Id: runEifModelTransfer.py 236587 2019-12-09 16:46:17Z rgoodman $

import optparse
import logging
import configparser
import sys
import datetime
from riskmodels import Utilities
from riskmodels.Connections import createConnections, finalizeConnections
from riskmodels.transfer import transferSubIssueData

if __name__ == '__main__':
    usage = "usage: %prog [options] config-file countries iso-date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("-d", action="store",
                             default=5, dest="numDays", type="int",
                             help="num Days to go back")

    (options_, args_) = cmdlineParser.parse_args()

    if len(args_) < 3:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    countries=args_[1]
    date=Utilities.parseISODate(args_[2])
    # create a config object based on the config file    
    logging.info('testOnly option (-n) is %s', options_.testOnly)
    connections_ = createConnections(config_)

    errorCase=False
    try:
        # find the last N+2 days; let the individual sections decide whether to ignore trading days
        prevdates=sorted(list(date - datetime.timedelta(i) for i in range(options_.numDays+2)))
        datestr=','.join([str(dd) for dd in prevdates])   
        # the sections to process are very specific for EIFs.  Set them up here and then set up the 
        # dates as needed
        for ctry in countries.split(','):
            config_.set('DEFAULT','sub-issue-ids',ctry)
            config_.set('DEFAULT','asset-type','futures')
            for section in ('SubIssueData','SubIssueReturn','SubIssueCumulativeReturn'):
                config_.set('DEFAULT','dates',datestr)
                transfer = config_.get(section, 'transfer')
                proc = eval('%s(config_, section, connections_, options_)' % transfer)
                
    except:
        logging.exception("Error transferring EIF Model Data")
        errorCase = True
    
    if options_.testOnly or errorCase:
        logging.info('Reverting changes')
        connections_.modelDB.revertChanges()
    else:
        logging.info('Committing changes')
        connections_.modelDB.commitChanges()
        
    finalizeConnections(connections_)
    if errorCase:
        sys.exit(1)

    
    

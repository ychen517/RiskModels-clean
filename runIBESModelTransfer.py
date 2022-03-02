
import optparse
import logging
import configparser
import sys
import datetime
from riskmodels import Utilities
from riskmodels.Connections import createConnections, finalizeConnections
from riskmodels.transfer import transferEstimateData

if __name__ == '__main__':
    usage = "usage: %prog [options] config-file subissue-ids iso-date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("-d", action="store",
                             default=30, dest="numDays", type="int",
                             help="num Days to go back")

    (options_, args_) = cmdlineParser.parse_args()

    if len(args_) < 3:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    subissues=args_[1]
    date=Utilities.parseISODate(args_[2])
    # create a config object based on the config file    
    logging.info('testOnly option (-n) is %s', options_.testOnly)
    connections_ = createConnections(config_)

    errorCase=False
    try:
        # just go back to N days from today
        for ctry in subissues.split(','):
            datestr='%s:%s' % (str(date+datetime.timedelta(days=-options_.numDays)),str(date))
        
            config_.set('DEFAULT','sub-issue-ids',ctry)
            for section in [ 'AssetEstimateData', 'EstimateCurrencyData']:
                config_.set('DEFAULT','dates',datestr)
                transfer = config_.get(section, 'transfer')
                proc = eval('%s(config_, section, connections_, options_)' % transfer)
                
    except:
        logging.exception("Error transferring IBES Model estimate Data")
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

    
    


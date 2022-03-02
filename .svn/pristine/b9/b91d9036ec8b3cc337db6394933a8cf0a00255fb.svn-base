import configparser
import logging
import optparse
import sys
from marketdb import Utilities
from riskmodels import Connections
from riskmodels.transfer import *

if __name__ == '__main__':
    usage = "usage: %prog [options] config-file [[section:]option=value ...]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--dw", action="store_true",
                             default=False, dest="dontWrite",
                             help="don't even attempt to write to the database")
    cmdlineParser.add_option("-f", action="store_true",
                             default=False, dest="override",
                             help="override certain aspects")
    cmdlineParser.add_option("-d", "-c", "--cleanup", action="store_true",
                             default=False, dest="cleanup",
                             help="set desc columns to null for all sub-issue-ids and dts before computation")
    cmdlineParser.add_option("--nuke", action="store_true",
                             default=False, dest="nuke",
                             help="set desc columns to null for all descriptor_exposure_CUR records before computation")
    cmdlineParser.add_option("--expand", "-x", action="store_true",
                             default=False, dest="expand",
                             help="Compute expanded set of home betas for a market")
    cmdlineParser.add_option("--traded-only", "--trd", action="store_true",
                             default=False, dest="tradedAssetsOnly",
                             help="Force descriptor code to load only assets traded on particular market")
    cmdlineParser.add_option("--not-in-models", action="store_true",
                             default=False, dest="notInRiskModels",
                             help="flag to allow transfer of sub-issues not mapped to a model")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="Extra debugging output")
    cmdlineParser.add_option("--track", action="store",
                             default=None, dest="trackID",
                             help="Sub-issue-id for tracking")

    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()

    # process command-line options to override config file
    for arg in args_[1:]:
        fields = arg.split('=')
        if len(fields) != 2:
            logging.fatal('Incorrect command-line assignment "%s"'
                          ', exactly one "=" is required', arg)
            sys.exit(1)
        (option, value) = fields
        section = 'DEFAULT'
        if option.find(':') != -1:
            fields = option.split(':')
            if len(fields) != 2:
                logging.fatal('Incorrect command-line assignment "%s"'
                              ', at most one ":" is allowed in option "%s"',
                              arg, option)
                sys.exit(1)
            (section, option) = fields
        if section != 'DEFAULT' and not config_.has_section(section):
            logging.fatal('Section "%s" is not present in config file.',
                          section)
            sys.exit(1)
        config_.set(section, option, value)
    connections_ = Connections.createConnections(config_)
    errorCase = False
    # process selected sections
    try:
        for section in config_.get('DEFAULT', 'sections').split(','):
            section = section.strip()
            logging.info('going to transfer section: %s'% section)
            if config_.has_section(section):
                transfer = config_.get(section, 'transfer')
                proc = eval('%s(config_, section, connections_, options_)'% transfer)
            else:
                logging.error('No "%s" section in config file. Skipping'
                              % section)
    except Exception as e:
        logging.fatal('Exception during processing. Reverting all changes',
                      exc_info=True)
        connections_.modelDB.revertChanges()
        errorCase = True
    if options_.testOnly:
        logging.info('Reverting changes')
        connections_.modelDB.revertChanges()
    else:
        connections_.modelDB.commitChanges()
    Connections.finalizeConnections(connections_)
    if errorCase:
        sys.exit(1)

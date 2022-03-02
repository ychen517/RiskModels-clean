# Synchronize the cumulative return levels between two database instances
# on a given day so that cumulative returns from that day on can be used
# with either history.
# Updates sub_issue_cumulative_return, rms_factor_return,
# and currency_risk_free_rate.


import configparser
import optparse

from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels.synccumulativereturns import *


def main():
    usage_ = "usage: %prog [options] config-file"
    cmdlineParser_ = optparse.OptionParser(usage=usage_)
    cmdlineParser_.add_option("-l", "--log-config", action="store",
                             default='log.config', dest="logConfigFile",
                             help="logging configuration file")
    cmdlineParser_.add_option("-n", action="store_true",
                              default=False, dest="testOnly",
                              help="don't change the database")
    (options_, args_) = cmdlineParser_.parse_args()
    if len(args_) != 1:
        cmdlineParser_.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser_)
    
    p_ = configparser.ConfigParser()
    p_.read(args_[0])
    sourceMdlUser_ = p_.get('source', 'user')
    sourceMdlPasswd_ = p_.get('source', 'passwd')
    sourceMdlSID_ = p_.get('source', 'sid')
    sourceDB_ = ModelDB.ModelDB(user=sourceMdlUser_, passwd=sourceMdlPasswd_,
                                sid=sourceMdlSID_)
    
    targetMdlUser_ = p_.get('target', 'user')
    targetMdlPasswd_ = p_.get('target', 'passwd')
    targetMdlSID_ = p_.get('target', 'sid')
    targetDB_ = ModelDB.ModelDB(user=targetMdlUser_, passwd=targetMdlPasswd_,
                               sid=targetMdlSID_)
    
    date_ = Utilities.parseISODate(p_.get('general', 'date'))
    logging.info('Synchronizing cumulative returns on %s', date_)
    if p_.has_section('sub-issue-cumulative-return'):
        syncAssetReturns(sourceDB_, targetDB_, date_, p_)
    if p_.has_section('cumulative-risk-free-rate'):
        syncCurrencyReturns(sourceDB_, targetDB_, date_, p_)
    if p_.has_section('factor-cumulative-return'):
        syncFactorReturns(sourceDB_, targetDB_, date_, p_)
    
    if options_.testOnly:
        logging.info('Reverting changes')
        targetDB_.revertChanges()
    else:
        logging.info('Commiting changes')
        targetDB_.commitChanges()
    
    sourceDB_.revertChanges()
    targetDB_.finalize()
    sourceDB_.finalize()

if __name__ == '__main__':
    main()

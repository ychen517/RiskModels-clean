

import configparser
import logging
import optparse

from riskmodels import Connections
from riskmodels import transfer
from riskmodels import Utilities
from riskmodels.creatermsid import CreateRMSID, InsertUpdateRMSID

if __name__=="__main__":
    usage = "usage: %prog [options] configfile issue-id"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database",action="store_false",
                             default=True,dest="testOnly",
                             help="change the database")
    cmdlineParser.add_option("--no-additions", action="store_false",
                             default=True, dest="addRecords",
                             help="don't add new rms_issue records")
    cmdlineParser.add_option("--no-estu-additions", action="store_false",
                             default=True, dest="addESTURecords",
                             help="don't add new rms_estu_excluded records")
    cmdlineParser.add_option("--allow-deletions", action="store_true",
                             default=False, dest="removeRecords",
                             help="allow rms_issue records to be removed")
    cmdlineParser.add_option("--allow-estu-deletions", action="store_true",
                             default=False, dest="removeESTURecords",
                             help="allow rms_estu_excluded records to be removed")
    cmdlineParser.add_option("--allow-updates", action="store_true",
                             default=False, dest="updateRecords",
                             help="allow existing rms_issue records to be changed")
    cmdlineParser.add_option("--allow-estu-updates", action="store_true",
                             default=False, dest="updateESTURecords",
                             help="allow existing rms_estu_excluded records to be changed (based on other ESTU records)")
    cmdlineParser.add_option("--use-other-estu", action="store",
                             default=None, dest="otherESTU",
                             help="include ESTU exclusion records from this model (mnemonic or rms_id)")
    cmdlineParser.add_option("--target-country", action="store",
                             default=None, dest="targetRMG",
                             help="only update assets with the target country exposure")
    cmdlineParser.add_option("--write-out-file", action="store_true",
                             default=False, dest="writeOutFile",
                             help="write out the proposed insert/update/delete action")
    cmdlineParser.add_option("--include-neq", action="store_true",
                             default=False, dest="includeNonEquity",
                             help="force inclusion of non-equity ETFs regardless of model type")
    cmdlineParser.add_option("--v3", action="store_true",
                             default=True, dest="version3",
                             help="model is a 3rd generation model")
    cmdlineParser.add_option("--allow-all", action="store_true",
                             default=False, dest="allowAll",
                             help="allow updating of pretty much everything for a new model build")

    (options_, args_) = cmdlineParser.parse_args()

    if len(args_)<2:
        cmdlineParser.error("Missing issue-id")

    if options_.allowAll:
        options_.version3 = True
        options_.testOnly = False
        options_.updateRecords = True
        options_.removeRecords = True
        options_.updateESTURecords = True
        options_.removeESTURecords = True

    print('Version 3', options_.version3)
    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()
    connections = Connections.createConnections(config_)
    if options_.modelName != None or (options_.modelID != None and options_.modelRev != None):
        riskModelClass = Utilities.processModelAndDefaultCommandLine(options_, cmdlineParser)
        riskModel_ = riskModelClass(connections.modelDB, connections.marketDB)
        options_.targetRM = riskModelClass
        logging.info('Working on %s'%riskModel_.mnemonic)
    else:
        Utilities.processDefaultCommandLine(options_, cmdlineParser)
        options_.targetRM = None
        logging.info('Working on all the rms IDs')

    rmsGenerator = CreateRMSID(connections, options_.targetRM, options_.targetRMG)
    # lookup marketID if that is what was passed
    if len(args_[1]) == 10 and args_[1][0] != 'D':
        modelID = ','.join(rmsGenerator.mapMarketToModel([args_[1]]))
    else:
        modelID = args_[1]
    idList = transfer.createIssueIDList(modelID, connections, assetType='equities')[0]

    rmsInfo = rmsGenerator.getRMSInfo(idList,
            version3=options_.version3, includeNonEquity=options_.includeNonEquity)
    rmsInsertUpdate = InsertUpdateRMSID(connections, options_)
    rmsInsertUpdate.process(rmsInfo)

    if not options_.testOnly:
        connections.modelDB.commitChanges()
        logging.info('Committing changes')
    else:
        connections.modelDB.revertChanges()
        logging.info('Reverting changes')


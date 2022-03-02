
import logging
import optparse
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels import EquityModel

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="force purge of model flagged for distribution")
    cmdlineParser.add_option("--no-univ", action="store_false",
                             default=True, dest="deleteUniverse",
                             help="do not delete risk model universe")
    cmdlineParser.add_option("--no-exposures", action="store_false",
                             default=True, dest="deleteExposures",
                             help="do not delete factor exposures")
    cmdlineParser.add_option("--no-factors", action="store_false",
                             default=True, dest="deleteFactors",
                             help="do not delete factor/specific returns")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 0:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(
        sid=options.marketDBSID, user=options.marketDBUser,
        passwd=options.marketDBPasswd)
    riskModel = riskModelClass(modelDB, marketDB)
    rmInfo = modelDB.getRiskModelInfo(riskModel.rm_id, riskModel.revision)
    if rmInfo.distribute and not options.force:
        logging.warning('Risk model is enabled for distribution, skipping purge')
        sys.exit(1)

    status = 0
    try:
        if not options.deleteFactors or not options.deleteExposures:
            options.deleteUniverse = False
            if not options.deleteFactors:
                options.deleteExposures = False
        if isinstance(riskModel, EquityModel.FundamentalModel) or isinstance(riskModel, EquityModel.StatisticalModel):
            modelDB.flushRiskModelSerieData(riskModel.rms_id, True, deleteUniverse=options.deleteUniverse,
                                            deleteExposures=options.deleteExposures, deleteFactors=options.deleteFactors)
        else:
            modelDB.flushRiskModelSerieData(riskModel.rms_id, riskModel.newExposureFormat,
                                        deleteUniverse=options.deleteUniverse,
                                        deleteExposures=options.deleteExposures,
                                        deleteFactors=options.deleteFactors)
        modelDB.commitChanges()
    except Exception as ex:
        logging.error('Exception caught during processing', exc_info=True)
        modelDB.revertChanges()
        status = 1

    marketDB.finalize()
    modelDB.finalize()
    sys.exit(status)

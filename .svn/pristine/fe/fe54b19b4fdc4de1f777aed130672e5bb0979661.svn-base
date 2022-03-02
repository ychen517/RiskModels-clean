
import math
import numpy.linalg as linalg
import numpy
import numpy.ma as ma
import logging
import optparse
from marketdb import MarketDB
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels import Utilities

#------------------------------------------------------------------------------
def getISCData(modelDB, rmi, dt, idList):
    ISCDict = modelDB.getSpecificCovariances(rmi)
    rskDict = modelDB.getSpecificRisks(rmi)
    if (idList[0] in ISCDict) and (idList[1] in ISCDict[idList[0]]):
        covar = ISCDict[idList[0]][idList[1]]
    else:
        covar = ISCDict[idList[1]][idList[0]]
    correl = covar / (rskDict[idList[0]] * rskDict[idList[1]])
    trk_err = (rskDict[idList[0]] * rskDict[idList[0]]) + (rskDict[idList[1]] * rskDict[idList[1]]) - (2.0 * covar)
    trk_err = numpy.sqrt(trk_err)
    logging.info('%s,%s,%.4f,%s,%.4f,%.4f,%.4f',
            dt, idList[0].getSubIDString(), rskDict[idList[0]], idList[1].getSubIDString(), rskDict[idList[1]], correl, trk_err)
    return

#------------------------------------------------------------------------------
if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--ids", action="store", default=None, dest="idList", help="input ID list")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 3:
        cmdlineParser.error("Incorrect number of arguments")
    
    modelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)

    startDate = Utilities.parseISODate(args[1])
    endDate = Utilities.parseISODate(args[2])
    dates = modelDB.getDateRange(riskModel.rmg, startDate, endDate, excludeWeekend=True)
    ids = str(args[0]).split(',')
    
    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        allSubIDs = modelDB.getRiskModelInstanceUniverse(rmi)
        sidStringMap = dict([(sid.getSubIDString(), sid) for sid in allSubIDs])
        trackList = [sidStringMap[ss] for ss in ids if ss in sidStringMap]
        if len(trackList) == 2:
            getISCData(modelDB, rmi, d, trackList)
    
    modelDB.finalize()
    marketDB.finalize()

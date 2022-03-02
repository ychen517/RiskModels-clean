import math
import numpy.linalg as linalg
import numpy
import numpy.ma as ma
import logging
import optparse
import pandas
from marketdb import MarketDB
from riskmodels import Utilities
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels import AssetProcessor_V4

if __name__ == '__main__':

    # Parse I/O
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--type", action="store", default=None, dest="getAssetType", help="input ID list by type")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2:
        cmdlineParser.error("Incorrect number of arguments")

    # Set up DB info and model class
    modelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    rm = modelClass(modelDB, marketDB)

    # Set up dates
    startDate = Utilities.parseISODate(args[1])
    endDate = startDate
    if len(args) == 3:
        endDate = Utilities.parseISODate(args[2])
    dates = modelDB.getDateRange(rm.rmg, startDate, endDate, excludeWeekend=True)

    # Get descriptor info
    allDescDict = dict(modelDB.getAllDescriptors())
    descriptors = sorted(set(str(args[0]).split(',')))
    descriptorMap = dict()

    for dt in dates:

        # Output high level model info
        rm.setFactorsForDate(dt, modelDB)
        rmi = rm.getRiskModelInstance(dt, modelDB)
        numeraire = rm.numeraire.currency_code

        # Get asset info
        universe = modelDB.getRiskModelInstanceUniverse(rmi)
        assetData = AssetProcessor_V4.AssetProcessor(dt, modelDB, marketDB, rm.getDefaultAPParameters())
        assetData.process_asset_information(rm.rmg, universe=universe)
        if options.getAssetType is None:
            estu = set(rm.loadEstimationUniverse(rmi, modelDB, assetData))
        else:
            typeList = str(options.getAssetType).split(',')
            estu = set([sid for sid in assetData.universe if assetData.getAssetType().get(sid, None) in typeList])
        estu = sorted(estu)

        # Get asset descriptors
        descValueDict, okDescriptorCoverageMap = rm.loadDescriptors(
                descriptors, allDescDict, dt, estu, modelDB, assetData.getCurrencyAssetMap())
        for ds in descriptors:
            dsSeries = descValueDict.loc[estu, ds]
            dsSeries.rename(dt)
            if ds in descriptorMap:
                descriptorMap[ds] = pandas.concat([descriptorMap[ds], dsSeries], axis=1)
            else:
                descriptorMap[ds] = pandas.DataFrame(dsSeries)

    # Output descriptors to flatfile
    for ds in descriptors:
        dsDF = descriptorMap[ds].fillna(0.0)
        universe = sorted(set(dsDF.index))
        dsDF = dsDF.loc[universe, :]
        dsDF.index = [sid.getSubIDString() for sid in universe]
        dsDF.columns = dates
        fileName = 'tmp/ds-%s-%s.csv' % (ds, dt)
        dsDF.to_csv(fileName)

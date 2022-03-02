import logging
import math
import numpy
import numpy.linalg as linalg
import numpy.ma as ma
import optparse
from marketdb import MarketDB
from riskmodels import AssetProcessor
from riskmodels import Utilities
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB

if __name__ == '__main__':

    # Parse I/O
    usage = "usage: %prog [options] <YYYY-MM-DD> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--cid", action="store_true", default=False, dest="getCIDAssets", help="input ID list")
    (options, args) = cmdlineParser.parse_args()
    
    # Set up DB info and model class
    modelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    rm = modelClass(modelDB, marketDB)
    numeraire = rm.numeraire.currency_code

    # Set up dates
    if len(args) == 2:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
    else:
        startDate = [ms.from_dt for ms in modelDB.getModelSeries(rm.rm_id) \
                if ms.rms_id==rm.rms_id][0]
        endDate = Utilities.parseISODate(args[0])
    dates = modelDB.getDateRange(rm.rmg, startDate, endDate, excludeWeekend=True)

    # Get descriptor info
    rm.setFactorsForDate(dates[0], modelDB)
    allDescDict = dict(modelDB.getAllDescriptors())
    if rm.regionalDescriptorStructure:
        localDescDict = dict(modelDB.getAllDescriptors(local=True))
    descriptors = []
    if hasattr(rm, 'DescriptorMap'):
        for f in rm.DescriptorMap.keys():
            dsList = [ds for ds in rm.DescriptorMap[f] if ds[-3:] != '_md']
            descriptors.extend(dsList)
        descriptors = list(set(descriptors))
        if rm.regionalDescriptorStructure:
            funDesc = sorted(ds for ds in descriptors if ds not in localDescDict)
            locDesc = sorted(ds for ds in descriptors if ds in localDescDict)
            descriptors = funDesc + locDesc
        else:
            descriptors.sort()
    
    outfileName = 'tmp/sanity-%s-%s.csv' % (rm.mnemonic, endDate.year)
    ofl = open(outfileName, 'w')
    ofl.write('Model,RMS_ID,Numeraire,\n')
    ofl.write('%s,%d,%s,\n' % (rm.mnemonic, rm.rms_id, rm.numeraire.currency_code))
    ofl.write('******************Asset Info******************,\n')
    ofl.write(',UNIV,ESTU,ESTU_UNMAPPED,')
    for ds in descriptors:
        ofl.write('%s,' % ds)
    for fac in rm.styles:
        ofl.write('%s,' % fac.name)
    ofl.write('\n')

    for dt in dates:
        logging.info('%s', dt)
        # Output high level model info
        rm.setFactorsForDate(dt, modelDB)
        rmi = rm.getRiskModelInstance(dt, modelDB)

        # Get list of sub-issues
        universe = modelDB.getRiskModelInstanceUniverse(rmi)

        # Get asset info
        data = AssetProcessor.process_asset_information(
            dt, universe, rm.rmg, modelDB, marketDB,
            checkHomeCountry=rm.coverageMultiCountry,
            numeraire_id=rm.numeraire.currency_id,
            legacyDates=rm.legacyMCapDates,
            forceRun=True,
            nurseryRMGList=rm.nurseryRMGs,
            rmgOverride=rm.rmgOverride)

        estu = rm.loadEstimationUniverse(rmi, modelDB, data)
        estu_unmapped = set(estu).difference(set(data.universe))
        ofl.write('%s,%d,%d,%d,' % (dt, len(data.universe), len(estu), len(estu_unmapped)))

        # Get asset descriptors
        if len(descriptors) > 0:
            descValueDict, okDescriptorCoverageMap = rm.loadDescriptorData(
                    descriptors, allDescDict, dt, data.universe,
                    modelDB, data.currencyAssetMap, rollOver=0)
            for ds in descriptors:
                if ds not in descValueDict:
                    ofl.write('1.0,')
                else:
                    misscov = numpy.flatnonzero(ma.getmaskarray(descValueDict[ds]))
                    misscov = len(misscov) / float(len(data.universe))
                    ofl.write('%.4f,' % misscov)

        # Get asset exposures
        if len(rm.styles) > 0:
            expM = rm.loadExposureMatrix(rmi, modelDB, addExtraCountries=True, assetList=data.universe)
            for fac in rm.styles:
                if fac.name not in expM.factors_:
                    ofl.write('1.0,')
                else:
                    fdx = expM.getFactorIndex(fac.name)
                    misscov = numpy.flatnonzero(ma.getmaskarray(expM.data_[fdx,:]))
                    misscov = len(misscov) / float(len(data.universe))
                    ofl.write('%.4f,' % misscov)
        ofl.write('\n')

    ofl.close()
    modelDB.finalize()
    marketDB.finalize()

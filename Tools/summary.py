import math
import numpy.linalg as linalg
import numpy
import numpy.ma as ma
import logging
import optparse
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
    cmdlineParser.add_option("--cid", action="store_true", default=False, dest="getCIDAssets", help="input ID list")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2:
        cmdlineParser.error("Incorrect number of arguments")
    ids = str(args[0]).split(',')
    
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
    descriptors = ['ISC_ADV_Score','ISC_Ret_Score','ISC_IPO_Score']
    if hasattr(rm, 'DescriptorMap'):
        for f in rm.DescriptorMap.keys():
            dsList = [ds for ds in rm.DescriptorMap[f] if ds[-3:] != '_md']
            descriptors.extend(dsList)
    descriptors = sorted(set(descriptors))
    
    for dt in dates:
        outfileName = 'tmp/summary-%s.csv' % str(dt)
        ofl = open(outfileName, 'w')
        ofl.write('******************Meta-Info******************,\n')

        # Output high level model info
        rm.setFactorsForDate(dt, modelDB)
        rmi = rm.getRiskModelInstance(dt, modelDB)
        numeraire = rm.numeraire.currency_code
        ofl.write('Date,Model,RMS_ID,Numeraire,\n')
        ofl.write('%s,%s,%d,%s,\n' % (str(dt), rm.mnemonic, rmi.rms_id, rm.numeraire.currency_code))

        # Get list of sub-issues
        universe = modelDB.getRiskModelInstanceUniverse(rmi)
        sidStringMap = dict([(sid.getSubIDString(), sid) for sid in universe])
        trackList = [sidStringMap[ss] for ss in ids if ss in sidStringMap]
        if len(trackList) < 1:
            logging.warning('No valid subissues in the model for %s', dt)
            continue

        # Get siblings if required
        if options.getCIDAssets:
            sidCompanyMap = modelDB.getIssueCompanies(dt, trackList, marketDB)
            trackList = []
            for cid in set(sidCompanyMap.values()):
                sidList = list(modelDB.getCompanySubIssues(dt, [cid], marketDB).keys())
                trackList.extend(sidList)

        # Get asset info
        if not hasattr(rm, 'coverageMultiCountry'):
            rm.coverageMultiCountry = len(rm.rmg) > 1
        assetData = AssetProcessor_V4.AssetProcessor(dt, modelDB, marketDB, rm.getDefaultAPParameters())
        assetData.process_asset_information(rm.rmg, universe=universe)
        try:
            estu = rm.loadEstimationUniverse(rmi, modelDB, assetData)
        except:
            estu = set()
        trdRMGMap = dict([(sid, rmg) for (rmg, sids) in assetData.tradingRmgAssetMap.items() for sid in sids])
        curRMGMap = dict([(sid, rmg) for (rmg, sids) in assetData.rmgAssetMap.items() for sid in sids])
        assetTradingCurrencyMap = modelDB.getTradingCurrency(dt, assetData.universe, marketDB)
        assetRMGMap = Utilities.flip_dict_of_lists(assetData.rmgAssetMap)
        ofl.write('******************Asset Info******************,\n')
        ofl.write('sub-issue,CID,Name,Sedol,ISIN,Home RMG,Trading RMG,Home Currency,Trading Currency,Exchange,Type,ESTU,\n')
        for sid in trackList:
            ofl.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,\n' % (\
                sid.getSubIDString(), assetData.getSubIssue2CidMapping().get(sid, None), assetData.getNameMap().get(sid, None),
                assetData.getSEDOLMap().get(sid, None), assetData.getISINMap().get(sid, None),
                assetRMGMap[sid].mnemonic, trdRMGMap[sid].mnemonic, curRMGMap[sid].getCurrencyCode(dt),
                assetTradingCurrencyMap.get(sid, None), assetData.getMarketType().get(sid, None),
                assetData.getAssetType().get(sid, None), sid in estu))

        # Output mcap info
        mcapDF = AssetProcessor_V4.computeTotalIssuerMarketCaps(dt, assetData.marketCaps, rm.numeraire, modelDB, marketDB)
        ofl.write('******************MCap Info******************,\n')
        ofl.write('sub-issue,MCap,Total MCap, DLC MCap,\n')
        for sid in trackList:
            ofl.write('%s,%.2f,%.2f,%.2f,\n' % (\
                sid.getSubIDString(), assetData.marketCaps[sid], mcapDF.loc[sid, 'totalCap'], mcapDF.loc[sid, 'dlcCap']))

        # Output recent returns
        ofl.write('******************Returns Info******************,\n')
        tradeReturns = modelDB.loadTotalReturnsHistoryV3(rm.rmg, dt, trackList, 7,
                            assetConvMap=None, excludeWeekend=False)
        homeReturns = modelDB.loadTotalReturnsHistoryV3(rm.rmg, dt, trackList, 7,
                            assetConvMap=assetData.drCurrData, excludeWeekend=False)
        ofl.write('sub-issue,currency,')
        for rDt in tradeReturns.dates:
            ofl.write('%s,' % rDt)
        ofl.write('\n')
        for (idx, sid) in enumerate(trackList):
            tradeCurr = assetTradingCurrencyMap.get(sid, None)
            ofl.write('%s,%s,' % (sid.getSubIDString(), tradeCurr))
            for (jdx, rtDt) in enumerate(tradeReturns.dates):
                if tradeReturns.data[idx, jdx] is ma.masked:
                    ofl.write(',')
                else:
                    ofl.write('%.6f,' % tradeReturns.data[idx, jdx])
            ofl.write('\n')
            homeCurr = curRMGMap[sid].getCurrencyCode(dt)
            if tradeCurr != homeCurr:
                ofl.write('%s,%s,' % (sid.getSubIDString(), homeCurr))
                for (jdx, rtDt) in enumerate(homeReturns.dates):
                    if homeReturns.data[idx, jdx] is ma.masked:
                        ofl.write(',')
                    else:
                        ofl.write('%.6f,' % homeReturns.data[idx, jdx])
                ofl.write('\n')

        # Get asset descriptors
        if len(descriptors) > 0:
            descValueDict, okDescriptorCoverageMap = rm.loadDescriptors(
                    descriptors, allDescDict, dt, trackList,
                    modelDB, assetData.getCurrencyAssetMap(), rollOver=rm.rollOverDescriptors)
            ofl.write('******************Descriptor Info******************,\n')
            ofl.write('sub-issue,')
            for ds in descValueDict.keys():
                ofl.write('%s,' % ds)
            ofl.write('\n')
            for (idx, sid) in enumerate(trackList):
                ofl.write('%s,' % sid.getSubIDString())
                for ds in descValueDict.keys():
                    ofl.write('%.6f,' % descValueDict[ds][idx])
                ofl.write('\n')

        # Get asset exposures
        expM = rm.loadExposureMatrix(rmi, modelDB, addExtraCountries=True, assetList=trackList)
        ofl.write('******************Exposure Info******************,\n')
        ofl.write('sub-issue,')
        fTypes = [expM.getFactorType(f) for f in expM.factors_]
        compactTypes = [expM.IndustryFactor, expM.CountryFactor, expM.CurrencyFactor]
        compactTypes = [t for t in compactTypes if t in fTypes]
        for (fdx, fct) in enumerate(expM.factors_):
            if fTypes[fdx] not in compactTypes:
                ofl.write('%s,' % fct.replace(',',''))
        for t in compactTypes:
            ofl.write('%s,' % t.name)
        ofl.write('\n')
        expdataMask = ma.getmaskarray(expM.data_)
        for (idx, sid) in enumerate(trackList):
            ofl.write('%s,' % sid.getSubIDString())
            cmpData = list(compactTypes)
            for (fdx, fct) in enumerate(expM.factors_):
                if fTypes[fdx] in compactTypes:
                    if expdataMask[fdx,idx] == False and expM.data_[fdx,idx] > 0.0:
                        cp_idx = compactTypes.index(fTypes[fdx])
                        cmpData[cp_idx] = fct.replace(',','')
                else:
                    if expdataMask[fdx,idx] == False:
                        ofl.write('%.6f,' % expM.data_[fdx,idx])
                    else:
                        ofl.write(',')
            for (j,t) in enumerate(compactTypes):
                ofl.write('%s,' % cmpData[j])
            ofl.write('\n')

        ofl.close()

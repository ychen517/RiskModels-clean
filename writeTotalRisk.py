import logging
import optparse
import numpy
import numpy.ma as ma
import os

from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
import generateRiskModelQA as QA
from riskmodels.writeFlatFiles import FlatFilesV3

if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--flat-file-dir", action="store",
                             default=".", dest="flatDir",
                             help="name of Flat file directory")
    cmdlineParser.add_option("--no-sub-dirs", action="store_false",
                             default=True, dest="appendDateDirs",
                             help="Don't append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--file-format-version", action="store", type="float",
                             default=3.2, dest="fileFormatVersion",
                             help="version of flat file format to create")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    riskModel_Class = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, 
                              user=options.modelDBUser, 
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser, 
                                 passwd=options.marketDBPasswd)
    riskModel_ = riskModel_Class(modelDB, marketDB)

    startDate = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[1])
    dates = modelDB.getDateRange(riskModel_.rmg, startDate, endDate, True)

    runTime = modelDB.revDateTime
    for d in dates:
        logging.info('Processing %s' % d)
        currModelData = QA.RiskModelDataBundle(riskModel_, d, modelDB, marketDB)
        # much of this is cribbed from generateRiskModelQA.RiskModelDataBundle.displayPortfolioCorrelations()
        univ = currModelData.getAllSubIssues()
        expM = currModelData.getExposureMatrix()
        mdl2sub = dict([(n.getModelID(),n) for n in univ])
        assetIds = [s.getModelID().getIDString() for s in expM.getAssets()]
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(assetIds)])
        fIdx = [expM.getFactorIndex(f.name) for f in riskModel_.styles]
        exposureMatrix = ma.take(expM.getMatrix(), fIdx, axis=0).filled(0.0)

        folios = list()
        resultsDict = dict()
        folioList = list(QA.MODEL_INDEX_MAP.get(currModelData.modelSelector.mnemonic[:4]))
        if not folioList:
            continue
        imp = modelDB.getIssueMapPairs(currModelData.rmi.date)
        for f in folioList:
            # portfolio weights
            port = modelDB.getIndexConstituents(f, currModelData.rmi.date, 
                            marketDB, rollBack=20, issueMapPairs=imp)
            if len(port) > 0 :
                (assets, weights) = zip(*[(mdl2sub[a], w) for (a,w) \
                                        in port if a in mdl2sub])
                weights = numpy.array(weights)
                weights /= numpy.sum(weights)
                folios.append(list(zip(assets, weights)))
            else:
                folios.append([(univ[0],0.0)])
            
            # benchmark weights
            IdsWghtMap=dict()
            bmIdx = folioList.index(f)
            
            for (sid, wt) in folios[bmIdx]:
                key = sid.getModelID().getIDString()
                IdsWghtMap[key] = wt
            
            validIds = [n for n in IdsWghtMap.keys() if n in assetIdxMap]
            wgts = [IdsWghtMap[n] for n in validIds]
            indices = [assetIdxMap[n] for n in validIds]
            if len(indices) > 0:
                wgts = wgts / numpy.sum(wgts)
                expMat = ma.take(exposureMatrix, indices, axis=1)
                expMat *= wgts
                resultsDict[f] = numpy.sum(expMat, axis=1)

        # calculate ESTU risk        
        estu = currModelData.getESTU()
        w = [currModelData.getSubIssueMarketCap(s) for s in estu]
        w /= numpy.sum(w)
        folios.append(list(zip(estu, w)))
        modelESTUName = '%s ESTU' % riskModel_.mnemonic
        folioList.append(modelESTUName)

        #get factor exposure of estimation universe
        estuWeightsMap = modelDB.getRMIESTUWeights(currModelData.rmi)
        # change keys from subissue ids to string model ids
        estuWgtsMap = dict() 
        for i in estuWeightsMap.keys(): 
            estuWgtsMap[i.getModelID().getIDString()]=estuWeightsMap[i]
        
        estuWgts = numpy.array(list(estuWgtsMap.values()))
        estuWgts /= numpy.sum(estuWgts)
        
        validIds = [n for n in estuWgtsMap.keys() if n in assetIdxMap]
        wgts = [estuWgtsMap[n] for n in validIds]
        indices = [assetIdxMap[n] for n in validIds]
        if len(indices) > 0:
            wgts = wgts / numpy.sum(wgts)
            expMat = ma.take(exposureMatrix, indices, axis=1)
            expMat *= wgts
            resultsDict[modelESTUName] = numpy.sum(expMat, axis=1)
        
        (vols, corr) = Utilities.compute_correlation_portfolios(
                        expM,
                        currModelData.getFactorCov(), 
                        currModelData.getSpecificRisks()[0], folios, 
                        currModelData.getSpecificRisks()[1])
        vols = vols.filled(0.0)
        
        styleNames = list()    
        for i in range(len(riskModel_.styles)):
            styleNames.append(riskModel_.styles[i].name)

        targetDir = options.flatDir
        if options.appendDateDirs:
            targetDir += '/%4d/%02d' % (d.year, d.month)
        try:
            os.makedirs(targetDir)
        except OSError as e:
            if e.errno != 17:
                raise
            else:
                pass
        outfile = open('%s/%s.%s.bmk' % (targetDir, riskModel_.mnemonic, d.strftime('%Y%m%d')), 'w', encoding='utf-8')
        ffv3 = FlatFilesV3()
        ffv3.dataDate_ = d
        ffv3.createDate_ = runTime
        ffv3.writeDateHeader(options, outfile)
        outfile.write('#Columns:Benchmark|Names|Total Risk|')
        outfile.write('|'.join(['%s' % n for n in styleNames]))
        outfile.write('\n')
        outfile.write('#Unit: ID|Number|Percent|')
        outfile.write('|'.join(['Number']*len(styleNames)))
        outfile.write('\n')
        for (i, f) in enumerate(folioList):
            line = '%s|%d|%5.2f' \
                    % (f, len(folios[i]), vols[i] * 100.0)
            for j in range(len(styleNames)):
                line += '|%1.9f' %(resultsDict[f][j])    
            outfile.write(line+'\n')
        outfile.close()
    marketDB.finalize()
    modelDB.finalize()
    

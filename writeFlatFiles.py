import io
import datetime
import logging
import numpy
import numpy.ma as ma
import optparse
import os
import os.path
import pymssql
import re
import shutil
import tempfile
from math import sqrt

from marketdb import MarketDB
from marketdb import MarketID
from marketdb.Utilities import listChunkIterator
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities
from riskmodels.Matrices import ExposureMatrix
from riskmodels.ModelDB import SubIssue, FromThruCache
from riskmodels.wombat import *
from riskmodels.writeFlatFiles import FlatFilesV3, makeFinalFileName, numOrNull, getFactorType, zeroPad, \
    getExposureAssets


class FlatFilesV2:
    """Class to create flat files in version 2 format.
    """
    def writeExposures(self, date, expMatrix, svDict, outFile):
        """Write a pipe delimited exposure file to outFile.
        """
        if outFile != None:
            nonIndustryFactors = sorted(expMatrix.getFactorNames(ExposureMatrix.StyleFactor) + expMatrix.getFactorNames(ExposureMatrix.StatisticalFactor))
            outFile.write('%s\n' % date)
            outFile.write("""Axioma Id|%s|Industry|Specific Risk\n"""
                          % ('|'.join([f for f in nonIndustryFactors])))
        mat = expMatrix.getMatrix()
        exposureAssets = []
        exposureAssetIDs = []
        for aIdx in range(len(expMatrix.getAssets())):
            asset = expMatrix.getAssets()[aIdx]
            modelID = asset.getModelID()
            if outFile != None:
                line = io.StringIO()
                if svDict != None:
                    if asset in svDict:
                        specificRisk = 100.0*svDict[asset]
                    else:
                        continue
                else:
                    specificRisk = -1.0
                line.write('%s' % (modelID.getPublicID()))
                for f in nonIndustryFactors:
                    line.write('|%4.3f' % (mat[expMatrix.getFactorIndex(f), aIdx]))
                numInd = 0
                for f in numpy.flatnonzero(mat[:,aIdx]):
                    if expMatrix.checkFactorType(expMatrix.getFactorNames()[f],
                                                 ExposureMatrix.IndustryFactor):
                        numInd += 1
                        fName = expMatrix.getFactorNames()[f]
                        line.write('|%s' % (fName))
                if numInd == 0:
                    line.write('|')
                if numInd > 1:
                    logging.warning('%d industries for %s',
                                 numInd, modelID.getIDString())
                    continue
                line.write('|%9.8f' % (specificRisk))
                line.write('\n')
                outFile.write(line.getvalue())
            exposureAssetIDs.append(modelID)
            exposureAssets.append(asset)
        return (exposureAssets, exposureAssetIDs)
    
    def writeFactorCov(self, date, covMatrix, factors, outFile):
        """Write a pipe delimited factor covariance file to outFile.
        """
        outFile.write('%s\n' % date)
        for factor in factors:
           outFile.write('|%s' % factor.name)
        outFile.write("\n")
        
        for i in range(len(factors)):
           outFile.write('%s' % factors[i].name)
           for j in range(len(factors)):
              outFile.write('|%.12g' % (10000.0*covMatrix[i,j]))
           outFile.write("\n")
    
    def writeIdentifierMapping(self, d, exposureAssets, modelDB, marketDB, rmg, outFile, outFile_nosedol=None,
                               outFile_nocusip=None, outFile_neither=None):
        cusipMap = modelDB.getIssueCUSIPs(d, exposureAssets, marketDB)
        sedolMap = modelDB.getIssueSEDOLs(d, exposureAssets, marketDB)
        isinMap = modelDB.getIssueISINs(d, exposureAssets, marketDB)
        nameMap = modelDB.getIssueNames(d, exposureAssets, marketDB)
        tickerMap = modelDB.getIssueTickers(d, exposureAssets, marketDB)
        outFiles = [outFile, outFile_nosedol, outFile_nocusip, outFile_neither]
        for f in outFiles:
            if f:
                f.write('%s\n' % d)
                if rmg.rmg_id != 1:
                    f.write('Axioma Id|ISIN|CUSIP|SEDOL(7)|Ticker|Ticker %s|Description\n' % rmg.mnemonic)
                else:
                    f.write('Axioma Id|ISIN|CUSIP|SEDOL(7)|Ticker|Description\n')
        for a in exposureAssets:
            ticker = tickerMap.get(a,'')
            if ticker != '' and rmg.rmg_id != 1:
                ticker = ticker + "-" + rmg.mnemonic
            if rmg.rmg_id != 1:
                # for non-us, include original ticker also
                for (idx, f) in enumerate(outFiles):
                    if f:
                        f.write('%s|%s|%s|%s|%s|%s|%s\n' % (a.getPublicID(),
                                                            (idx == 0) and isinMap.get(a, '') or '',
                                                            (idx == 0 or idx == 1) and cusipMap.get(a, '') or '',
                                                            (idx == 0 or idx == 2) and zeroPad(sedolMap.get(a, ''), 7) or '',
                                                            ticker,
                                                            tickerMap.get(a,''),
                                                            nameMap.get(a, '')))
            else:
                for (idx, f) in enumerate(outFiles):
                    if f:
                        f.write('%s|%s|%s|%s|%s|%s\n' % (a.getPublicID(),
                                                         (idx == 0) and isinMap.get(a, '') or '',
                                                         (idx == 0 or idx == 1) and cusipMap.get(a, '') or '',
                                                         (idx == 0 or idx == 2) and zeroPad(sedolMap.get(a, ''), 7) or '', ticker,
                                                         nameMap.get(a, '')))
    
    def writeCurrencyRates(self, d, marketDB, modelDB, outFile):
        outFile.write('%s\n' % d)
        outFile.write('CurrencyCode|Description|Exchange Rate|Risk-Free Rate|Cumulative RFR\n')
        currencyConverter = MarketDB.CurrencyConverter(marketDB, d)
        for k in currencyConverter.rates.keys():
            rate = currencyConverter.getRate(k, k)
            code = marketDB.getCurrencyISOCode(k,d)
            desc = marketDB.getCurrencyDesc(k, d)
            rfr = modelDB.getRiskFreeRateHistory(
                [code], [d], marketDB, annualize=True).data[0,0]
            cumR = modelDB.getCumulativeRiskFreeRate(code, d)
            if not rfr is None or not cumR is None:
                outFile.write('%s|%s|%s' % (code, desc, rate))
                if rfr is ma.masked:
                    outFile.write('|')
                else:
                    outFile.write('|%s' % rfr)
                if cumR is None:
                    outFile.write('|')
                else:
                    outFile.write('|%s' % cumR)
                outFile.write('\n')
    
    def writeFactorReturns(self, d, riskModel, modelDB, outFile):
        outFile.write('%s\n' % d)
        outFile.write('Factor|return|Cumulative Return\n')
        factorReturns = riskModel.loadFactorReturns(d, modelDB)
        factorReturnMap = dict(zip(factorReturns[1], factorReturns[0]))
        cumReturns = riskModel.loadCumulativeFactorReturns(d, modelDB)
        cumReturnMap = dict(zip(cumReturns[1], cumReturns[0]))
        for i in factorReturnMap.keys():
            outFile.write('%s|%4.10f|%s\n'
                          % (i.name,
                             factorReturnMap[i] * 100.0,
                             numOrNull(cumReturnMap[i], '%4.10f')))
    
    def writeIndustryHierarchy(self, d, riskModel, factors, modelDB, marketDB, outFile):
        outFile.write('%s\n' % d)
        outFile.write('Name|Parent|Level\n')
        # Read all classifications--find root parent and non-root parent of each
        allClassifications = riskModel.getAllClassifications(modelDB)
        for c in allClassifications:
            outFile.write('%s|' % c.name)
            parents = riskModel.getClassificationParents(c, modelDB)
            nonrootParents = [p for p in parents if not p.isRoot]
            if len(nonrootParents) > 0:
                outFile.write('%s' % nonrootParents[0].name)
            outFile.write('|')
            rootParents = [p for p in parents if p.isRoot]
            if len(rootParents) > 0:
                outFile.write('%s' % rootParents[0].name)
            outFile.write('\n')
    
    def writeAssetAttributes(self, date, riskModel, exposureAssets, expM, factors,
                             factorCov, svDict, modelDB, marketDB, outFile):
        histLen = 60
        for i in range(len(factors)):
            factor = factors[i]
            if expM.getFactorIndex(factor.name) != i:
                logging.error("Factors in covariance matrix and exposure matrix are not in same order")
                return 0
        
        dateList = modelDB.getDates(riskModel.rmg, date, histLen)
        cashSubIssues = list()
        for rmg in riskModel.rmg:
            cashSubIssues.extend([i for i in modelDB.getActiveSubIssues(
                        rmg, date) if i.isCashAsset()])
        subIssues = sorted(set(exposureAssets) | set(cashSubIssues))
        ucp = modelDB.loadUCPHistory(dateList, subIssues, None)
        latestUCP = ma.masked_where(True, ucp.data[:,0])
        for i in range(len(subIssues)):
            for j in range(len(dateList)-1,-1,-1):
                if not ucp.data[i,j] is ma.masked:
                    latestUCP[i] = ucp.data[i,j]
                    break
        mcapDates = modelDB.getDates(riskModel.rmg, date, histLen)
        marketCaps = modelDB.loadMarketCapsHistory(mcapDates, subIssues, None)
        latestMarketCap = ma.masked_where(True, marketCaps[:,0])
        for i in range(len(subIssues)):
            for j in range(len(dateList)-1,-1,-1):
                if not marketCaps[i,j] is ma.masked:
                    latestMarketCap[i] = marketCaps[i,j]
                    break
        last20Days = self.getDates(riskModel.rmg, date, 20)
        dailyVolumes = modelDB.loadVolumeHistory(
            last20Days, subIssues, None)
        adv20 = ma.average(dailyVolumes.data[:,-20:], axis=1)
        totalReturns = modelDB.loadTotalReturnsHistory(riskModel.rmg, date,
                                                       subIssues, 120, None)
        ret1 = totalReturns.data[:,-1] * 100.0
        totalReturns.data = totalReturns.data.filled(0.0)
        cumReturns = modelDB.loadCumulativeReturnsHistory([date], subIssues).data[:,-1]
        mcapDates = modelDB.getDates(riskModel.rmg, date, 19)
        # Only get historic betas for US Models
        histbetas = dict()
        if len(riskModel.rmg) == 1 \
                and riskModel.rmg[0].mnemonic in ('US', 'CA'):
            for rmg in riskModel.rmg:
                if riskModel.modelHack.nonCheatHistoricBetas:
                    histbetas.update(modelDB.getRMGHistoricBeta(rmg, date))
                else:
                    histbetas.update(modelDB.getRMGHistoricBetaOld(rmg, date))
        else:
            logging.info('Skipping historic betas')
        predbetas = modelDB.getRMIPredictedBeta(riskModel.getRiskModelInstance(date, modelDB))
        #assert(len(predbetas)>0)
        outFile.write('%s\n' % date)
        outFile.write("Axioma Id|RolloverFlag|Price|Market Cap|20-Day ADV|1-Day Return|Historical Beta|Predicted Beta|Cumulative Return\n")
        for i in range(len(subIssues)):
            line = io.StringIO()
            asset = subIssues[i]
            modelID = asset.getModelID()
            line.write('%s|' % (modelID.getPublicID()))
            if ucp.data[i, -1] is ma.masked:
                line.write('*')
            line.write('|%s' % numOrNull(latestUCP[i], '%.4f'))
            line.write('|%s' % numOrNull(latestMarketCap[i], '%.0f'))
            line.write('|%s' % numOrNull(adv20[i], '%.0f'))
            line.write('|%s' % numOrNull(ret1[i], '%.4f'))
            line.write('|%s' % numOrNull(histbetas.get(asset, ma.masked), '%.4f'))
            line.write('|%s' % numOrNull(predbetas.get(asset, ma.masked), '%.4f'))
            line.write('|%s' % numOrNull(cumReturns[i], '%.6g'))
            line.write('\n')
            outFile.write(line.getvalue())
    
    def writeGSFile(self, marketDB, modelDB, d, inFileName, outFile):
        import os.path
        if not os.path.isfile(inFileName):
            logging.info("Can't find Goldman file %s", inFileName)
            return
        infile = open(inFileName, 'r')
        #marketDB.setTransDateTime(d)
        tickerDict = marketDB.loadIdentifierInformation(d, 'asset_dim_ticker')
        cusipDict = marketDB.loadIdentifierInformation(d, 'asset_dim_cusip')
        isinDict = marketDB.loadIdentifierInformation(d, 'asset_dim_isin')
        tickerReverseDict = dict([(j,i) for (i,j) in tickerDict.items()])
        cusipReverseDict = dict([(j,i) for (i,j) in cusipDict.items()])
        issueMapPairs = modelDB.getIssueMapPairs(d)
        marketModelMap = dict([(j,i) for (i,j) in issueMapPairs])
        
        outFile.write('%s\n' % d)
        outFile.write('Axioma Id|CUSIP|Ticker|ISIN|Shortfall|Sum\n')
        for inline in infile:
            if inline.startswith('#'):
                continue
            fields = inline.split(',')
            ticker = fields[1]
            cusip = fields[2].strip('"')
            n1 = float(fields[4])    # goldman "A"
            n2 = float(fields[5])    # goldman "B"
            n3 = float(fields[6])    # goldman "C"
            axiomaID1 = tickerReverseDict.get(ticker)
            axiomaID2 = cusipReverseDict.get(cusip)
            marketAXID = axiomaID1
            if axiomaID1 is None:
                logging.warning('No axioma ID for ticker %s on date  %s',
                             ticker, d)
                marketAXID = axiomaID2
            if axiomaID2 is None:
                logging.warning('No axioma ID for cusip %s  on date %s', cusip, d)
                marketAXID = axiomaID1
            if axiomaID1 is not None and axiomaID2 is not None and axiomaID1 != axiomaID2:
                logging.fatal('AXID for ticker "%s" is %s while AXID for cusip "%s" is %s', ticker, axiomaID1, cusip, axiomaID2)
                raise ValueError('AXID for ticker "%s" is %s while AXID for cusip "%s" is %s' % \
                      (ticker, axiomaID1, cusip, axiomaID2))
            if marketAXID is None:
                continue
            axid = marketModelMap.get(marketAXID)
            if axid is None:
                logging.warning('No model ID for market ID %s', marketAXID)
                continue
            axid = axid.getPublicID()
            n1 = n1 / 10000 # convert from basis points to dollars
            n3 = n3 / 10000 # convert from basis points to dollars
            n2 = n2 / 10000 / sqrt(1000) # convert from basis points to dollars, plus additional conversion
            numberString = '%#.5e!%#.5e!%#.5e' % (n1, n2, n3)
            logging.debug('writing: %s|%s|%s|%s|%s', axid, cusip, ticker, isinDict.get(marketAXID, '' ), numberString)
            outFile.write('%s|%s|%s|%s|%s|%d\n' % (axid, cusip, ticker, isinDict.get(marketAXID, '' ),
                                                   wombat3(numberString),wombat(wombat3(numberString))))
    
    def writeDay(self, options, rmi, riskModel, modelDB, marketDB):
        if len(riskModel.rmg) > 1:
            logging.fatal('Version 2 format only supports single-country'
                          ' models')
            return 1
        d = rmi.date
        mnemonic = riskModel.mnemonic
        (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
            rmi, modelDB)
        expM = riskModel.loadExposureMatrix(rmi, modelDB)
        expM.fill(0.0)
        svDataDict = riskModel.loadSpecificRisks(rmi, modelDB)
        
        # Write factor covariance
        outFileName = '%s/%s.%04d%02d%02d.cov' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        self.writeFactorCov(d, factorCov, factors, outFile)
        outFile.close()
        
        # write exposures and specific risk, or just get the list of assets
        outFileName = '%s/%s.%04d%02d%02d.exp' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        (exposureAssets, exposureAssetIDs) = self.writeExposures(
            d, expM, svDataDict, outFile)
        outFile.close()
        
        cashIssues = [i for i in modelDB.getActiveIssues(d) if i.isCashAsset()]
        for i in cashIssues:
            exposureAssetIDs.append(i)
        # write identifier mapping for exposure assets
        outFileName = '%s/%s.%04d%02d%02d.idm' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'wt', encoding='utf-8')
        self.writeIdentifierMapping(d, exposureAssetIDs, modelDB, marketDB,
                                    riskModel.rmg[0], outFile, riskModel=riskModel)
        outFile.close()
        # write factor returns
        outFileName = '%s/%s.%04d%02d%02d.ret' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        self.writeFactorReturns(d, riskModel, modelDB, outFile)
        outFile.close()
        # write factor hierarchy
        outFileName = '%s/%s.%04d%02d%02d.hry' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        self.writeIndustryHierarchy(d, riskModel, factors, modelDB, marketDB,
                                    outFile)
        outFile.close()
        # write asset attributes
        outFileName = '%s/%s.%04d%02d%02d.att' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        self.writeAssetAttributes(d, riskModel, exposureAssets, expM, factors,
                                  factorCov, svDataDict, modelDB, marketDB, outFile)
        outFile.close()
        # write currency rate information
        outFileName = '%s/%s.%04d%02d%02d.cur' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        self.writeCurrencyRates(d, marketDB, modelDB, riskModel.rmg[0], outFile)
        outFile.close()
        # write Goldman model file
        inFileName = "/axioma/operations/daily/Goldman/GSTCM%04d%02d%02d-%s.csv" % (d.year, d.month, d.day, riskModel.rmg[0].mnemonic)
        outFileName = '%s/%s.%04d%02d%02d.gss' % (options.targetDir, mnemonic,
                                                  d.year, d.month, d.day)
        outFile = open(outFileName, 'w')
        self.writeGSFile(marketDB, modelDB, d, inFileName, outFile)
        outFile.close()


class FlatFilesGeneric(FlatFilesV3):
    """Class to create flat files compatible with AxiomaPortfolio's
    Delimited Factor Model DAO.  Note that only exposures and factor
    covariances are written.  Specific covariance, for example, is
    not supported.
    """
    
    def writeExposures(self, date, expMatrix, cashAssets, estU, svDict, options, outFile):
        """Write a triplet exposure file to outFile.
        """
        logging.debug("Writing exposures file")
        factorIdxNameMap = dict(zip(expMatrix.getFactorIndices(), expMatrix.getFactorNames()))
        outFile.write('|%s\n' % (
            '|'.join(factorIdxNameMap[i] for i in range(len(expMatrix.getFactorNames())))))
        mat = expMatrix.getMatrix()
        assetIdxMap = dict()
        for (aIdx, asset) in enumerate(expMatrix.getAssets()):
            assetIdxMap[asset] = aIdx
        if options.newRiskFields:
            allAssets = sorted(expMatrix.getAssets())
            for currencyFactor in expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor):
                cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
                allAssets.append(cashSubIssue)
            allAssets = sorted(allAssets)
        else:
            allAssets = expMatrix.getAssets()
        for asset in allAssets:
            if svDict != None and asset not in svDict and asset not in cashAssets:
                continue
            modelID = asset.getModelID()
            outFile.write(modelID.getPublicID())
            if asset in cashAssets:
                currencyFactor = modelID.getIDString().split('_')[1]
                for f in expMatrix.getFactorNames():
                    if currencyFactor == f:
                        outFile.write('|1')
                    else:
                        outFile.write('|0')
                outFile.write('\n')
                continue
            aIdx = assetIdxMap[asset]
            for fval in mat[:,aIdx]:
                if fval is not ma.masked:
                    outFile.write('|%.8g' % fval)
                else:
                    outFile.write('|0')
            outFile.write('|%.8g' % svDict[asset])
            outFile.write('\n')
        # add exposures to currency factors for corresponding currency assets
        if not self.vanilla and not options.newRiskFields:
            for currencyFactor in expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor):
                cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
                assert(cashSubIssue in cashAssets)
                outFile.write('%s' % cashSubIssue.getModelID().getPublicID())
                for f in expMatrix.getFactorNames():
                    if currencyFactor == f:
                        outFile.write('|1')
                    else:
                        outFile.write('|')
                outFile.write('|0.0')
                outFile.write('\n')
    
    def writeFactorCov(self, date, covMatrix, factors, options, outFile):
        """Write a pipe delimited factor covariance file to outFile.
        """
        outFile.write('%s\n' % (
            '|'.join([f.name for f in factors])))
        for i in range(len(factors)):
            outFile.write('|'.join('%.12g' % covMatrix[i,j] for j in range(len(factors))))
            outFile.write('\n')
    
    def writeDay(self, options, rmi, riskModel, modelDB, marketDB):
        d = rmi.date
        self.dataDate_ = rmi.date
        self.createDate_ = modelDB.revDateTime
        mnemonic = riskModel.mnemonic
        if options.writeCov:
            (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
                rmi, modelDB)
        if options.writeExp:
            expM = riskModel.loadExposureMatrix(rmi, modelDB)
            (svDataDict, specCov) = riskModel.loadSpecificRisks(rmi, modelDB)
            cashAssets = set()
            estU = list()
            exposureAssets = [n for n in expM.getAssets() if n in svDataDict]
            # exclude assets which shouldn't be extracted
            excludes = modelDB.getProductExcludedSubIssues(self.dataDate_)
            if options.preliminary or not rmi.is_final:
                excludes.extend(modelDB.getDRSubIssues(rmi))
            exposureAssets = list(set(exposureAssets) - set(excludes))
            exposureAssetIDs = [si.getModelID() for si in exposureAssets]
       
        shortdate='%04d%02d%02d' % (d.year, d.month, d.day)
        target=options.targetDir
        if options.appendDateDirs:
            target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass
        # Write factor covariance
        if options.writeCov:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='cov',dir=target)
            outFileName = '%s/%s.%04d%02d%02d.cov' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            outFile = open(tmpfilename, 'w')
            self.writeFactorCov(d, factorCov, factors, options, outFile)
            outFile.close()
            logging.info("Move covariance file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        
        # write exposures
        if options.writeExp:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='exp',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            
            outFileName = '%s/%s.%04d%02d%02d.exp' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            logging.info("Writing to %s", tmpfilename)
            #outFile = open(outFileName, 'w')
            outFile=open(tmpfilename,'w')
            self.writeExposures(d, expM, cashAssets, estU, svDataDict, options, outFile)
            outFile.close()
            logging.info("Move exposures file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)

class FlatFilesPhoenixCO(FlatFilesV3):
    """Class to create flat files for Phoenix commodities model
    """
    vanilla = False
    def __init__(self):
        self.issueSeriesMap = dict()
        self.seriesCurrencyMap = dict()
        self.seriesExchangeMap = dict()
        self.seriesNameMap = dict()
        self.seriesTickerMap = dict()

    def buildESTUWeights(self, modelDB, marketDB, estU, date, riskModel):
        """Retrieves estimation universe weights directly from rmi_nestu table,
        """
        modelDB.dbCursor.execute("""SELECT sub_issue_id, weight FROM rmi_nestu WHERE
        rms_id=:rms_id AND dt=:dt""", rms_id=riskModel.rms_id, dt=date)
        weightDict = dict(r for r in modelDB.dbCursor.fetchall())
        fullSum = sum(weightDict.values())
        estuWeights = dict()
        for subIssue in estU:
            estuWeights[subIssue] = weightDict.get(subIssue.getSubIDString())/fullSum
        return estuWeights

    def writeAssetAttributes(self, date, riskModelGroups, subIssues,
                             modelDB, marketDB, options, outFile, rollOverInfo=None,
                             useNonCheatHistoricBetas=True, riskModel=None):
        """Write asset attributes per risk model group.
        rollOverInfo: if present it is a map of SubIssue to Boolean. If a
           sub-issue is present in the map then its value should be
           used to determine roll-over status instead of the price history.
        useNonCheatHistoricBetas: if true, use the correct historic betas,
           otherwise use the old-style betas with the incorrect market return definition.
        """
        logging.debug("Writing asset attributes for %d assets", len(subIssues))
        self.writeDateHeader(options, outFile)
        outFile.write('#Columns: AxiomaID|Currency|RolloverFlag|Future Price'
                      '|Open Interest|20-Day ADV|1-Day Return|Historical Beta'
                      '|Cumulative Return|Axioma Series ID|Contract Year|Contract Month'
                      '|Last Trade Date\n')
        outFile.write('#Type: ID|NA|Set|Attribute|Attribute'
                      '|Attribute|Attribute|Attribute'
                      '|NA|Attribute|Attribute|Attribute|Attribute\n')
        outFile.write('#Unit: ID|Text|NA|CurrencyPerShare|Currency'
                      '|Currency|Percent|Number'
                      '|Number|ID|Number|Number|Date\n')
        rmgIdMap = dict([(rmg.rmg_id, rmg) for rmg in riskModelGroups])
        rmgSubIssues = dict([(rmg, set()) for rmg in riskModelGroups])
        subIssuesRMG = dict()
        modelDB.dbCursor.execute("""SELECT issue_id, rmg_id, from_dt
           FROM sub_issue WHERE from_dt <= :date_arg
           AND :date_arg < thru_dt""", date_arg=date)
        issueRMGDict = dict([(ModelID.ModelID(string=mid),
                              (rmg_id, fromDt.date()))
                             for (mid, rmg_id, fromDt)
                             in modelDB.dbCursor.fetchall()])
        currencyProvider = MarketDB.CurrencyProvider(marketDB, 10, None)
        riskModelGroups = list(riskModelGroups)
        issueFromDates = dict()
        for sid in subIssues:
            # if rmg not in rmgIdMap, then it is from a non-model
            # rmg. In that case add that rmg to riskModelGroups
            # and rmgIdMap
            (issueRMG, fromDt) = issueRMGDict[sid.getModelID()]
            if issueRMG not in rmgIdMap:
                newRMG = modelDB.getRiskModelGroup(issueRMG)
                rmgIdMap[newRMG.rmg_id] = newRMG
                rmgSubIssues[newRMG] = set()
                riskModelGroups.append(newRMG)
            rmg = rmgIdMap.get(issueRMG)
            rmgSubIssues[rmg].add(sid)
            subIssuesRMG[sid] = rmg
            issueFromDates[sid] = fromDt
        self.writeAssetData(outFile, riskModelGroups, subIssues, rmgSubIssues, subIssuesRMG,
                            currencyProvider, date, options, modelDB, marketDB,
                            issueFromDates, rollOverInfo=rollOverInfo,
                            useNonCheatHistoricBetas=useNonCheatHistoricBetas)

    def writeAssetData(self, outFile, rmgs, allSubIssues, rmgSubIssues, subIssuesRMG,
                       currencyProvider, date, options, modelDB, marketDB,
                       issueFromDates, rollOverInfo, useNonCheatHistoricBetas):
        # rmgSubIssues is dict of rmg->set of subIssues
        if options.newRiskFields:
            allSubIssues = sorted(allSubIssues)
        if not self.seriesCurrencyMap:
            self.issueSeriesMap = modelDB.getFutureSeries(date, allSubIssues, marketDB)
            self.getSeriesData(date, allSubIssues, marketDB)
        (axidStr, modelMarketMap) = modelDB.getMarketDB_IDs(marketDB, [a.getModelID() for a in allSubIssues if not a.isCashAsset()])
        (cashIdStr, cashMarketMap) = modelDB.getMarketDB_IDs(marketDB, [a.getModelID() for a in allSubIssues if a.isCashAsset()])
        sidCurrencies = dict()
        seriesData = dict()
        for s in allSubIssues:
            if s.getModelID() in modelMarketMap:
                mktMap = [a for a in modelMarketMap[s.getModelID()] if a[0] <= date and a[1] > date]
                mktId = mktMap[0][2]
                if mktId in self.issueSeriesMap:
                    seriesData[s] = self.issueSeriesMap[mktId]
                    logging.debug('Series for %s is %s', s, self.issueSeriesMap[mktId])
                    seriesId = MarketID.MarketID(string=self.issueSeriesMap[mktId][0]) 
                    if seriesId in self.seriesCurrencyMap:
                        sidCurrencies[s] = marketDB.getCurrencyID(self.seriesCurrencyMap[seriesId], date)
                    else:
                        logging.warning('%s has no currency', seriesId)
                else:
                    logging.error('%s has no series', mktId)
            else:
                if not s.isCashAsset():
                    logging.error('%s has no market ID', s)
            
        sidCurrencies.update(modelDB.getTradingCurrency(
            date, [s for s in allSubIssues if s.isCashAsset()], marketDB, 'id'))

        currencyMap = marketDB.getCurrencyISOCodeMap()
        tradingCurrencyMap = dict((s, currencyMap.get(c)) for (s, c) in sidCurrencies.items() if c in currencyMap)
        
        missingCurrency = [sid for sid in allSubIssues if sid not in sidCurrencies]
        if len(missingCurrency) > 0:
            if options.notCrash:
                allSubIssues = [sid for sid in allSubIssues if sid not in missingCurrency]
                logging.warning('Missing trading currency on %s for %s',
                            date, ','.join([sid.getSubIDString() for sid
                                            in missingCurrency]))
            else:
                logging.fatal('Missing trading currency on %s for %s',
                        date, ','.join([sid.getSubIDString() for sid
                            in missingCurrency]))
                raise Exception('Missing trading currencies on %s' % date)
        histLen = 20
        allDays = set()
        rmgTradingDays = dict()
        sidRMGIndices = dict()
        adv20Dict = dict()
        ret1Dict = dict()
        for rmg in rmgs:
            dateList = set(modelDB.getDates([rmg], date, histLen))
            rmgTradingDays[rmg] = dateList
            allDays |= dateList
        allDays = sorted(allDays)
        sidIdxMap = dict([(sid, idx) for (idx, sid) in enumerate(allSubIssues)])
        dateIdxMap = dict([(d, idx) for (idx, d) in enumerate(allDays)])
        dailyVolumes = modelDB.loadVolumeHistory(
            allDays, allSubIssues, sidCurrencies)
        dailyVolumes.data = Matrices.fillAndMaskByDate(
            dailyVolumes.data, [issueFromDates[sid] for sid in allSubIssues],
            allDays)
        # RLG, 11/13/08
        # just get the latest (1-day) return
        totalReturns = modelDB.loadTotalLocalReturnsHistory(
            [date], allSubIssues)
        totalReturns.data = totalReturns.data.filled(0.0)
        ucp = modelDB.loadUCPHistory(allDays, allSubIssues, sidCurrencies)
        rawUCP = modelDB.loadRawUCPHistory([date], allSubIssues)
        latestUCP = Matrices.allMasked((len(allSubIssues),))
        cumReturns = modelDB.loadCumulativeReturnsHistory(
            [date, date - datetime.timedelta(1), date - datetime.timedelta(2),
             date - datetime.timedelta(3)], allSubIssues)
        latestCumReturns = cumReturns.data[:,0]
        for i in range(3):
            latestCumReturns = ma.where(ma.getmaskarray(latestCumReturns),
                                        cumReturns.data[:,i+1],
                                        latestCumReturns)
        openInterest = modelDB.loadOpenInterestHistory(marketDB,
            allDays, allSubIssues)
        latestOpenInterest = Matrices.allMasked((len(allSubIssues),))
        if useNonCheatHistoricBetas:
            histbetas = modelDB.getPreviousHistoricBeta(date, allSubIssues)
        else:
            histbetas = modelDB.getPreviousHistoricBetaOld(date, allSubIssues)
        for rmg in rmgs:
            subIssues = sorted(rmgSubIssues[rmg])
            if options.notCrash:
                subIssueDict = dict((s, -1) for s in allSubIssues)
                droppedSubIssues = [sid for sid in subIssues if sid not in subIssueDict]
                if len(droppedSubIssues) > 0:
                    droppedDict = dict((s, -1) for s in droppedSubIssues)
                    subIssues = [sid for sid in subIssues if sid not in droppedDict]
                    for sid in droppedSubIssues:
                        logging.warning('Dropped subissue: %s for RMG: %s on Date: %s',
                                sid.getSubIDString(), rmg.mnemonic, date)
            rmgSidIndices = [sidIdxMap[sid] for sid in subIssues]
            sidRMGIndices.update(dict([(sid, idx) for (idx, sid) in enumerate(subIssues)]))
            if len(rmgSidIndices) == 1:
                # Add another asset to prevent numpy from converting
                # arrays with only one element to a number
                rmgSidIndices.append(0)
            tradingDays = sorted(rmgTradingDays[rmg], reverse=True)
            logging.debug('Writing data for %d assets in %s', len(subIssues), rmg)
            for sid in subIssues:
                sidIdx = sidIdxMap[sid]
                for tDate in tradingDays:
                    tIdx = dateIdxMap[tDate]
                    if not ucp.data[sidIdx,tIdx] is ma.masked:
                        latestUCP[sidIdx] = ucp.data[sidIdx,tIdx]
                        break
            for sid in subIssues:
                sidIdx = sidIdxMap[sid]
                for tDate in tradingDays:
                    tIdx = dateIdxMap[tDate]
                    if not openInterest.data[sidIdx,tIdx] is ma.masked:
                        latestOpenInterest[sidIdx] = openInterest.data[sidIdx,tIdx]
                        break
            tradingDayIndices = sorted([dateIdxMap[td] for td in tradingDays])
            rmgDailyVolumes = ma.take(dailyVolumes.data, rmgSidIndices, axis=0)
            rmgDailyVolumes = ma.take(rmgDailyVolumes, tradingDayIndices, axis=1)
            rmgReturns = ma.take(totalReturns.data, rmgSidIndices, axis=0)
            adv20Dict[rmg] = ma.average(rmgDailyVolumes[:,-20:], axis=1)
            ret1Dict[rmg] = 100.0 * rmgReturns[:,-1]
            if not options.newRiskFields:
                for (rmgIdx, sid) in enumerate(subIssues):
                    idx = sidIdxMap[sid]
                    self.writeAssetLine(outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                                        rollOverInfo, ucp, rawUCP, latestUCP,
                                        latestOpenInterest, adv20Dict[rmg], ret1Dict[rmg],
                                        histbetas, latestCumReturns, seriesData.get(sid))
        if options.newRiskFields:
            for (idx, sid) in enumerate(allSubIssues):
                rmg = subIssuesRMG[sid]
                rmgIdx = sidRMGIndices[sid]
                self.writeAssetLine(outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                                    rollOverInfo, ucp, rawUCP, latestUCP,
                                    latestOpenInterest, adv20Dict[rmg], ret1Dict[rmg],
                                    histbetas, latestCumReturns, seriesData.get(sid))
    
    def getSeriesData(self, date, seriesAssets, marketDB):
        INCR=200
        argList = [(':axid%d' % i) for i in range(INCR)]
        query = """SELECT ffa.id, ffa.stock_exchange_id, ffa.name, cr.code, ffa.external_ticker
        FROM future_family_attr_active_int ffa
        JOIN currency_ref cr ON cr.id=ffa.currency_id
        WHERE ffa.id IN (%(args)s)
        AND ffa.from_dt <= :dt AND ffa.thru_dt > :dt""" % {
        'args': ','.join('%s' % arg for arg in argList)}
        defaultDict = dict([(i, None) for i in argList])
        exchanges = dict()
        currencies = dict()
        names = dict()
        for axidChunk in listChunkIterator(seriesAssets, INCR):
            myDict = dict(defaultDict)
            myDict.update({'dt': date})
            updateDict = dict(zip(argList, [a.getIDString() for a in axidChunk]))
            myDict.update(updateDict)
            marketDB.dbCursor.execute(query, myDict)
            r = marketDB.dbCursor.fetchall()
            for (axid, classification, name, cur_code, ticker) in r:
                mktid = MarketID.MarketID(string=axid)
                self.seriesExchangeMap[mktid] = marketDB.getClassificationByID(classification)
                self.seriesCurrencyMap[mktid] = cur_code
                self.seriesNameMap[mktid] = name
                self.seriesTickerMap[mktid] = ticker

    def writeIdentifierMapping(self, d, exposureAssets, modelDB, marketDB,
                               options, outFile, outFile_nosedol=None,
                               outFile_nocusip=None, outFile_neither=None, riskModel=None):
        (axidStr, modelMarketMap) = modelDB.getMarketDB_IDs(marketDB, [a for a in exposureAssets if not a.isCashAsset()])
        (cashIdStr, cashMarketMap) = modelDB.getMarketDB_IDs(marketDB, [a for a in exposureAssets if a.isCashAsset()])
        tickerMap = modelDB.getFutureTickers(d, exposureAssets, marketDB)
        tickerMap.update(modelDB.getIssueTickers(d, [a for a in exposureAssets if a.isCashAsset], marketDB))
        regionFamily = marketDB.getClassificationFamily('REGIONS')
        regionMembers = marketDB.getClassificationFamilyMembers(regionFamily)
        marketMember = [i for i in regionMembers if i.name=='Market'][0]
        marketRev = marketDB.getClassificationMemberRevision(
            marketMember, d)
        countryMap = modelDB.getMktAssetClassifications(
            marketRev, exposureAssets, d, marketDB, level=1)
        currencyMap = modelDB.getFutureTradingCurrency(d, exposureAssets,
                                                       marketDB, 'code')
        self.issueSeriesMap = modelDB.getFutureSeries(d, exposureAssets, marketDB)
        self.seriesAssets = set(MarketID.MarketID(string=v[0]) for v in self.issueSeriesMap.values())
        self.getSeriesData(d, list(self.seriesAssets), marketDB)
        # uppercase exchange name
        self.exchangeMap = dict([(a, i.description.upper()) for (a, i) in self.seriesExchangeMap.items()])
        nameMap = modelDB.getFutureNames(d, exposureAssets, marketDB)
        cashNameMap = modelDB.getIssueNames(d, list(cashMarketMap.keys()), marketDB)
        outFiles = [outFile, outFile_nosedol, outFile_nocusip, outFile_neither]
        for f in outFiles:
            if f:
                self.writeDateHeader(options, f)
                f.write('#Columns: AxiomaID|Ticker|Description|Country|Currency')
                f.write('\n')
        for a in exposureAssets:
            if a.isCashAsset():
                for (idx, f) in enumerate(outFiles):
                    if f:
                        f.write('%s|%s|%s|%s|%s\n' % \
                                (a.getPublicID(), tickerMap.get(a, ''),
                                 cashNameMap.get(a,''),
                                 '', a.getCashCurrency()))
                continue
            else:
                if a in currencyMap:
                    currency = currencyMap[a]
                else:
                    if options.notCrash:
                        logging.warning('Missing trading currency on %s for %s',
                                d, a)
                    else:
                        logging.fatal('Missing trading currency on %s for %s',
                                    d, a)
                        raise Exception('Missing trading currencies on %s' % d)
                if a in countryMap:
                    country = countryMap[a].classification.code
                else:
                    if options.notCrash:
                        logging.warning('Missing trading country on %s for %s',
                                d, a)
                    else:
                        logging.fatal('Missing trading country on %s for %s',
                                    d, a)
                        raise Exception('Missing trading country on %s' % d)
            for m in modelMarketMap[a]:
                if m[0] <= d and m[1] > d:
                    mktId = m[2]
            if not mktId:
                logging.fatal("No current market ID for %s", a)
                raise ValueError("No current market ID for %s" % a)
            for (idx, f) in enumerate(outFiles):
                if f:
                    outString = '%s|%s|%s|%s|%s' % \
                            (a.getPublicID(), tickerMap.get(a, ''),
                             nameMap.get(a, ''),
                             country, currency)
                    f.write(outString)
                    f.write('\n')

    def writeSeriesFile(self, options, seriesFile):
        self.writeDateHeader(options, seriesFile)
        seriesFile.write('#Columns: Series ID|Description|Exchange|Exchange Ticker|Currency\n')
        for series in sorted(self.seriesAssets):
            # write series information
            seriesFile.write('%s|%s|%s|%s|%s\n' % \
                             (series.getIDString(),
                              self.seriesNameMap.get(series,''),
                              self.exchangeMap.get(series,''),
                              self.seriesTickerMap.get(series,''),
                              self.seriesCurrencyMap.get(series,''),
                              )
                             )
        seriesFile.close()
        
    def writeDay(self, options, rmi, riskModel, modelDB, marketDB):
        super(FlatFilesPhoenixCO, self).writeDay(options, rmi, riskModel, modelDB, marketDB)
        if options.writeAssetIdm:
            d = rmi.date
            mnemonic = riskModel.mnemonic
            shortdate='%04d%02d%02d' % (d.year, d.month, d.day)
            target=options.targetDir
            if options.appendDateDirs:
                target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
                try:
                    os.makedirs(target)
                except OSError as e:
                    if e.errno != 17:
                        raise
                    else:
                        pass
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='ser',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            logging.info("Writing to %s", tmpfilename)
            seriesFile = open(tmpfilename, 'w')
            outFileName = '%s/%s.%04d%02d%02d.ser' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            self.writeSeriesFile(options, seriesFile)
            logging.info("Move series file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)

    def writeAssetLine(self, outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                       rollOverInfo, ucp, rawUCP, latestUCP,
                       latestMarketCap, adv20, ret1,
                       histbetas, latestCumReturns, seriesData):
        line = io.StringIO()
        modelID = sid.getModelID()
        currency = tradingCurrencyMap.get(sid)
        if currency is None:
            logging.warning("No currency for %s", sid.getModelID())
            currency = ''
        
        line.write('%s|%s|' % (modelID.getPublicID(), currency))
        if rollOverInfo is not None and sid in rollOverInfo:
            flagRollOver =  rollOverInfo[sid]
        else:
            missingPrice = ucp.data[idx, -1] is ma.masked
            stalePrice = rawUCP[idx,0] is not None and rawUCP[idx,0].price_marker == 3
            flagRollOver = missingPrice or stalePrice
        if flagRollOver:
            line.write('*')
        mktId = None
        line.write('|%s' % numOrNull(latestUCP[idx], '%.4f', 0.0001, '%.4e'))
        line.write('|%s' % numOrNull(latestMarketCap[idx], '%.0f'))
        line.write('|%s' % numOrNull(adv20[rmgIdx], '%.0f'))
        line.write('|%s' % numOrNull(ret1[rmgIdx], '%.4f'))
        line.write('|%s' % numOrNull(histbetas.get(sid), '%.4f'))
        line.write('|%s' % numOrNull(latestCumReturns[idx], '%.6g'))
        line.write('|%s|%s|%s|%s' % (seriesData and (seriesData[0],
                                                     seriesData[2].year,
                                                     seriesData[2].month,
                                                     seriesData[1].date())
                                     or ('', '', '', '')))
        line.write('\n')
        outFile.write(line.getvalue())
    
class FlatFilesPhoenixFI(FlatFilesV3):
    """Class to create flat files for Phoenix fixed-income model
    """
    vanilla = False
    def __init__(self):
        #FlatFilesV3.__init__()
        self.macDB = pymssql.connect(user='intercom', password='inter2011com', host='dev-mac-db')
        self.macDB.autocommit(True)
        self.marketDataIdCache = FromThruCache()
        self.marketDataIdReverseCache = FromThruCache(useModelID=False)
        self.cusipCache = FromThruCache()
        self.sedolCache = FromThruCache()
        self.isinCache = FromThruCache()
        self.nameCache = FromThruCache()
        self.issuerCache = FromThruCache()
        self.currencyCache = FromThruCache()
        self.countryCache = FromThruCache()
        self.maturityCache = FromThruCache()
        self.typeCache = FromThruCache()
        self.gicsCache = FromThruCache()
        self.categoryCache = FromThruCache()
        self.ratingsCache = FromThruCache()
    
    def getExposureAssets(self, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB):
        exposureAssets = getExposureAssets(self.dataDate_, expMatrix, None, cashAssets, rmi, options, modelDB, marketDB)
        
        return exposureAssets 

    def writeExposures(self, date, expMatrix, cashAssets, estU, svDict, options, outFile):
        """Write a triplet exposure file to outFile.
        """
        logging.debug("Writing exposures file")
        self.writeDateHeader(options, outFile)
        outFile.write('#Columns: AxiomaID|%s\n' % (
            '|'.join(expMatrix.getFactorNames())))
        outFile.write('#Type: AxiomaID|%s\n' % (
            '|'.join([getFactorType(expMatrix, f).name for f
                      in expMatrix.getFactorNames()])))
        mat = expMatrix.getMatrix()
        assetIdxMap = dict()
        for (aIdx, asset) in enumerate(expMatrix.getAssets()):
            assetIdxMap[asset] = aIdx
        if options.newRiskFields:
            allAssets = sorted(expMatrix.getAssets())
            for currencyFactor in expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor):
                cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
                allAssets.append(cashSubIssue)
            allAssets = sorted(allAssets)
        else:
            allAssets = expMatrix.getAssets()
        for asset in allAssets:
            modelID = asset.getModelID()
            aIdx = assetIdxMap[asset]
            if ma.count(mat[:,aIdx]) == 0:
                continue
            outFile.write(modelID.getPublicID())
            if asset in cashAssets:
                currencyFactor = modelID.getIDString().split('_')[1]
                for f in expMatrix.getFactorNames():
                    if currencyFactor == f:
                        outFile.write('|1')
                    else:
                        outFile.write('|')
                outFile.write('\n')
                continue
            for fval in mat[:,aIdx]:
                if fval is not ma.masked:
                    outFile.write('|%.8g' % fval)
                else:
                    outFile.write('|')
            outFile.write('\n')
        # add exposures to currency factors for corresponding currency assets
        if not self.vanilla and not options.newRiskFields:
            for currencyFactor in expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor):
                cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
                assert(cashSubIssue in cashAssets)
                outFile.write('%s' % cashSubIssue.getModelID().getPublicID())
                for f in expMatrix.getFactorNames():
                    if currencyFactor == f:
                        outFile.write('|1')
                    else:
                        outFile.write('|')
                outFile.write('\n')
    
    def writeAssetRiskAttributes(self, date, riskModel, exposureAssets, expM,
                                 svDict, estU, options, modelDB, marketDB, outFile):
        rmi = riskModel.getRiskModelInstance(date, modelDB)
        if not self.vanilla:
            predbetas = modelDB.getRMIPredictedBeta(rmi)
            #assert(len(predbetas)>0)
            if len(riskModel.industries) > 0:
                classifications = riskModel.industryClassification.getAssetConstituents(modelDB, exposureAssets, date)
                hasIndustries = True
            else:
                classifications = set()
                hasIndustries = False
            estUWeightdict = modelDB.getRMIESTUWeights(rmi)
            specRtnsMatrix = modelDB.loadSpecificReturnsHistory(rmi.rms_id, exposureAssets, [date])
            specRtnsMatrix.data *= 100
            totalRisks = modelDB.getRMITotalRisk(rmi)
        logging.debug("Writing risk attributes for %d assets ", len(exposureAssets))
        self.writeDateHeader(options, outFile)
        if self.vanilla:
            outFile.write('#Columns: AxiomaID|Specific Risk\n')
            outFile.write('#Type: ID|Attribute\n')
            outFile.write('#Unit: ID|Percent\n')
        else:
            if hasIndustries:
                outFile.write('#Columns: AxiomaID|Specific Risk|Predicted Beta|Industry Source'
                              '|Estimation Universe')
                if options.newRiskFields:
                    outFile.write('|Estimation Universe Weight|Specific Return|Total Risk')
                outFile.write('\n')
                outFile.write('#Type: ID|Attribute|Attribute|Attribute|Set')
                if options.newRiskFields:
                    outFile.write('|Attribute|Attribute|Attribute')
                outFile.write('\n')
                outFile.write('#Unit: ID|Percent|Number|Text|NA')
                if options.newRiskFields:
                    outFile.write('|Percent|Percent|Percent')
                outFile.write('\n')
            else:
                outFile.write('#Columns: AxiomaID|Specific Risk|Predicted Beta'
                              '|Estimation Universe')
                if options.newRiskFields:
                    outFile.write('|Estimation Universe Weight|Specific Return|Total Risk')
                outFile.write('\n')
                outFile.write('#Type: ID|Attribute|Attribute|Set')
                if options.newRiskFields:
                    outFile.write('|Attribute|Attribute|Attribute')
                outFile.write('\n')
                outFile.write('#Unit: ID|Percent|Number|NA')
                if options.newRiskFields:
                    outFile.write('|Percent|Percent|Percent')
                outFile.write('\n')
        # temporary set to speed lookup
        estUset = set(estU)
        for (idx, asset) in enumerate(exposureAssets):
            modelID = asset.getModelID()
            outFile.write('%s' % (modelID.getPublicID()))
            if asset in svDict:
                outFile.write('|%s' % numOrNull(100.0 * svDict.get(asset), '%.12g'))
            else:
                outFile.write('|')
            if self.vanilla:
                outFile.write('\n')
                continue
            outFile.write('|%s' % numOrNull(predbetas.get(asset), '%.4f'))
            if hasIndustries:
                # Get source for asset's industry
                source = ''
                if asset in classifications:
                    src_id = classifications[asset].src
                    source = 'Axioma'
                    if (src_id>=300 and src_id<=399):
                        source = 'GICS-Direct'
                outFile.write('|%s' % source)
            if asset in estUset:
                outFile.write('|*')
            else:
                outFile.write('|')
            if options.newRiskFields:
                # estu weight
                if asset in estUWeightdict and estUWeightdict[asset] != None:
                    estUWeightdict[asset] *= 100
                outFile.write('|%s' % numOrNull(estUWeightdict.get(asset), '%.4f'))
                # specific returns, total risk
                if asset in totalRisks and totalRisks[asset] != None:
                    totalRisks[asset] *= 100
                outFile.write('|%s|%s' % (numOrNull(specRtnsMatrix.data[idx, 0], '%4.10f'),
                                          numOrNull(totalRisks.get(asset), '%.4f')))
            outFile.write('\n')

    def cacheMarketDataIds(self, assetList):
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = self.marketDataIdCache.getMissingIds(issueList)
        cursor = self.macDB.cursor()
        INCR=1000
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        dataIdQuery = """SELECT AxiomaDataId, SecurityIdentifier, FromDate, ToDate FROM
        MarketData.dbo.InstrumentXref WHERE SecurityIdentifierType='MODELDB_ID' AND
        SecurityIdentifier IN  (%(keys)s)
        """ % {'keys': ','.join(['%%(%s)s' % arg for arg in keyList])}
        for idChunk in listChunkIterator(missingIds, INCR):
            valueDict = defaultDict.copy()
            valueDict.update(dict(zip(keyList, [i.getIDString() for i in idChunk])))
            cursor.execute(dataIdQuery, valueDict)
            r = cursor.fetchmany(INCR)
            while len(r) > 0:
                for (val, aid, from_dt, thru_dt) in r:
                    mid = ModelID.ModelID(string=aid)
                    mdInfo = Utilities.Struct()
                    mdInfo.fromDt = Utilities.parseISODate(from_dt)
                    mdInfo.thruDt = Utilities.parseISODate(thru_dt)
                    mdInfo.id = val
                    self.marketDataIdCache.addAssetValues(mid, [mdInfo])
                    mdInfo = Utilities.Struct(mdInfo)
                    mdInfo.id = aid
                    self.marketDataIdReverseCache.addAssetValues(val, [mdInfo])
                r = cursor.fetchmany(INCR)
    
    def loadMarketDataCache(self, identifierType, date, assetList, cache):
        xrefIds = ['CUSIP','SEDOL','ISIN']
        views = {'issuername': 'v_MacIssuerNameInt',
                 'currency': 'v_MacCurrencyInt',
                 'name': 'v_MacNameInt',
                 'country': 'v_MacCountryInt',
                 'exchange': 'v_MacExchangeInt',
                 'maturity': 'v_MacMaturityInt',
                 'issuertype': 'v_MacIssuerIssuerTypeInt',
                 'gicslevel1': 'v_MacIssuerGicsLevel1Int',
                 'category': 'v_MacIssuerCategoryInt',
                 'compositerating': 'v_MacIssuerCompositeRatingInt',
                 }
        self.cacheMarketDataIds(assetList)
        issueList = list()
        for a in assetList:
            if isinstance(a, SubIssue):
                issueList.append(a.getModelID())
            else:
                issueList.append(a)
        missingIds = cache.getMissingIds(issueList)
        for m in missingIds:
            cache.addAssetValues(m, [])
        notFoundIds = dict((i,1) for i in self.marketDataIdCache.getMissingIds(issueList))
        cursor = self.macDB.cursor()
        INCR=1000
        keyList = ['key%d' % i for i in range(INCR)]
        defaultDict = dict([(a, None) for a in keyList])
        if identifierType.upper() in xrefIds:
            dataQuery = """SELECT AxiomaDataId, SecurityIdentifier, FromDate, ToDate FROM
            MarketData.dbo.InstrumentXref WHERE SecurityIdentifierType='%(idType)s' AND
            AxiomaDataId IN  (%(keys)s)
            """ % {'idType': identifierType.upper(), 
                   'keys': ','.join(['%%(%s)s' % arg for arg in keyList])}
        elif identifierType.lower() in views:
            dataQuery = """SELECT AxiomaId, %(idType)s, from_dt, thru_dt FROM
            MarketData.dbo.%(view)s WHERE 
            AxiomaId IN  (%(keys)s)
            """ % {'idType': identifierType.lower(),
                   'view': views[identifierType.lower()], 
                   'keys': ','.join(['%%(%s)s' % arg for arg in keyList])}
        else:
            raise KeyError('Unknown attribute %s' % identifierType)
        for idChunk in listChunkIterator(missingIds, INCR):
            valueDict = defaultDict.copy()
            keyVals = [self.marketDataIdCache.getAssetValue(i, date) for i in idChunk if i not in notFoundIds]
            valueDict.update(dict(zip(keyList, [k.id for k in keyVals])))
            cursor.execute(dataQuery, valueDict)
            r = cursor.fetchmany(100)
            while len(r) > 0:
                for (dataId, val, from_dt, thru_dt) in r:
                    aid = self.marketDataIdReverseCache.getAssetValue(dataId, date)
                    mid = ModelID.ModelID(string=aid.id)
                    info = Utilities.Struct()
                    info.fromDt = Utilities.parseISODate(from_dt)
                    info.thruDt = Utilities.parseISODate(thru_dt)
                    info.id = val
                    cache.addAssetValues(mid, [info])
                r = cursor.fetchmany(100)
    
    def getMarketDataCUSIPs(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('CUSIP', date, assetList, self.cusipCache)
        cusips = [(mid, self.cusipCache.getAssetValue(mid, date))
                  for mid in assetList]
        cusips = dict([(i, j.id) for (i,j) in cusips if j is not None])
        return cusips
    
    def getMarketDataSEDOLs(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('SEDOL', date, assetList, self.sedolCache)
        sedols = [(mid, self.sedolCache.getAssetValue(mid, date))
                  for mid in assetList]
        sedols = dict([(i, j.id) for (i,j) in sedols if j is not None])
        return sedols
    
    def getMarketDataISINs(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('ISIN', date, assetList, self.isinCache)
        isins = [(mid, self.isinCache.getAssetValue(mid, date))
                  for mid in assetList]
        isins = dict([(i, j.id) for (i,j) in isins if j is not None])
        return isins
    
    def getMarketDataIssuers(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('issuername', date, assetList, self.issuerCache)
        issuers = [(mid, self.issuerCache.getAssetValue(mid, date))
                  for mid in assetList]
        issuers = dict([(i, j.id) for (i,j) in issuers if j is not None])
        return issuers
    
    def getMarketDataNames(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('name', date, assetList, self.nameCache)
        names = [(mid, self.nameCache.getAssetValue(mid, date))
                  for mid in assetList]
        names = dict([(i, j.id) for (i,j) in names if j is not None])
        return names
    
    def getMarketDataCurrencies(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('currency', date, assetList, self.currencyCache)
        currencies = [(mid, self.currencyCache.getAssetValue(mid, date))
                      for mid in assetList]
        currencies = dict([(i, j.id) for (i,j) in currencies if j is not None])
        return currencies
    
    def getMarketDataIssuerType(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('issuerType', date, assetList, self.typeCache)
        issuers = [(mid, self.typeCache.getAssetValue(mid, date))
                      for mid in assetList]
        issuers = dict([(i, j.id) for (i,j) in issuers if j is not None])
        return issuers
    
    def getMarketDataMaturity(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('maturity', date, assetList, self.maturityCache)
        maturities = [(mid, self.maturityCache.getAssetValue(mid, date))
                      for mid in assetList]
        maturities = dict([(i, j.id) for (i,j) in maturities if j is not None])
        return maturities
    
    def getMarketDataGICS(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('gicslevel1', date, assetList, self.gicsCache)
        gics = [(mid, self.gicsCache.getAssetValue(mid, date))
                      for mid in assetList]
        gics = dict([(i, j.id) for (i,j) in gics if j is not None])
        return gics
    
    def getMarketDataCategory(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('category', date, assetList, self.categoryCache)
        categories = [(mid, self.categoryCache.getAssetValue(mid, date))
                      for mid in assetList]
        categories = dict([(i, j.id) for (i,j) in categories if j is not None])
        return categories
    
    def getMarketDataRatings(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('compositerating', date, assetList, self.ratingsCache)
        ratings = [(mid, self.ratingsCache.getAssetValue(mid, date))
                      for mid in assetList]
        ratings = dict([(i, j.id) for (i,j) in ratings if j is not None])
        return ratings
    
    def getSubIssueCurrencies(self, modelDB, dateList, subIssueList):
        retval = dict()
        INCR=1000
        DINCR=10
        keyList = ['key%d' % i for i in range(INCR)]
        dtArgs = ['dt%d' % i for i in range(DINCR)]
        defaultDict = dict([(a, None) for a in keyList])
        dataIdQuery = """SELECT sub_issue_id, currency_id FROM sub_issue_data
        WHERE sub_issue_id IN  (%(keys)s) AND dt in (%(dtArgs)s)
        """ % {'keys': ','.join([':%s' % arg for arg in keyList]),
               'dtArgs': ','.join([':%s' % arg for arg in dtArgs])}
        defaultDict = dict([(i, None) for i in dtArgs]
                           + [(i, None) for i in keyList])
        sidStrings = sorted([sid.getSubIDString() for sid in subIssueList])
        for dateChunk in listChunkIterator(dateList, DINCR):
            myDateDict = dict(zip(dtArgs, dateChunk))
            for sidChunk in listChunkIterator(sidStrings, INCR):
                mySidDict = dict(zip(keyList, sidChunk))
                valueDict = dict(defaultDict)
                valueDict.update(myDateDict)
                valueDict.update(mySidDict)
                modelDB.dbCursor.execute(dataIdQuery, valueDict)
                r = modelDB.dbCursor.fetchmany()
                while r:
                    retval.update(dict((SubIssue(string=rec[0]), rec[1]) for rec in r))
                    r = modelDB.dbCursor.fetchmany()
        return retval
    
    def getMarketDataCountries(self, date, assetList):
        # get data for assets from the MAC SQL server database
        self.loadMarketDataCache('country', date, assetList, self.countryCache)
        countries = [(mid, self.countryCache.getAssetValue(mid, date))
                      for mid in assetList]
        countries = dict([(i, j.id) for (i,j) in countries if j is not None])
        return countries
        
    def writeIdentifierMapping(self, d, exposureAssets, modelDB, marketDB,
                               options, outFile, outFile_nosedol=None,
                               outFile_nocusip=None, outFile_neither=None):
        cusipMap = self.getMarketDataCUSIPs(d, exposureAssets)
        logging.debug('got %d cusips', len(cusipMap))
        sedolMap = self.getMarketDataSEDOLs(d, exposureAssets)
        logging.debug('got %d sedols', len(sedolMap))
        isinMap = self.getMarketDataISINs(d, exposureAssets)
        logging.debug('got %d isins', len(isinMap))
        nameMap = self.getMarketDataNames(d, exposureAssets)
        logging.debug('got %d names', len(nameMap))
        issuerMap = self.getMarketDataIssuers(d, exposureAssets)
        logging.debug('got %d issuers', len(issuerMap))
        currencyMap = self.getMarketDataCurrencies(d, exposureAssets)
        logging.debug('got %d currencies', len(currencyMap))
        countryMap = self.getMarketDataCountries(d, exposureAssets)
        logging.debug('got %d countries', len(countryMap))
        typeMap = self.getMarketDataIssuerType(d, exposureAssets)
        logging.debug('got %d issuertypes', len(typeMap))
        maturityMap = self.getMarketDataMaturity(d, exposureAssets)
        logging.debug('got %d maturities', len(maturityMap))
        gicsMap = self.getMarketDataGICS(d, exposureAssets)
        logging.debug('got %d GICS mappings', len(gicsMap))
        categoryMap = self.getMarketDataCategory(d, exposureAssets)
        logging.debug('got %d categories', len(categoryMap))
        ratingMap = self.getMarketDataRatings(d, exposureAssets)
        logging.debug('got %d ratings', len(ratingMap))
        outFiles = [outFile, outFile_nosedol, outFile_nocusip, outFile_neither]
        for f in outFiles:
            if f:
                self.writeDateHeader(options, f)
                f.write('#Columns: AxiomaID|ISIN|CUSIP|SEDOL(7)|Issuer|Name'
                        '|Country|Currency|IssuerType|Maturity|GICS (level 1)|Category|Composite Rating')
                f.write('\n')
        for a in exposureAssets:
            if a.isCashAsset():
                currency = a.getCashCurrency()
                country = ''
            else:
                if a in currencyMap:
                    currency = currencyMap[a]
                else:
                    currency = ''
                    if options.notCrash:
                        logging.warning('Missing trading currency on %s for %s',
                                d, a)
                    else:
                        logging.fatal('Missing trading currency on %s for %s',
                                    d, a)
                        raise Exception('Missing trading currencies on %s' % d)
                if a in countryMap:
                    country = countryMap[a]
                else:
                    country = ''
                    if options.notCrash:
                        logging.warning('Missing trading country on %s for %s',
                                d, a)
                    else:
                        logging.fatal('Missing trading country on %s for %s',
                                    d, a)
                        raise Exception('Missing trading country on %s' % d)
                type = typeMap.get(a, '')
                maturity = maturityMap.get(a, '')
                gics = gicsMap.get(a, '')
                category = categoryMap.get(a, '')
                rating = ratingMap.get(a, '')
            for (idx, f) in enumerate(outFiles):
                if f:
                    outString = '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' % \
                            (a.getPublicID(),
                             (idx == 0) and isinMap.get(a, '') or '',
                             (idx == 0 or idx == 1) and cusipMap.get(a, '') or '',
                             (idx == 0 or idx == 2) and zeroPad(sedolMap.get(a, ''), 7) or '',
                             issuerMap.get(a,''),
                             nameMap.get(a, ''),
                             country, currency, type, maturity, gics, category, rating)
                    f.write(outString)

    def writeAssetData(self, outFile, rmgs, allSubIssues, rmgSubIssues, subIssuesRMG,
                       currencyProvider, date, options, modelDB, marketDB,
                       issueFromDates, rollOverInfo, useNonCheatHistoricBetas):
        # rmgSubIssues is dict of rmg->set of subIssues
        if options.newRiskFields:
            allSubIssues = sorted(allSubIssues)
        histLen = 20
        allDays = set()
        rmgTradingDays = dict()
        for rmg in rmgs:
            dateList = set(modelDB.getDates([rmg], date, histLen))
            rmgTradingDays[rmg] = dateList
            allDays |= dateList
        allDays = sorted(allDays)
        sidCurrencies = self.getSubIssueCurrencies(modelDB, allDays, allSubIssues)
        currencyMap = marketDB.getCurrencyISOCodeMap()
        tradingCurrencyMap = dict((s, currencyMap.get(c)) for (s, c) in sidCurrencies.items() if c in currencyMap)
        missingCurrency = dict((sid,-1) for sid in allSubIssues if sid not in sidCurrencies)
        if len(missingCurrency) > 0:
            if options.notCrash:
                allSubIssues = [sid for sid in allSubIssues if sid not in missingCurrency]
                logging.warning('Missing trading currency on %s for %s',
                            date, ','.join([sid.getSubIDString() for sid
                                            in missingCurrency]))
            else:
                logging.fatal('Missing trading currency on %s for %s',
                        date, ','.join([sid.getSubIDString() for sid
                            in missingCurrency]))
                raise Exception('Missing trading currencies on %s' % date)
        sidRMGIndices = dict()
        adv20Dict = dict()
        ret1Dict = dict()
        sidIdxMap = dict([(sid, idx) for (idx, sid) in enumerate(allSubIssues)])
        dateIdxMap = dict([(d, idx) for (idx, d) in enumerate(allDays)])
        dailyVolumes = modelDB.loadVolumeHistory(
            allDays, allSubIssues, sidCurrencies)
        dailyVolumes.data = Matrices.fillAndMaskByDate(
            dailyVolumes.data, [issueFromDates[sid] for sid in allSubIssues],
            allDays)
        # RLG, 11/13/08
        # just get the latest (1-day) return
        totalReturns = modelDB.loadTotalLocalReturnsHistory(
            [date], allSubIssues)
        totalReturns.data = totalReturns.data.filled(0.0)
        ucp = modelDB.loadUCPHistory(allDays, allSubIssues, sidCurrencies)
        rawUCP = modelDB.loadRawUCPHistory([date], allSubIssues)
        latestUCP = Matrices.allMasked((len(allSubIssues),))
        cumReturns = modelDB.loadCumulativeReturnsHistory(
            [date, date - datetime.timedelta(1), date - datetime.timedelta(2),
             date - datetime.timedelta(3)], allSubIssues)
        latestCumReturns = cumReturns.data[:,0]
        for i in range(3):
            latestCumReturns = ma.where(ma.getmaskarray(latestCumReturns),
                                        cumReturns.data[:,i+1],
                                        latestCumReturns)
        marketCaps = modelDB.loadMarketCapsHistory(
            allDays, allSubIssues, sidCurrencies)
        latestMarketCap = Matrices.allMasked((len(allSubIssues),))
        if useNonCheatHistoricBetas:
            histbetas = modelDB.getPreviousHistoricBeta(date, allSubIssues)
        else:
            histbetas = modelDB.getPreviousHistoricBetaOld(date, allSubIssues)
        for rmg in rmgs:
            subIssues = sorted(rmgSubIssues[rmg])
            if options.notCrash:
                subIssueDict = dict((s, -1) for s in allSubIssues)
                droppedSubIssues = [sid for sid in subIssues if sid not in subIssueDict]
                if len(droppedSubIssues) > 0:
                    droppedDict = dict((s, -1) for s in droppedSubIssues)
                    subIssues = [sid for sid in subIssues if sid not in droppedDict]
                    for sid in droppedSubIssues:
                        logging.warning('Dropped subissue: %s for RMG: %s on Date: %s',
                                sid.getSubIDString(), rmg.mnemonic, date)
            rmgSidIndices = [sidIdxMap[sid] for sid in subIssues]
            sidRMGIndices.update(dict([(sid, idx) for (idx, sid) in enumerate(subIssues)]))
            if len(rmgSidIndices) == 1:
                # Add another asset to prevent numpy from converting
                # arrays with only one element to a number
                rmgSidIndices.append(0)
            tradingDays = sorted(rmgTradingDays[rmg], reverse=True)
            logging.debug('Writing data for %d assets in %s', len(subIssues), rmg)
            for sid in subIssues:
                sidIdx = sidIdxMap[sid]
                for tDate in tradingDays:
                    tIdx = dateIdxMap[tDate]
                    if not ucp.data[sidIdx,tIdx] is ma.masked:
                        latestUCP[sidIdx] = ucp.data[sidIdx,tIdx]
                        break
            for sid in subIssues:
                sidIdx = sidIdxMap[sid]
                for tDate in tradingDays:
                    tIdx = dateIdxMap[tDate]
                    if not marketCaps[sidIdx,tIdx] is ma.masked:
                        latestMarketCap[sidIdx] = marketCaps[sidIdx,tIdx]
                        break
            tradingDayIndices = sorted([dateIdxMap[td] for td in tradingDays])
            rmgDailyVolumes = ma.take(dailyVolumes.data, rmgSidIndices, axis=0)
            rmgDailyVolumes = ma.take(rmgDailyVolumes, tradingDayIndices, axis=1)
            rmgReturns = ma.take(totalReturns.data, rmgSidIndices, axis=0)
            adv20Dict[rmg] = ma.average(rmgDailyVolumes[:,-20:], axis=1)
            ret1Dict[rmg] = 100.0 * rmgReturns[:,-1]
            if not options.newRiskFields:
                for (rmgIdx, sid) in enumerate(subIssues):
                    idx = sidIdxMap[sid]
                    self.writeAssetLine(outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                                        rollOverInfo, ucp, rawUCP, latestUCP,
                                        latestMarketCap, adv20Dict[rmg], ret1Dict[rmg],
                                        histbetas, latestCumReturns)
        if options.newRiskFields:
            for (idx, sid) in enumerate(allSubIssues):
                rmg = subIssuesRMG[sid]
                rmgIdx = sidRMGIndices[sid]
                self.writeAssetLine(outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                                    rollOverInfo, ucp, rawUCP, latestUCP,
                                    latestMarketCap, adv20Dict[rmg], ret1Dict[rmg],
                                    histbetas, latestCumReturns)
    

def main():
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--ignore-missing", action="store_true",
                             default=False, dest="ignoreMissingModels",
                             help="don't fail if no model is available for a specified date")
    cmdlineParser.add_option("--version", action="store",
                             default=3, type='int', dest="formatVersion",
                             help="version of flat files to create")
    cmdlineParser.add_option("--file-format-version", action="store",
                             default=3.2, type='float', dest="fileFormatVersion",
                             help="version of flat file format to create")
    cmdlineParser.add_option("--no-cov", action="store_false",
                             default=True, dest="writeCov",
                             help="don't create .cov file")
    cmdlineParser.add_option("--no-exp", action="store_false",
                             default=True, dest="writeExp",
                             help="don't create .exp file")
    cmdlineParser.add_option("--no-ccy", action="store_false",
                             default=True, dest="writeCurrencies",
                             help="don't create Currencies.att file")
    cmdlineParser.add_option("--no-rsk", action="store_false",
                             default=True, dest="writeRiskAttributes",
                             help="don't create .rsk file")
    cmdlineParser.add_option("--no-isc", action="store_false",
                             default=True, dest="writeSpecificCovariances",
                             help="don't create .isc file")
    cmdlineParser.add_option("--no-ret", action="store_false",
                             default=True, dest="writeFactorReturns",
                             help="don't create .ret file")
    cmdlineParser.add_option("--add-iret", action="store_true",
                             default=False, dest="writeInternalFactorReturns",
                             help="output .iret file")
    cmdlineParser.add_option("--no-hry", action="store_false",
                             default=True, dest="writeFactorHry",
                             help="don't create .hry file")
    cmdlineParser.add_option("--no-idm", action="store_false",
                             default=True, dest="writeAssetIdm",
                             help="don't create .idm file")
    cmdlineParser.add_option("--no-att", action="store_false",
                             default=True, dest="writeAssetAtt",
                             help="don't create model.att file")
    cmdlineParser.add_option("--no-gss", action="store_false",
                             default=True, dest="writeGSSModel",
                             help="don't create .gss file")
    cmdlineParser.add_option("--no-hist", action="store_false",
                             default=True, dest="writeFactorHistory",
                             help="don't create .pret file with factor return history")
    cmdlineParser.add_option("-p", "--preliminary", action="store_true",
                             default=False, dest="preliminary",
                             help="Preliminary run--ignore DR assets")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--new-rsk-fields", action="store_true",
                             default=False, dest="newRiskFields",
                             help="Include new fields in .rsk files")
    cmdlineParser.add_option("--warn-not-crash", action="store_true",
                             default=False, dest="notCrash",
                             help="Output warning rather than crashing when some fields are missing")
    cmdlineParser.add_option("--chron-dates", action="store_true",
                             default=False, dest="chronDates",
                             help="Write files in chronological order")
    cmdlineParser.add_option("--stat-rethist", action="store_true",
                             default=False, dest="writeStatFactorHistory",
                             help="create daily  .pret file with stat factor return data")
    cmdlineParser.add_option("--histbeta-new", action="store_true",
                             default=False, dest="histBetaNew",
                             help="process historic beta new way or legacy way")
    cmdlineParser.add_option("--force", "-f", action="store_true",
                             default=False, dest="force",
                             help="override model restrictions")
    cmdlineParser.add_option("--factor-suffix", action="store",
                             default="", dest="factorSuffix",
                             help="Factor suffix to use, if at all")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    modelDB.setTotalReturnCache(150)
    modelDB.setMarketCapCache(150)
    modelDB.setVolumeCache(150)
    modelDB.setHistBetaCache(30)
    modelDB.cumReturnCache = None
    print('suffix = |%s|' % options.factorSuffix)
    if len(args) == 1:
        dateRanges = [i.strip() for i in args[0].split(',')]
        dates = set()
        for dRange in dateRanges:
            if dRange.find(':') == -1:
                dates.add(Utilities.parseISODate(dRange))
            else:
                (startDate, endDate) = [i.strip() for i in dRange.split(':')]
                startDate = Utilities.parseISODate(startDate)
                endDate = Utilities.parseISODate(endDate)
                dates.update([startDate + datetime.timedelta(i)
                              for i in range((endDate-startDate).days + 1)])
        dates = sorted(dates,reverse=(options.chronDates==False))
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = sorted([startDate + datetime.timedelta(i)
                      for i in range((endDate-startDate).days + 1)], reverse=(options.chronDates==False))
    
    if options.formatVersion == 2:
        fileFormat_ = FlatFilesV2()
    elif options.formatVersion == 3:
        fileFormat_ = FlatFilesV3()
    elif options.formatVersion == 4:
        fileFormat_ = FlatFilesPhoenixFI()
    elif options.formatVersion == 5:
        fileFormat_ = FlatFilesPhoenixCO()
    elif options.formatVersion == -1:
        fileFormat_ = FlatFilesV3()
    elif options.formatVersion == -2:
        fileFormat_ = FlatFilesGeneric()
        fileFormat_.vanilla = True
        options.writeCurrencies = False
        options.writeFactorReturns = False
        options.writeStatFactorReturns = False
        options.writeFactorHry = False
        options.writeAssetIdm = False
        options.writeAssetAtt = False
        options.writeGSSModel = False
    else:
        logging.fatal('Unsupported format version %d', options.formatVersion)
        return 1
    if not hasattr(riskModel, 'modelHack'):
        riskModel.modelHack = Utilities.Struct()
        riskModel.modelHack.nonCheatHistoricBetas = True
    if options.force:
        riskModel.forceRun = True
        riskModel.variableStyles = True
    
    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi != None and rmi.has_risks:
            logging.info('Processing %s', d)
            fileFormat_.writeDay(options, rmi, riskModel, modelDB, marketDB)
        elif not options.ignoreMissingModels:
            logging.fatal('No risk model instance on %s', d)
            sys.exit(1)
        else:
            if rmi:
                logging.info('Processing even though no risks %s', d)
                fileFormat_.writeDay(options, rmi, riskModel, modelDB, marketDB)
            else:
                logging.info('Skipping %s, no model instance present', d)
                continue


        # write stat factor return history - only for MH fundamental models and US3-SH model
        if options.writeStatFactorHistory and \
           riskModel.mnemonic[-2:]=='-S':
           #issubclass(riskModel.__class__, (MFM.StatisticalModel )) :
           #(riskModel.mnemonic in ('AXUS3-MH-S', 'AXUS3-SH-S')): 
            target=options.targetDir
            if options.appendDateDirs:
               target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
               try:
                  os.makedirs(target)
               except OSError as e:
                  if e.errno != 17:
                     raise
                  else:
                     pass
            mydir = target
            tmpfile=tempfile.mkstemp(prefix='pret',dir=mydir)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            os.chmod(tmpfile[1],0o644)
            tmpfilename=tmpfile[1]
            outFileName = '%s/%s.%s.pret' % (mydir, riskModel.mnemonic, str(d).replace('-',''))
            outFileName = makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(tmpfilename, 'w')
            fileFormat_.writeStatFactorReturnHist(riskModel, d, modelDB, options, outFile)
            outFile.close()
            logging.info("Move Returns history file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
    


        logging.info("Done writing %s", d)
    
    # write factor return history - only for MH fundamental models and US3-SH model
    if options.writeFactorHistory and not riskModel.isStatModel() and \
           ('-MH' in riskModel.mnemonic or riskModel.mnemonic == 'AXUS3-SH' or riskModel.mnemonic == 'AXUS4-SH' or options.fileFormatVersion >= 4.0):
        mydir = options.targetDir
        if re.match('.*/\d{4}/\d{2}/{0,1}', mydir):
            mydir = '/'.join(mydir.rstrip('/').split('/')[:-2])
        tmpfile=tempfile.mkstemp(prefix='pret',dir=mydir)
        #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
        os.close(tmpfile[0])
        os.chmod(tmpfile[1],0o644)
        tmpfilename=tmpfile[1]
        outFileName = '%s/%s.history.pret' % (mydir, riskModel.mnemonic)
        outFileName = makeFinalFileName(outFileName, options.fileFormatVersion)
        outFile = open(tmpfilename, 'w')
        fileFormat_.writeFactorReturnHist(riskModel, dates[-1], modelDB, options, outFile)
        outFile.close()
        logging.info("Move Returns history file %s to %s", tmpfilename, outFileName)
        shutil.move(tmpfilename, outFileName)
    
    modelDB.finalize()
    marketDB.finalize()

if __name__ == '__main__':
    main()

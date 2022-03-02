
import sys
import copy
import numpy.ma as ma
import numpy
import datetime
import logging
import scipy.stats as sm
import logging
import os
import operator
from marketdb import MarketDB
from riskmodels import Matrices
from riskmodels import ModelDB
from riskmodels import LegacyUtilities as Utilities
from riskmodels import StyleExposures
from riskmodels import MFM
from riskmodels import RegressionToolbox
from riskmodels import RiskCalculator
from riskmodels import Standardization
from riskmodels import EstimationUniverse
from riskmodels import Ugliness
from riskmodels import FactorReturns
from riskmodels import AssetProcessor
from riskmodels import ReturnCalculator
from riskmodels import GlobalExposures
from riskmodels.Matrices import ExposureMatrix


def generate_fundamental_exposures_v3(riskModel, descriptors,
                modelDate, data, modelDB, marketDB):
    """Compute multiple-descriptor fundamental style exposures
    for assets in data.universe for all CompositeFactors in riskModel.factors.
    data should be a Struct() containing the ExposureMatrix
    object as well as any required market data, like market
    caps, asset universe, etc.
    Factor exposures are computed as the equal-weighted average
    of all the normalized descriptors associated with the factor.
    """
    logging.debug('generate_fundamental_exposures: begin')
    compositeFactors = [f for f in riskModel.styles 
            if hasattr(riskModel.styleParameters[f.name], 'descriptors')]
    if len(compositeFactors) == 0:
        logging.warning('No CompositeFactors found!')
        return data.exposureMatrix
    
    descriptorExposures = Matrices.ExposureMatrix(data.universe)
    dateResultDict = dict()
#    testFile = open('testing123.csv','w')
#    testDict = dict()
    for modelDateIdx, modelDate in enumerate(modelDates):
        descriptorExposures = Matrices.ExposureMatrix(data.universe)
        for dIdx, d  in enumerate(descriptors):
            if d == 'Book-to-Price' :
                values = StyleExposures.generate_book_to_price(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None, useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Earnings-to-Price':
                values = StyleExposures.generate_earnings_to_price(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None, useQuarterlyData=riskModel.quarterlyFundamentalData,
                            legacy=riskModel.modelHack.legacyETP)
            elif d == 'Est Earnings-to-Price':
                params = Utilities.Struct()
                params.maskNegative = False
                params.winsoriseRaw = True
                params.estOnly = False
                # Compute est EPS
                est_eps = StyleExposures.generate_est_earnings_to_price(
                        modelDate, data, riskModel, modelDB, marketDB, params,
                        restrict=None, useQuarterlyData=False)
                # Compute realised BTP
                btp = StyleExposures.generate_book_to_price(
                        modelDate, data, riskModel, modelDB, marketDB,
                        restrict=None, useQuarterlyData=riskModel.quarterlyFundamentalData)
                # Compute realised EPS
                eps = StyleExposures.generate_earnings_to_price(
                        modelDate, data, riskModel, modelDB, marketDB,
                        restrict=None, useQuarterlyData=riskModel.quarterlyFundamentalData,
                        legacy=riskModel.modelHack.legacyETP, maskNegative=params.maskNegative)
                # Take average of EPS and est. EPS
                values = ma.average(ma.array([est_eps, eps]),axis=0)
            elif d == 'Est Revenue':
                values = StyleExposures.generate_estimate_revenue(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Sales-to-Price':
                values = StyleExposures.generate_sales_to_price(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Debt-to-Assets':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData,
                            useTotalAssets=True)
                values = ma.where(values < 0.0, 0.0, values)   # Negative debt -> 0.0
            elif d == 'Debt-to-MarketCap':
                values = StyleExposures.generate_debt_to_marketcap(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Plowback times ROE':
                roe = StyleExposures.generate_return_on_equity(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
                divPayout = StyleExposures.generate_dividend_payout(
                            modelDate, data, riskModel, modelDB, marketDB,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
                values = (1.0 - divPayout) * roe
            elif d == 'Dividend Payout':
                values = StyleExposures.generate_dividend_payout(
                            modelDate, data, riskModel, modelDB, marketDB,
                            useQuarterlyData=riskModel.quarterlyFundamentalData,
                            maskZero=True)
            elif d == 'Proxied Dividend Payout':
                values = StyleExposures.generate_proxied_dividend_payout(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None, maskZero=True,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Return-on-Equity':
                values = StyleExposures.generate_return_on_equity(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
#                for sIdx, subIssue in enumerate(data.universe):
#                    testDict.setdefault(subIssue,list()).append(values[sIdx])
            elif d == 'Return-on-Assets':
                values = StyleExposures.generate_return_on_assets(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Sales Growth':
                values = StyleExposures.generate_sales_growth(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Est Sales Growth':
                values = StyleExposures.generate_est_sales_growth(
                        modelDate, data, riskModel, modelDB, marketDB,
                        restrict=None, winsoriseRaw=True,
                        useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Earnings Growth':
                values = StyleExposures.generate_earnings_growth(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)    
            elif d == 'Est Earnings Growth':
                values = StyleExposures.generate_est_earnings_growth(
                        modelDate, data, riskModel, modelDB, marketDB,
                        restrict=None, winsoriseRaw=True,
                        useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Robust Sales Growth':
                values = StyleExposures.generate_robust_sales_growth(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Robust Earnings Growth':
                values = StyleExposures.generate_robust_earnings_growth(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=riskModel.quarterlyFundamentalData)
            elif d == 'Est EBITDA':
                values = StyleExposures.generate_estimate_EBITDA(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Enterprise Value':
                values = StyleExposures.generate_estimate_enterprise_value(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Cash-Flow-per-Share':
                values = StyleExposures.generate_estimate_cash_flow_per_share(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Avg Return-on-Equity':
                values = StyleExposures.generate_est_return_on_equity(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Est Avg Return-on-Assets':
                values = StyleExposures.generate_est_return_on_assets(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None,
                            useQuarterlyData=False)
            elif d == 'Dividend Yield':
                values = StyleExposures.generate_dividend_yield(
                            modelDate, data, riskModel, modelDB, marketDB,
                            restrict=None)
            elif d == 'Market Sensitivity Descriptor':
                params = riskModel.styleParameters['Market Sensitivity']
                mm = Utilities.run_market_model_v3(
                        riskModel.rmg[0], data.returns, modelDB, marketDB, params,
                        debugOutput=riskModel.debuggingReporting,
                        clippedReturns=data.clippedReturns)
                values = mm.beta
            elif d == 'Volatility Descriptor':
                params = riskModel.styleParameters['Volatility']
                values = StyleExposures.generate_cross_sectional_volatility_v3(
                        data.returns, params, indices=data.estimationUniverse,
                        clippedReturns=data.clippedReturns)
            elif d == 'Liquidity Descriptor':
                params = riskModel.styleParameters['Liquidity']
                values = StyleExposures.generate_trading_volume_exposures_v3(
                        modelDate, data, riskModel.rmg, modelDB,
                        params, riskModel.numeraire.currency_id)
            elif d == 'Amihud Liquidity Descriptor':
                params = riskModel.styleParameters['Amihud Liquidity']
                iliq = StyleExposures.generateAmihudLiquidityExposures(
                        modelDate, data.returns, data, riskModel.rmg, modelDB,
                        params, riskModel.numeraire.currency_id, scaleByTO=True)
                values = ma.log(1.0 + iliq)
            elif d == 'Share Buyback':
                values = StyleExposures.generate_share_buyback(
                         modelDate, data, riskModel, modelDB, marketDB,
                         restrict = None, maskZero=True)
            elif d == 'Short Interest':
                values = StyleExposures.generate_short_interest(
                         modelDate, data, riskModel, modelDB, marketDB, 
                         restrict = None, maskZero=True)
            else:
                raise Exception('Undefined descriptor %s!' % d)
            subIssuePosDict = dict([(sIdx, subIssue) for sIdx, subIssue in enumerate(data.universe)])
            
            missingValue = numpy.flatnonzero(ma.getmaskarray(values))
            ok_list = [(i,iIdx) for iIdx, i in enumerate(values) if iIdx not in missingValue]
            ok_values, ok_keys = zip(*ok_list)
            
            missingPercent  = (1-float(len(ok_values))/float(len(values)))
            mean =  numpy.mean(ok_values, axis=0)
            stdev = numpy.std(ok_values, axis=0)
            min_value, min_pos = sorted(ok_list, key=operator.itemgetter(0,1))[0]
            max_value, max_pos = sorted(ok_list, key=operator.itemgetter(0,1))[-1]
            min_issue_id = subIssuePosDict[min_pos].getModelID()
            max_issue_id = subIssuePosDict[max_pos].getModelID()
            dateResultDict.setdefault(modelDate, list()).append((d, missingPercent, mean, stdev,
                                                                 min_value, max_value, min_issue_id, max_issue_id))
    # testFile.write('Return-on-Equity\n')
    # for sub_issue in testDict.keys():
    #     value = testDict[sub_issue]
    #     testFile.write('%s,'%sub_issue)
    #     testFile.write(','.join(['%s'%i for i in value]))
    #     testFile.write('\n')    
                
    return dateResultDict

def printResult( resultDict, ofile, ffile):
        ffile.write('\n\n')
        ofile.write('\n\nFACTOR LIBRARY QA reports:\n________________\n\n')
        ofile.write('%s\n'%('-'*200))
        mean_diff_threshold = 100
        missing_diff_threshold = 10
        rvalDict = dict()
        for dIdx, d in sorted(enumerate(resultDict.keys())):
            highlightFlag = False
            rval = Utilities.Struct()
            descriptor = d.replace(' ','_')
            rval.reportName = '%-20s_(TO_QA = Y if MEAN_DIFF_PERCENT_YtoY > %s or MISSING_PERCENT_DIFF >= %s)'%(descriptor,  mean_diff_threshold, missing_diff_threshold)
            rval.header = '%-6s | %-20s | %-10s | %-20s | %-20s | %-10s | %-50s \n' %\
                            ('DATE','MISSING_PERCENT','MEAN','STANDARD_DEVIATION',
                             'MEAN_DIFF_PERCENT_YtoY', 'TO QA?', 'MIN_VALUE:MAX_VALUE')
            valueList = list()
            for pos, (date, missingPercent, mean, stdev, min_value, max_value, min_issue_id, max_issue_id)  in  enumerate(resultDict[d]):
                qtr_mean_diff = -1
                missingPercent_diff = 0.01
                TO_QA = 'N'
                if pos > 3:
                    qtr_mean_diff = abs(round(mean/resultDict[d][pos-4][2],4)-1)*100
                    missingPercent_diff = round(missingPercent-resultDict[d][pos-4][1],4)*100
                if date.month > 0 and date.month < 4:
                    qtr = 'Q1'
                elif date.month > 3 and date.month < 7:
                    qtr = 'Q2'
                elif date.month > 6 and date.month < 10:
                    qtr = 'Q3'
                else:
                    qtr = 'Q4'

                if (qtr_mean_diff > mean_diff_threshold  and qtr_mean_diff != -1) or (missingPercent_diff >= missing_diff_threshold and missingPercent_diff != 0.01):
                    TO_QA = 'Y'
                    highlightFlag = True
                valueList.append(('%-4s_%-2s | %-20s%% | %-10s | %-20s | %-20s%%| %-10s | %-10s(%-10s):   %-10s(%-10s) '%(date.isoformat()[0:4], qtr, round(missingPercent,4), round(mean,4), round(stdev,4), qtr_mean_diff, TO_QA, round(min_value,4), min_issue_id.getIDString(), round(max_value,4), max_issue_id.getIDString())))
            rval.valueList = valueList    
            if highlightFlag:
                rval.reportName += ' (QA required) \n'
            else:
                rval.reportName += ' (QA not necessary) \n'
            rvalDict[d] = rval
        for descriptor in sorted(rvalDict.keys()):
            rval = rvalDict[descriptor]
            ofile.write(rval.reportName)
            ffile.write(rval.reportName)
            ofile.write(rval.header)
            ffile.write(rval.header)
            for i in rval.valueList :
               ofile.write('%s\n'%i)
               ffile.write('%s\n'%i)
               
            ofile.write('\n\n')
            ffile.write('\n\n')

#####################################
# Run an example when called as main.
if __name__ == '__main__':
    import optparse
    import pickle

    usage = "usage: %prog [options] date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--report-file", action="store",
                             default='reconReport.txt', dest="reportFile",
                             help="report file name")
    
    (options, args) = cmdlineParser.parse_args()
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, 
                                 user=options.marketDBUser, passwd=options.marketDBPasswd)
    filename = options.reportFile
    i = 0
    if os.path.isfile(filename + ".v" + str(i)):
        logging.info(filename  + ".v" + str(i) + " already exists")
        i = 1
        while os.path.isfile(filename + ".v" + str(i)):
            logging.info(filename + ".v" + str(i) + " already exists")
            i = i + 1
    filename = filename + ".v" + str(i)
    dirName=os.path.dirname(filename)
    if  dirName and not os.path.isdir(dirName):
        try:
            os.makedirs(dirName)
        except:
            excstr=str(sys.exc_info()[1])
            if excstr.find('File exists') >= 0 and excstr.find(dirName) >= 0:
                logging.info('Error can be ignored - %s' % excstr)
            else:
                raise

    ofile = open(filename, "w")
    ffile = open(filename+'.fmt', "w")
    riskModel = riskModelClass(modelDB, marketDB)
    modelDate = Utilities.parseISODate(args[0])
    allModeldates = sorted(modelDB.getDates(riskModel.rmg, modelDate, 365, excludeWeekend=True))
    divisor = -1

    modelDates = list()
    while -(divisor)< len(allModeldates): 
        modelDates.append(allModeldates[divisor])
        divisor += -60
    modelDates = sorted(modelDates)
    consolResultDict = dict()

    rmi = modelDB.getRiskModelInstance(riskModel.rms_id, modelDate)
    riskModel.setFactorsForDate(modelDate, modelDB)
    modelMarketMap = dict(modelDB.getIssueMapPairs(modelDate) )

    compositeFactors = [f for f in riskModel.styles
                        if hasattr(riskModel.styleParameters[f.name], 'descriptors')]
    if len(compositeFactors) == 0:
        logging.warning('No CompositeFactors found!')
    descriptors = []
    proxyDescSet = set(['Book-to-Price Nu'])
    for f in compositeFactors:
        descriptors.extend(riskModel.styleParameters[f.name].descriptors)
    descriptors = list(set(descriptors)-proxyDescSet)
    data = Utilities.Struct()
    data.universe = modelDB.getRiskModelInstanceUniverse(rmi)

    # index = modelDB.getIndexConstituents('RUSSELL TOP50', modelDate, marketDB)
    # universeList = []
    # for (s, w) in index:
    #     sub_issue = '%s'%s.getIDString()+'11'
    #     universeList.append(ModelDB.SubIssue(sub_issue))
    # data.universe = universeList
    # data.universe = [ModelDB.SubIssue('D3ATP4SFN811')]
    data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
    try:
        (mcapDates, goodRatio) = riskModel.getRMDates(
                modelDate, modelDB, 20, ceiling=False)
    except:
        try:
            (mcapDates, goodRatio) = riskModel.getRMDatesLegacy(
                    modelDate, modelDB, 20, ceiling=False)
        except:
            logging.error('No getRMDates or getRMDatesLegacy method in risk model class')
    data.marketCaps = modelDB.getAverageMarketCaps(
                mcapDates, data.universe, riskModel.numeraire.currency_id, marketDB)
    data.marketCaps = numpy.array(data.marketCaps)

    if hasattr(riskModel, 'modelHack') and riskModel.modelHack.issuerMarketCaps:
        data.issuerMarketCaps = GlobalExposures.computeTotalIssuerMarketCaps(
            data, modelDate, riskModel.numeraire, modelDB, marketDB)
        data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()
    else:
        data.issuerMarketCaps = data.marketCaps.copy()
        data.issuerTotalMarketCaps = data.issuerMarketCaps.copy()
    
    estu = riskModel.loadEstimationUniverse(rmi, modelDB, data)
    assert(len(estu) > 0)
    data.exposureMatrix = Matrices.ExposureMatrix(data.universe)
    data.estimationUniverse =  [data.assetIdxMap[n] for n in estu if n in data.assetIdxMap]
    dtResultDict = generate_fundamental_exposures_v3(riskModel, descriptors, modelDates, data, modelDB, marketDB)
    for dIdx, d in enumerate(descriptors):
        for dtIdx, date in enumerate(sorted(dtResultDict.keys())):
            d, missingPercent , mean, stdev, min_value, max_value, min_sub_issue_id, max_sub_issue_id = dtResultDict[date][dIdx]
            consolResultDict.setdefault(d,[]).append((date, missingPercent, mean, stdev, min_value, max_value, min_sub_issue_id, max_sub_issue_id))

    printResult(consolResultDict, ofile, ffile)
    modelDB.revertChanges()
    marketDB.finalize()
    modelDB.finalize()

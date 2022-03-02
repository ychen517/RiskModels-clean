

import datetime
import numpy
import logging
import numpy.ma as ma
import numpy.linalg as linalg
from marketdb import MarketDB
from riskmodels import Classification
from riskmodels import CurrencyRisk
from riskmodels import EstimationUniverse
from riskmodels import GlobalExposures
from riskmodels import MarketIndex
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels import MFM
from riskmodels.Factors import ModelFactor, CompositeFactor
from riskmodels import RegressionToolbox
from riskmodels import ReturnCalculator
from riskmodels import RiskCalculator
from riskmodels import Standardization
from riskmodels import StyleExposures
from riskmodels import TOPIX
from riskmodels import LegacyUtilities as Utilities
from riskmodels.RiskModels_V3 import defaultFundamentalCovarianceParameters, defaultStatisticalCovarianceParameters, \
     defaultRegressionParameters
from riskmodels.RiskModels_V3 import WWAxioma2011MH, FXAxioma2010USD


class FixedIncomeSingleCountryModel(MFM.StatisticalFactorModel):
    
    def generateStatCovariances(self, modelDate, modelDB, marketDB):
        """Cribbed from StatisticalFactorModel.generateStatisticalModel
        """
        # Get risk model universe
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        data = Utilities.Struct()
        data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
        (mcapDates, goodRatio) = self.getRMDates(
                modelDate, modelDB, 20, ceiling=False)
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        data.subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        # Load historical asset returns
        if len(self.rmg) > 1:
            needDays = int(self.returnHistory / goodRatio)
            baseCurrencyID = self.numeraire.currency_id
        else:
            needDays = self.returnHistory
            baseCurrencyID = None

        if self.modelHack.specialDRTreatment:
            self.log.info('Using NEW ISC Treatment for returns processing')
            self.rmgAssetMap = self.loadRiskModelGroupAssetMap(
                    modelDate, data.universe, modelDB, marketDB, False)
            # SHS Should drCurrData be baseCurrencyID here instead of None?
            (returns, zeroMask, ipoMask) = self.process_returns_history(
                    modelDate, data.universe, needDays-1, modelDB, marketDB,
                    drCurrData=None, subIssueGroups=data.subIssueGroups)
            del self.rmgAssetMap
        else:
            returns = modelDB.loadTotalReturnsHistory(self.rmg, modelDate, 
                        data.universe, needDays-1, assetConvMap=baseCurrencyID)
        # Create TimeSeriesMatrix objects for factor and specific returns
        self.log.debug('building time-series matrices: begin')
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        dateList = returns.dates
        newDateList = Utilities.change_date_frequency(dateList, 'rolling')
        factorReturns = modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, dateList)
        # compute weekly returns in factorReturns matrix
        rolling_returns_data, newDateList = Utilities.compute_compound_returns_v3(factorReturns.data, dateList, newDateList)
        
        # reverse lists
        dateList.reverse()
        newDateList.reverse()
        #rolling_returns_data = numpy.fliplr(rolling_returns_data)
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        omegaObs = min(len(dateList), maxOmegaObs)
        deltaObs = min(len(dateList), maxDeltaObs)
        self.log.info('Using %d of %d days of factor return history',
                      omegaObs, len(dateList))
        self.log.info('Using %d of %d days of specific return history',
                      deltaObs, len(dateList))
        
        data.frMatrix = Matrices.TimeSeriesMatrix(
                                subFactors, newDateList[:omegaObs])
        data.frMatrix.data = ma.array(numpy.fliplr(rolling_returns_data))[:,:omegaObs]
        # Compute factor covariances
        data.factorCov = self.covarianceCalculator.\
                computeFactorCovarianceMatrix(data.frMatrix)
        nonTriangular = False
        for f1 in range(len(subFactors)):
            if nonTriangular:
                break
            for f2 in range(f1 + 1):
                if abs(data.factorCov[f1,f2] - data.factorCov[f2,f1]) >= 1e-12:
                    print(f1, f2, data.factorCov[f1, f2], data.factorCov[f2, f1], data.factorCov[f1,f2] - data.factorCov[f2,f1])
                    nonTriangular = True
                    break
        if nonTriangular:
            self.log.warning('Covariance matrix is non-triangular; attempting to normalize')
            data.factorCov = (data.factorCov + numpy.transpose(data.factorCov))/2
        self.log.debug('computed factor covariances')
        return data
        

class FixedIncomeRegionalModel(MFM.RegionalStatisticalFactorModel):
    def generateStatCovariances(self, modelDate, modelDB, marketDB):
        """Cribbed from StatisticalFactorModel.generateStatisticalModel
        """
        # Get risk model universe
        rmi = modelDB.getRiskModelInstance(self.rms_id, modelDate)
        data = Utilities.Struct()
        data.universe = modelDB.getRiskModelInstanceUniverse(rmi)
        (mcapDates, goodRatio) = self.getRMDates(
                modelDate, modelDB, 20, ceiling=False)
        data.assetIdxMap = dict([(j,i) for (i,j) in enumerate(data.universe)])
        data.subIssueGroups = modelDB.getIssueCompanyGroups(
                modelDate, data.universe, marketDB)
        # Load historical asset returns
        if len(self.rmg) > 1:
            needDays = int(self.returnHistory / goodRatio)
            baseCurrencyID = self.numeraire.currency_id
        else:
            needDays = self.returnHistory
            baseCurrencyID = None

        if self.modelHack.specialDRTreatment:
            self.log.info('Using NEW ISC Treatment for returns processing')
            self.rmgAssetMap = self.loadRiskModelGroupAssetMap(
                    modelDate, data.universe, modelDB, marketDB, False)
            # SHS Should drCurrData be baseCurrencyID here instead of None?
            (returns, zeroMask, ipoMask) = self.process_returns_history(
                    modelDate, data.universe, needDays-1, modelDB, marketDB,
                    drCurrData=None, subIssueGroups=data.subIssueGroups)
            del self.rmgAssetMap
        else:
            returns = modelDB.loadTotalReturnsHistory(self.rmg, modelDate, 
                        data.universe, needDays-1, assetConvMap=baseCurrencyID)
        # Create TimeSeriesMatrix objects for factor and specific returns
        self.log.debug('building time-series matrices: begin')
        if isinstance(self.covarianceCalculator,
                RiskCalculator.CompositeCovarianceMatrix2009):
            (minVarObs, maxVarObs) = self.vp.getCovarianceSampleSize()
            (minCorrObs, maxCorrObs) = self.cp.getCovarianceSampleSize()
            (minOmegaObs, maxOmegaObs) = (max(minVarObs, minCorrObs), max(maxVarObs, maxCorrObs))
            (minDeltaObs, maxDeltaObs) = self.sp.getCovarianceSampleSize()
        else:
            (minOmegaObs, maxOmegaObs) = self.rp.getCovarianceSampleSize()
            (minDeltaObs, maxDeltaObs) = self.rp.getSpecificRiskSampleSize()
        subFactors = modelDB.getSubFactorsForDate(modelDate, self.factors)
        dateList = returns.dates
        newDateList = Utilities.change_date_frequency(dateList, 'rolling')
        factorReturns = modelDB.loadFactorReturnsHistory(self.rms_id, subFactors, dateList)
        # compute weekly returns in factorReturns matrix
        rolling_returns_data, newDateList = Utilities.compute_compound_returns_v3(factorReturns.data, dateList, newDateList)
        
        # reverse lists
        dateList.reverse()
        newDateList.reverse()
        #rolling_returns_data = numpy.fliplr(rolling_returns_data)
        if len(dateList) < max(minOmegaObs, minDeltaObs):
            required = max(minOmegaObs, minDeltaObs)
            self.log.warning('%d missing risk model instances for required days',
                          required - len(dateList))
            raise LookupError(
                '%d missing risk model instances for required days'
                % (required - len(dateList)))
        omegaObs = min(len(dateList), maxOmegaObs)
        deltaObs = min(len(dateList), maxDeltaObs)
        self.log.info('Using %d of %d days of factor return history',
                      omegaObs, len(dateList))
        self.log.info('Using %d of %d days of specific return history',
                      deltaObs, len(dateList))
        
        data.frMatrix = Matrices.TimeSeriesMatrix(
                                subFactors, newDateList[:omegaObs])
        data.frMatrix.data = ma.array(numpy.fliplr(rolling_returns_data))[:,:omegaObs]
        # Compute factor covariances
        data.factorCov = self.covarianceCalculator.\
                computeFactorCovarianceMatrix(data.frMatrix)
        nonTriangular = False
        for f1 in range(len(subFactors)):
            if nonTriangular:
                break
            for f2 in range(f1 + 1):
                if abs(data.factorCov[f1,f2] - data.factorCov[f2,f1]) >= 1e-12:
                    print(f1, f2, data.factorCov[f1, f2], data.factorCov[f2, f1], data.factorCov[f1,f2] - data.factorCov[f2,f1])
                    nonTriangular = True
                    break
        if nonTriangular:
            self.log.warning('Covariance matrix is non-triangular; attempting to normalize')
            data.factorCov = (data.factorCov + numpy.transpose(data.factorCov))/2
        self.log.debug('computed factor covariances')
        return data

class CommodityFutureRegionalModel(MFM.RegionalFundamentalModel):
    useRobustRegression = False

class USFIAxioma_R(FixedIncomeSingleCountryModel):
    """United States Single Currency Fixed-income Factor Risk Model - Research
    """
    rm_id = -1
    revision = 1
    rms_id = 153
    industryClassification = Classification.GICSIndustries(
        datetime.date(2008,8,30))
    blind = [ModelFactor("AB3D","AB3D"),
             ModelFactor("COF11DINS","COF11DINS"),
             ModelFactor("CPIU","CPIU"),
             ModelFactor("DEM6MLIB","DEM6MLIB"),
             ModelFactor("EU03MLIB","EU03MLIB"),
             ModelFactor("EU06MLIB","EU06MLIB"),
             ModelFactor("EURSWE10Y","EURSWE10Y"),
             ModelFactor("EURSWE2Y","EURSWE2Y"),
             ModelFactor("EURSWE30Y","EURSWE30Y"),
             ModelFactor("FBL3INVQW","FBL3INVQW"),
             ModelFactor("FBL6INVQW","FBL6INVQW"),
             ModelFactor("FCN10YY","FCN10YY"),
             ModelFactor("FCN10YYW","FCN10YYW"),
             ModelFactor("FCN1YY","FCN1YY"),
             ModelFactor("FCN2YY","FCN2YY"),
             ModelFactor("FCN2YYW","FCN2YYW"),
             ModelFactor("FCN30YY","FCN30YY"),
             ModelFactor("FCN3YY","FCN3YY"),
             ModelFactor("FCN5YY","FCN5YY"),
             ModelFactor("FCN5YYW","FCN5YYW"),
             ModelFactor("FFO","FFO"),
             ModelFactor("FFQ","FFQ"),
             ModelFactor("FFT","FFT"),
             ModelFactor("GBP3MLIB","GBP3MLIB"),
             ModelFactor("LIUSD2W","LIUSD2W"),
             ModelFactor("NIK225S","NIK225S"),
             ModelFactor("PRQ","PRQ"),
             ModelFactor("RM1","RM1"),
             ModelFactor("RMFEDFUNDNS","RMFEDFUNDNS"),
             ModelFactor("SP500C","SP500C"),
             ModelFactor("US.USD.AFLP(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.AFLP(IG).ABS.5Y"),
             ModelFactor("US.USD.AGNCY.FFCBFC.SPRSWP.5Y","RefType|Name=CurveNode|US.USD.AGNCY.FFCBFC.SPRSWP.5Y"),
             ModelFactor("US.USD.AGNCY.FHLB.SPRSWP.5Y","RefType|Name=CurveNode|US.USD.AGNCY.FHLB.SPRSWP.5Y"),
             ModelFactor("US.USD.AGNCY.FHLMC.SPRSWP.5Y","RefType|Name=CurveNode|US.USD.AGNCY.FHLMC.SPRSWP.5Y"),
             ModelFactor("US.USD.AGNCY.FNMA.SPRSWP.5Y","RefType|Name=CurveNode|US.USD.AGNCY.FNMA.SPRSWP.5Y"),
             ModelFactor("US.USD.AIRL(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.AIRL(IG).ABS.5Y"),
             ModelFactor("US.USD.ALEA(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.ALEA(IG).ABS.5Y"),
             ModelFactor("US.USD.AUTO(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.AUTO(IG).ABS.5Y"),
             ModelFactor("US.USD.BIKE(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.BIKE(IG).ABS.5Y"),
             ModelFactor("US.USD.BOAT(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.BOAT(IG).ABS.5Y"),
             ModelFactor("US.USD.CARD(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.CARD(IG).ABS.5Y"),
             ModelFactor("US.USD.CBO(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.CBO(IG).ABS.5Y"),
             ModelFactor("US.USD.CLO(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.CLO(IG).ABS.5Y"),
             ModelFactor("US.USD.CMBS(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.CMBS(IG).ABS.5Y"),
             ModelFactor("US.USD.CMO(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.CMO(IG).ABS.5Y"),
             ModelFactor("US.USD.COO(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.COO(IG).ABS.5Y"),
             ModelFactor("US.USD.EQIP(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.EQIP(IG).ABS.5Y"),
             ModelFactor("US.USD.EXIM(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.EXIM(IG).ABS.5Y"),
             ModelFactor("US.USD.GVT.ZC.10Y","RefType|Name=CurveNode|US.USD.GVT.ZC.10Y"),
             ModelFactor("US.USD.GVT.ZC.1Y","RefType|Name=CurveNode|US.USD.GVT.ZC.1Y"),
             ModelFactor("US.USD.GVT.ZC.2Y","RefType|Name=CurveNode|US.USD.GVT.ZC.2Y"),
             ModelFactor("US.USD.GVT.ZC.30Y","RefType|Name=CurveNode|US.USD.GVT.ZC.30Y"),
             ModelFactor("US.USD.GVT.ZC.5Y","RefType|Name=CurveNode|US.USD.GVT.ZC.5Y"),
             ModelFactor("US.USD.HLOC(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.HLOC(IG).ABS.5Y"),
             ModelFactor("US.USD.HOME(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.HOME(IG).ABS.5Y"),
             ModelFactor("US.USD.MANU(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.MANU(IG).ABS.5Y"),
             ModelFactor("US.USD.NIM(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.NIM(IG).ABS.5Y"),
             ModelFactor("US.USD.OTHR(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.OTHR(IG).ABS.5Y"),
             ModelFactor("US.USD.RECR(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.RECR(IG).ABS.5Y"),
             ModelFactor("US.USD.RMBS(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.RMBS(IG).ABS.5Y"),
             ModelFactor("US.USD.STUD(IG).ABS.5Y","RefType|Name=CurveNode|US.USD.STUD(IG).ABS.5Y"),
             ModelFactor("US1MLIB","US1MLIB"),
             ModelFactor("US1YLIB","US1YLIB"),
             ModelFactor("US2MLIB","US2MLIB"),
             ModelFactor("US3MLIB","US3MLIB"),
             ModelFactor("US3MTREF","US3MTREF"),
             ModelFactor("US4MLIB","US4MLIB"),
             ModelFactor("US5MLIB","US5MLIB"),
             ModelFactor("US6MLIB","US6MLIB"),
             ModelFactor("US6MLIBID","US6MLIBID"),
             ModelFactor("USD.(A).SPRSWP.5Y","RefType|Name=CurveNode|USD.(A).SPRSWP.5Y"),
             ModelFactor("USD.(AA).SPRSWP.5Y","RefType|Name=CurveNode|USD.(AA).SPRSWP.5Y"),
             ModelFactor("USD.(AAA).SPRSWP.5Y","RefType|Name=CurveNode|USD.(AAA).SPRSWP.5Y"),
             ModelFactor("USD.(BBB).SPRSWP.5Y","RefType|Name=CurveNode|USD.(BBB).SPRSWP.5Y"),
             ModelFactor("USD.(SUB-IG).SPRSWP.5Y","RefType|Name=CurveNode|USD.(SUB-IG).SPRSWP.5Y"),
             ModelFactor("USD.RATGICsSEC(A,Consumer Discretionary).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,CD).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Consumer Staples).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,CS).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Energy).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,Energy).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Financials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,Financials).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Health Care).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,Health Care).5Y"),
             ModelFactor("USD.RATGICsSEC(A,IT).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,IT).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Industrials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,Industrials).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Materials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,Materials).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Telecomm Service).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,TS).5Y"),
             ModelFactor("USD.RATGICsSEC(A,Utilities).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(A,Utilities).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Consumer Discretionary).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,CD).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Consumer Staples).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,CS).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Energy).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,Energy).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Financials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,Financials).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Health Care).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,Health Care).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,IT).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,IT).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Industrials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,Industrials).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Materials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,Materials).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Telecomm Service).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,TS).5Y"),
             ModelFactor("USD.RATGICsSEC(AA,Utilities).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AA,Utilities).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Consumer Discretionary).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,CD).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Consumer Staples).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,CS).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Energy).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,Energy).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Financials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,Financials).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Health Care).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,Health Care).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,IT).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,IT).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Industrials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,Industrials).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Materials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,Materials).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Telecomm Service).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,TS).5Y"),
             ModelFactor("USD.RATGICsSEC(AAA,Utilities).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(AAA,Utilities).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Consumer Discretionary).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,CD).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Consumer Staples).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,CS).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Energy).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,Energy).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Financials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,Financials).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Health Care).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,Health Care).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,IT).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,IT).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Industrials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,Industrials).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Materials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,Materials).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Telecomm Service).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,TS).5Y"),
             ModelFactor("USD.RATGICsSEC(BBB,Utilities).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(BBB,Utilities).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Consumer Discretionary).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,CD).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Consumer Staples).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,CS).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Energy).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,Energy).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Financials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,Financials).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Health Care).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,Health Care).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,IT).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,IT).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Industrials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,Industrials).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Materials).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,Materials).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Telecomm Service).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,TS).5Y"),
             ModelFactor("USD.RATGICsSEC(SUB-IG,Utilities).5Y","RefType|Name=CurveNode|USD.RATGICsSEC(SUB-IG,Utilities).5Y"),
             ModelFactor("USD.SUPN.(AAA).5Y","RefType|Name=CurveNode|USD.SUPN.(AAA).5Y"),
             ModelFactor("USD.SUPN.(IG).5Y","RefType|Name=CurveNode|USD.SUPN.(IG).5Y"),
             ModelFactor("USD01W","USD01W"),
             ModelFactor("USDSF10Y","USDSF10Y"),
             ModelFactor("USDSF1Y","USDSF1Y"),
             ModelFactor("USDSF2Y","USDSF2Y"),
             ModelFactor("USDSF30Y","USDSF30Y"),
             ModelFactor("USDSF3Y","USDSF3Y"),
             ModelFactor("USDSF4Y","USDSF4Y"),
             ModelFactor("USDSF5Y","USDSF5Y"),
             ModelFactor("USDSF6Y","USDSF6Y"),
             ModelFactor("USDSF7Y","USDSF7Y"),
             ModelFactor("USDSF8Y","USDSF8Y"),
             ModelFactor("USDSF9Y","USDSF9Y"),
             ]

    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('PhoenixModels.USFIAxioma')
        FixedIncomeSingleCountryModel.__init__(
                            self, ['CUSIP'], modelDB, marketDB)

        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        self.numFactors = len(self.blind)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, nwLag=0, dva=None)

    def setFactorsForDate(self, date, modelDB=None):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)

        # Create country and currency factors
        self.currencies = [ModelFactor(f, None)
                           for f in set([r.currency_code for r in self.rmg])]

        # Add additional currency factors (to allow numeraire changes)
        # if necessary
        self.currencies.extend([f for f in self.additionalCurrencyFactors
                                if f not in self.currencies
                                and f.isLive(date)])
        # Assign factor IDs and names
        regional = self.currencies
        for f in regional:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        allFactors = self.styles + self.industries + regional
        if self.intercept is not None:
            allFactors = [self.intercept] + allFactors
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factors = allFactors
        self.validateFactorStructure(date)

class UNIAxioma2013MH(FixedIncomeRegionalModel):
    rm_id = 2000
    revision = 1
    rms_id = 20000
    industryClassification = Classification.GICSIndustries(
        datetime.date(2008,8,30))

    blind = []


    def __init__(self, modelDB, marketDB, factorNames=[]):
        self.log = logging.getLogger('PhoenixModels.UNIAxioma')
        self.initBlind(modelDB, factorNames)
        FixedIncomeRegionalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.modelHack.useDRCLoningAndCointegration(False)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.createCurrencyCache(marketDB)
        self.indexSelector = MarketIndex.MarketIndexSelector(modelDB, marketDB)
        self.numFactors = len(self.blind)
        
    def initBlind(self, modelDB, factorNames):

        resolvedFactors = []
        query = """SELECT f.name, f.description
        FROM rms_factor rf JOIN factor f ON f.factor_id=rf.factor_id
        WHERE rf.rms_id=:rms_id AND f.name NOT IN (SELECT currency_code FROM currency_instrument_map)
        ORDER BY f.name
        """
        modelDB.dbCursor.execute(query, rms_id=self.rms_id)
        for r in modelDB.dbCursor.fetchall():
            if r[0] in factorNames:
                resolvedFactors.append(r[0])
            self.blind.append(ModelFactor(r[0], r[1]))

        remaining = list(set([x for x in factorNames if x not in resolvedFactors]))
        factorIDs = modelDB.getFactors(remaining, upper=True)
        toInsert = [x for x in factorIDs if x is not None]
        print(toInsert)
        modelDB.insertRMSFactorsForDates(self.rms_id, toInsert, '1980-01-01', '2999-12-31')

    def setFactorsForDate(self, date, modelDB=None):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)

        # Create country and currency factors
        self.currencies = [ModelFactor(f, None)
                           for f in set([r.currency_code for r in self.rmg])]

        # Add additional currency factors (to allow numeraire changes)
        # if necessary
        self.currencies.extend([f for f in self.additionalCurrencyFactors
                                if f not in self.currencies
                                and f.isLive(date)])
        # Assign factor IDs and names
        regional = self.currencies
#        self.nameFactorMap['USD'] = ModelFactor('USD', 'U.S. Dollar')
#        for f in regional:
#            dbFactor = self.nameFactorMap[f.name]
#            f.description = dbFactor.description
#            f.factorID = dbFactor.factorID
#            f.from_dt = dbFactor.from_dt
#            f.thru_dt = dbFactor.thru_dt
#
#        allFactors = self.blind + self.industries + regional
#        if self.intercept is not None:
#            allFactors = [self.intercept] + allFactors
#        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
#        self.factors = allFactors
        self.factors = self.blind
        self.validateFactorStructure(date, warnOnly = True)

    def setCalculators(self, modelDB, overrider = False):
        # Set up risk parameters
        defaultStatisticalCovarianceParameters(self, nwLag=0, dva=None)

class COAxioma2013MH(CommodityFutureRegionalModel):
    rm_id = 1000
    revision = 2
    rms_id = 10001
    styles = [ModelFactor("Aluminium Shift", "Commodity Futures, Aluminium Shift"),
              ModelFactor("Aluminium Tilt", "Commodity Futures, Aluminium Tilt"),
              ModelFactor("Canola Shift", "Commodity Futures, Canola Shift"),
              ModelFactor("Canola Tilt", "Commodity Futures, Canola Tilt"),
              ModelFactor("Cattle Feeder Shift", "Commodity Futures, Cattle Feeder Shift"),
              ModelFactor("Cattle Feeder Tilt", "Commodity Futures, Cattle Feeder Tilt"),
              ModelFactor("Cattle Live Shift", "Commodity Futures, Cattle Live Shift"),
              ModelFactor("Cattle Live Tilt", "Commodity Futures, Cattle Live Tilt"),
              ModelFactor("Cocoa Shift", "Commodity Futures, Cocoa Shift"),
              ModelFactor("Cocoa Tilt", "Commodity Futures, Cocoa Tilt"),
              ModelFactor("Coffee Shift", "Commodity Futures, Coffee Shift"),
              ModelFactor("Coffee Tilt", "Commodity Futures, Coffee Tilt"),
              ModelFactor("Copper Basis Shift", "Commodity Futures, Copper Basis Shift"),
              ModelFactor("Copper Shift", "Commodity Futures, Copper Shift"),
              ModelFactor("Corn Shift", "Commodity Futures, Corn Shift"),
              ModelFactor("Corn Tilt", "Commodity Futures, Corn Tilt"),
              ModelFactor("Cotton Shift", "Commodity Futures, Cotton Shift"),
              ModelFactor("Cotton Tilt", "Commodity Futures, Cotton Tilt"),
              ModelFactor("Ecxeau Shift", "Commodity Futures, Ecxeau Shift"),
              ModelFactor("Gas Curve", "Commodity Futures, Gas Curve"),
              ModelFactor("Gas Shift", "Commodity Futures, Gas Shift"),
              ModelFactor("Gas Tilt", "Commodity Futures, Gas Tilt"),
              ModelFactor("Gasoil Basis Shift", "Commodity Futures, Gasoil Basis Shift"),
              ModelFactor("Gasoil Basis Tilt", "Commodity Futures, Gasoil Basis Tilt"),
              ModelFactor("Gold Shift", "Commodity Futures, Gold Shift"),
              ModelFactor("Gold Tilt", "Commodity Futures, Gold Tilt"),
              ModelFactor("Heating Oil Basis Shift", "Commodity Futures, Heating Oil Basis Shift"),
              ModelFactor("Heating Oil Basis Tilt", "Commodity Futures, Heating Oil Basis Tilt"),
              ModelFactor("Kansas Wheat Shift", "Commodity Futures, Kansas Wheat Shift"),
              ModelFactor("Kansas Wheat Tilt", "Commodity Futures, Kansas Wheat Tilt"),
              ModelFactor("Lead Shift", "Commodity Futures, Lead Shift"),
              ModelFactor("Lean Hogs Shift", "Commodity Futures, Lean Hogs Shift"),
              ModelFactor("Lean Hogs Tilt", "Commodity Futures, Lean Hogs Tilt"),
              ModelFactor("Nickel Shift", "Commodity Futures, Nickel Shift"),
              ModelFactor("Oats Shift", "Commodity Futures, Oats Shift"),
              ModelFactor("Oats Tilt", "Commodity Futures, Oats Tilt"),
              ModelFactor("Oil Basis Shift", "Commodity Futures, Oil Basis Shift"),
              ModelFactor("Oil Basis Tilt", "Commodity Futures, Oil Basis Tilt"),
              ModelFactor("Oil Curve", "Commodity Futures, Oil Curve"),
              ModelFactor("Oil Shift", "Commodity Futures, Oil Shift"),
              ModelFactor("Oil Tilt", "Commodity Futures, Oil Tilt"),
              ModelFactor("Orange Juice Shift", "Commodity Futures, Orange Juice Shift"),
              ModelFactor("Orange Juice Tilt", "Commodity Futures, Orange Juice Tilt"),
              ModelFactor("Palladium Shift", "Commodity Futures, Palladium Shift"),
              ModelFactor("Platinum Shift", "Commodity Futures, Platinum Shift"),
              ModelFactor("RBOB Basis Shift", "Commodity Futures, RBOB Basis Shift"),
              ModelFactor("RBOB Basis Tilt", "Commodity Futures, RBOB Basis Tilt"),
              ModelFactor("Rice Shift", "Commodity Futures, Rice Shift"),
              ModelFactor("Rice Tilt", "Commodity Futures, Rice Tilt"),
              ModelFactor("Silver Shift", "Commodity Futures, Silver Shift"),
              ModelFactor("Soybean Oil Shift", "Commodity Futures, Soybean Oil Shift"),
              ModelFactor("Soybean Oil Tilt", "Commodity Futures, Soybean Oil Tilt"),
              ModelFactor("Soybeans Shift", "Commodity Futures, Soybeans Shift"),
              ModelFactor("Soybeans Tilt", "Commodity Futures, Soybeans Tilt"),
              ModelFactor("Soymeal Shift", "Commodity Futures, Soymeal Shift"),
              ModelFactor("Soymeal Tilt", "Commodity Futures, Soymeal Tilt"),
              ModelFactor("Sugar Shift", "Commodity Futures, Sugar Shift"),
              ModelFactor("Sugar Tilt", "Commodity Futures, Sugar Tilt"),
              ModelFactor("Wheat Shift", "Commodity Futures, Wheat Shift"),
              ModelFactor("Wheat Tilt", "Commodity Futures, Wheat Tilt"),
              ModelFactor("Zinc Shift", "Commodity Futures, Zinc Shift"),]
    industryClassification = Classification.CommoditiesEmpty(
        datetime.date(2008,8,30))
    newExposureFormat = False
    intercept = None
    countryBetas = False
    specReturnTimingId = 1
    additionalCurrencyFactors = [ModelFactor('EUR', 'Euro')]
    debuggingReporting = True
    
    def __init__(self, modelDB, marketDB):
        self.log = logging.getLogger('PhoenixModels.COAxioma2013MH')
        CommodityFutureRegionalModel.__init__(
                            self, ['SEDOL'], modelDB, marketDB)
        self.setCalculators(modelDB)
        modelDB.setTotalReturnCache(367*2)
        self.currencyModel = FXAxioma2010USD(modelDB, marketDB)
        self.industries = []
        self.countries = []
        self.modelHack.useDRCLoningAndCointegration(False)

    def setFactorsForDate(self, date, modelDB=None):
        """Determine which country/currency factors are in the
        model for the given date.
        """
        # Determine risk model groups (countries) in the model
        self.setRiskModelGroupsForDate(date)

        # Create country and currency factors
        self.currencies = [ModelFactor(f, None)
                           for f in set([r.currency_code for r in self.rmg])]

        for f in self.additionalCurrencyFactors:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        # Add additional currency factors (to allow numeraire changes)
        # if necessary
        self.currencies.extend([f for f in self.additionalCurrencyFactors
                                if f not in self.currencies
                                and f.isLive(date)])
        # Assign factor IDs and names
        regional = self.currencies
        for f in regional:
            dbFactor = self.nameFactorMap[f.name]
            f.description = dbFactor.description
            f.factorID = dbFactor.factorID
            f.from_dt = dbFactor.from_dt
            f.thru_dt = dbFactor.thru_dt

        allFactors = self.styles + self.industries + regional
        if self.intercept is not None:
            allFactors = [self.intercept] + allFactors
        self.factorIDMap = dict([(f.factorID, f) for f in allFactors])
        self.factors = allFactors
        self.validateFactorStructure(date)

    def setCalculators(self, modelDB, overrider=False):
        # special regression parameters
        regressionList = [[
                ExposureMatrix.StyleFactor, ExposureMatrix.CountryFactor
            ]]
        constraintList = [[
                RegressionToolbox.ConstraintSumToZero(ExposureMatrix.CountryFactor),
            ]]
        regressionList = [[ExposureMatrix.StyleFactor]]
        constraintList = [[]]
        regParameters = {
            'regressionOrder': regressionList,
            'factorConstraints': constraintList,
            'weightedRLM': False,
            }
        self.returnCalculator = ReturnCalculator.RobustRegression2(
                RegressionToolbox.RegressionParameters(regParameters))
        defaultFundamentalCovarianceParameters(self, nwLag=0, dva=None)

    def computeExcessReturns(self, date, returns, modelDB, marketDB,
                            drAssetCurrMap=None):
        self.log.debug('computeExcessReturns: begin')
        # Get mapping from asset ID to currency ID
        assetCurrMap = modelDB.getTradingCurrency(returns.dates[-1],
                returns.assets, marketDB, returnType='id')
        if drAssetCurrMap is not None:
            for sid in drAssetCurrMap.keys():
                assetCurrMap[sid] = drAssetCurrMap[sid]
        # Report on assets which are missing a trading currency
        missingTC = [sid for sid in assetCurrMap.keys()\
                if assetCurrMap[sid] == None]
        if len(missingTC) > 0:
            self.log.warning('Missing trading currency for %d assets: %s',
                          len(missingTC), missingTC)
        assetCurrMap = dict([(sid, assetCurrMap.get(sid,
                self.numeraire.currency_id)) \
                    for sid in returns.assets])
        # Create a map of currency indices to ID
        currencies = list(set(assetCurrMap.values()))
        # Get currency ISO mapping - need ISOs for rfrates
        isoCodeMap = marketDB.getCurrencyISOCodeMap()
        currencyISOs = [isoCodeMap[i] for i in currencies]
        # Pull up history of risk-free rates
        rfHistory = modelDB.getRiskFreeRateHistory(
                currencyISOs, returns.dates, marketDB)
        rfHistory.data = rfHistory.data.filled(0.0)
        self.log.debug('computeExcessReturns: end')
        return (returns, rfHistory)

    def loadEstimationUniverse(self, rmi, modelDB):
        """Loads the estimation universe of the given risk model instance.
        Returns a list of sub-issue IDs.
        """
        modelDB.dbCursor.execute("""SELECT sub_issue_id FROM rmi_nestu
            WHERE rms_id = :rms_arg AND dt = :date_arg""",
                              rms_arg=rmi.rms_id, date_arg=rmi.date)
        r = [ModelDB.SubIssue(i[0]) for i in modelDB.dbCursor.fetchall()]
        self.log.info('Loaded %d assets from estimation universe', len(r))
        return r
    
    def computePredictedBeta(self, date, modelData, 
                             modelDB, marketDB, globalBetas=False):
        return []


def computeTotalRisks(date, riskModel, modelDB, marketDB):
    """Cribbed from generateStatisticalModel
    """
    logging.info('Processing total risks and betas for %s', date)
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi is None:
        logging.warning('No risk model instance for %s, skipping', date)
        return
    if not rmi.has_risks:
        logging.warning('Incomplete risk model instance for %s, skipping', date)
        return
    modelData = Utilities.Struct()
    modelData.exposureMatrix = riskModel.loadExposureMatrix(rmi, modelDB)
    modelData.exposureMatrix.fill(0.0)
    (modelData.specificRisk, modelData.specificCovariance) = \
                        riskModel.loadSpecificRisks(rmi, modelDB)
    (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
        rmi, modelDB)
    modelData.factorCovariance = factorCov
    totalRisk = riskModel.computeTotalRisk(modelData, modelDB)
    
    modelDB.deleteRMITotalRisk(rmi)
    modelDB.insertRMITotalRisk(rmi, totalRisk)

if __name__=='__main__':
    import generateFundamentalModel as generateFund
    import optparse

    usage = "usage: %prog [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("--update-database",action="store_false",
                             default=True,dest="testOnly",
                             help="change the database")
    cmdlineParser.add_option("--model", default="FI", dest="model", help="FI or CO model")
    (options_, args_) = cmdlineParser.parse_args()
    logging.config.fileConfig('log.config')
    #Utilities.processDefaultCommandLine(options_, cmdlineParser)

    modelDB=ModelDB.ModelDB(user=options_.modelDBUser,passwd=options_.modelDBPasswd,sid=options_.modelDBSID)
    marketDB=MarketDB.MarketDB(user=options_.marketDBUser,passwd=options_.marketDBPasswd,sid=options_.marketDBSID)
    if options_.model == 'FI':
        usfi = USFIAxioma_R(modelDB,marketDB)
        print(usfi)
        usfi.forceRun = True
        usfi.setRiskModelGroupsForDate(datetime.date(2012,1,5))
        modelDB.setTotalReturnCache(367)
        modelDB.setVolumeCache(190)
        fullData = usfi.generateStatCovariances(datetime.date(2012,1,5),modelDB,marketDB)
        rmi = modelDB.getRiskModelInstance(usfi.rms_id, datetime.date(2012,1,5))
        subFactors = modelDB.getSubFactorsForDate(datetime.date(2012,1,5), usfi.factors)
        modelDB.dbCursor.execute("""DELETE FROM rmi_covariance
            WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                                  date_arg=rmi.date)
        usfi.insertFactorCovariances(rmi, fullData.factorCov, subFactors, modelDB)
        computeTotalRisks(datetime.date(2012,1,5), usfi, modelDB, marketDB)
    elif options_.model == 'UNI':
        unimodel = UNIAxioma2013MH(modelDB, marketDB)
        print(unimodel)
        unimodel.forceRun = True
        unimodel.setRiskModelGroupsForDate(datetime.date(2013,4,10))
        #modelDB.setTotalReturnCache(367)
        #modelDB.setVolumeCache(190)
        unimodel.setFactorsForDate(datetime.date(2013,4,10), modelDB)
        fullData = unimodel.generateStatCovariances(datetime.date(2013,4,10),modelDB,marketDB)
        rmi = modelDB.getRiskModelInstance(unimodel.rms_id, datetime.date(2013,4,10))
        subFactors = modelDB.getSubFactorsForDate(datetime.date(2013,4,10), unimodel.factors)
        modelDB.dbCursor.execute("""DELETE FROM rmi_covariance
            WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=rmi.rms_id,
                                  date_arg=rmi.date)
        unimodel.insertFactorCovariances(rmi, fullData.factorCov, subFactors, modelDB)
        computeTotalRisks(datetime.date(2013,4,10), unimodel, modelDB, marketDB)
    elif options_.model == 'CO':
        wwco = WWCOAxioma_R(modelDB, marketDB)
        startDate = datetime.date(2008,1,2)
        endDate = datetime.date(2013,3,8)
        dtRange = modelDB.getDateRange(wwco.rmg, startDate, endDate, excludeWeekend=True)
        startDate = dtRange[0]
        endDate = dtRange[-1]
        rmi = modelDB.getRiskModelInstance(wwco.rms_id, startDate)
        if not rmi:
            rmi = modelDB.createRiskModelInstance(wwco.rms_id, startDate)
            rmi.setIsFinal(True, modelDB)
        rmi.setHasExposures(True, modelDB)
        for dt in dtRange[1:]:
            print('Processing factor returns for %s' % dt)
            wwco.setRiskModelGroupsForDate(dt)
            wwco.setFactorsForDate(dt, modelDB)
            print(wwco)
#            rmi = modelDB.getRiskModelInstance(wwco.rms_id, dt)
#            if rmi:
#                modelDB.deleteFactorReturns(rmi)
#                modelDB.deleteSpecificReturns(rmi)
#                modelDB.deleteRMSFactorStatistics(rmi)
#                modelDB.deleteRMSStatistics(rmi)
#                modelDB.deleteRMIFactorSpecificRisk(rmi)
#                modelDB.deleteRMIPredictedBeta(rmi)
#                modelDB.deleteRMITotalRisk(rmi)
#                modelDB.dbCursor.execute("""DELETE FROM risk_model_instance
#                WHERE rms_id = :rms_arg AND dt=:date_arg""", rms_arg=wwco.rms_id,
#                                         date_arg=dt)
#            modelDB.createRiskModelInstance(wwco.rms_id, dt)
            rmi = modelDB.getRiskModelInstance(wwco.rms_id, dt)
            assert rmi.has_exposures, "No exposures for %s" % dt
            if dt <= datetime.date(2008,1,2):
                dt += datetime.timedelta(1)
                while dt.weekday() > 4:
                    dt += datetime.timedelta(1)
                continue
            generateFund.options = Utilities.Struct()
            generateFund.options.dontWrite=False
            #generateFund.generateEstimationUniverse(dt, wwco, modelDB, marketDB)
            #generateFund.generateExposures(dt, wwco, modelDB, marketDB)
            #generateFund.generateFactorAndSpecificReturns(dt, wwco, modelDB, marketDB)
            generateFund.generateCumulativeFactorReturns(dt, wwco, modelDB, dt == datetime.date(2008,1,3))
            assert rmi.has_returns, "No factor returns for %s" % dt
            if dt >= datetime.date(2009,1,5):
                generateFund.computeFactorSpecificRisk(dt, wwco, modelDB, marketDB)
                generateFund.computeTotalRisksAndBetas(dt, wwco, modelDB, marketDB)
            rmi = modelDB.getRiskModelInstance(wwco.rms_id, dt)
            print(rmi)
            if not options_.testOnly:
                logging.info("Committing changes for %s", dt)
                modelDB.commitChanges()
                
    if not options_.testOnly:
        logging.info("Committing changes")
        modelDB.commitChanges()
    else:
        logging.info("Reverting changes")
        modelDB.revertChanges()
    marketDB.revertChanges()
    modelDB.finalize()
    marketDB.finalize()
    

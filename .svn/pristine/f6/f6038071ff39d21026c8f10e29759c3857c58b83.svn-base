import calendar
import logging
import numpy as np
import numpy.ma as ma
import os
import pickle
import datetime
import inspect
import pandas
import plotly
import cufflinks
from collections import defaultdict
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import RiskModels
from riskmodels import Utilities

def createPath(pathname):
    if not os.path.exists(pathname):
        os.system('mkdir -p %s' % pathname)
    return pathname

class RMT:
    def __init__(self, rmClass, date, modelDB, marketDB, rmtDir, startDate=None):
        self.rmc = rmClass
        self.endDate = date
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.rmtDir = createPath('%s/%s' % (rmtDir, rmClass.name))

        # Set up dates
        self.startDate = Utilities.parseISODate('1980-01-01')
        if startDate is not None:
            self.startDate = startDate
        self.exposureStartDate = None
        self.returnStartDate = None
        self.riskStartDate = None

    def pre_process(self):

        # Get model dates and RMIs on each
        allCalendarDates = sorted([self.startDate + datetime.timedelta(i) for i in range((self.endDate-self.startDate).days + 1)])
        allRMIs = self.modelDB.getRiskModelInstances(self.rmc.rms_id, dateList=allCalendarDates)
        allRMIMap = dict([(rmi.date, rmi) for rmi in allRMIs])
        allModelDates = sorted(allRMIMap.keys())

        # Find important start dates
        found = 0
        for dt in allModelDates:
            if self.exposureStartDate is None and allRMIMap[dt].has_exposures:
                self.exposureStartDate = dt
                found += 1
            if self.returnStartDate is None and allRMIMap[dt].has_returns:
                self.returnStartDate = dt
                found += 1
            if self.riskStartDate is None and allRMIMap[dt].has_risks:
                self.riskStartDate = dt
                found += 1
            if found >= 3:
                break

        # Get model factors for entire history
        allModelFactors = self.modelDB.getRiskModelSerieFactors(self.rmc.rms_id)
        self.allModelFactors = [f for f in allModelFactors if f.thru_dt > self.exposureStartDate]
        self.fTypeIDMap = self.modelDB.getFactorTypeIDMap().sort_index()

        # Order factors for output
        self.factorNameList = []
        self.factorNameMap = dict()
        for ftype in self.fTypeIDMap.index:
            subset = sorted([f.name for f in self.allModelFactors if f.type_id == ftype])
            self.factorNameList.extend(subset)
            for fct in subset:
                self.factorNameMap[fct] = fct.replace(',','')

        # Save important model data
        self.allModelIssues = self.modelDB.getRiskModelSerieIssues(None, self.rmc.rms_id)
        self.allRMIMap = allRMIMap
        self.modelDates = allModelDates

        return

    def output_model_info(self):
        # Output top-level model data
        outfile = open('%s/_modelInfo.csv' % self.rmtDir, 'w')
        outfile.write('Model, %s,\n' % self.rmc.name)
        outfile.write('Type, %s,\n' % self.rmc.description)
        outfile.write('Exposure start date, %s,\n' % self.exposureStartDate)
        outfile.write('Factor start date, %s,\n' % self.returnStartDate)
        outfile.write('Risk start date, %s,\n' % self.riskStartDate)
        outfile.write('End date, %s,\n' % self.endDate)
        outfile.write('Total issues, %d,\n' % len(set(self.allModelIssues)))

        # Output parameters from model class
        allowedTypes = [str, int, bool, float, datetime.date, dict]
        outfile.write('Model global parameters')
        for vType in inspect.getmembers(self.rmc):
            if type(vType[1]) in allowedTypes and (vType[0][:2] != '__'):
                outfile.write(',%s, %s\n' % (vType[0], vType[1]))

        # Output data on model factors
        outfile.write('Total number of factors, %d,\n' % len(self.allModelFactors))
        for fti in self.fTypeIDMap.index:
            subset = [f for f in self.allModelFactors if f.type_id == fti]
            if len(subset) > 0:
                outfile.write('%s factors, %d,\n' % (self.fTypeIDMap[fti], len(subset)))
            if self.fTypeIDMap[fti] == 'Style':
                styleFactors = subset

        # Output names of styles and descriptors (if relevant)
        if len(styleFactors) > 0:
            outfile.write('Style factors,\n')
            for fac in styleFactors:
                outfile.write(',%s,--------------------------------,\n' % fac.name)
                if hasattr(self.rmc, 'DescriptorMap') and (fac.name in self.rmc.DescriptorMap):
                    if hasattr(self.rmc, 'DescriptorWeights') and (fac.name in self.rmc.DescriptorWeights):
                        for (desc, wt) in zip(self.rmc.DescriptorMap[fac.name], self.rmc.DescriptorWeights[fac.name]):
                            outfile.write(',,%s,Wt %s,\n' % (desc, wt))
                    else:
                        for desc in self.rmc.DescriptorMap[fac.name]:
                            outfile.write(',,%s,\n' % desc)

        # Output estimation universe parameters
        if hasattr(self.rmc, 'estu_parameters'):
            outfile.write('Estimation universe parameters')
            for (key, item) in self.rmc.estu_parameters.items():
                outfile.write(',%s,%s,\n' % (key, item))
        elif hasattr(self.rmc, 'baseModelDateMap'):
            bmDates = list(self.rmc.baseModelDateMap.keys())
            bmDates.sort()
            if hasattr(self.rmc.baseModelDateMap[bmDates[-1]], 'estu_parameters'):
                outfile.write('Estimation universe parameters')
                for (key, item) in self.rmc.baseModelDateMap[bmDates[-1]].estu_parameters.items():
                    outfile.write(',%s,%s,\n' % (key, item))

        # Output regression parameters
        if hasattr(self.rmc, 'returnCalculator') and hasattr(self.rmc.returnCalculator, 'allParameters'):
            outfile.write('Regression parameters,\n')
            if hasattr(self.rmc, 'twinRegressions'):
                outfile.write(',Twin regressions, %s,\n' % self.rmc.twinRegressions)
            for (ip, pars) in self.rmc.returnCalculator.allParameters.items():
                outfile.write(',Reg_%d,Name, %s,\n' % (ip, pars.name))
                outfile.write(',Reg_%d,ESTU, %s,\n' % (ip, pars.estuName))
                outfile.write(',Reg_%d,Kappa, %s,\n' % (ip, pars.kappa))
                outfile.write(',Reg_%d,Weight, %s,\n' % (ip, pars.regWeight))
                ftypes = [ft.name for ft in pars.regressionList]
                outfile.write(',Reg_%d,Factor Types, %s,\n' % (ip, ','.join(ftypes)))
                outfile.write(',Reg_%d,Dummy return, %s,\n' % (ip, pars.dummyType))
                outfile.write(',Reg_%d,Dummy threshold, %s,\n' % (ip, pars.dummyThreshold))
                for con in pars.constraintTypes:
                    outfile.write(',Reg_%d,Factor constraints, %s,\n' % (ip, con.name))
                outfile.write(',Reg_%d,Use real mcap, %s,\n' % (ip, pars.useRealMCapsForConstraints))

        # Output covariance parameters
        if hasattr(self.rmc, 'fvParameters'):
            vp = self.rmc.fvParameters
            outfile.write('Variance parameters,\n')
            outfile.write(',Half life, %d,\n' % vp.halfLife)
            outfile.write(',NW Lag, %d,\n' % vp.NWLag)
            outfile.write(',DVA %s, Window %s, Bounds: (%s %s), DW Ends %s,\n' % \
                    (vp.DVAType, vp.DVAWindow, vp.DVALowerBound, vp.DVAUpperBound, vp.downweightEnds))
            outfile.write(',Demean, %s, halflife, %s, Factors, %s,\n' % \
                    (vp.selectiveDeMean, vp.deMeanHalfLife, vp.deMeanFactorTypes))

        if hasattr(self.rmc, 'fcParameters'):
            cp = self.rmc.fcParameters
            outfile.write('Correlation parameters,\n')
            outfile.write(',Half life, %d,\n' % cp.halfLife)
            outfile.write(',NW Lag, %d,\n' % cp.NWLag)
            outfile.write(',DVA %s, Window %s, Bounds: (%s %s), DW Ends %s,\n' % \
                    (cp.DVAType, cp.DVAWindow, cp.DVALowerBound, cp.DVAUpperBound, cp.downweightEnds))
    
        if hasattr(self.rmc, 'srParameters'):
            sp = self.rmc.srParameters
            outfile.write('SP Risk parameters,\n')
            outfile.write(',Half life, %d,\n' % sp.halfLife)
            outfile.write(',NW Lag, %d,\n' % sp.NWLag)
            outfile.write(',DVA %s, Window %s, Bounds: (%s %s), DW Ends %s,\n' % \
                    (sp.DVAType, sp.DVAWindow, sp.DVALowerBound, sp.DVAUpperBound, sp.downweightEnds))

        outfile.close()

    def writeRegressionStats(self):

        outDir = createPath('%s/RegressStats-External' % self.rmtDir)
        if hasattr(self.rmc, 'twoRegressionStructure'):
            hasInternal = self.rmc.twoRegressionStructure
            intDir = createPath('%s/RegressStats-Internal' % self.rmtDir)
        else:
            hasInternal = False

        # Initialise
        dateList = sorted([d for d in self.modelDates if d>= self.returnStartDate])
        dateIndex = pandas.to_datetime(dateList)
        estuMap = self.rmc.masterEstuMap
        r2Cols = ['adjusted R-squared', 'rWtMin', 'rWtMean', 'rWtNum']
        estuCols = ['length', 'leavers', 'joiners', 'correl', 'mcap', 'adv']
        # External stats
        xtS = Utilities.Struct()
        xtS.tstats = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)
        xtS.pvalues = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)
        xtS.adjr2 = pandas.DataFrame(np.nan, index=dateIndex, columns=r2Cols)
        if hasInternal:
            # Internal stats
            inS = Utilities.Struct()
            inS.tstats = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)
            inS.pvalues = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)
            inS.adjr2 = pandas.DataFrame(np.nan, index=dateIndex, columns=r2Cols)
        # Estu stats
        estuDict = {'univ': pandas.DataFrame(np.nan, index=dateIndex, columns=estuCols)}
        prevEstu = {'univ': None}
        for ky in self.rmc.masterEstuMap.keys():
            estuDict[ky] = pandas.DataFrame(np.nan, index=dateIndex, columns=estuCols)
            prevEstu[ky] = None

        # Get some data date by date
        for dt in dateList:
            self.rmc.setFactorsForDate(dt, self.modelDB)
            rmi = self.allRMIMap[dt]

            # Load in ESTU and mktcap
            estu = set(self.rmc.loadEstimationUniverse(rmi, self.modelDB))
            univ = set(self.modelDB.getRiskModelInstanceUniverse(rmi))
            mktcap = self.modelDB.getAverageMarketCaps([dt], univ, self.rmc.numeraire.currency_id, returnDF=True)
            adv = self.modelDB.getAverageTradingVolume([dt], univ, self.rmc.numeraire.currency_id, returnDF=True)

            # Process the ESTUs
            for estuKey in estuDict.keys():
                if estuKey == 'univ':
                    estu = set(univ)
                elif estuKey in self.rmc.estuMap:
                    estu = set(self.rmc.estuMap[estuKey].assets)
                else:
                    continue
                estuDict[estuKey].loc[dt, 'length'] = len(estu)
                estuDict[estuKey].loc[dt, 'mcap'] = mktcap[estu].sum(axis=None)
                estuDict[estuKey].loc[dt, 'adv'] = adv.sum(axis=None)

                if prevEstu[estuKey] is not None:
                    estuDict[estuKey].loc[dt, 'leavers'] = len(prevEstu[estuKey].difference(estu))
                    estuDict[estuKey].loc[dt, 'joiners'] = len(estu.difference(prevEstu[estuKey]))
                    allAssets = estu.union(prevEstu[estuKey])
                    estuDict[estuKey].loc[dt, 'correl'] = mktcap[allAssets].fillna(0.0).corr(pMktcap[allAssets].fillna(0.0))

                prevEstu[estuKey] = set(estu)
            pMktcap = mktcap.copy(deep=True)

            # Load in regression data
            regStats = self.rmc.loadRegressionStatistics(dt, self.modelDB)
            curFacNames = [fct.name for fct in regStats[1]]
            xtS.tstats.loc[dt, curFacNames] = regStats[0][:,1]
            xtS.pvalues.loc[dt, curFacNames] = regStats[0][:,2]
            xtS.adjr2.loc[dt, 'adjusted R-squared'] = regStats[2]
            rwt = self.modelDB.loadRobustWeights(\
                    self.rmc.rms_id, univ, dt, reg_id=0, returnDF=True).loc[:,dt]
            rwt = rwt[np.isfinite(rwt)]
            if len(rwt) > 0:
                xtS.adjr2.loc[dt, 'rWtMin'] = rwt.min(axis=None)
                xtS.adjr2.loc[dt, 'rWtMean'] = rwt.mean(axis=None)
                xtS.adjr2.loc[dt, 'rWtNum'] = len(rwt) / float(len(rwt.index))
            else:
                xtS.adjr2.loc[dt, 'rWtMin'] = 1.0
                xtS.adjr2.loc[dt, 'rWtMean'] = 1.0
                xtS.adjr2.loc[dt, 'rWtNum'] = 0.0

            # Do internal regression stats if applicable
            if hasInternal:
                regStats = self.rmc.loadRegressionStatistics(dt, self.modelDB, flag='internal')
                curFacNames = [fct.name for fct in regStats[1]]
                inS.tstats.loc[dt, curFacNames] = regStats[0][:,1]
                inS.pvalues.loc[dt, curFacNames] = regStats[0][:,2]
                inS.adjr2.loc[dt, 'adjusted R-squared'] = regStats[2]
                rwt = self.modelDB.loadRobustWeights(\
                        self.rmc.rms_id, univ, dt, reg_id=0, returnDF=True, flag='internal').loc[:,dt]
                rwt = rwt[np.isfinite(rwt)]
                if len(rwt) > 0:
                    inS.adjr2.loc[dt, 'rWtMin'] = rwt.min(axis=None)
                    inS.adjr2.loc[dt, 'rWtMean'] = rwt.mean(axis=None)
                    inS.adjr2.loc[dt, 'rWtNum'] = len(rwt) / float(len(rwt.index))
                else:
                    inS.adjr2.loc[dt, 'rWtMin'] = 1.0
                    inS.adjr2.loc[dt, 'rWtMean'] = 1.0
                    inS.adjr2.loc[dt, 'rWtNum'] = 0.0

        # Load in the various factor returns
        xtS.factorReturns = self.rmc.loadFactorReturnsHistory(\
                dateList, self.modelDB, factorList=self.allModelFactors, screen_data=True,\
                returnDF=True).loc[self.factorNameList, :]
        if hasInternal:
            inS.factorReturns = self.rmc.loadFactorReturnsHistory(\
                    dateList, self.modelDB, factorList=self.allModelFactors, screen_data=True,
                    table_suffix='internal', returnDF=True).loc[self.factorNameList, :]
            corrDict = dict()
            for fct in self.factorNameList:
                data1 = xtS.factorReturns.loc[fct, :].dropna().values
                data2 = inS.factorReturns.loc[fct, :].dropna().values
                corrDict[fct] = ma.filled(ma.corrcoef(data1, data2), 0.0)[0,1]
            xtS.corrDF = pandas.Series(corrDict)
            xtS.corrDF.to_csv('%s/returnCorrelations-All.csv' % outDir)

        # Output regression stats
        self.outputRegressionStats(xtS, outDir)
        if hasInternal:
            self.outputRegressionStats(inS, intDir)

        # Output ESTU data
        for ky in estuDict.keys():
            estuDict[ky].to_csv('%s/estu-%s.csv' % (outDir, ky))
        subDir = createPath('%s/ESTU' % outDir)
        for estuKey in estuDict.keys():
            if estuKey == 'univ':
                continue
            estuDF = estuDict[estuKey]
            estuDF.loc[:, 'univ_length'] = estuDict['univ'].loc[:, 'length']
            estuDF.loc[:, 'univ_mcap'] = estuDict['univ'].loc[:, 'mcap']
            estuDF.loc[:, 'univ_adv'] = estuDict['univ'].loc[:, 'adv']
            title = '%s Estimation Universe: %d to %d' % (estuKey, self.returnStartDate.year, self.endDate.year)
            fname = '%s/%s.html' % (subDir, estuKey)
            plt = estuDF.fillna(0.0).iplot(asFigure=True, title=title)
            plotly.offline.plot(plt, filename=fname, auto_open=False)

        return

    def outputRegressionStats(self, rsObj, outDir):
        # Write raw data to csv
        rsObj.factorReturns.T.rename(columns=self.factorNameMap).to_csv('%s/FactorReturns.csv' % outDir)
        rsObj.adjr2.to_csv('%s/AdjR-Squared.csv' % outDir)
        rsObj.tstats.rename(columns=self.factorNameMap).to_csv('%s/TStatistics.csv' % outDir)
        rsObj.pvalues.rename(columns=self.factorNameMap).to_csv('%s/PValues.csv' % outDir)

        # Output time-series of r-squares
        title = 'Trailing 30-Day Average Adjusted R-Squared: %d to %d' % (self.returnStartDate.year, self.endDate.year)
        plt = pandas.rolling_mean(rsObj.adjr2, window=30, min_periods=30).dropna().iplot(asFigure=True, title=title)
        plotly.offline.plot(plt, filename='%s/AdjR-Squared.html' % outDir, auto_open=False)

        # Output time series of factor returns
        cumFacRets = (rsObj.factorReturns.fillna(0.0).T + 1.0).cumprod(axis=0) - 1.0
        cumFacRets.rename(columns=self.factorNameMap).to_csv('%s/FactorReturnsCumulative.csv' % outDir)
        rawDir = createPath('%s/FactorReturns-Raw' % outDir)
        cumDir = createPath('%s/FactorReturns-Cumulative' % outDir)
        for fti in self.fTypeIDMap.index:
            subset = [f.name for f in self.allModelFactors if f.type_id == fti]
            ftype = self.fTypeIDMap[fti]
            if (len(subset) > 0) and (ftype != 'Currency'):
                subDir1 = createPath('%s/%s' % (rawDir, ftype))
                subDir2 = createPath('%s/%s' % (cumDir, ftype))
                for fct in subset:
                    # Output raw factor return
                    title = '%s Factor Returns: %d to %d' % (fct, self.returnStartDate.year, self.endDate.year)
                    fname = '%s/%s.html' % (subDir1, fct.replace(',',''))
                    plt = rsObj.factorReturns.loc[fct,:].fillna(0.0).iplot(asFigure=True, kind='bar', title=title)
                    plotly.offline.plot(plt, filename=fname, auto_open=False)
                    # Output cumulative factor return
                    fname = '%s/%s.html' % (subDir2, fct.replace(',',''))
                    plt = cumFacRets.loc[:,fct].fillna(0.0).iplot(asFigure=True, title=title)
                    plotly.offline.plot(plt, filename=fname, auto_open=False)

        if hasattr(rsObj, 'corrDF'):
            # Output correlation between internal and external returns
            corrDir = createPath('%s/ReturnCorrelations' % outDir)
            for fti in self.fTypeIDMap.index:
                subset = [f.name for f in self.allModelFactors if f.type_id == fti]
                ftype = self.fTypeIDMap[fti]
                if (len(subset) > 0) and (ftype != 'Currency'):
                    title = 'Internal/External %s Factor Return Correlations: %d to %d' %\
                                     (ftype, self.returnStartDate.year, self.endDate.year)
                    fname = '%s/ReturnCorrelation-%s.html' % (corrDir, ftype)
                    plt = rsObj.corrDF[subset].iplot(asFigure=True, kind='bar', title=title)
                    plotly.offline.plot(plt, filename=fname, auto_open=False)

        # Output time-series of t-stats
        absTStats = rsObj.tstats.abs()
        rollDir = createPath('%s/TStatTimeSeries' % outDir)
        rollTStat = pandas.rolling_mean(rsObj.tstats.abs().fillna(0.0), window=30, min_periods=30)
        for fti in self.fTypeIDMap.index:
            subset = [f.name for f in self.allModelFactors if f.type_id == fti]
            ftype = self.fTypeIDMap[fti]
            if (len(subset) > 0) and (ftype != 'Currency'):
                subDir = createPath('%s/%s' % (rollDir, ftype))
                for fct in subset:
                    title = '%s 30-day T-Statistics: %d to %d' % (fct, self.returnStartDate.year, self.endDate.year)
                    fname = '%s/TStatistics-%s.html' % (subDir, fct.replace(',',''))
                    plt = rollTStat.loc[:,fct].dropna().iplot(asFigure=True, title=title, bestfit=True)
                    plotly.offline.plot(plt, filename=fname, auto_open=False)

        # Compute summary t-stat statistics
        tsPctSig = ((absTStats >= 2.0).sum(axis=0) / absTStats.count(axis=0)).rename('% Sig')
        meanTStats = absTStats.mean(axis=0).rename('Abs T-Stat')
        tsPctSig = pandas.concat([tsPctSig, meanTStats], axis=1)

        # Plot all t-stats
        title = 'Frequency and Magnitude of Factor Significance: %d to %d' % (self.returnStartDate.year, self.endDate.year)
        fname = '%s/TStatistics.html' % outDir
        plt = tsPctSig.iplot(asFigure=True, kind='bar', title=title)
        plotly.offline.plot(plt, filename=fname, auto_open=False)

        # Plot t-stats by factor type
        for fti in self.fTypeIDMap.index:
            subset = [f.name for f in self.allModelFactors if f.type_id == fti]
            ftype = self.fTypeIDMap[fti]
            if (len(subset) > 0) and (ftype != 'Currency'):
                title1 = '%s factor %s' % (ftype, title)
                fname = '%s/TStatistics-%s.html' % (outDir, ftype)
                plt = tsPctSig.loc[subset,:].iplot(asFigure=True, kind='bar', title=title1)
                plotly.offline.plot(plt, filename=fname, auto_open=False)

    def writeCovStats(self):

        outDir = createPath('%s/CovarianceStats' % self.rmtDir)
        if hasattr(self.rmc, 'twoRegressionStructure'):
            hasInternal = self.rmc.twoRegressionStructure
        else:
            hasInternal = False

        # Initialise
        retDateList = sorted([d for d in self.modelDates if d>= self.returnStartDate])
        rskDateList = sorted([d for d in self.modelDates if d>= self.riskStartDate])
        dateIndex = pandas.to_datetime(rskDateList)
        predFactorVol = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)

        # External stats
        xtS = Utilities.Struct()
        xtS.realFactorVol = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)
        if hasInternal:
            # Internal stats
            inS = Utilities.Struct()
            inS.realFactorVol = pandas.DataFrame(np.nan, index=dateIndex, columns=self.factorNameList)

        # Load in the various factor returns
        xtS.factorReturns = self.rmc.loadFactorReturnsHistory(\
                retDateList, self.modelDB, factorList=self.allModelFactors, screen_data=True,\
                returnDF=True).loc[self.factorNameList, :]
        if hasInternal:
            inS.factorReturns = self.rmc.loadFactorReturnsHistory(\
                    retDateList, self.modelDB, factorList=self.allModelFactors, screen_data=True,
                    table_suffix='internal', returnDF=True).loc[self.factorNameList, :]

        # Get some data date by date
        for ixdt, dt in enumerate(rskDateList):
            print (dt)
            self.rmc.setFactorsForDate(dt, self.modelDB)
            rmi = self.allRMIMap[dt]
            nextMonthDates = rskDateList[ixdt:]
            if len(nextMonthDates) > 20:
                nextMonthDates = nextMonthDates[:21]
            else:
                nextMonthDates = None

            # Load in covariance data
            predCov = self.rmc.loadFactorCovarianceMatrix(rmi, self.modelDB, returnDF=True)

            # Compute realised factor volatility
            if nextMonthDates is not None:
                realVol = Utilities.compute_NWAdj_covariance(\
                                xtS.factorReturns.loc[predCov.columns, nextMonthDates].values,
                                2, deMean=True, varsOnly=True, axis=1)
                realVol = pandas.Series(realVol, index=predCov.columns)
                if hasInternal:
                    realIntVol = Utilities.compute_NWAdj_covariance(\
                                    inS.factorReturns.loc[predCov.columns, nextMonthDates].values,
                                    2, deMean=True, varsOnly=True, axis=1)
                    realIntVol = pandas.Series(realIntVol, index=predCov.columns)

            # Save predicted and realised factor volatilities
            for fct in predCov.columns:
                predFactorVol.loc[dt, fct] = np.sqrt(predCov.loc[fct, fct])
                xtS.realFactorVol.loc[dt, fct] = np.sqrt(252.0*realVol.loc[fct])
                if hasInternal:
                    inS.realFactorVol.loc[dt, fct] = np.sqrt(252.0*realIntVol.loc[fct])

        # Output factor volatilities
        volDir = createPath('%s/FactorVolatility' % outDir)
        predFactorVol.rename(columns=self.factorNameMap).to_csv('%s/PredFactorVol.csv' % outDir)
        
        for fti in self.fTypeIDMap.index:
            subset = [f.name for f in self.allModelFactors if f.type_id == fti]
            ftype = self.fTypeIDMap[fti]
            if len(subset) > 0:
                subDir = createPath('%s/%s' % (volDir, ftype))
                for fct in subset:
                    title = '%s Factor Volatility %d to %d' % (fct, self.riskStartDate.year, self.endDate.year)
                    fname = '%s/%s.html' % (subDir, fct.replace(',',''))
                    data = pandas.concat([predFactorVol.loc[:, fct], xtS.realFactorVol.loc[:, fct].rename('Realised')], axis=1)
                    if hasInternal:
                        data = pandas.concat([data, inS.realFactorVol.loc[:, fct].rename('Realised-Internal')], axis=1)
                    plt = data.fillna(0.0).iplot(asFigure=True, title=title)
                    plotly.offline.plot(plt, filename=fname, auto_open=False)
        predFactorVol.rename(columns=self.factorNameMap).to_csv('%s/PredFactorVol.csv' % outDir)
        xtS.realFactorVol.rename(columns=self.factorNameMap).to_csv('%s/RealFactorVol.csv' % outDir)
        if hasInternal:
            inS.realFactorVol.rename(columns=self.factorNameMap).to_csv('%s/RealInternalFactorVol.csv' % outDir)
        return

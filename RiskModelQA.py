"""Module to run Risk Model QA"""

import logging
import numpy as np
import optparse
import pandas as pd
import pandas.io.sql as sql
import sys
import os
from zipfile import ZipFile
from datetime import datetime, timedelta
import EstuLargeLeaverReport as UniCheck

from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
from Tools.checkUniverse import UniverseChecker

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
matplotlib.style.use('seaborn-whitegrid')


class RMQA:
    def __init__(self, modelDB, marketDB, args, rms, riskModel, testOnly, updateFlag, chart):
        self.modelDB = modelDB
        self.marketDB = marketDB
        self.rms = rms
        self.args = args
        self.riskModel = riskModel
        self.testOnly = testOnly
        self.updateFlag = updateFlag
        self.chart = chart
        self.stdbound = [1, 3, 9]
        self.ChartToZip = list()

    def getdateset(self, window):
        #Create a set of unique dates for checking
        args = self.args
        conn = self.modelDB
        rmg = self.riskModel.rmg
        
        if len(args) == 1:
            dateRanges = [i.strip() for i in args[0].split(',')]
            dates = set()
            for dRange in dateRanges:
                if len(dRange) == 4:
                    dRange = dRange + '-01-01:' + dRange + '-12-31'
                if dRange.find(':') == -1:
                    dates.add(Utilities.parseISODate(dRange))
                else:
                    (startDate, endDate) = [i.strip() for i in dRange.split(':')]
                    startDate = Utilities.parseISODate(startDate)
                    endDate = Utilities.parseISODate(endDate)
                    dates.update([startDate + timedelta(i) for i in range((endDate-startDate).days + 1)])

            modelfullDates = list()
            for dt in dates:
                modelDates = conn.getDateRange(rmg, dt, dt, excludeWeekend=True, tradingDaysBack=window)
                modelfullDates += modelDates
        else:
            datelist = list()
            for i in range(len(args)):
                Date = Utilities.parseISODate(args[i])
                datelist.append(Date)

            modelfullDates = list()

            for dt in datelist:
                modelDates = conn.getDateRange(rmg, dt, dt, excludeWeekend=True, tradingDaysBack=window)
                modelfullDates += modelDates

        return set(modelfullDates)

    def getBound(self, code, mean=None, std=None, df=None):
        #Get bounds formed by mean and std
        if (mean == None or std == None) and df.shape[0] == 0:
            logging.error("No valid inputs for bound calculation")

        if mean is not None and std is not None:
            bound = []
            if code == '0' or code == '2' or code == '4' or 'FE-' in code:
                for i in self.stdbound:
                    bound.append(mean - (i*std))
            elif code == '1' or code == '3':
                for i in self.stdbound:
                    bound.append(mean + (i*std))
            elif code == 'FR' or 'FC-' in code:
                #for Factor Return and Forecast Change
                for i in self.stdbound:
                    bound.append(mean + (i*std))
                    bound.append(mean - (i*std))
            else:
                logging.error("Not recognized code for bound calculation")

            return bound

        if df.shape[0] > 0:
            if code == '0' or code == '2' or code == '4' or 'FE-' in code:
                for i in self.stdbound:
                    df['%d- SIGMA'%i] = df['MEAN'] - (i * df['VALUE'].shift().rolling(250).std())
            elif code == '1' or code == '3':
                for i in self.stdbound:
                    df['%d+ SIGMA'%i] = df['MEAN'] + (i * df['VALUE'].shift().rolling(250).std())
            elif code == 'FR' or 'FC' in code:
                #for Factor Return and Forecast Change
                for i in self.stdbound:
                    df['%d+ SIGMA'%i] = df['MEAN'] + (i * df['VALUE'].shift().rolling(250).std())
                    df['%d- SIGMA'%i] = df['MEAN'] - (i * df['VALUE'].shift().rolling(250).std())
            else:
                logging.error("Not recognized code for bound calculation")
                    
            return df

    def getQAStat(self, rms_id, date, code, rollingCheck=False):
        #Get past records and compute mean & std
        qa_hist_df = pd.DataFrame()
        qaQuery = """select * from (select b.dt, b.value from MODEL_QA_STATS a, MODEL_QA_STATS b where a.RMS_ID = b.RMS_ID and a.CODE = b.CODE and a.RMS_ID = '%(rms_id)s' and a.DT = '%(date)s' and a.dt """ +'%(sign)s'+ """b.dt and a.CODE = '%(code)s' order by b.dt desc) where rownum <%(numdays)d"""
        
        if rollingCheck:
            numdays = 1502
            sign = '>='
        else:
            numdays = 251
            sign = '>'
            
        qaQry = qaQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'numdays': numdays, 'code': code, 'sign': sign}
        qa_hist_df = sql.read_sql(qaQry, self.modelDB.dbConnection)

        if qa_hist_df.shape[0] != (numdays - 1):
            logging.warning("Count of historical QA stats is below/above %d day, please check" %(numdays-1))
            if rollingCheck:
                logging.warning("Not enough QA values for rolling Check, skip generating graph")
                return None

        if rollingCheck:
            logging.debug("rollingCheck is True")
            qa_hist_df.sort_values(by=['DT'], inplace=True)
            qa_hist_df['MEAN'] = qa_hist_df['VALUE'].shift().rolling(250).mean()
            qa_hist_df = self.getBound(code, None, None, df=qa_hist_df)
            return qa_hist_df
        else:
            qa_mean = qa_hist_df['VALUE'].mean()
            qa_std = qa_hist_df['VALUE'].std()
            return [qa_mean, qa_std]

    def outputChart(self, date, section, mnemonic, df, factor=None):
        section_dict={'1': 'ESTU Correlation',
                      '2': 'ESTU Leaver Count',
                      '3': 'Model Universe Correlation',
                      '4': 'Model Universe Leaver Count',
                      '5': 'Factor Return',
                      '6': 'Forecast Change',
                      '7': 'Exposure Correlation',
                      '8': 'Specific Risk Correlation'}
        df.dropna(inplace=True)
        df = df.drop(['MEAN'], axis=1)
        cmap = plt.get_cmap('Set1')
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        for col in df.columns[1:]:
            if col == 'ANALYSIS':
                continue
            elif 'VALUE' in col:
                #Put Marker at the latest value
                plt.plot(df['DT'], df[col], color=cmap(0.13), linewidth = 1.5, marker="o", markersize=5, markevery=[-1])
            elif '1' in col:
                plt.plot(df['DT'], df[col], color=cmap(0.38))
            elif '3' in col:
                plt.plot(df['DT'], df[col], color=cmap(0.5))
            elif '9' in col:
                plt.plot(df['DT'], df[col], color=cmap(0))
        if factor == None:
            title = 'Historical QA for Section %s - %s'%(section, mnemonic)
            fname = 'RMQA_section%s_%s_%s.png'%(section, mnemonic, date)
        else:
            title = 'Historical QA for Section %s [%s] - %s'%(section, factor, mnemonic)
            fname = 'RMQA_section%s_%s_%s_%s.png'%(section, factor, mnemonic, date)
            
        ax.set(ylabel=section_dict.get(section), title=title)
        ax.set_xlim(right=mdates.date2num(date+timedelta(days=20)))

        myFmt = mdates.DateFormatter("%b-%Y")
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis='x', direction='out')
        #Auto-rotation
        fig.autofmt_xdate()
        plt.legend(df.columns[1:], loc="best", prop={'size':8})

        if section!= '5':
            today_row = 0
        else:
            today_row = 1500
            
        #Annotation on the latest value
        if df.loc[today_row, 'ANALYSIS'] == 'Within 3+(inc) & 9+':
            edgecolor = '#FF8000'
        elif df.loc[today_row, 'ANALYSIS'] == 'Above 9+(inc)':
            edgecolor = 'r'

        bbox_param = dict(boxstyle='larrow', fc='w', ec=edgecolor, lw=1)
        ax.annotate("{:.4f}".format(df.loc[today_row, 'VALUE'])+', '+date.strftime("%Y-%m-%d"), (df.loc[today_row, 'DT'], df.loc[today_row, 'VALUE']),
                    xytext=(df.loc[today_row, 'DT']+timedelta(days=30), df.loc[today_row, 'VALUE']), fontsize=8, bbox=bbox_param)

        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.88, top=0.93)

        #Save to directory
        rptDirName = os.path.dirname(self.chart)
        if rptDirName and not os.path.isdir(rptDirName):
            os.makedirs(rptDirName)
            
        FileDir = os.path.abspath(self.chart)
        picPath = os.path.join(FileDir, fname)
        fig.savefig(picPath)
        plt.close(fig)

        #DIA-2840 HTML link of graph
        picPath = picPath.replace("axioma", "")
        picPath = picPath.replace("intranet", "rm-web")
        print(fname, ': ', '<a href="http:%s">%s</a>'%(picPath, 'Graph'))

        #To zip at the end
        self.ChartToZip.append(fname)
            
        return

    def outputZip(self, date):
        if len(self.ChartToZip) == 0:
            return

        dirName = os.path.abspath(self.chart)
        ZipFileName = os.path.join(dirName, 'RMQA_%s_%s.zip'%(self.riskModel.mnemonic, date))
        ChartToZip = self.ChartToZip

        with ZipFile(ZipFileName, 'w') as zipObj:
            for folderName, subfolder, filename in os.walk(dirName):
                for fn in filename:
                    if fn in ChartToZip:
                        chartPath = os.path.join(folderName, fn)
                        #Add to zip file (path of chart, name to display) <--will break into subfolders in zip without name/arcname
                        zipObj.write(chartPath, fn)

        winPath = ZipFileName.replace("axioma", "")
        winPath = winPath.replace("intranet", "rm-web")
        logging.info('<a href="http:%s">%s</a>'%(winPath, 'RMQA_%s_%s.zip'%(self.riskModel.mnemonic, date)))
        
        return

    def generateUniCorrReport(self, date, UniType):
        correlationdf = UniCheck.getCorrelation(self.modelDB, self.marketDB, self.riskModel, date, self.riskModel.rms_id, UniType)
        logging.info("%s CorrelCoef on %s is %s", UniType, date, correlationdf.iloc[0]['CorrelCoef'])

        if UniType == "ESTU":
            code = '0'
        elif UniType == "Model":
            code = '2'
        #Check if today's value exist in DB for stat calculation
        checkQuery = """select * from MODEL_QA_STATS where RMS_ID = '%(rms_id)s' and DT = '%(date)s' and CODE = '%(code)s'"""
        checkQry = checkQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'code': code}
        check_df = sql.read_sql(checkQry, self.modelDB.dbConnection)
        if check_df.shape[0] == 0:
            cur = self.modelDB.dbCursor
            insertQuery = """INSERT INTO MODEL_QA_STATS VALUES('%(date)s', '%(rms_id)s', sysdate, %(correlation)s, '%(code)s')"""
            cur.execute(insertQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'code': code, 'correlation': correlationdf.iloc[0]['CorrelCoef']})
            logging.info("Inserting 1 value - %s CorrelCoef", UniType)
        elif self.updateFlag:
            cur = self.modelDB.dbCursor
            updateQuery = """UPDATE MODEL_QA_STATS SET VALUE = %(correlation)s, REV_DT = sysdate 
                             where DT = '%(date)s' and RMS_ID = '%(rms_id)s' and code = '%(code)s'"""
            cur.execute(updateQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'code': code, 'correlation': correlationdf.iloc[0]['CorrelCoef']})
            logging.info("Updating 1 value - %s CorrelCoef", UniType)
        else:
            logging.debug("No insertion/update")

        if check_df.shape[0] == 0 or self.updateFlag:
            if self.testOnly:
                self.modelDB.dbConnection.rollback()
                logging.info("Rolling-back Changes")
            else:
                self.modelDB.dbConnection.commit()
                logging.info("Committing Changes")

        #stat for as-of-date obs
        stat = self.getQAStat(self.riskModel.rms_id, date, code)
        bound = self.getBound(code, stat[0], stat[1])

        alert = 0
        if correlationdf.iloc[0]['CorrelCoef'] > bound[0]:
            logging.info("%s Correlation Check Result: Below 1+ (SAFE)", UniType)
        elif bound[1] < correlationdf.iloc[0]['CorrelCoef'] <= bound[0]:
            logging.info("%s Correlation Check Result: Within 1+(inc) & 3+ (SAFE)", UniType)
        elif bound[2] < correlationdf.iloc[0]['CorrelCoef'] <= bound[1]:
            logging.info("%s Correlation Check Result: Within 3+(inc) & 9+ (SAFE)", UniType)
        else:
            logging.error("%s Correlation Check Result: Above 9+(inc) (ALERT)", UniType)
            logging.error("(Threshold: %.5f (9+))", bound[2])
            alert += 1

        if alert == 0:
            return alert
        elif alert == 1:
            statdf = self.getQAStat(self.riskModel.rms_id, date, code, True)
            conditions = [statdf['VALUE'] > statdf['1- SIGMA'],
                          (statdf['3- SIGMA'] < statdf['VALUE']) & (statdf['VALUE'] <= statdf['1- SIGMA']),
                          (statdf['9- SIGMA'] < statdf['VALUE']) & (statdf['VALUE'] <= statdf['3- SIGMA']),
                          statdf['VALUE'] <= statdf['9- SIGMA']]
            analysis = ["Below 1+", "Within 1+(inc) & 3+", "Within 3+(inc) & 9+", "Above 9+(inc)"]
            statdf['ANALYSIS'] = np.select(conditions, analysis, default = None)
            if self.chart is not None:
                if UniType == 'ESTU':
                    section = '1'
                elif UniType == 'Model':
                    section = '3'
                self.outputChart(date, section, self.riskModel.mnemonic, statdf)
            countdf = statdf.iloc[:-1, :].groupby(['ANALYSIS']).size().to_frame(name='COUNTS').reindex(analysis)
            print("Past %d-day Count Analysis:\n"%countdf['COUNTS'].sum(), countdf.to_string())
            return alert
        else:
            logging.error("alert=%d, please check" %alert)
            sys.exit(1)

    def generateLeaverReport(self, date, UniType):
        V3 = UniCheck.getVersionMap(self.modelDB, self.riskModel.rms_id, date)
        #Check # of leavers (Exclude delisted assets for ESTU)
        leaversdf = UniCheck.getLeaver(self.modelDB, date, self.riskModel.rms_id, self.riskModel.rmg, UniType, v3=V3)
        if leaversdf.shape[0] == 0:
            leavercount = 0
        else:
            leavercount = leaversdf.iloc[0]['COUNT']
        logging.info("%s Leaver count on %s is %s", UniType, date, leavercount)

        if UniType == "ESTU":
            code = '1'
        elif UniType == "Model":
            code = '3'
        #Check if today's value exist in DB for stat calculation
        checkQuery = """select * from MODEL_QA_STATS where RMS_ID = '%(rms_id)s' and DT = '%(date)s' and CODE = '%(code)s'"""
        checkQry = checkQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'code': code}
        check_df = sql.read_sql(checkQry, self.modelDB.dbConnection)
        if check_df.shape[0] == 0:
            cur = self.modelDB.dbCursor
            insertQuery = """INSERT INTO MODEL_QA_STATS VALUES('%(date)s', '%(rms_id)s', sysdate, %(leavercount)s, '%(code)s')"""
            cur.execute(insertQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'code': code, 'leavercount': leavercount})
            logging.info("Inserting 1 value - %s Leaver Count", UniType)
        elif self.updateFlag:
            cur = self.modelDB.dbCursor
            updateQuery = """UPDATE MODEL_QA_STATS SET VALUE = %(leavercount)s, REV_DT = sysdate
                             where DT = '%(date)s' and RMS_ID = '%(rms_id)s' and code = '%(code)s'"""
            cur.execute(updateQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'code': code, 'leavercount': leavercount})
            logging.info("Updating 1 value - %s Leaver Count", UniType)
        else:
            logging.debug("No insertion/update")

        if check_df.shape[0] == 0 or self.updateFlag:
            if self.testOnly:
                self.modelDB.dbConnection.rollback()
                logging.info("Rolling-back Changes")
            else:
                self.modelDB.dbConnection.commit()
                logging.info("Committing Changes")

        #stat for as-of-date obs
        stat = self.getQAStat(self.riskModel.rms_id, date, code)
        bound = self.getBound(code, stat[0], stat[1])

        alert = 0
        if leavercount < bound[0]:
            logging.info("%s Leaver Count Check Result: Below 1+ (SAFE)", UniType)
        elif bound[0] <= leavercount < bound[1]:
            logging.info("%s Leaver Count Check Result: Within 1+(inc) & 3+ (SAFE)", UniType)
        elif bound[1] <= leavercount < bound[2]:
            logging.info("%s Leaver Count Check Result: Within 3+(inc) & 9+ (SAFE)", UniType)
        else:
            logging.error("%s Leaver Count Check Result: Above 9+(inc) (ALERT)", UniType)
            logging.error("(Threshold: %.5f (9+))", bound[2])
            alert += 1

        if alert == 0:
            return alert
        elif alert == 1:
            statdf = self.getQAStat(self.riskModel.rms_id, date, code, True)
            conditions = [statdf['VALUE'] < statdf['1+ SIGMA'],
                          (statdf['3+ SIGMA'] > statdf['VALUE']) & (statdf['VALUE'] >= statdf['1+ SIGMA']),
                          (statdf['9+ SIGMA'] > statdf['VALUE']) & (statdf['VALUE'] >= statdf['3+ SIGMA']),
                          statdf['VALUE'] >= statdf['9+ SIGMA']]
            analysis = ["Below 1+", "Within 1+(inc) & 3+", "Within 3+(inc) & 9+", "Above 9+(inc)"]
            statdf['ANALYSIS'] = np.select(conditions, analysis, default = None)
            if self.chart is not None:
                if UniType == 'ESTU':
                    section = '2'
                elif UniType == 'Model':
                    section = '4'
                self.outputChart(date, section, self.riskModel.mnemonic, statdf)
            countdf = statdf.iloc[:-1, :].groupby(['ANALYSIS']).size().to_frame(name='COUNTS').reindex(analysis)
            print("Past %d-day Count Analysis:\n"%countdf['COUNTS'].sum(), countdf.to_string())
            return alert
        else:
            logging.error("alert=%d, please check" %alert)
            sys.exit(1)

    def generateFactorReturnReport(self, date):
        dateList = self.modelDB.getDateRange(self.riskModel.rmg, date, date, excludeWeekend=True, tradingDaysBack=251)

        #Excluding Stat factor check
        self.modelDB.dbCursor.execute("""select f.FACTOR_ID from factor f where f.FACTOR_TYPE_ID in (6)""")
        ret = self.modelDB.dbCursor.fetchall()
        excludelist = [r[0] for r in ret]

        #Find factors for specfic rms_id on specific date and setup factor-related variable
        self.riskModel.setFactorsForDate(dt, self.modelDB)
        Factors = self.riskModel.factors
        Factors = [f for f in self.riskModel.factors if f.factorID not in excludelist]
        if len(Factors) == 0:
            logging.info("All factors are excluded, skipping Factor Return check")
            return 0
        subFactors = self.modelDB.getSubFactorsForDate(date, Factors)
        logging.info("Loaded %d Factors, %d subFactors from risk model code" %(len(Factors), len(subFactors)))
        #Load Factor Return and turn output to DF
        facReturnM = self.modelDB.loadFactorReturnsHistory(self.riskModel.rms_id, subFactors, dateList)
        facReturnDF = facReturnM.toDataFrame()
        facReturnDF = facReturnDF.transpose().fillna(0.0)

        #Loop for each subFactor, compute stats and output QA message
        totalalert = 0
        for col in facReturnDF.columns:
            alert = 0
            tempDF = pd.DataFrame()
            tempDF['VALUE'] = facReturnDF[col][:250]
            todayReturn = facReturnDF[col][250]
            qa_mean = tempDF['VALUE'].mean()
            qa_std = tempDF['VALUE'].std()
            #Skip check for subFactors always with zero return -- meaningless bound
            if qa_mean == 0 and qa_std == 0:
                continue
            bound = self.getBound('FR', qa_mean, qa_std)
            bound.sort()

            if todayReturn > bound[2] and todayReturn < bound[3]:
                logging.debug("'%s'(subFactorID=%d, todayReturn=%.5f) Factor Return Check Result: Below 1+ (SAFE)", col.factor.name, col.subFactorID, todayReturn)
            elif (todayReturn >= bound[3] and todayReturn < bound[4]) or (todayReturn > bound[1] and todayReturn <= bound[2]):
                logging.debug("'%s'(subFactorID=%d, todayReturn=%.5f) Factor Return Check Result: Within 1+(inc) & 3+ (SAFE)", col.factor.name, col.subFactorID, todayReturn)
            elif (todayReturn >= bound[4] and todayReturn < bound[5]) or (todayReturn > bound[0] and todayReturn <= bound[1]):
                logging.debug("'%s'(subFactorID=%d, todayReturn=%.5f) Factor Return Check Result: Within 3+(inc) & 9+ (SAFE)", col.factor.name, col.subFactorID, todayReturn)
            else:
                logging.error("'%s'(subFactorID=%d, todayReturn=%.5f) Factor Return Check Result: Above 9+(inc) (ALERT)", col.factor.name, col.subFactorID, todayReturn)
                logging.error("(Threshold: %.5f (9+) or %.5f (9-))", bound[5], bound[0])
                alert += 1
            #Count how many factors with alert
            totalalert += alert

            if alert == 0:
                continue
            elif alert == 1:
                #rollingCheck for subFactor(s) with alert
                RollDateList = self.modelDB.getDateRange(self.riskModel.rmg, date, date, excludeWeekend=True, tradingDaysBack=1501)
                RollReturnM = self.modelDB.loadFactorReturnsHistory(self.riskModel.rms_id, [col], RollDateList)
                RollReturnDF = RollReturnM.toDataFrame()
                RollReturnDF = RollReturnDF.transpose().fillna(0.0)
                RollReturnDF.rename(columns={col: 'VALUE'}, inplace=True)

                RollReturnDF['MEAN'] = RollReturnDF['VALUE'].shift().rolling(250).mean()
                RollReturnDF = self.getBound('FR', None, None, df=RollReturnDF)
                conditions = [(RollReturnDF['VALUE'] > RollReturnDF['1- SIGMA']) & (RollReturnDF['VALUE'] < RollReturnDF['1+ SIGMA']),
                              ((RollReturnDF['VALUE'] >= RollReturnDF['1+ SIGMA']) & (RollReturnDF['VALUE'] < RollReturnDF['3+ SIGMA'])) |
                              ((RollReturnDF['VALUE'] > RollReturnDF['3- SIGMA']) & (RollReturnDF['VALUE'] <= RollReturnDF['1- SIGMA'])),
                              ((RollReturnDF['VALUE'] >= RollReturnDF['3+ SIGMA']) & (RollReturnDF['VALUE'] < RollReturnDF['9+ SIGMA'])) |
                              ((RollReturnDF['VALUE'] > RollReturnDF['9- SIGMA']) & (RollReturnDF['VALUE'] <= RollReturnDF['3- SIGMA'])),
                              ((RollReturnDF['VALUE'] >= RollReturnDF['9+ SIGMA']) | (RollReturnDF['VALUE'] <= RollReturnDF['9- SIGMA']))]
                analysis = ["Below 1+", "Within 1+(inc) & 3+", "Within 3+(inc) & 9+", "Above 9+(inc)"]
                RollReturnDF['ANALYSIS'] = np.select(conditions, analysis, default = None)
                RollReturnDF.index.name = 'DT'
                RollReturnDF.reset_index(inplace=True)
                if self.chart is not None:
                    self.outputChart(date, '5', self.riskModel.mnemonic, RollReturnDF, col.factor.name)
                countdf = RollReturnDF.iloc[:-1, :].groupby(['ANALYSIS']).size().to_frame(name='COUNTS').reindex(analysis)
                print("Past %d-day Count Analysis:\n"%countdf['COUNTS'].sum(), countdf.to_string())
            else:
                logging.error("alert=%d, please check" %alert)
                sys.exit(1)

        if totalalert == 0:
            msg = "SAFE"
        else:
            msg = "ALERT"
        logging.info("Final Factor Return Check Result: %d out of %d subFactors pass (%s)", len(subFactors)-totalalert, len(subFactors), msg)
        return totalalert

    def generateFactorVolTestReport(self, date):
        #No Covariance data for Factor Library
        if 'FL' in self.riskModel.mnemonic or 'EL' in self.riskModel.mnemonic:
            logging.info("Skipping Forecast Change Check: Factor Library")
            return 0
        
        dateList = self.modelDB.getDateRange(self.riskModel.rmg, date, date, excludeWeekend=True, tradingDaysBack=2)

        #Excluding Stat factor check
        self.modelDB.dbCursor.execute("""select f.FACTOR_ID from factor f where f.FACTOR_TYPE_ID in (6)""")
        ret = self.modelDB.dbCursor.fetchall()
        excludelist = [r[0] for r in ret]

        #Find factors for specfic rms_id on specific date and setup factor-related variable
        self.riskModel.setFactorsForDate(dt, self.modelDB)
        Factors = self.riskModel.factors
        Factors = [f for f in self.riskModel.factors if f.factorID not in excludelist]
        if len(Factors) == 0:
            logging.info("All factors are excluded, skipping Factor Volatility Test Check")
            return 0
        subFactors = self.modelDB.getSubFactorsForDate(date, Factors)
        logging.info("Loaded %d Factors, %d subFactors from risk model code" %(len(Factors), len(subFactors)))
        #Load Factor risk (root of variance)
        facVarianceM = self.modelDB.loadFactorVolatilityHistory(self.riskModel.rms_id, subFactors, dateList[0], dateList[1])
        facVarianceDF = facVarianceM.toDataFrame()
        #Compute FC and remove factor with zero variances/inf/NaN FC
        facVarianceDF['FC'] = (facVarianceDF[dateList[1]] - facVarianceDF[dateList[0]])/facVarianceDF[dateList[0]]
        if facVarianceDF[(facVarianceDF[dateList[0]] <= 0)].shape[0] > 0 or facVarianceDF[(facVarianceDF[dateList[1]] < 0)].shape[0] > 0:
            logging.info("Skipping VolTest for %s - Zero variance (prev_dt)" % facVarianceDF[facVarianceDF[dateList[0]] <= 0].index.values)
            logging.info("Skipping VolTest for %s - NaN FC" % facVarianceDF[facVarianceDF['FC'].isnull()].index.values)
        facVarianceDF = facVarianceDF[(facVarianceDF[dateList[0]] > 0) & (facVarianceDF[dateList[1]] >= 0)]
        logging.info("Total of %d factors to check Forecast Change " % len(facVarianceDF.index.values))

        #QA
        totalalert = 0
        for index, row in facVarianceDF.iterrows():
            codeQuery = """select * from QA_CODE_REF where CODE = 'FC-%d'""" % index.subFactorID
            code_df = sql.read_sql(codeQuery, self.modelDB.dbConnection)
            #Add code for new factor
            if code_df.shape[0] == 0:
                cur = self.modelDB.dbCursor
                insertcodeQuery = """INSERT INTO QA_CODE_REF VALUES ('FC-%(subFactorID)d', 941, '%(factor_name)s Forecast Change', sysdate)"""
                cur.execute(insertcodeQuery % {'subFactorID': index.subFactorID , 'factor_name': index.factor.name})
                logging.info("Inserting 1 code - %s Forecast Change" % index.factor.name)
                if self.testOnly:
                    self.modelDB.dbConnection.rollback()
                    logging.info("Rolling-back Changes")
                else:
                    self.modelDB.dbConnection.commit()
                    logging.info("Committing Changes")

            #Check if today's value exist in DB for stat calculation
            checkQuery = """select * from MODEL_QA_STATS where RMS_ID = '%(rms_id)s' and DT = '%(date)s' and CODE = 'FC-%(subFactorID)d'"""
            checkQry = checkQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'subFactorID': index.subFactorID}
            check_df = sql.read_sql(checkQry, self.modelDB.dbConnection)
            if check_df.shape[0] == 0:
                cur = self.modelDB.dbCursor
                insertQuery = """INSERT INTO MODEL_QA_STATS VALUES('%(date)s', '%(rms_id)s', sysdate, %(FC)s, 'FC-%(subFactorID)d')"""
                cur.execute(insertQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'subFactorID': index.subFactorID, 'FC': row['FC']})
                logging.info("Inserting 1 value - %s Forecast Change" % index.factor.name)
            elif self.updateFlag:
                cur = self.modelDB.dbCursor
                updateQuery = """UPDATE MODEL_QA_STATS SET VALUE = %(FC)s, REV_DT = sysdate where DT = '%(date)s' 
                                 and RMS_ID = '%(rms_id)s' and code = 'FC-%(subFactorID)d'"""
                cur.execute(updateQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'subFactorID': index.subFactorID, 'FC': row['FC']})
                logging.info("Updating 1 value - %s Forecast Change" % index.factor.name)
            else:
                logging.debug("No insertion/update")

            if check_df.shape[0] == 0 or self.updateFlag:
                if self.testOnly:
                    self.modelDB.dbConnection.rollback()
                    logging.info("Rolling-back Changes")
                else:
                    self.modelDB.dbConnection.commit()
                    logging.info("Committing Changes")

            code = 'FC-%d' % index.subFactorID
            #stat for as-of-date obs
            stat = self.getQAStat(self.riskModel.rms_id, date, code)
            #Skip QA if it's the new factor's 1st date having FC
            if np.isnan(stat[0]) or np.isnan(stat[1]):
                continue
            bound = self.getBound(code, stat[0], stat[1])
            bound.sort()

            alert = 0
            if row['FC'] > bound[2] and row['FC'] < bound[3]:
                logging.debug("'%s'(subFactorID=%d, todayFC=%.5f) Forecast Change Check Result: Below 1+ (SAFE)", index.factor.name, index.subFactorID, row['FC'])
            elif (row['FC'] >= bound[3] and row['FC'] < bound[4]) or (row['FC'] > bound[1] and row['FC'] <= bound[2]):
                logging.debug("'%s'(subFactorID=%d, todayFC=%.5f) Forecast Change Check Result: Within 1+(inc) & 3+ (SAFE)", index.factor.name, index.subFactorID, row['FC'])
            elif (row['FC'] >= bound[4] and row['FC'] < bound[5]) or (row['FC'] > bound[0] and row['FC'] <= bound[1]):
                logging.debug("'%s'(subFactorID=%d, todayFC=%.5f) Forecast Change Check Result: Within 3+(inc) & 9+ (SAFE)", index.factor.name, index.subFactorID, row['FC'])
            else:
                #[DIA-2838] Ignore inflated FC due to extremely low factor volatility
                if row[dateList[0]] < 0.00001 and row[dateList[1]] < 0.00001:
                    logging.info("'%s'(subFactorID=%d, todayFC=%.5f) Forecast Change Check Result: Above 9+(inc) but ignore due today & prev factor risks < 0.00001 (SAFE)", index.factor.name, index.subFactorID,row['FC'])
                else:
                    logging.error("'%s'(subFactorID=%d, todayFC=%.5f) Forecast Change Check Result: Above 9+(inc) (ALERT)", index.factor.name, index.subFactorID, row['FC'])
                    logging.error("(Threshold: %.5f (9+) or %.5f (9-))", bound[5], bound[0])
                    alert += 1

            #Count how many factors with alert
            totalalert += alert

            if alert == 0:
                continue
            elif alert == 1:
                statdf = self.getQAStat(self.riskModel.rms_id, date, code, True)
                conditions = [(statdf['VALUE'] > statdf['1- SIGMA']) & (statdf['VALUE'] < statdf['1+ SIGMA']),
                              ((statdf['VALUE'] >= statdf['1+ SIGMA']) & (statdf['VALUE'] < statdf['3+ SIGMA'])) |
                              ((statdf['VALUE'] > statdf['3- SIGMA']) & (statdf['VALUE'] <= statdf['1- SIGMA'])),
                              ((statdf['VALUE'] >= statdf['3+ SIGMA']) & (statdf['VALUE'] < statdf['9+ SIGMA'])) |
                              ((statdf['VALUE'] > statdf['9- SIGMA']) & (statdf['VALUE'] <= statdf['3- SIGMA'])),
                              ((statdf['VALUE'] >= statdf['9+ SIGMA']) | (statdf['VALUE'] <= statdf['9- SIGMA']))]
                analysis = ["Below 1+", "Within 1+(inc) & 3+", "Within 3+(inc) & 9+", "Above 9+(inc)"]
                statdf['ANALYSIS'] = np.select(conditions, analysis, default = None)
                if self.chart is not None:
                    self.outputChart(date, '6', self.riskModel.mnemonic, statdf, index.factor.name)
                countdf = statdf.iloc[:-1, :].groupby(['ANALYSIS']).size().to_frame(name='COUNTS').reindex(analysis)
                print("Past %d-day Count Analysis:\n"%countdf['COUNTS'].sum(), countdf.to_string())
            else:
                logging.error("alert=%d, please check" %alert)
                sys.exit(1)

        if totalalert == 0:
            msg = "SAFE"
        else:
            msg = "ALERT"
        logging.info("Final Forecast Change Check Result: %d out of %d subFactors pass (%s)", len(facVarianceDF.index.values)-totalalert, len(facVarianceDF.index.values), msg)
        return totalalert
        
    def generateExposureCorrReport(self, date):
        #No interested factor for statistical models
        if '-S' in self.riskModel.mnemonic:
            self.modelDB.dbCursor.execute("""select * from RMS_FACTOR rf, factor f , factor_type ft where rf.RMS_ID = '%s' 
            and rf.FACTOR_ID=f.FACTOR_ID and f.FACTOR_TYPE_ID=ft.FACTOR_TYPE_ID and ft.FACTOR_TYPE_ID in (0, 9, 10, 11) and rf.THRU_DT>'%s' and rf.FROM_DT<='%s'"""
            %(self.riskModel.rms_id, str(date), str(date)))
            
            if len(self.modelDB.dbCursor.fetchall()) == 0:
                logging.info("Skipping Factor Exposure Correlation Check: Statistical Model")
                return 0
            
        dateList = self.modelDB.getDateRange(self.riskModel.rmg, date, date, excludeWeekend=True, tradingDaysBack=2)

        #rms_id=501: interested in factor_type_id = 7 as well
        if self.riskModel.rms_id == 501:
            self.modelDB.dbCursor.execute("""select f.FACTOR_ID from factor f where f.FACTOR_TYPE_ID in (0, 7, 9, 10, 11)""")
            ret = self.modelDB.dbCursor.fetchall()
            checklist = [r[0] for r in ret]
        else:
            #Only interested in Factor Type ID in (0, 9, 10, 11)
            self.modelDB.dbCursor.execute("""select f.FACTOR_ID from factor f where f.FACTOR_TYPE_ID in (0, 9, 10, 11)""")
            ret = self.modelDB.dbCursor.fetchall()
            checklist = [r[0] for r in ret]

        expDFMap = dict()
        
        for index, dt in list(enumerate(dateList)):
            logging.info("Screening factors for QA on %s" % str(dt))
            #Find factors for specfic rms_id on specific date and setup factor-related variable
            self.riskModel.setFactorsForDate(dt, self.modelDB)
            rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, dt)
            estu = self.modelDB.getRiskModelInstanceESTU(rmi)
            expM = self.riskModel.loadExposureMatrix(rmi, self.modelDB)
            expDF = expM.toDataFrame()
            logging.info("Extracted %d factors from risk model code" % (len(list(expDF.columns.values))))

            #Screening out not required factor(s) -- columns
            for f in self.riskModel.factors:
                if f.factorID not in checklist:
                    logging.debug("Skipping Exposure CorrelCoef check for not required factor %s" % f.name)
                    if f.name in list(expDF.columns.values):
                        expDF = expDF.drop([f.name], axis=1)
                else:
                    if f.name not in list(expDF.columns.values):
                        logging.error("No Exposure data for required factor %s" % f.name)
                        sys.exit(1)
            logging.info("Total of %d factors to check Exposure CorrelCoef" % len(list(expDF.columns.values)))

            if len(list(expDF.columns.values)) < 1:
                logging.error("No Factor Exposure to check for model %s" % self.riskModel.mnemonic)
                sys.exit(1)
                            
            #Screening out non ESTU assets from ExpM -- rows
            id_tocheck = list(set(estu).intersection(set(expM.getAssets())))
            expDF = expDF[expDF.index.isin(id_tocheck)]
            expDF = expDF.fillna(0.0)
            expDFMap[dt] = expDF

        L = dict()
        #Keep only assets alive in both today and prev Exp Matrix (limit to ESTU already)
        common_index = expDFMap.get(dateList[1]).index.intersection(expDFMap.get(dateList[0]).index)
        todaydf = expDFMap.get(dateList[1])[expDFMap.get(dateList[1]).index.isin(common_index)]
        prevdf = expDFMap.get(dateList[0])[expDFMap.get(dateList[0]).index.isin(common_index)]

        missingfactor_tocheck = [i for i in list(todaydf.columns.values) if i not in list(prevdf.columns.values)]
        for i, factor_name in list(enumerate(todaydf.columns.values)):
            S = dict()
            #Skip check if prev Exp Matrix doesn't contain today's factor
            if factor_name in missingfactor_tocheck:
                logging.error("Factor Exposure cannot be compared: %s not in previous day's expM" % factor_name)
                continue
            else:
                S['Date'] = dt
                S['Factor'] = factor_name
                S['CorrelCoef'] = np.corrcoef(todaydf[factor_name], prevdf[factor_name])[0][1]
            L[i] = S
        
        corrdf = pd.DataFrame.from_dict(L, orient='index').reindex_axis(['Date', 'Factor', 'CorrelCoef'], axis=1)
        #Get subFactors for mapping code/column
        subFactors = self.modelDB.getSubFactorsForDate(date, self.riskModel.factors)
        for sf in subFactors:
            corrdf.loc[corrdf['Factor']==sf.factor.name, 'subFactorID'] = sf.subFactorID
        corrdf = corrdf.astype({'subFactorID': int})

        #QA
        totalalert = 0
        skipped = 0
        for index, row in corrdf.iterrows():
            codeQuery = """select * from QA_CODE_REF where CODE = 'FE-%d'""" % row['subFactorID']
            code_df = sql.read_sql(codeQuery, self.modelDB.dbConnection)
            #Add code for new factor 
            if code_df.shape[0] == 0:
                cur = self.modelDB.dbCursor
                insertcodeQuery = """INSERT INTO QA_CODE_REF VALUES ('FE-%(subFactorID)d', 941, '%(factor_name)s Exposure Correlation', sysdate)"""
                cur.execute(insertcodeQuery % {'subFactorID': row['subFactorID'], 'factor_name': row['Factor']})
                logging.info("Inserting 1 code - %s Exposure Correlation" % row['Factor'])
                if self.testOnly:
                    self.modelDB.dbConnection.rollback()
                    logging.info("Rolling-back Changes")
                else:
                    self.modelDB.dbConnection.commit()
                    logging.info("Committing Changes")

            #Check if today's value exist in DB for stat calculation
            checkQuery = """select * from MODEL_QA_STATS where RMS_ID = '%(rms_id)s' and DT = '%(date)s' and CODE = 'FE-%(subFactorID)d'"""
            checkQry = checkQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'subFactorID': row['subFactorID']}
            check_df = sql.read_sql(checkQry, self.modelDB.dbConnection)
            if check_df.shape[0] == 0:
                cur = self.modelDB.dbCursor
                insertQuery = """INSERT INTO MODEL_QA_STATS VALUES('%(date)s', '%(rms_id)s', sysdate, %(FECorrel)s, 'FE-%(subFactorID)d')"""
                cur.execute(insertQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'subFactorID': row['subFactorID'], 'FECorrel': row['CorrelCoef']})
                logging.info("Inserting 1 value - %s Exposure CorrelCoef" % row['Factor'])
            elif self.updateFlag:
                cur = self.modelDB.dbCursor
                updateQuery = """UPDATE MODEL_QA_STATS SET VALUE = %(FECorrel)s, REV_DT = sysdate where DT = '%(date)s' and 
                                 RMS_ID = '%(rms_id)s' and code = 'FE-%(subFactorID)d'"""
                cur.execute(updateQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'subFactorID': row['subFactorID'], 'FECorrel': row['CorrelCoef']})
                logging.info("Updating 1 value - %s Exposure CorrelCoef" % row['Factor'])
            else:
                logging.debug("No insertion/update")

            if check_df.shape[0] == 0 or self.updateFlag:
                if self.testOnly:
                    self.modelDB.dbConnection.rollback()
                    logging.info("Rolling-back Changes")
                else:
                    self.modelDB.dbConnection.commit()
                    logging.info("Committing Changes")

            code = 'FE-%d' % row['subFactorID']
            #stat for as-of-date obs
            stat = self.getQAStat(self.riskModel.rms_id, date, code)
            #Skip QA if it's the new factor's 1st date having FE
            if np.isnan(stat[0]) or np.isnan(stat[1]):
                continue
            if stat[1] == 0:
                logging.info("Skipping Factor Exposure Check for %s: zero std", row['Factor'])
                skipped += 1
                continue
            bound = self.getBound(code, stat[0], stat[1])

            alert = 0
            if row['CorrelCoef'] > bound[0]:
                logging.debug("'%s'(subFactorID=%d, todayCorrel=%.5f) Exposure Correlation Check Result: Below 1+ (SAFE)", row['Factor'], row['subFactorID'], row['CorrelCoef'])
            elif bound[1] < row['CorrelCoef'] <= bound[0]:
                logging.debug("'%s'(subFactorID=%d, todayCorrel=%.5f) Exposure Correlation Check Result: Within 1+(inc) & 3+ (SAFE)", row['Factor'], row['subFactorID'], row['CorrelCoef'])
            elif bound[2] < row['CorrelCoef'] <= bound[1]:
                logging.debug("'%s'(subFactorID=%d, todayCorrel=%.5f) Exposure Correlation Check Result: Within 3+(inc) & 9+ (SAFE)", row['Factor'], row['subFactorID'], row['CorrelCoef'])
            else:
                logging.error("'%s'(subFactorID=%d, todayCorrel=%.5f) Exposure Correlation Check Result: Above 9+(inc) (ALERT)", row['Factor'], row['subFactorID'], row['CorrelCoef'])
                logging.error("(Threshold: %.5f (9+))", bound[2])
                alert += 1
            #Count how many factors with alert
            totalalert += alert

            if alert == 0:
                continue
            elif alert == 1:
                statdf = self.getQAStat(self.riskModel.rms_id, date, code, True)
                if statdf is None:
                    continue
                conditions = [statdf['VALUE'] > statdf['1- SIGMA'],
                             (statdf['3- SIGMA'] < statdf['VALUE']) & (statdf['VALUE'] <= statdf['1- SIGMA']),
                             (statdf['9- SIGMA'] < statdf['VALUE']) & (statdf['VALUE'] <= statdf['3- SIGMA']),
                             statdf['VALUE'] <= statdf['9- SIGMA']]
                analysis = ["Below 1+", "Within 1+(inc) & 3+", "Within 3+(inc) & 9+", "Above 9+(inc)"]
                statdf['ANALYSIS'] = np.select(conditions, analysis, default = None)
                if self.chart is not None:
                    self.outputChart(date, '7', self.riskModel.mnemonic, statdf, row['Factor'])
                countdf = statdf.iloc[:-1, :].groupby(['ANALYSIS']).size().to_frame(name='COUNTS').reindex(analysis)
                print("Past %d-day Count Analysis:\n"%countdf['COUNTS'].sum(), countdf.to_string())
            else:
                logging.error("alert=%d, please check" %alert)
                sys.exit(1)

        if totalalert == 0:
            msg = "SAFE"
        else:
            msg = "ALERT"
        logging.info("Final Factor Exposure Check Result: %d out of %d subFactors pass (%s)", corrdf.shape[0]-totalalert-skipped, corrdf.shape[0]-skipped, msg)
        return totalalert

    def generateSpecificRiskCorrReport(self, date):
        #No Specific Risk data for Factor Library
        if 'FL' in self.riskModel.mnemonic or 'EL' in self.riskModel.mnemonic:
            logging.info("Skipping Specific Risk Correlation Check: Factor Library")
            return 0

        dateList = self.modelDB.getDateRange(self.riskModel.rmg, date, date, excludeWeekend=True, tradingDaysBack=2)

        SpRskMap = dict()
        for index, dt in list(enumerate(dateList)):
            rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, dt)
            estu = self.modelDB.getRiskModelInstanceESTU(rmi)
            #This function gets both specifc risk [0] and covariance [1]
            SpRsk = self.riskModel.loadSpecificRisks(rmi, self.modelDB)[0]
            SpRskDF = pd.DataFrame(list(SpRsk.items()), columns=['SUBISSUE', 'VALUE'])
            #Only compare data for ESTU assets
            SpRskDF = SpRskDF[SpRskDF['SUBISSUE'].isin(estu)].reset_index(drop=True)
            SpRskMap[dt] = SpRskDF

        todaydf = SpRskMap.get(dateList[1])
        prevdf = SpRskMap.get(dateList[0]).rename(columns={'VALUE': 'VALUE_PREV'})
        #Common assets in ESTU
        mergedf = pd.merge(todaydf, prevdf, how='inner', on=['SUBISSUE'])
        logging.info("Specific Risk CorrelCoef on %s is %s", dt, np.corrcoef(mergedf.VALUE, mergedf.VALUE_PREV)[0][1])
        todaySpRskCorrel = np.corrcoef(mergedf.VALUE, mergedf.VALUE_PREV)[0][1]
            
        #Check if today's value exist in DB for stat calculation
        checkQuery = """select * from MODEL_QA_STATS where RMS_ID = '%(rms_id)s' and DT = '%(date)s' and CODE = '4'"""
        checkQry = checkQuery % {'date': date, 'rms_id': self.riskModel.rms_id}
        check_df = sql.read_sql(checkQry, self.modelDB.dbConnection)
        if check_df.shape[0] == 0:
            cur = self.modelDB.dbCursor
            insertQuery = """INSERT INTO MODEL_QA_STATS VALUES('%(date)s', '%(rms_id)s', sysdate, %(SpRskCorrel)s, '4')"""
            cur.execute(insertQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'SpRskCorrel': todaySpRskCorrel})
            logging.info("Inserting 1 value - Specific Risk CorrelCoef")
        elif self.updateFlag:
            cur = self.modelDB.dbCursor
            updateQuery = """UPDATE MODEL_QA_STATS SET VALUE = %(SpRskCorrel)s, REV_DT = sysdate where DT = '%(date)s' and RMS_ID = '%(rms_id)s' and code = '4'"""
            cur.execute(updateQuery % {'date': date, 'rms_id': self.riskModel.rms_id, 'SpRskCorrel': todaySpRskCorrel})
            logging.info("Updating 1 value - Specific Risk CorrelCoef")
        else:
            logging.debug("No insertion/update")

        if check_df.shape[0] == 0 or self.updateFlag:
            if self.testOnly:
                self.modelDB.dbConnection.rollback()
                logging.info("Rolling-back Changes")
            else:
                self.modelDB.dbConnection.commit()
                logging.info("Committing Changes")

        #stat for as-of-date obs
        stat = self.getQAStat(self.riskModel.rms_id, date, '4')
        bound = self.getBound('4', stat[0], stat[1])

        alert = 0
        if todaySpRskCorrel > bound[0]:
            logging.info("Specific Risk Correlation Check Result: Below 1+ (SAFE)")
        elif bound[1] < todaySpRskCorrel <= bound[0]:
            logging.info("Specific Risk Correlation Check Result: Within 1+(inc) & 3+ (SAFE)")
        elif bound[2] < todaySpRskCorrel <= bound[1]:
            logging.info("Specific Risk Correlation Check Result: Within 3+(inc) & 9+ (SAFE)")
        else:
            logging.error("Specific Risk Correlation Check Result: Above 9+(inc) (ALERT)")
            logging.error("(Threshold: %.5f (9+))", bound[2])
            alert += 1

        if alert == 0:
            return alert
        elif alert == 1:
            statdf = self.getQAStat(self.riskModel.rms_id, date, '4', True)
            conditions = [statdf['VALUE'] > statdf['1- SIGMA'],
                          (statdf['3- SIGMA'] < statdf['VALUE']) & (statdf['VALUE'] <= statdf['1- SIGMA']),
                          (statdf['9- SIGMA'] < statdf['VALUE']) & (statdf['VALUE'] <= statdf['3- SIGMA']),
                          statdf['VALUE'] <= statdf['9- SIGMA']]
            analysis = ["Below 1+", "Within 1+(inc) & 3+", "Within 3+(inc) & 9+", "Above 9+(inc)"]
            statdf['ANALYSIS'] = np.select(conditions, analysis, default = None)
            if self.chart is not None:
                self.outputChart(date, '8', self.riskModel.mnemonic, statdf)
            countdf = statdf.iloc[:-1, :].groupby(['ANALYSIS']).size().to_frame(name='COUNTS').reindex(analysis)
            print("Past %d-day Count Analysis:\n"%countdf['COUNTS'].sum(), countdf.to_string())
            return alert
        else:
            logging.error("alert=%d, please check" %alert)
            sys.exit(1)
                
    def generateESTURankReport(self, date):
        if 'USSC4' in self.riskModel.mnemonic:
            logging.info("Skipping ESTU Ranking Jump Check: US Small-Cap")
            return 0
            
        dateList = self.modelDB.getDateRange(self.riskModel.rmg, date, date, excludeWeekend=True, tradingDaysBack=2)

        #Get 20-day Average MCap for today & prev's
        mcapDates = self.modelDB.getDates(self.riskModel.rmg, dateList[1], 19)
        rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, dateList[1])
        estu = self.modelDB.getRiskModelInstanceESTU(rmi)
        mcap = self.modelDB.getAverageMarketCaps(mcapDates, estu, self.riskModel.numeraire.currency_id, self.marketDB)
        assetcapDict = dict(zip(estu, mcap.filled(0.0)))

        prevmcapDates = self.modelDB.getDates(self.riskModel.rmg, dateList[0], 19)
        prevrmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, dateList[0])
        prevestu = self.modelDB.getRiskModelInstanceESTU(prevrmi)
        prevmcap = self.modelDB.getAverageMarketCaps(prevmcapDates, prevestu, self.riskModel.numeraire.currency_id, self.marketDB)
        prevassetcapDict = dict(zip(prevestu, prevmcap.filled(0.0)))

        #Get ranking for top 40
        currTop = np.array([assetcapDict.get(sid) for sid in estu])
        currRank = np.argsort(-currTop)[:40]
        currRankIdx = dict([(estu[j],i) for (i,j) in enumerate(currRank)])

        prevTop = np.array([prevassetcapDict.get(sid) for sid in prevestu])
        prevRank = np.argsort(-prevTop)[:40]
        prevRankIdx = dict([(prevestu[j],i) for (i,j) in enumerate(prevRank)])

        rankDiff = {}
        totalalert = 0

        for asset in currRankIdx:
            alert = 0
            if asset in prevRankIdx:
                rankDiff[asset] = prevRankIdx[asset] - currRankIdx[asset]
                if abs(rankDiff[asset]) > 5:
                    alert += 1
                    logging.info("%s: Ranking Change > 5"%asset)
            else:
                rankDiff[asset] = "--"
                if currRankIdx.get(asset) <= 34:
                    alert += 1
                    logging.info("%s: Newly join Top 35"%asset)
            totalalert += alert

        if totalalert > 0:
            logging.error("Big jumps in ESTU Ranking (ALERT)")
            resultlist = []
            for asset in currRankIdx:
                resultlist.append((asset, assetcapDict.get(asset)/1e6, currRankIdx[asset]+1, rankDiff[asset]))
            resultDF = pd.DataFrame(resultlist, columns=['SubIssue', '20-day Ave.MCap (M)', 'Rank', 'Rank Diff']).sort_values(by='Rank').reset_index(drop=True)
            print(resultDF.to_string())
        elif totalalert == 0:
            logging.info("No Big jumps in ESTU Ranking (SAFE)")
        else:
            logging.error("totalalert=%d, please check" %totalalert)
            sys.exit(1)

        return totalalert

    def generateMissingAssetReport(self, date):
        #Temp tweak for NALM
        if 'NALM4' in self.riskModel.mnemonic:
            logging.info("Temp skip section 10 Missing Asset check for NALM4 models")
            return 0
        
        rmi = self.modelDB.getRiskModelInstance(self.riskModel.rms_id, date)
        univ = self.modelDB.getRiskModelInstanceUniverse(rmi, False)
        universeChecker = UniverseChecker(self.riskModel, self.modelDB, self.marketDB, date, univ, list(), list())

        missingBM, terminated = universeChecker.getMissingBenchmark(showAllTerminated=True)
        missingBM = list(missingBM)
        missingCM = list(universeChecker.getMissingComposite())

        alert = 0
        #Terminated Benchmark Asset. Re-transfer Affected Index if Needed
        if len(terminated) > 0:
            cur1 = self.marketDB.dbCursor
            query = """select axioma_id, name, value, index_id from index_constituent_easy where axioma_id in (%(axid)s) and dt = '%(date)s'"""
            cur1.execute(query % {'date': date, 'axid': ','.join("'%s'" %argid for argid in terminated)})
            
            TerminatedBmk = cur1.fetchall()
            BmkAssetWgtDict = dict()
            for item in TerminatedBmk:
                if "OPEN" not in item[1]:
                    if item[1] not in BmkAssetWgtDict.keys():
                        BmkAssetWgtDict[item[1]] = dict([(item[0],round(item[2]*100,5))])
                    else:
                        BmkAssetWgtDict[item[1]].update(dict([(item[0],round(item[2]*100,5))]))

            #Number of affected index
            if len(list(BmkAssetWgtDict.values())) > 0:
                alert += len(list(BmkAssetWgtDict.values()))

                logging.error("Terminated Benchmark Asset. Re-transfer Affected Index if Needed (%) / check Consolidated Report (ALERT)")
                TerminatedBmkDF = pd.DataFrame.from_dict(BmkAssetWgtDict, orient='index')
                print(TerminatedBmkDF.to_string())

        #Missing Composite assets
        if len(missingCM) > 0:
            MarketDB_ID, imp = self.modelDB.getMarketDB_IDs(self.marketDB, missingCM)

            cur2 = self.marketDB.dbCursor
            query = """select axioma_id from modeldb_global.ISSUE_MAP im, classification_active_int cai where
            im.MARKETDB_ID = cai.axioma_id and
            cai.axioma_id in (%(axid)s) and cai.from_dt <= '%(date)s' and cai.thru_dt > '%(date)s' and
            cai.classification_type_name='Axioma Asset Type' and cai.name like '%(ETF)s'"""
            cur2.execute(query % {'date': date, 'axid': ','.join("'%s'" %argid for argid in MarketDB_ID), 'ETF': 'ETF'+'%'})

            resultMktList = cur2.fetchall()
            diff = len(missingCM) - len(resultMktList)
            if diff == 0:
                logging.info("Asset type is ETF and hence is not in the model")
            else:
                alert += diff
                resultMdlList = [universeChecker.imp[i[0]] for i in resultMktList]
                logging.error("Missing Composite Assets: %s / check Consolidated Report (ALERT)", (set(missingCM) - set(resultMdlList)))

        #Missing benchmark assets
        totalalert = alert + len(missingBM)
        if len(missingBM) > 0:
            logging.error("Missing Benchmark Assets: %s / check Consolidated Report (ALERT)", (','.join("%s" %bm for bm in missingBM)))
            
        if totalalert == 0:
            logging.info("No Missing Benchmark/Composite Assets: (SAFE)")
        elif totalalert < 0:
            logging.error("totalalert=%d, please check" %totalalert)
            sys.exit(1)
            
        return totalalert
            
if __name__ == '__main__':
    usage = "usage: %prog [options] <startdate or datelist> [<end-date>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)

    cmdlineParser.add_option("--chart", action="store", default=None, dest="chart", help="Generate chart(s) to directory")
    cmdlineParser.add_option("--all", action="store_true", default=False, dest="runAll", help="Run for all RMS_ID")
    cmdlineParser.add_option("--update", action="store_true", default=False, dest="updateFlag", help="Update QA stats in ModelDB")
    cmdlineParser.add_option("-n", action="store_true", default=False, dest="testOnly", help="don't change the database")

    (options, args) = cmdlineParser.parse_args()

    if len(args)< 1:
        cmdlineParser.error("Incorrect number of arguments")

    if options.runAll and (options.modelName or (options.modelID and options.modelRev)):
        cmdlineParser.error("Please specify either one risk model or opt for all")

    logging.info('testOnly option (-n) is %s', options.testOnly)
    
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)

    riskModelDict = dict()

    #Get riskmodel info based on inputs
    if not options.runAll:
        try:
            riskModelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
            logging.info("Accessing RiskModelClass = %s",riskModelClass)
            riskModel = riskModelClass(modelDB, marketDB)
            riskModelDict[riskModel.rms_id] = riskModel
        except Exception:
                logging.error("Exception Caught in processing a single model", exc_info=True)
                sys.exit(1)
    else:
        try:
            #Get all active risk model series and loop for each
            allModelSeries = UniCheck.getAllDistributedSeries(modelDB)
            for key, values in allModelSeries.items():
                options.modelID = values[0]
                options.modelRev = values[1]
                riskModelClass = Utilities.processModelAndDefaultCommandLine(options, cmdlineParser)
                logging.info("Accessing RiskModelClass = %s",riskModelClass)
                riskModel = riskModelClass(modelDB, marketDB)
                riskModelDict[key] = riskModel
                if key != riskModel.rms_id:
                    logging.error("Skipping mismatched rms_id: Given rms_id = %s, Extracted rms_id = %s", key, riskModel.rms_id)
                    continue
        except Exception:
            logging.error("Exception Caught in processing all active rms_id", exc_info=True)
            sys.exit(1)
            
    if len(riskModelDict)==0:
        logging.error("No rms_id is found, please check!")
        sys.exit(1)
                
    logging.info("Initiating RiskModelQA.py")

    resultdf = pd.DataFrame()
    for rms, riskModel in riskModelDict.items():
        rmqa = RMQA(modelDB, marketDB, args, rms, riskModel, options.testOnly, options.updateFlag, options.chart)
        #Dates to run QA
        modelfullDates = sorted(list(rmqa.getdateset(None)))
        logging.info("RMS_ID=%s: dates under checking:%s", rms, sorted([dt.strftime('%Y-%m-%d') for dt in modelfullDates]))

        resultdf = pd.DataFrame()
        for dt in modelfullDates:
            errorNum = 0
            print("*********************************************************************************************************************")
            logging.info("#####1: Processing ESTU Correlation Check")
            errorNum += rmqa.generateUniCorrReport(dt, "ESTU")
            logging.info("#####1: Finished Processing ESTU Correlation Check")
                
            print("*********************************************************************************************************************")
            logging.info("#####2: Processing ESTU Leaver Count Check")
            errorNum += rmqa.generateLeaverReport(dt, "ESTU")
            logging.info("#####2: Finished Processing ESTU Leaver Count Check")

            print("*********************************************************************************************************************")
            logging.info("#####3: Processing Model Universe Correlation Check")
            errorNum += rmqa.generateUniCorrReport(dt, "Model")
            logging.info("#####3: Finished Processing Model Universe Correlation Check")

            print("*********************************************************************************************************************")
            logging.info("#####4: Processing Model Universe Leaver Count Check")
            errorNum += rmqa.generateLeaverReport(dt, "Model")
            logging.info("#####4: Finished Processing Model Universe Leaver Count Check")

            print("*********************************************************************************************************************")
            logging.info("#####5: Processing Factor Return Check")
            errorNum += rmqa.generateFactorReturnReport(dt)
            logging.info("#####5: Finished Processing Factor Return Check")

            print("*********************************************************************************************************************")
            logging.info("#####6: Processing Factor Volatility Test Check")
            errorNum += rmqa.generateFactorVolTestReport(dt)
            logging.info("#####6: Finished Processing Factor Volatility Test Check")

            print("*********************************************************************************************************************")
            logging.info("#####7: Processing Factor Exposure Correlation Check")
            errorNum += rmqa.generateExposureCorrReport(dt)
            logging.info("#####7: Finished Processing Factor Exposure Correlation Check")

            print("*********************************************************************************************************************")
            logging.info("#####8: Processing Specific Risk Correlation Check")
            errorNum += rmqa.generateSpecificRiskCorrReport(dt)
            logging.info("#####8: Finished Processing Specific Risk Correlation Check")

            print("*********************************************************************************************************************")
            logging.info("#####9: Processing ESTU Ranking Jump Check")
            errorNum += rmqa.generateESTURankReport(dt)
            logging.info("#####9: Finished Processing ESTU Ranking Jump Check")

            print("*********************************************************************************************************************")
            logging.info("#####10: Processing Missing Benchmark/Composite Assets Check")
            errorNum += rmqa.generateMissingAssetReport(dt)
            logging.info("#####10: Finished Processing Missing Benchmark/Composite Assets Check")

            if options.chart is not None:
                print("*********************************************************************************************************************")
                rmqa.outputZip(dt)

    if errorNum > 0:
        sys.exit(1)
    else:
        sys.exit(0)

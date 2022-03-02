
import os
import optparse
import logging
import configparser
import sys
import datetime
import numpy as np
from marketdb import MarketID
from riskmodels import Connections
from riskmodels import VisualTool
from riskmodels import Utilities

class FundamentalDataQA:
    def __init__(self, connections):
        self.connections=connections
        self.marketDB=connections.marketDB

    reportContents = list()

    def populateReportContent(self, reportSection, reportName, colheader=None, content=None,
                              description=None, reportType=None,rowheader=None,  position=None):
        data = Utilities.Struct()
        data.reportSection = reportSection
        data.reportName = reportName
        if description is not None:
            data.description = description
        if reportType is not None:
            data.reportType = reportType
        if content is not None:
            data.content = content
            data.header = colheader
        if position is not None:
            data.position = position
        self.reportContents.append(data)

    def getCodeNames(self,item_code):
        query= """select id, name, description from meta_codes where code_type like '%fund_currency%' and id=:item_code """
        marketDB.dbCursor.execute(query,{'item_code':item_code})
        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))
        result = result[0][2].replace(", in million currency","")
        return result

    def prevTradingDay(self, date_arg,ctry_arg):
        query= """select max(dt) from meta_trading_calendar_active where iso_ctry_code=:ctry_arg  and dt<:date_arg"""
        marketDB.dbCursor.execute(query,{'ctry_arg':ctry_arg,'date_arg':date_arg})
        result=marketDB.dbCursor.fetchall()
        return result[0][0].date()

    def getAssetInfo(self,date_arg,axids):
        """ Get Asset Info that will be used as a reference in the report
        e.g. Name, Country, Sedol, GVKEY, Mcap)"""

        prev_dt = self.prevTradingDay(date_arg,'US')

        mids= []
        for i in range(len(axids)):
            mids.append(MarketID.MarketID(string=axids[i]))

        currName = marketDB.getAssetNames(date_arg, mids)
        currSEDOL = marketDB.getSEDOLs(date_arg, mids)
        currGVKEY = marketDB.getGVKEY(date_arg, mids)
        currTradingCountry=  marketDB.getTradingCountry(date_arg, mids)
        currTso = marketDB.getSharesOutstanding([prev_dt], mids)
        currUcpLoc = marketDB.getPrices([prev_dt],mids,'USD')

        for key in currTradingCountry.keys():
       #     if currTradingCountry[key] is not None:
            currTradingCountry[key] = currTradingCountry[key].encode('ascii','replace')
            #else:
            #     currTradingCountry[key] = 'NA'
                        
        """Calculate Mcap"""
        axout_p_mktid= currUcpLoc.assets
        data_pl= currUcpLoc.data
        data_p=[]
        
        for i in range(len(data_pl)):
            data_p.append(data_pl[i][0])
            
        prices = dict(zip(axout_p_mktid,data_p))
        axout_s_mktid=currTso.assets
        data_sl= currTso.data
        data_s=[]
        
        for i in range(len(data_sl)):
            data_s.append(data_sl[i][0])
        
        shares  = dict(zip(axout_s_mktid,data_s))
        currMcap = {i: prices[i]*shares[i] for i in prices}

        for axid in currMcap:
            currMcap[axid] /= 1000000
            currMcap[axid] = int(currMcap[axid])
            
        currInfo={}
        for i in range(len(axids)):
            currInfo[currName[i][0]] = [currName[i][1],currTradingCountry[currName[i][0]],currSEDOL[i][1],currGVKEY[currName[i][0]],currMcap[currName[i][0]]]
            

        for key in list(currInfo.keys()):
            currInfo[key.getIDString()] = currInfo.pop(key)
        
        return currInfo

    def getDailyChanges(self, start_date,change_value,item_code):
        """Returns daily change for a given measyre updated on a given day"""

        end_date= start_date + datetime.timedelta(days=1)        

        query='''select * from 
        (select xt.axioma_id, xt.value, to_char(xt.dt), to_char(xt.eff_dt), xt1.value prev_value, to_char(xt1.dt) prev_dt,  crc.code  , to_char(xt1.eff_dt) PREV_VALUE_EFF_DT, ROUND(abs(xt.value/xt1.value-1),4) ABS_CHANGE
        from 
        (select xp.* from asset_dim_fund_currency xp,
        (select axioma_id, item_code, max(dt) MAX_DT, max(eff_dt) MAX_EFF_DT, max(REV_DT) MAX_REV_DT from asset_dim_fund_currency where  rev_dt between  :start_date and :end_date group by axioma_id, item_code) xpm
         where   xp.axioma_id=xpm.axioma_id and xp.dt=xpm.max_dt  and xp.eff_dt=xpm.max_eff_dt  and xp.rev_dt=xpm.max_rev_dt 
        and xp.item_code=xpm.item_code and xp.eff_del_flag='N' and xp.rev_del_flag='N' 
        ) xt, 
        (
        select xp.* from asset_dim_fund_currency xp,
        (
        select stl.axioma_id,  stl.item_code, max(stl.dt) MAX_DT, max(stl.eff_dt) MAX_EFF_DT, max(stl.rev_dt) MAX_REV_DT from asset_dim_fund_currency stl, 
        (select AXIOMA_ID, item_code,  max(rev_dt) MAX_REV_DT from asset_dim_fund_currency where  rev_dt <:end_date  group by axioma_id, item_code) max_rev_dts
        where  stl.rev_dt<>max_rev_dts.max_rev_dt and stl.axioma_id=max_rev_dts.axioma_id and stl.item_code=max_rev_dts.item_code and stl.eff_del_flag='N' and stl.rev_del_flag='N'
        group by stl.axioma_id, stl.item_code
        ) xpm
         where   xp.axioma_id=xpm.axioma_id and xp.dt=xpm.max_dt and xp.eff_dt=xpm.max_eff_dt and xp.rev_dt=xpm.max_rev_dt
        and xp.item_code=xpm.item_code 
        ) xt1,
        asset_ref ar, currency_ref crc
        where   xt.item_code=:item_code and xt.eff_dt>=xt1.eff_dt and ar.thru_dt>:end_date and crc.id=xt.currency_id
        and xt.axioma_id=xt1.axioma_id and ar.axioma_id=xt.axioma_id and ar.trading_country='US' 
        and xt.item_code=xt1.item_code and xt.currency_id=xt1.currency_id)ch
        where ch.abs_change>:change_value and ch.prev_value>0 order by ch.abs_change desc'''
        
        marketDB.dbCursor.execute(query,{'start_date':start_date,'end_date':end_date,'change_value':change_value,'item_code':item_code})

        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))
        if len(result) > 0:
            """Format Dates"""
            for i in range(len(result)):
                 result[i][2] = result[i][2][:10]
                 result[i][3] = result[i][3][:10]
                 result[i][5] = result[i][5][:10]
                 result[i][7] = result[i][7][:10]
            axout=[] 
            for i in range(len(result)):
                axout.append(result[i][0])
            """Convert Result to Dictionary"""
            resultd= {t[0]:t[1:] for t in result}
            m = self.getAssetInfo(start_date,axout)
            """ Merge Dictionaries"""
            fdict = dict((k, [v] + ([resultd[k]] if k in resultd else [])) for (k, v) in m.items())
            for key in fdict.keys():
                fdict[key] = fdict[key][0] + fdict[key][1]
            result = []
            for key, value in fdict.items():
                result.append([key] + value)
            result.sort(key=lambda x: x[13],reverse=True)
            resultm = []
            for i in range(len(result)):
                if result[i][5] > 1000:
                    resultm.append(result[i])
                
            result = resultm
            reportSection = 'Changes from Daily Updates (Cstatdb)'
            description = self.getCodeNames(item_code) + ' >' + str(change_value) + ' Change' 
            header =['AxiomaID', 'Name', 'Ctry', 'SEDOL', 'GVKEY', 'MCap','New Value','Dt','Eff Dt','Prev Value', 'Prev Dt', 'Currency', 'Prev Eff Dt','Abs Change']
            reportName = self.getCodeNames(item_code) + " Change"

            if len(result) > 0:
                content = np.asmatrix(result)
                self.populateReportContent(reportSection, reportName, header, content, description)
            else:
                self.populateReportContent(reportSection, reportName, None, None, description)
        
            
    def xpsCstatCount(self,date_arg, cp_code,xp_code):
        "Number of records updated for a given date for a given measure between Cstatdb and Xpressfeed"""

        date_arg = str(date_arg)

        query = '''select cp.dt_, cp.ct CT_CP, xp.ct XP_CT from
        (select a.dt_,ct from (select a.dt_, case when ct is null then 0 else ct end ct from 
        ( select distinct to_char(dt,'YYYY-MM-DD')dt_ from meta_trading_calendar_active where DT >=to_date(:date_arg,'YYYY-MM-DD')-10 and dt <= to_date(:date_arg,'YYYY-MM-DD')
         and to_char(dt,'D')not in  ('1','7')order by to_char(dt,'YYYY-MM-DD') desc) a left outer join (select count(*) ct, to_char(rev_dt,'YYYY-MM-DD') rev_dt_ 
         from  asset_dim_fund_currency where rev_dt >= to_date(:date_arg,'YYYY-MM-DD')-10 and rev_dt<to_date(:date_arg,'YYYY-MM-DD')+1 and rev_del_flag ='N'  and item_code=:cp_code
        and axioma_id in (select axioma_id from asset_ref where trading_country='US')
        and axioma_id not in (select axioma_id from asset_ref ar where add_dt > to_date(:date_arg,'YYYY-MM-DD')-10 ) group by to_char(rev_dt,'YYYY-MM-DD')) b 
        on a.dt_=b.rev_dt_ order by a.dt_ desc) a where rownum < 6) cp ,
        (select a.dt_,ct from (select a.dt_, case when ct is null then 0 else ct end ct from 
        ( select distinct to_char(dt,'YYYY-MM-DD')dt_ from meta_trading_calendar_active where DT >=to_date(:date_arg,'YYYY-MM-DD')-10 and dt <= to_date(:date_arg,'YYYY-MM-DD')
        and to_char(dt,'D')not in  ('1','7')order by to_char(dt,'YYYY-MM-DD') desc) a left outer join (select count(*) ct, to_char(rev_dt,'YYYY-MM-DD') rev_dt_ 
        from  asset_dim_fundamental_data where rev_dt >= to_date(:date_arg,'YYYY-MM-DD')-10 and rev_dt<to_date(:date_arg,'YYYY-MM-DD')+1 and rev_del_flag ='N'  and item_code_id=:xp_code
        and axioma_id in (select axioma_id from asset_ref where trading_country='US')
        and axioma_id not in (select axioma_id from asset_ref ar where add_dt > to_date(:date_arg,'YYYY-MM-DD')-10 and trading_country<>'US' ) group by to_char(rev_dt,'YYYY-MM-DD')) b 
        on a.dt_=b.rev_dt_ order by a.dt_ desc) a where rownum < 6) xp
        where xp.dt_=cp.dt_'''

        marketDB.dbCursor.execute(query,{'date_arg':date_arg,'cp_code':cp_code,'xp_code':xp_code})
        result=marketDB.dbCursor.fetchall()

        reportSection = 'Cstatdb vs Xpressfeed'
        description = 'Count of Records Updated for ' + self.getCodeNames(cp_code)
        header = ['Date','Cstatdb','Xpressfeed']
        reportName= "Count of Records Cs and Cp " + self.getCodeNames(cp_code)
        reportType='BarChart'
        
        if len(result) > 0:
            content = np.asmatrix(result)
            self.populateReportContent(reportSection, reportName, header, content, description, reportType)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)

    def xpsfeedMissingInCstatdb(self,date,ctry_arg):
        """Checks for recently added assets that have data available in Cstatdb but not in Xpressfeed """

        date_arg = str(date)
        
        query = ''' select cp.axioma_id, to_char(min(cp.dt)) MIN_CSTATDB_DT , count(distinct(cp.item_code)) COUNT_OF_MEASURES_CSTATDB, to_char(min(rf.add_dt)) ASSET_ADD_DT, to_char(min(rf.from_dt)) ASSET_FROM_DT from  asset_dim_fund_currency_active cp, asset_ref rf where cp.axioma_id in (
        (select axioma_id from asset_ref  where  add_dt>(to_date(:date_arg,'YYYY-MM-DD')-15) and add_dt<to_date(:date_arg,'YYYY-MM-DD') and axioma_id not like 'F%' and axioma_id not like 'O%' and trading_country=:ctry_arg) 
        )  and cp.axioma_id not in (select distinct axioma_id from asset_dim_fundamental_data_act)
        and cp.axioma_id=rf.axioma_id
        and cp.item_code not in (6,7)
        group by cp.axioma_id
        '''        
        marketDB.dbCursor.execute(query,{'date_arg':date_arg,'ctry_arg':ctry_arg})
        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))

        """Add asset information and format"""
        axout=[]
        """Format Dates"""
        for i in range(len(result)):
            result[i][1] = result[i][1][:10]
            result[i][3] = result[i][3][:10]
            result[i][4] = result[i][4][:10]
            axout.append(result[i][0])

        """Convert Result to Dictionary"""
        resultd= {t[0]:t[1:] for t in result}
        m = self.getAssetInfo(date,axout)
        """ Merge Dictionaries"""
        fdict = dict((k, [v] + ([resultd[k]] if k in resultd else [])) for (k, v) in m.items())
        for key in fdict.keys():
            fdict[key] = fdict[key][0] + fdict[key][1]
        result = []
        for key, value in fdict.items():
            result.append([key] + value)
        result.sort(key=lambda x: x[5],reverse=True)

        reportSection = 'Cstatdb vs Xpressfeed'
        description = 'Securities with Fund. Data in Cstatdb but not in Xpsfeed'
        header =['AxiomaID', 'Name', 'Ctry', 'SEDOL', 'GVKEY', 'MCap','Min Cstatdb Dt', 'Count of Measures Cstatdb','Asset Add_Dt', 'Asset From_Dt']
        reportName= 'Securities with Fund. Data in Cstatdb but not in Xpsfeed'

        if len(result) > 0:
            content = np.asmatrix(result)
            self.populateReportContent(reportSection, reportName, header, content, description)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)


    def xpsGvkeyValueChange(self,start_date,cp_code,xp_code):
        """ From the updates received from Xpressfeed, it compares the gvkey of the most recent report vs the gvkey of the most recent previous one"""
        end_date= start_date + datetime.timedelta(days=1)
        query = '''select * from 
        (select xt.axioma_id, substr(xt.ref,length(xt.ref)-6,length(xt.ref)) new_gvkey,  substr(xt1.ref,length(xt1.ref)-6,length(xt1.ref)) prev_gvkey
        from 
        (select xp.* from asset_dim_fundamental_data xp,
        (select axioma_id, item_code_id, max(dt) MAX_DT, max(eff_dt) MAX_EFF_DT, max(REV_DT) MAX_REV_DT from asset_dim_fundamental_data where  rev_dt between  :start_date and :end_date group by axioma_id, item_code_id) xpm
         where   xp.axioma_id=xpm.axioma_id and xp.dt=xpm.max_dt  and xp.eff_dt=xpm.max_eff_dt  and xp.rev_dt=xpm.max_rev_dt 
        and xp.item_code_id=xpm.item_code_id and xp.eff_del_flag='N' and xp.rev_del_flag='N' 
        ) xt, 
        (
        select xp.* from asset_dim_fundamental_data xp,
        (
        select stl.axioma_id,  stl.item_code_id, max(stl.dt) MAX_DT, max(stl.eff_dt) MAX_EFF_DT, max(stl.rev_dt) MAX_REV_DT from asset_dim_fundamental_data stl, 
        (select AXIOMA_ID, ITEM_CODE_ID,  max(rev_dt) MAX_REV_DT from asset_dim_fundamental_data where  rev_dt <:end_date  group by axioma_id, item_code_id) max_rev_dts
        where  stl.rev_dt<>max_rev_dts.max_rev_dt and stl.axioma_id=max_rev_dts.axioma_id and stl.item_code_id=max_rev_dts.item_code_id and stl.eff_del_flag='N' and stl.rev_del_flag='N'
        group by stl.axioma_id, stl.item_code_id
        ) xpm
         where   xp.axioma_id=xpm.axioma_id and xp.dt=xpm.max_dt and xp.eff_dt=xpm.max_eff_dt and xp.rev_dt=xpm.max_rev_dt
        and xp.item_code_id=xpm.item_code_id 
        ) xt1
        where   xt.item_code_id=:xp_code and xt.eff_dt>=xt1.eff_dt
        and xt.axioma_id=xt1.axioma_id and xt.item_code_id=xt1.item_code_id and xt.currency_id=xt1.currency_id)ch
        where ch.new_gvkey<>ch.prev_gvkey
        '''        
        marketDB.dbCursor.execute(query,{'start_date':start_date,'end_date':end_date,'xp_code':xp_code})
        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))

        """Add asset information and format"""
        axout=[]
        
  
        """Convert Result to Dictionary"""
        resultd= {t[0]:t[1:] for t in result}
        m = self.getAssetInfo(start_date,axout)
        """ Merge Dictionaries"""
        fdict = dict((k, [v] + ([resultd[k]] if k in resultd else [])) for (k, v) in m.items())
        for key in fdict.keys():
            fdict[key] = fdict[key][0] + fdict[key][1]
        result = []
        for key, value in fdict.items():
            result.append([key] + value)
        result.sort(key=lambda x: x[5],reverse=True)
        
        reportSection = 'Cstatdb vs Xpressfeed'
        description = 'Companies with Gvkey Change based on fundamental data  [Xpressfeed] (' + self.getCodeNames(cp_code) + ')'
        header =['AxiomaID', 'Name', 'Ctry', 'SEDOL', 'GVKEY', 'MCap','Dt','New Gvkey','Old Gvkey']
        reportName= 'Gvkey Change Xpressfeed Fundamental Data ('  + self.getCodeNames(cp_code) + ')'
        
        if len(result) > 0:
            content = np.asmatrix(result)
            self.populateReportContent(reportSection, reportName, header, content, description)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)

        
            
    def xpsCstatValueCrossCheck(self,date_arg,cp_code,xp_code):
        """ From the updates received, It checks if there are different values between cstatdb and xpressfeed for a given measure"""
        start_date = date_arg - datetime.timedelta(days=1)
        end_date= date_arg + datetime.timedelta(days=1)
        
        query = '''select * from (select cp.axioma_id,  to_char(cp.dt),cp.value cp_value,  to_char(cp.eff_dt),crc.code cp_curr, xf.value xp_value, to_char(xf.eff_dt), crx.code  xp_curr,  ROUND(abs(xf.value/cp.value-1),6) ABS_DIFF from asset_dim_fund_currency cp,
        asset_dim_fundamental_data xf,
        (select axioma_id, item_code_id, dt, max(eff_dt) max_eff_dt from asset_dim_fundamental_data group by axioma_id, item_code_id, dt) md,
        (select axioma_id, item_code, dt, max(eff_dt) max_eff_dt from asset_dim_fund_currency group by axioma_id, item_code, dt) mdc,
        currency_ref crc, currency_ref crx , asset_ref ar
        where
        cp.axioma_id=xf.axioma_id and xf.item_code_id=:xp_code and cp.item_code=:cp_code
        and ar.axioma_id=cp.axioma_id and ar.trading_country='US' and ar.thru_dt>:end_date
        and xf.axioma_id=md.axioma_id and xf.item_code_id=md.item_code_id and xf.dt=md.dt and xf.eff_dt=md.max_eff_dt
        and cp.axioma_id=mdc.axioma_id and cp.item_code=mdc.item_code and cp.dt=mdc.dt and cp.eff_dt=mdc.max_eff_dt
        and cp.dt=xf.dt and crc.id=cp.currency_id and crx.id=xf.currency_id
        and cp.rev_dt < :end_date
        and xf.rev_dt between :start_date and :end_date
        and cp.eff_del_flag='N' and xf.eff_del_flag='N'
        and cp.value<>xf.value)
        where abs_diff>.001 
        '''        
        marketDB.dbCursor.execute(query,{'start_date':start_date,'end_date':end_date,'cp_code':cp_code,'xp_code':xp_code})
        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))

        """Add asset information and format"""
        axout=[]
        """Format Dates"""
        for i in range(len(result)):
             result[i][1] = result[i][1][:10]
             result[i][3] = result[i][3][:10]
             result[i][6] = result[i][6][:10]
             axout.append(result[i][0])

        """Convert Result to Dictionary"""
        resultd= {t[0]:t[1:] for t in result}
        m = self.getAssetInfo(start_date,axout)
        """ Merge Dictionaries"""
        fdict = dict((k, [v] + ([resultd[k]] if k in resultd else [])) for (k, v) in m.items())
        for key in fdict.keys():
            fdict[key] = fdict[key][0] + fdict[key][1]
        result = []
        for key, value in fdict.items():
            result.append([key] + value)
        result.sort(key=lambda x: x[5],reverse=True)
       
        reportSection = 'Cstatdb vs Xpressfeed'
        description = 'Companies with Value Mismatch between Cs and Xp (' + self.getCodeNames(cp_code) + ')'
        header =['AxiomaID', 'Name', 'Ctry', 'SEDOL', 'GVKEY', 'MCap','Dt','Cstat Value','Cstat Eff Dt','Cstat Currency','Xpsfeed Value','Xpsfeed Eff Dt','Xpsfeed Currency', 'Abs Change']
        reportName= 'Value Mismatch Cs and Xp ('  + self.getCodeNames(cp_code) + ')'
        
        if len(result) > 0:
            content = np.asmatrix(result)
            self.populateReportContent(reportSection, reportName, header, content, description)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)
       


        
    def cstatVsFactsetValue(self,date_arg,cstat_code,factset_code,change_value):
        """Displays assets that had an update in Cstatdb a specified date and compares them against FactSet existing data
        for a specified measure"""

        start_date = self.prevTradingDay(date_arg,'US')
        
        end_date= date_arg + datetime.timedelta(days=1)

        query = """ select * from (select fc.axioma_id, substr(fc.ref,-8) FSYM_ID, fc.VALUE FACTSET_VALUE , crf.code FACTSET_CURRENCY, cp.VALUE  CSTATDB_VALUE, crc.code CSTATDB_CURRENCY, to_char(cp.DT), to_char(cp.EFF_DT),
        ROUND(abs(cp.value/fc.value-1),4) DIFF
         from ASSET_DIM_FS_FUND_ATTR2 fc , asset_dim_fund_currency cp, asset_ref ar, currency_ref crc, currency_ref crf,
        (select axioma_id, max(eff_dt) m_eff_dt, item_code, dt  from asset_dim_fund_currency_active group by axioma_id, dt, item_code) cpm,
        (select axioma_id, item_code_id, period_end_dt, max(change_dt) max_change_dt from ASSET_DIM_FS_FUND_ATTR2 group by axioma_id, item_code_id, period_end_dt) fcm
         where cp.axioma_id=cpm.axioma_id and cp.dt=cpm.dt and cp.item_code=cpm.item_code and cp.eff_dt=cpm.m_eff_dt and 
        fc.axioma_id=fcm.axioma_id and fc.period_end_dt=fcm.period_end_dt and fc.item_code_id=fcm.item_code_id and fc.change_dt=fcm.max_change_dt and
        fc.axioma_id=cp.axioma_id and ar.axioma_id=cp.axioma_id and ar.trading_country='US' and ar.thru_dt>:end_date
        and fc.period_end_dt=cp.dt and  fc.item_code_id=:factset_code and cp.item_code=:cstat_code  and cp.rev_del_flag=fc.rev_del_flag 
        and fc.currency_id=cp.currency_id and cp.value<>0 and fc.value<>0 and cp.eff_del_flag='N' and cp.rev_del_flag='N'
        and cp.rev_dt between :start_date and :end_date and crc.id=cp.currency_id and crf.id=fc.currency_id
        ) where diff>:change_value
        """

        marketDB.dbCursor.execute(query,{'start_date':start_date,'end_date':end_date,'cstat_code':cstat_code,'factset_code':factset_code,'change_value':change_value})

        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))

        """Add asset information and format"""

        axout=[]
        """Format Dates"""
        for i in range(len(result)):
             result[i][6] = result[i][6][:10]
             result[i][7] = result[i][7][:10]
             axout.append(result[i][0])
        """Convert Result to Dictionary"""
        resultd= {t[0]:t[1:] for t in result}
        m = self.getAssetInfo(start_date,axout)
        """ Merge Dictionaries"""
        fdict = dict((k, [v] + ([resultd[k]] if k in resultd else [])) for (k, v) in m.items())
        for key in fdict.keys():
            fdict[key] = fdict[key][0] + fdict[key][1]
        result = []
        for key, value in fdict.items():
            result.append([key] + value)
        result.sort(key=lambda x: x[5],reverse=True)
       
        reportSection = 'Cstatdb vs FactSet'
        description = 'Companies with Value Mismatch between Cstat and FactSet (' + self.getCodeNames(cstat_code) + ')'
        header=['AxiomaID', 'Name', 'Ctry', 'SEDOL', 'GVKEY','MCap','FSYM ID','FactSet Value','FactSet Currency','Cstat Value','Cstat Currency','Dt','Cstat Eff Dt','Diff']
        reportName= 'Value Mismatch Cs and FactSet('  + self.getCodeNames(cstat_code) + ')'
       
        if len(result) > 0:
            content = np.asmatrix(result)
            self.populateReportContent(reportSection, reportName, header, content, description)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)
        
    
    def getStaleData(self,date_arg,item_code):
        """From the last X years, look for assets that have records with a max dt less than a specified date
        Only look for assets that have frequent and timely records within the lookback period"""
        
        max_dt=datetime.date(2016,12,31)
        lookback_dt=date_arg - datetime.timedelta(days=1080)
        
        query= ''' select A.*, B.AVG_LAG, B.COUNT from 
        (select * from 
        (select f.axioma_id, to_char(max(f.dt)) max_dt from asset_dim_fund_currency_active f,
        asset_ref rf
        where f.axioma_id=rf.axioma_id and
        f.item_code=:item_code and rf.trading_country='US' and rf.thru_dt>:date_arg
        group by f.axioma_id) mf
        where mf.max_dt <:max_dt
        ) A, 
        (
        select * from(
        select axioma_id, round(avg(lag)) avg_lag, count(*) count from (
        select f.*, (f.eff_dt-f.dt) LAG from asset_dim_fund_currency_active f,
        (select axioma_id, dt, item_code, min(eff_dt) eff_dt from asset_dim_fund_currency_active group by axioma_id, dt, item_code) fm ,
        asset_ref rf
        where f.axioma_id=rf.axioma_id and rf.thru_dt>:date_arg  and  f.axioma_id=fm.axioma_id and f.dt=fm.dt and f.eff_dt=fm.eff_dt and f.item_code=fm.item_code and
        f.item_code=:item_code and  f.dt>=:lookback_dt and rf.trading_country='US')
        group by axioma_id)
        where count>4 and avg_lag<50
        ) B
        where A.axioma_id=B.axioma_id
        '''
        
        marketDB.dbCursor.execute(query,{'date_arg':date_arg,'max_dt':max_dt,'lookback_dt':lookback_dt,'item_code':item_code})

        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))

        """Add asset information and format"""

        axout=[]
        """Format Dates"""
        for i in range(len(result)):
             result[i][1] = result[i][1][:10]
             axout.append(result[i][0])
        """Convert Result to Dictionary"""
        resultd= {t[0]:t[1:] for t in result}
        m = self.getAssetInfo(date_arg,axout)
        """ Merge Dictionaries"""
        fdict = dict((k, [v] + ([resultd[k]] if k in resultd else [])) for (k, v) in m.items())
        for key in fdict.keys():
            fdict[key] = fdict[key][0] + fdict[key][1]
        result = []
        for key, value in fdict.items():
            result.append([key] + value)
        result.sort(key=lambda x: x[5],reverse=True)
        resultm = []
        for i in range(len(result)):
            if result[i][5] > 1000:
                resultm.append(result[i])
                
        result = resultm

        reportSection = 'Timeliness of Data (Cstatdb)'
        description = 'Assets with Potential Stale Data'

    
        header=['AxiomaID', 'Name', 'Ctry', 'SEDOL', 'GVKEY', 'MCap','Max Dt', 'Avg Lag','# Obs']
        reportName= "Assets with Potential Stale Data"
        
        if len(result) > 0:
            content = np.asmatrix(result)
            self.populateReportContent(reportSection, reportName, header, content, description)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)


    def getDualHistogram(self, data1, data2, binNo=100):
        """Given two data array and number of bins
           Return two histograms with the same bin sizes
        """
        minval = min(np.append(data1, data2))
        maxval = max(np.append(data1, data2))
        increment = (maxval - minval)/binNo
        bins = np.arange(minval-increment, maxval+increment, increment)
        hist1, bins1 = np.histogram(data1, bins=bins)
        hist2, bins2 =  np.histogram(data2, bins=bins)
        return (hist1, hist2, bins)

    def getAgeDistInput(self,date_arg,dt_lookback,eff_dt_lookback,item_code):
        date_arg = str(date_arg)

        query = """select * from
        (select  to_date(:date_arg,'YYYY-MM-DD')-:eff_dt_lookback - max(dt) diff from asset_dim_fund_currency_active where item_code=:item_code 
        and dt>=(to_date(:date_arg,'YYYY-MM-DD')-:dt_lookback) and eff_dt<=(to_date(:date_arg,'YYYY-MM-DD')-:eff_dt_lookback)
        and axioma_id in
        (select axioma_id from asset_ref where trading_country='US' and thru_dt>to_date(:date_arg,'YYYY-MM-DD')) group by axioma_id)"""

        marketDB.dbCursor.execute(query,{'date_arg':date_arg,'dt_lookback':dt_lookback,'eff_dt_lookback':eff_dt_lookback,'item_code':item_code})
        result=marketDB.dbCursor.fetchall()
        result = list(map(list, result))
        return result
    
    def getAgeDist(self,date):
        data1 = self.getAgeDistInput(date,365,0,26)
        data2 = self.getAgeDistInput(date,730,365,26)
        (hist1, hist2, bins) = fmr.getDualHistogram(data1, data2, binNo=100)

        reportSection = 'Timeliness of Data (Cstatdb)'
        description = 'Age of Data'
        reportName= 'Age of Data'
        reportType = 'AreaChart'
        content = np.c_[bins[1:], np.row_stack((hist1, hist2)).transpose()]
        header=['Bins','Today','A year ago']

        if len(content) > 0:
             self.populateReportContent(reportSection, reportName, header, content, description, reportType)
        else:
            self.populateReportContent(reportSection, reportName, None, None, description)
    
if __name__ == '__main__':
    usage = "usage: %prog configfile date [options]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("--report-file", action="store",
                             default=None, dest='reportFile',
                             help='report file name')
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args_) = cmdlineParser.parse_args()
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    filename=options.reportFile
   
    if len(args_)<2:
        logging.error("Incorrect number of arguments")
        sys.exit(1)

    configFile_ = open(args_[0])
    config_ = configparser.ConfigParser()
    config_.read_file (configFile_)
    configFile_.close()

    date = Utilities.parseISODate(args_[1])
    connections = Connections.createConnections(config_)
    marketDB=connections.marketDB
    fmr = FundamentalDataQA(connections)
    header = "Fundamental Data QA Report for US"

    
    if filename == None:
        filename = 'FundamentalDataQAReportUS.%04d%02d%02d' % (date.year, date.month, date.day)
    i = 0
    path = os.path.dirname(filename)
    if not os.path.exists(path) and path != '':
        os.makedirs(path)
    if os.path.isfile(filename + ".v" + str(i)):
        logging.info(filename  + ".v" + str(i) + " already exists")
        i = 1
        while os.path.isfile(filename + ".v" + str(i)):
             logging.info(filename + ".v" + str(i) + " already exists")
             i = i + 1

    htmlName = filename + ".v" + str(i)

    fmr.getDailyChanges(date,.2,26)
    fmr.getDailyChanges(date,.5,2)
    fmr.xpsCstatCount(date,26,2002)
    fmr.xpsfeedMissingInCstatdb(date,'US')
    fmr.xpsGvkeyValueChange(date,26,2002)
    fmr.xpsCstatValueCrossCheck(date,26,2002)
    fmr.cstatVsFactsetValue(date,26,12,.1)
    fmr.getStaleData(date,26)
    fmr.getAgeDist(date)
    
    visualizer = VisualTool.visualizer(htmlName,
                                       fmr.reportContents,date,
                                       reportHeader = header,
                                       displayThought=False)
 
    visualizer.visualize()

    marketDB.finalize()
    sys.exit()


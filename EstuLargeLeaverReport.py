
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def getAllDistributedSeries(conn):
    #Get all active regional models' information, especially rm_id and revision
    #Including WW4PrelimAxiomaMH, Factor Library and excluding CommodityModelMH
    
    query = """select b.SERIAL_ID, b.RM_ID, b.REVISION from RISK_MODEL a
               join (select * from RISK_MODEL_SERIE) b
               on a.MODEL_ID=b.RM_ID where a.MODEL_REGION is not null and b.DISTRIBUTE = 1 and b.THRU_DT > sysdate"""

    conn.dbCursor.execute(query)

    #To be updated when new model is introduced (for --all option)
    RECOGNIZED_SERIES = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 83, 84, 87, 88, 96, 97, 99, 100, 101, 102,
                         105, 106, 109, 110, 113, 114, 120, 121, 123, 124, 125, 126, 129, 130, 131, 132,
                         133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 153, 154, 155, 156,
                         161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177,
                         178, 185, 186, 187, 188, 189, 190, 191, 192, 193, 200, 201, 202, 203, 210, 211,
                         212, 213, 230, 231, 232, 233, 234, 235, 246, 247, 248, 249, 262, 263, 264, 265,
                         266, 267, 270, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287,
                         288, 300, 310, 311, 312, 313, 314, 320, 321, 322, 323, 324, 330, 331, 332, 333,
                         334, 335, 336, 337, 338, 339, 400, 401, 410, 411, 501, 2001]

    activeseries = dict()
    for rms_id, rm_id, rev_id in conn.dbCursor.fetchall():
        if rms_id in RECOGNIZED_SERIES:
            activeseries[rms_id] = [rm_id, rev_id]
        else:
            logging.warning('Found rms_id %s but it is not enabled for checking, update when it is ready for production!' % rms_id)

    return activeseries

def getVersionMap(conn, rms_id, date):
    #This method determines the estu table to look for when a rmi_id is given
    v3 = False

    #Treatment for US3 models, both RMI_ESTU_V3 and RMI_ESTU have information
    US3 = [170, 171, 172, 173, 174, 2001]
    US3MH_rms_id = [170, 172]
    US3SH_rms_id = [171, 173]

    if rms_id in US3:
        logging.debug("Using RMS_ESTU table for rms_id = %s for date = %s", rms_id, date)
        return v3
    # if rms_id in US3MH_rms_id:
    #     logging.debug("Using RMI_ESTU_V3 table for rms_id = %s", rms_id)
    #     v3 = True
    #     return v3
    # elif rms_id in US3SH_rms_id:
    #     if date >= datetime.strptime('2014-06-27', "%Y-%m-%d").date() and date <= datetime.strptime('2016-11-03', "%Y-%m-%d").date():
    #         logging.debug("Using RMS_ESTU table for rms_id = %s for date = %s", rms_id, date)
    #         return v3
    #     else:
    #         logging.debug("Using RMS_ESTU_V3 table for rms_id = %s for date = %s", rms_id, date)
    #         v3 = True
    #         return v3        
    else:
        #Get all active V3 rms_id in rmi_estu_v3
        query_v3 = """select distinct RMS_ID from RMI_ESTU_V3 a
                      join (select serial_id from RISK_MODEL_SERIE where DISTRIBUTE = 1 and THRU_DT > sysdate) b 
                      on a.RMS_ID = b.SERIAL_ID where a.NESTED_ID = 1"""

        #Get all active non V3 rms_id in rmi_estu
        query_non_v3 = """select distinct RMS_ID from RMI_ESTU a
                          join (select serial_id from RISK_MODEL_SERIE where DISTRIBUTE = 1 and THRU_DT > sysdate) b 
                          on a.RMS_ID = b.SERIAL_ID"""

        V3_list = list()
        non_V3_list = list()
    
        conn.dbCursor.execute(query_v3)
        for v in conn.dbCursor.fetchall():
            V3_list.append(v[0])

        conn.dbCursor.execute(query_non_v3)
        for n in conn.dbCursor.fetchall():
            non_V3_list.append(n[0])
            
        if rms_id in V3_list:
            logging.debug("Using RMI_ESTU_V3 table for rms_id = %s", rms_id)
            v3 = True
        elif rms_id in non_V3_list:
            logging.debug("Using RMI_ESTU table for rms_id = %s", rms_id)
            v3 = False
        else:
            logging.error("No active mapping to estu table is found, please check!")

    return v3
        
def getLeaver(conn, date, rms_id, rmg, universe, v3=False):
    #Find ESTU/Model Universe leavers for a model for a given datelist
    queryparam = dict()

    queryparam['dt_arg'] = date
    queryparam['rms_arg'] = rms_id

    previousdate = conn.getDateRange(rmg, date, date, tradingDaysBack=2, excludeWeekend=True)
    queryparam['prev_dt_arg'] = previousdate[0]
    
    # if date.isoweekday() == 1:
    #     queryparam['prev_dt_arg'] = date - timedelta(days=3)
    # else:
    #     queryparam['prev_dt_arg'] = date - timedelta(days=1)

    if universe == "ESTU":
        if v3:
            queryparam['version'] = '_V3'
            queryparam['addcon'] = ' and a.NESTED_ID = 1'
            queryparam['type'] = 'V3'
        else:
            queryparam['version'] = ''
            queryparam['addcon'] = ''
            queryparam['type'] = 'Non_V3'

        query = """select a.RMS_ID, c.DT, '%(type)s Main' as TYPE, count(a.SUB_ISSUE_ID) COUNT from RMI_ESTU%(version)s a
               left join (select * from RMI_ESTU%(version)s bb where bb.dt = '%(dt_arg)s' and bb.RMS_ID in (%(rms_arg)s)) b on b.sub_issue_id = a.sub_issue_id
               left join (select * from rmi_universe cc where cc.dt = '%(dt_arg)s' and cc.RMS_ID in (%(rms_arg)s)) c on c.sub_issue_id = a.sub_issue_id
               where a.RMS_ID in (%(rms_arg)s) and a.DT ='%(prev_dt_arg)s'%(addcon)s and b.SUB_ISSUE_ID is NULL and c.SUB_ISSUE_ID is not NULL 
               group by a.RMS_ID, c.DT"""
        
    elif universe == "Model":
        query = """select a.RMS_ID, a.DT, count(a.SUB_ISSUE_ID) COUNT from RMI_UNIVERSE a 
        left join (select * from RMI_UNIVERSE bb where bb.DT='%(dt_arg)s' and bb.RMS_ID in (%(rms_arg)s)) b on a.SUB_ISSUE_ID = b.SUB_ISSUE_ID
        where a.RMS_ID in (%(rms_arg)s) and a.DT = '%(prev_dt_arg)s' and b.SUB_ISSUE_ID is NULL and a.SUB_ISSUE_ID is not NULL
        group by a.RMS_ID, a.DT"""

    qry = query % queryparam
    leaversdf = pd.read_sql(qry, conn.dbConnection)

    return leaversdf

def getCorrelation(modelDB, marketDB, riskModel, date, rms_id, universe):
    #Compute the cross-sectional correlation of ESTU/Model Universe weights for a rms_id from one day to the next for the latest 7 model generation days
    dateranges = modelDB.getDateRange(riskModel.rmg, date , date, excludeWeekend=True, tradingDaysBack=7)

    corrdf = pd.DataFrame(columns = ['Date', 'CorrelCoef'])
    todaydf = pd.DataFrame()
    prevdf = pd.DataFrame()
    x = 0

    for index, dt in reversed(list(enumerate(dateranges))):
        if index > 0:
            if index == (len(dateranges)-1-x):
                rmi = modelDB.getRiskModelInstance(rms_id, dt)
                if rmi == None:
                    x =+ 1
                    continue
                if universe == "ESTU":
                    estu = modelDB.getRiskModelInstanceESTU(rmi)
                elif universe == "Model":
                    estu = modelDB.getRiskModelInstanceUniverse(rmi)
                else:
                    logging.error("Unknown universe type for correlation check")
                mcapDates = modelDB.getDates(riskModel.rmg, date, 19)
                mcap = modelDB.getAverageMarketCaps(mcapDates, estu, riskModel.numeraire.currency_id, marketDB)
                assetcapDict = dict(zip(estu, mcap.filled(0.0)))
                estuCaps = np.array([assetcapDict.get(s, 0.0) for s in estu])
                estuWeights = [1/float(len(estu)) for s in estu]
                estuWeights = estuCaps / np.sum(estuCaps)
                estuWeightsDict = dict(zip(estu, estuWeights*100))
                todaydf = pd.DataFrame(list(estuWeightsDict.items()), columns = ['Date', 'Weight'])
            else:
                todaydf = prevdf
                todaydf.rename(columns={'Weight_prev':'Weight'}, inplace=True)
 
            rmi_prev = modelDB.getRiskModelInstance(rms_id, dateranges[index-1])
            if rmi_prev is None:
                continue
            if universe == "ESTU":
                estu_prev = modelDB.getRiskModelInstanceESTU(rmi_prev)
            elif universe == "Model":
                estu_prev = modelDB.getRiskModelInstanceUniverse(rmi_prev)
            else:
                logging.error("Unknown universe type for correlation check")
            mcapDates_prev = modelDB.getDates(riskModel.rmg, dateranges[index-1], 19)
            mcap_prev = modelDB.getAverageMarketCaps(mcapDates_prev, estu_prev, riskModel.numeraire.currency_id, marketDB)
            assetcapDict_prev = dict(zip(estu_prev, mcap_prev.filled(0.0)))
            estuCaps_prev = np.array([assetcapDict_prev.get(s, 0.0) for s in estu_prev])
            estuWeights_prev = [1/float(len(estu_prev)) for s in estu_prev]
            estuWeights_prev = estuCaps_prev / np.sum(estuCaps_prev)
            estuWeightsDict_prev = dict(zip(estu_prev, estuWeights_prev*100))
            prevdf = pd.DataFrame(list(estuWeightsDict_prev.items()), columns = ['Date', 'Weight_prev'])

            mergedf = pd.DataFrame()
            mergedf = pd.merge(todaydf, prevdf, how='outer')
            mergedf.fillna(0, inplace=True)

            logging.debug("The ESTU correlation coefficient between %s and %s is %s" ,dt, dateranges[index-1], np.corrcoef(mergedf.Weight, mergedf.Weight_prev)[1][0])
            resultdf = pd.DataFrame()
            resultdf = pd.DataFrame({'Date':[dt], 'CorrelCoef': [np.corrcoef(mergedf.Weight, mergedf.Weight_prev)[1][0]]})
            corrdf = pd.concat([corrdf, resultdf], ignore_index=True)
        else:
            break
        
    return corrdf

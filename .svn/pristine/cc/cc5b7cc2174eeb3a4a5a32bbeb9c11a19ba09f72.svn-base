
import os
import logging
import logging.config
import sys
import optparse
import copy
import datetime
import time
from marketdb import MarketDB

STOXX='stoxx'
RUSSELL='russell'
FTSE='ftse'
FTSE_HIST='ftse_hist'

BEGINNING_OF_TIME='19500101'
END_OF_TIME='99991231'

UPDATED_THRU_DATES={
        STOXX:'20011229',
        RUSSELL:'20080531',
        FTSE:'20080531',
        FTSE_HIST:'20050101',    
}
VENDOR_TABLES={
        STOXX:'vendordb.stoxx_close_a_final',
        RUSSELL:'vendordb.rsl_glbl_index_final',
        FTSE:'vendordb.ftse_global_const_final',
        FTSE_HIST:'vendordb.ftse_global_const_hist_final',    
}
ASSET_REF=dict()

PRINT_STAT_FREQUENCY=10

rev_dt = datetime.datetime.now()

FTID_GLOBAL='ftid_global'
FTID_CANADA='ftid_canada'
FTID_US='ftid_us'
DATASTREAM='datastream' 

#COLUMNS_FTID=('axioma_id', 'code', 'change_dt', 'change_del_flag', 'src_id', 'ref', 'rev_dt', 'rev_del_flag')
#COLUMNS_DATASTREAM=('axioma_id', 'info_code', 'exch_int_code', 'change_dt', 'change_del_flag', 'src_id', 'ref', 'rev_dt', 'rev_del_flag')

FTID_CODES=('code',)
DATASTREAM_CODES=('info_code', 'exch_int_code',)

AXIOMA_ID_TO_VENDOR_ID_MAP_TABLES={
        FTID_GLOBAL:'asset_dim_ftid_global',
        FTID_US:'asset_dim_ftid_us',
        FTID_CANADA:'asset_dim_ftid_canada',
        DATASTREAM:'asset_dim_datastream',  
}

class Statistics:
    """Class to hold statistics (NP: since I cannot increase global variable (it becomes local right away))"""
    counterAssetRefAll=0    
    counterAssetRefProcessed=0   
    counterAssetRefDatesOk=0
    counterAssetRefDatesError=0
    counterAssetRefDatesUpdate=0
    counterAssetRefNotFoundInVendorDb=0

    countersUpdateAxiomaIdToVendorIdMapTables={
            FTID_GLOBAL:0,
            FTID_CANADA:0,
            FTID_US:0,
            DATASTREAM:0,    
    }
    def __init__(self,startTime):
        self.startTime=startTime
        
    def __str__(self):             
        s = """
stat counterAssetRefAll: %s
stat counterAssetRefProcessed: %s
stat counterAssetRefDatesOk: %s
stat counterAssetRefDatesError: %s
stat counterAssetRefDatesUpdate: %s
stat counterAssetRefNotFoundInVendorDb: %s                 
stat countersUpdateAxiomaIdToVendorIdMapTables: FTID_GLOBAL: %s FTID_CANADA: %s FTID_US: %s DATASTREAM: %s   
stat elapsed time: %10.6f min         
        """%(str(self.counterAssetRefAll),
             str(self.counterAssetRefProcessed),             
             str(self.counterAssetRefDatesOk),
             str(self.counterAssetRefDatesError),
             str(self.counterAssetRefDatesUpdate),
             str(self.counterAssetRefNotFoundInVendorDb),
             
             str(self.countersUpdateAxiomaIdToVendorIdMapTables[FTID_GLOBAL]),
             str(self.countersUpdateAxiomaIdToVendorIdMapTables[FTID_CANADA]),
             str(self.countersUpdateAxiomaIdToVendorIdMapTables[FTID_US]),
             str(self.countersUpdateAxiomaIdToVendorIdMapTables[DATASTREAM]),
             ((time.time()-startTime)/60),
             )
        return s   
    
class AssetData:

    """ Class to hold asset's data """
    
    def __init__(self,axiomaId, fromDt, thruDt, allSedols, allCusips):   
        self.axiomaId=axiomaId
        self.fromDt=fromDt
        self.thruDt=thruDt
        self.allSedols=allSedols
        self.allCusips =allCusips   
        self.vendorFromMin=None
        self.vendorThruMax=None 
        self.newFromDt=None
        self.newThruDt=None 
        

    def __str__(self):             
        s = """axioma_id: %s allSedols: %s allCusips: %s
ref_from_dt: %s ref_thru_dt: %s 
vendorFromMin: %s  vendorThruMax:  %s
newFromDt: %s  newThruDt:  %s
        """%(self.axiomaId, str(self.allSedols), str(self.allCusips),
             str(self.fromDt), str(self.thruDt),             
             str(self.vendorFromMin),str(self.vendorThruMax),
             str(self.newFromDt),str(self.newThruDt)
             )
        return s   
    
def setUpdatedThruDates(marketdb):
    logging.debug('in setUpdatedThruDates()')
    
    if False:
        sql="select to_char(max(dt)+1,'YYYYMMDD') from %s"
        for (vendor,table) in VENDOR_TABLES.items():
            logging.debug( '%s %s %s'%(vendor,table,sql%table))
            marketdb.dbCursor.execute(sql%table)
            
            resultSet = marketdb.dbCursor.fetchall()                  
            if len(resultSet)>0:
                UPDATED_THRU_DATES[vendor]=resultSet[0][0]
                
        logging.info('selected from the vendor tables max thru dates:')
        
    else:
        logging.info('use hardcoded max thru dates for testing:')
        
    for (vendor,lastUpdateDate) in UPDATED_THRU_DATES.items():     
        logging.info('%s %s'%(vendor,lastUpdateDate))
    return
    
    

def tightenAxiomaIdDates(marketdb,axiomaIds,ctryList,stat):    
    logging.debug('in tightenAxiomaIdDates()') 
    
    loadAssetData(marketdb,axiomaIds,ctryList,stat)
    loadVendorsFromThruDates(stat)

def loadVendorsFromThruDates(stat):
    
    printStatCounter=0
    
    for assetData in ASSET_REF.values():
        
        if printStatCounter==PRINT_STAT_FREQUENCY:
            printStatCounter=0
        if printStatCounter==0:
            logging.info(str(stat))
        printStatCounter=printStatCounter+1     
        
        
        if len(assetData.allSedols)<1 and len(assetData.allCusips)<1:
            logging.debug('no sedols or cusip found for the asset: %s'%str(assetData))
            continue

        cusipInClause = createInClause(assetData.allCusips)
        sedolInClause = createInClause(assetData.allSedols)

        fullClause = ''
        if len(assetData.allCusips) >= 1:
            fullClause = 'cusip in (' + cusipInClause + ")"
            
        if len(assetData.allSedols) >= 1:
            if len(fullClause) > 0:
                fullClause = fullClause + " or sedol in (" + sedolInClause + ')'
            else:
                fullClause = 'sedol in (' + sedolInClause + ')'

        rslFullClause = fullClause.replace('sedol', 'sedol_code')
        sqlOneAssetDates="""
select 
        least(
            nvl((select to_char(min(dt)) from vendordb.rsl_glbl_index_final where (%(rslFullClause)s) and dt < '%(russellUpdatedThruDate)s' ), '%(eot)s'),
            nvl((select to_char(min(dt)) from vendordb.stoxx_close_a_final where sedol in (%(sedolInClause)s) and dt < '%(stoxxUpdatedThruDate)s'), '%(eot)s'),
            nvl((select to_char(min(dt)) from vendordb.ftse_global_const_final where (%(fullClause)s) and dt < '%(ftseUpdatedThruDate)s'), '%(eot)s'),
            nvl((select to_char(min(dt)) from vendordb.ftse_global_const_hist_final where (%(fullClause)s) and dt < '%(ftseHistUpdatedThruDate)s'), '%(eot)s')
            ) as min_dt
,
        greatest(
(select case
when t1.dt < '%(russellUpdatedThruDate)s' then t1.dt
when t1.dt = '%(russellUpdatedThruDate)s' then '%(eot)s'
when t1.dt is null then '%(bot)s'
end as dt
from ( select to_char(max(dt)+1,'YYYYMMDD') dt from vendordb.rsl_glbl_index_final where (%(rslFullClause)s) and dt < '%(russellUpdatedThruDate)s') t1),  

(select case
when t2.dt < '%(stoxxUpdatedThruDate)s' then t2.dt
when t2.dt = '%(stoxxUpdatedThruDate)s' then '%(eot)s'
when t2.dt is null then '%(bot)s'
end as dt
from ( select to_char(max(dt)+1,'YYYYMMDD') dt from vendordb.stoxx_close_a_final where sedol in (%(sedolInClause)s) and dt < '%(stoxxUpdatedThruDate)s') t2),   

(select case
when t3.dt < '%(ftseUpdatedThruDate)s' then t3.dt
when t3.dt = '%(ftseUpdatedThruDate)s' then '%(eot)s'
when t3.dt is null then '%(bot)s'
end as dt
from ( select to_char(max(dt)+1,'YYYYMMDD') dt from vendordb.ftse_global_const_final where (%(fullClause)s) and dt < '%(ftseUpdatedThruDate)s') t3),   

(select case
when t4.dt < '%(ftseHistUpdatedThruDate)s' then t4.dt
when t4.dt = '%(ftseHistUpdatedThruDate)s' then '%(eot)s'
when t4.dt is null then '%(bot)s'
end as dt
from ( select to_char(max(dt)+1,'YYYYMMDD') dt from vendordb.ftse_global_const_hist_final where (%(fullClause)s) and dt < '%(ftseHistUpdatedThruDate)s') t4) 
) as max_dt
from dual
"""%{
     'russellUpdatedThruDate':UPDATED_THRU_DATES[RUSSELL],
     'stoxxUpdatedThruDate':UPDATED_THRU_DATES[STOXX],
     'ftseUpdatedThruDate':UPDATED_THRU_DATES[FTSE],
     'ftseHistUpdatedThruDate':UPDATED_THRU_DATES[FTSE_HIST],
     'bot':BEGINNING_OF_TIME,
     'eot':END_OF_TIME,
     'sedolInClause':sedolInClause,
     'fullClause':fullClause,
     'rslFullClause':rslFullClause
     }    
        
        logging.debug('sqlOneAssetDates: %s'%sqlOneAssetDates)
        
        marketdb.dbCursor.execute(sqlOneAssetDates)        
        resultSet = marketdb.dbCursor.fetchall()                         

# the len is always one
        assetData.vendorFromMin=resultSet[0][0]
        assetData.vendorThruMax=resultSet[0][1]
#        print 'assetData' ,assetData

        tightenAsset(assetData,stat) 
        stat.counterAssetRefProcessed=stat.counterAssetRefProcessed+1
        
            
        
def tightenAsset(assetData,stat):
    logging.debug('in tightenAsset()')

    if assetData.fromDt< assetData.vendorFromMin:
        assetData.newFromDt=assetData.vendorFromMin
        
#    elif assetData.fromDt> assetData.vendorFromMin:
#        stat.counterAssetRefDatesError=stat.counterAssetRefDatesError+1
#        logging.error('ERROR: asset_ref dates are too narrow: %s'%str(assetData))
#        
    
    if assetData.thruDt > assetData.vendorThruMax:
        assetData.newThruDt=assetData.vendorThruMax
        
#    elif assetData.thruDt < assetData.vendorThruMax:
#        stat.counterAssetRefDatesError=stat.counterAssetRefDatesError+1
#        logging.error('ERROR: asset_ref dates are too narrow: %s'%str(assetData))
#        

    if assetData.fromDt> assetData.vendorFromMin or assetData.thruDt < assetData.vendorThruMax:
        stat.counterAssetRefDatesError=stat.counterAssetRefDatesError+1
        logging.error('ERROR: asset_ref dates are too narrow: %s'%str(assetData))
        return
        
    if assetData.vendorFromMin == END_OF_TIME and assetData.vendorThruMax== BEGINNING_OF_TIME:
        logging.info('SKIP: asset was not found in vendordb: %s'%str(assetData))
        stat.counterAssetRefNotFoundInVendorDb=stat.counterAssetRefNotFoundInVendorDb+1
        return
    
    if assetData.newFromDt is not None or assetData.newThruDt is not None:
        logging.info('UPDATE: asset_ref dates will be tightened: %s'%str(assetData))
        stat.counterAssetRefDatesUpdate=stat.counterAssetRefDatesUpdate+1            
        updateAssetRef(assetData)
        updateAssetIdToVendorMaps(assetData,stat)            
            
    else:
        stat.counterAssetRefDatesOk=stat.counterAssetRefDatesOk+1
        logging.info('SKIP: asset_ref dates will NOT be changed: %s'%str(assetData))
            
#    if assetData.vendorFromMin != END_OF_TIME or assetData.vendorThruMax!= BEGINNING_OF_TIME:
#        if assetData.newFromDt is not None or assetData.newThruDt is not None:
#            logging.info('UPDATE: asset_ref dates will be tightened: %s'%str(assetData))
#            stat.counterAssetRefDatesUpdate=stat.counterAssetRefDatesUpdate+1            
#            updateAssetRef(assetData)
#            updateAssetIdToVendorMaps(assetData,stat)            
#            
#        else:
#            stat.counterAssetRefDatesOk=stat.counterAssetRefDatesOk+1
#            logging.info('SKIP: asset_ref dates will NOT be changed: %s'%str(assetData))
#            
#    else:
#        logging.info('SKIP: asset was not found in vendordb: %s'%str(assetData))
#        stat.counterAssetRefNotFoundInVendorDb=stat.counterAssetRefNotFoundInVendorDb+1
  
def updateAssetIdToVendorMaps(assetData,stat):
    logging.debug('in updateAssetIdToVendorMaps()')
    
    sqlSelect="""
select axioma_id, ref, %s,to_char(change_dt,'YYYYMMDD'), change_del_flag
from %s_active where axioma_id = '%s'
order by change_dt 
    """
    
    for (vendor, table)  in AXIOMA_ID_TO_VENDOR_ID_MAP_TABLES.items():              
        codes=DATASTREAM_CODES
        if vendor.lower().startswith("ftid"):   
            codes=FTID_CODES
            
#        print sqlSelect%(', '.join(codes),table, assetData.axiomaId)
        marketdb.dbCursor.execute(sqlSelect%(', '.join(codes),table, assetData.axiomaId))        
        resultSet = marketdb.dbCursor.fetchall()      
        
        if len(resultSet)<1:
            logging.debug('no records found in %s'%table)
            continue
        
        logging.debug('%d record(s) found in %s table'%(len(resultSet),table))
    
        codes=", ".join(FTID_CODES)
        changeDtIndex=2
        addValues=''
    
        if vendor.lower().startswith("datastream"):           
            codes=", ".join(DATASTREAM_CODES)
            changeDtIndex=3  
            addValues=',:9'
    
        sql = """
insert into %s (axioma_id, %s, change_dt, change_del_flag, src_id, ref, rev_dt, rev_del_flag) 
values (:1,:2,:3,:4,:5,:6,:7,:8%s)"""%(table, codes, addValues)
         
        
        values=[]
        if options.updateFromDt:
            values.extend(updateAxiomaIdVendorIdMapFromDt(resultSet,assetData,changeDtIndex))
        if options.updateThruDt:
            values.extend(updateAxiomaIdVendorIdMapThruDt(resultSet,assetData,changeDtIndex))
        
        if len(values) >0:
            logging.debug(sql)
            for row in values:
                logging.debug(row)

            marketdb.dbCursor.executemany(sql, values)
            stat.countersUpdateAxiomaIdToVendorIdMapTables[vendor]=stat.countersUpdateAxiomaIdToVendorIdMapTables[vendor]+1
            
def updateAxiomaIdVendorIdMapFromDt(resultSet,assetData,changeDtIndex):
    """1. invalidate all the records before newFromDt
2. insert newFromDt record
    """
    logging.debug('in updateAxiomaIdVendorIdMapFromDt()')
    logging.debug(updateAxiomaIdVendorIdMapFromDt.__doc__)
    
    values=[]    
    
    fromDtRow=None#to keep data from the last change_dt before the newFromDt
    
    for rowTuple in resultSet:#rows are sorted asc
        row=[]
        ref = ''
        for i in range(0, len(rowTuple)):
            if i == 1:
                ref = rowTuple[i]
            else:
                row.append(rowTuple[i])
        change_dt=row[changeDtIndex]    
       
        if change_dt < assetData.newFromDt:#invalidate and copy for later insertion
            newRow=copy.deepcopy(row)
            newRow.append(0)
            newRow.append(ref + ' tightenAxiomaIdDates.py')
            newRow.append(rev_dt)          
            newRow.append('Y')
            values.append(newRow)
            logging.debug('AxiomaIdVendorIdMapFromDt will be invalidated %s'%str(newRow))    
            fromDtRow=copy.deepcopy(newRow)
        
        if change_dt == assetData.newFromDt:#fromDtRow is already in place
            logging.debug('AxiomaIdVendorIdMapFromDt is already in place %s'%str(row))    
            return values
        
        if change_dt > assetData.newFromDt:  # do nothing?
            break
    
    if fromDtRow is not None:    
        fromDtRow[changeDtIndex]=assetData.newFromDt
        fromDtRow[len(fromDtRow)-1]='N'
        logging.debug('AxiomaIdVendorIdMapFromDt will be updated %s'%str(fromDtRow))        
        values.append(fromDtRow)        
            
#    print values
    return values
    
def updateAxiomaIdVendorIdMapThruDt(resultSet,assetData,changeDtIndex):
    """1. if there is a termination record in AxiomaIdVendorIdMap:
    print error(only in case if the termin date is different form newthruDate?) and return
    
2. if assetData.newThruDt != END_OF_TIME:       
    a. if there are records in AxiomaIdVendorIdMap with change_dt > assetData.newThruDt:
        invalidate these records(?)
    b. if the latest change_dt in AxiomaIdVendorIdMap < assetData.newThruDt:
        insert termination record as of assetData.newThruDt
    """
    logging.debug('in updateAxiomaIdVendorIdMapThruDt()')
    logging.debug(updateAxiomaIdVendorIdMapThruDt.__doc__)
    
    values=[]
    print(resultSet)
    lastRecord = resultSet[len(resultSet)-1]
    
    change_dt=lastRecord[changeDtIndex]
    change_del_flag=lastRecord[changeDtIndex+1]
    
    if change_del_flag=='Y' and change_dt == assetData.newThruDt:
        logging.debug('termination record already exists')
        return values    
   
    if change_del_flag=='Y' and change_dt != assetData.newThruDt:
        logging.error('ERROR: termination record already exists on %s'%(change_dt))
        return values
    
    if assetData.newThruDt != END_OF_TIME:
        newRow=[]
        ref = ''
        for i in range(0, len(lastRecord)):
            if i == 1:
                ref = lastRecord[i]
            else:
                newRow.append(lastRecord[i])
        #newRow.extend(lastRecord)
        newRow[changeDtIndex]=assetData.newThruDt
        newRow[changeDtIndex+1]='Y'
        newRow.append(0)
        newRow.append(ref + ' tightenAxiomaIdDates.py')
        newRow.append(rev_dt)          
        newRow.append('N')
        values.append(newRow) 
        logging.debug('AxiomaIdVendorIdMapThruDt will be updated %s'%str(newRow)) 
    else:
        logging.debug("AxiomaIdVendorIdMapThruDt don't need a termination record because newThruDt is %s"%(assetData.newThruDt))
#    print values
    return values
   
def updateAssetRef(assetData):
    logging.debug('in updateAssetRef()')
    
    setList=[]
    if options.updateFromDt:
        if assetData.newFromDt is not None:
            setList.append("from_dt=to_date('%s','YYYYMMDD')"%assetData.newFromDt)
        else:
            assetData.newFromDt=assetData.fromDt

    if options.updateThruDt:
        if assetData.newThruDt is not None:
            setList.append("thru_dt=to_date('%s','YYYYMMDD')"%assetData.newThruDt)
        else:
            assetData.newThruDt=assetData.thruDt

    if len(setList) > 0:
        setClause=", ".join(setList)
    
        sql="""update asset_ref 
        set %(setClause)s, src_id=0, ref=ref||' '||'tightenAxiomaIdDates.py'
        where axioma_id='%(axiomaId)s' and from_dt=to_date('%(fromDt)s','YYYYMMDD') and  thru_dt=to_date('%(thruDt)s','YYYYMMDD')
        """%{
            'axiomaId':assetData.axiomaId,
            'fromDt':assetData.fromDt,
            'thruDt':assetData.thruDt,
            'setClause':setClause,
            }

        logging.debug(sql)
        marketdb.dbCursor.execute(sql)

def createInClause(listOfValues):    
    return "'%s'"%("', '".join(listOfValues))
    
def loadAssetData(marketdb,axiomaIds,ctryList,stat):
    sql="""
select distinct r.axioma_id, to_char(r.from_dt,'YYYYMMDD'), to_char(r.thru_dt,'YYYYMMDD') , s.id as sedol, c.id as cusip 
from asset_ref r
left outer join asset_dim_cusip_active c on r.axioma_id=c.axioma_id
left outer join asset_dim_sedol_active s on r.axioma_id=s.axioma_id
%s
order by r.axioma_id
"""

    ctryClause = ''
    if ctryList.lower() != 'all':
        ctryClause = """r.axioma_id in (
        select distinct axioma_id from classification_const_active a, 
        (
        select a.code code_0, e.level_0, b.code code_1, e.level_1, c.code code_2, e.level_2, d.code code_3, e.level_3 from 
        (
        select a.parent_classification_id level_0, a.child_classification_id level_1,
        b.child_classification_id level_2, c.child_classification_id level_3
        from classification_dim_hier a
        left join classification_dim_hier b
        on a.child_classification_id=b.parent_classification_id
        left join classification_dim_hier c
        on b.child_classification_id=c.parent_classification_id
        where a.parent_classification_id=0) e
        left join
        classification_ref a
        on e.level_0=a.id
        left join
        classification_ref b
        on e.level_1=b.id
        left join
        classification_ref c
        on e.level_2=c.id
        left join
        classification_ref d
        on e.level_3=d.id
        ) b
        where (a.classification_id=b.level_2 or a.classification_id=b.level_3) and b.code_1 in (%s))""" % ("'" + ctryList.replace(",", "','") + "'")
        logging.debug("ctryClause: " + ctryClause)
        
        
    whereClause=''
    inClause=''  
     
    if axiomaIds is not None:    
        axiomaIdList =  axiomaIds.split(',')
        if len(axiomaIdList)>0:
            inClause="r.axioma_id in (%s) "%(createInClause(axiomaIdList))
         
        logging.debug('inClause: %s'%inClause)  
        
        whereClause='where %s'%(inClause)  

    if ctryClause != '':
        if whereClause != '':
            whereClause = whereClause + ' and ' + ctryClause
        else:
            whereClause = 'where %s ' % ctryClause
            
    logging.info(sql%(whereClause))
    
    marketdb.dbCursor.execute(sql%(whereClause))
        
    resultSet = marketdb.dbCursor.fetchall()                  
    
    prevAxiomaId=''
    prevFromDt=''
    prevTrueDt=''
    allSedols=[]
    allCusips=[]
    if len(resultSet)>0:        
        for row in resultSet:
            if row[0]!=prevAxiomaId:
                if len(allSedols)>0 or len(allCusips)>0:
                    ASSET_REF[prevAxiomaId]=AssetData(prevAxiomaId,prevFromDt,prevTrueDt,allSedols,allCusips)
                prevAxiomaId=row[0]
                prevFromDt=row[1]
                prevTrueDt=row[2]
                allSedols=[]
                allCusips=[]
                
            if row[3] is not None:
                allSedols.append(row[3])
            if row[4] is not None:
                allCusips.append(row[4])
                
    if len(allSedols)>0 or len(allCusips)>0:
        ASSET_REF[prevAxiomaId]=AssetData(prevAxiomaId,prevFromDt,prevTrueDt,allSedols,allCusips)     
    
    stat.counterAssetRefAll=len(ASSET_REF)
     
    for assetData in ASSET_REF.values():
        logging.debug('loaded asset data: %s'%str(assetData))
    
    logging.info('assets loaded from asset_ref: %s'%str(len(ASSET_REF)))
    
if __name__ == '__main__':
    usage = "usage: %prog [options] " 
    cmdlineParser = optparse.OptionParser(usage=usage)
    
    cmdlineParser.add_option("-u", "--updateDb", action="store_true",
                            default=False, dest="updateDb",
                            help="commit changes to the database")

    cmdlineParser.add_option("-f", "--update-from-dt", action="store_true",
                            default=False, dest="updateFromDt",
                            help="update from dt for assets")

    cmdlineParser.add_option("-t", "--update-thru-dt", action="store_true",
                            default=False, dest="updateThruDt",
                            help="update thru dt for assets")

    cmdlineParser.add_option("--axiomaIds", action="store",
                            default=None, dest="axiomaIds",
                            help="comma-separated axiomaId list")

    cmdlineParser.add_option("--countries", action="store",
                            default=None, dest="ctryList",
                            help="comma-separated list of ISO ctry codes")

    cmdlineParser.add_option("--marketdb-user", action="store",
                            default=os.environ.get('MARKETDB_USER'), 
                            dest="marketDBUser", 
                            help="user for MarketDB access")
    cmdlineParser.add_option("--marketdb-passwd", action="store", 
                            default=os.environ.get('MARKETDB_PASSWD'), 
                            dest="marketDBPasswd", 
                            help="Password for MarketDB access")
    cmdlineParser.add_option("--marketdb-sid", action="store", 
                            default=os.environ.get('MARKETDB_SID'), 
                            dest="marketDBSID", 
                            help="Oracle SID for MarketDB access")
    cmdlineParser.add_option("-l", "--log-config", action="store", 
                            default='log.config', dest="logConfigFile", 
                            help="logging configuration file")
    
    cmdlineParser.add_option("--log-directory", action="store", 
                             default='.', dest="logDir", 
                             help="directory for log files")
    (options, args) = cmdlineParser.parse_args()
       
    if not os.path.exists(options.logConfigFile):
        cmdlineParser.error("Log Configuration file doesn't exist: "+options.logConfigFile)
        print(usage)
        sys.exit(1)
            
    logging.config.fileConfig(options.logConfigFile)   
     
    logging.debug('options:  '+ str(options))
    
    if options.marketDBSID is None or options.marketDBPasswd is None  or options.marketDBUser is None:
        cmdlineParser.error("marketDBSID or marketDBPasswd or marketDBUser is None")
        print(usage)
        sys.exit(1)   
    
    logging.info("Log directory: "+str(os.path.abspath(options.logDir)))
    
    axiomaIds=options.axiomaIds
    ctryList = options.ctryList
    updateDb = options.updateDb
    
    
    startTime =time.time()
    
    logging.info("Started at %s "%(str(datetime.datetime.now())[0:19]))
    
    marketdb=MarketDB.MarketDB(user=options.marketDBUser, passwd=options.marketDBPasswd, 
                                sid=options.marketDBSID)
        
    marketdb.dbCursor.execute("ALTER SESSION SET NLS_DATE_FORMAT='YYYYMMDD'")
    
    
    
    counter = 0
    stat = Statistics(startTime)
    try:
        setUpdatedThruDates(marketdb)
        tightenAxiomaIdDates(marketdb,axiomaIds,ctryList,stat)
        logging.info(str(stat))
    except Exception as e:
        logging.error('An Error ocurred: %s'%str(e), exc_info=True)
        marketdb.revertChanges()
    
    if stat.counterAssetRefDatesUpdate>0 and updateDb:
        logging.info("Commiting changes")
        #mktDB.revertChanges()        
        marketdb.commitChanges()
    else:
        logging.info("Rolling back changes")
        marketdb.revertChanges()
        
    marketdb.finalize() 
    logging.info("Finished at %s, elapsed time %10.6f min"%((str(datetime.datetime.now())[0:19]),((time.time()-startTime)/60)))
    
    sys.exit(0)

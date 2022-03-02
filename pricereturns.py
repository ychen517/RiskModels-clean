
import optparse
import configparser
import os
import datetime
import logging.config
import logging
import cx_Oracle
import numpy
from collections import defaultdict
import math

INCR=300
TradingDaysPerYear=252
annualizeFactor=math.sqrt(TradingDaysPerYear)
sampleStdAdjFactor = math.sqrt(float(TradingDaysPerYear)/float(TradingDaysPerYear-1))

def parseISODate(dateStr):
    """Parse a string in YYYY-MM-DD format and return the corresponding
    date object.
    """
    assert(len(dateStr) == 10)
    assert(dateStr[4] == '-' and dateStr[7] == '-')
    return datetime.date(int(dateStr[0:4]), int(dateStr[5:7]),
                         int(dateStr[8:10]))


if __name__ == '__main__':
    usage = "usage: %prog [options] date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--output-dir", action="store",
                            default='/axioma/products/current/RussellAxiomaIndexes/US_R3000_TotalReturn', dest="outputDir",
                            help="Output directory name")
    cmdlineParser.add_option("--modeldb-passwd", action="store",
                            default=None, dest="modelDBPasswd",
                            help="Password for ModelDB access")
    cmdlineParser.add_option("--modeldb-sid", action="store",
                            default=None, dest="modelDBSID",
                            help="Oracle SID for ModelDB access")
    cmdlineParser.add_option("--modeldb-user", action="store",
                            default=None, dest="modelDBUser",
                            help="user for ModelDB access")
    cmdlineParser.add_option("-l", "--log-config", action="store",
                            default='log.config', dest="logConfigFile",
                            help="logging configuration file")
    cmdlineParser.add_option("--recon", action="store_true",
                             default=False, dest="recon",
                             help="is this the recon")

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")

    # get the ealiest date
    conn = cx_Oracle.connect(options.modelDBUser, options.modelDBPasswd, options.modelDBSID)
    cursor = conn.cursor()
    logging.config.fileConfig(options.logConfigFile)
    date=parseISODate(args[0])
    query1="""
    select min(dt) dt from (
    select cal.dt, cal.sequence from Risk_model_group rmg,RMG_CALENDAR cal
    where cal.rmg_id = rmg.rmg_id
    and rmg.MNEMONIC='US'
    and dt<= :dt
    order by dt desc
    ) v where rownum <= (%d) 
    """ % (TradingDaysPerYear)

    cursor.execute(query1,dt=date)
    olddt=parseISODate(str(cursor.fetchall()[0][0])[:10])
    logging.info("Old date=%s",olddt)
    
    # get the most recent date
    query2="""
       select min(dt) from marketdb_global.index_revision_active ir where 
           index_id=(select id from marketdb_global.index_member where name='RUSSELL 3000 NEXT DAY OPEN')
           and dt > :dt
           and exists (select * from marketdb_global.index_constituent ic where ic.revision_id=ir.id)
    """
    cursor.execute(query2,dt=date)
    nextdate=parseISODate(str(cursor.fetchall()[0][0])[:10])
    logging.info("Russell next day open index date=%s", nextdate)

    # get all the assets satisfying the criteria
    if options.recon:
        indexName = 'RUSSELL 3000 RECON OPEN'
    else:
        indexName = 'RUSSELL 3000 NEXT DAY OPEN'
    query3="""
        select s.sub_id from
                marketdb_global.index_revision_active ir, marketdb_global.index_constituent ic, issue_map im,
                sub_issue s, rms_issue rms
        where 
                ir.id=ic.revision_id and ir.dt=:dt 
                and ir.index_id=(select id from marketdb_global.index_member where name='%s')
                and im.modeldb_id=s.issue_id and im.marketdb_id=ic.axioma_id
                and rms.rms_id=
                        (select serial_id from risk_model_serie 
                                where distribute=1 and 
                                RM_ID = (select  MODEL_ID from risk_model where name='US2AxiomaMH'))
                and rms.issue_id=im.modeldb_id
                and rms.from_dt <= :olddt and rms.thru_dt > :nextdate
    """ % (indexName)
    logging.info("Getting return constituents for %s", date)
    cursor.execute(query3,dt=nextdate, olddt=olddt, nextdate=nextdate)
    results=cursor.fetchall()
    logging.info("Got %d constituents", len(results))
    
    
     # create the output file: histvol_yyyymmdd.csv
    histvol = dict()
    if options.recon:
        fname = options.outputDir + '/histvol_'+ 'recons_'  + date.strftime('%Y%m%d')+'.csv'
    else:
        fname = options.outputDir + '/histvol_'+ date.strftime('%Y%m%d')+'.csv'
    
    if not os.path.exists(options.outputDir):
        os.makedirs(options.outputDir)
   
    
    f = open(fname, 'w')
    numAssetsDropped = 0   
    
    while results:
        histret = defaultdict(list)
        ids=[ res[0] for res  in results[:INCR]]
        idstr= ','.join(["'%s'" % i for i in ids])

        # get the return data for each asset between olddt and current date
        query4="""
            select sub_issue_id, dt, nvl(tr,0.0) from sub_issue_return_active ret where sub_issue_id in (%s)
            and exists (select * from rmi_universe rmi where rmi.dt=ret.dt and rmi.sub_issue_id=ret.sub_issue_id
                        and rmi.rms_id=
                        (select serial_id from risk_model_serie 
                                where distribute=1 and 
                                RM_ID = (select  MODEL_ID from risk_model where name='US2AxiomaMH'))
            )
            and dt between :olddt and :dt order by sub_issue_id, dt
        """ % (idstr)
        results=results[INCR:]
        cursor.execute(query4, olddt=olddt, dt=date)
        newresults=cursor.fetchall()
        # do processing. write results into options.outputDir 
        logging.info( '%d ids returned %d rows' , len(ids), len(newresults) )

        #compute the standard deviation and add to output later 
        for id, dt, tr in newresults: 
            if dt.strftime('%Y%m%d') > date.strftime('%Y%m%d'):
                logging.info('get data for next date[%s]' % dt.strftime('%Y%m%d'))
            histret[id[1:]].append(tr)
        
        logging.info('#return assets = %d', len(histret))

        for key, val in histret.items():
            if len(val) == TradingDaysPerYear:
                histvol[key] = numpy.std(val)*annualizeFactor*sampleStdAdjFactor
            else :
                logging.info("skip [%s] with only %d days of history" % (key[:-2], len(val)))
                numAssetsDropped = numAssetsDropped + 1 
                
    # get the number of assets in R3000 next day open for the current date
    query5="""
             SELECT COUNT(*) FROM marketdb_global.INDEX_CONSTITUENT_EASY 
                WHERE REVISION_ID=
        (SELECT ID FROM marketdb_global.INDEX_REVISION_ACTIVE
         WHERE INDEX_ID=(SELECT ID FROM marketdb_global.INDEX_MEMBER WHERE NAME='%s')
        AND DT=:dt)
                """ % (indexName)
    cursor.execute(query5, dt=nextdate)
    numAssetsInBenchmark=cursor.fetchall()[0][0]
    
    f.write("#Historical %d-Day Volatility (from %s to %s); total assets in R3K=%d; assets with full %d-day return history = %d\n" %(TradingDaysPerYear, olddt.strftime('%Y%m%d'), date.strftime('%Y%m%d'), numAssetsInBenchmark, TradingDaysPerYear, len(histvol) ))
    
    for key in sorted(histvol.keys()):
        f.write("%s,%g\n" % (key[:-2], histvol[key]))

    f.close()


import datetime
import logging
import optparse

from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

QUERY = """SELECT DISTINCT SUBSTR(si.issue_id,2,9) axioma_id,
    cref.name exchange,
    (SELECT dema.exch_int_code FROM datastream_exchange_map_active dema
     WHERE dema.classification_id=cref.id AND dema.change_dt=
     (SELECT MAX(change_dt) FROM datastream_exchange_map_active dema1
      WHERE dema1.change_dt<=rmiu.dt AND dema1.exch_int_code=dema.exch_int_code)
     AND dema.change_del_flag='N' AND ROWNUM<=1) datastream_exch_code,
    (SELECT fema.exchange ftid_exchange_code FROM
     ftid_global_exch_map_active fema
     WHERE fema.classification_id=cref.id AND fema.change_dt=
     (SELECT MAX(change_dt) FROM ftid_global_exch_map_active fema1
      WHERE fema1.change_dt<=rmiu.dt AND fema1.exchange=fema.exchange)
     AND fema.change_del_flag='N' AND ROWNUM<=1) ftid_exch_code,
    curr.code currency, 
    ucp.value price, 
    tdv.value volume
    FROM %(modelUser)s.rmi_universe rmiu, %(modelUser)s.sub_issue si,
    %(modelUser)s.issue_map im, asset_dim_market_active adma,
    classification_ref cref, asset_dim_ucp_active ucp, currency_ref curr,
    asset_dim_tdv_active tdv
    --WHERE rmiu.rms_id=:rms_arg and rmiu.dt=:dt_ar
    WHERE rmiu.rms_id > 0 and rmiu.dt=:dt_arg
    AND si.sub_id=rmiu.sub_issue_id
    -- next line excludes US assets
    AND si.rmg_id <> 1
    AND im.modeldb_id=si.issue_id AND im.from_dt<=rmiu.dt AND im.thru_dt>rmiu.dt
    AND adma.axioma_id=im.marketdb_id
    AND adma.change_dt=(SELECT MAX(change_dt) FROM asset_dim_market_active adma1
     WHERE adma1.change_dt<=rmiu.dt AND adma1.axioma_id=adma.axioma_id
    ) AND adma.change_del_flag='N'
    AND cref.id=adma.classification_id
    AND ucp.axioma_id(+)=im.marketdb_id AND ucp.dt(+)=:dt_arg
    AND curr.id(+)=ucp.currency_id
    AND tdv.axioma_id(+)=im.marketdb_id AND tdv.dt(+)=:dt_arg
"""

NEW_QUERY="""SELECT SUBSTR(v.modeldb_id,2,9) axioma_id, cref.name exchange,
 (SELECT dema.exch_int_code FROM datastream_exchange_map_active dema
     WHERE dema.classification_id=cref.id AND dema.change_dt=
     (SELECT MAX(change_dt) FROM datastream_exchange_map_active dema1
      WHERE dema1.change_dt<=:dt_arg AND dema1.exch_int_code=dema.exch_int_code)
     AND dema.change_del_flag='N' AND ROWNUM<=1) datastream_exch_code,
 (SELECT fema.exchange ftid_exchange_code FROM
     ftid_global_exch_map_active fema
     WHERE fema.classification_id=cref.id AND fema.change_dt=
     (SELECT MAX(change_dt) FROM ftid_global_exch_map_active fema1
      WHERE fema1.change_dt<=:dt_arg AND fema1.exchange=fema.exchange)
     AND fema.change_del_flag='N' AND ROWNUM<=1) ftid_exch_code,
 (SELECT curr.code FROM asset_dim_ucp_active ucp,currency_ref curr WHERE ucp.dt=:dt_arg AND ucp.axioma_id=v.marketdb_id
  AND curr.id=ucp.currency_id) currency,
 (SELECT value FROM asset_dim_ucp_active ucp WHERE ucp.dt=:dt_arg AND ucp.axioma_id=v.marketdb_id) price,
 (SELECT value FROM asset_dim_tdv_active tdv WHERE tdv.dt=:dt_arg AND tdv.axioma_id=v.marketdb_id) volume
 FROM
  (SELECT im.marketdb_id, im.modeldb_id,
   (SELECT classification_id FROM asset_dim_market_active_int adma 
    WHERE adma.axioma_id=im.marketdb_id AND adma.from_dt <= :dt_arg AND :dt_arg < adma.thru_dt) classid
   FROM modeldb_global.issue_map im, modeldb_global.sub_issue si
   WHERE si.rmg_id <> 1
   AND si.issue_id=im.modeldb_id
   AND EXISTS (SELECT * FROM modeldb_global.rmi_universe rmi WHERE rmi.dt=:dt_arg AND rmi.sub_issue_id=si.sub_id)
  ) v, classification_ref cref
 WHERE v.classid=cref.id"""

NULL = ''
def numOrNull(val, fmt, threshhold=None, fmt2=None):
    """Formats a number with the specified format or returns NULL
    if the value is masked.
    """
    if val is None:
        return NULL
    if threshhold and fmt2 and val<threshhold:
        return fmt2 % val
    else:
        return fmt % val

#def writeExchFile(options, date, riskModel, modelDB, marketDB):
def writeExchFile(options, date, modelDB, marketDB):
    #out = open('%s/%s.%s.exch' % (options.targetDir, riskModel.mnemonic, str(date).replace('-','')), 'w')
    out = open('%s/AXGL.%s.exch' % (options.targetDir, str(date).replace('-','')), 'w')
    out.write('#DataDate: %s\n' % date)
    out.write('#CreationTimestamp: %s\n' %
              datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    out.write('#Columns: Axioma ID|Exchange|Datastream Exch Code|FTID Exch Code|Currency|Price|Volume\n')
    out.write('#Type: ID|Attribute|Attribute|Attribute|NA|Attribute|Attribute\n')
    out.write('#Unit: ID|Text|Number|Text|Text|CurrencyPerShare|Number\n')
    query = NEW_QUERY % dict(modelUser=options.modelDBUser)
    logging.debug('query: %s; dt_arg: %s', query, date)
    marketDB.dbCursor.execute(query, dt_arg=date)
    r = marketDB.dbCursor.fetchmany()
    while r:
        logging.debug('got %d records', len(r))
        for rec in r:
            out.write('%s|%s|%s|%s|%s|%s|%s\n' %
                      (rec[0], rec[1], rec[2], rec[3], rec[4], numOrNull(rec[5], '%.5f'), numOrNull(rec[6], '%d')))
        r = marketDB.dbCursor.fetchmany()
    out.close()

if __name__=='__main__':
    usage = "usage: %prog [options] <startdate or datelist> <end-date>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)

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
        dates = sorted(dates,reverse=True)
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = sorted([startDate + datetime.timedelta(i)
                      for i in range((endDate-startDate).days + 1)], reverse=True)
    
    for d in dates:
        #riskModel.setFactorsForDate(d, modelDB)
        #rmi = riskModel.getRiskModelInstance(d, modelDB)
        #if rmi != None and rmi.has_risks:
        #    logging.info('Processing %s' % str(d))
        #    writeExchFile(options, d, riskModel, modelDB, marketDB)
        #else:
        #    if len(dates)==1:
        #        logging.fatal('No risk model instance on %s' % str(d))
        #        sys.exit(1)
        #    else:
        #        logging.error('No risk model instance on %s' % str(d))
        writeExchFile(options, d, modelDB, marketDB)

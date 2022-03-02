import cx_Oracle
import sys
import time
import optparse
import logging
import logging.config

if __name__ == '__main__':
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-u", "--user", dest="user",
                      action="store",help="User name", default='modeldb_global')
    parser.add_option("-s", "--sid", dest="sid",
                      action="store",help="SID name", default='glprod')
    parser.add_option("-p", "--pass", dest="passwd",
                      action="store",help="Password", default='modeldb_global')
    parser.add_option("-n", "--testOnly", dest="testOnly",
                      action="store_true", default=False, help="Don't commit changes")

    (options, args) = parser.parse_args()

    #print "DEBUG:", options.user, options.passwd, options.sid
    conn = cx_Oracle.connect(options.user, options.passwd, options.sid)
    cursor = conn.cursor()
    #cursor.execute('alter session set nls_date_format="YYYYMMDD"')
    cursor.execute('alter session set nls_date_format="YYYY-MM-DD"')

    rms_id = sys.argv[1]

    tables= [
        'RISK_MODEL_INSTANCE',
        'RMS_FACTOR_RETURN',
        ##'RMS_ESTU_EXCLUDED',
        'RMS_FACTOR_DESCRIPTOR',
        'RMS_FACTOR_STATISTICS',
        ##'RMG_MODEL_MAP',
        'RMS_ISSUE',
        'RMS_STATISTICS',
        'RMS_STND_MEAN',
        'RMS_STND_STDEV',
        'RMS_FACTOR_STATISTICS_INTERNAL', 
        'RMS_STATISTICS_INTERNAL', 
        'RMS_FACTOR_RETURN_INTERNAL', 
        'RMS_STND_EXP', 
        'RMS_STND_DESC',
    ]
    #tables=['RISK_MODEL_INSTANCE']
    logging.config.fileConfig('log.config')
    logging.info('Start')
    for tbl in tables:
        logging.info( 'Working on %s', tbl)
        query = """delete from modeldb_global.%s where rms_id=%s """ % (tbl, rms_id)
        print (query)
        cursor.execute(query)
        logging.info('%d rows deleted', cursor.rowcount)
        query = """insert into modeldb_global.%s select * from modeldb_global.%s@MDLDB_GLBL_RESEARCHODA.AXIOMAINC.COM where rms_id=%s """ % (tbl, tbl, rms_id)
        print (query)
        cursor.execute(query)
        logging.info('%d rows inserted', cursor.rowcount)
        if options.testOnly:
            logging.info('Reverting changes')
            conn.rollback()
        else:
            logging.info('Committing changes')
            conn.commit()
    logging.info('Done')

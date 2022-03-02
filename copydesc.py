
import cx_Oracle
import sys
import time
import optparse
import datetime
import logging
import logging.config

if __name__ == '__main__':
    logging.config.fileConfig('log.config')
    logging.info('Start')


    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("-u", "--user", dest="user",
                      action="store",help="User name", default='modeldb_global')
    parser.add_option("-s", "--sid", dest="sid",
                      action="store",help="SID name", default='glprod')
    parser.add_option("-p", "--pass", dest="passwd",
                      action="store",help="Password", default='modeldb_global')
    parser.add_option("--no-header", dest="noHeader",
                      action="store_true",help="show header", default=False)

    (options, args) = parser.parse_args()

    #print "DEBUG:", options.user, options.passwd, options.sid
    conn = cx_Oracle.connect(options.user, options.passwd, options.sid)
    cursor = conn.cursor()
    cursor.execute('alter session set nls_date_format="YYYY-MM-DD"')
    query="""
    insert into modeldb_golden.descriptor_local_currency (SUB_ISSUE_ID, DT, CURR_FIELD
    ,DS_209, DS_210
    ,DS_302 ,DS_303
    ,DS_304 ,DS_305 ,DS_306 ,DS_307 ,DS_316 ,DS_317 ,DS_318 ,DS_319 ,DS_308
    ,DS_309 ,DS_310 ,DS_311 ,DS_320 ,DS_321 ,DS_322 ,DS_323 ,DS_324 ,DS_325
    ,DS_326 ,DS_327 ,DS_328 ,DS_329 ,DS_330 ,DS_331 ,DS_332 ,DS_333 ,DS_334
    ,DS_335 ,DS_336 ,DS_337 ,DS_338 ,DS_339 ,DS_340 ,DS_341 ,DS_342 ,DS_343
    ,DS_344 ,DS_345 ,DS_346 ,DS_347 ,DS_348 ,DS_349 ,DS_350 ,DS_351 ,DS_352
    ,DS_353 ,DS_354 ,DS_355 ,DS_356 ,DS_357
    )
    select SUB_ISSUE_ID,DT,CURR_FIELD
    ,DS_209, DS_210 
    ,DS_302 ,DS_303
    ,DS_304 ,DS_305 ,DS_306 ,DS_307 ,DS_316 ,DS_317 ,DS_318 ,DS_319 ,DS_308
    ,DS_309 ,DS_310 ,DS_311 ,DS_320 ,DS_321 ,DS_322 ,DS_323 ,DS_324 ,DS_325
    ,DS_326 ,DS_327 ,DS_328 ,DS_329 ,DS_330 ,DS_331 ,DS_332 ,DS_333 ,DS_334
    ,DS_335 ,DS_336 ,DS_337 ,DS_338 ,DS_339 ,DS_340 ,DS_341 ,DS_342 ,DS_343
    ,DS_344 ,DS_345 ,DS_346 ,DS_347 ,DS_348 ,DS_349 ,DS_350 ,DS_351 ,DS_352
    ,DS_353 ,DS_354 ,DS_355 ,DS_356 ,DS_357
    from descriptor_local_currency@MDLDB_GLBL_RESEARCHODA.AXIOMAINC.COM p
    where dt=:dt
    """
    query1="""
    insert into modeldb_golden.descriptor_numeraire_usd (SUB_ISSUE_ID, DT, CURR_FIELD, DS_220, DS_221, DS_216, DS_217,DS_224,DS_164,DS_167, DS_107 )
    select SUB_ISSUE_ID, DT, CURR_FIELD, DS_220, DS_221, DS_216, DS_217, DS_224,DS_164 ,DS_167, DS_107 
    from descriptor_numeraire_usd@MDLDB_GLBL_RESEARCHODA.AXIOMAINC.COM p
    where dt=:dt
    """
    date=datetime.date(1996,12,31)
    enddate=datetime.date(2017,6,9)
    try:

        while date <= enddate:
            exists = cursor.execute("""select 1 from dual where exists (select * from modeldb_golden.descriptor_local_currency where dt=:dt)""",
                    dt=date)
            res = cursor.fetchall()
            if date.isoweekday() not in (6,7): 
                logging.info('%s....',  date)
                if len(res) == 1:
                    print('Already done with %s' % date)
                else:
                    cursor.execute(query, dt=date)
                    logging.info( 'Done with DESCRIPTOR_LOCAL_CURRENCY %s %d' % (date , cursor.rowcount))
                    cursor.execute(query1, dt=date)
                    logging.info( 'Done with DESCRIPTOR_NUMERAIRE_USD %s %d' % (date , cursor.rowcount))
                    conn.commit()
            else:
                logging.info("Ignoring %s ... weekend", date)
            date = date + datetime.timedelta(days=1)
    except:
        print("Unepected error:", sys.exc_info()[0], sys.exc_info()[1])


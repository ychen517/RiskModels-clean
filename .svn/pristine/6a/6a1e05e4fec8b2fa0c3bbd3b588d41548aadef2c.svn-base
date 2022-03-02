
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
    query = """update /*Index(t IDX_DT_CURR_FIELD_LOCAL_DESC) */ modeldb_global.descriptor_local_currency t
    set
    (t.DS_209,t.DS_210,t.DS_302,t.DS_303,t.DS_304,t.DS_305,t.DS_306,t.DS_307,t.DS_308,t.DS_309,t.DS_310,t.DS_311,t.DS_316,t.DS_317,t.DS_318,t.DS_319,t.DS_320,t.DS_321,t.DS_322,t.DS_323,t.DS_324,t.DS_325,t.DS_326,t.DS_327,t.DS_328,t.DS_329,t.DS_330,t.DS_331,t.DS_332,t.DS_333,t.DS_334,t.DS_335,t.DS_336,t.DS_337,t.DS_338,t.DS_339,t.DS_340,t.DS_341,t.DS_342,t.DS_343,t.DS_344,t.DS_345,t.DS_346,t.DS_347,t.DS_348,t.DS_349,t.DS_350,t.DS_351,t.DS_352,t.DS_353,t.DS_354,t.DS_355,t.DS_356,t.DS_357) =
    (
    select  /*+ Index(g PK_DESC_LOCAL_CURRENCY) */ 
    DS_209,DS_210,DS_302,DS_303,DS_304,DS_305,DS_306,DS_307,DS_308,DS_309,DS_310,DS_311,DS_316,DS_317,DS_318,DS_319,DS_320,DS_321,DS_322,DS_323,DS_324,DS_325,DS_326,DS_327,DS_328,DS_329,DS_330,DS_331,DS_332,DS_333,DS_334,DS_335,DS_336,DS_337,DS_338,DS_339,DS_340,DS_341,DS_342,DS_343,DS_344,DS_345,DS_346,DS_347,DS_348,DS_349,DS_350,DS_351,DS_352,DS_353,DS_354,DS_355,DS_356,DS_357
    from
    modeldb_golden.descriptor_local_currency g
    where g.sub_issue_id=t.sub_issue_id and g.dt = t.dt and g.curr_field = t.curr_field
    )
    where t.dt=:dt
    """
    query1="""
    update /*+Index(t IDX_DT_CURR_FIELD_DESC) */ modeldb_global.descriptor_numeraire_usd t set 
    (t.DS_220,t.DS_221,t.DS_216,t.DS_217,t.DS_224,t.DS_164,t.DS_167,t.DS_107) =
    (select  /*+ Index(g PK_DESC_NUMERAIRE_USD) */
    g.DS_220,g.DS_221,g.DS_216,g.DS_217,g.DS_224,g.DS_164,g.DS_167,g.DS_107
    from modeldb_golden.descriptor_numeraire_usd g where g.dt=t.dt and g.sub_issue_id=t.sub_issue_id and g.curr_field=t.curr_field)
    where t.dt=:dt
    """
    date=datetime.date(1996,12,31)
    enddate=datetime.date(2017,6,9)
    try:

        while date <= enddate:
            if date.isoweekday() not in (6,7): 
                logging.info('%s....',  date)
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


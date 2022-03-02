
import datetime
import logging
import optparse
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

def main():
    usage = "usage: %prog [-n --verbose|-v] [connection-args]  date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="Extra debugging output")
    cmdlineParser.add_option("--axids", "-a", action="store",
                             default=None, dest="axidList",
                             help="list of axiomaids")

    Utilities.addModelAndDefaultCommandLine(cmdlineParser)

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    modelClass = Utilities.processModelAndDefaultCommandLine( options, cmdlineParser)

    #Utilities.processDefaultCommandLine(options, cmdlineParser)

    dt = Utilities.parseISODate(args[0])
    if len(args) == 1:
        endDate = dt
    else:
        endDate = Utilities.parseISODate(args[1])

    modelDB = ModelDB.ModelDB(sid=options.modelDBSID,
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID,
                                 user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)

    riskModel = modelClass(modelDB, marketDB)
    rmsid = riskModel.rms_id
    currency = riskModel.numeraire.currency_code

    # check to see how many assets did not have descriptors created
    # cheat for now and use the marketdb_global directly.  SOmetime in future we should fix this
    # and not have cross database references
    query = """
        select v.dt, v.issue_id, v.ds_100, v.ds_102, v.ds_105, v.ds_106, v.ds_107, v.ds_108, v.ds_109, v.ds_182, v.ds_183, v.ds_184, v.ds_184, v.ds_185,
         i.marketdb_id, 
        (select name from %(marketdb)s.classification_active_int cl 
              where cl.axioma_id= i.marketdb_id
                and cl.revision_id=17
                and cl.from_dt <= :dt and :dt < cl.thru_dt) class_name
             
        from (
            select * from (
                select * from rms_issue s
                where s.from_dt <= :dt and :dt < s.thru_dt
                and s.rms_id=%(rmsid)d
                ) si left join DESCRIPTOR_EXPOSURE_%(currency)s d on d.sub_issue_id= si.issue_id || '11' and d.dt=:dt
            where 
                    d.sub_issue_id is null
                or (d.ds_100 is null or d.ds_102 is null 
                or d.ds_105 is null or d.ds_106 is null 
                or d.ds_107 is null or d.ds_108 is null 
                or d.ds_109 is null or d.ds_182 is null 
                or d.ds_183 is null or d.ds_184 is null 
                or d.ds_185 is null)
        ) v , 
        issue_map i
       where 
                  i.modeldb_id=v.issue_id
              and i.from_Dt <= :dt and :dt < i.thru_dt

    """ % {'marketdb':options.marketDBUser, 'rmsid':rmsid ,'currency':currency}
    cursor= modelDB.dbCursor
    errorCase = False

    while True:
        if dt > endDate:
            break
        if dt.isoweekday() in (6,7):
            dt = dt + datetime.timedelta(days=1)
            continue
        
        logging.info("Checking for descriptors on %s", dt)
        cursor.execute(query, dt=dt)
        cols=[i[0] for i in  cursor.description]
        results = cursor.fetchall()
        if len(results) > 0:
            errorCase = True
            print(','.join([i for i in cols]))
            for r in results:
                print(','.join([str(i) for i in r]))
        dt = dt + datetime.timedelta(days=1)

    ###modelDB.revertChanges()
    if errorCase:
        logging.info('Exiting with errors')
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()


import logging
import optparse
import subprocess
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

#------------------------------------------------------------------------------
if __name__ == '__main__':
    usage = "usage: %prog [options] <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("-C", action="store",
                             default='staging.config', dest="config",
                             help="which config to use")
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    
    modelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    riskModel = modelClass(modelDB, marketDB)
    date = Utilities.parseISODate(args[0])
    
    query="""
        select im.MODELDB_ID, ar.ADD_DT, ar.TRADING_COUNTRY 
                ,(select code from marketdb_global.classification_active_int a, marketdb_global.classification_ref b
                where a.classification_id=b.id and a.revision_id in (11,12) and a.thru_dt > sysdate
                        and ar.axioma_id=a.axioma_id) home_country
        from modeldb_global.ISSUE_MAP im,
        marketdb_global.asset_ref ar
        where im.MARKETDB_ID=ar.AXIOMA_ID
            and exists (select * from modeldb_global.RMS_ISSUE ri where ri.ISSUE_ID=im.MODELDB_ID and ri.RMS_ID=230)
            and not exists (select * from modeldb_global.descriptor_numeraire_usd ds where dt=:date_arg and
            ds.sub_issue_id= (im.modeldb_id || '11')
            and ds. ds_107 is not null)
            and im.from_dt <= :date_arg and im.thru_dt > :date_arg
    """
    print(query)
    checkParam = modelDB.dbCursor.execute(query, date_arg=date)
    results= modelDB.dbCursor.fetchall()
    countryList=set()
    for r in results:
        countryList=countryList.union([r[2]]).union([r[3]])
        logging.info('%s', r)

    exitStatus=0
    if len(countryList) > 0:
        rmgs=','.join(list(countryList))
    
        cmd="python3 transfer.py %s dates=%s sections=NumeraireDescriptorData rmgs=%s numeraire=USD -l log.config -f" % (options.config,str(date),rmgs)
        if not options.testOnly:
            print(cmd)
            obj=subprocess.Popen(cmd, shell=True, cwd='.')
            stat= obj.wait()
            logging.info( 'return status= %s',stat)
            print('------------------------')
            sys.stdout.flush()
            if stat != 0:
                exitStatus=1
        else:
            logging.info('Would execute %s', cmd)
            logging.info('Exit with error status')
            exitStatus=1

    else:
        logging.info('No descriptor transfer needs to be rerun today')
    modelDB.finalize()
    marketDB.finalize()
    sys.exit(exitStatus)

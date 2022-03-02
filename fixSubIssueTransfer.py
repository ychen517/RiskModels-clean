
import configparser
import datetime
import optparse
import logging
import sys
from riskmodels import Connections
from riskmodels import Utilities
from riskmodels import transfer

def finalize(conn):
    modelDB = conn.modelDB
    marketDB = conn.marketDB
    modelDB.finalize()
    marketDB.finalize()
    
def checkSubIssueData(connections_, sidList, date):
    modelDB = connections_.modelDB
    marketDB = connections_.marketDB
    modelDB.dbCursor.execute('alter session set nls_date_format="YYYY-MM-DD HH24:MI:SS"')
    finalquery="""
          select * from (
            select sd.sub_issue_id, sd.tso, 
            (select value from %(marketuser)s.asset_dim_tso_active_int t  where t.from_dt <= :dt and :dt < t.thru_dt
                and t.axioma_id = im.marketdb_id ) tso1,
            sd.tdv, 
            (select value from %(marketuser)s.asset_dim_tdv_active t where t.dt= :dt
                and t.axioma_id = im.marketdb_id ) tdv1,
            sd.ucp,
            (select value from %(marketuser)s.asset_dim_ucp_active t where t.dt= :dt
                and t.axioma_id = im.marketdb_id ) ucp1
            , im.marketdb_id
        from sub_issue_data_active sd, sub_issue si, issue_map im
            where 
                si.sub_id not like 'DCSH_%%' 
            and si.sub_id = sd.sub_issue_id
            and sd.dt = :dt
            and im.modeldb_id = substr(si.sub_id,1,10)
            and im.from_dt <= :dt and :dt <= im.thru_dt
            %(clause)s
        ) v where (nvl(v.ucp1,-1) <> nvl(v.ucp,-1)) or (nvl(v.tso,-1) <> nvl(v.tso1,-1)) or (nvl(v.tdv,-1) <> nvl(v.tdv1,-1))
           
        """

    if len(sidList) == 2:
        # this is a country
        query="""
            select rmg_id from risk_model_group rmg
            where rmg.mnemonic = :rmg_mnemonic 
        """
        modelDB.dbCursor.execute(query, rmg_mnemonic=sidList)
        results=modelDB.dbCursor.fetchall()
        if len(results) == 0:
           logging.fatal('No country %s present', sidList)
           sys.exit(1)

        rmgid=results[0][0]
        clause =  """
            and si.rmg_id = :rmgid 
        """
    else:
        subIssues=sidList.split(',')
        clause="""
            and :rmgid = :rmgid
            and si.sub_id in (%(sid)s)
        """ % {'sid': ','.join(["'%s'" % i for i in subIssues] )} 
        rmgid=0
        
    query = finalquery % {'clause':clause, 'marketuser':marketDB.dbConnection.username}
    logging.debug('%s %s %s', query, rmgid, date)
    logging.info('Executing query ....')
    modelDB.dbCursor.execute(query, rmgid=rmgid, dt=str(date))
    results=modelDB.dbCursor.fetchall()
    badids=[]
    for r in results:
        logging.info('%s', r)
        badids.append(r[0])
    return badids

def transferSubIssueData(configFileName, options, sidList, date):
    configFile_ = open(configFileName)
    config_ = configparser.ConfigParser()
    config_.read_file(configFile_)
    configFile_.close()

    config_.set('DEFAULT','dates',str(date))
    connections_ = Connections.createConnections(config_)

    config_.set('DEFAULT','sub-issue-ids',sidList)
    transfer.transferSubIssueData(config_, 'SubIssueData', connections_, options)

    return connections_

if __name__ == '__main__':
    usage = "usage: %prog [options] [[section:]option=value ...] date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    cmdlineParser.add_option("--verbose", "-v", action="store_true",
                             default=False, dest="verbose",
                             help="Extra debugging output")
    cmdlineParser.add_option("--sub-issue-ids", "-a", action="store",
                             default=None, dest="sidList",
                             help="list of axiomaids")

    Utilities.addModelAndDefaultCommandLine(cmdlineParser)

    (options, args) = cmdlineParser.parse_args()
    if len(args) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    #junk = Utilities.processModelAndDefaultCommandLine( options, cmdlineParser)

    Utilities.processDefaultCommandLine(options, cmdlineParser)

    sidList= options.sidList

    if options.sidList is None:
        logging.info('No SID specified. no work to do')
        sys.exit(0)

    configFileName = args[0]
    date = Utilities.parseISODate(args[1])

    if options.marketDBSID != options.marketDBSID:
        logging.fatal('Cannot run program against two different SIDs')
        sys.exit(1)

    logging.info('transfer data for %s', sidList)
    conn = transferSubIssueData(configFileName, options, sidList, date)
    badids = checkSubIssueData(conn, sidList, date)    
    finalize(conn)
    
    attempt = 0
    while len(badids) > 0 and attempt < 3:
       attempt = attempt + 1
       logging.info('need to transfer data for %s (attempt #%d)', badids, attempt)
       sidList = ','.join(badids)
       conn = transferSubIssueData(configFileName, options, sidList, date)
       badids = checkSubIssueData(conn, sidList, date)    
       finalize(conn)

    if len(badids) > 0:
       logging.error('Even after %d attempts, there are bad ids for %s', attempt, options.sidList)
       #sys.exit(1)
       sys.exit(0)

    
    logging.info('All sub-issue IDs have good data')
    sys.exit(0)

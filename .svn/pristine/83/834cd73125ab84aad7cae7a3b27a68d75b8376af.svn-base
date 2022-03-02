
import configparser
import datetime
import optparse
import logging
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

TABLES='rmg_proxy_return,sub_issue_data,sub_issue_return'

def main():
    usage = "usage: %prog [connection args] date"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", action="store",
                             default="1830", dest="numDays",
                             help="how many days to look back")
    cmdlineParser.add_option("-c", action="store",
                             default="US", dest="country",
                             help="countries of interest")
    cmdlineParser.add_option("-t", action="store",
                             default=TABLES, dest="tables",
                             help="tables of interest")
    cmdlineParser.add_option("-f", action="store_true",
                             default=False, dest="force",
                             help="force a run")
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) < 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)

    dt = Utilities.parseISODate(args_[0])
    startdt = dt + datetime.timedelta(days=-int(options_.numDays))

    modelDB = ModelDB.ModelDB(user=options_.modelDBUser, passwd=options_.modelDBPasswd,
                              sid=options_.modelDBSID)

    cursor = modelDB.dbCursor
    if dt.isoweekday() != 5 and not options_.force:
        logging.info('%s not a Friday, so ignore', dt)
        sys.exit(0)
    while startdt <= dt:
        if not startdt.isoweekday() in (6,7):
            logging.info("Working on %s", startdt )
            for tbl in options_.tables.split(','):
                for ctry in options_.country.split(','):
                    query="""select t.* from %s t where dt = :dt 
                        and exists (select * from sub_issue s where s.sub_id=t.sub_issue_id 
                        and s.rmg_id=(select rmg_id from risk_model_group where mnemonic = '%s'))""" % (tbl, ctry)
                    cursor.execute(query, dt=startdt)
                    results=cursor.fetchall()
                    logging.info('     ..... %s %-20s returned %10d rows', ctry, tbl, len(results))
        startdt = startdt + datetime.timedelta(days=1)
 
        
if __name__ == '__main__':
    main()


import logging
import numpy
import os
import numpy.ma as ma
import optparse
import sys
import pandas as pd
import pandas.io.sql as sql
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

#df = pd.read_sql(qry, conn.dbConnection)

QUERIES={
'RMG_MARKET_VOLATILITY': """select * from RMG_MARKET_VOLATILITY where DT=:dt and RMG_ID=(select RMG_ID from RISK_MODEL_GROUP rmg where
          mnemonic=:rmg)
          """,
'RMG_MARKET_PORTFOLIO': """select * from RMG_MARKET_PORTFOLIO where DT=:dt and RMG_ID=(select RMG_ID from RISK_MODEL_GROUP rmg where
          mnemonic=:rmg) 
          order by SUB_ISSUE_ID""",
'RMG_MARKET_RETURN': """select * from RMG_MARKET_RETURN where DT=:dt and RMG_ID=(select RMG_ID from RISK_MODEL_GROUP rmg where
          mnemonic=:rmg) """,
'SUB_ISSUE_DATA': """select s.* from SUB_ISSUE_DATA s 
          where DT=:dt and
          rmg_id = (select RMG_ID from RISK_MODEL_GROUP rmg where mnemonic=:rmg)
          order by s.sub_issue_id, rev_dt
          """,
'SUB_ISSUE_RETURN': """select s.* from SUB_ISSUE_RETURN s 
          where DT=:dt and
          rmg_id = (select RMG_ID from RISK_MODEL_GROUP rmg where mnemonic=:rmg)
          order by s.sub_issue_id, rev_dt
          """,
'DESCRIPTOR_LOCAL_CURRENCY':"""select d.* from DESCRIPTOR_LOCAL_CURRENCY d
        join SUB_ISSUE si on si.SUB_ID = d.SUB_ISSUE_ID and si.RMG_ID=
                (select RMG_ID from RISK_MODEL_GROUP rmg where mnemonic=:rmg)
        where DT=:dt 
        order by d.SUB_ISSUE_ID
""",
'DESCRIPTOR_NUMERAIRE_USD':""" select d.* from DESCRIPTOR_NUMERAIRE_USD d
        join SUB_ISSUE si on si.SUB_ID = d.SUB_ISSUE_ID and si.RMG_ID=
                (select RMG_ID from RISK_MODEL_GROUP rmg where mnemonic=:rmg)
        where DT=:dt 
        order by d.SUB_ISSUE_ID
""",
'RETURNS_TIMING_ADJUSTMENT':"""select * from RETURNS_TIMING_ADJUSTMENT where dt=:dt""",
'RMG_RETURNS_TIMING_ADJ':"""select * from RMG_RETURNS_TIMING_ADJ where dt=:dt""",
'REGION_RETURN':"""select * from REGION_RETURN where dt=:dt""",
}
NO_RMG_TABLES=['REGION_RETURN','RETURNS_TIMING_ADJUSTMENT','RMG_RETURNS_TIMING_ADJ']

def getTable(cursor,date,table,rmg=None):
    """ get data and store it in data frame and return to caller
    """
    qry=QUERIES[table]
    if rmg:
        cursor.execute(qry, dt=date, rmg=rmg)
    else:
        cursor.execute(qry, dt=date)
    names= [ x[0] for x in cursor.description]
    results=cursor.fetchall()
    df=pd.DataFrame(results, columns=names)
    return df
    
def main():

    usage = "usage: %prog [options] <YYYY-MM-DD> <rmg>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-d", "--directory", action="store",
                         default='.', dest="targetDir",
                         help="directory for output files")
    cmdlineParser.add_option("-t", "--tables", action="store",
                         default=None, dest="tables",
                         help="directory for output files")
    cmdlineParser.add_option("--prelim", action="store_true",
                         default=False, dest="prelim",
                         help="prelim version or not")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2:
        cmdlineParser.error("Incorrect number of arguments")

    Utilities.processDefaultCommandLine(options, cmdlineParser)
    logging.info('Start')
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                          passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    date = Utilities.parseISODate(args[0])
    rmgs = args[1]
    dirName=os.path.dirname(options.targetDir)
    if dirName and not os.path.exists(dirName):
        try:
            os.makedirs(dirName)
        except OSError:
            excstr=str(sys.exc_info()[1])
            if excstr.find('File exists') >= 0 and excstr.find(dirName) >= 0:
                logging.info('Error can be ignored - %s' % excstr)
            else:
                raise
    if options.prelim:
        prelim='-PRELIM'
    else:
        prelim='-FINAL'
    dtstr=str(date)[:10]
    for rmg in rmgs.split(','):
        for table in [t.upper() for t in options.tables.split(',')]:
            logging.info('Working on %s for %s %s %s', rmg,table, dtstr, prelim)
            if table in NO_RMG_TABLES:
                outFileName='%s/%s%s-%s.csv' % (options.targetDir.rstrip('/'), table, prelim, dtstr)
                df=getTable(modelDB.dbCursor, date, table) 
            else:
                outFileName='%s/%s-%s%s-%s.csv' % (options.targetDir.rstrip('/'), table, rmg, prelim, dtstr)
                df=getTable(modelDB.dbCursor, date, table, rmg) 
            outFile=open(outFileName,'w')
            outFile.write(df.to_csv())
            outFile.close()
    modelDB.finalize()
    logging.info('Done')


if __name__ == '__main__':
    main()


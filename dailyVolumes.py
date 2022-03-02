
import datetime
import sys

from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels.Utilities import parseISODate

OUTDIR='/data/intranet/development'

if __name__ == '__main__':
    #print sys.argv
    if len(sys.argv)>1 and sys.argv[1]:
        dt=parseISODate(sys.argv[1])
    else:
        dt=datetime.date.today()

    marketDB = MarketDB.MarketDB(user='marketdb',passwd='marketdbprod',sid='rmprod')
    modelDB = ModelDB.ModelDB(user='modeldb',passwd='modeldbprod',sid='rmprod')
    query = """select substr(im.MODELDB_ID,2) "AXIOMAID",
               --t.value "VOLUME",s.VALUE "TSO",
               round(100*t.value/s.value,8) "RATIO",
               t.dt "DATE" from ASSET_DIM_TDV t, INDEX_REVISION ir, INDEX_CONSTITUENT ic,
               modeldb.ISSUE_MAP im, ASSET_DIM_TSO s
               where ic.REVISION_ID=ir.ID and ir.DT=t.DT and ir.INDEX_ID=35 and ic.AXIOMA_ID=t.AXIOMA_ID
               and t.DT=:dt and im.MARKETDB_ID=t.AXIOMA_ID
               and s.AXIOMA_ID=t.AXIOMA_ID and s.FROM_DT<=t.DT and s.THRU_DT>t.dt and s.TRANS_FROM_DT<=sysdate and s.TRANS_THRU_DT>sysdate"""
    modelDB.dbCursor.execute("select max(dt) from RISK_MODEL_INSTANCE where dt<=:date_arg", date_arg=dt)
    r = modelDB.dbCursor.fetchall()
    if not r:
        raise ValueError('No date in RISK_MODEL_INSTANCE')
    lastdate = modelDB.oracleToDate(r[0][0])
    marketDB.dbCursor.execute(query, dt=lastdate)
    r = marketDB.dbCursor.fetchall()
    if r:
        outFile = open('%s/volume_%04d%02d%02d.txt' % (OUTDIR, lastdate.year, lastdate.month, lastdate.day), 'w')
        outFile.write('AXIOMAID|RATIO|DATE\n')
        for rec in r:
            outFile.write('%s|%.8f|%s\n' % (rec[0], rec[1], modelDB.oracleToDate(rec[2])))
        outFile.close()
    marketDB.finalize()
    modelDB.finalize()

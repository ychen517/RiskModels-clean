
from riskmodels import ModelDB

def syncMetaData(srcDb, tgtDb):
    # get target model IDs
    tgtDb.execute("""SELECT issue_id FROM issue""")
    tgtIDs = set([i[0] for i in tgtDb.fetchall()])
    # get target ca_merger survivor
    tgtDb.execute("""SELECT dt, ca_sequence, modeldb_id,
       new_marketdb_id, old_marketdb_id, share_ratio, cash_payment,
       currency_id, ref, rmg_id FROM ca_merger_survivor""")
    tgtMergers = set([i for i in tgtDb.fetchall()])
    tgtMergDict = dict([(i[:3], i[3:]) for i in tgtMergers])
    # get source ca_merger survivor
    srcDb.execute("""SELECT dt, ca_sequence, modeldb_id,
       new_marketdb_id, old_marketdb_id, share_ratio, cash_payment,
       currency_id, ref, rmg_id FROM ca_merger_survivor""")
    srcMergers = set([i for i in srcDb.fetchall() if i[2] in tgtIDs])
    srcMergDict = dict([(i[:3], i[3:]) for i in srcMergers])
    
    for missingKey in set(srcMergDict.keys()) - set(tgtMergDict.keys()):
        print('Inserting %s' % (missingKey,))
        missing = list(missingKey) + list(srcMergDict[missingKey])
        print(missing)
        tgtDb.execute("""INSERT INTO ca_merger_survivor
          (dt, ca_sequence, modeldb_id,
          new_marketdb_id, old_marketdb_id, share_ratio, cash_payment,
          currency_id, ref, rmg_id)
          VALUES(:dt, :cas, :mid, :nmkt, :omkt, :ratio, :cash,
          :ccy, :ref, :rmg)""",
                      dt=missing[0],
                      cas=missing[1],
                      mid=missing[2],
                      nmkt=missing[3],
                      omkt=missing[4],
                      ratio=missing[5],
                      cash=missing[6],
                      ccy=missing[7],
                      ref=missing[8],
                      rmg=missing[9])
    # get target ca_spin_off
    tgtDb.execute("""SELECT dt, ca_sequence, parent_id,
       child_id, share_ratio, implied_div,
       currency_id, ref, rmg_id FROM ca_spin_off""")
    tgtSpinOffs = set([i for i in tgtDb.fetchall()])
    # get source ca_spin_off
    srcDb.execute("""SELECT dt, ca_sequence, parent_id,
       child_id, share_ratio, implied_div,
       currency_id, ref, rmg_id FROM ca_spin_off""")
    srcSpinOffs = set([i for i in srcDb.fetchall()
                        if i[2] in tgtIDs])
    for missing in srcSpinOffs - tgtSpinOffs:
        print('Inserting %s' % (missing,))
        tgtDb.execute("""INSERT INTO ca_spin_off
          (dt, ca_sequence, parent_id,
          child_id, share_ratio, implied_div,
          currency_id, ref, rmg_id)
          VALUES(:dt, :cas, :pid, :cid, :ratio, :cash,
          :ccy, :ref, :rmg)""",
                      dt=missing[0],
                      cas=missing[1],
                      pid=missing[2],
                      cid=missing[3],
                      ratio=missing[4],
                      cash=missing[5],
                      ccy=missing[6],
                      ref=missing[7],
                      rmg=missing[8])
          
    
if __name__ == '__main__':
    glprod = ModelDB.ModelDB(user='modeldb_global', passwd='modeldb_global', sid='research')
    oldUS = ModelDB.ModelDB(user='modeldb', passwd='modeldbprod', sid='rmprod')

    syncMetaData(oldUS.dbCursor, glprod.dbCursor)

    glprod.revertChanges()
    #glprod.commitChanges()
    oldUS.revertChanges()
    glprod.finalize()
    oldUS.finalize()

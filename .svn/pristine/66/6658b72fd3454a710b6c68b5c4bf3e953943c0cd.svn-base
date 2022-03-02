
import datetime
import logging
import optparse
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities

SRC_ID=905

def addCurrencyAsset(rmgId, currency, marketDB, modelDB):
    """Add an issue for the given currency and a sub-issue for it
    int the given risk model group.
    Add axioma-id marketdb.
    """
    cashModelTemplate = 'DCSH_%s__'
    cashMarketTemplate = 'CSH_%s___'
    cashAxid = cashMarketTemplate % currency
    cashMarketID = 'CSH_%s' % currency
    currencyIssue = ModelDB.CANewIssue(
        cashModelTemplate % currency, cashAxid,
        'Cash asset for %s' % currency)
    currencySubIssue = ModelDB.SubIssue(string=currencyIssue.modeldb_id
                                        .getIDString() + '11')

    
    marketDB.dbCursor.execute("""SELECT from_dt, thru_dt, description, id
      FROM currency_ref WHERE code=:iso""", iso=currency)
    r = marketDB.dbCursor.fetchall()
    if len(r) != 1:
        logging.fatal('No MarketDB records for %s' % currency)
        return False
    (startDate, thruDate, description, currency_id) = r[0]
    startDate = startDate.date()
    thruDate = thruDate.date()
    print('Creating Axioma ID %s for %s with IDs %s' % (
        cashAxid, description, cashMarketID))

    # Create MarketDB records
    marketDB.dbCursor.execute("""INSERT INTO asset_ref
      (axioma_id, from_dt, thru_dt, src_id, ref)
      VALUES(:axid, :fromDt, :thruDt, :src, :ref)""",
                              axid=cashAxid, fromDt=startDate,
                              thruDt=thruDate, src=SRC_ID,
                              ref='Cash assets for %s' % currency)
    marketDB.dbCursor.execute("""INSERT INTO asset_dim_name
      (axioma_id, id, change_dt, change_del_flag, src_id, ref,
             rev_dt, rev_del_flag)
      VALUES(:axid, :nameArg, :changeDt, 'N', :src, :ref, systimestamp, 'N')""",
                              axid=cashAxid, changeDt=startDate,
                              src=SRC_ID, nameArg=description,
                              ref='Cash assets for %s' % currency)
    marketDB.dbCursor.execute("""INSERT INTO asset_dim_trading_currency
      (axioma_id, id, change_dt, change_del_flag, src_id, ref,
             rev_dt, rev_del_flag)
      VALUES(:axid, :idArg, :changeDt, 'N', :src, :ref, systimestamp, 'N')""",
                              axid=cashAxid, changeDt=startDate,
                              src=SRC_ID, idArg=currency_id,
                              ref='Cash asset for %s' % currency)
    if thruDate.year < 2999:
        marketDB.dbCursor.execute("""INSERT INTO asset_dim_trading_currency
          (axioma_id, id, change_dt, change_del_flag, src_id, ref,
                 rev_dt, rev_del_flag)
          VALUES(:axid, :idArg, :changeDt, 'Y', :src, :ref, systimestamp, 'N')""",
                                  axid=cashAxid, changeDt=thruDate,
                                  src=SRC_ID, idArg=currency_id,
                                  ref='Cash asset for %s' % currency)
    for t in ['asset_dim_cusip', 'asset_dim_isin', 'asset_dim_sedol',
              'asset_dim_ticker']:
        marketDB.dbCursor.execute("""INSERT INTO %(table)s
           (axioma_id, id, change_dt, change_del_flag, src_id, ref,
                  rev_dt, rev_del_flag)
           VALUES(:axid, :nameArg, :changeDt, 'N', :src, :ref,
            systimestamp, 'N')""" % {'table': t},
                              axid=cashAxid, changeDt=startDate,
                              src=SRC_ID, nameArg=cashMarketID,
                              ref='Cash assets for %s' % currency)
        if thruDate < datetime.date.today():
            marketDB.dbCursor.execute("""INSERT INTO %(table)s
              (axioma_id, id, change_dt, change_del_flag, src_id, ref,
                     rev_dt, rev_del_flag)
              VALUES(:axid, :nameArg, :changeDt, 'Y', :src, :ref,
               systimestamp, 'N')""" % {'table': t},
                              axid=cashAxid, changeDt=thruDate,
                              src=SRC_ID, nameArg=cashMarketID,
                              ref='Cash assets for %s' % currency)
    # Create modeldb issue
    modelDB.createNewIssue([currencyIssue], startDate)
    # Create modeldb sub-issue
    modelDB.dbCursor.execute("""SELECT rmg_id, from_dt, thru_dt
      FROM rmg_currency WHERE currency_code=:code""", code=currency)
    r = dict([(i[0], (i[1:])) for i in  modelDB.dbCursor.fetchall()])
    if len(r) != 1:
        if rmgId is None:
            print('More than one risk model group found for %s' % currency)
            return False
        elif rmgId not in r:
            print('More than one risk model group found for %s' \
                ' and %d is not among them: %s' % (currency, rmgId, list(r.keys())))
            return False
    else:
        if rmgId is not None and rmgId not in r:
            print('Provided risk model group (%d) does not match available' \
                ' group (%d)' % (rmgId, list(r.keys())[0]))
            return False
        else:
            rmgId = list(r.keys())[0]
    rmg = modelDB.getRiskModelGroup(rmgId)
    print('Adding cash asset to %s' % rmg)
    modelDB.createNewSubIssue([currencySubIssue], rmg, startDate)
    if thruDate < datetime.date.today():
        print('Terminating cash asset for %s on %s' % (currency, thruDate))
        modelDB.deactivateIssues([currencyIssue.modeldb_id], thruDate)
        modelDB.deactivateSubIssues([currencySubIssue], rmg, thruDate)
    return True

if __name__ == '__main__':
    usage = "usage: %prog [options] <currency ISO code> [<rmg-id>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options_, args_) = cmdlineParser.parse_args()
    if len(args_) not in (1,2):
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options_, cmdlineParser)
    
    modelDB_ = ModelDB.ModelDB(sid=options_.modelDBSID,
                               user=options_.modelDBUser,
                               passwd=options_.modelDBPasswd)
    marketDB_ = MarketDB.MarketDB(sid=options_.marketDBSID,
                                  user=options_.marketDBUser,
                                  passwd=options_.marketDBPasswd)

    currency_ = args_[0]
    if len(args_) > 1:
        rmgId_ = int(args_[1])
    else:
        rmgId_ = None
    assert(len(currency_) == 3)

    try: 
        success = addCurrencyAsset(rmgId_, currency_, marketDB_, modelDB_)
    except Exception as ex:
        logging.fatal('Exception during processing', exc_info=True)
        success = False
    
    if options_.testOnly or not success:
        logging.info('Reverting changes')
        marketDB_.revertChanges()
        modelDB_.revertChanges()
    else:
        marketDB_.commitChanges()
        modelDB_.commitChanges()
    
    modelDB_.finalize()
    marketDB_.finalize()

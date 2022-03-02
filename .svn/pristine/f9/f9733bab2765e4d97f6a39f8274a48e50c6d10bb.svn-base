
import logging
import optparse
import pymssql
from riskmodels import ModelDB
from riskmodels import Utilities

def getModelId(modelDB, mnemonic):
    modelDB.dbCursor.execute("""SELECT model_id, name FROM risk_model WHERE mnemonic=:mnemonic_arg""",
                             mnemonic_arg=mnemonic)
    return modelDB.dbCursor.fetchone()

def getModelFactors(modelDB, rmId):
    modelDB.dbCursor.execute("""SELECT f.Name, ft.Name, rms.serial_id, sf.sub_id FROM risk_model rm
        JOIN risk_model_serie rms ON rm.model_id=rms.rm_id
        JOIN rms_factor rf ON rms.serial_id=rf.rms_id
        JOIN factor f ON rf.factor_id=f.factor_id
        JOIN factor_type ft ON f.factor_type_id=ft.factor_type_id
        JOIN sub_factor sf ON sf.factor_id=f.factor_id
        WHERE rm.model_id=:rm_arg AND rms.distribute=1 ORDER BY rf.thru_dt DESC""",
                             rm_arg=rmId)
    factorList = modelDB.dbCursor.fetchall()
    # We can have multiple factors with the same rms_id and sub_factor sub_id. In those cases only keep
    # the first one we encounter, i.e. the most recent one
    subIdSet = set()
    finalList = list()
    for factor in factorList:
        if (factor[2], factor[3]) not in subIdSet:
            finalList.append(factor)
            subIdSet.add((factor[2], factor[3]))
    print(len(finalList), len(factorList))
    return sorted(finalList)
    
def createAxiomaDataId(cursor, category, name, vendorIdType, vendorId):
    cursor.execute("""truncate table #InstrumentData""")
    cursor.execute("""INSERT INTO #InstrumentData (AxiomaDataId, Action, CategoryEnum, Name, Lub, Lud,
        VendorDataIdTypeEnum, VendorDataInternalId) VALUES (-1, %s, %s, %s, 'MarketDataLoader',
        getdate(), %s, %s)""", ('I', category, name, vendorIdType, vendorId))
    cursor.execute("""exec dbo.InstrumentData_put""")
    cursor.execute("""SELECT AxiomaDataid FROM #InstrumentData""")
    return cursor.fetchone()[0]

def createOrUpdateXrefs(cursor, axid, targetXrefs):
    cursor.execute("""SELECT SecurityIdentifierType, SecurityIdentifier, FromDate, ToDate
       FROM InstrumentXref WHERE AxiomaDataId = %s""", axid)
    currentXrefs = cursor.fetchall()
    #print currentXrefs
    #print targetXrefs
    for tXref in targetXrefs:
        cXrefs = [cx for cx in currentXrefs if cx[0] == tXref[0]]
        if len(cXrefs) == 0:
            # Insert targetXref
            logging.info('Inserting into InstrumentXref for %d: %s', axid, tXref)
            cursor.execute("""INSERT INTO InstrumentXref (AxiomaDataId, SecurityIdentifierType,
                SecurityIdentifier, FromDate, ToDate, Lub, Lud) VALUES (%s, %s, %s,
                %s, %s, 'MarketDataLoader', getdate())""",
                        (axid, tXref[0], tXref[1], tXref[2], tXref[3]))
        else:
            # Compare and update first match, delete all the others
            for cXref in cXrefs[1:]:
                logging.info('Deleting from InstrumentXref of %d: %s', axid, cXref)
                cursor.execute("""DELETE FROM InstrumentXref
                    WHERE AxiomaDataId=%s AND SecurityIdentifierType=%s
                        AND SecurityIdentifier=%s AND FromDate=%s""",
                            (axid, cXref[0], cXref[1], cXref[2]))
            if tXref != cXrefs[0]:
                logging.info('Updating InstrumentXref of %d: %s -> %s', axid, cXrefs[0], tXref)
                cursor.execute("""UPDATE InstrumentXref SET SecurityIdentifier=%s,
                    FromDate=%s, ToDate=%s, Lub='MarketDataLoader', Lud=getdate()
                    WHERE AxiomaDataId=%s AND SecurityIdentifierType=%s
                        AND SecurityIdentifier=%s AND FromDate=%s""",
                            (tXref[1], tXref[2], tXref[3], axid, cXrefs[0][0],
                             cXrefs[0][1], cXrefs[0][2]))
            #print tXref, cXrefs, tXref == cXrefs[0]
    
def createOrUpdateModelAxiomaDataId(marketData, rmId, rmName):
    cur = marketData.cursor()
    cur.execute("""SELECT AxiomaDataId FROM InstrumentData
       WHERE CategoryEnum = 41 AND VendorDataIdTypeEnum = 'Name' AND VendorDataInternalId = %s""",
                rmName)
    modelAxid = cur.fetchone()
    if modelAxid is None:
        # Model not present yet, assign new AxiomaDataId in InstrumentData
        modelAxid = createAxiomaDataId(cur, 41, rmName, 'Name', rmName)
        logging.info('Created AxiomaDataId %d for model %s', modelAxid, rmName)
    else:
        modelAxid = modelAxid[0]

    targetXrefs = [('Name', rmName, '1900-01-01', '9999-12-31'),
                  ('DataId', 'RefType|Name=EquityFactorModel|{}'.format(rmName),
                   '1900-01-01', '9999-12-31')]
    createOrUpdateXrefs(cur, modelAxid, targetXrefs)
    cur.close()

def createOrUpdateFactorAxiomaDataId(marketData, factorInfo, rmName):
    factorName, factorType, rmsId, sfId = factorInfo
    internalId = 'SF_{}_{}'.format(rmsId, sfId)
    cur = marketData.cursor()
    cur.execute("""SELECT AxiomaDataId FROM InstrumentData
       WHERE CategoryEnum = 42 AND VendorDataIdTypeEnum = 'Name' AND VendorDataInternalId = %s""",
                internalId)
    factorAxid = cur.fetchone()
    if factorAxid is None:
        # Factor not present yet, assign new AxiomaDataId in InstrumentData
        internalName = '{} {} factor in {}: {}'.format(factorName, factorType, rmName, internalId)
        factorAxid = createAxiomaDataId(cur, 42, internalName, 'Name', internalId)
        logging.info('Created AxiomaDataId %d for factor %s', factorAxid, factorName)
    else:
        factorAxid = factorAxid[0]
    targetXrefs = [('Name', internalId, '1900-01-01', '9999-12-31'),
                  ('DataId', 'RefType|Name=EquityRiskFactor|{}'.format(internalId),
                   '1900-01-01', '9999-12-31')]
    createOrUpdateXrefs(cur, factorAxid, targetXrefs)

def setupInstrumentDataTempTable(marketData):
    marketData.cursor().execute("""create table [#InstrumentData] (
                AxiomaDataId              int          not null,
                CategoryEnum              int          not null,
                Name                      varchar(100) not null,
                CurrencyEnum              varchar(3)       null,
                DayCountEnum              varchar(10)      null,
                TermPeriodUnitEnum        varchar(1)       null,
                TermNumberOfUnit          int              null,
                ForwardTermPeriodUnitEnum varchar(1)       null,
                ForwardTermNumberOfUnit   int              null,
                StrikePrice               float            null,
                UnderlyingIdentifierType  varchar(100)     null,
                UnderlyingIdentifier      varchar(100)     null,
                OptionStyleEnum           varchar(100)     null,
                MaturityDate              date             null,
                IssueDate                 date             null,
                Coupon                    float            null,
                AmortTypeEnum             int              null,
                CountryEnum               varchar(2)       null,
                VendorDataLocationEnum    varchar(100)     null,
                VendorDataIdTypeEnum      varchar(100)     null,
                VendorDataInternalId      varchar(100)     null,
                IndustrySubSector         varchar(100)     null,
                IndustrySector            varchar(100)     null,
                CallPutEnum               varchar(1)       null,
                BusinessConventionsEnum   varchar(10)      null,
                VolSurfaceId              int              null,
                IsCurrentPrimaryListing   char(1)          null,
                Lud                       datetime         null,
                Lub                       varchar(100)     null,
                Action char(1) null
            )""")
    
def main():
    usage = "usage: %prog [options] <model mnemonic>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="don't change the database")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID,
                              user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketData = pymssql.connect(user='MarketDataLoader', password='mdl1234', host='prod-mac-mkt-db',
                         database='MarketData')
    rmId, rmName = getModelId(modelDB, args[0])
    setupInstrumentDataTempTable(marketData)
    logging.info('Processing model %s', rmName)
    createOrUpdateModelAxiomaDataId(marketData, rmId, rmName)
    factors = getModelFactors(modelDB, rmId)
    for factor in factors:
        logging.info('Processing factor %s (%d, %d)', factor[0], factor[2], factor[3])
        createOrUpdateFactorAxiomaDataId(marketData, factor, rmName)
    modelDB.revertChanges()
    modelDB.finalize()
    if options.testOnly:
        marketData.rollback()
    else:
        marketData.commit()
    marketData.close()

if __name__ == '__main__':
    main()

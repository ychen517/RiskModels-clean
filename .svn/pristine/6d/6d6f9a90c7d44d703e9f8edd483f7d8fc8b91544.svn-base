import numpy
import time
try:
    import numpy.ma as ma
except:
    import numpy.core.ma as ma
from collections import defaultdict
import datetime
import logging
import math
from math import sqrt
import optparse
import configparser
import struct
import os
import io
from marketdb import MarketDB
from marketdb import MarketID
import riskmodels
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import ModelDB
from riskmodels.ModelDB import SubIssue
from riskmodels import ModelID
from riskmodels import AssetProcessor
from riskmodels import Utilities
from riskmodels import MFM
from riskmodels.wombat import *
from riskmodels.writeFlatFiles import getWeekdaysBack, loadAllClassificationConstituents, buildESTUWeights, \
    checkCompositesForCompleteness, recomputeSpecificReturns, getExtractExcludeClassifications, CNH_START_DATE, getFixedFactorNames

NULL = ''
MAJOR_VERSION = 2
MINOR_VERSION = 1

REDUCED_CURRENCY_DATE=datetime.date(2014,3,1)

# ALL_IDS contains the IDs for the Master file, along with the length of each.
ALL_IDS = dict([('Ticker Map',(1,0, 'asset_dim_ticker', False)),
                ('Cusip Map',(2,9, 'asset_dim_cusip', False)),
                ('8 Digit Cusip Map',(3,8, None, False)),
                ('SEDOL Map',(4,7, 'asset_dim_sedol', True)),
                ('6 Digit SEDOL Map',(5,6, None, True)),
                ('Isin Map',(6,12, 'asset_dim_isin', False)),
                ('ISSUER',(7,0, 'asset_dim_name', False)),
                ('Currency',(8,3, 'asset_dim_trading_currency', False)),
                ('Country',(9,2,None, False)),
                ('Company',(10,9,'asset_dim_company',False)),
                ('Company Name',(11,0,None,False)),
                ('Asset Type', (12,0, 'Axioma Asset Type', False)),
                ('Asset Class', (13,0, 'Axioma Asset Type', False)),
                ('Asset Subclass', (14,0, 'Axioma Asset Type', False)),
# we need the asset type code information for the RMM
                ('Asset Type Code', (15,0,None, False)),
                ('MIC', (16,0,'MIC', False)),
                ])

V4_IDS = dict([('Hard Clone',(17,9, 'issue_exposure_linkage', False)),
               ('Force Coint',(18,9, 'issue_exposure_linkage', False)),
               ('No Coint',(19,9, 'issue_exposure_linkage', False)),
                ('DR to Root',(20,9, 'asset_dim_root_id', False)),
               ])

GICS_IDS=dict([('GICS Sector', (21,0, 'GICS', False)),
                ('GICS Industry Group', (22,0, 'GICS', False)),
                ('GICS Industry', (23,0, 'GICS', False)),
                ('GICS SubIndustry', (24,0, 'GICS', False)),
               ])

ASSET_TYPE={'Asset Type':1,'Asset Class':2,'Asset Subclass':3}

GICS_TYPE={'GICS Sector':1,'GICS Industry Group':2,'GICS Industry':3, 'GICS SubIndustry':4}

NUM_IDS = len(ALL_IDS)
assert(len(set(val[0] for val in ALL_IDS.values())) == NUM_IDS)

#RMM_ASSET_FIELDS=['isc_score_legacy','isc_adv_score','isc_ipo_score','isc_ret_score']
RMM_ASSET_FIELDS=['isc_adv_score','isc_ipo_score','isc_ret_score']
# for now, RMM_MODEL_FIELDS data comes out of the loadDescriptor call via the Asset Processor
RMM_MODEL_FIELDS=['market_sensitivity_104w']

RMM_ASSET_FIELDS_DICT={}
RMM_MODEL_FIELDS_DICT={}

for fld in RMM_ASSET_FIELDS:
    RMM_ASSET_FIELDS_DICT[fld] = -1

for fld in RMM_MODEL_FIELDS:
    RMM_MODEL_FIELDS_DICT[fld] = -1

#assert(len(set([idnum for (idnum, length, table, pad) in ALL_IDS.itervalues()])) == NUM_IDS)
NUM_RESERVED = 25

def MAclip(a, lb, ub):
    """Clip a masked array just like Numeric.clip() does for a regular array.
    """
    a = ma.where(a < lb, lb, a)
    a = ma.where(a > ub, ub, a)
    return a

def numOrNull(val, prec=16):
    """Formats a number with sixteen digits of precision or returns NULL
    if the value is masked.
    """
    if val is ma.masked or val is None or math.isnan(val):
        return NULL
    return ('%%.%dg' % prec) % val

def fixedLength(val, length):
    return bytes((val + length * '\0')[:length], 'utf-8')

def variableLength(val):
    val = bytes(val, 'utf-8')
    return bytes([len(val)]) + val

def zeroPad(val, length):
    val = bytes(val, 'utf-8')
    if val == b'':
        return val
    if len(val) >= length:
        return val[:length]
    zeros = length*'0'
    return zeros[:length - len(val)] + val

def binaryDouble(val):
    if val is None or val is ma.masked:
        # Python up to and including 2.4 can handle NaN only in native
        # byte-order. As a work-around, use Q to do the reordering.
        try:
            return struct.pack('>Q', struct.unpack('Q',struct.pack('d', float('NaN')))[0])
        except ValueError:
            return '\x7f\xf8\x00\x00\x00\x00\x00\x00'
    return struct.pack('>d', val)

def encodeDate(date):
    baseDate = datetime.date(1970,1,1)
    myDate = datetime.date(date.year, date.month, date.day)
    if (myDate < baseDate):
        myDate = baseDate
    delta = myDate - baseDate
    if delta.days > 0xffff:
        delta = datetime.timedelta(0xffff)
    return struct.pack('!H', delta.days)

def getNumFactors(riskModel):
    """Return the maximum number of non-zero factor exposures per asset
    in the given model.
    """
    localStructureFactors = [f for f in riskModel.localStructureFactors if f not in riskModel.styles]
    num = len(riskModel.styles) + len(riskModel.blind) \
           + min(1, len(riskModel.industries)) \
           + len(localStructureFactors)
    if riskModel.intercept is not None:
        num += 1
    if isinstance(riskModel, MFM.MacroeconomicModel):
        num += len(riskModel.macro_core) + len(riskModel.macro_market_traded) \
            + len(riskModel.macro_equity) + len(riskModel.macro_sectors)
    if riskModel.isProjectionModel():
        num += len(riskModel.macros)
    if riskModel.isRegionalModel():
        if riskModel.isStatModel():
            num += min(1, len(riskModel.currencies))
        else:
            num += min(1, len(riskModel.currencies)) \
                    + min(1, len(riskModel.countries))
    return num

def computeIncomeGrowth(modelDB, date, subIssues, sidCurrencies, marketDB):
    startDate = date - datetime.timedelta(2.5*366)
    ni_q = modelDB.getFundamentalCurrencyItem(
        'ni_qtr', startDate, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust=None)
    ni_y = modelDB.getFundamentalCurrencyItem(
        'ni_ann', startDate, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust=None)
    thisYearNI = Matrices.allMasked((len(subIssues),))
    prevYearNI = Matrices.allMasked((len(subIssues),))
    for i in range(len(subIssues)):
        if len(ni_q[i]) >= 8:
            # Use last eight quarters to build yearly net income
            endDate = ni_q[i][-1][0]
            values = [val for (dt, val, ccy) in ni_q[i]
                      if dt >= endDate - datetime.timedelta(23 * 31)]
            if len(values) == 8:
                thisYearNI[i] = sum(values[4:8])
                prevYearNI[i] = sum(values[0:4])
        if thisYearNI[i] is ma.masked and len(ni_y[i]) >= 2:
            # Use last two yearly filings
            endDate = ni_y[i][-1][0]
            values = [val for (dt, val, ccy) in ni_y[i]
                      if dt >= endDate - datetime.timedelta(23 * 31)]
            if len(values) == 2:
                thisYearNI[i] = values[1]
                prevYearNI[i] = values[0]
    growth_inc = (thisYearNI / prevYearNI - 1.0) * 100.0
    return growth_inc

def computeEarningsGrowth(modelDB, date, subIssues, sidCurrencies, marketDB):
    startDate = date - datetime.timedelta(2.5*366)
    aepsfx_q = modelDB.getFundamentalCurrencyItem(
        'epsfx_qtr', startDate, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust='divide')
    aepsfx_y = modelDB.getFundamentalCurrencyItem(
        'epsfx_ann', startDate, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust='divide')
    thisYearEarn = Matrices.allMasked((len(subIssues),))
    prevYearEarn = Matrices.allMasked((len(subIssues),))
    for i in range(len(subIssues)):
        if len(aepsfx_q[i]) >= 8:
            # Use last eight quarters to build yearly earnings
            endDate = aepsfx_q[i][-1][0]
            values = [val for (dt, val, ccy) in aepsfx_q[i]
                      if dt >= endDate - datetime.timedelta(23 * 31)]
            if len(values) == 8:
                thisYearEarn[i] = sum(values[4:8])
                prevYearEarn[i] = sum(values[0:4])
        if thisYearEarn[i] is ma.masked and len(aepsfx_y[i]) >= 2:
            # Use last two yearly filings
            endDate = aepsfx_y[i][-1][0]
            values = [val for (dt, val, ccy) in aepsfx_y[i]
                      if dt >= endDate - datetime.timedelta(23 * 31)]
            if len(values) == 2:
                thisYearEarn[i] = values[1]
                prevYearEarn[i] = values[0]
    growth_earn = (thisYearEarn / prevYearEarn - 1.0) * 100.0
    return growth_earn

def getLatestSales(modelDB, date, subIssues, sidCurrencies, marketDB):
    twoYearsAgo = date - datetime.timedelta(2*365)
    quarterlySales = modelDB.getFundamentalCurrencyItem(
        'sale_qtr', twoYearsAgo, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust=None)
    annualSales = modelDB.getFundamentalCurrencyItem(
        'sale_ann', twoYearsAgo, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust=None)
    latestSales = Matrices.allMasked((len(subIssues),))
    for i in range(len(subIssues)):
        endDate = None
        if len(quarterlySales[i]) >= 4:
            endDate = quarterlySales[i][-1][0]
            values = [val for (dt, val, ccy) in quarterlySales[i]
                      if dt >= endDate - datetime.timedelta(11 * 31)]
            if len(values) == 4:
                latestSales[i] = sum(values)
        if len(annualSales[i]) > 0 and (latestSales[i] is ma.masked
                                     or endDate < annualSales[i][-1][0]):
            latestSales[i] = annualSales[i][-1][1]
    return latestSales * 1e6

def getDividendsPerShare(modelDB, date, subIssues, sidCurrencies, marketDB):
    twoYearsAgo = date - datetime.timedelta(2*365)
    adps_q = modelDB.getFundamentalCurrencyItem(
        'dps_qtr', twoYearsAgo, date, subIssues, date,
        marketDB, convertTo=sidCurrencies, splitAdjust='divide')
    latestADPS_y = Matrices.allMasked((len(subIssues),))
    for i in range(len(subIssues)):
        # sum the last four quarterly filings within a year
        if len(adps_q[i]) > 0:
            fileDate = adps_q[i][-1][0]
            valueSum = 0
            prevYear = fileDate - datetime.timedelta(11*31)
            for (d, value, ccy) in adps_q[i][-4:]:
                if d >= prevYear:
                    valueSum += value
            latestADPS_y[i] = valueSum
    return latestADPS_y


def combineMatchingPairs(list1, list2, combineFunction):
    combined = list()
    for (l1, l2) in zip(list1, list2):
        myList = list()
        combined.append(myList)
        iter1 = iter(l1)
        iter2 = iter(l2)
        try:
            val1 = next(iter1)
            val2 = next(iter2)
            while True:
                if val1[0] == val2[0]:
                    # Match, use combineFunction to merge the two values
                    combinedVal = combineFunction(val1[1], val2[1])
                    if combinedVal is not None:
                        myList.append((val1[0], combinedVal, val1[2]))
                    val1 = next(iter1)
                    val2 = next(iter2)
                elif val1[0] < val2[0]:
                    val1 = next(iter1)
                else:
                    val2 = next(iter2)
        except StopIteration:
                pass
    return combined

def getLatestDebtToEquity(modelDB, date, subIssues, sidCurrencies, marketDB):
    """Take the ratio of total debt to common equity from the latest filing,
    quarterly or annual, where both numbers are available.
    """
    twoYearsAgo = date - datetime.timedelta(2*365)
    debt_q = modelDB.getQuarterlyTotalDebt(
        twoYearsAgo, date, subIssues, date, marketDB, sidCurrencies)
    debt_y = modelDB.getAnnualTotalDebt(
        twoYearsAgo, date, subIssues, date, marketDB, sidCurrencies)
    ce_q = modelDB.getFundamentalCurrencyItem(
        'ce_qtr', twoYearsAgo, date, subIssues, date, marketDB,
        convertTo=sidCurrencies, splitAdjust=None)
    ce_y = modelDB.getFundamentalCurrencyItem(
        'ce_ann', twoYearsAgo, date, subIssues, date, marketDB,
        convertTo=sidCurrencies, splitAdjust=None)
    #Match the lists and extract the latest
    def ratio(x,y):
        if y == 0.0:
            return None
        return x / y
    d2e_q = combineMatchingPairs(debt_q, ce_q, ratio)
    d2e_y = combineMatchingPairs(debt_y, ce_y, ratio)
    assert('Not Implemented')
    # Merge the two lists
    d2e = [i + j for (i,j) in zip(d2e_q, d2e_y)]
    for i in d2e:
        i.sort()
    return Utilities.extractLatestValue(d2e)
    
def createVersionFile(options, dbName, modelDB):
    out = open('%s/Db%s_version.binary' % (options.targetDir, dbName), 'wb')
    out.write(struct.pack('!ii', MAJOR_VERSION, MINOR_VERSION ))
    out.write(bytes('#CreationTimestamp: %s\n' %
                    modelDB.revDateTime.strftime('%Y-%m-%d %H:%M:%S'), 'utf-8'))
    out.close()

def createPartialModelIndicatorFile(options, dbName, riskModel):
    with open('%s/Db%s_partial.binary' % (options.targetDir, dbName), 'wb') as out:
        out.write(bytes('%s|%s' % (riskModel.name, riskModel.extractName), 'utf-8'))

def createCurrencyConvFile(options, fileType,marketDB):
    out = open('%s/Db%s.binary' % (options.targetDir, fileType), 'wb')
    
    marketDB.dbCursor.execute("""
      select (select code from currency_ref where id = old_id) as old_code, 
       (select code from currency_ref where id = new_id) as new_code, 
       dt, old_to_new_rate rate
       from currency_mod_convergence
       order by dt, old_code
       """)
    result = marketDB.dbCursor.fetchall()
    for row in result:
        out.write(fixedLength(row[0], 3))
        out.write(fixedLength(row[1], 3))
        out.write(encodeDate(row[2]))
        out.write(binaryDouble(row[3]))  
    out.close()
        
def writeMeta(modelDB, marketDB, options, date):
    TWO2MODELS=['AXAPxJP22-MH', 'AXAPxJP22-MH-S', 'AXAPxJP22-SH', 'AXAPxJP22-SH-S', 'AXAP22-MH', 'AXAP22-MH-S', 'AXAP22-SH', 'AXAP22-SH-S',
                'AXWW22-MH', 'AXWW22-MH-S', 'AXWW22-SH', 'AXWW22-SH-S', ]
    createVersionFile(options, 'Meta', modelDB)
    createCurrencyConvFile(options, 'Meta_CurrencyConv',marketDB)
    out = open('%s/Meta_Region' % options.targetDir, 'w')
    # regions file now contains risk models, so that DbMaster and DbAsset
    # are per model as is done in the flat-files.
    # Add US, UK, and TW regions for compatibility with 2.0
    allModels = modelDB.getRiskModelsForExtraction(date)
    for i in allModels:
        out.write('%s|%s\n' % (i.name, i.description))
    # TODO: add all RMGs for which Goldman have Shortfall models
    # (currently only US)
    out.write('US|US Model Group\n')
    out.write('UK|UK Model Group\n')
    out.write('TW|Taiwan Model Group\n')
    out.write('WW|Global Group\n')
    out.close()
    # write models
    out = open('%s/Meta_Model' % options.targetDir, 'w')
    # include all models in DbMeta - except the 2.2 models if the flag is turned on
    for i in allModels:
        try:
            model = riskmodels.getModelByVersion(i.rm_id, i.revision)
            model = model(modelDB, marketDB)
            model.setFactorsForDate(date, modelDB)
            if 'exclude22Models' in options.getFieldNames() and options.exclude22Models and i.mnemonic in TWO2MODELS:
                logging.info('Ignoring %s model', i.mnemonic)
                continue
            out.write('%s|%s|%d|%s\n' % (
                    i.name, i.description, getNumFactors(model), i.region))
        except KeyError:
            logging.error("No model class available for model ID %s, revision %s", i.rm_id, i.revision)
    out.close()
    #
    # write composite family information
    #
    out = open('%s/Meta_Composites' % options.targetDir, 'w')
    etfFamilies = marketDB.getETFFamilies()
    for compFamily in etfFamilies:
        if compFamily.distribute:
            out.write('%s|%s\n' % (compFamily.name, compFamily.description))
    out.close()
    #
    # write model<->composite family mapping
    #
    modelCompositePairs = modelDB.getCompositeFamilyModelMap()
    #modelIDNameMap = dict((i.rm_id, i.name) for i in allModels)
    if 'exclude22Models' in options.getFieldNames() and options.exclude22Models:
        modelIDNameMap = dict((i.rm_id, i.name) for i in allModels if i.mnemonic not in TWO2MODELS)
    else:
        modelIDNameMap = dict((i.rm_id, i.name) for i in allModels)
    compFamilyIDNameMap = dict((i.id, i.name) for i in etfFamilies
                               if i.distribute)
    out = open('%s/Meta_CompositeFamilyModelMap' % options.targetDir, 'w')
    for (modelID, compFamilyID) in modelCompositePairs:
        if modelID in modelIDNameMap and compFamilyID in compFamilyIDNameMap:
            out.write('%s|%s\n' % (compFamilyIDNameMap[compFamilyID],
                                   modelIDNameMap[modelID]))
    out.close()
    #
    # write index family information
    #
    out = open('%s/Meta_IndexFamily' % options.targetDir, 'w')
    distributedIdxFamilies = list()
    for idxFamily in marketDB.getIndexFamilies():
        if idxFamily.distribute:
            out.write('%s|%s\n' % (idxFamily.name, idxFamily.description))
            distributedIdxFamilies.append(idxFamily)
    out.close()
    #
    # write index information for each distributed family
    #
    out = open('%s/Meta_Index' % options.targetDir, 'w')
    for (famNum, idxFamily) in enumerate(distributedIdxFamilies):
        indexes = marketDB.getAllIndexFamilyIndices(idxFamily)
        # Merge names with multiple records into one
        merged = dict()
        for index in indexes:
            if index.name not in merged:
                merged[index.name] = index
            else:
                merged[index.name].from_dt = min(merged[index.name].from_dt,
                                                 index.from_dt)
                merged[index.name].thru_dt = max(merged[index.name].thru_dt,
                                                 index.thru_dt)
        indexes = list(merged.values())
        for index in indexes:
            out.write('%d|%s|%s|%s|%s\n' % (
                    famNum + 1, index.name, index.description,
                    index.from_dt, index.thru_dt))
    out.close()
    #
    # write industry classification information
    # We don't have distribute flag for classification_member
    # so this hard-codes GICS and TSEC for now.
    #
    out = open('%s/Meta_Classification' % options.targetDir, 'w')
    out.write('%s|%s\n' % ('GICS', 'GICS Industry Classification'))
    out.write('%s|%s\n' % ('TSEC', 'TSEC Industry Classification'))
    out.close()


    # 
    # write out name and code information for asset types
    #

    out = open('%s/Asset_Type_Codes' % options.targetDir, 'w')
    # find all the code to name mappings for the various axioma asset types
    marketDB.dbCursor.execute("""
                select code, name from classification_ref c
                   where revision_id= (
                      select id from classification_revision crev 
                         where crev.from_dt <= :dt and :dt < crev.thru_dt
                         and crev.member_id=(select id from classification_member m where m.name='Axioma Asset Type')
                   )
        """, dt=date)
    for resultrow in marketDB.dbCursor.fetchall():
        atypecode, atypename = resultrow
        out.write('%s|%s\n' % (atypename, atypecode))
    out.close()

def remapNurseryAssets(modelDB, marketDB, model, date, subIssues, subIssueRMGDict):
    if not hasattr(model, 'nurseryRMGs') or len(model.nurseryRMGs) == 0:
        return

    # Load in new asset processor
    assetData = AssetProcessor.process_asset_information(
            date, list(subIssues), model.rmg, modelDB, marketDB,
            checkHomeCountry=(len(model.rmg) > 1),
            legacyDates=model.legacyMCapDates,
            numeraire_id=model.numeraire.currency_id,
            forceRun=model.forceRun)

    # Compare nursery assets assigned country with the home or exposure country
    assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
            assetData.rmgAssetMap.items() for sid in ids])
    for sid in assetData.universe:
        if assetRMGMap[sid] != subIssueRMGDict[sid].rmg_id:
            assetRMG = modelDB.getRiskModelGroup(assetRMGMap[sid])
            if subIssueRMGDict[sid] in model.nurseryRMGs:
                logging.debug('Asset %s trades on nursery: %s, mapped to %s for exposures',
                        sid.getSubIDString(), subIssueRMGDict[sid].mnemonic, assetRMG.mnemonic)
    return

def buildMasterAssetList(modelDB, marketDB, model, options, date):
    # Only write assets that have a specific risk
    model.setFactorsForDate(date, modelDB)
    issues = set()
    subIssues = set()
    rmi = model.getRiskModelInstance(date, modelDB)
    excludes = modelDB.getProductExcludedSubIssues(date)
    if options.preliminary or not rmi.is_final:
        excludes.extend(modelDB.getDRSubIssues(rmi))
    # map issues to their RMG
    subIssueRMGDict = dict(modelDB.getSubIssueRiskModelGroupPairs(date))

    if rmi != None and (rmi.has_risks or options.allowPartialModels):
        universe = modelDB.getRiskModelInstanceUniverse(rmi)
        subIssues = set(universe) - set(excludes)

        # if it is a supplemental data case and there are only some countries, 
        # ensure that only those RMGs are being included
        if hasattr(model, 'supplementalData') and model.supplementalData and model.includeCountries:
            rmgids = [cc[0] for cc in model.includeCountries]
            subIssues=set([si for si in subIssues if subIssueRMGDict[si].rmg_id in rmgids])

        issues = set([sid.getModelID() for sid in subIssues])
    if len(issues) == 0:
        logging.critical("No issues for %s on %s ", model.name, date)
        raise ValueError('No issues in risk model')
    logging.debug('got model assets')
    allIssues = dict([(id, (from_dt, thru_dt)) for (id, from_dt, thru_dt)
                      in modelDB.getAllIssues()
                      if id in issues])
    logging.debug('got asset from/thru info')
    for e in excludes:
        subIssueRMGDict.pop(e, None)
    issueRMGDict = dict([(si.getModelID(), rmg.rmg_id)
                         for (si, rmg) in subIssueRMGDict.items()])
    remapNurseryAssets(modelDB, marketDB, model, date, subIssues, subIssueRMGDict)
    logging.debug('got RMG mappings')

    # find cash assets for active RMGs and add to master
    modelDB.dbCursor.execute("""SELECT DISTINCT rmg.rmg_id, si.sub_id,
      ie.from_dt, ie.thru_dt
    FROM risk_model_group rmg JOIN rmg_model_map rmm ON rmm.rmg_id=rmg.rmg_id
       JOIN sub_issue si ON si.rmg_id=rmg.rmg_id
       JOIN issue ie ON si.issue_id=ie.issue_id
    WHERE si.sub_id like 'DCSH/_%' ESCAPE '/'
       AND si.from_dt <= :dt AND si.thru_dt > :dt
       AND ie.from_dt <= :dt AND ie.thru_dt > :dt
       AND rmm.from_dt <= :dt AND rmm.thru_dt > :dt
    """, dt=date)

    # keep track of the supplemental data countries if we need to and only include those cash assets
    if hasattr(model, 'supplementalData') and model.supplementalData and model.includeCountries:
        rmgids = [cc[0] for cc in model.includeCountries]
    for (rmg_id, sid, fromDt, thruDt) in modelDB.dbCursor.fetchall():
        sid = ModelDB.SubIssue(string=sid)
        if hasattr(model,'supplementalData') and model.supplementalData and model.includeCountries and subIssueRMGDict[sid].rmg_id not in rmgids:
            continue
        if sid not in excludes:
            mid = sid.getModelID()
            subIssues.add(sid)
            issues.add(mid)
            issueRMGDict[mid] = rmg_id
            allIssues[mid] = (fromDt.date(), thruDt.date())
    subIssueFromDates = dict([(si, allIssues[si.getModelID()][0])
                              for si in subIssues])
    logging.debug('added cash assets')
    return (allIssues, issueRMGDict, subIssues, subIssueFromDates)

def writeMasterFile(modelDB, marketDB, model, options, date):
    logging.debug('build asset list')
    (allIssues, issueRMGDict, subIssues, subIssueFromDates) = \
        buildMasterAssetList(modelDB, marketDB, model, options, date)
    # write asset master
    writeMaster('DbMaster', model.extractName, modelDB, date, marketDB,
                allIssues, issueRMGDict, options)
    # write Goldman database(s) - US 
    if model.mnemonic == 'AXUS3-MH':
        writeGSDerby(marketDB, modelDB, 'US', date, subIssues,
                     "/axioma/operations/daily/Goldman/GSTCM%04d%02d%02d-WORLD.csv" % (date.year, date.month, date.day), 
                     options)
    
    # write Goldman database(s) - WW 
    if model.mnemonic == 'AXWW21-MH':
        writeGSDerby(marketDB, modelDB, 'WW', date, subIssues,
                     "/axioma/operations/daily/Goldman/GSTCM%04d%02d%02d-WORLD.csv" % (date.year, date.month, date.day), 
                     options)
    
    # write currencies
    logging.debug('write currencies')
    currencyProvider = MarketDB.CurrencyProvider(marketDB, 10, None,
                                                 maxNumConverters=120)
    writeMasterCurrency(modelDB, marketDB, model, options, date,
                        currencyProvider)
    createVersionFile(options, 'Master-%s' % model.extractName, modelDB)
    return (subIssues, issueRMGDict, subIssueFromDates)

def writeAssetFile(modelDB, marketDB, model, options, date, assetInfo):
    if assetInfo is None:
        (allIssues, issueRMGDict, subIssues, subIssueFromDates) =\
            buildMasterAssetList(modelDB, marketDB, model, options, date)
    else:
        (subIssues, issueRMGDict, subIssueFromDates) = assetInfo
    
    # may need to hack CNH in here to add to Asset File
    if date >= CNH_START_DATE:
        cnh=ModelDB.SubIssue(string='DCSH_CNH__11')
        subIssues=subIssues.union(set([cnh]))
        subIssueFromDates[cnh]= subIssueFromDates[[s for s in subIssues if s.getSubIDString().find('DCSH_CNY')==0][0]]

    # write asset data for each risk model group
    logging.info('writeAssetFile %d assets', len(subIssues))
    createVersionFile(options, 'Asset-%s' % model.extractName, modelDB)
    outAsset = open('%s/Asset-%s_Asset' % (options.targetDir, model.extractName), 'w')
    createVersionFile(options, 'Fundamental-%s' % model.extractName, modelDB)
    outFundamental = open('%s/Fundamental-%s_Fundamental' % (options.targetDir, model.extractName), 'w')
    
    # Build mapping from RMG to subissues
    rmgIdMap = dict([(rmg.rmg_id, rmg) for rmg in model.rmg])
    rmgSubIssues = dict([(rmg, set()) for rmg in model.rmg])
    riskModelGroups = list(model.rmg)
    addedList = []
    runningList = []
    for sid in subIssues:
        issueRMG_ID = issueRMGDict[sid.getModelID()]

        # if rmg not in rmgIdMap, then it is from a non-model rmg.
        # In that case add that rmg to riskModelGroups and rmgIdMap
        if issueRMG_ID not in rmgIdMap:
            rmg = modelDB.getRiskModelGroup(issueRMG_ID)
            logging.info('Market %s not in list, adding', rmg.mnemonic)
            rmgIdMap[issueRMG_ID] = rmg
            rmgSubIssues[rmg] = set()
            riskModelGroups.append(rmg)
            addedList.append(rmg)
        else:
            rmg = rmgIdMap[issueRMG_ID]

        # Report on status of tracked or nursery assets
        if sid in options.trackList:
            logging.info('Tracked asset: %s, assigned to market: %s',
                    sid.getSubIDString(), rmg.mnemonic)

        if hasattr(model, 'nurseryRMGs') and (rmg in model.nurseryRMGs):
            logging.debug('Asset %s trades on nursery market: %s',
                    sid.getSubIDString(), rmg.mnemonic)
        if rmg in addedList:
            logging.debug('Asset %s trades on extra-model market: %s',
                    sid.getSubIDString(), rmg.mnemonic)

        # Add asset to RMG to subissue mapping
        rmgSubIssues[rmg].add(sid)
        runningList.append(sid)

    logging.info('List after market processing: %d', len(runningList))
    currencyProvider = MarketDB.CurrencyProvider(marketDB, 10, None, maxNumConverters=120)
    if len(model.rmg) == 1 and model.rmg[0].mnemonic in ('US', 'CA') \
            and date >= datetime.date(2008, 1, 1):
        publishFundamentalRMGs = model.rmg
    else:
        publishFundamentalRMGs = []
    
    # Prune rmg to remove empties
    for rmg in list(riskModelGroups):
        if (rmg not in rmgSubIssues) or (len(rmgSubIssues[rmg]) < 1):
            logging.info('RMG: %s has no assets associated so removing', rmg.mnemonic)
            riskModelGroups.remove(rmg)
            if rmg in rmgSubIssues:
                del rmgSubIssues[rmg]
        
    writeAssetData(outAsset, outFundamental, riskModelGroups, subIssues, rmgSubIssues,
                   date, modelDB, marketDB, currencyProvider,
                   publishFundamentalRMGs, subIssueFromDates, options=options, riskModel=model)
    outAsset.close()
    outFundamental.close()

def writeMasterCurrency(modelDB, marketDB, model, options, date,
                        currencyProvider):
    out = open('%s/DbMaster-%s_Currency.binary' % (
        options.targetDir, model.extractName), 'wb')
    numeraireCode = 'USD'
    currencyConverter = currencyProvider.getCurrencyConverter(date)
    numeraire = currencyProvider.getCurrencyID(numeraireCode, date)
    assert(numeraire is not None)
    # get active currency IDs
    marketDB.dbCursor.execute("""SELECT id FROM currency_ref
       WHERE from_dt <= :date_arg AND thru_dt > :date_arg""",
                              date_arg=date)
    currencyIDs = set([i[0] for i in marketDB.dbCursor.fetchall()])
    # get currencies used in production models.
    # we only care about risk-free rates for those
    query="""SELECT DISTINCT currency_code
    FROM rmg_currency rc JOIN rmg_model_map rm
      ON rc.rmg_id=rm.rmg_id AND rm.rms_id > 0
    WHERE rc.from_dt <= :dt AND rc.thru_dt > :dt
      AND rm.from_dt <= :dt AND rm.thru_dt > :dt
    """
    if date >= CNH_START_DATE:
        query=query + """ union select 'CNH' from dual """

    modelDB.dbCursor.execute(query, dt=date)
    modelCurrencies = set([i[0] for i in modelDB.dbCursor.fetchall()])
    logging.debug('%d currencies, %d in models', len(currencyIDs),
                  len(modelCurrencies))
    modelCurrencyList = list(modelCurrencies)
    riskFreeRates = modelDB.getRiskFreeRateHistory(
        list(modelCurrencies), [date], marketDB, annualize=True)
    modelCurrencyIdx = dict([(j,i) for (i,j)
                             in enumerate(riskFreeRates.assets)])
    riskFreeRates = riskFreeRates.data[:,0]
    for currencyID in currencyIDs:
        code = marketDB.getCurrencyISOCode(currencyID, date)
        desc = marketDB.getCurrencyDesc(currencyID, date)
        if date >= REDUCED_CURRENCY_DATE and code not in modelCurrencies and code != 'XDR':
            continue

        if code in modelCurrencies:
            rfr = riskFreeRates[modelCurrencyIdx[code]]
            if rfr is ma.masked:
                rfr = None
            else:
                rfr *= 100.0
            cumR = modelDB.getCumulativeRiskFreeRate(code, date)
        else:
            rfr = None
            cumR = None
        rate = None
        if currencyConverter.hasRate(currencyID, numeraire):
            rate = currencyConverter.getRate(currencyID, numeraire)
            out.write(fixedLength(code, 5))
            out.write(variableLength(desc))
            out.write(binaryDouble(rate))
            out.write(binaryDouble(rfr))
            out.write(binaryDouble(cumR))
        elif code in modelCurrencies:
            logging.error('Missing exchange rate for %s on %s',
                          code, date)
    out.close()

def writeAssetData(outAsset, outFundamental, rmgs, allSubIssues, rmgSubIssues, date,
                   modelDB, marketDB, currencyProvider,
                   publishFundamentalRMGs, subIssueFromDates,
                   marketDataOnly=False, rollOverInfo=None, options=None, riskModel=None):
    """Write the asset data in trading currency.
    """
    logging.info('Writing asset file')
    sidCurrencies = modelDB.getTradingCurrency(
        date, allSubIssues, marketDB, 'id')
    missingCurrency = [sid for sid in allSubIssues if sid not in sidCurrencies]
    if len(missingCurrency) > 0:
        logging.fatal('Missing trading currency on %s for %s',
                      date, ','.join([sid.getSubIDString() for sid
                                      in missingCurrency]))
        # To temporarily assign a trading currency, uncomment the line below
        # and comment out the exception
        #sidCurrencies.update((sid, 1) for sid in missingCurrency)


        #### comment out for now
        raise Exception('Missing trading currencies on %s' % date)
    histLen = 60
    # Build list of all days that are trading days
    allDays = set()
    rmgTradingDays = dict()
    for rmg in rmgs:
        dateList = set(modelDB.getDates([rmg], date, histLen))
        rmgTradingDays[rmg] = dateList
        allDays |= dateList
    allDays = sorted(allDays)
    dailyVolumes = modelDB.loadVolumeHistory(
        allDays, allSubIssues, sidCurrencies)
    dv1 = modelDB.loadVolumeHistory(
        [date], allSubIssues, sidCurrencies)
    dailyVolumes.data = Matrices.fillAndMaskByDate(
        dailyVolumes.data, [subIssueFromDates[sid] for sid in allSubIssues],
        allDays)
    # RLG, 11/13/08
    # instead of trading days, use all weekdays back from this date
    # when calculating returns
    startDay = getWeekdaysBack(date, histLen)[0]
    # get all days from startDay to date, including date
    retDays = [startDay+datetime.timedelta(i) for i in range((date - startDay).days + 1)]
    totalReturns = modelDB.loadTotalLocalReturnsHistory(
        retDays, allSubIssues)
    rawUCP = modelDB.loadRawUCPHistory([date], allSubIssues)
    ucp = modelDB.loadUCPHistory(allDays, allSubIssues, sidCurrencies)
    acp = modelDB.loadACPHistory(allDays, allSubIssues, marketDB,
                                 sidCurrencies)
    cumReturns = modelDB.loadCumulativeReturnsHistory(
        [date, date - datetime.timedelta(1), date - datetime.timedelta(2),
         date - datetime.timedelta(3)], allSubIssues)
    latestCumReturns = cumReturns.data[:,0]

    # load up the proxy returns
    proxyReturns = modelDB.loadProxyReturnsHistory(allSubIssues,[date])

    # load up the non traded indicator data
    nonTradInt = modelDB.loadNotTradedInd([date],allSubIssues)
    
    for i in range(3):
        latestCumReturns = ma.where(ma.getmaskarray(latestCumReturns),
                                    cumReturns.data[:,i+1],
                                    latestCumReturns)
    marketCaps = modelDB.loadMarketCapsHistory(
        allDays, allSubIssues, sidCurrencies)
    
    if hasattr(options, 'cnRMMFields') and options.cnRMMFields:
        freeFloat = modelDB.loadFreeFloatHistory([date], allSubIssues, marketDB)

    if hasattr(options, 'newRMMFields') and options.newRMMFields:
        # fish out all the descriptor values to be stored.  Use a temporary method for now
        descriptorMap={}
        for desc_name, desc_id in modelDB.getAllDescriptors():
            if desc_name.lower() in RMM_ASSET_FIELDS_DICT:
                RMM_ASSET_FIELDS_DICT[desc_name.lower()] = desc_id
        for desc_name, desc_id in RMM_ASSET_FIELDS_DICT.items():
            if desc_id == -1:
                logging.fatal('No descriptor for %s', desc_name)
            logging.info('Loading data %s for %s', desc_name, date)
            desc = modelDB.loadDescriptorData(date, allSubIssues, riskModel.descriptorNumeraire or riskModel.numeraire.currency_code, desc_id, rollOverData=False, tableName='descriptor_numeraire')
            #desc = modelDB.loadDescriptorData(date, allSubIssues, riskModel.numeraire.currency_code, desc_id, rollOverData=False, tableName='descriptor_numeraire')
            for idx, e in enumerate(desc.assets):
                descriptorMap[(e,desc_id)] = desc.data[idx,0]

    sidIdxMap = dict([(sid, idx) for (idx, sid) in enumerate(allSubIssues)])
    dateIdxMap = dict([(d, idx) for (idx, d) in enumerate(allDays)])
    for rmg in rmgs:
        subIssues = rmgSubIssues[rmg]
        if len(subIssues) < 1:
            logging.info('RMG %s is empty, skipping', rmg.mnemonic)
            continue
        rmgSidIndices = [sidIdxMap[sid] for sid in subIssues]
        if len(rmgSidIndices) == 1:
            # Add another asset to prevent numpy from converting
            # arrays with only one element to a number
            rmgSidIndices.append(0)
        tradingDays = sorted(rmgTradingDays[rmg], reverse=True)
        logging.debug('Writing data for %d assets in %s', len(subIssues), rmg)
        latestUCP = Matrices.allMasked((len(allSubIssues),))
        for sid in subIssues:
            sidIdx = sidIdxMap[sid]
            for tDate in tradingDays:
                tIdx = dateIdxMap[tDate]
                if not ucp.data[sidIdx,tIdx] is ma.masked:
                    latestUCP[sidIdx] = ucp.data[sidIdx,tIdx]
                    break
        latestACP = Matrices.allMasked((len(allSubIssues),))
        for sid in subIssues:
            sidIdx = sidIdxMap[sid]
            for tDate in tradingDays:
                tIdx = dateIdxMap[tDate]
                if not acp.data[sidIdx,tIdx] is ma.masked:
                    latestACP[sidIdx] = acp.data[sidIdx,tIdx]
                    break
        latestMarketCap = Matrices.allMasked((len(allSubIssues),))
        for sid in subIssues:
            sidIdx = sidIdxMap[sid]
            for tDate in tradingDays:
                tIdx = dateIdxMap[tDate]
                if not marketCaps[sidIdx,tIdx] is ma.masked:
                    latestMarketCap[sidIdx] = marketCaps[sidIdx,tIdx]
                    break
        tradingDayIndices = sorted([dateIdxMap[td] for td in tradingDays])
        rmgDailyVolumes = ma.take(dailyVolumes.data, rmgSidIndices, axis=0)
        rmgDailyVolumes = ma.take(rmgDailyVolumes, tradingDayIndices, axis=1)
        rmgReturns = ma.take(totalReturns.data, rmgSidIndices, axis=0)
        # All days in the totalReturns matrix are considered trading days
        # rmgReturns = ma.take(rmgReturns, tradingDayIndices, axis=1)
        adv5 = ma.average(rmgDailyVolumes[:,-5:], axis=1)
        adv20 = ma.average(rmgDailyVolumes[:,-20:], axis=1)
        mdv20 = ma.median(rmgDailyVolumes[:,-20:], axis=1)
        mdv60 = ma.median(rmgDailyVolumes[:,-60:], axis=1)

#        if len(rmgDailyVolumes.tolist())==0:
#            mdv20=[] 
#            mdv60=[]
#        else:
#            mdv20 = ma.median(rmgDailyVolumes[:,-20:], axis=1)
#            mdv60 = ma.median(rmgDailyVolumes[:,-60:], axis=1)

        ret1 = 100.0 * rmgReturns[:,-1]
        # five weekdays ago
        daysAgo = getWeekdaysBack(date, 5)[0]
        # include all days between daysAgo and this date
        ret5 = 100.0 * ma.product((rmgReturns + 1.0)[:,(daysAgo-date).days:], axis=1) - 100.0
        # twenty weekdays ago
        daysAgo = getWeekdaysBack(date, 20)[0]
        ret20 = 100.0 * ma.product((rmgReturns + 1.0)[:,(daysAgo-date).days:], axis=1) - 100.0
        # sixty weekdays ago
        daysAgo = getWeekdaysBack(date, 60)[0]
        ret60 = 100.0 * ma.product((rmgReturns + 1.0)[:,(daysAgo-date).days:], axis=1) - 100.0
        if rmg in publishFundamentalRMGs and not marketDataOnly:
            logging.debug('Getting fundamental attributes for %s', rmg.mnemonic)
            twoYearsAgo = date - datetime.timedelta(2*365)
            rmgACP = ma.take(latestACP, rmgSidIndices, axis=0)
            rmgMCap = ma.take(latestMarketCap, rmgSidIndices, axis=0)
            
            # EBITDA
            ebitda = Utilities.extractLatestValue(
                modelDB.getFundamentalCurrencyItem(
                    'ebitda_ann', twoYearsAgo, date, subIssues, date,
                    marketDB, sidCurrencies))
            # get fundamental data elements for US
            bookValuePS = Utilities.extractLatestValue(
                modelDB.getFundamentalCurrencyItem(
                    'bkvlps_ann', twoYearsAgo, date, subIssues, date,
                    marketDB, sidCurrencies, splitAdjust='divide'))
            p2book = rmgACP / bookValuePS
            annualEarningsPS = Utilities.extractLatestValue(
                modelDB.getFundamentalCurrencyItem(
                'epsx12_qtr', twoYearsAgo, date, subIssues, date,
                marketDB, sidCurrencies, splitAdjust='divide'))
            p2earn = rmgACP / annualEarningsPS
            
            # Compute Debt to Equity
            debtToEquity = getLatestDebtToEquity(
                modelDB, date, subIssues, sidCurrencies, marketDB)
            
            # Compute price to sales
            latestSales = getLatestSales(
                modelDB, date, subIssues, sidCurrencies, marketDB)
            p2sales = rmgMCap / latestSales
            
            # Compute dividend yield
            latestADPS_y = getDividendsPerShare(
                modelDB, date, subIssues, sidCurrencies, marketDB)
            div_yield = latestADPS_y * 100.0 / rmgACP
            
            # Compute 1 year earnings growth
            growth_earn = computeEarningsGrowth(
                modelDB, date, subIssues, sidCurrencies, marketDB)
            
            # Compute 1 year income growth
            growth_inc = computeIncomeGrowth(
                modelDB, date, subIssues, sidCurrencies, marketDB)
        else:
            debtToEquity = Matrices.allMasked((len(subIssues),))
            div_yield = Matrices.allMasked((len(subIssues),))
            ebitda = Matrices.allMasked((len(subIssues),))
            growth_earn = Matrices.allMasked((len(subIssues),))
            growth_inc = Matrices.allMasked((len(subIssues),))
            p2earn = Matrices.allMasked((len(subIssues),))
            p2book = Matrices.allMasked((len(subIssues),))
            p2sales = Matrices.allMasked((len(subIssues),)) 
        # clip ratios at 50,000
        CUT_OFF = 50000.0
        debtToEquity = MAclip(debtToEquity, -CUT_OFF, CUT_OFF)
        growth_earn = MAclip(growth_earn, -CUT_OFF, CUT_OFF)
        growth_inc = MAclip(growth_inc, -CUT_OFF, CUT_OFF)
        p2earn = MAclip(p2earn, -CUT_OFF, CUT_OFF)
        p2book = MAclip(p2book, -CUT_OFF, CUT_OFF)
        p2sales = MAclip(p2sales, -CUT_OFF, CUT_OFF)

       
        for (rmgIdx, sid) in enumerate(subIssues):
            idx = sidIdxMap[sid]
            values = dict()
            values['id'] = sid.getModelID().getPublicID()
            values['ret1'] = numOrNull(ret1[rmgIdx])
            values['ucp'] = numOrNull(latestUCP[idx])
            values['ret5'] = numOrNull(ret5[rmgIdx])
            values['ret20'] = numOrNull(ret20[rmgIdx])
            values['ret60'] = numOrNull(ret60[rmgIdx])
            values['cumret'] = numOrNull(latestCumReturns[idx], 16)
            if sid.isCashAsset():
                # hardcode the cash asset price if it turns out to be null
                if latestUCP[idx] is ma.masked:
                    logging.info('hardcoding ucp to 1 for %s',  values['id'])
                    values['ucp'] = '1'
                values['adv5'] = NULL
                values['adv20'] = NULL
                values['mdv20'] = NULL
                values['mdv60'] = NULL
            else:
                values['adv5'] = numOrNull(adv5[rmgIdx])
                values['adv20'] = numOrNull(adv20[rmgIdx])
                values['mdv20'] = numOrNull(mdv20[rmgIdx])
                values['mdv60'] = numOrNull(mdv60[rmgIdx])
            values['mcap'] = numOrNull(latestMarketCap[idx])
            if rollOverInfo is not None and sid in rollOverInfo:
                flagRollOver = rollOverInfo[sid]
            else:
                missingPrice = ucp.data[idx, -1] is ma.masked
                stalePrice = rawUCP[idx,0] is not None and rawUCP[idx,0].price_marker == 3
                flagRollOver = missingPrice or stalePrice
            values['rollover'] = flagRollOver and '1' or '0'
            values['non_trad_indicator'] = numOrNull((nonTradInt.data[idx,0]))
            values['proxy_returns'] =  numOrNull(proxyReturns.data[idx,0])
            if sid.isCashAsset():
                values['tdv1'] = NULL
            else:
                values['tdv1'] = numOrNull(dv1.data[idx,0])

            if not marketDataOnly:
                values['corpAct'] = '0'
                values['debt2equity'] = numOrNull(debtToEquity[rmgIdx])
                values['div_yield'] = numOrNull(div_yield[rmgIdx])
                values['ebitda'] = numOrNull(ebitda[rmgIdx])
                values['growth_earn'] = numOrNull(growth_earn[rmgIdx])
                values['growth_inc'] = numOrNull(growth_inc[rmgIdx])
                values['p2earn'] = numOrNull(p2earn[rmgIdx])
                values['p2book'] = numOrNull(p2book[rmgIdx])
                values['p2sales'] = numOrNull(p2sales[rmgIdx])
                if sid in options.trackList:
                    logging.info('Tracked asset: %s being written to asset file', sid.getSubIDString())
                    logging.info('...values written: %s', values)

                # check to see if this is the new style or old style of file
                if options and options.fileFormatVersion >= 4.0:
                    outAsset.write('%(id)s|%(adv5)s|%(adv20)s|%(corpAct)s|%(debt2equity)s|%(div_yield)s|%(ebitda)s|%(growth_earn)s|%(growth_inc)s|%(mcap)s|%(p2earn)s|%(p2book)s|%(p2sales)s|%(ucp)s|%(ret1)s|%(ret5)s|%(ret20)s|%(ret60)s|%(rollover)s|%(non_trad_indicator)s|%(proxy_returns)s|%(tdv1)s|%(mdv20)s|%(mdv60)s' % values)
                else:
                    if options and options.newMDVFields:
                        outAsset.write('%(id)s|%(adv5)s|%(adv20)s|%(corpAct)s|%(debt2equity)s|%(div_yield)s|%(ebitda)s|%(growth_earn)s|%(growth_inc)s|%(mcap)s|%(p2earn)s|%(p2book)s|%(p2sales)s|%(ucp)s|%(ret1)s|%(ret5)s|%(ret20)s|%(ret60)s|%(cumret)s|%(rollover)s|%(non_trad_indicator)s|%(proxy_returns)s|%(tdv1)s|%(mdv20)s|%(mdv60)s' % values)
                    else:
                        outAsset.write('%(id)s|%(adv5)s|%(adv20)s|%(corpAct)s|%(debt2equity)s|%(div_yield)s|%(ebitda)s|%(growth_earn)s|%(growth_inc)s|%(mcap)s|%(p2earn)s|%(p2book)s|%(p2sales)s|%(ucp)s|%(ret1)s|%(ret5)s|%(ret20)s|%(ret60)s|%(cumret)s|%(rollover)s|%(non_trad_indicator)s|%(proxy_returns)s|%(tdv1)s' % values)

                # if there are new RMM fields add them now
                if hasattr(options, 'newRMMFields') and options.newRMMFields:
                    #  columns are:   isc_adv_score    isc_ipo_score   isc_ret_score  
                    for desc_values in RMM_ASSET_FIELDS:
                        desc_id = RMM_ASSET_FIELDS_DICT[desc_values]
                        values[desc_values] = numOrNull(descriptorMap.get((sid,desc_id),None))
                        outAsset.write('|%s' % values[desc_values])
                
                # need to write out the free float data here....
                if hasattr(options, 'cnRMMFields') and options.cnRMMFields:
                    #print("Free float", sid, idx, freeFloat.data[idx,0], numOrNull(freeFloat.data[idx,0]))
                    outAsset.write('|%s' % numOrNull(freeFloat.data[idx,0]))
            else:
                outAsset.write('%(id)s|%(adv5)s|%(adv20)s|%(mcap)s|%(ucp)s|%(ret1)s|%(ret5)s|%(ret20)s|%(ret60)s|%(cumret)s|%(rollover)s' % values)

            outAsset.write('\n')
            if outFundamental is not None:
                outFundamental.write('%(id)s|%(corpAct)s|%(debt2equity)s|%(div_yield)s|%(ebitda)s|%(growth_earn)s|%(growth_inc)s|%(p2earn)s|%(p2book)s|%(p2sales)s\n' % values)

def writeMaster(dbName, subName, modelDB, date, marketDB, allIssues, issueRMGDict,
                options, modelList=None):
    global ALL_IDS, NUM_IDS
    # may need to hack CNH in here to add to Master File
    
    cnyInfo= [allIssues[a] for a in allIssues.keys() if a.getIDString().find('DCSH_CNY')==0]
    if len(cnyInfo) > 0 and date >= CNH_START_DATE:
        cnh=ModelID.ModelID(string='DCSH_CNH__')
        allIssues[cnh] = cnyInfo[0]

    BOT = datetime.date(1950,1,1)
    EOT = datetime.date(9999,12,31)
    datestr=str(date)[:10].replace('-','')
    logging.debug('%d assets in master', len(allIssues))
    if hasattr(options, 'newRMMFields') and options.newRMMFields:
        ALL_IDS.update(V4_IDS)
        NUM_IDS =  len(ALL_IDS)
        assert(len(set(val[0] for val in ALL_IDS.values())) == NUM_IDS)

    if hasattr(options, 'writeText') and options.writeText==True:
        writeBinary=False
        # add GICS_IDS to ALL_IDS
        ALL_IDS.update(GICS_IDS)
        NUM_IDS = len(ALL_IDS)
        assert(len(set(val[0] for val in ALL_IDS.values())) == NUM_IDS)

    else:
        writeBinary=True
    fileNameList=[]
    if writeBinary:
        out1 = open('%s/%s-%s_Master.binary' % (options.targetDir, dbName,
                                                subName), 'wb')
        out2 = open('%s/%s-%s_Master.binary_CUSIP' % (options.targetDir, dbName,
                                                      subName), 'wb')
        out3 = open('%s/%s-%s_Master.binary_SEDOL' % (options.targetDir, dbName,
                                                      subName), 'wb')
        out4 = open('%s/%s-%s_Master.binary_NONE' % (options.targetDir, dbName,
                                                     subName), 'wb')
    else:
        target=options.targetDir
        if options.appendDateDirs:
            target = os.path.join(target, '%04d' % date.year, '%02d' % date.month)
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass

        file1='%s/%s.%s.idh' % (target, subName, datestr)
        file2='%s/%s-CUSIP.%s.idh' % (target, subName,datestr)
        file3='%s/%s-SEDOL.%s.idh' % (target, subName,datestr)
        file4='%s/%s-NONE.%s.idh' % (target, subName, datestr)
        out1=open(file1,'wb')
        out2=open(file2,'wb')
        out3=open(file3,'wb')
        out4=open(file4,'wb')
        fileNameList=[file1,file2,file3,file4]
    outFiles = [out1, out2, out3, out4]
    regionFamily = marketDB.getClassificationFamily('REGIONS')
    regionMembers = marketDB.getClassificationFamilyMembers(regionFamily)
    marketMember = [i for i in regionMembers if i.name=='Market'][0]
    marketRev = marketDB.getClassificationMemberRevision(
        marketMember, date)
    countryMap = modelDB.getMktAssetClassifications(
        marketRev, allIssues, date, marketDB, level=1)
    marketDB.dbCursor.execute("""SELECT id, code FROM currency_ref""")
    currencyDict = dict(marketDB.dbCursor.fetchall())
    localTickers = set([str(x.classification.code) for x in countryMap.values() if x.classification.code != 'US'])
    localTickerDict = dict()

    for f in outFiles:
        logging.info('Writing %s file', f)
        if writeBinary:
            f.write(encodeDate(date))
        else:
            # write header for flat file
            f.write('#DataDate: %s\n' % date)
            # write createDate in UTC
            gmtime = time.gmtime(time.mktime(modelDB.revDateTime.timetuple()))
            utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
            f.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
            f.write("#FlatFileVersion:3.3\n")
            if modelList:
                familyName=[i.name.rstrip('MH').rstrip('SH').rstrip('SH-S').rstrip('MH-S') for i in modelList][0]
                f.write('#ModelFamily: %s\n' % familyName)
                f.write('#ModelName:%s\n' % ','.join([m.name for m in modelList]))
                f.write('#ModelNumeraire:%s\n' % modelList[0].numeraire.currency_code)
            f.write('#Columns: AxiomaID|IDType|ID Value|From Date|Thru Date\n')
            f.write('#Type: ID|Set|Attribute|Attribute|Attribute\n')
            f.write('#Unit: ID|NA|Text|Date|Date\n')
            f.write('#Types supported:%s' % ','.join(list(ALL_IDS.keys())))

        # write header record which indicates how many and which IDs will be
        # in the file. There are NUM_IDS regular IDs plus one country ticker
        # per risk model group
        numTickers = len(localTickers)
        if writeBinary:
            f.write(bytes([NUM_IDS+numTickers, NUM_RESERVED]))

        for (name, (idnum, length, table, pad)) in ALL_IDS.items():
            if writeBinary:
                f.write(bytes([idnum]))
                f.write(variableLength(name))
                f.write(bytes([length]))
            #else:
            #    f.write('%s |' % (name))
        for (tickerNum, code) in enumerate(localTickers):
            localTickerDict[code] = NUM_IDS + tickerNum + 1
            if writeBinary:
                f.write(bytes([localTickerDict[code]]))
            #else:
            #    f.write('%s |' % localTickerDict[code])
            name = 'Ticker %s' % code
            if writeBinary:
                f.write(variableLength(name))
                f.write(bytes([0]))
            # removed on 2013-04-04 to get rid of country specific tickers
            ###else:
            ###    f.write(', %s' % name)
        if not writeBinary:
            f.write('\n')
    excludeException = False
    # write out all issues
    allIDHistories = dict()
    for (name, (idnum, length, table, pad)) in ALL_IDS.items():
        if table is None:
            continue
        cache = None
        idCol = 'id'
        isClassification = ''
        if table == 'asset_dim_ticker':
            cache = modelDB.tickerCache
        elif table == 'asset_dim_name':
            cache = modelDB.nameCache
        elif table == 'asset_dim_sedol':
            cache = modelDB.sedolCache
        elif table == 'asset_dim_cusip':
            cache = modelDB.cusipCache
        elif table == 'asset_dim_isin':
            cache = modelDB.isinCache
        elif table == 'asset_dim_trading_currency':
            cache = modelDB.tradeCcyCache
        elif table == 'asset_dim_company':
            cache = modelDB.companyCache
            idCol = 'company_id'
        elif table == 'Axioma Asset Type':
            isClassification = 'AssetType'
        elif table == 'GICS':
            isClassification = 'GICS'
        elif table == 'MIC':
            isClassification = 'Market'

        elif table == 'asset_dim_root_id':
            idCol = 'ROOT_AXIOMA_ID'
        elif table == 'issue_exposure_linkage':
            idCol = 'MASTER_ISSSUE_ID'
            mainID = 'SLAVE_ISSUE_ID'
          

        if isClassification == 'AssetType':
            clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
            clsMember = marketDB.getClassificationFamilyMembers(clsFamily)
            thisMember = [mem for mem in clsMember if mem.name==table][0]
            # Assumes that we only want to publish the current revision
            # That might not be correct for the history files but for now we only have one revision
            thisRevision = marketDB.getClassificationMemberRevision(thisMember, date)
            # Load asset level data which will populate the cache with the entire history
            clsData=modelDB.getMktAssetClassifications(thisRevision, list(allIssues.keys()), date, marketDB, None)
            level = ASSET_TYPE[name]
            fromThruHistory = modelDB.marketClassificationCaches[thisRevision.id].asFromThruCache(list(allIssues.keys()), level)
            idHistories = dict((asset, fromThruHistory.getAssetHistory(asset)) for asset in allIssues.keys())
            # just for one of the asset types load up the asset type code dictionary
            if name=='Asset Type':
                 assetTypeCodeDict={}
                 for issueid, item in clsData.items():
                     if item.classification is not None:
                         assetTypeCodeDict[issueid] = str(item.classification.code)
                     else:
                         assetTypeCodeDict[issueid] = '---'
        elif isClassification == 'GICS':
            clsFamily = marketDB.getClassificationFamily('INDUSTRIES')
            clsMember = marketDB.getClassificationFamilyMembers(clsFamily)
            thisMember = [mem for mem in clsMember if mem.name==table][0]
            # Assumes that we only want to publish the current revision
            # That might not be correct for the history files but for now we only have one revision
            thisRevision = marketDB.getClassificationMemberRevision(thisMember, date)
            # Load asset level data which will populate the cache with the entire history
            modelDB.getMktAssetClassifications(thisRevision, list(allIssues.keys()), date, marketDB, None)
            level = GICS_TYPE[name]
            fromThruHistory = modelDB.marketClassificationCaches[thisRevision.id].asFromThruCache(list(allIssues.keys()), level)
            idHistories = dict((asset, fromThruHistory.getAssetHistory(asset)) for asset in allIssues.keys())
        elif isClassification == 'Market':
            # Load asset level data which will populate the cache with the entire history
            modelDB.getMktAssetClassifications(marketRev, list(allIssues.keys()), date, marketDB, None)
            fromThruHistory = modelDB.marketClassificationCaches[marketRev.id].asFromThruCache(
                list(allIssues.keys()), None, idField='id')
            clsHistories = {asset: fromThruHistory.getAssetHistory(asset) for asset in allIssues}
            query = """SELECT classification_id, operating_mic, from_dt, thru_dt FROM class_mic_map_active_int"""
            marketDB.dbCursor.execute(query)
            marketMicMap = defaultdict(list)
            for market, mic, fromDt, thruDt in marketDB.dbCursor.fetchall():
                marketMicMap[market].append((mic, fromDt.date(), thruDt.date()))
            idHistories = dict()
            for asset, history in clsHistories.items():
                newHistory = list()
                for histEntry in history:
                    crId = histEntry.id
                    mics = marketMicMap[crId]
                    for mic in mics:
                        if histEntry.fromDt < mic[2] and mic[1] < histEntry.thruDt:
                            newVal = Utilities.Struct()
                            newVal.fromDt = max(histEntry.fromDt, mic[1])
                            newVal.thruDt = min(histEntry.thruDt, mic[2])
                            newVal.id = mic[0]
                            newHistory.append(newVal)
                if len(newHistory) > 0:
                    idHistories[asset] = newHistory
        else:
            # we have a special case for the issue linkage
            if table == 'issue_exposure_linkage':
                if name == 'Hard Clone':
                    linkageType=1
                elif name == 'Force Coint':
                    linkageType=2 
                elif name == 'No Coint':
                    linkageType=3 
                idHistories = modelDB.loadIssueExposureLinkage(list(allIssues.keys()), linkageType)
            elif table == 'asset_dim_root_id':
                idHistories = modelDB.loadRootIDLinkage(list(allIssues.keys()), marketDB)
            else:
                idHistories = modelDB.loadMarketIdentifierHistory(
                    list(allIssues.keys()), marketDB, table, idCol, cache=cache)

        allIDHistories[name] = idHistories

        # last check to see if there are any dubious classification types here 
        if name == 'Asset Subclass'  and options.fileFormatVersion >=  4.0:
           extractExcludeList = getExtractExcludeClassifications(None,marketDB)
           for k,v in idHistories.items(): 
              for el in v:
                 if el.id in extractExcludeList:
                     logging.fatal('Prohibited classification encountered during extraction, %s %s', k, el.id)
                     excludeException = True

           if excludeException:
               raise Exception('Extraction exclusion exceptions')

        if name in ('Asset Subclass', 'MIC'):
            pass
        # build up the cache for company ID information now if the table is company_id
        if table == 'asset_dim_company':
            companyIds=set()
            companyIdDict={}
            for key, vals in idHistories.items():
                cid=[v.id for v in vals]
                if cid:
                    companyIdDict[key]=cid[0]
                else:
                    companyIdDict[key]=None

                companyIds |= set([MarketID.MarketID(string=v.id) for v in vals])
            companyIds=list(companyIds)
            # for the list of companyIDs, now load up a cache that contains the company names
            companyNameCache=MarketDB.TimeVariantCache()
            companyNameDict={} 
            results=marketDB.loadTimeVariantTableToCache('company_dim_name', companyIds, ['id'],companyNameCache, 'company_id')
            for compid, compval in companyNameCache.assetValueMap.items():
                cidstr=compid.getIDString()
                companyNameDict[cidstr]=[]
                vals=[[vv[0],vv[1].id, vv[1].change_del_flag] for vv in sorted(compval)]
                for fdt, tdt in zip(vals, vals[1:]+[[datetime.date(2999,12,31),'junk',True]]):
                    if fdt[2]==True:
                        continue
                    # append to dictionary the from-dt/thru-dt/name list
                    idVal=Utilities.Struct()
                    idVal.id=fdt[1]
                    idVal.fromDt=fdt[0]
                    idVal.thruDt=tdt[0]
                    companyNameDict[cidstr].append(idVal)
    
                    
    rmgIDDict = dict()
    assetCount=0
    for (i, (assetFromDt, assetThruDt)) in sorted(allIssues.items()):
        assetCount=assetCount+1
        allIdString = io.BytesIO()
        allIds_nosedols = io.BytesIO()
        allIds_nocusips = io.BytesIO()
        allIds_neither = io.BytesIO()
        textString=''
        textString_nosedols=''
        textString_nocusips=''
        textString_neither=''
        allStrings = [allIdString, allIds_nosedols, allIds_nocusips, allIds_neither]
        textStrings=[textString, textString_nosedols, textString_nocusips, textString_neither]
        numIDs = numIDs_nosedols = numIDs_nocusips = numIDs_neither = 0
        allNumIDs = [numIDs, numIDs_nosedols, numIDs_nocusips, numIDs_neither]
        if i.isCashAsset():
            idRMG = None
        else:
            issueRMGID = issueRMGDict[i]
            if issueRMGID not in rmgIDDict:
                rmg = modelDB.getRiskModelGroup(issueRMGID)
                rmg.setRMGInfoForDate(date)
                rmgIDDict[rmg.rmg_id] = rmg
            idRMG = rmgIDDict[issueRMGID]
        # textStrings[idx] is the string representation of the data
        for (idx, (s, n)) in enumerate(zip(allStrings, allNumIDs)):
            # Assume the asset has always been in this country
            if i.isCashAsset():
                countryCode = ''
            elif i in countryMap:
                countryCode = countryMap[i].classification.code
            else:
                logging.error('No country classification for %s,'
                              ' using rmg instead.', i)
                countryCode = idRMG.mnemonic
            if not i.isCashAsset():
                (idnum, length, table, pad) = ALL_IDS['Country']
                s.write(bytes([idnum]))
                s.write(fixedLength(countryCode, length))
                s.write(encodeDate(assetFromDt))
                s.write(encodeDate(assetThruDt))
                textStrings[idx]='%s|%s|%s|%s|%s' % (i.getPublicID(),'Country', countryCode, assetFromDt,assetThruDt)
                n += 1
            # find and count all IDs for this issue's market IDs throughout time
            for (name, (idnum, length, table, pad)) in ALL_IDS.items():
                if table is None and name != 'Company Name':
                    continue
                # cash assets don't have company names, so skip it
                if i.isCashAsset() and name =='Company Name':
                    continue
                if name.lower().startswith('isin') and idx != 0:
                    continue
                if name.lower().startswith('sedol') and idx not in (0,2):
                    continue
                if name.lower().startswith('cusip') and idx not in (0,1):
                    continue
                if name=='Company Name':
                    idList=[]
                    cid = companyIdDict.get(i)
                    if cid:
                        idList=companyNameDict[cid]
                elif name=='MIC':
                    # if file version is less than 4.0, by pass this
                    if options.fileFormatVersion < 4.0:
                        continue
                    # Cash assets don't have MICs.  so simply leave that out
                    if i.isCashAsset():
                        idList = []
                    else:
                        if i in allIDHistories[name]:
                            idList = allIDHistories[name][i]
                        else:
                            idList = []
                else:
                    idList = allIDHistories[name][i]
                if not idList and name == 'Company' and not i.isCashAsset():
                    logging.error('No company ID for %s', i)

                # cash assets may not have Asset Type classifications set up.  If not, hack it for now
                # and create a 'Cash' classification type here.  Do it only for the 4.0 version, though - ugly hack
                if i.isCashAsset() and name in ASSET_TYPE.keys() and not idList: # and options.fileFormatVersion >= 4.0:
                    idVal=Utilities.Struct()
                    if name == 'Asset Type':
                        idVal.id='Cash-T'
                    elif name == 'Asset Class':
                        idVal.id='Cash-C'
                    if name == 'Asset Subclass':
                        idVal.id='Cash-S'
                    idVal.fromDt=assetFromDt
                    idVal.thruDt=assetThruDt
                    idList=[idVal]
                    #print i, table, idList
                for idValue in idList:
                    value = idValue.id
                    if name == 'Currency':
                        value = currencyDict[value]
                    from_dt = idValue.fromDt
                    thru_dt = idValue.thruDt
                    s.write(bytes([idnum]))
                    n += 1
                    if name == 'Ticker Map' and not i.isCashAsset() \
                           and idRMG.rmg_id != 1: # not for US and cash assets
                        # add mnemonic to ticker
                        originalValue = value
                        value = value + "-" + countryCode
                    if length == 0:
                        s.write(variableLength(value))
                    elif pad:
                        s.write(zeroPad(value, length))
                    else:
                        s.write(fixedLength(value, length))
                    s.write(encodeDate(from_dt))
                    s.write(encodeDate(thru_dt))
                    if name in ['ISSUER', 'Company Name']:
                        val = value.encode('utf-8')
                        textStrings[idx]='%s\n%s|%s|%s|%s|%s' % (textStrings[idx], i.getPublicID(),name, val, from_dt, thru_dt)
                    elif name == 'Ticker Map':
                        #ignore the country specific ticker
                        pass
                    else:
                        textStrings[idx]='%s\n%s|%s|%s|%s|%s' % (textStrings[idx], i.getPublicID(),name, value, from_dt, thru_dt)
                    if name == 'Cusip Map':
                        # add 8-digit CUSIP from full CUSIP
                        (idnum1, length1, table, pad1) = ALL_IDS['8 Digit Cusip Map']
                        n += 1
                        s.write(bytes([idnum1]))
                        s.write(fixedLength(value[:length1], length1))
                        s.write(encodeDate(from_dt))
                        s.write(encodeDate(thru_dt))
                        textStrings[idx]='%s\n%s|%s|%s|%s|%s' % (textStrings[idx], i.getPublicID(),'8 Digit Cusip Map', value[:length1], from_dt, thru_dt)
                    elif name == 'SEDOL Map':
                        # add 6-digit SEDOL from full SEDOL
                        (idnum1, length1, table, pad1) = ALL_IDS['6 Digit SEDOL Map']
                        n += 1
                        s.write(bytes([idnum1]))
                        s.write(fixedLength(value[:length1], length1))
                        s.write(encodeDate(from_dt))
                        s.write(encodeDate(thru_dt))
                        textStrings[idx]='%s\n%s|%s|%s|%s|%s' % (textStrings[idx], i.getPublicID(), '6 Digit SEDOL Map', value[:length1], from_dt, thru_dt)

                    elif name == 'Ticker Map' and not i.isCashAsset() \
                             and idRMG.rmg_id != 1: # not for US and cash assets
                        # add original ticker map
                        n += 1
                        s.write(bytes([localTickerDict[countryCode]]))
                        s.write(variableLength(originalValue))
                        s.write(encodeDate(from_dt))
                        s.write(encodeDate(thru_dt))
                        ###textStrings[idx]='%s\n%s|Ticker %s|%s|%s|%s' % (textStrings[idx], i.getPublicID(),countryCode, originalValue, from_dt, thru_dt)
                        textStrings[idx]='%s\n%s|Ticker|%s|%s|%s' % (textStrings[idx], i.getPublicID(),originalValue, from_dt, thru_dt)
                    elif name == 'Asset Type': # and ((options.fileFormatVersion < 4.0 and not i.isCashAsset()) or (options.fileFormatVersion >= 4.0)): 
                        # add the asset type code
                        # if it is 4.0 do this for all assets.  if it is prior to 4.0 file version, do this for NON-cash assets only
                        n += 1
                        s.write(bytes([15]))
                        if i.isCashAsset():
                            s.write(variableLength('Cash'))
                        else:
                            s.write(variableLength(assetTypeCodeDict.get(i,'---')))
                        s.write(encodeDate(from_dt))
                        s.write(encodeDate(thru_dt))
                        textStrings[idx]='%s\n%s|Asset Type Code|%s|%s|%s' % (textStrings[idx], i.getPublicID(),assetTypeCodeDict.get(i,'---'), from_dt, thru_dt)
            if n > 255:
                logging.fatal('More than 255 IDs for Model ID %s (%d)', i.getIDString(),n)
                raise ValueError('More than 255 IDs for Model ID %s' % i.getIDString())
            allNumIDs[idx] = n
        # write count and string to file
        outputs = zip(outFiles, allStrings, textStrings,allNumIDs)
        for (f, s, txt, n) in outputs:
            if writeBinary:
                f.write(fixedLength(i.getPublicID(), 9))
                f.write(encodeDate(assetFromDt))
                f.write(encodeDate(assetThruDt))
                f.write(b' ' * NUM_RESERVED)
                f.write(bytes([n]) + s.getvalue())
                #s.close()
            else:
                f.write('%s|Asset Life|%s|%s|%s\n' % (i.getPublicID(), i.getPublicID(),assetFromDt, assetThruDt))
                f.write(txt)
                f.write('\n\n')
            s.close()

    for f in outFiles:
        f.close()
    return fileNameList

def writeGSDerby(marketDB, modelDB, rmg, date, exposureAssets, inFileName, options):
    if not os.path.isfile(inFileName):
        logging.debug("Can't find Goldman file %s" % inFileName)
        return
    infile = open(inFileName, 'r')
    logging.info('Load all axioma IDs')
    allaxids = marketDB.getAllAxiomaIDs()
    
    tickerDict = dict(marketDB.getTickers(date, allaxids))
    cusipDict = dict(marketDB.getCUSIPs(date, allaxids))
    sedolDict = dict(marketDB.getSEDOLs(date, allaxids))
    isinDict = dict(marketDB.getISINs(date, allaxids))
    countryDict = marketDB.getTradingCountry(date, allaxids)
    tickerReverseDict = dict((j+'-'+(countryDict.get(i) and countryDict.get(i) or ''),i) for (i,j) in tickerDict.items())
    cusipReverseDict = dict([(j+'-'+(countryDict.get(i) and countryDict.get(i) or ''),i) for (i,j) in cusipDict.items()])
    sedolReverseDict = dict([(j,i) for (i,j) in sedolDict.items()])
    issueMapPairs = modelDB.getIssueMapPairs(date)
    marketModelMap = dict([(j,i) for (i,j) in issueMapPairs])
    marketModelActive = dict(marketModelMap)
    outFileName = '%s/Shortfall-%s_Shortfall' % (options.targetDir, rmg)
    outFile = open(outFileName, 'w')
    for inline in infile:
        if inline.startswith('#'):
            continue
        fields = inline.strip().split(',')
        ticker = fields[1]
        cusip = fields[2].strip('"')
        n1 = float(fields[4])    # goldman "A"
        n2 = float(fields[5])    # goldman "B"
        n3 = float(fields[6])    # goldman "C"
        if len(fields)>9:
            sedol = fields[8].strip('"')
            country = fields[9].strip('"')
        else:
            sedol = None
            country = 'US'
        if rmg == 'US' and country != 'US':
            continue
        axiomaID1 = tickerReverseDict.get(ticker + (country and '-' + country or ''))
        if cusip and country in ('US', 'CA'):
            useCUSIP = True
            axiomaID2 = cusipReverseDict.get(cusip + '-' + country)
        else:
            useCUSIP = False
            axiomaID2 = sedolReverseDict.get(sedol)
        marketAXID = axiomaID1
        if axiomaID1 is None:
            logging.warning('No axioma ID for ticker %s in country %s on date %s',
                         ticker, country, date)
            marketAXID = axiomaID2
        if axiomaID2 is None:
            if useCUSIP:
                logging.warning('No axioma ID for CUSIP %s on date %s',
                             cusip, date)
            else:
                logging.warning('No axioma ID for SEDOL %s on date %s',
                             sedol, date)
            marketAXID = axiomaID1
        if axiomaID1 is not None and axiomaID2 is not None and axiomaID1 != axiomaID2:
            axid1 = marketModelMap.get(axiomaID1)
            axid2 = marketModelMap.get(axiomaID2)
            if axid1 is None:
                marketAXID = axiomaID2
            elif axid2 is None:
                marketAXID = axiomaID1
            elif axid1 != axid2:
                subid1 = None
                subid2 = None
                query = """SELECT sub_id FROM sub_issue WHERE issue_id=:id
                AND from_dt <= :dt AND thru_dt > :dt"""
                modelDB.dbCursor.execute(query, id=axid1.getIDString(), dt=date)
                r = modelDB.dbCursor.fetchall()
                if r:
                    subid1 = SubIssue(string=r[0][0])
                modelDB.dbCursor.execute(query, id=axid2.getIDString(), dt=date)
                r = modelDB.dbCursor.fetchall()
                if r:
                    subid2 = SubIssue(string=r[0][0])
                if subid1 is None:
                    marketAXID = axiomaID2
                elif subid2 is None:
                    marketAXID = axiomaID1
                elif subid1 != subid2:
                    if subid1 in exposureAssets and subid2 in exposureAssets:
                        # special case for IE assets with GB line using same SEDOL
                        if country == 'IE' and countryDict.get(axiomaID2) == 'GB':
                            logging.info('SEDOL %s is used by Goldman for %s and by Axioma for %s',
                                         sedol, axiomaID1, axiomaID2)
                        else:
                            logging.fatal('AXID for ticker "%s" is %s while AXID for %s "%s" is %s',
                                          ticker, axiomaID1, useCUSIP and 'CUSIP' or 'SEDOL',
                                          useCUSIP and cusip or sedol, axiomaID2)
                            raise ValueError('AXID for ticker "%s" is %s while AXID for %s "%s" is %s' % \
                                (ticker, axiomaID1, useCUSIP and 'CUSIP' or 'SEDOL',
                                          useCUSIP and cusip or sedol, axiomaID2))
                    else:
                        logging.warning('AXID for ticker "%s" is %s while AXID for %s "%s" is %s',
                                     ticker, axiomaID1, useCUSIP and 'CUSIP' or 'SEDOL',
                                     useCUSIP and cusip or sedol, axiomaID2)
                        if subid1 in exposureAssets:
                            marketAXID = axiomaID1
                        elif subid2 in exposureAssets:
                            marketAXID = axiomaID2
                        else:
                            marketAXID = None
                        if marketAXID:
                            logging.warning("Using %s because it's in exposure table", marketAXID)
        if marketAXID is None:
            continue
        axid = marketModelActive.pop(marketAXID, None)
        if axid is None:
            if marketAXID not in marketModelMap:
                logging.warning('No model ID for market ID %s', marketAXID.getIDString())
            else:
                logging.warning('Duplicate entry in file for market ID %s', marketAXID.getIDString())
            continue
        axid = axid.getPublicID()
        n1 = n1 / 10000 # convert from basis points to dollars
        n3 = n3 / 10000 # convert from basis points to dollars
        n2 = n2 / 10000 / sqrt(1000) # convert from basis points to dollars, plus additional conversion
        numberString = '%#.5e!%#.5e!%#.5e' % (n1, n2, n3)
        logging.debug('writing: %s|%s' % (axid, numberString))
        outFile.write('%s|%s|%d\n' % (axid, wombat3(numberString),wombat(wombat3(numberString))))
    outFile.close()
    createVersionFile(options, 'Shortfall-%s' % country, modelDB)

def writeModel(modelDB, marketDB, options, date, riskModel): 
    rmi = riskModel.getRiskModelInstance(date, modelDB)
    if rmi == None:
        logging.fatal('No risk model instance %s on %s', riskModel.name, date)
        raise KeyError('No risk model instance on %s' % date)
    if (not rmi.has_exposures) or (not rmi.has_risks):
        if options.allowPartialModels:
            logging.warning('Partial risk model instance %s on %s. Continuing anyway', riskModel.name, date)
        else:
            logging.fatal('Partial risk model instance %s on %s', riskModel.name, date)
            raise KeyError('Partial risk model instance on %s' % date)
    if len(riskModel.rmg) > 1:
        SCM = False
    else:
        SCM = True
    scmCoverage = SCM
    if hasattr(riskModel, 'coverageMultiCountry'):
        scmCoverage= (riskModel.coverageMultiCountry==False)
    
    expM = riskModel.loadExposureMatrix(rmi, modelDB)
    expM.fill(0.0)
    estU = riskModel.loadEstimationUniverse(rmi, modelDB)
    if rmi.has_risks:
        (svDataDict, specCov) = riskModel.loadSpecificRisks(rmi, modelDB)
    else:
        # extracting a partial model. Create fake specific risk of 1.0 for each asset in the exposure matrix
        svDataDict = dict((sid, 1.0) for sid in expM.getAssets())
        specCov = dict()
    exposureAssets = sorted(svDataDict.keys())
    (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
        rmi, modelDB)
    factorReturns = riskModel.loadFactorReturns(date, modelDB)
    cumReturns = riskModel.loadCumulativeFactorReturns(date, modelDB)
    factorReturnMap = dict(zip(factorReturns[1], 100.0 * factorReturns[0]))
    cumReturnMap = dict(zip(cumReturns[1], cumReturns[0]))
    # write characteristics
    createVersionFile(options, 'Model-%s' % riskModel.name, modelDB)
    createPartialModelIndicatorFile(options, 'Model-%s' % riskModel.extractName, riskModel)

    out = open('%s/Model-%s_Characteristic' %
               (options.targetDir, riskModel.extractName), 'w')
    estUDict = dict(zip(estU, [1] * (len(estU))))
    # exclude assets which shouldn't be extracted
    excludes = modelDB.getProductExcludedSubIssues(date)
    if options.preliminary or not rmi.is_final:
        excludes.extend(modelDB.getDRSubIssues(rmi))
    for e in excludes:
        svDataDict.pop(e, None)
    exposureAssets = list(set(exposureAssets) - set(excludes))
    if len(riskModel.industries):
        classifications = riskModel.industryClassification.getAssetConstituents(modelDB, exposureAssets, date)
        hasIndustries = True
    else:
        classifications = set()
        hasIndustries = False
    if riskModel.modelHack.nonCheatHistoricBetas:
        histbetas = modelDB.getPreviousHistoricBeta(date, exposureAssets)
    else:
        histbetas = modelDB.getPreviousHistoricBetaOld(date, exposureAssets)
    predbetas = modelDB.getRMIPredictedBeta(rmi)
    totalrisks = modelDB.getRMITotalRisk(rmi)
    if hasattr(riskModel, 'legacyMCapDates') and riskModel.legacyMCapDates is False:
        marketCaps, mcapDates = AssetProcessor.robustLoadMCaps(
                date, exposureAssets, None, modelDB, marketDB)
    else:
        mcapDates = modelDB.getDates(riskModel.rmg, date, 19)
        marketCaps = modelDB.getAverageMarketCaps(
            mcapDates, exposureAssets, None)
    if options.newRiskFields:
        estUWeightdict = buildESTUWeights(modelDB, estU, date, riskModel)
        specRtnsMatrix = modelDB.loadSpecificReturnsHistory(rmi.rms_id, exposureAssets, [date])
        specRtnsMatrix.data *= 100
        newSpecReturns = recomputeSpecificReturns(date, exposureAssets, riskModel, modelDB, marketDB, False)
        newSpecReturns  *= 100 # conver to percentages

    if options.newBetaFields or options.histBetaNew or options.newRMMFields:
        # fish out all the various beta fields now
        #assets = modelDB.getRiskModelInstanceUniverse(rmi)

        data = AssetProcessor.process_asset_information(
                    date, exposureAssets, riskModel.rmg, modelDB, marketDB,
                    checkHomeCountry=(scmCoverage==False),
                    legacyDates=riskModel.legacyMCapDates,
                    numeraire_id=riskModel.numeraire.currency_id,
                    forceRun=riskModel.forceRun)

        historicBeta_home = modelDB.getHistoricBetaDataV3(
                    date, data.universe, field='value', home=1, rollOverData=True)
        historicBeta_trad = modelDB.getHistoricBetaDataV3(
                    date, data.universe, field='value', home=0, rollOverData=True)

        nobs_home = modelDB.getHistoricBetaDataV3(
                    date, data.universe, field='nobs', home=1, rollOverData=True)

        # load the predicted beta fields now
        localCurrBeta = modelDB.getRMIPredictedBetaV3(rmi, field='local_num_beta')
        localBeta = modelDB.getRMIPredictedBetaV3(rmi, field='local_beta')
        globalBeta = modelDB.getRMIPredictedBetaV3(rmi, field='global_beta')

    if options.newRMMFields:
        # data is filled out already
        # fish out all the descriptor values to be stored.  Use a temporary method for now
        descriptorMap={}
        for desc_name, desc_id in modelDB.getAllDescriptors():
            if desc_name.lower() in RMM_MODEL_FIELDS_DICT:
                RMM_MODEL_FIELDS_DICT[desc_name.lower()] = desc_id
        for desc_name, desc_id in RMM_MODEL_FIELDS_DICT.items():
            if desc_id == -1:
                logging.fatal('No descriptor for %s', desc_name)
            logging.info('Loading data %s for %s', desc_name, date)
            ###desc = modelDB.loadDescriptorData(date, exposureAssets, riskModel.numeraire.currency_code, desc_id, rollOverData=False)
            desc =  modelDB.loadLocalDescriptorData(date, data.universe, data.currencyAssetMap, [desc_id])
            for idx, e in enumerate(data.universe):
                descriptorMap[(e,desc_id)] = desc.data[idx,0]

        estuMap=modelDB.getEstuMappingTable(riskModel.rms_id)
        if 'ChinaA' in estuMap:
            chinaASubIssues=modelDB.getRiskModelInstanceESTU(rmi, estu_idx=estuMap['ChinaA'].id)
            chinaAMap = dict([(i,1) for i in  chinaASubIssues])
        else:
            chinaAMap = {}
    if options.cnRMMFields:
        chinaOffshoreMap = {}
        if 'ChinaOff' in estuMap:
            chinaOffSubIssues=modelDB.getRiskModelInstanceESTU(rmi, estu_idx=estuMap['ChinaOff'].id)
            chinaOffshoreMap = dict([(i,1) for i in  chinaOffSubIssues])

    for (idx, subIssue) in enumerate(exposureAssets):
        values = dict()
        source = ''
        if hasIndustries and subIssue in classifications:
            src_id = classifications[subIssue].src
            source = 'Axioma'
            if (src_id>=300 and src_id<=399):
                source = 'GICS-Direct'
        values['id'] = subIssue.getModelID().getPublicID()
        values['beta_pred'] = predbetas.get(subIssue, '')
        #values['beta_hist'] = histbetas.get(subIssue, '')
        ### RSK-3973 play games regarding the historic beta

        if options.histBetaNew:
            if idx==0:
                logging.info('Using HistoricBeta new style...')
            if SCM:
                values['beta_hist'] = historicBeta_trad.get(subIssue, historicBeta_home.get(subIssue,''))
            else:
                values['beta_hist'] = historicBeta_home.get(subIssue, '')
        else:
            values['beta_hist'] = histbetas.get(subIssue, '')
        values['risk_spec'] = numOrNull(svDataDict[subIssue], prec=16)
        values['risk_total'] = totalrisks.get(subIssue, '')
        values['estu'] = estUDict.get(subIssue, 0)
        values['mcap'] = numOrNull(marketCaps[idx])
        values['src'] = source
        if options.newRiskFields:
            if subIssue in estUWeightdict:
                estUWeightdict[subIssue] *= 100
            values['estuWeight'] = numOrNull(estUWeightdict.get(subIssue))
            if options.fileFormatVersion >= 4.0:
                values['specRtn'] = numOrNull(newSpecReturns[idx,0])
            else:
                values['specRtn'] = numOrNull(specRtnsMatrix.data[idx, 0])
            out.write('%(id)s|%(beta_pred)s|%(beta_hist)s|%(risk_spec)s|%(risk_total)s|%(estu)s|%(estuWeight)s|%(specRtn)s|%(mcap)s|%(src)s' % values)
        else:
            out.write('%(id)s|%(beta_pred)s|%(beta_hist)s|%(risk_spec)s|%(risk_total)s|%(estu)s|%(mcap)s|%(src)s' % values)
        if options.newBetaFields:
            # fields to populate are:
            # Regional = num_of_betas | hbeta_trad | bheta_home | pbeta_local_hedged| pbeta_local_unhedged | pbeta_global_unhedged
            # SCM      = num_of_betas | hbeta_trad | bheta_home | pbeta_local_hedged 
            values['num_beta_returns'] = nobs_home.get(subIssue, '')
            values['hbeta_trad'] = numOrNull(historicBeta_trad.get(subIssue, None) or historicBeta_home.get(subIssue, None))
            values['hbeta_home'] = numOrNull(historicBeta_home.get(subIssue, None))
            if values['num_beta_returns']:
                out.write('|%d' % int(values['num_beta_returns']))
            else:
                out.write('|')
            out.write('|%(hbeta_trad)s|%(hbeta_home)s' % values)
            if SCM:
                values['pbeta_local_hedged'] = numOrNull(localCurrBeta.get(subIssue, None))
                out.write('|%(pbeta_local_hedged)s' % values)
            else:
                values['pbeta_local_unhedged'] = numOrNull(localCurrBeta.get(subIssue, None))
                values['pbeta_local_hedged'] = numOrNull(localBeta.get(subIssue, None))
                values['pbeta_global_unhedged'] = numOrNull(globalBeta.get(subIssue, None))
                out.write('|%(pbeta_local_hedged)s|%(pbeta_local_unhedged)s|%(pbeta_global_unhedged)s' % values)
        if options.newRMMFields:
            #  columns are:   isc_adv_score    isc_ipo_score   isc_ret_score  
            #  columns are:   market_sensitivity_104w    
            for desc_values in RMM_MODEL_FIELDS:
                desc_id = RMM_MODEL_FIELDS_DICT[desc_values]
                values[desc_values] = numOrNull(descriptorMap.get((subIssue,desc_id),None))
                out.write('|%s' % values[desc_values])

            out.write('|%s' % numOrNull(chinaAMap.get(subIssue,None)))
        if options.cnRMMFields:
            out.write('|%s' % numOrNull(chinaOffshoreMap.get(subIssue,None)))
        out.write('\n')
    out.close()
    # write factor types that have factors attached
    out = open('%s/Model-%s_Factor_type' %
               (options.targetDir, riskModel.extractName), 'w')
    fTypeIdxMap = dict()
    fTypeNum = 1
    for (fType, idxList) in expM.factorIdxMap_.items():
        if len(idxList) > 0:
            fTypeIdxMap[fType] = fTypeNum
            fTypeNum += 1
            out.write('%s|%s\n' % (fType.name, fType.description))
    out.close()
    # write factors and their classification
    out = open('%s/Model-%s_Factor' %
               (options.targetDir, riskModel.extractName), 'w')
    classFile = open('%s/Model-%s_Classification'
                     % (options.targetDir, riskModel.extractName), 'w')
    hierFile = open('%s/Model-%s_Classification_hierarchy'
                    % (options.targetDir, riskModel.extractName), 'w')
    rootClassifications = riskModel.getClassificationRoots(modelDB)
    curNumber = 0
    suffix=''
    if hasattr(options, 'factorSuffix') and options.factorSuffix is not None:
        suffix=options.factorSuffix
    #print("suffix=%s" % suffix)
    if suffix != '' and riskModel:
        factorDict=riskModel.factorTypeDict
        factorNames=getFixedFactorNames([f.name for f in factors], suffix, factorDict )
    else:
        factorNames=[f.name for f in factors]

    #print(factorNames)
    for idx, factor in enumerate(factors):
        assert(expM.getFactorIndex(factor.name) != None)
        type_id = fTypeIdxMap[expM.getFactorType(factor.name)]
        facRet = numOrNull(factorReturnMap[factor])
        cumRet = numOrNull(cumReturnMap[factor], 16)
        # for some models put a suffix in the factor names
        factorName=factorNames[idx]
        # hack to see if description has to be fixed up as well
        if factorName != factor.name:
            desc = factor.description + suffix
        else:
            desc = factor.description
        out.write('%s|%s|%s|%d|%s\n'
                  % (factorName, desc, facRet,
                     type_id, cumRet))
        if len(rootClassifications) > 0:
            classFile.write('%s|%s|%d|%d\n' % (factorName, factor.description,
                                               1, 0))
            curNumber += 1
    out.close()
    
    unprocessed = rootClassifications
    descIdxMap = dict(zip([f.description for f in factors], list(range(len(factors)))))
    processedClassIDs = set()
    idIdxMap = dict()
    curNumberList = [curNumber]
    for r in rootClassifications:
        writeModelHierarchy(r, classFile, hierFile, curNumberList,
                            processedClassIDs, idIdxMap, descIdxMap,
                            riskModel, modelDB)
    classFile.close()
    hierFile.close()
    
    # write covariances
    out = open('%s/Model-%s_Covariance' %
               (options.targetDir, riskModel.extractName), 'w')
    for i in range(len(factors)):
       for j in range(i+1):
          out.write('%d|%d|%.16g\n' % (i, j, factorCov[i,j]))
    out.close()
    # write specific covariances
    out = open('%s/Model-%s_SpecificCovariance' %
               (options.targetDir, riskModel.extractName), 'w')
    expAssets = set(exposureAssets)
    for (sid1, sid1Covs) in specCov.items():
        for (sid2, value) in sid1Covs.items():
            if sid1 in expAssets and sid2 in expAssets:
                out.write('%s|%s|%.16g\n' % (
                        sid1.getModelID().getPublicID(),
                        sid2.getModelID().getPublicID(), value))
    out.close()
    # write exposures
    out = open('%s/Model-%s_Exposure' %
               (options.targetDir, riskModel.extractName), 'w')
    mat = expM.getMatrix()
    numFields = getNumFactors(riskModel)
    marketCapMap = dict(zip(exposureAssets, marketCaps))
    for (j, sid) in enumerate(expM.getAssets()):
        if sid not in svDataDict:
            continue
        out.write('%s' % sid.getModelID().getPublicID())
        fields = 0
        for (fIdx, fval) in enumerate(mat[:,j]):
            if fval != 0.0:
                out.write('|%d|%.16g' % (fIdx, fval))
                fields += 1
        assert(fields <= numFields)
        out.write('||' * (numFields - fields))
        out.write('|%s' % numOrNull(marketCapMap[sid]))
        out.write('\n')
    logging.debug('add cash assets')
    # add cash assets with exposure to corresponding currency factor
    for currencyFactor in expM.getFactorNames(ExposureMatrix.CurrencyFactor):
        cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
        out.write('%s' % cashSubIssue.getModelID().getPublicID())
        fIdx = expM.getFactorIndex(currencyFactor)
        out.write('|%d|1' % fIdx)
        out.write('||' * (numFields - 1))
        out.write('|\n')

    out.close()
    # write exposure sql template
    out = open('%s/Model-%s_create' %
               (options.targetDir, riskModel.extractName), 'w')
    out.write('CREATE TABLE EXPOSURE ( axioma_id VARCHAR(9) NOT NULL,')
    for j in range(numFields):
        out.write('factor_id%(factor)d SMALLINT, value%(factor)d REAL,'
                  % {'factor' : j})
    out.write('mcap REAL, CONSTRAINT AXIOMAID_FACTORID_PK PRIMARY KEY (axioma_id)')
    for j in range(numFields):
        out.write(', CONSTRAINT EXP_FACTOR_REF%(factor)d FOREIGN KEY (factor_id%(factor)d) REFERENCES FACTOR(id)' % {'factor': j})
    out.write(');')
    out.close()
    out = open('%s/Model-%s_load' %
               (options.targetDir, riskModel.extractName), 'w')
    for j in range(numFields):
        out.write('ALTER TABLE EXPOSURE DROP CONSTRAINT EXP_FACTOR_REF%(factor)d;\n'
                  % {'factor' : j})
    out.close()

    # write risk model instance
    out = open('%s/Model-%s_RiskModelInstance' %
               (options.targetDir, riskModel.extractName), 'w')
    out.write('%s|%d|%d|%d|%d'% (str(date)[:10],rmi.has_exposures, rmi.has_returns, rmi.has_risks, rmi.is_final))
    out.write('\n')
    out.close()
     

def writeModelHierarchy(node, classFile, hierFile, currNumberList,
                        processedClassIDs, idIdxMap, descIdxMap,
                        riskModel, modelDB):
    if node.id in processedClassIDs:
        return
    if node.isLeaf:
        idIdxMap[node.id] = descIdxMap[node.description]
        processedClassIDs.add(node.id)
        return
    children = riskModel.getClassificationChildren(node, modelDB)
    for c in children:
        writeModelHierarchy(c, classFile, hierFile, currNumberList,
                            processedClassIDs, idIdxMap, descIdxMap,
                            riskModel, modelDB)
    idIdxMap[node.id] = currNumberList[0]
    currNumberList[0] += 1
    classFile.write('%s|%s|%d|%d\n'
                    % (node.name, node.description,
                       node.isLeaf, node.isRoot))
    for c in children:
        hierFile.write('%d|%d|%.16g\n' %
                       (idIdxMap[node.id], idIdxMap[c.id], c.weight))
    processedClassIDs.add(node.id)


def writePortFamily(modelDB, marketDB, options, date, familyName):
    family = modelDB.getModelPortfolioFamily(date, familyName)
    if family == None or len(family) == 0:
        raise LookupError('Unknown model portfolio family %s' % familyName)
    #print('....', family)
    createVersionFile(options, 'ModelPortfolio-%s' % familyName, modelDB)
    portfileName = '%s/ModelPortfolio-%s_ModelPortfolio' % (options.targetDir, familyName)
    expfileName = '%s/ModelPortfolio-%s_ModelPortfolio_exposure' % (options.targetDir, familyName)
    outMport = open(portfileName, 'w')
    outExposure = open(expfileName, 'w')
    mportNumber = 0
    for mp in family:
        member = modelDB.getModelPortfolioByID(mp.id, date) 
        #print('...member...', member)
        const = modelDB.getModelPortfolioConstituents(date, mp.id)
        if len(const) > 0:
            outMport.write('%s|%s\n' % (member.name, member.description))
            mportNumber += 1
            for (asset, weight) in const:
                outExposure.write('%s|%d|%.16g\n' % (asset.getModelID().getPublicID(), mportNumber, weight * 100.0))
        else:
            logging.error('No consituents for Model Portfolio %s on %s', mp.name, date) 
    outMport.close()
    outExposure.close()
    return portfileName, expfileName

def writePortFamily2(modelDB, marketDB, options, date, familyName, client=None):
    family = modelDB.getModelPortfolioFamily(date, familyName)
    if family == None or len(family) == 0:
        return None,None
        #raise LookupError, 'Unknown model portfolio family %s' % familyName
    outIndexName = '%s/Index-%s_Index' % (options.targetDir, familyName)
    outIndex = open(outIndexName, 'w')
    outExposureName = '%s/Index-%s_Index_exposure' % (options.targetDir, familyName)
    outExposure = open(outExposureName, 'w')
    indexNumber = 0
    # if client is specified, find the list of indexes in that family the client is entitled to get
    if client:
        query="""select member_id from mdl_port_member_client where client='%s' and from_dt <= :dt and :dt < thru_dt""" % client
        modelDB.dbCursor.execute(query, dt=date)
        clientIDs=[r[0] for r in modelDB.dbCursor.fetchall()]
        logging.info('Setting up indexes for %s', client) 
    for mp in family:
        member = modelDB.getModelPortfolioByID(mp.id, date) 
        if client and member.id not in clientIDs:
            logging.debug('Ignore %s since %s is not eligible to get it', member.name, client)
            continue
        logging.info('adding member %s for %s', member, '' if client else client)
        const = modelDB.getModelPortfolioConstituents(date, mp.id)
        if len(const) > 0:
            outIndex.write('%s|%s|PERCENT\n' % (member.name, member.description))
            indexNumber += 1
            for (asset, weight) in const:
                outExposure.write('%s|%d|%.16g\n' % (asset.getModelID().getPublicID(), indexNumber, weight * 100.0))
        else:
            logging.error('No consituents for Model Portfolio %s on %s', mp.name, date) 

    outIndex.close()
    outExposure.close()
    return outIndexName, outExposureName

def writeIndexFamily(modelDB, marketDB, options, date, familyName, nextDayOpen=False):
    family = marketDB.getIndexFamily(familyName)
    if family == None:
        raise LookupError('Unknown index family %s' % familyName)
    indices = marketDB.getIndexFamilyIndices(family, date, True)
    createVersionFile(options, 'Index-%s' % family.name, modelDB)
    outIndex = open('%s/Index-%s_Index' %
               (options.targetDir, family.name), 'w')
    outExposure = open('%s/Index-%s_Index_exposure' %
               (options.targetDir, family.name), 'w')
    indexNumber = 0
    issueMapPairs = modelDB.getIssueMapPairs(date)
    # if the nextDayOpen flag is sent down, make sure to use the date at least one day 
    # greater than the current one to find the indexes
    if nextDayOpen:
        found=False
        for dayidx in range(14)[1:]:
           d1=date+datetime.timedelta(days=dayidx)
           if d1.isoweekday() in [6,7]:
               continue
           logging.info("Looking at date %s", d1)
           indicesMissing=[]       
           for i in range(len(indices)):
              idx = indices[i]
              bcomp = modelDB.getIndexConstituents(idx.name, d1, marketDB,
                                             issueMapPairs=issueMapPairs)
              if len(bcomp) > 0: 
                  found=True
                  outIndex.write('%s|%s|%s\n' % (idx.name, idx.description,
                                           idx.unit))
                  indexNumber += 1
                  for (asset, weight) in bcomp:
                    if idx.unit == 'PERCENT':
                        weight = weight * 100.0
                    outExposure.write('%s|%d|%.16g\n' %
                                  (asset.getPublicID(), indexNumber, weight))
              else:   
                  indicesMissing.append(idx)
           if found:
              logging.info("Found constituents(derby) for %s on %s/%s %s/%s", familyName, d1, d1.isoweekday(), date, date.isoweekday())
              for missingidx in indicesMissing:
                logging.error('No constituents for Benchmark %s on %s' , missingidx.name, date)
              break
        if not found:
            for i in range(len(indices)):
               idx=indices[i]
               logging.error('No constituents for Benchmark %s on %s' , idx.name, date)

    else:
        for i in range(len(indices)):
            idx = indices[i]
            bcomp = modelDB.getIndexConstituents(idx.name, date, marketDB,
                                             issueMapPairs=issueMapPairs)
            if len(bcomp) > 0:
                outIndex.write('%s|%s|%s\n' % (idx.name, idx.description,
                                           idx.unit))
                indexNumber += 1
                for (asset, weight) in bcomp:
                    if idx.unit == 'PERCENT':
                        weight = weight * 100.0
                    outExposure.write('%s|%d|%.16g\n' %
                                  (asset.getPublicID(), indexNumber, weight))
            else:
                logging.error('No constituents for Benchmark %s on %s' , idx.name, date)
    outIndex.close()
    outExposure.close()


def writeMarketClassification(marketDB, modelDB, options, date, rootName):
    family = marketDB.getClassificationFamily('INDUSTRIES')
    member  = marketDB.getClassificationFamilyMembers(family)
    thisMember = [mem for mem in member if mem.name=='ICB'][0]

    # if this is AxModelGICS2016 use the hardcoded date
    legacy = False
    if rootName=='AXModelICB':
        if date >= datetime.date(2020,9,21):
            logging.info('Hardcoding revision date to be 2020-09-21 for ICB')
            thisRevision = marketDB.getClassificationMemberRevision(thisMember, datetime.date(2020,9,21))
        elif date >= datetime.date(2004,9,30):
            logging.info('Hardcoding revision date to be 2004-09-30 for ICB')
            thisRevision = marketDB.getClassificationMemberRevision(thisMember, datetime.date(2004,9,30))
        else:
            logging.info('Hardcoding revision date to be 1999-12-31 for ICB')
            thisRevision = marketDB.getClassificationMemberRevision(thisMember, datetime.date(1999,12,31))
            legacy = True
        #print(thisRevision)
        className='ICB'
    else:
        return None

    rootClassification = marketDB.getClassificationRevisionRoot(thisRevision)
    unprocessed = [rootClassification]
    curNumber = 1
    classDict={}
    subIssues = modelDB.getAllActiveSubIssues(date, inModels=True)
    excludes = modelDB.getProductExcludedSubIssues(date)
    subIssues = list(set(subIssues) - set(excludes))

    allIssues=[s.getModelID() for s in subIssues]
    industryMap=modelDB.getMktAssetClassifications(thisRevision, allIssues, date, marketDB, None)

    flatFile=False
    fileNameList=[]

    newFile = open('%s/Classification-%s_Data-%s.txt' % (options.targetDir, className, str(date).replace('-','')), 'w')
    if not legacy:
        newFile.write('METAGROUP NAME|%(class)s.Industry|%(class)s.Supersector|%(class)s.Sector|%(class)s.Subsector\n' % {'class':className})
        newFile.write('METAGROUP DESC|%(class)s Industry|%(class)s Supersector|%(class)s Sector|%(class)s Subsector\n' % {'class':className})
        newFile.write('NAME PREFIX|%(class)s.I.|%(class)s.U.|%(class)s.S.|%(class)s.B.\n' % {'class':className})
    else:
        newFile.write('METAGROUP NAME|%(class)s.Industry\n' % {'class':className})
        newFile.write('METAGROUP DESC|%(class)s Industry\n' % {'class':className})
        newFile.write('NAME PREFIX|%(class)s.I.\n' % {'class':className})
    newFile.write('\n')

    clsIdMap = dict()
    while len(unprocessed) > 0:
        curr = unprocessed.pop()
        curr.myNumber = curNumber
        clsIdMap[curr.id] = curr
        classDict[curNumber]=[curr.name,curr.description]
        if not curr.isLeaf:
            children = marketDB.getClassificationChildren(curr)
            for i in children:
                i.parentNumber = curr.myNumber
            unprocessed.extend(children)
        curNumber += 1
    for (asset, clsStruct) in sorted(industryMap.items()):
        curr = clsIdMap[clsStruct.classification_id]
        weight = clsStruct.weight
        source = 'ICB'
        if not legacy:
            level1=clsStruct.classification.levelParent[1].name
            level2=clsStruct.classification.levelParent[2].name
            level3=clsStruct.classification.levelParent[3].name
            level4=clsStruct.classification.name
            newFile.write('%s|%s|%s|%s|%s\n' % (asset.getPublicID(),   level1, level2, level3, level4))
        else:
            level=clsStruct.classification.name
            newFile.write('%s|%s\n' % (asset.getPublicID(), level))
    newFile.close()
    return fileNameList

def writeMdlClassification(modelDB, options, date, rootName):
    family = modelDB.getMdlClassificationFamily('INDUSTRIES')
    member = [m for m in modelDB.getMdlClassificationFamilyMembers(family) if m.name == 'GICSIndustries'][0]
    # if this is AxModelGICS2016 use the hardcoded date
    if rootName=='AXModelGICS2016':
        logging.info('Hardcoding revision date to be 2016-09-01 for AXModelGICS2016')
        rev = modelDB.getMdlClassificationMemberRevision(member, datetime.date(2016,9,1))
        className='AxModelGICS2016'
    elif rootName=='AXModelGICS2018':
        logging.info('Hardcoding revision date to be 2018-09-29 for AXModelGICS2018')
        rev = modelDB.getMdlClassificationMemberRevision(member, datetime.date(2018,9,29))
        className='AxModelGICS2018'
    else:
        rev = modelDB.getMdlClassificationMemberRevision(member, date)
    rootClassification = modelDB.getMdlClassificationRevisionRoot(rev)
    unprocessed = [rootClassification]
    curNumber = 1
    classDict={}
    subIssues = modelDB.getAllActiveSubIssues(date, inModels=True)
    excludes = modelDB.getProductExcludedSubIssues(date)
    subIssues = list(set(subIssues) - set(excludes))

    industryMap = modelDB.getMdlAssetClassifications(rev, subIssues, date)


    flatFile=False
    if rootName=='INDUSTRIES':
        rootName = 'AXModelGICS2016'
    createVersionFile(options, 'Classification-%s' % rootName, modelDB)
    classFile = open('%s/Classification-%s_Classification'
                     % (options.targetDir, rootName), 'w')
    hierFile = open('%s/Classification-%s_Classification_hierarchy'
                    % (options.targetDir, rootName), 'w')
    assetFile = open('%s/Classification-%s_Asset_classification'
                    % (options.targetDir, rootName), 'w')
    srcFile = open('%s/Attribute-%s_Source-%s.txt' % (options.targetDir, rootName, str(date).replace('-','')), 'w')
    srcFile.write('#DataDate: %s\n' % str(date))
    gmtime = time.gmtime(time.mktime(modelDB.revDateTime.timetuple()))
    utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
    srcFile.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
    srcFile.write('#Columns: AxiomaID|Industry Source\n')
    srcFile.write('#Type: ID|Attribute\n')
    srcFile.write('#Unit: ID|Text\n')
    fileNameList=[]

    newFile = open('%s/Classification-%s_Data-%s.txt' % (options.targetDir, rootName, str(date).replace('-','')), 'w')
    newFile.write('METAGROUP NAME|%(gicsClass)s.SECTORS|%(gicsClass)s.INDUSTRYGROUPS|%(gicsClass)s.INDUSTRIES\n' % {'gicsClass':className})
    newFile.write('METAGROUP DESC|%(gicsClass)s Sectors|%(gicsClass)s Industry Groups|%(gicsClass)s Industries\n' % {'gicsClass':className})
    newFile.write('NAME PREFIX|%(gicsClass)s.S.|%(gicsClass)s.G.|%(gicsClass)s.I.\n' % {'gicsClass':className})
    newFile.write('\n')

    clsIdMap = dict()
    while len(unprocessed) > 0:
        curr = unprocessed.pop()
        curr.myNumber = curNumber
        clsIdMap[curr.id] = curr
        classDict[curNumber]=[curr.name,curr.description]
        if flatFile:
            classFile.write('%s.%d|%s|%s|%d|%d\n' 
                        % (rootName,curNumber,curr.name, curr.description,
                           curr.isLeaf, curr.isRoot))
        else:
            classFile.write('%s.%s|%s|%d|%d\n'
                        % (rootName,curr.name, curr.description,
                           curr.isLeaf, curr.isRoot))
        if not curr.isRoot:
            if flatFile:
                hierFile.write('%d|%d|%s|%s|%.16g\n' % (curr.parentNumber, curr.myNumber, classDict[curr.parentNumber][0],classDict[curr.myNumber][0],curr.weight))
            else:
                hierFile.write('%d|%d|%.16g\n' %
                           (curr.parentNumber, curr.myNumber, curr.weight))
        if not curr.isLeaf:
            children = modelDB.getMdlClassificationChildren(curr)
            for i in children:
                i.parentNumber = curr.myNumber
            unprocessed.extend(children)
        curNumber += 1
    classFile.close()
    hierFile.close()
    for (sid, clsStruct) in sorted(industryMap.items()):
        asset = sid.getModelID()
        curr = clsIdMap[clsStruct.classification_id]
        weight = clsStruct.weight
        if flatFile:
            assetFile.write('%s|%d|%s|%.16g\n'% (asset.getPublicID(), curr.myNumber, classDict[curr.myNumber][0],weight))
        else:
            assetFile.write('%s|%d|%.16g\n' % (asset.getPublicID(), curr.myNumber, weight))
            source = 'Axioma'
            if clsStruct.src >= 300 and clsStruct.src <= 399:
                source = 'GICS-Direct'
            srcFile.write('%s|%s\n' % (asset.getPublicID(), source))
            level1=clsStruct.classification.levelParent[1].name
            level2=clsStruct.classification.levelParent[2].name
            if level1[-2:] == '-S':
                level1 = level1[:-2]
            if level2[-2:] == '-G':
                level2 = level2[:-2]
            newFile.write('%s|%s|%s|%s\n' % (asset.getPublicID(),   level1, level2, classDict[curr.myNumber][0]))
    assetFile.close()
    srcFile.close()
    newFile.close()
    return fileNameList

def writeClassification(modelDB, marketDB, options, date, rootName):
    if rootName in ['AXModelGICS2016','AXModelGICS2018']:
        return writeMdlClassification(modelDB, options, date, rootName)
    if rootName in ['AXModelICB']:
        return writeMarketClassification(marketDB,modelDB, options, date, rootName)
    industryFamily = marketDB.getClassificationFamily('INDUSTRIES')
    assert(industryFamily != None)
    industryMembers = marketDB.getClassificationFamilyMembers(industryFamily)
    rootMember = dict([(i.name,i) for i in industryMembers]).get(rootName)
    if rootMember is None:
        return
    industryRev = marketDB.getClassificationMemberRevision(rootMember, date)
    if industryRev is None:
        return
    # Get current root for this classification
    rootClassification = marketDB.getClassificationRevisionRoot(industryRev)
    if rootClassification == None:
        raise LookupError('Unknown classification %s' % rootName)
    elif not rootClassification.isRoot:
        raise LookupError('Classification %s is not root' % rootName)
    if hasattr(options, 'writeText') and options.writeText==True:
        flatFile=True
        dtstr=str(date).replace('-','')
        classFileName= '%s/Classification-%s-%s.cls' % (options.targetDir, rootName, dtstr)
        classFile = open(classFileName, 'w')
        hierFileName= '%s/Classification-%s-%s.hry' % (options.targetDir, rootName, dtstr)
        hierFile = open(hierFileName, 'w')
        assetFileName = '%s/Classification-%s-%s.asc' % (options.targetDir, rootName, dtstr)
        assetFile = open(assetFileName, 'w')
        fileNameList=[classFileName,hierFileName,assetFileName]


        # write headers for flat file
        for f in [assetFile,classFile,hierFile]:
            f.write('#DataDate: %s\n' % str(date)[:10])
            # write createDate in UTC
            gmtime = time.gmtime(time.mktime(modelDB.revDateTime.timetuple()))
            utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
            f.write('#CreationTimestamp: %sZ\n' % utctime.strftime('%Y-%m-%d %H:%M:%S'))
            f.write("#FlatFileVersion:3.3\n")
            if f==classFile:
                f.write('#Columns: ID|Name|Description|is leaf?|is root?\n')
            elif f==hierFile:
                f.write('#Columns: ParentID|Child ID|Parent Name|Child Name|Weight\n')
            elif f==assetFile:
                f.write('#Columns: Axioma ID|Classification ID|Classification Name|Weight\n')
    else:
        flatFile=False
        createVersionFile(options, 'Classification-%s' % rootName, modelDB)
        classFile = open('%s/Classification-%s_Classification'
                     % (options.targetDir, rootName), 'w')
        hierFile = open('%s/Classification-%s_Classification_hierarchy'
                    % (options.targetDir, rootName), 'w')
        assetFile = open('%s/Classification-%s_Asset_classification'
                    % (options.targetDir, rootName), 'w')
        fileNameList=[]
    unprocessed = [rootClassification]
    curNumber = 1
    classDict={} 
    subIssues = modelDB.getAllActiveSubIssues(date, inModels=True)
    excludes = modelDB.getProductExcludedSubIssues(date)
    # Don't exclude from classification jar based on risk model,
    # as we might be extracting multiple models at once
    #if options.preliminary:
    #    excludes.extend(modelDB.getDRSubIssues(rmi))
    subIssues = list(set(subIssues) - set(excludes))
    industryMap = modelDB.getMktAssetClassifications(
        industryRev, subIssues, date, marketDB)
    clsIdMap = dict()
    while len(unprocessed) > 0:
        curr = unprocessed.pop()
        curr.myNumber = curNumber
        clsIdMap[curr.id] = curr
        classDict[curNumber]=[curr.name,curr.description]
        if flatFile:
            classFile.write('%d|%s|%s|%d|%d\n'
                        % (curNumber,curr.name, curr.description,
                           curr.isLeaf, curr.isRoot))
        else:
            classFile.write('%s|%s|%d|%d\n'
                        % (curr.name, curr.description,
                           curr.isLeaf, curr.isRoot))
        if not curr.isRoot:
            if flatFile:
                hierFile.write('%d|%d|%s|%s|%.16g\n' % (curr.parentNumber, curr.myNumber, classDict[curr.parentNumber][0],classDict[curr.myNumber][0],curr.weight))
            else:
                hierFile.write('%d|%d|%.16g\n' %
                           (curr.parentNumber, curr.myNumber, curr.weight))
        if not curr.isLeaf:
            children = marketDB.getClassificationChildren(curr)
            for i in children:
                i.parentNumber = curr.myNumber
            unprocessed.extend(children)
        curNumber += 1
    classFile.close()
    hierFile.close()
    for (sid, clsStruct) in sorted(industryMap.items()):
        asset = sid.getModelID()
        curr = clsIdMap[clsStruct.classification_id]
        weight = clsStruct.weight
        if flatFile:
            assetFile.write('%s|%d|%s|%.16g\n'% (asset.getPublicID(), curr.myNumber, classDict[curr.myNumber][0],weight))
        else:
            assetFile.write('%s|%d|%.16g\n' % (asset.getPublicID(), curr.myNumber, weight))
    assetFile.close()
    return fileNameList
    

def writeCompositeFamily(modelDB, marketDB, options, date, familyName, vendorDB):
    family = marketDB.getETFFamily(familyName)
    if family == None:
        raise LookupError('Unknown composite family %s' % familyName)
    composites = marketDB.getETFFamilyMembers(family, date, True)
    
    logging.info('%d composites in family %s', len(composites), familyName)
    createVersionFile(options, 'Composite-%s' % family.name, modelDB)
    outConstituents = open('%s/Composite-%s_Constituents' %
               (options.targetDir, family.name), 'w')
    activeCompMdlIDs = set()
    issueMapPairs = modelDB.getIssueMapPairs(date)
    marketModelMap = dict((j, i) for (i,j) in issueMapPairs)
    marketMap = dict([(i[1].getIDString(), i[0]) for i in issueMapPairs])
    rollOverInfo = dict()
    activeComposites = dict()
    for composite in composites:
        compMdlID = marketModelMap.get(composite.axioma_id)
        if compMdlID is None:
            logging.error('Skipping unmapped composite %s', composite)
            continue
        (constDate, constituents) = modelDB.getCompositeConstituents(
            composite.name, date, marketDB, marketMap=marketMap,
            rollBack=30, compositeAxiomaID=composite.axioma_id)
        if len(constituents) > 0:
            activeComposites[compMdlID] = (constDate, constituents)
            if constDate != date:
                rollOverInfo[compMdlID] = True
            for (asset, weight) in constituents:
                outConstituents.write('%s|%s|%.16g\n' %  (
                        compMdlID.getPublicID(), asset.getPublicID(), weight))
        else:
            logging.error('No constituents for composite %s (%s) - derby' % (composite.name, date))
    outConstituents.close()
    activeCompMdlIDs = list(activeComposites.keys())
    logging.info('%d active composites', len(activeCompMdlIDs))
    if len(activeCompMdlIDs) == 0:
        return
    checkCompositesForCompleteness(
        activeComposites, family, issueMapPairs, marketDB, date)
    #
    # Write data for DbComposite Asset table
    subIDs = modelDB.getAllActiveSubIssues(date)
    mdlSIDMap = dict((i.getModelID(), i) for i in subIDs)
    activeCompSIDs = [mdlSIDMap[i] for i in activeCompMdlIDs]
    rollOverInfo = dict((mdlSIDMap[mid], val) for (mid, val) in
                        rollOverInfo.items())
    
    compositeData = [(marketModelMap.get(c.axioma_id), c) for c in composites if c.axioma_id in marketModelMap]
    compositeDataDict=dict((c[1].name, mdlSIDMap[c[0]]) for c in compositeData)
    compositenames=[c.name for c in composites]
    vendorDB.dbCursor.execute("""
         select distinct etf_ticker from NETIK_ETF_CONST_ROLLED_OVER where dt=:dt
         """,dt=date) 
    rolledOverETFs = [r[0] for r in vendorDB.dbCursor.fetchall() if r[0] in compositenames]
    if len(rolledOverETFs) > 0:
        logging.info('Rolled over additional ETFs %s (%s)', ','.join(rolledOverETFs), ','.join([str(compositeDataDict[r]) for r in rolledOverETFs]))
        for r in rolledOverETFs:
            rollOverInfo[compositeDataDict[r]] = True

    # map issues to their RMG
    subIssueRMGDict = dict(modelDB.getSubIssueRiskModelGroupPairs(date))
    issueRMGDict = dict([(si.getModelID(), rmg.rmg_id)
                         for (si, rmg) in subIssueRMGDict.items()])
    out = open('%s/Composite-%s_Asset' % (options.targetDir, familyName), 'w')
    rmgIdMap = dict()
    rmgSubIssues = dict()
    riskModelGroups = list()
    for sid in activeCompSIDs:
        issueRMG = issueRMGDict[sid.getModelID()]
        if issueRMG not in rmgIdMap:
            newRMG = modelDB.getRiskModelGroup(issueRMG)
            rmgIdMap[newRMG.rmg_id] = newRMG
            rmgSubIssues[newRMG] = set()
            riskModelGroups.append(newRMG)
        rmg = rmgIdMap.get(issueRMG)
        rmgSubIssues[rmg].add(sid)
    # get sub-issue from-dates
    subIssueFromDates = dict()
    for sid in activeCompSIDs:
        modelDB.dbCursor.execute("""SELECT MIN(from_dt) FROM sub_issue
           WHERE sub_id = :sid_arg""", sid_arg=sid.getSubIDString())
        r = modelDB.dbCursor.fetchone()
        subIssueFromDates[sid] = r[0].date()
    currencyProvider = MarketDB.CurrencyProvider(marketDB, 10, None,
                                                 maxNumConverters=120)
    writeAssetData(out, None, riskModelGroups, activeCompSIDs, rmgSubIssues,
                   date, modelDB, marketDB, currencyProvider,
                   list(), subIssueFromDates,
                   marketDataOnly=True, rollOverInfo=rollOverInfo)
    out.close()
    #
    # Write DbComposite master file
    #
    allIssues = dict([(id, (from_dt, thru_dt)) for (id, from_dt, thru_dt)
                      in modelDB.getAllIssues()
                      if id in activeCompMdlIDs])
    writeMaster('DbComposite', familyName, modelDB, date, marketDB, allIssues,
                issueRMGDict, options)

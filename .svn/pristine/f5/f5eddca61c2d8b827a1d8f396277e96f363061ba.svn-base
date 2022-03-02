import io
import datetime
import logging
import numpy
import operator
import os
import os.path
import shutil
import tempfile
import time
from math import sqrt, isnan
from numpy import ma as ma

from marketdb import MarketID, CompanyID, MarketDB
from riskmodels import ModelID, ModelDB, Matrices, factor_matrix
from riskmodels import AssetProcessor
from riskmodels import AssetProcessor_V4
from riskmodels.Matrices import ExposureMatrix
from riskmodels.ModelDB import SubIssue
from riskmodels.wombat import wombat3, wombat

REDUCED_CURRENCY_DATE = datetime.date(2014, 3, 1)
CNH_START_DATE=datetime.date(2011,2,28)
NULL = ''


def getFixedFactorNames(factors, suffix, factorDict):
    fnames=[]
    for fname in factors:
        if factorDict[fname].name in ['Currency','Macro']:
            fnames.append(fname)
        else:
             fnames.append(fname+suffix)
    return fnames

def makeFinalFileName(fileName, version=3.2):
    return fileName
    if version >= 4.0:
        fstr = '.F%2d.' % (version * 10)
        l = fileName.split('.')
        return ('.'.join(l[:-2]) + fstr + '.'.join(l[-2:]))
    else:
        return fileName


def numOrNull(val, fmt, threshhold=None, fmt2=None):
    """Formats a number with the specified format or returns NULL
    if the value is masked.
    """
    if val is ma.masked or val is None or isnan(val):
        return NULL
    if threshhold and fmt2 and val < threshhold:
        return fmt2 % val
    else:
        return fmt % val


def getFactorType(expMatrix, f):
    for ftype in expMatrix.factorTypes_:
        if expMatrix.checkFactorType(f, ftype):
            return ftype
    raise KeyError('Factor of unknown type: %s' % f)


def zeroPad(val, length):
    if val == '':
        return val
    if len(val) >= length:
        return val[:length]
    zeros = length * '0'
    return zeros[:length - len(val)] + val


def getRolledOverETFs(vendorDB, date, inputETFs):
    """
        return a list of ETFs that had rolled over data for the given date
    """

    vendorDB.dbCursor.execute("""
         select distinct etf_ticker from NETIK_ETF_CONST_ROLLED_OVER where dt=:dt
         """, dt=date)
    return [r[0] for r in vendorDB.dbCursor.fetchall() if r[0] in inputETFs]


def getExposureAssets(date, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB):
    exposureAssets = []
    exposureAssetIDs = []
    spacs = AssetProcessor_V4.sort_spac_assets(date, expMatrix.getAssets(), modelDB, marketDB)
    for (aIdx, asset) in enumerate(expMatrix.getAssets()):
        if svDict != None and not options.ignoreMissingModels and asset not in svDict:
            continue
        if (ma.count(expMatrix.data_[:, aIdx]) == 0) and (asset not in spacs):
            continue
        modelID = asset.getModelID()
        exposureAssetIDs.append(modelID)
        exposureAssets.append(asset)
    exposureAssets.extend(cashAssets)
    exposureAssetIDs.extend([i.getModelID() for i in cashAssets])
    # exclude assets which shouldn't be extracted
    excludes = modelDB.getProductExcludedSubIssues(date)
    if options.preliminary:
        excludes.extend(modelDB.getDRSubIssues(rmi))
    for e in excludes:
        svDict.pop(e, None)
    exposureAssets = list(set(exposureAssets) - set(excludes))
    exposureAssetIDs = list(set(exposureAssetIDs) - set([i.getModelID() for i in excludes]))
    if options.newRiskFields:
        return (sorted(exposureAssets), sorted(exposureAssetIDs))
    else:
        return (exposureAssets, exposureAssetIDs)


def getWeekdaysBack(dt, numDays):
    """Returns a list of datetime.date objects for the weekdays before
    the date in dt, including dt if it is a weekday.  If dt is a weekday,
    then numDays + 1 dates are returned.
    numDays is the number of days to be returned.  The list is sorted
    so that the earliest date appears first.
    """
    retval = []
    curdt = datetime.date(dt.year, dt.month, dt.day)
    if curdt.weekday() < 5:
        retval.append(curdt)
    added = 0
    while added < numDays:
        curdt -= datetime.timedelta(1)
        if curdt.weekday() >= 5:
            continue
        retval.append(curdt)
        added += 1
    retval.sort()
    return retval


def getExtractExcludeClassifications(date,marketDB):
    # get list of prohibited asset subclasses RSK-4421
    query="""
        select r.name from classification_exclude e, classification_ref r where 
            exclude_logic_id = (select id from exclude_logic where exclude_logic = 'EXTRACTION_EXCLUDE')
            and e.CLASSIFICATION_ID = r.id
        """
    if date:
       query= query+ " and e.from_dt <= :dt and :dt < e.thru_dt"
       marketDB.dbCursor.execute(query,dt=date)
    else:
       marketDB.dbCursor.execute(query)

    return [e[0] for e in marketDB.dbCursor.fetchall()]


def recomputeSpecificReturns(date, univ, riskModel, modelDB, marketDB,
        useNumeraireReturns=True, useTradingReturns=False):
    """Given the current date, a list of subissues and a riskmodel instance
    recomputes the asset specific returns without all the pre-processing
    that the model regressions typically use. Thus, these are the returns
    that clients may compute when trying to replicate our numbers
    If useNumeraireReturns is True, model numeraire total returns will be loaded,
    and the currency returns included in the factor return contribution,
    thus mimicking the attribution methodology.
    Otherwise, local returns are used. If useTradingReturns is False,
    returns of "foreign" listings, e.g. ADRs will be converted to their local
    currency, otherwise they will be expressed in their trading currency.
    In both these cases, the currency return
    is excluded, thus mimicking the factor regression minus conversion
    of DRs and fill-in of missing data
    """

    import pandas

    # Load in current day's asset returns
    if useNumeraireReturns:
        # Load numeraire returns
        returns = modelDB.loadTotalReturnsHistoryV3(riskModel.rmg, date, univ, 1,
                assetConvMap=riskModel.numeraire.currency_id, dateList=[date], compound=False)
    else:
        drCurrData = None
        # If we are converting to home country returns for a regional model
        if (not useTradingReturns) and (len(riskModel.rmg) > 1):
            data = AssetProcessor.process_asset_information(
                    date, univ, riskModel.rmg, modelDB, marketDB,
                    legacyDates=riskModel.legacyMCapDates,
                    forceRun=True)
            drCurrData = data.drCurrData
        # Load local return
        returns = modelDB.loadTotalReturnsHistoryV3(riskModel.rmg, date, univ, 1,
                    assetConvMap=drCurrData, dateList=[date], compound=False)

    # Mask assets that haven't traded
    returns.data = ma.masked_where(returns.notTradedFlag, returns.data)
    if len(returns.data.shape) < 2:
        returns.data = returns.data[:, numpy.newaxis]

    # Compute excess returns
    if useNumeraireReturns:
        (excessReturns, rfr) = riskModel.computeExcessReturns(
                date, returns, modelDB, marketDB, None,
                forceCurrencyID=riskModel.numeraire.currency_id)
    else:
        (excessReturns, rfr) = riskModel.computeExcessReturns(
                date, returns, modelDB, marketDB, drCurrData)
    returnsDF = pandas.DataFrame(excessReturns.data[:,0], index=univ)

    # Get exposure matrix for previous trading day
    if riskModel.isStatModel():
        # Stat models use the current day
        prevDate = date
    else:
        prevDate = modelDB.getDates(riskModel.rmg, date, 1, excludeWeekend=True)[0]
    rmi = modelDB.getRiskModelInstance(riskModel.rms_id, prevDate)
    riskModel.setFactorsForDate(prevDate, modelDB)
    expM = riskModel.loadExposureMatrix(rmi, modelDB)
    expmatDF = pandas.DataFrame(expM.data_.T.copy(),
                                index=expM.assets_,
                                columns=expM.factors_).fillna(0.)

    # Get today's subfactors
    riskModel.setFactorsForDate(date, modelDB)
    subFactors = modelDB.getSubFactorsForDate(date, riskModel.factors)
    if (not useNumeraireReturns) and hasattr(riskModel, 'currencies'):
        subFactors = [s for s in subFactors if s.factor not in riskModel.currencies]
    factorNames = [f.factor.name for f in subFactors]

    # Pull out exposures in correct order
    expmatDF = expmatDF.reindex(index=univ, columns=factorNames).fillna(0.)

    # Load factor returns
    fr = modelDB.loadFactorReturnsHistory(riskModel.rms_id, subFactors, [date])
    facRetDF = pandas.DataFrame(fr.data, index=factorNames).fillna(0.)

    # Now combine the parts to get the specific returns
    specRets = returnsDF - numpy.dot(expmatDF, facRetDF)
    return specRets.loc[univ].values


def loadAllClassificationConstituents(classification, marketDB):
    if classification.isLeaf:
        marketDB.dbCursor.execute("""SELECT axioma_id
              FROM classification_const_active
              WHERE classification_id=:id_arg AND change_del_flag='N'""",
                                  id_arg=classification.id)
        constituents = set()
        r = marketDB.dbCursor.fetchmany()
        while len(r) > 0:
            constituents |= set([MarketID.MarketID(string=axid)
                                 for (axid,) in r])
            r = marketDB.dbCursor.fetchmany()
    else:
        children = marketDB.getClassificationChildren(classification)
        constituents = set()
        for c in children:
            childConstituents = loadAllClassificationConstituents(c, marketDB)
            constituents |= childConstituents
    return constituents


def buildESTUWeights(modelDB, estU, date, riskModel):
     """Returns a dictionary mapping the estimation universe
     asset to weight based on the square root of their market cap.
     Uses 20-day average market cap for single-country models,
     and 28-day average market cap for regional models (to account
     for weekend trading).
     """
     if len(riskModel.rmg) == 1:
         dates = modelDB.getDates(riskModel.rmg, date, 19)
     else:
         dates = modelDB.getDates(riskModel.rmg, date, 27)
     mcaps = modelDB.getAverageMarketCaps(
         dates, estU, riskModel.numeraire.currency_id).filled(0.0)
     sqrtMcaps = numpy.sqrt(mcaps)
     sqrtMcaps = sqrtMcaps / numpy.sum(sqrtMcaps)
     estuWeights = dict(zip(estU, sqrtMcaps.flat))
     return estuWeights


def checkCompositesForCompleteness(activeComposites, family, issueMapPairs, marketDB, date):
    """Check for completeness, all composites that appear as constituents
    must be present as active composites"""
    marketDB.dbCursor.execute("""SELECT axioma_id FROM composite_member
       WHERE from_dt <= :dt_arg AND :dt_arg < thru_dt""",
                              dt_arg=date)
    allMktComposites = set(i[0] for i in marketDB.dbCursor.fetchall())
    modelMap = dict((i[0], i[1].getIDString()) for i in issueMapPairs)
    activeCompositesMkt = set(modelMap[mdlId] for mdlId in activeComposites.keys())
    allConstituents = set()
    for (constDate, constituents) in activeComposites.values():
        allConstituents.update([c[0] for c in constituents])
    allConstituentsMkt = set(modelMap[mdlId] for mdlId in allConstituents)
    missing = (allConstituentsMkt & allMktComposites) - activeCompositesMkt
    if len(missing) > 0:
        msg = 'Composites are present in family %s on %s as constituents but not composites: %s' % (family.name, date, ','.join(missing))
        logging.fatal(msg)
        raise ValueError(msg)


class TempFile:
    """Helper class to create a file safely by first creating it as a temporary
    file and then moving it to its final name.
    """
    def __init__(self, name, shortDate):
        """Name is the full path of the final location of the file.
        The constructor creates a temporary file in the same directory with
        the suffix of the file as its prefix and the provided shortDate string
        as the suffix.
        """
        fileName = os.path.basename(name)
        directory = os.path.dirname(name)
        tmpfile = tempfile.mkstemp(suffix=shortDate, prefix=fileName[-3:],
                                   dir=directory)
        self.tmpName = tmpfile[1]
        self.tmpFile = os.fdopen(tmpfile[0], 'w')
        self.finalName = name
    
    def getFile(self):
        """Returns the file object of the temporary file created by the constructor.
        """
        return self.tmpFile
    
    def getTmpName(self):
        """Returns the name of the temporary file.
        """
        return self.tmpName
    
    def __enter__(self):
        self.tmpFile.__enter__()
        return self

    def __exit__(self, *args):
        """Alias for closeAndMove() to allow use in 'with' statements.
        """
        self.tmpFile.__exit__(*args)
        self.closeAndMove()

    def closeAndMove(self):
        """Close the temporary file and move it to its final location (name argument
        of the constructor).
        Also sets the file permissions to 0644.
        """
        self.tmpFile.close()
        logging.info("Move file %s to %s", self.tmpName, self.finalName)
        shutil.move(self.tmpName, self.finalName)
        os.chmod(self.finalName,0o644)


class FlatFilesV3(object):
    """Class to create flat files in version 3 format.
    """
    vanilla = False
    
    def writeDateHeader(self, options, outFile, riskModel=None, no40header=False):
        outFile.write('#DataDate: %s\n' % self.dataDate_)
        # write createDate in UTC
        gmtime = time.gmtime(time.mktime(self.createDate_.timetuple()))
        utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
        outFile.write('#CreationTimestamp: %sZ\n' %
                      utctime.strftime('%Y-%m-%d %H:%M:%S'))
        
        if options.fileFormatVersion >= 4.0 and riskModel:
            outFile.write('#ModelName: %s\n' % riskModel.name)
            outFile.write('#ModelNumeraire: %s\n' % riskModel.numeraire.currency_code)
            outFile.write('#FlatFileVersion: 4.0\n')
        elif options.fileFormatVersion >= 4.0:
            if not no40header:
                outFile.write('#FlatFileVersion: 4.0\n')

        
    def getExposureAssets(self, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB):
        return getExposureAssets(self.dataDate_, expMatrix, svDict, cashAssets, rmi, options, modelDB, marketDB)

    def writeExposures(self, date, expMatrix, cashAssets, estU, svDict, options, outFile, riskModel=None):
        """Write a triplet exposure file to outFile.
        """
        logging.debug("Writing exposures file")
        suffix=''
        if hasattr(options, 'factorSuffix') and options.factorSuffix is not None:
            suffix=options.factorSuffix
        if riskModel and suffix != '':
            factorDict=riskModel.factorTypeDict
            factorNames=getFixedFactorNames(expMatrix.getFactorNames(), suffix, factorDict)
        else:
            factorNames=expMatrix.getFactorNames()

        self.writeDateHeader(options, outFile, riskModel)
        #outFile.write('#Columns: AxiomaID|%s\n' % ( '|'.join(expMatrix.getFactorNames())))
        outFile.write('#Columns: AxiomaID|%s\n' % ( '|'.join(factorNames)))
        outFile.write('#Type: AxiomaID|%s\n' % (
            '|'.join([getFactorType(expMatrix, f).name for f
                      in expMatrix.getFactorNames()])))
        mat = expMatrix.getMatrix()
        assetIdxMap = dict()
        for (aIdx, asset) in enumerate(expMatrix.getAssets()):
            assetIdxMap[asset] = aIdx
        if options.newRiskFields:
            allAssets = sorted(expMatrix.getAssets())
            for currencyFactor in expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor):
                cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
                allAssets.append(cashSubIssue)
            allAssets = sorted(allAssets)
        else:
            allAssets = expMatrix.getAssets()

        for asset in allAssets:
            if svDict != None and asset not in svDict and asset not in cashAssets:
                continue
            modelID = asset.getModelID()
            outFile.write(modelID.getPublicID())
            if asset in cashAssets:
                currencyFactor = modelID.getIDString().split('_')[1]
                if asset.getSubIDString() == 'DCSH_CNH__11':
                    logging.warning('Ignoring %s', asset.getSubIDString())
                    continue
                for f in expMatrix.getFactorNames():
                    if currencyFactor == f:
                        outFile.write('|1')
                    else:
                        outFile.write('|')
                outFile.write('\n')
                continue
            aIdx = assetIdxMap[asset]
            for fval in mat[:,aIdx]:
                if fval is not ma.masked:
                    outFile.write('|%.8g' % fval)
                else:
                    outFile.write('|')
            outFile.write('\n')
        # add exposures to currency factors for corresponding currency assets
        if not self.vanilla and not options.newRiskFields:
            for currencyFactor in expMatrix.getFactorNames(ExposureMatrix.CurrencyFactor):
                cashSubIssue = SubIssue('DCSH_%s__11' % currencyFactor)
                assert(cashSubIssue in cashAssets)
                outFile.write('%s' % cashSubIssue.getModelID().getPublicID())
                for f in expMatrix.getFactorNames():
                    if currencyFactor == f:
                        outFile.write('|1')
                    else:
                        outFile.write('|')
                outFile.write('\n')
    
    def writeFactorCov(self, date, covMatrix, factors, options, outFile, riskModel=None):
        """Write a pipe delimited factor covariance file to outFile.
        """
        suffix=''
        if hasattr(options, 'factorSuffix') and options.factorSuffix is not None:
            suffix=options.factorSuffix
        self.writeDateHeader(options, outFile, riskModel)
        if riskModel and suffix != '':
            factorDict=riskModel.factorTypeDict
            factorNames=getFixedFactorNames([f.name for f in factors], suffix, factorDict)
        else:
            factorNames=[f.name for f in factors]
        outFile.write('#Columns: FactorName|%s\n' % ( '|'.join([fname for fname in factorNames ])))
        scale = 10000.0
        for i in range(len(factors)):
            fname=factorNames[i]
            outFile.write('%s' % fname)
            for j in range(len(factors)):
                outFile.write('|%.12g' % (scale*covMatrix[i,j]))
            outFile.write("\n")
    
    def writeIdentifierMapping(self, d, exposureAssets, modelDB, marketDB,
                               options, outFile, outFile_nosedol=None,
                               outFile_nocusip=None, outFile_neither=None, riskModel=None):
        # may need to hack CNH in here to add to exposureAssets
        if riskModel and d >= CNH_START_DATE:
            cnh=ModelID.ModelID(string='DCSH_CNH__')
            exposureAssets.append(cnh)

        excludeException = False
        cusipMap = modelDB.getIssueCUSIPs(d, exposureAssets, marketDB)
        sedolMap = modelDB.getIssueSEDOLs(d, exposureAssets, marketDB)
        isinMap = modelDB.getIssueISINs(d, exposureAssets, marketDB)
        nameMap = modelDB.getIssueNames(d, exposureAssets, marketDB)
        tickerMap = modelDB.getIssueTickers(d, exposureAssets, marketDB)
        currencyMap = modelDB.getTradingCurrency(d, exposureAssets,
                                                 marketDB, 'code')
        regionFamily = marketDB.getClassificationFamily('REGIONS')
        regionMembers = marketDB.getClassificationFamilyMembers(regionFamily)
        marketMember = [i for i in regionMembers if i.name=='Market'][0]
        marketRev = marketDB.getClassificationMemberRevision(
            marketMember, d)
        countryMap = modelDB.getMktAssetClassifications(
            marketRev, exposureAssets, d, marketDB, level=1)
        # company ID, exchange
        companyMap = modelDB.getIssueCompanies(d, exposureAssets, marketDB)
        classificationMap = modelDB.getMktAssetClassifications(marketRev, exposureAssets, d, marketDB, level=None)
        # uppercase exchange name
        exchangeMap = dict([(a, i.classification.description.upper()) for (a, i) in classificationMap.items()])
        exchangeIDMap = dict([(a, i.classification.id) for (a, i) in classificationMap.items()])
        # get all the MIC codes that we have in place from the DB for the given date
        marketDB.dbCursor.execute('select classification_id, OPERATING_MIC from class_mic_map_active_int where from_dt <= :dt and :dt < thru_dt', dt=d)
        micMap = dict([(r[0],r[1]) for r in marketDB.dbCursor.fetchall()])
        if options.fileFormatVersion >= 4.0:
            extractExcludeList = getExtractExcludeClassifications(d, marketDB)
            logging.info("Exclude the following classification types %s",  extractExcludeList)
            companyNameDict={}
            clsFamily = marketDB.getClassificationFamily('ASSET TYPES')
            clsMember = marketDB.getClassificationFamilyMembers(clsFamily)
            thisMember = [mem for mem in clsMember if mem.name=='Axioma Asset Type'][0]
            thisRevision = marketDB.getClassificationMemberRevision(thisMember, d)
            # for each of the three levels of axioma asset level go ahead and fill up thevalues
            typeMap=modelDB.getMktAssetClassifications(thisRevision, exposureAssets, d, marketDB, 1)
            classMap=modelDB.getMktAssetClassifications(thisRevision, exposureAssets, d, marketDB, 2)
            subclassMap=modelDB.getMktAssetClassifications(thisRevision, exposureAssets, d, marketDB, 3)
            
            # fill up the company name now
            companyIds = [CompanyID.CompanyID(string=i) for i in companyMap.values()]
            companyNameCache=MarketDB.TimeVariantCache()
            results=marketDB.loadTimeVariantTable('company_dim_name', [d], companyIds, ['id'],companyNameCache, 'company_id')
            for idx, cid in enumerate(companyIds):
                if results[0,idx]:
                    companyNameDict[cid.getIDString()] = results[0, idx].id
                else:
                    logging.warning( 'no information for %s' % cid.getIDString())
                    companyNameDict[cid.getIDString()] = ''

        outFiles = [outFile, outFile_nosedol, outFile_nocusip, outFile_neither]
        for f in outFiles:
            if f:
                self.writeDateHeader(options, f, riskModel)
                f.write('#Columns: AxiomaID|ISIN|CUSIP|SEDOL(7)|Ticker|Description'
                        '|Country|Currency')
                if options.newRiskFields:
                    f.write('|Exchange|CompanyID')
                if options.fileFormatVersion >= 4.0:
                    f.write('|Company Name|MIC|Asset Type|Asset Class|Asset Subclass')
                f.write('\n')
        for a in exposureAssets:
            if a.isCashAsset():
                currency = a.getCashCurrency()
                country = ''
            else:
                if a in currencyMap:
                    currency = currencyMap[a]
                else:
                    if options.notCrash:
                        logging.warning('Missing trading currency on %s for %s',
                                d, a)
                        currency='##'
                    else:
                        logging.fatal('Missing trading currency on %s for %s',
                                    d, a)
                        raise Exception('Missing trading currencies on %s' % d)
                if a in countryMap:
                    country = countryMap[a].classification.code
                else:
                    country='##'
                    if options.notCrash:
                        logging.warning('Missing trading country on %s for %s',
                                d, a)
                    else:
                        logging.fatal('Missing trading country on %s for %s',
                                    d, a)
                        raise Exception('Missing trading country on %s' % d)
            # warn if no data--should be error or fatal?
            if a not in exchangeMap and not a.isCashAsset():
                logging.warning('No exchange for %s', a.getIDString())
            if a not in companyMap and not a.isCashAsset():
                logging.warning('No company for %s', a.getIDString())
            for (idx, f) in enumerate(outFiles):
                if f:
                    outString = '%s|%s|%s|%s|%s|%s|%s|%s' % \
                            (a.getPublicID(),
                             (idx == 0) and isinMap.get(a, '') or '',
                             (idx == 0 or idx == 1) and cusipMap.get(a, '') or '',
                             (idx == 0 or idx == 2) and zeroPad(sedolMap.get(a, ''),7) or '',
                             tickerMap.get(a,''),
                             nameMap.get(a, ''),
                             country, currency)
                    f.write(outString)
                    if options.newRiskFields:
                        f.write('|%s|%s' % (exchangeMap.get(a, ''), companyMap.get(a, '')))
                    if options.fileFormatVersion >= 4.0:
                        if a.isCashAsset():
                            typeVal = 'Cash-T'
                            classVal = 'Cash-C'
                            subclassVal = 'Cash-S'
                            micVal=''
                        else:
                            if a in typeMap:
                                typeVal=typeMap.get(a).classification.name
                            else:
                                typeVal=''
                            if a in classMap:
                                classVal=classMap.get(a).classification.name
                            else:
                                classVal=''
                            if a in subclassMap:
                                subclassVal=subclassMap.get(a).classification.name
                            else:
                                subclassVal=''
                            if a in exchangeIDMap:
                                micVal=micMap.get(exchangeIDMap.get(a),'')
                            else:
                                micVal='----'
                        ### not the best place to complain, but last place check
                        ### if there is an asset that has subcassVal that is one of the proscribed values, bail
                        ### RSK-4541
                        if subclassVal in extractExcludeList:
                            logging.fatal('Prohibited classification encountered during extraction, %s %s', a, subclassVal)
                            excludeException=True
                          
                        f.write('|%s|%s|%s|%s|%s'  % (companyNameDict.get(companyMap.get(a,''),''), micVal, typeVal, classVal, subclassVal))
                    f.write('\n')
        if excludeException:
            raise Exception('Extraction exclusion exceptions on %s' % d)
  
    def writeCurrencyRates(self, marketDB, modelDB, options, outFile):
        self.writeDateHeader(options, outFile, no40header=True)
        outFile.write('#Columns: CurrencyCode|Description|Exchange Rate'
                      '|Risk-Free Rate|Cumulative RFR\n')
        currencyProvider = MarketDB.CurrencyProvider(marketDB, 10, None)
        currencyConverter = currencyProvider.getCurrencyConverter(
            self.dataDate_)
        # get active currency IDs
        marketDB.dbCursor.execute("""SELECT id FROM currency_ref
           WHERE from_dt <= :date_arg AND thru_dt > :date_arg""",
                                  date_arg=self.dataDate_)
        currencyIDs = set([i[0] for i in marketDB.dbCursor.fetchall()])
        numeraire = currencyProvider.getCurrencyID('USD', self.dataDate_)
        assert(numeraire is not None)
        # get currencies used in production models.
        # we only care about risk-free rates for those
        query="""SELECT DISTINCT currency_code
        FROM rmg_currency rc JOIN rmg_model_map rm
          ON rc.rmg_id=rm.rmg_id AND rm.rms_id > 0
        WHERE rc.from_dt <= :dt AND rc.thru_dt > :dt
          AND rm.from_dt <= :dt AND rm.thru_dt > :dt
        """
        if self.dataDate_ >= CNH_START_DATE:
            query = query + """ UNION select 'CNH' from dual"""
        modelDB.dbCursor.execute(query, dt=self.dataDate_)
        modelCurrencies = set([i[0] for i in modelDB.dbCursor.fetchall()])
        logging.debug('%d currencies, %d in models', len(currencyIDs),
                      len(modelCurrencies))
        riskFreeRates = modelDB.getRiskFreeRateHistory(
            list(modelCurrencies), [self.dataDate_], marketDB, annualize=True)
        modelCurrencyIdx = dict([(j,i) for (i,j)
                                 in enumerate(riskFreeRates.assets)])
        riskFreeRates = riskFreeRates.data[:,0]
        for currencyID in currencyIDs:
            code = marketDB.getCurrencyISOCode(currencyID, self.dataDate_)
            if self.dataDate_ >= REDUCED_CURRENCY_DATE and code not in modelCurrencies and code != 'XDR':
                 continue

            desc = marketDB.getCurrencyDesc(currencyID, self.dataDate_)
            if code in modelCurrencies:
                rfr = riskFreeRates[modelCurrencyIdx[code]]
                if rfr is ma.masked:
                    rfr = None
                cumR = modelDB.getCumulativeRiskFreeRate(code, self.dataDate_)
            else:
                rfr = None
                cumR = None
            rate = None
            if currencyConverter.hasRate(currencyID, numeraire):
                rate = currencyConverter.getRate(currencyID, numeraire)
                outFile.write('%s|%s|%s|%s|%s\n' % (
                    code, desc, rate, numOrNull(rfr, '%.8f'),
                    numOrNull(cumR, '%.8f')))
            elif code in modelCurrencies:
                logging.error('Missing exchange rate for %s on %s',
                              code, self.dataDate_)
    
    def writeFactorReturns(self, d, riskModel, modelDB, options, outFile, internal=False):
        suffix=''
        if hasattr(options, 'factorSuffix') and options.factorSuffix is not None:
            suffix=options.factorSuffix
        self.writeDateHeader(options, outFile, riskModel)
        outFile.write('#Columns: FactorName|Return|Cumulative Return\n')
        if internal:
            factorReturns = riskModel.loadFactorReturns(d, modelDB, flag='internal')
            cumReturns = (Matrices.allMasked((len(riskModel.factors),)), riskModel.factors)
        else:
            factorReturns = riskModel.loadFactorReturns(d, modelDB)
            cumReturns = riskModel.loadCumulativeFactorReturns(d, modelDB)
        factorReturnMap = dict(zip(factorReturns[1], factorReturns[0]))
        cumReturnMap = dict(zip(cumReturns[1], cumReturns[0]))

        allFactors = sorted(list(factorReturnMap.keys()), key=operator.attrgetter('name'))

        if riskModel and suffix != '':
            factorDict=riskModel.factorTypeDict
            factorNames=getFixedFactorNames([f.name for f in allFactors], suffix, factorDict)
        else:
            factorNames=[f.name for f in allFactors]


        for (idx, i) in enumerate(allFactors):
            if factorReturnMap[i] is not ma.masked:
                factorReturnMap[i] *= 100.0
            if options.fileFormatVersion >= 4.0:
                outFile.write('%s|%s|%s\n'
                          % (factorNames[idx],
                             numOrNull(factorReturnMap[i], '%4.10f'),
                             numOrNull(cumReturnMap[i], '%.16g')))
            else:
                outFile.write('%s|%s|%s\n'
                          % (i.name,
                             numOrNull(factorReturnMap[i], '%4.10f'),
                             numOrNull(cumReturnMap[i], '%4.10f', 0.001, "%.14e")))
    
    def writeIndustryHierarchy(self, d, riskModel, factors, modelDB, marketDB, options, outFile):
        self.writeDateHeader(options, outFile, riskModel)
        outFile.write('#Columns: Name|Parent|Level\n')
        # Read all classifications--find root parent and non-root parent of each
        allClassifications = riskModel.getAllClassifications(modelDB)
        if not allClassifications:
            return
        for c in allClassifications:
            if options.factorSuffix and \
               c.name in riskModel.factorTypeDict and \
               riskModel.factorTypeDict[c.name].name == 'Industry':
                outFile.write('%s|' % (c.name + options.factorSuffix))
            else:
                outFile.write('%s|' % c.name)
            parents = riskModel.getClassificationParents(c, modelDB)
            nonrootParents = [p for p in parents if not p.isRoot]
            if len(nonrootParents) > 0:
                outFile.write('%s' % nonrootParents[0].name)
            outFile.write('|')
            rootParents = [p for p in parents if p.isRoot]
            if len(rootParents) > 0:
                outFile.write('%s' % rootParents[0].name)
            outFile.write('\n')
    
    def writeAssetLine(self, outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                       rollOverInfo, ucp, rawUCP, latestUCP,
                       latestMarketCap, adv20, ret1,
                       histbetas, latestCumReturns, mdv20=None, flatFileVersion=3.2, mdv60=None):
        line = io.StringIO()
        modelID = sid.getModelID()
        currency = tradingCurrencyMap.get(sid)
        if currency is None:
            logging.warning("No currency for %s", sid.getModelID())
            currency = ''
        
        line.write('%s|%s|' % (modelID.getPublicID(), currency))
        if rollOverInfo is not None and sid in rollOverInfo:
            flagRollOver =  rollOverInfo[sid]
        else:
            missingPrice = ucp.data[idx, -1] is ma.masked
            stalePrice = rawUCP[idx,0] is not None and rawUCP[idx,0].price_marker == 3
            flagRollOver = missingPrice or stalePrice
        if flagRollOver:
            line.write('*')
        if flatFileVersion >=4.0:
            if sid.isCashAsset and latestUCP[idx] is ma.masked:
                logging.warning('Defaulting cash asset %s to 1.0 price', sid)
                line.write('|1.00000')
            else:
                line.write('|%s' % numOrNull(latestUCP[idx], '%.5f', 0.00001, '%.5e'))
        else:
            if sid.isCashAsset and latestUCP[idx] is ma.masked:
                logging.warning('Defaulting cash asset %s to 1.0 price', sid)
                line.write('|1.0000')
            else:
                line.write('|%s' % numOrNull(latestUCP[idx], '%.4f', 0.0001, '%.4e'))
        line.write('|%s' % numOrNull(latestMarketCap[idx], '%.0f'))
        if sid.isCashAsset():
            line.write('|%s' % NULL)
        else:
            line.write('|%s' % numOrNull(adv20[rmgIdx], '%.0f'))
        if flatFileVersion >=4.0:
            line.write('|%s' % numOrNull(ret1[rmgIdx], '%.12f'))
            if histbetas.get(sid):
                line.write('|%s' % numOrNull(histbetas.get(sid), '%.4f'))
            else:
                line.write('|')
            if sid.isCashAsset():
                line.write('|%s|%s' % (NULL,NULL))
            else:
                line.write('|%s' % numOrNull(mdv20[rmgIdx], '%.0f'))
                line.write('|%s' % numOrNull(mdv60[rmgIdx], '%.0f'))
        else:
            line.write('|%s' % numOrNull(ret1[rmgIdx], '%.4f'))
            #else flatFileVersion <=3.2:
            line.write('|%s' % numOrNull(histbetas.get(sid), '%.4f'))
            line.write('|%s' % numOrNull(latestCumReturns[idx], '%.6g'))
        line.write('\n')
        outFile.write(line.getvalue())
    
    def writeAssetAttributes(self, date, riskModelGroups, subIssues,
                             modelDB, marketDB, options, outFile, rollOverInfo=None,
                             useNonCheatHistoricBetas=True, riskModel=None):
        """Write asset attributes per risk model group.
        rollOverInfo: if present it is a map of SubIssue to Boolean. If a
           sub-issue is present in the map then its value should be
           used to determine roll-over status instead of the price history.
        useNonCheatHistoricBetas: if true, use the correct historic betas,
           otherwise use the old-style betas with the incorrect market return definition.
        """
        # may need to hack CNH in here to add to exposureAssets
        if riskModel and date >= CNH_START_DATE:
            cnh=ModelDB.SubIssue(string='DCSH_CNH__11')
            subIssues=subIssues.union(set([cnh]))

        logging.debug("Writing asset attributes for %d assets", len(subIssues))
        self.writeDateHeader(options, outFile, riskModel)
        if options.fileFormatVersion >= 4.0:
            outFile.write('#Columns: AxiomaID|Currency|RolloverFlag|Price'
                      '|Market Cap|20-Day ADV|1-Day Return|Historical Beta|20-Day MDV|60-Day MDV\n')
            outFile.write('#Type: ID|NA|Set|Attribute|Attribute'
                      '|Attribute|Attribute|Attribute|Attribute|Attribute\n')
            outFile.write('#Unit: ID|Text|NA|CurrencyPerShare|Currency'
                      '|Currency|Percent|Currency|Currency|Currency\n')
        else:
            outFile.write('#Columns: AxiomaID|Currency|RolloverFlag|Price'
                      '|Market Cap|20-Day ADV|1-Day Return|Historical Beta'
                      '|Cumulative Return\n')
            outFile.write('#Type: ID|NA|Set|Attribute|Attribute'
                      '|Attribute|Attribute|Attribute'
                      '|NA\n')
            outFile.write('#Unit: ID|Text|NA|CurrencyPerShare|Currency'
                      '|Currency|Percent|Number'
                      '|Number\n')
        rmgIdMap = dict([(rmg.rmg_id, rmg) for rmg in riskModelGroups])
        rmgSubIssues = dict([(rmg, set()) for rmg in riskModelGroups])
        subIssuesRMG = dict()
        modelDB.dbCursor.execute("""SELECT issue_id, rmg_id, from_dt
           FROM sub_issue WHERE from_dt <= :date_arg
           AND :date_arg < thru_dt""", date_arg=date)
        issueRMGDict = dict([(ModelID.ModelID(string=mid),
                              (rmg_id, fromDt.date()))
                             for (mid, rmg_id, fromDt)
                             in modelDB.dbCursor.fetchall()])
        currencyProvider = MarketDB.CurrencyProvider(marketDB, 10, None)
        riskModelGroups = list(riskModelGroups)
        issueFromDates = dict()
        for sid in subIssues:
            # if rmg not in rmgIdMap, then it is from a non-model
            # rmg. In that case add that rmg to riskModelGroups
            # and rmgIdMap
            (issueRMG, fromDt) = issueRMGDict[sid.getModelID()]
            if issueRMG not in rmgIdMap:
                newRMG = modelDB.getRiskModelGroup(issueRMG)
                rmgIdMap[newRMG.rmg_id] = newRMG
                rmgSubIssues[newRMG] = set()
                riskModelGroups.append(newRMG)
            rmg = rmgIdMap.get(issueRMG)
            rmgSubIssues[rmg].add(sid)
            subIssuesRMG[sid] = rmg
            issueFromDates[sid] = fromDt
        self.writeAssetData(outFile, riskModelGroups, subIssues, rmgSubIssues, subIssuesRMG,
                            currencyProvider, date, options, modelDB, marketDB,
                            issueFromDates, rollOverInfo=rollOverInfo,
                            useNonCheatHistoricBetas=useNonCheatHistoricBetas, riskModel=riskModel)

    def writeAssetData(self, outFile, rmgs, allSubIssues, rmgSubIssues, subIssuesRMG,
                       currencyProvider, date, options, modelDB, marketDB,
                       issueFromDates, rollOverInfo, useNonCheatHistoricBetas, flatFileVersion=3.2, riskModel=None):
        # rmgSubIssues is dict of rmg->set of subIssues
        if options.newRiskFields:
            allSubIssues = sorted(allSubIssues)
        tradingCurrencyMap = modelDB.getTradingCurrency(
            date, allSubIssues, marketDB, 'code')
        sidCurrencies = modelDB.getTradingCurrency(
            date, allSubIssues, marketDB, 'id')
        missingCurrency = [sid for sid in allSubIssues if sid not in sidCurrencies]
        if len(missingCurrency) > 0:
            if options.notCrash:
                allSubIssues = [sid for sid in allSubIssues if sid not in missingCurrency]
                logging.warning('Missing trading currency on %s for %s',
                            date, ','.join([sid.getSubIDString() for sid
                                            in missingCurrency]))
            else:
                logging.fatal('Missing trading currency on %s for %s',
                        date, ','.join([sid.getSubIDString() for sid
                            in missingCurrency]))
                raise Exception('Missing trading currencies on %s' % date)
        histLen = 20
        histLen = 60
        allDays = set()
        rmgTradingDays = dict()
        sidRMGIndices = dict()
        adv20Dict = dict()
        mdv20Dict = dict()
        mdv60Dict = dict()
        ret1Dict = dict()
        for rmg in rmgs:
            dateList = set(modelDB.getDates([rmg], date, histLen))
            rmgTradingDays[rmg] = dateList
            allDays |= dateList
        allDays = sorted(allDays)
        sidIdxMap = dict([(sid, idx) for (idx, sid) in enumerate(allSubIssues)])
        dateIdxMap = dict([(d, idx) for (idx, d) in enumerate(allDays)])
        dailyVolumes = modelDB.loadVolumeHistory(
            allDays, allSubIssues, sidCurrencies)
        dailyVolumes.data = Matrices.fillAndMaskByDate(
            dailyVolumes.data, [issueFromDates[sid] for sid in allSubIssues],
            allDays)
        # RLG, 11/13/08
        # just get the latest (1-day) return
        totalReturns = modelDB.loadTotalLocalReturnsHistory(
            [date], allSubIssues)
        totalReturns.data = totalReturns.data.filled(0.0)
        ucp = modelDB.loadUCPHistory(allDays, allSubIssues, sidCurrencies)
        rawUCP = modelDB.loadRawUCPHistory([date], allSubIssues)
        latestUCP = Matrices.allMasked((len(allSubIssues),))
        cumReturns = modelDB.loadCumulativeReturnsHistory(
            [date, date - datetime.timedelta(1), date - datetime.timedelta(2),
             date - datetime.timedelta(3)], allSubIssues)
        latestCumReturns = cumReturns.data[:,0]
        for i in range(3):
            latestCumReturns = ma.where(ma.getmaskarray(latestCumReturns),
                                        cumReturns.data[:,i+1],
                                        latestCumReturns)
        marketCaps = modelDB.loadMarketCapsHistory(allDays, allSubIssues, sidCurrencies)
        latestMarketCap = Matrices.allMasked((len(allSubIssues),))
        if useNonCheatHistoricBetas:
            histbetas = modelDB.getPreviousHistoricBeta(date, allSubIssues)
        else:
            histbetas = modelDB.getPreviousHistoricBetaOld(date, allSubIssues)

        if riskModel:
           if len(riskModel.rmg) > 1:
              SCM = False
           else:
              SCM = True
        else:
           SCM=True
        scmCoverage = SCM
        if riskModel and hasattr(riskModel, 'coverageMultiCountry'):
            scmCoverage = (riskModel.coverageMultiCountry==False)

        if options.histBetaNew:
           logging.info('Using historic beta new style...')
           subIssues = [ s for s in allSubIssues if not s.isCashAsset()]
           data = AssetProcessor.process_asset_information(
              date, subIssues, riskModel.rmg, modelDB, marketDB,
              checkHomeCountry=(scmCoverage==False),
              legacyDates=riskModel.legacyMCapDates,
              numeraire_id=riskModel.numeraire.currency_id,
              forceRun=riskModel.forceRun)

           historicBeta_home = modelDB.getHistoricBetaDataV3(
                    date, data.universe, field='value', home=1, rollOverData=True)
           historicBeta_trad = modelDB.getHistoricBetaDataV3(
                    date, data.universe, field='value', home=0, rollOverData=True)

           
        for rmg in rmgs:
            subIssues = sorted(rmgSubIssues[rmg])
            if options.notCrash:
                subIssueDict = dict((s, -1) for s in allSubIssues)
                droppedSubIssues = [sid for sid in subIssues if sid not in subIssueDict]
                if len(droppedSubIssues) > 0:
                    droppedDict = dict((s, -1) for s in droppedSubIssues)
                    subIssues = [sid for sid in subIssues if sid not in droppedDict]
                    for sid in droppedSubIssues:
                        logging.warning('Dropped subissue: %s for RMG: %s on Date: %s',
                                sid.getSubIDString(), rmg.mnemonic, date)
            rmgSidIndices = [sidIdxMap[sid] for sid in subIssues]
            sidRMGIndices.update(dict([(sid, idx) for (idx, sid) in enumerate(subIssues)]))
            if len(rmgSidIndices) == 1:
                # Add another asset to prevent numpy from converting
                # arrays with only one element to a number
                rmgSidIndices.append(0)
            tradingDays = sorted(rmgTradingDays[rmg], reverse=True)
            logging.debug('Writing data for %d assets in %s', len(subIssues), rmg)
            for sid in subIssues:
                sidIdx = sidIdxMap[sid]
                for tDate in tradingDays:
                    tIdx = dateIdxMap[tDate]
                    if not ucp.data[sidIdx,tIdx] is ma.masked:
                        latestUCP[sidIdx] = ucp.data[sidIdx,tIdx]
                        break
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
            adv20Dict[rmg] = ma.average(rmgDailyVolumes[:,-20:], axis=1)
            if len(rmgDailyVolumes) > 0:
                mdv20Dict[rmg] = ma.median(rmgDailyVolumes[:,-20:], axis=1)
                mdv60Dict[rmg] = ma.median(rmgDailyVolumes[:,-60:], axis=1)
            else:
                mdv20Dict[rmg] = rmgDailyVolumes
                mdv60Dict[rmg] = rmgDailyVolumes
            ret1Dict[rmg] = 100.0 * rmgReturns[:,-1]
            if not options.newRiskFields:
                for (rmgIdx, sid) in enumerate(subIssues):
                    idx = sidIdxMap[sid]
                    self.writeAssetLine(outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                                        rollOverInfo, ucp, rawUCP, latestUCP,
                                        latestMarketCap, adv20Dict[rmg], ret1Dict[rmg],
                                        histbetas, latestCumReturns)
        if options.newRiskFields:
            for (idx, sid) in enumerate(allSubIssues):
                rmg = subIssuesRMG[sid]
                rmgIdx = sidRMGIndices[sid]
                ### RSK-3973 play games regarding the historic beta

                if options.histBetaNew:
                     if SCM:
                         histbetas[sid] = historicBeta_trad.get(sid, historicBeta_home.get(sid,''))
                     else:
                         histbetas[sid]= historicBeta_home.get(sid, '')

                self.writeAssetLine(outFile, sid, idx, rmgIdx, tradingCurrencyMap,
                                    rollOverInfo, ucp, rawUCP, latestUCP,
                                    latestMarketCap, adv20Dict[rmg], ret1Dict[rmg],
                                    histbetas, latestCumReturns, mdv20Dict[rmg], options.fileFormatVersion, mdv60Dict[rmg])

    def buildESTUWeights(self, modelDB, marketDB, estU, date, riskModel):
        """Returns a dictionary mapping the estimation universe
        asset to weight based on the square root of their market cap.
        Uses 20-day average market cap for single-country models,
        and 28-day average market cap for regional models (to account
        for weekend trading).
        """
        if len(riskModel.rmg) == 1:
            dates = modelDB.getDates(riskModel.rmg, date, 19)
        else:
            dates = modelDB.getDates(riskModel.rmg, date, 27)
        mcaps = modelDB.getAverageMarketCaps(
                dates, estU, riskModel.numeraire.currency_id, marketDB).filled(0.0)
        sqrtMcaps = numpy.sqrt(mcaps)
        sqrtMcaps = sqrtMcaps / numpy.sum(sqrtMcaps)
        estuWeights = dict(zip(estU, sqrtMcaps.flat))
        return estuWeights

    def writeAssetRiskAttributes(self, date, riskModel, exposureAssets, expM,
                                 svDict, estU, options, modelDB, marketDB, outFile):
        rmi = riskModel.getRiskModelInstance(date, modelDB)
        if not self.vanilla:
            predbetas = modelDB.getRMIPredictedBeta(rmi)
            #assert(len(predbetas)>0)
            if len(riskModel.industries) > 0:
                classifications = riskModel.industryClassification.getAssetConstituents(modelDB, exposureAssets, date)
                hasIndustries = True
            else:
                classifications = set()
                hasIndustries = False
            estUWeightdict = self.buildESTUWeights(modelDB, marketDB, estU, date, riskModel)
            specRtnsMatrix = modelDB.loadSpecificReturnsHistory(rmi.rms_id, exposureAssets, [date])
            newSpecReturns = recomputeSpecificReturns(date, exposureAssets, riskModel, modelDB, marketDB, False)
            specRtnsMatrix.data *= 100
            newSpecReturns  *= 100.0 # conver to percentages
            totalRisks = modelDB.getRMITotalRisk(rmi)
        logging.debug("Writing risk attributes for %d assets ", len(exposureAssets))
        self.writeDateHeader(options, outFile, riskModel)
        if self.vanilla:
            outFile.write('#Columns: AxiomaID|Specific Risk\n')
            outFile.write('#Type: ID|Attribute\n')
            outFile.write('#Unit: ID|Percent\n')
        else:
            if hasIndustries:
                if options.fileFormatVersion < 4.0:
                    outFile.write('#Columns: AxiomaID|Specific Risk|Predicted Beta|Industry Source'
                              '|Estimation Universe')
                    if options.newRiskFields:
                        outFile.write('|Estimation Universe Weight|Specific Return|Total Risk')
                    outFile.write('\n')
                    outFile.write('#Type: ID|Attribute|Attribute|Attribute|Set')
                    if options.newRiskFields:
                        outFile.write('|Attribute|Attribute|Attribute')
                    outFile.write('\n')
                    outFile.write('#Unit: ID|Percent|Number|Text|NA')
                    if options.newRiskFields:
                        outFile.write('|Percent|Percent|Percent')
                    outFile.write('\n')
                else:
                    outFile.write('#Columns: AxiomaID|Specific Risk|Industry Source'
                              '|Estimation Universe')
                    outFile.write('|Estimation Universe Weight|Specific Return|Total Risk')
                    outFile.write('\n')
                    outFile.write('#Type: ID|Attribute|Attribute|Set')
                    outFile.write('|Attribute|Attribute|Attribute')
                    outFile.write('\n')
                    outFile.write('#Unit: ID|Percent|Text|NA')
                    outFile.write('|Percent|Percent|Percent')
                    outFile.write('\n')
            else:
                if options.fileFormatVersion < 4.0:
                    outFile.write('#Columns: AxiomaID|Specific Risk|Predicted Beta'
                              '|Estimation Universe')
                    if options.newRiskFields:
                        outFile.write('|Estimation Universe Weight|Specific Return|Total Risk')
                    outFile.write('\n')
                    outFile.write('#Type: ID|Attribute|Attribute|Set')
                    if options.newRiskFields:
                        outFile.write('|Attribute|Attribute|Attribute')
                    outFile.write('\n')
                    outFile.write('#Unit: ID|Percent|Number|NA')
                    if options.newRiskFields:
                        outFile.write('|Percent|Percent|Percent')
                    outFile.write('\n')
                else:
                    outFile.write('#Columns: AxiomaID|Specific Risk|Estimation Universe')
                    outFile.write('|Estimation Universe Weight|Specific Return|Total Risk')
                    outFile.write('\n')
                    outFile.write('#Type: ID|Attribute|Set')
                    outFile.write('|Attribute|Attribute|Attribute')
                    outFile.write('\n')
                    outFile.write('#Unit: ID|Percent|NA')
                    outFile.write('|Percent|Percent|Percent')
                    outFile.write('\n')
        # temporary set to speed lookup
        estUset = set(estU)
        count=0
        for (idx, asset) in enumerate(exposureAssets):
            if asset not in svDict:
                assert(asset.isCashAsset())
                continue
            modelID = asset.getModelID()
            outFile.write('%s' % (modelID.getPublicID()))
            outFile.write('|%s' % numOrNull(100.0 * svDict.get(asset), '%.12g'))
            if self.vanilla:
                outFile.write('\n')
                continue
            # only write out predicted betas if file format is < 3.3
            if options.fileFormatVersion < 3.3:
                outFile.write('|%s' % numOrNull(predbetas.get(asset), '%.4f'))
            if hasIndustries:
                # Get source for asset's industry
                source = ''
                if asset in classifications:
                    src_id = classifications[asset].src
                    source = 'Axioma'
                    if (src_id>=300 and src_id<=399):
                        source = 'GICS-Direct'
                outFile.write('|%s' % source)
            if asset in estUset:
                outFile.write('|*')
            else:
                outFile.write('|')

            if options.newRiskFields:
                # estu weight
                if asset in estUWeightdict and estUWeightdict[asset] != None:
                    estUWeightdict[asset] *= 100
                outFile.write('|%s' % numOrNull(estUWeightdict.get(asset), '%.4f'))
                # specific returns, total risk
                if asset in totalRisks and totalRisks[asset] != None:
                    totalRisks[asset] *= 100

                if options.fileFormatVersion < 3.3:
                    outFile.write('|%s|%s' % (numOrNull(specRtnsMatrix.data[idx,0], '%4.10f'),
                                              numOrNull(totalRisks.get(asset), '%.4f')))
                else:
                    # for the 4.0 version use the new format
                    outFile.write('|%s|%s' % (numOrNull(newSpecReturns[idx,0], '%.12f'),
                                              numOrNull(totalRisks.get(asset), '%.12g')))
            outFile.write('\n')
    
    def writeSpecificCovariances(self, date, exposureAssets, specCov, options, outFile, riskModel=None):
        logging.debug("Writing specific covariances for %d assets ", len(exposureAssets))
        self.writeDateHeader(options, outFile, riskModel)
        if options.fileFormatVersion >= 4.0:
            outFile.write('#Columns: AxiomaID|AxiomaID|Covariance\n')
        else:
            outFile.write('#Columns: AxiomaID|AxiomaID|Covariance\n')
        exposureAssets = set(exposureAssets)
        triplets = list()
        for (sid1, sid1Covs) in specCov.items():
            for (sid2, value) in sid1Covs.items():
                if sid1 in exposureAssets and sid2 in exposureAssets:
                    if sid1 < sid2:
                        triplets.append((sid1, sid2, value))
                    else:
                        triplets.append((sid2, sid1, value))
        for (sid1, sid2, value) in sorted(triplets):
            if options.fileFormatVersion >= 4.0:
                outFile.write('%s|%s|%s\n' % (
                    sid1.getModelID().getPublicID(),
                    sid2.getModelID().getPublicID(),
                    numOrNull(value, '%.12g')))
            else:
                outFile.write('%s|%s|%s\n' % (
                    sid1.getModelID().getPublicID(),
                    sid2.getModelID().getPublicID(),
                    numOrNull(value, '%.12g')))


 
    def writeStatFactorReturnHist(self, model, d, modelDB, options, outFile):
        self.writeDateHeader(options, outFile, model)
        factor_matrix.writeStatFactorMatrix(model, d, modelDB, outFile, delimiter='|', dateformat='%Y-%m-%d', standardHeader=True)

    def writeFactorReturnHist(self, model, d, modelDB, options, outFile):
        self.writeDateHeader(options, outFile, model)
        startDate = min([rmg.from_dt for rmg in model.rmgTimeLine])
        # start at the earliest 1 Jan 2009
        if d > datetime.date(2009,1,1):
            startDate = max(startDate, datetime.date(2009,1,1))
        # use today's date as the end date so we pick up all available information
        # since we overwrite the whole file each time
        # note dates are inclusive
            endDate = datetime.date.today()
        else:
            endDate = datetime.date(2008,12,31)
        factor_matrix.writeFactorMatrix(model, startDate, endDate, modelDB, outFile, delimiter='|',
                                        dateformat='%Y-%m-%d', standardHeader=True, version=options.fileFormatVersion)
    
    def writeGSFile(self, options, marketDB, modelDB, d, rmg,
                    exposureAssets,
                    outFile, outFile1=None, outFile2=None, outFile3=None, outFile4=None):
        import os.path
        inFileName = "/axioma/operations/daily/Goldman/GSTCM%04d%02d%02d-WORLD.csv" % (d.year, d.month, d.day)
        if not os.path.isfile(inFileName):
            logging.info("Can't find Goldman file %s", inFileName)
            return
        infile = open(inFileName, 'r')
        for inline in infile:
            if inline.startswith('#'):
                continue
            fields = inline.strip().split(',')
            if len(fields)>9 and not rmg == 'US':
                oldFormat = False
            else:
                oldFormat = True
            break
        infile.seek(0)
        logging.info('Load all axioma IDs')
        allaxids = marketDB.getAllAxiomaIDs()
        
        tickerDict = dict(marketDB.getTickers(d, allaxids))
        cusipDict = dict(marketDB.getCUSIPs(d, allaxids))
        sedolDict = dict(marketDB.getSEDOLs(d, allaxids))
        isinDict = dict(marketDB.getISINs(d, allaxids))
        countryDict = marketDB.getTradingCountry(d, allaxids)
        tickerReverseDict = dict((j+'-'+(countryDict.get(i) and countryDict.get(i) or ''),i) for (i,j) in tickerDict.items())
        cusipReverseDict = dict([(j+'-'+(countryDict.get(i) and countryDict.get(i) or ''),i) for (i,j) in cusipDict.items()])
        sedolReverseDict = dict([(j,i) for (i,j) in sedolDict.items()])
        issueMapPairs = modelDB.getIssueMapPairs(d)
        marketModelMap = dict([(j,i) for (i,j) in issueMapPairs])
        gssoutfiles = [outFile, outFile2, outFile3, outFile4]
        
        for f in gssoutfiles:
            if f:
                self.writeDateHeader(options, f)
                if oldFormat:
                    f.write('#Columns: Axioma Id|CUSIP|Ticker|ISIN|Shortfall|Sum\n')
                else:
                    f.write('#Columns: Axioma Id|Country|SEDOL|CUSIP|Ticker|ISIN|Shortfall|Sum\n')
        if outFile1:
            self.writeDateHeader(options, outFile1)
            if oldFormat:
                outFile1.write('#Columns: Axioma Id|Ticker|CUSIP|MDV|Last Close')
            else:
                outFile1.write('#Columns: Axioma Id|Country|Ticker|SEDOL|CUSIP|MDV|Last Close')
            for pct in range(20):
                outFile1.write('|%d%% MDV Cost bps' % (pct+1))
            outFile1.write('\n')
        for inline in infile:
            if inline.startswith('#'):
                continue
            fields = inline.strip().split(',')
            ticker = fields[1]
            cusip = fields[2].strip('"')
            price = float(fields[3])
            goldman_a = float(fields[4])    # goldman "A"
            goldman_b = float(fields[5])    # goldman "B"
            goldman_c = float(fields[6])    # goldman "C"
            mdv = None
            if len(fields)>7:
                mdv = float(fields[7])
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
                logging.debug('No axioma ID for ticker %s in country %s on date %s',
                             ticker, country, d)
                marketAXID = axiomaID2
            if axiomaID2 is None:
                if useCUSIP:
                    logging.debug('No axioma ID for CUSIP %s on date %s',
                             cusip, d)
                else:
                    logging.warning('No axioma ID for SEDOL %s on date %s',
                             sedol, d)
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
                    modelDB.dbCursor.execute(query, id=axid1.getIDString(), dt=d)
                    r = modelDB.dbCursor.fetchall()
                    if r:
                        subid1 = SubIssue(string=r[0][0])
                    modelDB.dbCursor.execute(query, id=axid2.getIDString(), dt=d)
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
            axid = marketModelMap.get(marketAXID)
            if axid is None:
                logging.warning('No model ID for market ID %s', marketAXID)
                continue
            axid = axid.getPublicID()
            n1 = goldman_a / 10000 # convert from basis points to dollars
            n3 = goldman_c / 10000 # convert from basis points to dollars
            n2 = goldman_b / 10000 / sqrt(1000) # convert from basis points to dollars, plus additional conversion
            numberString = '%#.5e!%#.5e!%#.5e' % (n1, n2, n3)
            logging.debug('writing: %s|%s|%s|%s|%s', axid, cusip, ticker, isinDict.get(marketAXID, '' ), numberString)
            for (i, f) in enumerate(gssoutfiles):
                # 0 and 2 have CUSIP & ISIN info; 1 and 3 do not
                # 0 and 1 have SEDOL info; 2 and 3 do not
                if f:
                    if oldFormat:
                        f.write('%s|%s|%s|%s|%s|%d\n' % (axid, (i % 2 == 0 and cusip or ''), ticker, (i % 2 == 0 and isinDict.get(marketAXID, '' ) or ''),
                                                         wombat3(numberString),wombat(wombat3(numberString))))
                    else:
                        f.write('%s|%s|%s|%s|%s|%s|%s|%d\n' % (axid, country,
                                                               (i < 2 and sedol or ''),
                                                               (i % 2 == 0 and cusip or ''),
                                                               ticker,
                                                               (i % 2 == 0 and isinDict.get(marketAXID, '' ) or ''),
                                                               wombat3(numberString),wombat(wombat3(numberString))))
            if outFile1 and mdv:
                if oldFormat:
                    outFile1.write('%s|%s|%s|%s|%.4f' % (axid, ticker, cusip, mdv, price))
                else:
                    outFile1.write('%s|%s|%s|%s|%s|%s|%.4f' % (axid, country, ticker, sedol, cusip, mdv, price))
                for pct in range(20):
                    outFile1.write('|%.5f' % \
                                   max(goldman_c, \
                                       goldman_a + (goldman_b * \
                                                    sqrt(mdv * price * ((pct+1)/100.0) / 1000.0))))
                outFile1.write('\n')
        infile.close()
    
    def writeDay(self, options, rmi, riskModel, modelDB, marketDB):
        d = rmi.date
        self.dataDate_ = rmi.date
        self.createDate_ = modelDB.revDateTime
        mnemonic = riskModel.mnemonic
        if options.writeCov or options.writeFactorHry:
            (factorCov, factors) = riskModel.loadFactorCovarianceMatrix(
                rmi, modelDB)
        if options.writeExp or options.writeRiskAttributes \
                or options.writeAssetIdm or options.writeAssetAtt \
                or options.writeSpecificCovariances \
                or options.writeGSSModel:
            expM = riskModel.loadExposureMatrix(rmi, modelDB)
            (svDataDict, specCov) = riskModel.loadSpecificRisks(rmi, modelDB)
            cashAssets = set()
            if not self.vanilla:
                estU = riskModel.loadEstimationUniverse(rmi, modelDB)
                # find cash assets for active RMGs and add to master
                modelDB.dbCursor.execute("""SELECT rmg.rmg_id, rmg.mnemonic, rmg.description
                FROM risk_model_group rmg
                WHERE EXISTS (SELECT * FROM rmg_model_map rmm WHERE rmm.rmg_id = rmg.rmg_id
                                AND rmm.from_dt<=:dt AND rmm.thru_dt>:dt)""", dt=d)
                for r in modelDB.dbCursor.fetchall():
                    # add cash assets
                    rmg = modelDB.getRiskModelGroup(r[0])
                    for i in modelDB.getActiveSubIssues(rmg, d):
                        if i.isCashAsset():
                            cashAssets.add(i) 
                (exposureAssets, exposureAssetIDs) = self.getExposureAssets(
                    expM, svDataDict, cashAssets, rmi, options, modelDB, marketDB)
            else:
                estU = list()
                exposureAssets = [n for n in expM.getAssets() if n in svDataDict]
                # exclude assets which shouldn't be extracted
                excludes = modelDB.getProductExcludedSubIssues(self.dataDate_)
                if options.preliminary:
                    excludes.extend(modelDB.getDRSubIssues(rmi))
                exposureAssets = list(set(exposureAssets) - set(excludes))
                exposureAssetIDs = [si.getModelID() for si in exposureAssets]
       
        shortdate='%04d%02d%02d' % (d.year, d.month, d.day)
        target=options.targetDir
        if options.appendDateDirs:
            target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass
        # Write factor covariance
        if options.writeCov:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='cov',dir=target)
            outFileName = '%s/%s.%04d%02d%02d.cov' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            outFile = open(tmpfilename, 'w')
            self.writeFactorCov(d, factorCov, factors, options, outFile, riskModel)
            outFile.close()
            logging.info("Move covariance file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        
        # write exposures
        if options.writeExp:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='exp',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            
            outFileName = '%s/%s.%04d%02d%02d.exp' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            logging.info("Writing to %s", tmpfilename)
            #outFile = open(outFileName, 'w')
            outFile=open(tmpfilename,'w')
            self.writeExposures(d, expM, cashAssets, estU, svDataDict, options, outFile, riskModel)
            outFile.close()
            logging.info("Move exposures file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        # write currency rate information
        if options.writeCurrencies:
            # write currencies file into a tempfile on the remote directory
            # and then do a move.  That is the safest thing to do
            
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='currencies',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            outFileName = '%s/Currencies.%04d%02d%02d.att' % (
                target, d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            logging.info("Writing to %s", tmpfilename)
            outFile = open(tmpfilename, 'w')
            self.writeCurrencyRates(marketDB, modelDB, options, outFile)
            outFile.close()
            logging.info("Move currencies file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        # write asset risk characteristics
        if options.writeRiskAttributes:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='rsk',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            logging.info("Writing to %s", tmpfilename)
            
            outFileName = '%s/%s.%04d%02d%02d.rsk' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            #outFile = open(outFileName, 'w')
            outFile=open(tmpfilename, 'w')
            self.writeAssetRiskAttributes(d, riskModel, exposureAssets, expM,
                                          svDataDict, estU, options, modelDB, marketDB, outFile)
            outFile.close()
            logging.info("Move Risk attribute file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        # write specific covariances
        if options.writeSpecificCovariances:
            outFileName = '%s.%s.isc' % (mnemonic, shortdate)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = TempFile(os.path.join(target, outFileName),
                               shortdate)
            self.writeSpecificCovariances(d, exposureAssets, specCov,
                                          options, outFile.getFile(), riskModel)
            outFile.closeAndMove()
        # write factor returns
        if options.writeFactorReturns:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='ret',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            logging.info("Writing to %s", tmpfilename)
            
            outFileName = '%s/%s.%04d%02d%02d.ret' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(tmpfilename, 'w')
            self.writeFactorReturns(d, riskModel, modelDB, options, outFile)
            outFile.close()
            logging.info("Move Factor returns file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        # write internal factor returns
        if options.writeInternalFactorReturns:
            if hasattr(riskModel, 'twoRegressionStructure') and riskModel.twoRegressionStructure:
                tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='iret',dir=target)
                #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
                os.close(tmpfile[0])
                tmpfilename=tmpfile[1]
                logging.info("Writing to %s", tmpfilename)

                outFileName = '%s/%s.%04d%02d%02d.iret' % (target, mnemonic,
                                                        d.year, d.month, d.day)
                outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
                outFile = open(tmpfilename, 'w')
                self.writeFactorReturns(d, riskModel, modelDB, options, outFile, internal=True)
                outFile.close()
                logging.info("Move Internal Factor returns file %s to %s", tmpfilename, outFileName)
                shutil.move(tmpfilename, outFileName)
                os.chmod(outFileName,0o644)
        # write factor hierarchy
        if options.writeFactorHry:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='hry',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            logging.info("Writing to %s", tmpfilename)
            
            outFileName = '%s/%s.%04d%02d%02d.hry' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(tmpfilename, 'w')
            self.writeIndustryHierarchy(d, riskModel, factors, modelDB, marketDB, options, outFile)
            outFile.close()
            logging.info("Move hry file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
        # write identifier mapping for exposure assets
        if options.writeAssetIdm:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='idm',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            tmpfilename=tmpfile[1]
            
            tmpfile2=tempfile.mkstemp(suffix=shortdate,prefix='idm',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile2[0])
            tmpfilename2=tmpfile2[1]
            
            tmpfile3=tempfile.mkstemp(suffix=shortdate,prefix='idm',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile3[0])
            tmpfilename3=tmpfile3[1]
            
            tmpfile4=tempfile.mkstemp(suffix=shortdate,prefix='idm',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile4[0])
            tmpfilename4=tmpfile4[1]
            
            outFileName = '%s/%s.%04d%02d%02d.idm' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(tmpfilename, 'wt', encoding='utf-8')
            outFileName2 = '%s/%s-CUSIP.%04d%02d%02d.idm' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName2=makeFinalFileName(outFileName2, options.fileFormatVersion)
            outFile2 = open(tmpfilename2, 'wt', encoding='utf-8')
            outFileName3 = '%s/%s-SEDOL.%04d%02d%02d.idm' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName3=makeFinalFileName(outFileName3, options.fileFormatVersion)
            outFile3 = open(tmpfilename3, 'wt', encoding='utf-8')
            outFileName4 = '%s/%s-NONE.%04d%02d%02d.idm' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName4=makeFinalFileName(outFileName4, options.fileFormatVersion)
            outFile4 = open(tmpfilename4, 'wt', encoding='utf-8')
            logging.info("Writing to %s, %s, %s, %s", tmpfilename, tmpfilename2, tmpfilename2, tmpfilename4)
            
            self.writeIdentifierMapping(d, exposureAssetIDs, modelDB, marketDB, options,
                                        outFile, outFile2, outFile3, outFile4, riskModel=riskModel)
            outFile.close()
            outFile2.close()
            outFile3.close()
            outFile4.close()
            
            logging.info("Move file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
            os.chmod(outFileName,0o644)
            logging.info("Move file %s to %s", tmpfilename2, outFileName2)
            shutil.move(tmpfilename2, outFileName2)
            os.chmod(outFileName2,0o644)
            logging.info("Move file %s to %s", tmpfilename3, outFileName3)
            shutil.move(tmpfilename3, outFileName3)
            os.chmod(outFileName3,0o644)
            logging.info("Move file %s to %s", tmpfilename4, outFileName4)
            shutil.move(tmpfilename4, outFileName4)
            os.chmod(outFileName4,0o644)
        # write asset attributes
        if options.writeAssetAtt:
            tmpfile=tempfile.mkstemp(suffix=shortdate,prefix='att',dir=target)
            #tmpfile is a tuple, contain the unix style FD in [0] and the name in [1]
            os.close(tmpfile[0])
            os.chmod(tmpfile[1],0o644)
            tmpfilename=tmpfile[1]                
            outFileName = '%s/%s.%04d%02d%02d.att' % (target, mnemonic,
                                                      d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(tmpfilename, 'w')
            self.writeAssetAttributes(d, riskModel.rmg, set(exposureAssets),
                                      modelDB, marketDB, options, outFile,
                                      useNonCheatHistoricBetas=
                                      riskModel.modelHack.nonCheatHistoricBetas, riskModel=riskModel)
            outFile.close()
            logging.info("Move Asset attribute file %s to %s", tmpfilename, outFileName)
            shutil.move(tmpfilename, outFileName)
        
        # write Goldman model file
        if options.writeGSSModel and riskModel.mnemonic == 'AXUS3-MH':
            rmg = 'US'
            outFileName = '%s/AXUS.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(outFileName, 'w')
            outFile1Name = '%s/AXUS.%04d%02d%02d.esm' % (target, d.year, d.month, d.day)
            outFile1Name=makeFinalFileName(outFile1Name, options.fileFormatVersion)
            outFile1 = open(outFile1Name, 'w')
            outFile2Name = '%s/AXUS-SEDOL.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFile2Name=makeFinalFileName(outFile2Name, options.fileFormatVersion)
            outFile2 = open(outFile2Name, 'w')
            outFile3Name = '%s/AXUS-CUSIP.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFile3Name=makeFinalFileName(outFile3Name, options.fileFormatVersion)
            outFile3 = open(outFile3Name, 'w')
            outFile4Name = '%s/AXUS-NONE.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFile4Name=makeFinalFileName(outFile4Name, options.fileFormatVersion)
            outFile4 = open(outFile4Name, 'w')
            self.writeGSFile(options, marketDB, modelDB, d, rmg,
                             set(exposureAssets),
                             outFile, outFile1, outFile2, outFile3, outFile4)
            outFile.close()
            outFile1.close()
            outFile2.close()
            outFile3.close()
            outFile4.close()
        if options.writeGSSModel and riskModel.mnemonic == 'AXWW21-MH':
            rmg = 'WORLD'
            outFileName = '%s/AXWW.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFileName=makeFinalFileName(outFileName, options.fileFormatVersion)
            outFile = open(outFileName, 'w')
            outFile1Name = '%s/AXWW.%04d%02d%02d.esm' % (target, d.year, d.month, d.day)
            outFile1Name=makeFinalFileName(outFile1Name, options.fileFormatVersion)
            outFile1 = open(outFile1Name, 'w')
            outFile2Name = '%s/AXWW-SEDOL.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFile2Name=makeFinalFileName(outFile2Name, options.fileFormatVersion)
            outFile2 = open(outFile2Name, 'w')
            outFile3Name = '%s/AXWW-CUSIP.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFile3Name=makeFinalFileName(outFile3Name, options.fileFormatVersion)
            outFile3 = open(outFile3Name, 'w')
            outFile4Name = '%s/AXWW-NONE.%04d%02d%02d.gss' % (
                target, d.year, d.month, d.day)
            outFile4Name=makeFinalFileName(outFile4Name, options.fileFormatVersion)
            outFile4 = open(outFile4Name, 'w')
            self.writeGSFile(options, marketDB, modelDB, d, rmg,
                             set(exposureAssets),
                             outFile, outFile1, outFile2, outFile3, outFile4)
            outFile.close()
            outFile1.close()
            outFile2.close()
            outFile3.close()
            outFile4.close()
        
    def writeCompositeConstituents(self, outFile, composites, modelDB, marketDB, options):
        self.writeDateHeader(options, outFile)
        outFile.write('#Columns: Composite AxiomaID|Constituent AxiomaID|Weight\n')
        outFile.write('#Type: ID|ID|NA\n')
        outFile.write('#Unit: ID|ID|Number\n')
        if options.fileFormatVersion >= 4.0:
            outFile.write('#FlatFileVersion: 4.0\n')
        for (compMdlID, (constDate, constituents)) in sorted(
            composites.items()):
            for (asset, weight) in constituents:
                outFile.write('%s|%s|%.5g\n' % (compMdlID.getPublicID(),
                                                asset.getPublicID(),
                                                weight))
        outFile.write('#EOF\n')
        
    def getActiveComposites(self, family, compositeMembers, modelDB, marketDB):
        """Returns a dictionary mapping composite members to their constituent/weight
        pairs for dataDate_.
        Only those composites that have a least one constituent are included.
        """
        issueMapPairs = modelDB.getIssueMapPairs(self.dataDate_)
        marketModelMap = dict((j, i) for (i,j) in issueMapPairs)
        composites = [(marketModelMap.get(c.axioma_id), c) for c in compositeMembers]
        activeComposites = dict()
        marketMap = dict([(i[1].getIDString(), i[0]) for i in issueMapPairs])
        for (compMdlID, composite) in composites:
            if compMdlID is None:
                logging.error('Skipping unmapped composite %s', composite)
                continue
            (constDate, constituents) = modelDB.getCompositeConstituents(
                composite.name, self.dataDate_, marketDB, marketMap=marketMap,
                rollBack=30, compositeAxiomaID=composite.axioma_id)
            if len(constituents) > 0:
                constituents.sort()
                activeComposites[compMdlID] = (constDate, constituents)
            else:
                logging.error('No constituents for composite %s' % composite.name)
        checkCompositesForCompleteness(
            activeComposites, family, issueMapPairs, marketDB, self.dataDate_)
        return activeComposites
    
    def writeCompositeDay(self, options, date, familyName, modelDB, marketDB, vendorDB):
        self.dataDate_ = date
        self.createDate_ = modelDB.revDateTime
        shortdate='%04d%02d%02d' % (date.year, date.month, date.day)
        family = marketDB.getETFFamily(familyName)
        if family == None:
            raise LookupError('Unknown composite family %s' % familyName)
        compositemembers = marketDB.getETFFamilyMembers(family, date, True)
        compositenames=[c.name for c in compositemembers]
        logging.info('%d composites in family %s', len(compositemembers), familyName)
        composites = self.getActiveComposites(family, compositemembers, modelDB, marketDB)
        if len(composites) == 0:
            logging.info('Skipping inactive family %s', familyName)
            return
        logging.info('%d active composites in family %s', len(composites), familyName)
        activeCompMdlIDs = sorted(composites.keys())
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
        #
        # Write .cst (constituents) file
        #
        if options.writeCstFile:
            outFile = TempFile('%s/Composite-%s.%s.cst' %
                               (target, family.name, shortdate),
                               shortdate)
            self.writeCompositeConstituents(outFile.getFile(), composites,
                                            modelDB, marketDB, options)
            outFile.closeAndMove()
        #
        # Write .att file
        #
        if options.writeAttFile:
            issueMapPairs = modelDB.getIssueMapPairs(date)
            marketModelMap = dict((j, i) for (i,j) in issueMapPairs)
            compositeData = [(marketModelMap.get(c.axioma_id), c) for c in compositemembers if c.axioma_id in marketModelMap]
            
            # set up a dictionary of composite names to model subissues and use it to fill in rollover information
            subIDs = modelDB.getAllActiveSubIssues(date)
            mdlSIDMap = dict((i.getModelID(), i) for i in subIDs)
            activeCompSIDs = [mdlSIDMap[i] for i in activeCompMdlIDs]
            
            compositeDataDict=dict((c[1].name, mdlSIDMap[c[0]]) for c in compositeData)
            
            rollOverInfo = dict((mdlSIDMap[i], True) for i in activeCompMdlIDs
                                if date != composites[i][0])
            
            # find all the ETFs that had a rollover for this date and figure out which assets
            # are affected and add that to the rollover information as well
            # 
            rolledOverETFs=getRolledOverETFs(vendorDB, date, compositenames)
            if len(rolledOverETFs) > 0:
                logging.info('Rolled over additional ETFs %s (%s)', ','.join(rolledOverETFs), ','.join([str(compositeDataDict[r]) for r in rolledOverETFs]))
                for r in rolledOverETFs:
                    rollOverInfo[compositeDataDict[r]] = True
            
            outFile = TempFile('%s/Composite-%s.%s.att' %
                               (target, family.name, shortdate),
                               shortdate)
            self.writeAssetAttributes(date, list(), set(activeCompSIDs),
                                      modelDB, marketDB, options, outFile.getFile(),
                                      rollOverInfo=rollOverInfo)
            outFile.closeAndMove()
        #
        # Write .idm file
        #
        if options.writeIdmFile:
            outFile = TempFile('%s/Composite-%s.%s.idm' % (
                    target, family.name, shortdate), shortdate)
            outFile2 = TempFile('%s/Composite-%s-CUSIP.%s.idm' % (
                    target, family.name, shortdate), shortdate)
            outFile3 = TempFile('%s/Composite-%s-SEDOL.%s.idm' % (
                    target, family.name, shortdate), shortdate)
            outFile4 = TempFile('%s/Composite-%s-NONE.%s.idm' % (
                    target, family.name, shortdate), shortdate)
            logging.info("Writing to %s, %s, %s, %s", outFile.getTmpName(), outFile2.getTmpName(), outFile3.getTmpName(), outFile4.getTmpName())
            
            self.writeIdentifierMapping(date, activeCompMdlIDs, modelDB, marketDB,
                                        options, outFile.getFile(), outFile2.getFile(),
                                        outFile3.getFile(), outFile4.getFile())
            outFile.closeAndMove()
            outFile2.closeAndMove()
            outFile3.closeAndMove()
            outFile4.closeAndMove()
    def getTargetDirectory(self, options, d):
        target=options.targetDir
        if options.appendDateDirs:
            target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass
        return target
    
    def writeCountryInfoDay(self, options, date, modelDB, marketDB):
        self.dataDate_ = date
        self.createDate_ = modelDB.revDateTime
        shortdate='%04d%02d%02d' % (date.year, date.month, date.day)
        rmgs = modelDB.getAllRiskModelGroups()
        logging.info('%d active countries', len(rmgs))
        if len(rmgs) == 0:
            logging.info('No active countries. Skipping date')
            return
        target = self.getTargetDirectory(options, date)
        try:
            os.makedirs(target)
        except OSError as e:
            if e.errno != 17:
                raise
            else:
                pass
        #
        # Write .att (country) file
        #
        outFile = TempFile('%s/Countries.%s.att' % (target, shortdate),
                           shortdate)
        self.writeCountryAttributes(outFile.getFile(), rmgs,
                                    modelDB, marketDB, options)
        outFile.closeAndMove()
    
    def writeCountryAttributes(self, outFile, rmgs, modelDB, marketDB, options):
        self.writeDateHeader(options, outFile)
        rmgs = sorted(rmgs, key=lambda a: a.mnemonic)
        retTiming = modelDB.loadReturnsTimingAdjustmentsHistory(1, rmgs, [self.dataDate_])
        retTimingV2 = modelDB.loadReturnsTimingAdjustmentsHistory(1, rmgs, [self.dataDate_], legacy=False)
        retTimingProxy = modelDB.loadReturnsTimingAdjustmentsHistory(1, rmgs, [self.dataDate_], loadProxy=True, legacy=False)
        tradingMarkets = modelDB.getTradingMarkets(self.dataDate_)
        outFile.write('#Columns: ISO Code2|ISO Code3|Description|GMT OffSet|Trading Day|Returns Timing|Currency|Returns Timing V2|Returns Timing Proxy\n')
        outFile.write('#Type: ID|NA|NA|NA|Attribute|Attribute|Attribute|Attribute|Attribute\n')
        outFile.write('#Unit: Text|Text|Text|Number|Number|Number|Text|Number|Number\n')
        for (idx, rmg) in enumerate(rmgs):
            outFile.write('%s|%s|%s|%.5g|%s|%s|%s|%s|%s\n' % (
                    rmg.mnemonic, rmg.iso_code, rmg.description,
                    rmg.gmt_offset, rmg.rmg_id in tradingMarkets and '1' or '',
                    numOrNull(retTiming.data[idx,0], '%.5g'),
                    rmg.getCurrencyCode(self.dataDate_) or '',
                    numOrNull(retTimingV2.data[idx,0], '%.5g'),
                    numOrNull(retTimingProxy.data[idx,0], '%.5g')))
        outFile.write('#EOF\n')

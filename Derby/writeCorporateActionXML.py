import base64
import codecs
import datetime
import logging
import operator
import optparse
import os
import xml.dom.minidom as minidom

from marketdb import MarketDB
from marketdb import MarketID
import riskmodels
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities
from riskmodels import wombat
from riskmodels import writeDerbyFiles
from riskmodels import writeFlatFiles

class TerminationAction(MarketDB.CorporateAction):
    def __init__(self, modelID, marketID, sequenceNumber):
        MarketDB.CorporateAction.__init__(self, sequenceNumber)
        self.modelID = modelID
        self.asset = marketID
    def __str__(self):
        return '(%d, %s, Termination)' % (self.sequenceNumber, self.modelID)
    def __repr__(self):
        return self.__str__()

class AcquisitionAction(MarketDB.CorporateAction):
    def __init__(self, oldModelID, newModelID, newMarketID, ratio, sequenceNumber):
        MarketDB.CorporateAction.__init__(self, sequenceNumber)
        self.modelID = oldModelID
        self.asset = MarketID.MarketID(string=oldModelID.getIDString()[1:] + '_')
        self.newModelID = newModelID
        self.newMarketID = newMarketID
        self.oldToNewRatio = ratio
    def __str__(self):
        return '(%d, %s, Acquisition %s, %g)' % (
            self.sequenceNumber, self.modelID, self.newModelID, self.oldToNewRatio)
    def __repr__(self):
        return self.__str__()

def addCorporateActions(document, root, assetActions, currencyMap, adjTypes,
                        date, encryptActions, assetTypeMapper, assets, showEmptyAssets):
    dateElement = document.createElement('ex_dt')
    dateElement.setAttribute('val', str(date))
    root.appendChild(dateElement)
    for asset in showEmptyAssets:
        if asset not in assetActions:
            assetActions[asset] = list()
    for (asset, actions) in sorted(assetActions.items()):
        if asset not in assets:
            continue
        assetElement = document.createElement('asset')
        assetElement.setAttribute('ax_id', asset.getPublicID())
        assetTypeMapper.addAssetType(assetElement, asset, date)
        dateElement.appendChild(assetElement)
        for ca in actions:
            caElem = document.createElement('action')
            caElem.setAttribute('ca_sequence', str(ca.sequenceNumber))
            if isinstance(ca, MarketDB.StockSplit):
                caElem.setAttribute('type', 'adj_factor')
                caElem.setAttribute('sub_type', adjTypes[ca.adjType])
                caElem.setAttribute('shares_af', str(1.0 / ca.priceRatio))
            elif isinstance(ca, MarketDB.CashDividend):
                caElem.setAttribute('type', 'cash_div')
                caElem.setAttribute('sub_type', ca.subType)
                grossDiv = ca.grossDividend
                caElem.setAttribute('val', str(ca.grossDividend))
                caElem.setAttribute('currency', currencyMap[ca.currency])
            elif isinstance(ca, TerminationAction):
                caElem.setAttribute('type', 'termination')
            elif isinstance(ca, ModelDB.MergerSurvivor):
                # ca_merger_survivor records are treated like a split since the change
                # is only in the underlying Axioma ID
                caElem.setAttribute('type', 'adj_factor')
                caElem.setAttribute('sub_type', 'Exchange')
                caElem.setAttribute('shares_af', str(ca.share_ratio))
            elif isinstance(ca, AcquisitionAction):
                caElem.setAttribute('type', 'acquisition')
                caElem.setAttribute('val', str(1.0 / ca.oldToNewRatio))
                caElem.setAttribute('new_ax_id', ca.newModelID.getPublicID())
                assetTypeMapper.addAssetType(caElem, ca.newModelID, date, 'new_ax_id_')
            else:
                raise ValueError('unsupported corporate action type: %s' % ca)
            if encryptActions:
                assetElement.appendChild(encryptAction(asset, date, caElem, document))
            else:
                assetElement.appendChild(caElem)
    return document

class AssetTypeMapper:
    def __init__(self, mktDb, mdlDb):
        self.etfMap = dict()
        for family in mktDb.getETFFamilies():
            members = mktDb.getAllETFFamilyMembers(family)
            for etf in members:
                if etf.axioma_id is not None:
                    self.etfMap.setdefault(etf.axioma_id, list()).append((etf.from_dt, etf.thru_dt))
                    mdlDb.dbCursor.execute("""SELECT modeldb_id, from_dt, thru_dt FROM issue_map
                       WHERE marketdb_id=:axid_arg""", axid_arg=etf.axioma_id.getIDString())
                    for (modelID, fromDt, thruDt) in mdlDb.dbCursor.fetchall():
                        fromDt = fromDt.date()
                        thruDt = thruDt.date()
                        if fromDt < etf.thru_dt and etf.from_dt < thruDt:
                            self.etfMap.setdefault(ModelID.ModelID(string=modelID),
                                                   list()).append((max(etf.from_dt, fromDt),
                                                                   min(etf.thru_dt, thruDt)))
    
    def addAssetType(self, assetElement, modelId, date, prefix=''):
        """Add the 'type' and for composites the 'composition' attribute to the asset XML element
        depending on the active asset type.
        The two attributes are prefixed with the provided 'prefix' argument
        """
        isComposite = False
        assert isinstance(modelId, ModelID.ModelID)
        for (fromDt, thruDt) in self.etfMap.get(modelId, list()):
            if date >= fromDt and date < thruDt:
                isComposite = True
        if isComposite:
            assetType = 'composite'
            assetElement.setAttribute(prefix + 'composition', 'market.Composition of ' + modelId.getPublicID())
        elif modelId.isCashAsset():
            assetType = 'cash'
        else:
            assetType = 'simple'
        assetElement.setAttribute(prefix + 'type', assetType)

def encryptAction(asset, date, actionElem, document):
    encElem = document.createElement('encryptedaction')
    attributes = [actionElem.attributes.item(i) for i in range(actionElem.attributes.length)]
    val = ':'.join(attr.name + '=' + attr.value for attr in attributes)
    encVal = wombat.scrambleString(val, 25, asset.getPublicID(), date, '')
    base64val = base64.b64encode(encVal)
    encElem.setAttribute('val', str(base64val, 'utf8'))
    return encElem

def processMergerSurvivor(ca, assetActions):
    assetActions.setdefault(ca.modelID, list()).append(ca)
    if ca.cash_payment is not None and ca.cash_payment != 0.0:
        # Create artificial dividend corporate action for the cash payment
        # It has to comes first in sequence so give it the original sequence number
        # and increment the one on the merger-survivor
        divCA = MarketDB.CashDividend(ca.sequenceNumber, ca.asset, ca.cash_payment,
                                      ca.cash_payment, ca.currency_id, False)
        divCA.subType = "gross-dividend"
        divCA.mergerDiv = True
        divCA.modelID = ca.modelID
        assetActions.setdefault(ca.modelID, list()).append(divCA)
        ca.sequenceNumber = ca.sequenceNumber + 1

def processCashDividend(ca, assetActions, assetRmgs):
    """Special processing for cash dividends.
    For AU trading assets, we report the asset_dim_cdiv net value as the "gross-dividend"
    and the difference between the net and gross (if any) as "franking-credit".
    For all other trading countries, the divAmt field is reported as the "gross-dividend".
    That is the asset_dim_cdiv net value for all assets trading in net-dividend countries
    (i.e., GB) and the gross value for all others.
    """
    netVal = ca.netDividend
    if netVal is None or netVal == 0.0:
        netVal = ca.grossDividend
    grossVal = ca.divAmt
    mdlID = ca.modelID
    countryISO = assetRmgs.get(mdlID, (None, None))[0]
    if not countryISO:
        logging.warning("No country for %s", mdlID)
        return
    if countryISO == 'AU':
        if abs(netVal) > 1e-8:
            caNet = MarketDB.CashDividend(ca.sequenceNumber, ca.asset, netVal, netVal,
                                          ca.currency)
            caNet.subType = "gross-dividend"
            caNet.modelID = mdlID
            assetActions.setdefault(mdlID, list()).append(caNet)
        frankingCredit = grossVal - netVal
        if abs(frankingCredit) > 1e-8:
            caFrank = MarketDB.CashDividend(ca.sequenceNumber + 1, ca.asset, frankingCredit,
                                            frankingCredit, ca.currency)
            caFrank.subType = "franking-credit"
            caFrank.modelID = mdlID
            assetActions.setdefault(mdlID, list()).append(caFrank)
    else:
        if abs(grossVal) > 1e-8:
            caGross = MarketDB.CashDividend(ca.sequenceNumber, ca.asset, grossVal, grossVal,
                                            ca.currency)
            caGross.subType = "gross-dividend"
            caGross.modelID = mdlID
            assetActions.setdefault(mdlID, list()).append(caGross)

def addCorporateActionToList(ca, assetActions, assetRmgs):
    if isinstance(ca, MarketDB.CashDividend):
        processCashDividend(ca, assetActions, assetRmgs)
    elif isinstance(ca, ModelDB.MergerSurvivor):
        processMergerSurvivor(ca, assetActions)
    else:
        assetActions.setdefault(ca.modelID, list()).append(ca)

def sortActionsByAssetAndSequence(actionList, assetRmgs):
    assetActions = dict()
    for ca in sorted(actionList, key=operator.attrgetter('sequenceNumber')):
        addCorporateActionToList(ca, assetActions, assetRmgs)
    
    for mdlID in assetActions.keys():
        cas = assetActions[mdlID]
        if len(cas) > 1:
            mergers = [ca for ca in cas if isinstance(ca, ModelDB.MergerSurvivor)]
            if len(mergers) > 0:
                # Remove all corporate actions up to the merger and replace them with
                # the corporate actions from the old MarketID mapped to the current ModelID
                mergerCa = mergers[0]
                mergerIdx = cas.index(mergerCa)
                if mergerIdx > 0 and hasattr(cas[mergerIdx-1], 'mergerDiv'):
                    # If prev dividend belongs to merger, keep it as well
                    mergerIdx = mergerIdx - 1
                oldAssetCas = [ca for ca in actionList if ca.asset.getIDString() == mergerCa.old_marketdb_id and ca.sequenceNumber < mergerCa.sequenceNumber]
                cas = cas[mergerIdx:]
                assetActions[mdlID] = cas
                for ca in oldAssetCas:
                    ca.modelID = mdlID
                    addCorporateActionToList(ca, assetActions, assetRmgs)
                cas = sorted(cas, key=operator.attrgetter('sequenceNumber'))
            for idx in range(len(cas) - 1):
                if cas[idx].sequenceNumber >= cas[idx+1].sequenceNumber:
                    logging.warning('Sequence number conflict for asset %s, seq %d',
                                 mdlID.getIDString(), cas[idx].sequenceNumber)
                    cas[idx + 1].sequenceNumber = cas[idx].sequenceNumber + 1
            assetActions[mdlID] = cas
    return assetActions

def getAdjustmentFactorTypes(mktDb):
    query = """select id, name from meta_codes where code_type='asset_dim_af:adjtype'"""
    mktDb.dbCursor.execute(query)
    afTypes = {1: 'split', 2: 'split',
               3: 'spin-off',
               4: 'stock-dividend',
               5: 'rights-issue',
               10: 'CN-stock-reform', 11: 'CN-stock-reform',
               6: 'other', 7: 'other', 8: 'other', 9: 'other'
               }
    unmappedAfIds = set()
    for afID, afName in mktDb.dbCursor.fetchall():
        if afID not in afTypes:
            logging.fatal('Unmapped adjustment factor type [%s, %s]. Please update the extraction code',
                          afID, afName)
            unmappedAfIds.add(afID)
    if len(unmappedAfIds) > 0:
        raise KeyError('Unmapped adjustment factor types')
    return afTypes

def getAssetRMGMapping(mdlDb):
    mdlDb.dbCursor.execute("""SELECT si.issue_id, rmg.mnemonic, rmg.rmg_id FROM sub_issue si
       JOIN risk_model_group rmg ON si.rmg_id=rmg.rmg_id""")
    assetRmgs = dict((ModelID.ModelID(string=i[0]), i[1:]) for i in  mdlDb.dbCursor.fetchall())
    return assetRmgs

def getTerminationActions(mdlDb, date):
    # We don't have good termination information right now so skip
    # termination actions for now which also matches how cumulative
    # returns are treating assets that go away.
    return list()
    mdlDb.dbCursor.execute("""SELECT issue_id, marketdb_id FROM issue i JOIN issue_map im
        ON i.issue_id = im.modeldb_id AND im.from_dt <= :dt_arg AND im.thru_dt >= :dt_arg
        WHERE i.thru_dt=:dt_arg""", dt_arg=date)
    return [TerminationAction(ModelID.ModelID(string=i[0]), MarketID.MarketID(string=i[1]), 100)
            for i in  mdlDb.dbCursor.fetchall()]

def getCurrencyConvergences(mktDb, currencyMap, date):
    """Create a corporate action for each currency convergence.
    """
    mktDb.dbCursor.execute("""SELECT ccyOld.code, ccyNew.code, old_to_new_rate
        FROM currency_mod_convergence
        JOIN currency_ref ccyOld ON ccyOld.id=old_id
        JOIN currency_ref ccyNew ON ccyNew.id=new_id
        WHERE dt=:dt_arg""", dt_arg=date)
    return [AcquisitionAction(ModelID.ModelID(string='DCSH_%s__' % oldCode),
                              ModelID.ModelID(string='DCSH_%s__' % newCode),
                              MarketID.MarketID(string='CSH_%s___' % newCode), ratio, -1)
            for (oldCode, newCode, ratio) in mktDb.dbCursor.fetchall()]

def getCashAssetDividends(mdlDb, currencyMap, date):
    """Create cash dividend corporate actions for each day that a cash asset has a return.
    Cash assets have a fixed price and the return comes from interest payments of the
    risk-free rate.
    """
    revCurrencyMap = dict((iso, code) for (code, iso) in currencyMap.items())
    mdlDb.dbCursor.execute("""SELECT marketdb_id, tr FROM sub_issue_return_active
        JOIN sub_issue si ON si.sub_id=sub_issue_id AND si.from_dt <= :dt_arg AND si.thru_dt > :dt_arg
        JOIN issue_map im ON im.modeldb_id=si.issue_id AND im.from_dt <= :dt_arg AND im.thru_dt > :dt_arg
        WHERE dt=:dt_arg AND sub_issue_id  LIKE 'DCSH\\_%' escape '\\' 
        AND tr IS NOT NULL""", dt_arg=date)
    cashDivs = list()
    for (mktId, retVal) in mdlDb.dbCursor.fetchall():
        asset = MarketID.MarketID(string=mktId)
        currencyISO = mktId[4:7]
        currency = revCurrencyMap[currencyISO]
        caCash = MarketDB.CashDividend(0, asset, retVal, retVal, currency)
        caCash.subType = "net-dividend"
        cashDivs.append(caCash)
    return cashDivs

def buildDateList(args):
    if len(args) == 1:
        dateRanges = [i.strip() for i in args[0].split(',')]
        dates = set()
        for dRange in dateRanges:
            if dRange.find(':') == -1:
                dates.add(Utilities.parseISODate(dRange))
            else:
                (startDate, endDate) = [i.strip() for i in dRange.split(':')]
                startDate = Utilities.parseISODate(startDate)
                endDate = Utilities.parseISODate(endDate)
                dates.update([startDate + datetime.timedelta(i)
                              for i in range((endDate-startDate).days + 1)])
    else:
        startDate = Utilities.parseISODate(args[0])
        endDate = Utilities.parseISODate(args[1])
        dates = [startDate + datetime.timedelta(i)
                 for i in range((endDate-startDate).days + 1)]
    dates = sorted(dates, reverse=True)
    return dates

def getModelFamilies(mdlDb):
    """
        get all model family to model name mappings from the database
    """
    modelDict={}
    mdlDb.dbCursor.execute("""select rm.name, rm.mnemonic from risk_model rm
        where exists (select * from risk_model_serie rms where rms.distribute=1 and rm.model_id=rms.rm_id)""")
    for r in mdlDb.dbCursor.fetchall():
        modelName,mnemonic=r
        familyName=modelName[0:modelName.find('Axioma') + len('Axioma')]
        modelDict[mnemonic]=familyName
    return modelDict
        
def validModelFamilies(modelFamilies, mdlDb):
    mdlDb.dbCursor.execute("""SELECT rm.name FROM risk_model rm JOIN risk_model_serie rms
       ON rm.model_id=rms.rm_id
       WHERE rms.distribute=1""")
    activeModelNames = [i[0] for i in mdlDb.dbCursor.fetchall()]
    activeModelFamilies = set(i[0:i.find('Axioma') + len('Axioma')] for i in activeModelNames)
    unknownFamilies = set(modelFamilies) - activeModelFamilies
    if len(unknownFamilies) > 0:
        logging.fatal('Unknown model families: %s', ','.join(unknownFamilies))
        logging.info('Supported model families: %s', ','.join(sorted(activeModelFamilies)))
        return False
    return True

def validCompositeFamilies(compositeFamilies, mktDb):
    activeFamilies = set(f.name for f in mktDb.getETFFamilies() if f.distribute)
    unknownFamilies = set(compositeFamilies) - activeFamilies
    if len(unknownFamilies) > 0:
        logging.fatal('Unknown composite families: %s', ','.join(unknownFamilies))
        logging.info('Supported composite families: %s', ','.join(sorted(activeFamilies)))
        return False
    return True

def getRiskModelFamilyToModelMap(modelFamilies, mdlDb, mktDb):
    familyMap = dict()
    for family in modelFamilies:
        mdlDb.dbCursor.execute("""SELECT rm.name, rms.rm_id, rms.revision
           FROM risk_model rm JOIN risk_model_serie rms ON rm.model_id=rms.rm_id
           WHERE rms.distribute=1 AND rm.name like :family_arg || '%'""", family_arg=family)
        for (rmName, rmID, rmRevision) in mdlDb.dbCursor.fetchall():
            logging.debug('Adding model %s, revision %s to %s family', rmName, rmRevision, family)
            riskModel = riskmodels.modelRevMap[(rmID, rmRevision)](mdlDb, mktDb)
            familyMap.setdefault(family, list()).append(riskModel)
    return familyMap

def getCompositeFamilyObjects(compositeFamilies, mdlDb, mktDb):
    return set(mktDb.getETFFamily(name) for name in compositeFamilies)

def processModelFamilyDay(activeModels, d, modelDB, marketDB, options, requireCurrentDate=True):
    # Determine previous trading day for each active model (should be the same, no?
    dateRange = [d - datetime.timedelta(days=i) for i in range(30)]
    modelPrevRMI = dict()
    modelCurrRMI = dict()
    for model in activeModels:
        modelDays = modelDB.getRiskModelInstances(model.rms_id, dateRange)
        assert(len(modelDays) >= 1)
        if requireCurrentDate and modelDays[0].date != d:
            raise ValueError('model %s not active on requested date %s' % (model.mnemonic, d))
        modelCurrRMI[model] = modelDays[0]
        if  len(modelDays) > 1 and modelDays[0].date == d:
            modelPrevRMI[model] = modelDays[1]
    # get active assets, i.e. in model on requested date and previous trading day
    activeAssets = set()
    options.preliminary = False
    options.newRiskFields = False
    for model in activeModels:
        rmis = [modelCurrRMI[model]]
        if model in modelPrevRMI:
            rmis.append(modelPrevRMI[model])
        for rmi in rmis:
            model.setFactorsForDate(rmi.date, modelDB)
            try:
                exposureAssets = writeDerbyFiles.buildMasterAssetList(modelDB, marketDB, model, options, rmi.date)[2]
                activeAssets |= set(exposureAssets)
            except ValueError:
                exposureAssets = list()
            logging.debug('%d exposure assets on %s, %s total', len(exposureAssets), rmi.date, len(activeAssets))
    activeAssets = set(i.getModelID() for i in activeAssets)
    # get corporate actions for all dates between oldest previous trading day and d
    lookBack = 0
    if len(modelPrevRMI) > 0:
        lookBack = d - min(i.date for i in modelPrevRMI.values())
    return (activeAssets, lookBack)

def createCADocument():
    # create skeleton XML document
    imp = minidom.getDOMImplementation()
    dtd = imp.createDocumentType('corporate_actions', '', 'axiomaCorpActions.dtd')
    document = imp.createDocument('http://www.w3.org/1999/xhtml', 'corporate_actions', dtd)
    root = document.documentElement
    root.setAttribute('createtime', str(datetime.datetime.now()))
    return (document, root)

def determineRMGCalendars(d, lookBack, assets, assetRmgs, modelDB):
    """Per country, determine the trading days that should be included in this
    extraction. That's all trading days between d - lookBack (exclusive) and d (inclusive).
    The date range for which corporate actions should be retrieved is then
    from the trading day prior to the ones in the list above (exclusive), up to the last trading
    day in the list (inclusive).
    """
    rmgIds = set(assetRmgs[a][1] for a in assets if (not a.isCashAsset()) and a in assetRmgs)
    rmgs = [modelDB.getRiskModelGroup(rmgId) for rmgId in rmgIds]
    rmgCalendars = dict()
    for rmg in rmgs:
        tradingDays = modelDB.getDateRange(rmg, d - lookBack + datetime.timedelta(days=1),
                                                  d, excludeWeekend=False)
        if len(tradingDays) > 0:
            priorDays = modelDB.getDateRange(rmg, tradingDays[0] - datetime.timedelta(days=90),
                                                    tradingDays[0] - datetime.timedelta(days=1),
                                                    excludeWeekend=False)
            prevDay = (len(priorDays) == 0 and tradingDays[0]) or priorDays[-1]
            calendarDays = sorted([tradingDays[-1] - datetime.timedelta(days=i)
                                   for i in range((tradingDays[-1] - prevDay).days)])
        else:
            calendarDays = list()
        rmgCalendars[rmg.rmg_id] = (tradingDays, calendarDays)
    return rmgCalendars

def convertToModelID(actions, date, modelDB):
    """Ensure each corporate action is tagged with the corresponding modeldb ID.
    """
    issueMapPairs = modelDB.getIssueMapPairs(date)
    marketModelMap = dict((mkt, mdl) for (mdl, mkt) in issueMapPairs)
    for ca in actions:
        mdlID = None
        if isinstance(ca, ModelDB.MergerSurvivor):
            mdlID = ca.modeldb_id
            ca.asset = MarketID.MarketID(string=ca.new_marketdb_id)
        elif hasattr(ca, 'modelID'):
            mdlID = ca.modelID
        elif ca.asset in marketModelMap:
            mdlID = marketModelMap[ca.asset]
        if mdlID is None:
            logging.warning('Droping corporate action [%s]. No mapping to an active model ID', ca)
            ca.modelID = None
        ca.modelID = mdlID
    return [ca for ca in actions if ca.modelID is not None]

def mergeCorporateActionLists(holidayActions, tradingDayActions):
    """Merge the corporate actions from trading holidays with the trading day
    actions.
    The sequence numbers for the holiday actions start from 1. If that conflicts
    with the trading day actions then those are pushed higher.
    """
    for idx, ca in enumerate(holidayActions):
        ca.sequenceNumber = idx + 1
    currentIdx = len(holidayActions)
    for ca in tradingDayActions:
        if currentIdx >= ca.sequenceNumber:
            ca.sequenceNumber = currentIdx + 1
        currentIdx = ca.sequenceNumber
    return holidayActions + tradingDayActions

def gatherCorporateActions(calendarDays, currencyMap, withCurrencyConvergences, assetRmgs, marketDB, modelDB):
    """Get all corporate actions for the listed calendar days.
    Returns a tuple with a dictionary (date -> (asset -> list(CA)))
    and the set of assets that have corporate actions in the dictionary.
    """
    corporateActions = dict()
    assetsWithCAs = set()
    for caDate in calendarDays:
        actionList = marketDB.getStockSplits(caDate) + marketDB.getCashDividends(caDate) + getTerminationActions(modelDB, caDate)
        actionList += getCashAssetDividends(modelDB, currencyMap, caDate) + modelDB.getCAMergerSurvivors(caDate)
        if withCurrencyConvergences:
            actionList += getCurrencyConvergences(marketDB, currencyMap, caDate)
        actionList = convertToModelID(actionList, caDate, modelDB)
        assetCADict = sortActionsByAssetAndSequence(actionList, assetRmgs)
        corporateActions[caDate] = assetCADict
        assetsWithCAs.update(assetCADict.keys())
    return (corporateActions, assetsWithCAs)

def moveCorporateActionsToTradingDays(corporateActions, activeAssets, assetsWithCAs, assetRmgs, rmgCalendars,
                                      extractDays):
    # Move corporate actions on trading holidays to their corresponding trading day
    # and remove them altogether if the trading holiday doesn't correspond to a trading
    # day that is being extracted
    for asset in activeAssets & assetsWithCAs:
        if asset.isCashAsset():
            # Don't apply trading calendars to cash assets, they "trade" every calendar day
            continue
        assetTradingDays, assetCalendarDays = rmgCalendars[assetRmgs[asset][1]]
        assetCalendarDays = sorted(set(assetCalendarDays) | set(extractDays))
        assetActions = list()
        for caDate in assetCalendarDays:
            if caDate in assetTradingDays:
                if len(assetActions) > 0:
                    currentActions = corporateActions[caDate].pop(asset, list()) 
                    assetActions = mergeCorporateActionLists(assetActions, currentActions)
                    corporateActions[caDate][asset] = assetActions
                    assetActions = list()
            else:
                assetActions.extend(corporateActions[caDate].pop(asset, list()))

def createXML(d, lookBack, marketDB, modelDB, currencyMap, adjTypes, assetRmgs, options,
              assetTypeMapper, activeAssets):
    (document, root) = createCADocument()
    rmgCalendars = determineRMGCalendars(d, lookBack, activeAssets, assetRmgs, modelDB)
    extractDays = list(d - datetime.timedelta(days=i) for i in range(lookBack.days))
    calendarDays = set(extractDays)
    for rmgTradingDays, rmgCalendarDays in rmgCalendars.values():
        calendarDays.update(rmgCalendarDays)
    corporateActions, assetsWithCAs = gatherCorporateActions(
        calendarDays, currencyMap, options.withCurrencyConvergences, assetRmgs, marketDB, modelDB)
    moveCorporateActionsToTradingDays(corporateActions, activeAssets, assetsWithCAs, assetRmgs, rmgCalendars,
                                      extractDays)
    for caDate in extractDays:
        assetCADict = corporateActions[caDate]
        if len(assetCADict) > 0:
            addCorporateActions(document, root, assetCADict, currencyMap, adjTypes,
                                caDate, options.encryptActions, assetTypeMapper,
                                activeAssets, list())
        else:
            logging.info('No corporate actions on %s', d)
    return document

def buildTargetDirPath(d, options, makeDirs):
    target = options.targetDir
    if options.appendDateDirs:
        target = os.path.join(target, '%04d' % d.year, '%02d' % d.month)
        if makeDirs:
            try:
                os.makedirs(target)
            except OSError as e:
                if e.errno != 17:
                    raise
                else:
                    pass
    return target

def writeXMLToFile(document, family, d, options):
    # write XML file 
    target = buildTargetDirPath(d, options, True)
    dateStr = d.strftime('%Y%m%d')
    outName = os.path.join(target, 'CorpActions-%s-%s.xml' % (family, dateStr))
    with writeFlatFiles.TempFile(outName, dateStr) as outFile:
        with outFile.getFile() as g:
            document.writexml(g, '  ', '  ', newl='\n', encoding='UTF-8')
    logging.info('Done writing %s', d)

def mergeAuxFile(document, familyName, date, options):
    """If an auxiliary file exists (CorpActions-<family>-<date>.aux), merge its date sections
    with the current document.
    """
    target = buildTargetDirPath(date, options, True)
    dateStr = date.strftime('%Y%m%d')
    auxFileName = os.path.join(target, 'CorpActions-%s-%s.aux' % (familyName, dateStr))
    if os.path.exists(auxFileName):
        logging.info('Found auxiliary file %s', auxFileName)
        auxDoc = minidom.parse(auxFileName)
        removeTextElements(auxDoc.documentElement)
        dateElements = auxDoc.getElementsByTagName('ex_dt')
        root = document.documentElement
        for dateElement in dateElements:
            root.appendChild(dateElement.parentNode.removeChild(dateElement))

def removeTextElements(node):
    for el in list(node.childNodes):
        if el.nodeType == minidom.Node.TEXT_NODE:
            node.removeChild(el)
        else:
            removeTextElements(el)

def processCompositeFamilyDay(family, currencyMap, assetTypeMapper, date, adjTypes,
                              assetRmgs, modelDB, marketDB, options):
    # We publish composites every weekday except Jan 1, so the previous day is the first preceeding day
    # that is not Jan 1 or a weekend
    prevWeek = [date - datetime.timedelta(days + 1) for days in range(7)]
    prevWeekDays = [i for i in prevWeek if i.isoweekday() <= 5 and not (i.month == 1 and i.day == 1)]
    fileFormat_ = writeFlatFiles.FlatFilesV3()
    activeComposites = set()
    for d in [prevWeekDays[0], date]:
        fileFormat_.dataDate_ = d
        compositemembers = marketDB.getETFFamilyMembers(family, d, True)
        compositenames=[c.name for c in compositemembers]
        logging.info('%d composites in family %s on %s', len(compositemembers), family.name, d)
        composites = fileFormat_.getActiveComposites(family, compositemembers, modelDB, marketDB)
        if len(composites) > 0:
            logging.info('%d active composites in family %s on %s', len(composites), family.name, d)
            activeComposites |= set(composites.keys())
    return (activeComposites, date - prevWeekDays[0])

def getActiveModels(modelList, d, modelDB, allowPartialModels):
    activeModels = list()
    for riskModel in modelList:
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi != None and (rmi.has_risks or allowPartialModels):
            riskModel.setFactorsForDate(d, modelDB)
            activeModels.append(riskModel)
    return activeModels

def main():
    usage = "usage: %prog [options] <startdate or datelist> <end-date>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("--no-encryption", action="store_false",
                             dest="encryptActions", default=True,
                             help="Don't encrypt action values")
    cmdlineParser.add_option("--model-families", action="store",
                             default='', dest="modelFamilies",
                             help="comma-separated list of risk model families")
    cmdlineParser.add_option("--model-names", action="store",
                             default='', dest="modelNames",
                             help="comma-separated list of risk model names")
    
    cmdlineParser.add_option("-c", "--composite-families", action="store",
                             default='', dest="compositeFamilies",
                             help="comma-separated list of composite families")
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for output files")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    cmdlineParser.add_option("--allow-partial-models", action="store_true",
                         default=False, dest="allowPartialModels",
                         help="extract even if the risk model is incomplete")
    cmdlineParser.add_option("--with-currency-convergences", action="store_true",
                         default=False, dest="withCurrencyConvergences",
                         help="include currency convergences as corporate actions")
    
    Utilities.addDefaultCommandLine(cmdlineParser)
    
    (options, args) = cmdlineParser.parse_args()
    options.supplementalData =False

    if len(args) < 1 or len(args) > 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)
    dates = buildDateList(args)
    modelDict=modelDB.getModelFamilies()
    
    if options.modelFamilies:
        modelFamilies = [i.strip() for i in options.modelFamilies.split(',') if len(i) > 0]
    else:
        modelNames = [i.strip() for i in options.modelNames.split(',') if len(i) > 0]
        modelNames=[i for i in modelNames if (i=='AXTW-MH-S' or i[-3:]=='-MH')]
        logging.info("Working on model names %s", ','.join(modelNames))
        # make sure to only do this for -MH models other than TW where we allow MH-S model
        modelFamilies=list(set([modelDict[i] for i in modelNames if i in modelDict]))
    logging.info("Working on families %s", ','.join(modelFamilies))    
    if not validModelFamilies(modelFamilies, modelDB):
            return
                                                                
    familyModelMap = getRiskModelFamilyToModelMap(modelFamilies, modelDB, marketDB)
    
    compositeFamilies = [i.strip() for i in options.compositeFamilies.split(',') if len(i) > 0]
    if not validCompositeFamilies(compositeFamilies, marketDB):
        return
    compositeFamilyObjects = getCompositeFamilyObjects(compositeFamilies, modelDB, marketDB)
    
    currencyMap = marketDB.getCurrencyISOCodeMap()
    assetTypeMapper = AssetTypeMapper(marketDB, modelDB)
    adjTypes = getAdjustmentFactorTypes(marketDB)
    assetRmgs = getAssetRMGMapping(modelDB)
    for d in dates:
        logging.info('Processing %s', d)
        for (modelFamily, models) in familyModelMap.items():
            activeModels = getActiveModels(models, d, modelDB, options.allowPartialModels)
            if len(activeModels) > 0:
                (activeAssets, lookBack) = processModelFamilyDay(activeModels, d, modelDB,
                                                                 marketDB, options)
                document = createXML(d, lookBack, marketDB, modelDB, currencyMap, adjTypes,
                                     assetRmgs, options, assetTypeMapper, activeAssets)
                mergeAuxFile(document, modelFamily, d, options)
                writeXMLToFile(document, modelFamily, d, options)
            else:
                logging.info('Skipping %s for %s model family. No active models', d, modelFamily)
        for compositeFamily in compositeFamilyObjects:
            if d.isoweekday() > 5 or (d.day == 1 and d.month == 1):
                logging.info('Skipping weekends and Jan-1, %s', d)
                continue
            (activeComposites, lookBack) = processCompositeFamilyDay(
                compositeFamily, currencyMap, assetTypeMapper, d, adjTypes,
                assetRmgs, modelDB, marketDB, options)
            if len(activeComposites) == 0:
                logging.info('Skipping composite family %s. No active members', compositeFamily.name)
            else:
                document = createXML(d, lookBack, marketDB, modelDB, currencyMap, adjTypes,
                                     assetRmgs, options, assetTypeMapper, activeComposites)
                mergeAuxFile(document, compositeFamily.name, d, options)
                writeXMLToFile(document, compositeFamily.name, d, options)
    
    modelDB.finalize()
    marketDB.finalize()

if __name__ == '__main__':
    main()

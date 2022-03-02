import collections
import datetime
import logging
import numpy
import operator
import copy
from marketdb import MarketDB
from marketdb import MarketID
from marketdb import Utilities as Utilities
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities as rmUtils
from riskmodels import TransferSources
from riskmodels import DescriptorSources
from riskmodels import ResearchDescriptorSources
from riskmodels import AssetProcessor
from riskmodels import AssetProcessor_V4
from riskmodels import Matrices
from riskmodels.Matrices import ExposureMatrix
from riskmodels import Classification

class ProcessClassification:
    PrefInd = collections.namedtuple('PrefInd', ['date', 'subIndFilter'])
    
    """Class to combine various classification sources into the
    model classification.
    The preferIndustry configuration option contains a list of
    industries and cut-off dates in the target distribution. For those
    industries, if the asset was classified in this industry on the closest
    date after the cut-off date, then it should be classified in that
    industry before the cut-off date even if an original GICS assignment
    exists that puts it elsewhere.
    The preferIndustry setting takes an optional list of sub-industry names
    which can be used to restrict which industry assignments should be overridden.
    By default, the preferred industry is applied irrespective of what the normal
    assignment would have been. But if the list of sub-industries is populated,
    the override only applies if the normal assignment is based on one of the
    source sub-industries in the list. This can be used if a (sub-)industry
    is terminated and its constituents distributed among a list of pre-existing industries.
    By declaring these industries preferred for assets classified in the terminated
    (sub-)industry, they will be assigned to their new homes without impacting the
    history of assets that ended up there coming from other industries.
    """
    def __init__(self, connections, config, section):
        self.excludeTypeList = [None] + AssetProcessor.noTransferList
        self.connections = connections
        self.axidRanges = self.connections.axidRanges
        if config.has_option(section, 'preferIndustry'):
            preferIndustries = config.get(
                section, 'preferIndustry').split('|')
        else:
            preferIndustries = list()
        preferIndustries = [pi.strip().split(':') for pi in preferIndustries]
        preferIndustries = [(pi[0], self.PrefInd(Utilities.parseISODate(pi[1]), pi[2:]))
                            for pi in preferIndustries]
        if len(preferIndustries) > 0:
            modelDB = connections.modelDB
            # Get target classification and replace industry names with IDs
            targetCls = config.get(section, 'targetCls')
            (familyName, memberName, clsDateStr) = targetCls.split(':')
            clsDate = Utilities.parseISODate(clsDateStr)
            clsFamily = modelDB.getMdlClassificationFamily(familyName)
            clsMembers = modelDB.getMdlClassificationFamilyMembers(clsFamily)
            clsMember = [i for i in clsMembers if i.name==memberName][0]
            clsRevision = modelDB.getMdlClassificationMemberRevision(
                clsMember, clsDate)
            modelDB.dbCursor.execute("""SELECT name, id
               FROM classification_ref where revision_id=:rev_arg""",
                                      rev_arg=clsRevision.id)
            nameIdMap = dict(modelDB.dbCursor.fetchall())
            self.preferIndustry =  { nameIdMap[name]: value for (name, value)
                                     in preferIndustries}
        else:
            self.preferIndustry = dict()

        assetTypeFamily = self.connections.marketDB.getClassificationFamily('ASSET TYPES')
        atMembers = dict([(i.name, i) for i in self.connections.marketDB.getClassificationFamilyMembers(assetTypeFamily)])
        self.atMember = atMembers.get('Axioma Asset Type', None)

    def process(self, axiomaId, date, values):
        """Return the first value in the list that is not None
        and None otherwise. If preferred industries are configured and present for an asset,
        use them to override the normal selection if the normal assignment is in the filter list."""
        # check if the asset was classified in one of the preferred
        # industries active after the given date as not a guess.
        axidRange = self.axidRanges[axiomaId]
        if date < axidRange[0] or date >= axidRange[1]:
            return None
        availableIndustries = [v for v in values if v is not None]
        preferredInd = self.__findPreferredIndustry(date, availableIndustries)
        normalSelection = availableIndustries[0] if len(availableIndustries) > 0 else None
        if preferredInd is not None:
            # only apply the preferred industry if the normal assignment is in the filter list
            # or the list is empty
            subIndFilter = self.preferIndustry[preferredInd.classificationIdAndWeight[0][0]].subIndFilter
            if len(subIndFilter) > 0 and normalSelection.origClassification not in subIndFilter:
                preferredInd = None
        # If we have an assignment to a preferred industry and the normal
        # selection isn't one, check if we should backdate
        if preferredInd is not None and \
           normalSelection.classificationIdAndWeight[0][0] != preferredInd.classificationIdAndWeight[0][0]:
            val = Utilities.Struct(copy=preferredInd)
            val.src_id = 5
            if val.ref[:8] != 'backdate':
                val.ref = 'backdate preferred industry; %s' % val.ref
            normalSelection = val
        if normalSelection is None and not axiomaId.isCashAsset():
            aTypeRevision = self.connections.marketDB.getClassificationMemberRevision(self.atMember, date)
            aTypeData = self.connections.modelDB.getMktAssetClassifications(\
                                aTypeRevision, [axiomaId], date, self.connections.marketDB).get(axiomaId, None)
            assetType = None
            if aTypeData is not None:
                assetType = aTypeData.classification.code
            if assetType not in self.excludeTypeList:
                logging.warning('No classification for type %s asset %s on %s', assetType, axiomaId, date)
        return normalSelection
    
    def __findPreferredIndustry(self, date, values):
        """Find the classifications that come from future classifications associated with the preferred industries.
        That means the source revision from_dt has to match the cut-off date and the assignment has to be to the
        preferred industry and has to be an original GICS assignment (non-guess).
        Prefer industries are only applied from future classification, i.e. from_dt > transfer date - grace period
        where the grace period accounts for the fact the GICS doesn't always move assets on the revision start date.
        If multiple preferred industry assignment are found, take the one with the largest cut-off date.
        """
        GRACE_DAYS = TransferSources.FutureClassification.PREF_GRACE_DAYS
        # Filter out guesses and multiple assignments
        values = [val for val in values if not val.isGuess]
        # Filter out past preferred industries
        values = [val for val in values if val.sourceRevision.from_dt >= date - GRACE_DAYS]
        prefIndustries = [v for v in values if v.classificationIdAndWeight[0][0] in self.preferIndustry
                          and v.sourceRevision.from_dt == self.preferIndustry[v.classificationIdAndWeight[0][0]].date]
        prefIndustries = sorted(prefIndustries, key=operator.attrgetter('changeDate'))
        return prefIndustries[0] if len(prefIndustries) > 0 else None
        

class ProcessPriorityList:
    """Class that uses the first value that is not None.
    """
    def __init__(self, connections, config, section):
        self.connections = connections
    
    def process(self, axiomaId, date, values):
        """Return the first value in the list that is not None
        and None otherwise."""
        for val in values:
            if val is not None:
                return val
        return None

class ProcessPriorityListFund:
    """Class that uses the first value that is not None.
    """
    def __init__(self, connections, config, section):
        self.connections = connections
    
    def process(self, sid, date, effDate, values):
        """Return the first value in the list that is not None
        and None otherwise."""
        for val in values:
            if val is not None:
                return val
        return None

class ProcessRFRates:
    """Class that uses the first value that is not None.
    Warn if a currency does not receive a rate during its lifetime
    """
    def __init__(self, connections, config, section):
        self.connections = connections
    
    def process(self, currency, date, values):
        """Return the first value in the list that is not None
        and None otherwise."""
        for val in values:
            if val is not None:
                return val
        if date >= currency[1] and date < currency[2]:
            logging.warning('No risk-free rate for %s on %s', currency[0], date)
        return None

def pruneIDList(idList, connections, assetType=['all']):
    """prune the list of IDs passed in by assetType.  Use the asset-usage to do the pruning
       equities = STATMOD,EXMOD,CONST,CASH,MOD
       futures = EXFUTCOM,FUTCOM 
       mutualfunds = PRCTS,IDONLY

       Also do the pruning on either issue-ids or sub-issue-ids regardless of which is passed in
    """
    if 'all' in assetType:
        return idList

    if len(idList) == 0:
        return idList

    usageTypes=[]
    tableName = 'ISSUE_MAP'
    if 'futures' in assetType:
        usageTypes +=['EXFUTCOM', 'FUTCOM']
        tableName = 'FUTURE_ISSUE_MAP'
    if 'mutualfunds' in assetType:
        usageTypes += ['PRCTS', 'IDONLY']
    if 'equities' in assetType:
        usageTypes +=['STATMOD', 'EXMOD', 'CONST' ,'CASH', 'MOD']

    # need to use both modelDB and marketDB connections
    mdlCur = connections.modelDB.dbCursor
    mktCur = connections.marketDB.dbCursor
  
    # first get all the marketdb IDs out of the modeldb so that we can use those to pump into the marketdb
    allidsMap = dict([(i.string[:10],i) for i in idList])
    query="""select modeldb_id, marketdb_id from issue_map union select modeldb_id, marketdb_id from future_issue_map"""
    mdlCur.execute(query)
    results = mdlCur.fetchall()
    idMap= {}
    for r in results:
        if r[0] in allidsMap:
            idMap.setdefault(MarketID.MarketID(string=r[1]), list()).append(r[0])
    #idMap = dict([(MarketID.MarketID(string=r[1]),r[0]) for r in results if r[0] in allidsMap])
   
    marketDB =  connections.marketDB
    clsFamily = marketDB.getClassificationFamily('USAGES')
    clsMember = marketDB.getClassificationFamilyMembers(clsFamily)
    thisMember = [mem for mem in clsMember if mem.name=='Axioma Usages'][0]
    dt=datetime.date.today()
    thisRevision = marketDB.getClassificationMemberRevision(thisMember, dt)
    # for each of the three levels of axioma asset level go ahead and fill up thevalues
    typeMap=marketDB.getAssetClassificationRanges(thisRevision, list(idMap.keys()))
    oldLength = len(idList)
    res=[idMap[k] for (k, v) in typeMap.items()  if len([vv for vv in v if vv.clsName in usageTypes]) > 0 ]
    resultList= list(set([allidsMap[item] for sublist in res for item in sublist]))

    #resultList = list(set([allidsMap[idMap[k]] for k,v in typeMap.items()  if len([vv for vv in v if vv.clsName in usageTypes]) > 0 ]))
    logging.info('*** Pruned idList of type %s from %d to %d', assetType, oldLength, len(resultList))
    return resultList
 
def createIssueIDList(idString, connections, assetType=['all']):
    # idString is assumed to be all, a comma delimited string
    # of issue ids or a comma delimited string of risk model group
    # mnemonics (two-letter ISO country codes)
    idStrings = [i.strip() for i in idString.split(',')]
    ids = set()
    mdlCur = connections.modelDB.dbCursor
    for idString in idStrings:
        if idString == 'all':
            logging.info('Load all issue IDs')
            mdlCur.execute("""SELECT issue_id FROM sub_issue si JOIN
                                     future_issue_map im ON im.modeldb_id=si.issue_id
                               UNION
                               SELECT issue_id FROM issue""")
            ids |= set([ModelID.ModelID(string=i[0])
                        for i in mdlCur.fetchall()])
        elif len(idString) == 10:
            ids.add(ModelID.ModelID(string=idString))
        elif len(idString) == 2:
            logging.info('Load all %s issue IDs' % idString)
            mdlCur.execute("""SELECT issue_id FROM sub_issue si JOIN
                risk_model_group rmg ON si.rmg_id=rmg.rmg_id
                JOIN future_issue_map im ON im.modeldb_id = si.issue_id 
                WHERE rmg.mnemonic=:rmg_mnemonic
                UNION
                SELECT issue_id FROM sub_issue si JOIN
                risk_model_group rmg ON si.rmg_id=rmg.rmg_id
                WHERE rmg.mnemonic=:rmg_mnemonic""", rmg_mnemonic=idString)
            ids |= set([ModelID.ModelID(string=i[0])
                        for i in mdlCur.fetchall()])
        else:
            logging.error('Incorrect format for Issue ID: %s. Skipping'
                          % idString)
    ids = pruneIDList(ids, connections, assetType)
    ids = sorted(ids)
    ranges = dict()
    INCR = 400
    argList = [('iid%d' % i) for i in range(INCR)]
    query = """SELECT issue_id, from_dt, thru_dt FROM issue
        WHERE issue_id in (%(args)s)""" % {
            'args': ','.join([':%s' % arg for arg in argList]) }
    defaultDict = dict([(arg, None) for arg in argList])
    idStrs = [i.getIDString() for i in ids]
    for idChunk in Utilities.listChunkIterator(idStrs, INCR):
        myDict = dict(defaultDict)
        myDict.update(dict(zip(argList, idChunk)))
        mdlCur.execute(query, myDict)

        for (iid, fromDt, thruDt) in mdlCur.fetchall():
            iid = ModelID.ModelID(string=iid)
            ranges[iid] = (fromDt.date(), thruDt.date())
    logging.info('Loaded %d issues', len(ids))
    ids = [elt for elt in ids if elt in ranges]
    return (ids, ranges)

def createCurrencyList(ccyString, connections):
    # ccyString is assumed to be all, a comma delimited string
    # of three letter currency ISO codes,
    # or a comma delimited string of risk model group
    # mnemonics (two-letter ISO country codes)
    ccyStrings = [i.strip() for i in ccyString.split(',')]
    ccyIDs = set()
    mdlCur = connections.modelDB.dbCursor
    for ccyString in ccyStrings:
        if ccyString == 'all':
            logging.info('Load all risk model currencies')
            mdlCur.execute("""SELECT currency_code FROM rmg_currency rc
              """)
              #JOIN rmg_model_map rm ON rc.rmg_id=rm.rmg_id""")
            ccyIDs |= set([i[0] for i in mdlCur.fetchall()])
        elif len(ccyString) == 3:
            ccyIDs.add(ccyString)
        elif len(ccyString) == 2:
            mdlCur.execute("""SELECT currency_code FROM rmg_currency c
            JOIN risk_model_group rmg ON c.rmg_id=rmg.rmg_id
            WHERE mnemonic=:rmg_mnemonic""", rmg_mnemonic=ccyString)
            ccyIDs |= set([i[0] for i in mdlCur.fetchall()])
        else:
            logging.error('Incorrect format for currency: %s. Skipping'
                          % ccyString)
    ccyIDs = sorted(ccyIDs)
    ccyInfo = list()
    for ccyID in ccyIDs:
        mdlCur.execute("""SELECT MIN(from_dt), MAX(thru_dt) FROM rmg_currency
           WHERE currency_code=:ccy GROUP BY currency_code""", ccy=ccyID)
        r = mdlCur.fetchall()
        assert(len(r) == 1)
        ccyInfo.append((ccyID, r[0][0].date(), r[0][1].date()))
    logging.info('Loaded %d currencies', len(ccyInfo))
    return ccyInfo

def createInstrumentList(connections):
    """Returns a list of the values from the currency_instrument_map table,
    as a list of tuples of (currency_code, instrument_name, from_dt, thru_dt)
    """
    mdlCur = connections.modelDB.dbCursor
    retval = []
    mdlCur.execute("""SELECT currency_code, instrument_name, from_dt, thru_dt
    FROM currency_instrument_map""")
    for r in mdlCur.fetchall():
        retval.append((r[0], r[1], r[2].date(), r[3].date()))
    return retval

def createRMGList(rmgString, connections):
    # rmgString is assumed to be all, a comma delimited string
    # of integers (risk model group IDs)
    # or a comma delimited string of risk model group
    # mnemonics (two-letter ISO country codes)
    rmgStrings = [i.strip() for i in rmgString.split(',')]
    rmgIDs = set()
    mdlCur = connections.modelDB.dbCursor
    for rmgString in rmgStrings:
        if rmgString == 'all':
            logging.info('Load all risk model groups')
            mdlCur.execute("""SELECT rmg_id FROM rmg_model_map
              WHERE rms_id > 0 AND rmg_id > 0""")
            rmgIDs |= set([int(i[0]) for i in mdlCur.fetchall()])
        elif rmgString.isdigit():
            rmgIDs.add(int(rmgString))
        elif len(rmgString) == 2:
            mdlCur.execute("""SELECT rmg_id FROM risk_model_group
            WHERE mnemonic=:rmg_mnemonic""", rmg_mnemonic=rmgString)
            rmgIDs |= set([i[0] for i in mdlCur.fetchall()])
        else:
            logging.error('Incorrect format for RMG: %s. Skipping'
                          % rmgString)
    rmgIDs = sorted(rmgIDs)
    logging.info('Loaded %d risk model group(s)', len(rmgIDs))
    return rmgIDs

def createAMPList(ampString, date, connections):
    # ampString is assumed to be a comma-delimited string
    # of AMP short names
    ampStrings = [i.strip() for i in ampString.split(',')]
    allAMPs = connections.modelDB.getAllModelPortfolioMembers(date)
    ampRmgMap = {}
    ampList = []
    for ampString in ampStrings:
        amps = [a for a in allAMPs if a.short_name == ampString]
        if len(amps) == 1:
            amp = amps[0]
            ampRmgMap[amp.id] = connections.modelDB.getModelPortfolioRmgMap(date, amp.id)
            if amp not in ampList:
                ampList.append(amp)
        elif len(amps) == 0:
            # get AMP from a more current date
            allAMPs = connections.modelDB.getAllModelPortfolioMembers(datetime.date.today())
            amps = [a for a in allAMPs if a.short_name == ampString]
            if len(amps) == 1:
                amp = amps[0]
                ampRmgMap[amp.id] = connections.modelDB.getModelPortfolioRmgMap(datetime.date.today(), amp.id)
                if amp not in ampList:
                    ampList.append(amp)
    return ampList, ampRmgMap

def createRegionList(regString, connections):
    # regString is assumed to be all, a comma delimited string
    # of integers (region IDs)
    # or a comma delimited string of region names
    regStrings = [i.strip() for i in regString.split(',')]
    regIDs = set()
    mdlCur = connections.modelDB.dbCursor
    for rs in regStrings:
        if rs == 'all':
            logging.info('Load all regions')
            mdlCur.execute("""SELECT id FROM region""")
            regIDs |= set([int(i[0]) for i in mdlCur.fetchall()])
        elif rs.isdigit():
            regIDs.add(int(rs))
        else:
            mdlCur.execute("""SELECT id FROM region
            WHERE name=:region_name""", region_name=rs)
            regIDs |= set([i[0] for i in mdlCur.fetchall()])
    regIDs = sorted(regIDs)
    logging.info('Loaded %d regions', len(regIDs))
    return regIDs

def createSubIssueIDList(idString, connections):
    # idString is assumed to be all, a comma delimited string
    # of issue ids or a comma delimited string of risk model group
    # mnemonics (two-letter ISO country codes)
    idStrings = [i.strip() for i in idString.split(',')]
    ids = set()
    mdlCur = connections.modelDB.dbCursor
    for idString in idStrings:
        if idString == 'all':
            logging.info('Load all sub-issue IDs')
            mdlCur.execute("""SELECT distinct sub_id FROM sub_issue where sub_id not like 'DCSH/_%' ESCAPE '/'""")
            ids |= set([ModelDB.SubIssue(string=i[0])
                        for i in mdlCur.fetchall()])
        elif idString[:5] == 'newer':
            revDt = Utilities.parseISODate(idString[6:])
            logging.info('Load all sub-issue IDs with updates'
                         ' on or after %s', revDt)
            mktCur = connections.marketDB.dbCursor
            tables = ['asset_dim_tso', 'asset_dim_ucp', 'asset_dim_tdv',
                      'asset_dim_return']
            changedAxids = set()
            for table in tables:
                mktCur.execute("""SELECT distinct axioma_id FROM %(table)s
                  WHERE rev_dt >= :revDt""" % {'table': table}, revDt=revDt)
                changedAxids |= set([i[0] for i in mktCur.fetchall()])
            changedSubIssues = mapAxiomaIdToSubIssue(list(changedAxids), mdlCur)
            ids |= set([ModelDB.SubIssue(string=i)
                        for i in changedSubIssues])
        elif idString == 'cash':
            logging.info('Load all cash sub-issue IDs')
            mdlCur.execute("""SELECT distinct sub_id FROM sub_issue
               WHERE issue_id like 'DCSH/_%' ESCAPE '/'""")
            ids |= set([ModelDB.SubIssue(string=i[0])
                        for i in mdlCur.fetchall()])
        elif len(idString) == 12:
            ids.add(ModelDB.SubIssue(string=idString))
        elif len(idString) == 2:
            logging.info('Load all %s sub-issue IDs' % idString)
            mdlCur.execute("""SELECT distinct sub_id FROM sub_issue si JOIN
            risk_model_group rmg ON si.rmg_id=rmg.rmg_id
            WHERE rmg.mnemonic=:rmg_mnemonic and si.sub_id not like 'DCSH/_%' ESCAPE '/'""", rmg_mnemonic=idString)
            ids |= set([ModelDB.SubIssue(string=i[0])
                        for i in mdlCur.fetchall()])
        else:
            logging.error('Incorrect format for Issue ID: %s. Skipping'
                          % idString)
    ids = list(ids)
    ranges = getSidRanges(ids, mdlCur)
    logging.info('Loaded %d sub-issues', len(ids))
    tmp = sorted((ranges[sid][2], sid) for sid in ids)
    ids = [i[1] for i in tmp]
    
    return (ids, ranges)

def getSidRanges(ids, mdlCur):
    ranges = dict()
    INCR = 400
    argList = [('iid%d' % i) for i in range(INCR)]
    query = """SELECT sub_id, from_dt, thru_dt, rmg_id FROM sub_issue
    WHERE sub_id in (%(args)s)""" % {
            'args': ','.join([':%s' % arg for arg in argList]) }
    defaultDict = dict([(arg, None) for arg in argList])
    idStrs = [i.getSubIDString() for i in ids]
    for idChunk in Utilities.listChunkIterator(idStrs, INCR):
        myDict = dict(defaultDict)
        myDict.update(dict(zip(argList, idChunk)))
        mdlCur.execute(query, myDict)
        for (iid, fromDt, thruDt, rmg_id) in mdlCur.fetchall():
            iid = ModelDB.SubIssue(string=iid)
            ranges[iid] = (fromDt.date(), thruDt.date(), rmg_id)
    logging.info('Loaded %d sub-issues', len(ids))
    return ranges

def mapAxiomaIdToIssue(axids, mdlCur):
    INCR = 100
    argList = ['axid%d' % i for i in range(INCR)]
    query = """SELECT modeldb_id FROM issue_map
      WHERE marketdb_id in (%(args)s)""" % {
        'args': ','.join([':%s' % arg for arg in argList])}
    futureQuery = """SELECT modeldb_id FROM future_issue_map
      WHERE marketdb_id in (%(args)s)""" % {
        'args': ','.join([':%s' % arg for arg in argList])}
    defaultDict = dict([(arg, None) for arg in argList])
    issues = set()
    futureids = [a for a in axids if a.startswith('F')]
    axids = [a for a in axids if not a.startswith('F')]
    for axidChunk in Utilities.listChunkIterator(axids, INCR):
        myDict = defaultDict.copy()
        myDict.update(dict(zip(argList, axidChunk)))
        mdlCur.execute(query, myDict)
        issues |= set([i[0] for i in mdlCur.fetchall()])
    for axidChunk in Utilities.listChunkIterator(futureids, INCR):
        myDict = defaultDict.copy()
        myDict.update(dict(zip(argList, axidChunk)))
        mdlCur.execute(futureQuery, myDict)
        issues |= set([i[0] for i in mdlCur.fetchall()])
    return issues

def mapAxiomaIdToSubIssue(axids, mdlCur):
    INCR = 100
    argList = ['axid%d' % i for i in range(INCR)]
    query = """SELECT distinct sub_id FROM sub_issue
      JOIN issue_map ON issue_id=modeldb_id
      WHERE marketdb_id in (%(args)s)""" % {
        'args': ','.join([':%s' % arg for arg in argList])}
    futureQuery = """SELECT distinct sub_id FROM sub_issue
      JOIN future_issue_map ON issue_id=modeldb_id
      WHERE marketdb_id in (%(args)s)""" % {
        'args': ','.join([':%s' % arg for arg in argList])}
    defaultDict = dict([(arg, None) for arg in argList])
    subIssues = set()
    futureids = [a for a in axids if a.startswith('F')]
    axids = [a for a in axids if not a.startswith('F')]
    for axidChunk in Utilities.listChunkIterator(axids, INCR):
        myDict = defaultDict.copy()
        myDict.update(dict(zip(argList, axidChunk)))
        mdlCur.execute(query, myDict)
        subIssues |= set([i[0] for i in mdlCur.fetchall()])
    for axidChunk in Utilities.listChunkIterator(futureids, INCR):
        myDict = defaultDict.copy()
        myDict.update(dict(zip(argList, axidChunk)))
        mdlCur.execute(futureQuery, myDict)
        subIssues |= set([i[0] for i in mdlCur.fetchall()])
    return subIssues


def createDateList(dateString):
    """
    calling RisModels.MarketDB.Utilities.createDateList(dateString)
    """
    return Utilities.createDateList(dateString)

def configureProcessorsAndLogic(config, section, connections, gPar=None):
    logging.info('in configureProcessorsAndLogic section %s %s',section, connections.modelDB.dbConnection.dsn)

    if not config.has_option(section, 'target'):
        logging.error('target option is missing in %s section' % section)
        return
    targetFields = config.get(section, 'target').split(':')
    targetType = targetFields[0]
    targetArgs = targetFields[1:]
    if gPar is None:
        targetProcessor = eval('%s(connections, *targetArgs)'% targetType)
    else:
        targetProcessor = eval('%s(connections, *targetArgs, gp=gPar)'% targetType)
    if not config.has_option(section, 'logic'):
        logging.error('logic option is missing in %s section' % section)
        raise KeyError
    businessLogic = eval('%s(connections, config, section)'
                         % config.get(section, 'logic'))
    if not config.has_option(section, 'sources'):
        logging.error('sources option is missing in %s section' % section)
        raise KeyError
    sourceProcessors = list()
    for sourceDef in config.get(section, 'sources').split(','):
        sourceDef = sourceDef.strip()
        logging.info('sourceDef: %s',sourceDef)
        sourceFields = sourceDef.split(':')
        sourceType = sourceFields[0]
        sourceArgs = sourceFields[1:]
        if gPar is None:
            proc = eval('%s(connections, *sourceArgs)' % sourceType)
        else:
            proc = eval('%s(connections, *sourceArgs, gp=gPar)' % sourceType)
        sourceProcessors.append(proc)
        
    return (targetProcessor, businessLogic, sourceProcessors)

def transferIssueData(config, section, connections, options):
    """Run the data transfer for the issue IDs and dates given
    by the 'issue-ids' and 'dates' options.
    The 'target' option defines which table the data will be inserted
    into. The format is 'table-type:table-name'.
    The 'sources' options lists the sources, most preferred first.
    Each source is defined as 'module-name.class-name'.
    """
    dateList = createDateList(config.get(section, 'dates'))
    ids=None
    axidList=None
    if config.has_option(section, 'asset-type'):
        assetType = config.get(section, 'asset-type').split(',')
    else:
        assetType=['all']
    if  config.has_option(section,'issue-ids'):
        ids=config.get(section,'issue-ids')
        # check to see if this is a special case of GICSCustom transfers for AU,CN,CA,JP,GB,NA
        if len(ids)==2:
            if section.find('GICSCustom')==0:
                ids=ids.upper()
                gicsctry=section[len('GICSCustom')+section.find('GICSCustom'):][:2].upper()
                #if (gicsctry in ['AU','CN','CA','JP','GB'] and gicsctry != ids) or (gicsctry=='NA' and not ids in ['US','CA']):
                if (gicsctry in ['AU','CN','CA','JP','GB','TW','US'] and gicsctry != ids):
                    logging.info('Can ignore this section %s for %s', section, ids)
                    return 
                if section=='GICSCustomGB4' and ids != 'GB':
                    logging.info('Can ignore this section %s for %s', section, ids)
                    return 
                if section=='GICSCustomCA4' and ids != 'CA':
                    logging.info('Can ignore this section %s for %s', section, ids)
                    return 
    if ids:
        (axidList, axidRanges) = createIssueIDList(config.get(
            section, 'issue-ids'), connections, assetType=assetType)
    else:
        if config.has_option(section,'axioma-ids'):
            axids=config.get(section,'axioma-ids')
            mdlCur = connections.modelDB.dbCursor
            issuelist=','.join(list(mapAxiomaIdToIssue(axids.split(','), mdlCur)))
            if issuelist:
                (axidList, axidRanges) = createIssueIDList(issuelist,connections, assetType=assetType)
    if not axidList:
        logging.info('No issue-ids or axioma-ids to process')
        return
    connections.axidRanges = axidRanges
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections)
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'id-chunk'):
        AINCR = int(config.get(section, 'id-chunk'))
    else:
        AINCR = 2000
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False
    for dates in Utilities.listChunkIterator(dateList, DINCR):
        logging.info('processing %s for %s:%s', section, dates[0], dates[-1])
        for axids in Utilities.listChunkIterator(axidList, AINCR):
            values = [proc.getBulkData(dates, axids)
                      for proc in sourceProcessors]
            values = numpy.array(values)
            #print values
            selectedValues = numpy.empty((len(dates),len(axids)), dtype=object)
            for (subIdx, date) in enumerate(dates):
                newValues = [businessLogic.process(
                    axids[aIdx], date, values[:,subIdx,aIdx])
                             for aIdx in range(len(axids))]
                newValues = numpy.array(newValues)
                selectedValues[subIdx] = newValues
                # Don't transfer any values outside the lifetime of the
                # Axioma ID
                for (aIdx, axid) in enumerate(axids):
                    (fromDt, thruDt) = axidRanges[axid]
                    if date < fromDt or date >= thruDt:
                        selectedValues[subIdx, aIdx] = None
            targetProcessor.bulkProcess(dates, axids, selectedValues)
        if commitEachChunk and not options.testOnly:
            logging.info('Committing changes')
            connections.modelDB.commitChanges()
    
def buildRMGTradingList(rmgIds, dateList, connections):
    """Given a set of risk model group IDs and a list of dates,
    returns a dictionary that maps each day to the set of rmg IDs
    that trade on that day.
    """
    modelDB = connections.modelDB
    if len(dateList) == 0:
        return dict()
    minDt = min(dateList)
    maxDt = max(dateList)
    tradingList = dict([(date, set()) for date in dateList])
    rmgList = modelDB.getAllRiskModelGroups(inModels=False)
    xuRMG = [r for r in rmgList if r.description=='United States Small Cap'][0]
    usRMG = [r for r in rmgList if r.description=='United States'][0]
    xcRMG = [r for r in rmgList if r.description=='Domestic China'][0]
    chinaRMG = [r for r in rmgList if r.description=='China'][0]
    for rmgId in rmgIds:
        rmg = Utilities.Struct()
        rmg.rmg_id = rmgId
        if rmgId == xcRMG.rmg_id:
            # For XC (Domestic China) use the CN trading calendar
            rmg.rmg_id = chinaRMG.rmg_id
        elif rmgId == xuRMG.rmg_id:
            rmg.rmg_id = usRMG.rmg_id
        tradingDays = modelDB.getDateRange([rmg], minDt, maxDt)
        for td in tradingDays:
            if td in tradingList:
                tradingList[td].add(rmgId)
    return tradingList
        
def transferRMGData(config, section, connections, options):
    """Run the data transfer for the risk model groups and dates given
    by the 'rmgs' and 'dates' options.
    Risk model groups are specified by their mnemonic or integer ID.
    The 'target' option defines which table the data will be inserted
    into. The format is 'table-type:table-name'.
    The 'sources' options lists the sources, most preferred first.
    Each source is defined as 'module-name.class-name'.
    """
    dateList = createDateList(config.get(section, 'dates'))
    rmgIds = createRMGList(config.get(section, 'rmgs'), connections)
    tradingRMGs = buildRMGTradingList(rmgIds, dateList, connections)
    gp = Utilities.Struct()
    gp.tradingRMGs = tradingRMGs
    gp.override = options.override
    gp.notInRiskModels = options.notInRiskModels
    gp.expand = options.expand
    gp.verbose = options.verbose
    gp.cleanup = options.cleanup
    gp.nuke = options.nuke
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections, gPar=gp)
    
    source = config.get(section, 'sources')
    tradingDaysOnly = False
    if config.has_option(section, 'trading-days-only'):
        val = config.get(section, 'trading-days-only')
        tradingDaysOnly = (val.lower().strip() == 'true')
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False
    if config.has_option(section, 'restrict-to-sub-issue-ids'):
        (restrictList, restrictRanges) = createSubIssueIDList(config.get(
                section, 'restrict-to-sub-issue-ids'), connections)
        restrictSet = set(restrictList)
        logging.info('Restricting changes to %d specified sub-issues',
                     len(restrictSet))
    else:
        restrictSet = None
    
    for dates in Utilities.listChunkIterator(dateList, DINCR):
        logging.info('processing %s for %s:%s', section, dates[0], dates[-1])
        values = [proc.getBulkData(dates, rmgIds)
                  for proc in sourceProcessors]
        values = numpy.array(values)
        selectedValues = numpy.empty((len(dates),len(rmgIds)), dtype=object)
        for (dIdx, date) in enumerate(dates):
            trading = tradingRMGs[date]
            newValues = [businessLogic.process(
                rmgIds[rIdx], date, values[:,dIdx,rIdx])
                         for rIdx in range(len(rmgIds))]
            newValues = numpy.array(newValues)
            selectedValues[dIdx] = newValues
            # Don't transfer any values on non-trading days
            for (rIdx, rmgId) in enumerate(rmgIds):
                if tradingDaysOnly and rmgId not in trading:
                    selectedValues[dIdx, rIdx] = None
                elif selectedValues[dIdx, rIdx] is not None:
                    selectedValues[dIdx, rIdx].rmg_id = rmgId
        if not options.dontWrite:
            targetProcessor.bulkProcess(dates, rmgIds, selectedValues, restrictSet)
            if commitEachChunk and not options.testOnly:
                logging.info('Committing changes for %s step for %d RMGS and dates %s to %s',
                        source, len(rmgIds), dates[0], dates[-1])
                connections.modelDB.commitChanges()

def transferAMPIndustryReturnData(config, section, connections, options):
    """Run the data transfer for the AMPs and dates given
    by the 'amps', 'dates', and 'classification' options.
    """
    # get dateList and ampList
    dateList = createDateList(config.get(section, 'dates'))
    ampList, ampRmgMap = createAMPList(config.get(section, 'amps'), 
            max(dateList), connections)

    # get rmgs associated with amps in ampList 
    rmgIds = []
    for ampId, ampRmgDict in ampRmgMap.items():
        for rmg, wgt in ampRmgDict.items():
            if rmg.rmg_id not in rmgIds:
                rmgIds.append(rmg.rmg_id)
    tradingRMGs = buildRMGTradingList(rmgIds, dateList, connections)
  
    # get industry classification 
    gicsCls = config.get(section, 'classification')
    (clsName, clsDateStr) = gicsCls.split(':')
    clsDate = Utilities.parseISODate(clsDateStr)
    gicsCls_ = getattr(Classification, clsName)
    gicsCls = gicsCls_(clsDate)
    clsFamily = connections.modelDB.getMdlClassificationFamily('INDUSTRIES')
    clsMembers = connections.modelDB.getMdlClassificationFamilyMembers(clsFamily)
    clsMember = [i for i in clsMembers if i.name==gicsCls.name][0]
    clsRevision = connections.modelDB.getMdlClassificationMemberRevision(
        clsMember, clsDate)
    industryList = []
    industryIdDict = {}
    for f in gicsCls.getLeafNodes(connections.modelDB).values():
        industryList.append(f.description)
        industryIdDict[f.description] = f.id

    gp = Utilities.Struct()
    gp.gicsCls = gicsCls  # gics classificaiton
    gp.revision_id = clsRevision.id # gics classificaiton revision id
    gp.tradingRMGs = tradingRMGs 
    gp.ampRmgMap = ampRmgMap 
    gp.industryList = industryList 
    gp.industryIdDict = industryIdDict # ref id
    gp.useRmgMarketPortfolio = True # set to True to always use RMG Market Portfolio instead of AMP
    gp.override = options.override
    gp.notInRiskModels = options.notInRiskModels
    gp.expand = options.expand
    gp.verbose = options.verbose
    gp.cleanup = options.cleanup
    gp.nuke = options.nuke
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections, gPar=gp)
    
    source = config.get(section, 'sources')
    tradingDaysOnly = False
    if config.has_option(section, 'trading-days-only'):
        val = config.get(section, 'trading-days-only')
        tradingDaysOnly = (val.lower().strip() == 'true')
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False

    restrictSet = None
    
    for dates in Utilities.listChunkIterator(dateList, DINCR):
        for amp in ampList:
            logging.info('processing %s for %s:%s', section, dates[0], dates[-1])
            selectedValues = numpy.empty((len(dates),len(industryList)), dtype=object)
            for (dIdx, date) in enumerate(dates):
                trading = tradingRMGs[date]
                if not set([r.rmg_id for r in ampRmgMap[amp.id].keys()]).issubset(trading):
                    continue
                # compute returns if at least one rmg associated with amp is trading
                values = [proc.getBulkData(amp, date)
                          for proc in sourceProcessors][0]
                selectedValues[dIdx, :] = values

            if not options.dontWrite:
                targetProcessor.bulkProcess(amp, dates, industryList, selectedValues)
                if commitEachChunk and not options.testOnly:
                    logging.info('Committing changes for %s step for %d RMGS and dates %s to %s',
                            source, len(rmgIds), dates[0], dates[-1])
                    connections.modelDB.commitChanges()

def transferRegionData(config, section, connections, options):
    """Run the data transfer for the regions and dates given
    by the 'regs' and 'dates' options.
    Regions are specified by their name or integer ID.
    The 'target' option defines which table the data will be inserted
    into. The format is 'table-type:table-name'.
    The 'sources' options lists the sources, most preferred first.
    Each source is defined as 'module-name.class-name'.
    """
    dateList = createDateList(config.get(section, 'dates'))
    regIds = createRegionList(config.get(section, 'regs'), connections)
    mdlCur = connections.modelDB.dbCursor
    mdlCur.execute("""SELECT rmg_id FROM rmg_model_map
                    WHERE rms_id > 0 AND rmg_id > 0""")
    rmgIds = set([int(i[0]) for i in mdlCur.fetchall()])
    tradingRMGs = buildRMGTradingList(rmgIds, dateList, connections)
    gp = Utilities.Struct()
    gp.tradingRMGs = tradingRMGs
    gp.override = options.override
    gp.verbose = options.verbose
    gp.cleanup = options.cleanup
    gp.nuke = options.nuke
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections, gPar=gp)

    tradingDaysOnly = False
    if config.has_option(section, 'trading-days-only'):
        val = config.get(section, 'trading-days-only')
        tradingDaysOnly = (val.lower().strip() == 'true')
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False
    restrictSet = None

    for dates in Utilities.listChunkIterator(dateList, DINCR):
        logging.info('processing %s for %s:%s', section, dates[0], dates[-1])
        values = [proc.getBulkData(dates, regIds)
                  for proc in sourceProcessors]
        values = numpy.array(values)
        selectedValues = numpy.empty((len(dates),len(regIds)), dtype=object)
        for (dIdx, date) in enumerate(dates):
            trading = tradingRMGs[date]
            newValues = [businessLogic.process(
                regIds[rIdx], date, values[:,dIdx,rIdx])
                         for rIdx in range(len(regIds))]
            newValues = numpy.array(newValues)
            selectedValues[dIdx] = newValues
            # Don't transfer any values on non-trading days
            if len(trading) == 0:
                for rIdx in range(len(regIds)):
                    selectedValues[dIdx, rIdx] = None
        if not options.dontWrite:
            targetProcessor.bulkProcess(dates, regIds, selectedValues, restrictSet)
            if commitEachChunk and not options.testOnly:
                logging.info('Committing changes')
                connections.modelDB.commitChanges()

def transferCurrencies(config, section, connections, options):
    """Run the data transfer for the currencies and dates given
    by the 'currencies' and 'dates' options.
    Currencies are specified by their ISO code.
    The 'target' option defines which table the data will be inserted
    into. The format is 'table-type:table-name'.
    The 'sources' options lists the sources, most preferred first.
    Each source is defined as 'module-name.class-name'.
    """
    dateList = createDateList(config.get(section, 'dates'))
    currencyISOs = createCurrencyList(config.get(section, 'currencies'),
                                      connections)
    connections.instruments = createInstrumentList(connections)
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections)
    
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False
    
    for dates in Utilities.listChunkIterator(dateList, DINCR):
        logging.info('processing %s for %s:%s', section, dates[0], dates[-1])
        values = [proc.getBulkData(dates, currencyISOs)
                  for proc in sourceProcessors]
        values = numpy.array(values)
        selectedValues = numpy.empty((len(dates),len(currencyISOs)),
                                     dtype=object)
        #print values
        for (dIdx, date) in enumerate(dates):
            newValues = [businessLogic.process(
                currencyISOs[cIdx], date, values[:,dIdx,cIdx])
                         for cIdx in range(len(currencyISOs))]
            newValues = numpy.array(newValues)
            selectedValues[dIdx] = newValues
        targetProcessor.bulkProcess(dates, [c[0] for c in currencyISOs],
                                    selectedValues)
        if commitEachChunk and not options.testOnly:
            connections.modelDB.commitChanges()
  
def transferDescriptorData(config, section, connections, options):
    """Builds the descriptor data that underlies all 2016-onwards fundamental risk models
    """
    ids=None
    sidList=None
    allRMGList = connections.modelDB.getAllRiskModelGroups(inModels=True)

    # Define global parameters
    gp = Utilities.Struct()
    gp.cleanup = options.cleanup
    gp.override = options.override
    connections.modelDB.forceRun = options.override
    gp.nuke = options.nuke
    gp.verbose = options.verbose
    gp.trackList = []
    gp.tradedAssetsOnly = options.tradedAssetsOnly
    gp.gicsDate = datetime.date(2016,9,1)
    rmgMnemonicIDMap = dict([(r.mnemonic, r.rmg_id) for r in allRMGList])
    rmgIDMnemonicMap = dict([(r.rmg_id, r.mnemonic) for r in allRMGList])
    gp.useFixedFrequency = None
    if config.has_option(section, 'fixed-freq'):
        gp.useFixedFrequency = config.get(section, 'fixed-freq')

    # Sort out various command line options
    dateList = createDateList(config.get(section, 'dates'))
    ids=config.get(section, 'rmgs')
    if config.has_option(section, 'rmg-list'):
        allowedRMGs = config.get(section, 'rmg-list')
        if ids == 'all':
            ids = allowedRMGs
        else:
            allowedRMGs = allowedRMGs.split(',')
            ids = ids.split(',')
            idsString = [idx for idx in ids if idx in allowedRMGs]
            ids = ','.join(idsString)
    exclude_rmg = []
    if config.has_option(section, 'exclude-rmg-list'):
        exclude_list = config.get(section, 'exclude-rmg-list').split(',')
        exclude_rmg = [connections.modelDB.getRiskModelGroupByISO(r) for r in exclude_list]

    (sidList, sidRanges) = createSubIssueIDList(ids, connections)

    # Get list of assets to be tracked for debugging
    if options.trackID is not None:
        trackList = options.trackID.split(',')
        allSubIDs = connections.modelDB.getAllActiveSubIssues(dateList[-1])
        sidStringMap = dict([(sid.getSubIDString(), sid) for sid in allSubIDs])
        gp.trackList = [sidStringMap[ss] for ss in trackList if ss in sidStringMap]

    # weed out based on asset type by calling prune earlier
    if config.has_option(section, 'asset-type'):
        # restrict to equities or futures or mutual funds
        assetType = config.get(section, 'asset-type')
        sidList = pruneIDList(sidList, connections, assetType.split(','))
        mdlCur = connections.modelDB.dbCursor
        sidDict = dict((s,1) for s in sidList)
        for s in list(sidRanges.keys()):
            if s not in sidDict:
                sidRanges.pop(s)

    logging.info('** Loaded %d sub-issues', len(sidList))


    # If we're updating only a handful of assets, get the subissues
    if config.has_option(section, 'gics-date'):
        gp.gicsDate = Utilities.parseISODate(config.get(section, 'gics-date'))
    if config.has_option(section,'sub-issue-ids'):
        ids=config.get(section,'sub-issue-ids')
        (updateSidList, updateSidRanges) = createSubIssueIDList(ids, connections)
    else:
        updateSidList = None

    if config.has_option(section,'numeraire'):
        numeraire = config.get(section, 'numeraire')
        hasNumeraire = True
    else:
        logging.info('No numeraire specified - will use default')
        hasNumeraire = False

    if sidList is None or (len(sidList) < 1):
        logging.info('No sub-issue-ids or axioma-ids to process')
        return

    # Deal with list of items to be computed
    itemList = [i.strip() for i in config.get(section, 'items').split(',')]
    itemList = list(set(itemList))
    itemList.sort()
    if config.has_option(section, 'shortlist'):
        shortList = config.get(section, 'shortlist').split(',')
        itemList = [item for item in itemList if item in shortList]
    descDict = dict(connections.modelDB.getAllDescriptors())
    itemList = [item for item in itemList if item in descDict]
    gp.descIDList = ['ds_%d' % descDict[item] for item in itemList]
    if len(gp.descIDList) < 1:
        logging.warning('No valid descriptors specified, bailing....')
        return

    gp.sidRanges = sidRanges
    (targetProcessor, businessLogic, sourceProcessors) = \
        configureProcessorsAndLogic(config, section, connections, gPar=gp)

    if config.has_option(section, 'id-chunk'):
        AINCR = int(config.get(section, 'id-chunk'))
    else:
        AINCR = 2000
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False

    # Initialise Descriptor Class
    dsCls = sourceProcessors[0]
    # Define numeraire if necessary
    if not hasNumeraire:
        numeraire = 'USD'
    assert(len(numeraire) == 3)
    dsCls.setNumeraire(numeraire)
    dsCls.forceRun = True

    # Optionally set returnHistory for base descriptor class
    if config.has_option(section, 'returnHistory'):
        retHist = int(config.get(section, 'returnHistory'))
        dsCls.setReturnHistory(retHist)

    # Set up GICS date
    if hasattr(dsCls, 'descriptorType'):
        descriptorType = dsCls.descriptorType
    else:
        descriptorType = 'SCM'
    logging.info('Descriptor Type: %s, GICS Date: %s', descriptorType, dsCls.gicsDate)

    for date in dateList:
        logging.info('Processing descriptor data for %s', date.isoformat())
        if date.isoweekday() > 5:
            logging.info('Date is a %s, skipping', date.strftime('%A'))
            continue

        allSubIssues = list()
        rmg_id_List = []
        rmgSidMap = dict()
        assetCurrPairs = dict()

        # Screen out those invalid SubIssues
        for sid in sidList:
            if sid in sidRanges:

                # Get RMG and asset dates
                (fromDt, thruDt, rmg_id) = sidRanges[sid]
                rmg_id_List.append(rmg_id)
                if rmg_id not in rmgSidMap:
                    rmgSidMap[rmg_id] = []

                if date >= thruDt or date < fromDt:
                    # Those without valid lifespan will be ruled out
                    continue
                else:
                    allSubIssues.append(sid)
                    rmgSidMap[rmg_id].append(sid)
        rmg_id_List = list(set(rmg_id_List))

        # Report on initial list of subissues
        logging.info('Using %d subissues on %s', len(allSubIssues), date)

        # Process lists of active sub-issues
        logging.info('Processing assets by type')
        if (descriptorType == 'local'):
            allActiveSubIssues = connections.modelDB.getAllActiveSubIssues(date)
            allActiveSubIssues = pruneAssets(connections, date, allActiveSubIssues, descriptorType)
            allSubIssues = sorted(set(allSubIssues).intersection(set(allActiveSubIssues)))
            logging.info('Main asset universe consists of %d assets', len(allSubIssues))
        else:
            allSubIssues = pruneAssets(connections, date, allSubIssues, descriptorType)

        if (descriptorType != 'local') and len(allSubIssues) < 1:
            logging.warning('No valid subissues for %s', date)
            continue

        # Set up date history for local descriptor types
        dsCls.setDateHistory(descriptorType, date)
        dsCls.setNumeraireID(dsCls.numeraire_ISO, date)

        # Loop round RMGs
        for rmg_id in rmg_id_List:

            # Set up RMG
            rmg = connections.modelDB.getRiskModelGroup(rmg_id)
            if rmg in exclude_rmg:
                logging.info('Skipping descriptors for RMG %s', rmg.mnemonic)
                continue
            rmg.setRMGInfoForDate(date)
            logging.info('Processing data for RMG: %s', rmg.mnemonic)

            # Set up local currency information
            latestTrdDate = connections.modelDB.getDates([rmg], date, 1)
            if len(latestTrdDate) < 1:
                logging.warning('No trade date found for RMG %s, date %s', rmg.mnemonic, date)
            elif latestTrdDate[-1] < date:
                logging.warning('Most recent trading date is %s', latestTrdDate[-1])
            localCurrencyMap = dict([(d['from_dt'], d['currency_code']) \
                    for d in rmg.timeVariantDicts if d['from_dt']<=date])
            if len(localCurrencyMap) < 1:
                logging.error('No currencies associated with %s on or before %s',
                        rmg.mnemonic, date)
                assert(len(localCurrencyMap)>0)
            dsCls.setLocalCurrency(localCurrencyMap, date)
            tradingCurrency = dsCls.localCurrencyISO[dsCls.latestDate]
            logging.info('Using numeraire: %s, trading currency: %s', numeraire, tradingCurrency)

            # Get initial universe
            univ = list(set(rmgSidMap[rmg_id]).intersection(set(allSubIssues)))
            tableName = 'descriptor_exposure_%s' % numeraire
            allModelRMSIDs = []

            # Remove assets from the main group whose home and trading markets differ but
            # have the same currencies (e.g. Euro)
            removeList = []
            if descriptorType == 'local':

                # Load primary and secondary home markets
                homeMarket1 = AssetProcessor.get_home_country(
                        univ, date, connections.modelDB, connections.marketDB, clsType='HomeCountry')
                homeMarket2 = AssetProcessor.get_home_country(
                        univ, date, connections.modelDB, connections.marketDB, clsType='HomeCountry2')

                # Get list of assets whose trading country and home market differ
                removeList1 = [sid for sid in homeMarket1.keys() if (homeMarket1[sid].classification.code != rmg.mnemonic)]
                removeList2 = [sid for sid in homeMarket2.keys() if (homeMarket1[sid].classification.code != rmg.mnemonic) \
                        and (homeMarket2.get(sid, None) is not None) and (homeMarket2[sid].classification.code != rmg.mnemonic)]

                for sid in removeList1 + removeList2:
                    # Get currency of primary home market
                    homeMarketCode = homeMarket1[sid].classification.code
                    if homeMarketCode in rmgMnemonicIDMap:
                        homeRMG = connections.modelDB.getRiskModelGroupByISO(homeMarketCode)
                        homeIsoCode = homeRMG.getCurrencyCode(date)
                        # Primary home and trading market differ...
                        if homeIsoCode == tradingCurrency:
                            # ... but currencies are the same, so remove from list
                            removeList.append(sid)
                            logging.info('Dropping asset %s from main list (home: %s, currency: %s)',
                                    sid.getSubIDString(), homeRMG.mnemonic, homeIsoCode)

                    if homeMarket2.get(sid, None) is not None:
                        # Compare secondary home market now (if it exists)
                        homeMarket2Code = homeMarket2[sid].classification.code
                        if homeMarket2Code in rmgMnemonicIDMap:
                            home2RMG = connections.modelDB.getRiskModelGroupByISO(homeMarket2Code)
                            home2IsoCode = home2RMG.getCurrencyCode(date)
                            if home2IsoCode == tradingCurrency:
                                # If secondary home market currency is the same as the trading currency, then drop
                                removeList.append(sid)
                                logging.info('Dropping asset %s from main list (home2: %s, currency: %s)',
                                sid.getSubIDString(), home2RMG.mnemonic, home2IsoCode)

                # Remove unwanted assets
                removeList = set(removeList)
                if len(removeList) > 0:
                    logging.info('Dropping %d %s-traded assets with different home/trading markets but identical currency',
                            len(removeList), rmg.mnemonic)
                    univ = list(set(univ).difference(removeList))

            # Build asset data
            assetData = AssetProcessor.process_asset_information(
                    date, univ, [rmg], connections.modelDB, connections.marketDB,
                    checkHomeCountry=False, numeraire_id=dsCls.numeraire_id,
                    forceRun=True, legacyDates=False)

            # Determine type of descriptor we're updating
            if descriptorType == 'numeraire':
                tableName = 'descriptor_numeraire_%s' % numeraire

            elif (descriptorType == 'local'):
                tableName = 'descriptor_local_currency'
                allModelRMSIDs = connections.modelDB.getRMSIDsByModelType()

                # If we're doing descriptors based on trading-currencies, load in extra assets
                # with home country assignment to current RMG
                logging.info('Loading additional asset universe... *********************************')
                extraSubIssues = set(allActiveSubIssues).difference(set(assetData.universe))
                if len(removeList) > 0:
                    extraSubIssues = list(extraSubIssues.difference(removeList))

                # Load primary and secondary home markets
                homeMarket1 = AssetProcessor.get_home_country(
                        extraSubIssues, date, connections.modelDB, connections.marketDB, clsType='HomeCountry')
                homeMarket2 = AssetProcessor.get_home_country(
                        extraSubIssues, date, connections.modelDB, connections.marketDB, clsType='HomeCountry2')

                # Get list of assets with secondary home market
                home2List = [sid for sid in homeMarket2.keys() if (homeMarket2.get(sid, None) is not None) and
                        (homeMarket2[sid].classification.code == rmg.mnemonic) and
                        (homeMarket2[sid].classification.code != homeMarket1[sid].classification.code)]

                removeList = []
                for sid in home2List:
                    # Check whether current trading RMG is primary or secondary home market
                    homeMarketCode = homeMarket1[sid].classification.code
                    if (homeMarketCode != rmg.mnemonic) and (homeMarketCode in rmgMnemonicIDMap):
                        homeRMG = connections.modelDB.getRiskModelGroupByISO(homeMarketCode)
                        homeIsoCode = homeRMG.getCurrencyCode(date)
                        homeMarket2Code = homeMarket2[sid].classification.code
                        # Get secondary home currency
                        if homeMarket2Code in rmgMnemonicIDMap:
                            home2RMG = connections.modelDB.getRiskModelGroupByISO(homeMarket2Code)
                            home2IsoCode = home2RMG.getCurrencyCode(date)

                            # Primary and secondary home markets differ...
                            if homeIsoCode == home2IsoCode:
                                # ... but currencies are the same, so drop from secondary home market
                                removeList.append(sid)
                                logging.info('Excluding asset %s from secondary home market %s, as currency %s same as primary home %s',
                                        sid.getSubIDString(), home2RMG.mnemonic, homeIsoCode, homeRMG.mnemonic)

                # Remove all assets mapped to secondary home country where the primary home has the same currency
                removeList = set(removeList)
                if len(removeList) > 0:
                    logging.info('Dropping %d non-%s-traded assets whose primary and secondary home markets have same currency',
                            len(removeList), rmg.mnemonic)
                    extraSubIssues = list(set(extraSubIssues).difference(removeList))

                rmgAssetMap = AssetProcessor.loadRiskModelGroupAssetMap(
                        date, extraSubIssues, allRMGList, connections.modelDB, connections.marketDB,
                        False, quiet=True)

                # If doing a preliminary run, exclude assets trading after our particular region
                if options.tradedAssetsOnly:
                    # Sort regions into three super-regions:
                    # 1. Americas (11 and 12)
                    # 2. Europe, ME and Africa (13, 16, 17)
                    # 3. Asia (14, 15)
                    regionOrder = {11:0, 12:0, 13:1, 14:2, 15:2, 16:1, 17:1}
                    extraSubIssues = []
                    curRegionPosition = regionOrder[rmg.region_id]
                    for extraRMG in allRMGList:
                        if extraRMG.rmg_id in rmgAssetMap:
                            if regionOrder[extraRMG.region_id] >= curRegionPosition:
                                extraSubIssues.extend(rmgAssetMap[extraRMG.rmg_id])
                                logging.info('Adding %d assets from %s', len(rmgAssetMap[extraRMG.rmg_id]), extraRMG.mnemonic)
                            else:
                                logging.info('Excluding assets from %s', extraRMG.mnemonic)

                # Process the list to find those with home market equal to the current market
                extraAssetData = AssetProcessor.process_asset_information(
                        date, extraSubIssues, [rmg], connections.modelDB, connections.marketDB,
                        checkHomeCountry=True, numeraire_id=dsCls.numeraire_id,
                        forceRun=True, legacyDates=False)

                logging.info('Finished loading additional asset universe... ************************')

                if len(extraAssetData.universe) > 0:
                    logging.info('Adding %d assets to RMG: %s that are traded elsewhere',
                            len(extraAssetData.universe), rmg.mnemonic)

                    # Append extra assets to universe
                    assetData.universe.extend(extraAssetData.universe)
                    logging.info('%s universe is now %d assets', rmg.mnemonic, len(assetData.universe))

                    # Rebuild some of the asset maps
                    assetData.marketCaps = numpy.concatenate((\
                            assetData.marketCaps, extraAssetData.marketCaps), axis=0)
                    if hasattr(assetData, 'monthlyADV'):
                        assetData.monthlyADV = numpy.concatenate((\
                                assetData.monthlyADV, extraAssetData.monthlyADV), axis=0)
                    assetData.hardCloneMap.update(extraAssetData.hardCloneMap)
                    assetData.forceCointMap.update(extraAssetData.forceCointMap)
                    assetData.noCointMap.update(extraAssetData.noCointMap)
                    assetData.dr2Underlying.update(extraAssetData.dr2Underlying)
                    assetData.sidToCIDMap.update(extraAssetData.sidToCIDMap)
                    assetData.subIssueGroups = connections.modelDB.getIssueCompanyGroups(
                            date, assetData.universe, connections.marketDB)
                    assetData.assetIdxMap = dict(zip(assetData.universe, range(len(assetData.universe))))
                    assetData.rmgAssetMap[rmg_id] = assetData.universe
                    assetData.rmcAssetMap = copy.deepcopy(assetData.rmgAssetMap)
                    for (tmp_rmg_id, rmgSidList) in extraAssetData.tradingRmgAssetMap.items():
                        if tmp_rmg_id in assetData.tradingRmgAssetMap:
                            assetData.tradingRmgAssetMap[tmp_rmg_id].extend(extraAssetData.tradingRmgAssetMap[tmp_rmg_id])
                        else:
                            assetData.tradingRmgAssetMap[tmp_rmg_id] = list(extraAssetData.tradingRmgAssetMap[tmp_rmg_id])
                    assetData.assetTypeDict.update(extraAssetData.assetTypeDict)
                    assetData.marketTypeDict.update(extraAssetData.marketTypeDict)
                    assetData.assetISINMap.update(extraAssetData.assetISINMap)
                    assetData.assetNameMap.update(extraAssetData.assetNameMap)
                    assetData.freeFloat_marketCaps = numpy.concatenate((\
                            assetData.freeFloat_marketCaps, extraAssetData.freeFloat_marketCaps), axis=0)

            if len(assetData.universe) < 1:
                logging.warning('No valid subissues for %s', date)
                continue

            # Get dict of trading and current market currencies per asset
            assetTradingRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                    assetData.tradingRmgAssetMap.items() for sid in ids])
            for sid in assetData.universe:
                tradingRMG = connections.modelDB.getRiskModelGroup(assetTradingRMGMap[sid])
                isoCode = tradingRMG.getCurrencyCode(date)
                if sid in assetCurrPairs:
                    assetCurrPairs[sid].append([rmg, tradingCurrency, tradingRMG.mnemonic, isoCode, date])
                else:
                    assetCurrPairs[sid] = [[rmg, tradingCurrency, tradingRMG.mnemonic, isoCode, date]]

            # Set some basic data items
            assetData.sidRanges = getSidRanges(assetData.universe, connections.modelDB.dbCursor)
            if updateSidList is not None:
                updateRMGSidList = [sid for sid in updateSidList if sid in assetData.universe]
            else:
                updateRMGSidList = list(assetData.universe)
            if len(updateRMGSidList) < 1:
                logging.warning('No valid subissues found, skipping...')
                continue
            dsCls.buildBasicData(date, assetData, rmg, allRMGList)
            outputArray = Matrices.allMasked((len(updateRMGSidList), len(itemList)))
            descriptorExposures = Matrices.ExposureMatrix(updateRMGSidList)

            # Some housekeeping and debugging output
            if gp.verbose:
                cls = Classification.GICSIndustries(gp.gicsDate)
                industryList = [f.description for f in cls.getLeafNodes(connections.modelDB).values()]
                exposures = cls.getExposures(date, updateRMGSidList, industryList, connections.modelDB)
                descriptorExposures.addFactors(industryList, exposures, ExposureMatrix.IndustryFactor)

            if gp.nuke:
                targetProcessor.nuke(tableName)
            
            if dsCls.cleanup:
                if descriptorType == 'SCM':
                    targetProcessor.deleteData(tableName, updateRMGSidList, date, None)
                elif descriptorType == 'numeraire':
                    targetProcessor.deleteData(tableName, updateRMGSidList, date, numeraire)
                elif (descriptorType == 'local'):
                    targetProcessor.deleteData(tableName, updateRMGSidList, date, tradingCurrency)

            # Loop round descriptors
            for (itemDim, item) in enumerate(itemList):
                if hasattr(DescriptorSources, item):
                    descriptorInstance = eval('DescriptorSources.%s(connections, gp=gp)' % item)
                    logging.info('Constructing %s/%s for %s:%s, rmg: %s, numeraire: %s, trading currency: %s',
                                section, item, date, date, rmg.mnemonic, numeraire, tradingCurrency)
                else:
                    logging.info('%s not found in DescriptorSources module ', item)
                    descriptorInstance = eval('ResearchDescriptorSources.%s(connections, gp=gp)' % item)
                    logging.info('Constructing research %s/%s for %s:%s, rmg: %s, numeraire: %s, trading currency: %s',
                            section, item, date, date, rmg.mnemonic, numeraire, tradingCurrency)

                # Create the particular descriptor
                retVal = descriptorInstance.buildDescriptor(assetData, dsCls)
                for (iDim, sid) in enumerate(updateRMGSidList):
                    targetIdx = assetData.assetIdxMap[sid]
                    if retVal[0, targetIdx] is not None:
                        outputArray[iDim, itemDim] = retVal[0, targetIdx].value
                descriptorExposures.addFactor(item, outputArray[:,itemDim], ExposureMatrix.StyleFactor)

            if gp.verbose:
                if (descriptorType == 'local'):
                    descriptorExposures.dumpToFile('tmp/Descriptors-%s-%s-%s.csv' % \
                            (rmg.mnemonic, tradingCurrency, date),
                            connections.modelDB, connections.marketDB, date,
                            assetType=assetData.assetTypeDict, dp=8, assetData=assetData)
                else:
                    descriptorExposures.dumpToFile('tmp/Descriptors-%s-%s-%s.csv' % \
                            (rmg.mnemonic, numeraire, date),
                            connections.modelDB, connections.marketDB, date,
                            assetType=assetData.assetTypeDict, dp=8, assetData=assetData)

            if not options.dontWrite:
                outputArray = rmUtils.screen_data(outputArray)
                if descriptorType == 'SCM':
                    targetProcessor.bulkProcess(
                            date, updateRMGSidList, outputArray, tableName, None)
                elif descriptorType == 'numeraire':
                    targetProcessor.bulkProcess(
                            date, updateRMGSidList, outputArray, tableName, numeraire)
                elif (descriptorType == 'local'):
                    targetProcessor.bulkProcess(
                            date, updateRMGSidList, outputArray, tableName, tradingCurrency)
    
                if commitEachChunk and not options.testOnly:
                    logging.info('Committing changes')
                    connections.modelDB.commitChanges()

        # Report on possible duplicate entries - where an asset has different markets but the same currency
        import collections
        from itertools import chain
        for (sid, listings) in assetCurrPairs.items():
            listings = assetCurrPairs[sid]
            if len(listings)>1:
                currList = [lst[1] for lst in listings]
                dupCurr = [itm for itm, cnt in collections.Counter(currList).items() if cnt>1]
                dupItems = [(lst[0].mnemonic, lst[1], lst[2], lst[3]) for lst in listings if lst[1] in dupCurr]
                if len(dupItems) > 0:
                    dupItems = list(chain.from_iterable(dupItems))
                    logging.warning('Duplicates: foreign assets with identical home and trading currencies, %s, %s',
                            sid.getSubIDString(), ','.join(dupItems))

        if len(rmg_id_List) == 1:
            thisRmg = connections.modelDB.getRiskModelGroup(rmg_id_List[0])
            logging.info('Finished processing %s descriptor data for RMG: %s on %s', section, thisRmg.mnemonic, date.isoformat())
        else:
            logging.info('Finished processing %s descriptor data for %d RMGs on %s', section, len(rmg_id_List), date.isoformat())

def pruneAssets(connections, date, subIssues, descriptorType):
    """Remove certain unwanted asset types from the list of assets used by the desciptor code
    """
    # Exclude some undesirables in advance
    # To do: replace this with call to exclude table on marketDB
    excludeTypes = AssetProcessor.noTransferList
    assetTypeDict = AssetProcessor.get_asset_info(date, subIssues,
                connections.modelDB, connections.marketDB, 'ASSET TYPES', 'Axioma Asset Type')
    excludeFields = [ic.lower() for ic in excludeTypes]
    excludeStocks = set([sid for sid in subIssues \
            if (sid in assetTypeDict) and assetTypeDict[sid].lower() in excludeFields])
    if len(excludeStocks) > 0:
        logging.info('%d stocks marked for exclusion by type: %s', len(excludeStocks), ','.join(excludeTypes))
        subIssues = list(set(subIssues).difference(excludeStocks))
        logging.info('Asset universe now %d assets', len(subIssues))

    # Report on number of SPACs if they are pre-announcement date
    # Note: we don't yet have announcment date, so we use ex-date for now
    preSPAC = AssetProcessor_V4.sort_spac_assets(date, subIssues, connections.modelDB, connections.marketDB)
    if len(preSPAC) > 0:
        logging.info('%d SPACs marked for special treatment', len(preSPAC))

    # Exclude issues with asset type of None
    excludeStocks = set([sid for sid in subIssues if (sid not in assetTypeDict) or \
            (sid in assetTypeDict and assetTypeDict[sid] is None)])
    if len(excludeStocks) > 0:
        logging.info('%d stocks marked for exclusion with no Asset type', len(excludeStocks))
        logging.debug('Excluded assets: %s', excludeStocks)
        subIssues = list(set(subIssues).difference(excludeStocks))
        logging.info('Asset universe now %d assets', len(subIssues))

    # Exclude issues with an unrecognised asset type
    allAssetTypes = [ic.lower() for ic in AssetProcessor.allAssetTypes]
    excludeStocks = set([sid for sid in subIssues if (sid in assetTypeDict) and \
            assetTypeDict[sid].lower() not in allAssetTypes])
    if len(excludeStocks) > 0:
        newTypes = set([assetTypeDict[sid] for sid in excludeStocks])
        logging.info('%d stocks marked for exclusion with unrecognised asset type(s): %s',
                len(excludeStocks), ','.join(newTypes))
        logging.debug('Excluded assets: %s', excludeStocks)
        subIssues = list(set(subIssues).difference(excludeStocks))
        logging.info('Asset universe now %d assets', len(subIssues))

    # Get rid of open-ended funds
    # get all subissues and then send it over to get market classifications for the specific revision
    marketDB=connections.marketDB
    clsFamily = marketDB.getClassificationFamily('USAGES')
    clsMember = marketDB.getClassificationFamilyMembers(clsFamily)
    thisMember = [mem for mem in clsMember if mem.name=='Axioma Usages'][0]
    dt=datetime.date.today()
    thisRevision = marketDB.getClassificationMemberRevision(thisMember, dt)
    usages=connections.modelDB.getMktAssetClassifications(thisRevision, subIssues, date, marketDB)
    newList = [k for k,v in usages.items() if v.classification and v.classification.code in ( 'MOD','CONST', 'STATMOD', 'EXMOD') ]
    logging.info('Keeping only MOD,CONST,STATMOD,EXMOD  usages took list from %d to %d', len(subIssues), len(newList))
    subIssues = newList

    ###oeFunds = set(connections.modelDB.getIssuesByMarketDBType('O%'))
    ###if len(oeFunds) > 0:
    ###    logging.info('%d open-ended funds excluded', len(oeFunds))
    ###    subIssues = list(set(subIssues).difference(oeFunds))
    ###    logging.info('Asset universe now %d assets', len(subIssues))

    # Get rid of cash assets
    cashIDs = set([sid for sid in subIssues if sid.isCashAsset()])
    if len(cashIDs) > 0:
        logging.info('%d cash assets  excluded', len(cashIDs))
        subIssues = list(set(subIssues).difference(cashIDs))
        logging.info('Asset universe now %d assets', len(subIssues))

    subIssues.sort()
    return subIssues

'''TODO: Document me. N.B. new transfer estimate data '''
def transferEstimateData(config, section, connections, options):
    return transferData(config, section, connections, options, Utilities.DATA_TYPE.ESTIMATE)

def transferFundamentalData(config, section, connections, options):
    return transferData(config, section, connections, options, Utilities.DATA_TYPE.FUNDAMENTAL)
    
def transferXpressfeedFundamentalData(config, section, connections, options):
    return transferData(config, section, connections, options, Utilities.DATA_TYPE.XPRESSFEED_FUNDAMENTAL)

'''N.B. new transfer all (generic) data '''
#def transferFundamentalData(config, section, connections, options):
def transferData(config, section, connections, options, dataType):
    """Run the data transfer for the sub-issue IDs and effective dates given
    by the 'sub-issue-ids' and 'dates' options.
    The 'target' option defines which table the data will be inserted
    into.
    The 'sources' options lists the sources, most preferred first.
    Each source is defined as 'module-name.class-name'.
    'items' contains a comma-separated list of item names to transfer.
    The transfer is run in two steps: first the sources are asked to provide
    a list of asset/effective-date pairs for which they have updates.
    The second step then asks all sources to provide their values for those
    pairs and calls the processing logic and target processor to update
    the database accordingly.
    """
    dateList = createDateList(config.get(section, 'dates'))
    itemList = [i.strip() for i in config.get(section, 'items').split(',')]
    ids=None
    sidList=None
    if  config.has_option(section,'sub-issue-ids'):
        ids=config.get(section,'sub-issue-ids')
    if ids:
        (sidList, sidRanges) = createSubIssueIDList(config.get(
            section, 'sub-issue-ids'), connections)
    else:
        if config.has_option(section,'axioma-ids'):
            axids=config.get(section,'axioma-ids')
            mdlCur = connections.modelDB.dbCursor
            subissuelist=','.join(list(mapAxiomaIdToSubIssue(axids.split(','), mdlCur)))
            if subissuelist:
                (sidList, sidRanges) = createSubIssueIDList(subissuelist,connections)
    if config.has_option(section, 'asset-type'):
        # restrict to equities or futures or mutual funds
        mdlCur = connections.modelDB.dbCursor 
        assetType = config.get(section, 'asset-type')
        sidList = pruneIDList(sidList, connections, assetType.split(','))
        sidDict = dict((s,1) for s in sidList)
        for s in list(sidRanges.keys()):
            if s not in sidDict:
                sidRanges.pop(s)
        sidList=newsidList

    logging.info('** Loaded %d sub-issues', len(sidList))
    if not sidList:
        logging.info('No sub-issue-ids or axioma-ids to process')
        return
    connections.sidRanges = sidRanges
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections)
    
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'id-chunk'):
        AINCR = int(config.get(section, 'id-chunk'))
    else:
        AINCR = 2000
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False
    
    for dates in Utilities.listChunkIterator(dateList, DINCR):
        for item in itemList:
            logging.info('processing %s/%s for %s:%s', section, item, dates[0], dates[-1])
            #dateIdxMap = dict([(j,i) for (i,j) in enumerate(dates)])
            for sids in Utilities.listChunkIterator(sidList, AINCR):
                updatesFound = set()
                for proc in sourceProcessors:
                    updatesFound.update(proc.findUpdates(dates, sids, item))
                updatesFound = sorted(updatesFound)
                values = [proc.getBulkData(updatesFound, item) for proc in sourceProcessors]
                values = numpy.array(values)
                selectedValues = numpy.empty((len(updatesFound),), dtype=object)
                for (tIdx, (sid, dt, effDt)) in enumerate(updatesFound):
                    newValue = businessLogic.process(sid, dt, effDt, values[:,tIdx])
                    selectedValues[tIdx] = newValue
                                 
                    # Don't transfer any values after the thru_dt of the Sub-Issue ID
                    # TODO: N.B. This is not True for EstimateData: Do not ignore estimate values under these condition
                    if dataType != Utilities.DATA_TYPE.ESTIMATE:
                        (fromDt, thruDt, rmg_id) = sidRanges[sid]
                        if dt >= thruDt:
                            if selectedValues[tIdx] is not None:
                                logging.debug('Ignoring value for %s on %s: after asset thru_dt', sid, dt)
                                selectedValues[tIdx] = None
                
                #print selectedValues
                targetProcessor.bulkProcess(updatesFound, selectedValues, item)
            if commitEachChunk and not options.testOnly:
                logging.info('Committing changes')
                connections.modelDB.commitChanges()

def transferSubIssueData(config, section, connections, options):
    """Run the data transfer for the sub-issue IDs and dates given
    by the 'sub-issue-ids' and 'dates' options.
    The 'target' option defines which table the data will be inserted
    into. The format is 'table-type:table-name'.
    The 'sources' options lists the sources, most preferred first.
    Each source is defined as 'module-name.class-name'.
    """
    dateList = createDateList(config.get(section, 'dates'))
    ids=None
    sidList=None
    if  config.has_option(section,'sub-issue-ids'):
        ids=config.get(section,'sub-issue-ids')
    if ids:
        (sidList, sidRanges) = createSubIssueIDList(config.get(
            section, 'sub-issue-ids'), connections)
    else:
        if config.has_option(section,'axioma-ids'):
            axids=config.get(section,'axioma-ids')
            mdlCur = connections.modelDB.dbCursor
            subissuelist=','.join(list(mapAxiomaIdToSubIssue(axids.split(','), mdlCur)))
            if subissuelist:
                (sidList, sidRanges) = createSubIssueIDList(subissuelist,connections)

    if section in ['SubIssueData', 'SubIssueReturn', 'SubIssueDivYield']:
        # Exclude all the CASH sub issue ids
        cashsids = [sid for sid in sidList if sid.isCashAsset()]
        for cashsid in cashsids:
            logging.warning("Removing %s from list of sub ids to transfer for %s" % (cashsid.getSubIDString(), section))
            del sidRanges[cashsid]
        sidList = list(set(sidList) - set(cashsids))

    if config.has_option(section, 'asset-type'):
        # restrict to equities or futures
        mdlCur = connections.modelDB.dbCursor 
        assetType = config.get(section, 'asset-type')
        sidList = pruneIDList(sidList, connections, assetType.split(','))
        sidDict = dict((s,1) for s in sidList)
        for s in list(sidRanges.keys()):
            if s not in sidDict:
                sidRanges.pop(s)
    logging.info('** Loaded %d sub-issues', len(sidList))

    if len(sidList) == 0:
        sidList = None

    if not sidList:
        logging.info('No sub-issue-ids or axioma-ids to process')
        return

    tradingDaysOnly = False
    if config.has_option(section, 'trading-days-only'):
        val = config.get(section, 'trading-days-only')
        tradingDaysOnly = (val.lower().strip() == 'true')
    rmgIds = set([i[2] for i in sidRanges.values()])
    tradingRMGs = buildRMGTradingList(rmgIds, dateList, connections)
    connections.tradingRMGs = tradingRMGs
    connections.sidRanges = sidRanges
    (targetProcessor, businessLogic, sourceProcessors) = \
                    configureProcessorsAndLogic(config, section, connections)
    if config.has_option(section, 'date-chunk'):
        DINCR = int(config.get(section, 'date-chunk'))
    else:
        DINCR = 30
    if config.has_option(section, 'id-chunk'):
        AINCR = int(config.get(section, 'id-chunk'))
    else:
        AINCR = 2000
    if config.has_option(section, 'commit-each-chunk'):
        val = config.get(section, 'commit-each-chunk')
        commitEachChunk = (val.lower().strip() == 'true')
    else:
        commitEachChunk = False

    for dates in Utilities.listChunkIterator(dateList, DINCR):
        logging.info('processing %s for %s:%s', section, dates[0], dates[-1])
        for sids in Utilities.listChunkIterator(sidList, AINCR):
            values = [proc.getBulkData(dates, sids)
                      for proc in sourceProcessors]
            values = numpy.array(values)
            selectedValues = numpy.empty((len(dates),len(sids)), dtype=object)
            for (dIdx, date) in enumerate(dates):
                if tradingDaysOnly:
                    trading = tradingRMGs[date]
                newValues = [businessLogic.process(
                    sids[aIdx], date, values[:,dIdx,aIdx])
                             for aIdx in range(len(sids))]
                newValues = numpy.array(newValues)
                selectedValues[dIdx] = newValues
                # Don't transfer any values outside the lifetime of the
                # Axioma ID
                for (aIdx, sid) in enumerate(sids):
                    (fromDt, thruDt, rmg_id) = sidRanges[sid]
                    if (tradingDaysOnly and rmg_id not in trading) \
                           or date < fromDt or date >= thruDt:
                        selectedValues[dIdx, aIdx] = None
                    elif selectedValues[dIdx, aIdx] is not None:
                        selectedValues[dIdx, aIdx].rmg_id = rmg_id
            targetProcessor.bulkProcess(dates, sids, selectedValues)
        if commitEachChunk and not options.testOnly:
            logging.info('Committing changes')
            connections.modelDB.commitChanges()

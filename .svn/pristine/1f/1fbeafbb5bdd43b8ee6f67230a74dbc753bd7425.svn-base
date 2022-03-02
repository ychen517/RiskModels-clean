
import configparser
import datetime
import logging
import optparse
from riskmodels import ModelID
from riskmodels import Connections
from riskmodels import Utilities

def getRMGHistory(mdlDB):
    cur = mdlDB.dbCursor
    cur.execute("""SELECT si.issue_id, rmg.mnemonic, si.from_dt, si.thru_dt
       FROM sub_issue si JOIN risk_model_group rmg ON si.rmg_id=rmg.rmg_id
       WHERE issue_id NOT LIKE 'DCSH_%'""")
    rmgHistory = dict()
    startDate = datetime.date(1990,1,1)
    for (issue, rmgID, fromDt, thruDt) in cur.fetchall():
        rmgHistory.setdefault(ModelID.ModelID(string=issue), list()).append(
            (max(fromDt.date(), startDate), thruDt.date(), rmgID))
    for (issue, rmgList) in rmgHistory.items():
        rmgHistory[issue] = sorted(rmgList)
    return rmgHistory

def getQuotationHistory(mdlDB, mktDB, issues):
    qtFamily = mktDB.getClassificationFamily("REGIONS")
    qtMembers = mktDB.getClassificationFamilyMembers(qtFamily)
    qtMember = [m for m in qtMembers if m.name=='Market'][0]
    qtRevision = mktDB.getClassificationMemberRevision(
        qtMember, datetime.date.today())
    mdlDB.getMktAssetClassifications(
        qtRevision, issues, datetime.date.today(), mktDB, level=1)
    clsDict = mdlDB.marketClassificationCaches[qtRevision.id].classificationDict
    qtHist = mdlDB.marketClassificationCaches[qtRevision.id].historyDict
    future = datetime.date(2999, 12, 31)
    # Convert to from/thru history
    for (issue, issueQtHist) in qtHist.items():
        qtIsoHist = list()
        for (p1, p2) in zip(issueQtHist, issueQtHist[1:] + [(future, 'Y')]):
            if p1[1] == 'N':
                fromDt = p1[0]
                thruDt = p2[0]
                cls = clsDict[p1[2].classification_id]
                if cls.level == 1:
                    qtIso = cls.code
                else:
                    qtIso = cls.levelParent[1].code
                qtIsoHist.append((fromDt, thruDt, qtIso))
        qtHist[issue] = qtIsoHist
    # Merge history where possible
    for (issue, issueQtHist) in qtHist.items():
        if len(issueQtHist) == 0:
            continue
        mergedHist = [issueQtHist[0]]
        for p in issueQtHist[1:]:
            if p[2] == mergedHist[-1][2]:
                mergedHist[-1] = (mergedHist[-1][0], p[1], p[2])
            else:
                mergedHist.append(p)
        qtHist[issue] = mergedHist
    return qtHist

def getAxiomaIDMappings(mdlDB):
    mdlDB.dbCursor.execute("""SELECT modeldb_id, marketdb_id, from_dt, thru_dt
       FROM issue_map""")
    axiomaIDMap = dict()
    for (mdlID, mktID, fromDt, thruDt) in mdlDB.dbCursor.fetchall():
        axiomaIDMap.setdefault(mdlID, list()).append((fromDt, thruDt, mktID))
    return axiomaIDMap

def compareHistories(issue, rmgHist, qtHist, axiomaIDMappings):
    issueStr = issue.getIDString()
    if issueStr not in axiomaIDMappings:
        logging.error('No issue_map entries for %s', issueStr)
        return
    axiomaIDs = ','.join([i[2] for i in axiomaIDMappings[issueStr]])
    if len(qtHist) == 0:
        logging.warning('No country of quotation history for %s/%s',
                     issueStr, axiomaIDs)
        return
    # Check for missing qt history at beginning and end
    if qtHist[0][0] > rmgHist[0][0]:
        logging.warning('No country of quotation for %s/%s from %s to %s',
                     issueStr, axiomaIDs, rmgHist[0][0], qtHist[0][0])
    if qtHist[-1][1] < rmgHist[-1][1]:
        logging.warning('No country of quotation for %s/%s from %s to %s',
                     issueStr, axiomaIDs, qtHist[-1][1], rmgHist[-1][1])
    # Check for gaps in qt history
    for p1, p2 in zip(qtHist[:-1], qtHist[1:]):
        if p1[1] < p2[0]:
            logging.warning('Gap in country of quotation coverage for %s/%s'
                         ' from %s to %s', issueStr, axiomaIDs,
                         p1[1], p2[0])
    # Check for gaps in rmg history
    for p1, p2 in zip(rmgHist[:-1], rmgHist[1:]):
        if p1[1] < p2[0]:
            logging.warning('Gap in RMG coverage for %s/%s'
                         ' from %s to %s', issueStr, axiomaIDs,
                         p1[1], p2[0])
    # Now check that rmg and qt history agree where both exist
    for rmgFrom, rmgThru, rmgISO in rmgHist:
        for qtFrom, qtThru, qtISO in qtHist:
            if rmgFrom <= qtThru and qtFrom <= rmgThru and rmgISO != qtISO:
                logging.warning('Country of quotation (%s) differs from'
                             ' risk model group (%s) for %s/%s from %s to %s',
                             qtISO, rmgISO, issueStr, axiomaIDs,
                             max(qtFrom, rmgFrom), min(qtThru, rmgThru))

def main():
    usage = "usage: %prog [options] config-file"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 1:
        cmdlineParser.error("Incorrect number of arguments")
    configFile = open(args[0])
    config = configparser.ConfigParser()
    config.read_file(configFile)
    configFile.close()
    connections = Connections.createConnections(config)
    mdlDB = connections.modelDB
    mktDB = connections.marketDB
    rmgHistory = getRMGHistory(mdlDB)
    qtHistory = getQuotationHistory(mdlDB, mktDB,list(rmgHistory.keys()))
    axiomaIDMap = getAxiomaIDMappings(mdlDB)
    for (issue, issueRMGHist) in rmgHistory.items():
        if issue in qtHistory:
            compareHistories(issue, issueRMGHist, qtHistory[issue],
                             axiomaIDMap)
        else:
            logging.warning('No country of quotation information for %s',
                         issue.getIDString())
    mdlDB.finalize()
    mktDB.finalize()

if __name__ == '__main__':
    main()

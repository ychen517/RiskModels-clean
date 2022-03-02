
import datetime
import logging
import optparse
import sys
import numpy
import numpy.ma as ma
import os
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import Utilities
from riskmodels.Matrices import ExposureMatrix

if __name__ == '__main__':
    usage = "usage: %prog [options] <sub-factor-id> <YYYY-MM-DD> [<YYYY-MM-DD>]"
    cmdlineParser = optparse.OptionParser(usage=usage)
    Utilities.addModelAndDefaultCommandLine(cmdlineParser)
    cmdlineParser.add_option("-n", action="store_true", default=False, dest="testOnly", help="don't change the database")
    cmdlineParser.add_option("--ret", action="store_true", default=False, dest="returns", help="output asset returns")
    cmdlineParser.add_option("--mtm", action="store_true", default=False, dest="mtm", help="output exposures")
    cmdlineParser.add_option("--stm", action="store_true", default=False, dest="stm", help="output exposures")
    cmdlineParser.add_option("--siz", action="store_true", default=False, dest="siz", help="output exposures")
    cmdlineParser.add_option("--vol", action="store_true", default=False, dest="vol", help="output exposures")
    cmdlineParser.add_option("--liq", action="store_true", default=False, dest="liq", help="output exposures")
    cmdlineParser.add_option("--xrt", action="store_true", default=False, dest="xrt", help="output exposures")
    cmdlineParser.add_option("--val", action="store_true", default=False, dest="val", help="output exposures")
    cmdlineParser.add_option("--lev", action="store_true", default=False, dest="lev", help="output exposures")
    cmdlineParser.add_option("--gro", action="store_true", default=False, dest="gro", help="output exposures")
    cmdlineParser.add_option("--bms", action="store_true", default=False, dest="bms", help="create bms")
    cmdlineParser.add_option("--name", action="store_true", default=False, dest="names", help="get list of names")
    cmdlineParser.add_option("--idfile", action="store_true", default=False, dest="idFile", help="input ID list")
    (options, args) = cmdlineParser.parse_args()
    if len(args) < 2 or len(args) > 3:
        cmdlineParser.error("Incorrect number of arguments")
    riskModelClass = Utilities.processModelAndDefaultCommandLine(
        options, cmdlineParser)

    # Set things up
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(
        sid=options.marketDBSID, user=options.marketDBUser,
        passwd=options.marketDBPasswd)
    riskModel = riskModelClass(modelDB, marketDB)

    # Sort dates out
    startDate = Utilities.parseISODate(args[1])
    if len(args) == 1:
        endDate = startDate
    else:
        endDate = Utilities.parseISODate(args[2])
    startDate = modelDB.getDateRange(riskModel.rmg, startDate, endDate, True)[0]
    dates = modelDB.getDateRange(modelDB.getAllRiskModelGroups(), startDate, endDate, True)

    status = 0
    if not options.idFile:
        axiomaID = str(args[0]).split(',')
        nameDict = dict()
    else:
        axiomaID = []
        nameList = []
        infile = open(str(args[0]), 'r')
        for inline in infile:
            fields = inline.split(',')
            if len(fields[0]) > 0:
                axiomaID.append(str(fields[0]))
                nameList.append(str(fields[1]))
        infile.close()
        nameDict = dict(zip(axiomaID, nameList))
    prevDate = None
    p_rmi = None

    if options.names:
        nameList = [''] * len(axiomaID)
    for d in dates:
        riskModel.setFactorsForDate(d, modelDB)
        rmi = riskModel.getRiskModelInstance(d, modelDB)
        if rmi == None:
            rmi = p_rmi
        univ = modelDB.getRiskModelInstanceUniverse(rmi)
        subIDList = [s.getSubIDString() for s in univ]
        assetIdxMap = dict([(j,i) for (i,j) in enumerate(subIDList)])
        if options.names:
            logging.info('Processing: %s' % d)
            for (i,a) in enumerate(axiomaID):
                if a in assetIdxMap:
                    loc = assetIdxMap[a]
                    sid = univ[loc]
                    name = modelDB.getIssueNames(d, [sid], marketDB)
                    nameList[i] = name[sid]
        elif options.returns:
            # Output asset returns
            try:
                axiomaIDSub = [a for a in axiomaID if a in assetIdxMap]
                loc = [assetIdxMap[aID] for aID in axiomaIDSub]
                subID = [univ[l] for l in loc]
                numer = riskModel.numeraire.currency_id
                returns = modelDB.loadTotalReturnsHistoryV3(
                        riskModel.rmg, d, subID, 1, assetConvMap=numer)
                tradeFlag = returns.notTradedFlag[:,0]
                returns = Utilities.screen_data(returns.data[:,0])
                name = modelDB.getIssueNames(d, subID, marketDB)
                name = [name[sID].replace(',','') for sID in subID]
            except:
                returns = [0]*len(axiomaID)
                mcap = [0]*len(axiomaID)
                name = ['None']*len(axiomaID)
                tradeFlag = ['None']*len(axiomaID)
            outStr = ''
            jj = 0
            for ii in range(len(axiomaID)):
                if axiomaID[ii] in axiomaIDSub:
                    outStr += '%s,%s,%12.8f,%s,' % (axiomaID[ii], name[jj], returns[jj], tradeFlag[jj])
                    jj += 1
                else:
                    outStr += '%s,X,0.0,X,' % axiomaID[ii]
            logging.info(',%s,%s', d, outStr)

        elif options.bms:
            logging.info('Processing: %s' % d)
            (mcapDates, goodRatio) = riskModel.getRMDates(d, modelDB, 20, ceiling=False)
            mcaps = modelDB.getAverageMarketCaps(
                    mcapDates, univ, riskModel.numeraire.currency_id)
            (dum1, dum2, dum3) = riskModel.process_asset_country_assignments(
                    d, univ, mcaps, modelDB, marketDB)
            assetRMGMap = dict([(sid, rmg_id) for (rmg_id, ids) in \
                    riskModel.rmgAssetMap.items() for sid in ids])
            subIssueGroups = modelDB.getIssueCompanyGroups(d, univ, marketDB)
            if len(subIssueGroups) > 0:
                scores = riskModel.score_linked_assets(d, univ, modelDB, marketDB)
                plist = []
                bmlist = []
                dt = str(d).replace('-','')
                for (groupId, subIssueList) in subIssueGroups.items():
                    score = scores[groupId]
                    indSort = numpy.argsort(-score)
                    if len(subIssueList) > 1:
                        mainID = subIssueList[indSort[0]]
                        for idx in indSort[1:]:
                            nextID = subIssueList[idx]
                            if assetRMGMap[mainID] == assetRMGMap[nextID]:
                                plist.append(mainID)
                                bmlist.append(nextID)
                                break
                if len(bmlist) > 0:
                    wt = 1.0 / len(bmlist)
                    if not os.path.exists('ADR-%s' % riskModel.mnemonic[2:4]):
                        os.mkdir('ADR-%s' % riskModel.mnemonic[2:4])
                    outfile1 = 'ADR-%s/ADR-%s-%s.csv' % (riskModel.mnemonic[2:4], riskModel.mnemonic[2:4], dt)
                    outfile2 = 'ADR-%s/ADR-BM-%s-%s.csv' % (riskModel.mnemonic[2:4], riskModel.mnemonic[2:4], dt)
                    ofile1 = open(outfile1, 'w')
                    ofile2 = open(outfile2, 'w')
                    for (pid, bid) in zip(plist, bmlist):
                        ofile1.write('%s, %s\n' % (pid.getSubIDString()[1:-2], wt))
                        ofile2.write('%s, %s\n' % (bid.getSubIDString()[1:-2], wt))
                    ofile1.close()
                    ofile2.close()
        else:
            if options.mtm:
                style = 'Medium-Term Momentum'
            elif options.stm:
                style = 'Short-Term Momentum'
            elif options.siz:
                style = 'Size'
            elif options.vol:
                style = 'Volatility'
            elif options.liq:
                style = 'Liquidity'
            elif options.xrt:
                style = 'Exchange Rate Sensitivity'
            elif options.gro:
                style = 'Growth'
            elif options.val:
                style = 'Value'
            elif options.lev:
                style = 'Leverage'
            try:
                axiomaIDSub = [a for a in axiomaID if a in assetIdxMap]
                loc = [assetIdxMap[aID] for aID in axiomaIDSub]
                subID = [univ[l] for l in loc]
                name = modelDB.getIssueNames(d, subID, marketDB)
                name = [name[sID].replace(',','') for sID in subID]
                expM = riskModel.loadExposureMatrix(rmi, modelDB)
                styleFactorIdx = expM.getFactorIndex(style)
                styleExp = expM.getMatrix()[styleFactorIdx]
                styleExp = ma.filled(ma.take(styleExp, loc, axis=0), 0.0)
            except:
                axiomaIDSub = []
            outStr = ''
            jj = 0
            for ii in range(len(axiomaID)):
                if axiomaID[ii] in axiomaIDSub:
                    outStr += '%s,%s,%12.8f,' % (axiomaID[ii], name[jj], styleExp[jj])
                    jj += 1
                else:
                    outStr += '%s,XXX,0.0,' % axiomaID[ii]
            logging.info(',%s,%s,%s', d, style, outStr)
        p_rmi = rmi

    if options.names:
        outfile = 'adrList.csv'
        ofile = open(outfile, 'w')
        evens = False
        for (i,sid) in enumerate(axiomaID):
            nm = nameList[i]
            if evens:
                ofile.write('%s,%s-ADR,\n' % (sid, nm))
                evens = False
            else:
                ofile.write('%s,%s,\n' % (sid, nm))
                evens = True
        ofile.close()
    sys.exit(0)

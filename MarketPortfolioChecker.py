
import logging
import optparse
import sys
import configparser
from operator import itemgetter
import numpy
import traceback

from riskmodels import Connections
from riskmodels import Utilities
from riskmodels import TransferSources
from riskmodels import transfer

class MarketPortfolioChecker:
    """Generates the QA reports using the MarketPortfolio
    """
    def __init__(self, configfile, date, country):
        configFile_ = open(configfile)
        config_ = configparser.ConfigParser()
        config_.read_file(configFile_)
        configFile_.close()
        self.connections = Connections.createConnections(config_)
        self.country = country
        self.tradingRMGsList = transfer.createRMGList(country, self.connections)
        self.rmgId = self.tradingRMGsList[0]
        self.subIdWtPairs = {}
        self.subIdMCapPairs = {}
        self.subIdReturns = {}
        self.date = date
        self.prevdt = self.getPreviousTradingDay(self.date)
        gp = Utilities.Struct()
        gp.tradingRMGs = transfer.buildRMGTradingList(self.tradingRMGsList, 
                                                                    [self.date, self.prevdt], self.connections)
        self.rmgMktPortfolio = TransferSources.RMGMarketPortfolio(self.connections, gp)
        self.rmg = self.rmgMktPortfolio.rmgMap[self.rmgId]
        self.populateAllData(self.date)
        
    def getMarketPortfolio(self, dt):
        """ Create the market portfolio for dt
        """
        mp = self.rmgMktPortfolio.getBulkData([dt], [self.rmgId])
        return mp[0, 0].valuePairs

    def getMPMktCapAndReturns(self, dt):
        """ Create market cap and returns for the assets in the
        market portfolio
        """
        if self.subIdWtPairs.get(dt, None) is None:
            self.subIdWtPairs[dt] = self.getMarketPortfolio(dt)
        subIssues, weights = zip(*self.subIdWtPairs[dt])
        if self.rmg.description == 'Domestic China':
            rmgList = self.connections.modelDB.getAllRiskModelGroups()
            chinaRMG = [r for r in rmgList if r.description=='China'][0]
            self.rmg.rmg_id = chinaRMG.rmg_id
        currencyCode = self.rmg.getCurrencyCode(dt)
        currencyID = self.connections.modelDB.currencyCache.getCurrencyID(currencyCode, dt)
        if self.subIdReturns.get(dt, None) is None:
            returns = self.connections.modelDB.loadTotalReturnsHistory(
                [self.rmg], dt, subIssues, 0, assetConvMap=currencyID)
            self.subIdReturns[dt] = returns

        if self.subIdMCapPairs.get(dt, None) is None:
            mktCaps = self.connections.modelDB.loadMarketCapsHistory([dt], subIssues, currencyID)
            subIdMCapPairs = []
            for i in range(0, len(subIssues)):
                if not mktCaps[i].mask:            
                    subIdMCapPairs.append((subIssues[i], mktCaps[i].data[0]))
                else:
                    subIdMCapPairs.append((subIssues[i], 0.0))
            subIdMCapPairsSorted = sorted(subIdMCapPairs, key=itemgetter(1), reverse=True)
            self.subIdMCapPairs[dt] = subIdMCapPairsSorted
    
    def calculateMarketReturn(self, dt):
        subIssues, weights = zip(*self.subIdWtPairs[dt])
        weights = numpy.array(weights, dtype=float)
        ret = numpy.dot(self.subIdReturns[dt].data.filled(0.0)[:, 0], weights)
        ret = '%.4lf%%' % (ret*100.0)
        return ret

    def getPreviousTradingDay(self, dt):
        """ Get the previous trading date to dt for the rmg id
        """
        # For XC use the trading calendar of CN
        relatedRMGId = self.rmgId
        if self.rmgId == -2:
            relatedRMGId = 12
        query = """select max(dt) from rmg_calendar where rmg_id=:rmgid_arg and
        dt < :dt_arg"""
        self.connections.modelDB.dbCursor.execute(query, rmgid_arg=relatedRMGId, dt_arg=dt)
        r = self.connections.modelDB.dbCursor.fetchall()

        if len(r) == 1 and r[0][0]:
            return r[0][0].date()

        return None

    def populateAllData(self, dt):
        """ Populate market portfolio, market cap and returns
        """
        if self.subIdMCapPairs.get(dt, None) is None:
            self.getMPMktCapAndReturns(dt)
        self.prevdt = self.getPreviousTradingDay(dt)
        if self.subIdMCapPairs.get(self.prevdt, None) is None:
            self.getMPMktCapAndReturns(self.prevdt)

    def getTop10(self, dt):
        """ Get top 10 assets by weight and return
        a list of tuples with
        modeldb id, rank diff and weight
        """
        top10 = []
        self.populateAllData(dt)
        mcapdt = self.subIdMCapPairs[dt]
        mcapprevdt = self.subIdMCapPairs[self.prevdt]
        nconsts = min(len(mcapdt), 10)
        for i in range(nconsts):
            subid = mcapdt[i][0]
            mdlmktmap = self.getMarketIDs([subid.getModelID()], dt)
            (mktid, status) = mdlmktmap.get(subid.getModelID(), (None, 'Unknown'))
            rankdiff = 9999
            for j in range(len(mcapprevdt)):
                (sid, wt) = mcapprevdt[j] 
                if sid == subid:
                    rankdiff = j - i
                    break
            sidwt = -1.0
            for (sid, wt) in self.subIdWtPairs[dt]:
                if sid == subid:
                    sidwt = wt
                    break

            if sidwt < 0:
                logging.error("%s not found in portfolio for %s" % (subid, dt))
                
            top10.append((mktid.getIDString(), mcapdt[i][0].getSubIDString()[0:10], sidwt, rankdiff))
        return top10

    def getConstituentDiffs(self, dt):
        """ Get all the joiners and leavers with previous rank
        """
        diffs = []
        self.populateAllData(dt)
        dtConsts = [i[0] for i in self.subIdWtPairs[dt]]
        prevDtConsts = [i[0] for i in self.subIdWtPairs[self.prevdt]]
        joiners = set(dtConsts) - set(prevDtConsts)
        leavers = set(prevDtConsts) - set(dtConsts)

        allmdlids = [i.getModelID() for i in joiners] + [i.getModelID() for i in leavers]
        mdlmktmap = self.getMarketIDs(allmdlids, dt)
        for joiner in joiners:
            (mktid, status) = mdlmktmap.get(joiner.getModelID(), (None, 'Unknown'))
            diffs.append(("Joiner", mktid.getIDString(), joiner.getSubIDString()[0:10], '-', status, 'N'))
        for leaver in leavers:
            mcapprevdt = self.subIdMCapPairs[self.prevdt]
            rank = -9999
            for j in range(len(mcapprevdt)):
                (sid, wt) = mcapprevdt[j] 
                if sid == leaver:
                    rank = j
                    break
            (mktid, status) = mdlmktmap.get(leaver.getModelID(), (None, 'Unknown'))            
            bottomPer = (len(prevDtConsts) - float(rank))/len(prevDtConsts)
            if bottomPer >= 0.05:
                TO_QA = 'Y' # Flag if the leaver is above the bottom 5%
            else:
                TO_QA = 'N'
            diffs.append(("Leaver", mktid.getIDString(), leaver.getSubIDString()[0:10], rank, status, TO_QA))
        return diffs
    
    def getLargeWtDiffs(self, dt):
        """ Get all assets with significant change in
        weights
        """
        tol = 0.001
        diffs = []
        self.populateAllData(dt)
        dtConsts = [i[0] for i in self.subIdWtPairs[dt]]
        prevDtConsts = [i[0] for i in self.subIdWtPairs[self.prevdt]]
        joiners = set(dtConsts) - set(prevDtConsts)
        dtConstsDict = dict([(i[0], i[1]) for i in self.subIdWtPairs[dt]])
        prevDtConstsDict = dict([(i[0], i[1]) for i in self.subIdWtPairs[self.prevdt]])        
        common = list(set(dtConsts) - joiners)
        for subid in common:
            if abs(dtConstsDict[subid]-prevDtConstsDict[subid]) > tol:
                mdlmktmap = self.getMarketIDs([subid.getModelID()], dt)
                (mktid, status) = mdlmktmap.get(subid.getModelID(), (None, 'Unknown'))
                diffs.append((mktid.getIDString(), subid.getSubIDString()[0:10], dtConstsDict[subid], prevDtConstsDict[subid], 'Y'))

        return diffs
    
    def getConstituentCount(self, dt):
        """ Get Market portfolio constituent count
        """
        self.populateAllData(dt)
        return len(self.subIdWtPairs[dt])

    def getMarketIDs(self, mdlids, dt):
        """ Return a mapping of Market IDs on dt for the given list of mdlids
        """
        retmap = {}
        try:
            dt = Utilities.parseISODate(dt)
        except Exception:
            pass
        mktidstr, mdlmktmap = self.connections.modelDB.getMarketDB_IDs(self.connections.marketDB, mdlids)
        for mdlid in mdlids:
            mktidinfo = mdlmktmap.get(mdlid)
            mktidinfo = sorted(mktidinfo, key=itemgetter(1), reverse=True)
            for (fromdt, thrudt, mktid) in mktidinfo:
                if fromdt <= dt and thrudt > dt:
                    retmap[mdlid] = (mktid, 'Active')
                    break
                elif thrudt <= dt:
                    retmap[mdlid] = (mktid, 'Terminated')
                    break
                elif fromdt == dt:
                    retmap[mdlid] = (mktid, 'Added')
                    break
                retmap[mdlid] = (None, 'Unknown')
                
        return retmap

#------------------------------------------------------------------------------
if __name__ == '__main__':
    usage = "usage: %prog [options] <config file> <rmg> <date>"

    cmdlineParser = optparse.OptionParser(usage=usage)
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 3:
        cmdlineParser.error("Incorrect number of arguments")

    try:
        dt = Utilities.parseISODate(args[2])
        mpc = MarketPortfolioChecker(args[0], dt, args[1])
        nconsts = mpc.getConstituentCount(dt)
        rptPrefix = ''
        if args[1]=='XC':
            rptPrefix='%s_'%args[1]
        print('%sMKT_PORT_CONST_COUNT~'%(rptPrefix), nconsts)
        top10 = mpc.getTop10(dt)
        print('%sMKT_PORT_TOP10_MCAP~'%(rptPrefix), top10)
        diffs = mpc.getConstituentDiffs(dt)
        print('%sMKT_PORT_CONST_DIFFS~'%(rptPrefix), diffs)
        wtdiffs = mpc.getLargeWtDiffs(dt)
        print('%sMKT_PORT_WEIGHT_DIFFS~'%(rptPrefix), wtdiffs)
        ret = mpc.calculateMarketReturn(dt)
        print('%sMKT_PORT_RETURN~'%(rptPrefix), ret)
        sys.exit(0)
    except Exception as e:
        print(traceback.print_tb(sys.exc_info()[2]))
        print('ERROR', Exception)
        print('ERROR', e)
        sys.exit(1)

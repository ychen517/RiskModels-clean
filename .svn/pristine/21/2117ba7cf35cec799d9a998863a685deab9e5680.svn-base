# Take the corporate actions from the MarketDB and convert them
# into ModelDB actions where possible.
# Flag corporate actions that need to be handled manually.
#
# Usage: createCorporateActions.py <date>
#
import cx_Oracle
import datetime
import logging
import numpy.ma as ma
import numpy
import optparse
import sys
from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities

# dictionary of (MarketDB ID -> ModelID) pairs that are mapped already
mappedMarketModel = dict()
# dictionary of (ModelID -> MarketDB) for ModelIDs that no longer have a
# living MarketID match
unmappedModelIDs = dict()
# dictionary of (MarketDB ID -> 1/2) for MarketDB IDs that are not mapped
# to a ModelDB ID. 2 means that a ModelID should be created even if there
# is no price
unmappedMarketIDs = dict()
# list of merger Struct's
mergers = list()
# list of spin-off Struct's
spinoffs = list()
# set of MarketDB IDs that need to be handled manually
badMarketIDs = set()
# dictionary of (MarketDB ID -> ModelID) of newly created ModelIDs
newMarketModel = dict()

def generateDeadIssues(date, deadIssues, marketDB, modelDB):
    """Create dead issue corporate actions.
    """
    valueDicts = [dict([('date_arg', str(date)),
                        ('market_arg', marketID),
                        ('model_arg', modelID.getIDString()),
                        ('rmg_arg', rmg_.rmg_id)])
                  for (modelID, marketID) in deadIssues.items()]
    if len(valueDicts) > 0:
        modelDB.dbCursor.executemany("""INSERT INTO ca_dead_issue
        (dt, modeldb_id, marketdb_id, kill_market, rmg_id)
        VALUES (:date_arg, :model_arg, :market_arg, 0, :rmg_arg)""",
                                     valueDicts)
    logging.info('Marked %d ModelDB as dead' % (len(valueDicts)))

def generateNewIssues(date, unmappedMarketIDs, rmg, marketDB, modelDB):
    """Create new issue corporate actions.
    """
    # Add a new ModelDB ID for new MarketDB IDs that have a price
    # convertible to the common currency.
    alwaysIDs = [i[0] for i in unmappedMarketIDs.items() if i[1] == 2]
    maybeIDs = [i[0] for i in unmappedMarketIDs.items() if i[1] == 1]
    prices = marketDB.getPrices([date], maybeIDs, rmg.currency_code)
    havePriceIndices = numpy.flatnonzero(ma.getmaskarray(prices.data[:,0])==False)
    havePriceIDs = [maybeIDs[i] for i in havePriceIndices]
    newMarketIDs = alwaysIDs + havePriceIDs
    newIDs = modelDB.createNewModelIDs(len(newMarketIDs))
    newMarketModel.update(dict(zip(newMarketIDs, newIDs)))
    isins = dict(marketDB.getISINs(date, newMarketIDs))
    sedols = dict(marketDB.getSEDOLs(date, newMarketIDs))
    refs = ['ISIN=%s, SEDOL=%s' % (isins.get(a), sedols.get(a))
            for a in newMarketIDs]
    valueDicts = [dict([('date_arg', str(date)),
                        ('model_arg', model.getIDString()),
                        ('market_arg', market), ('ref_arg', ref),
                        ('rmg_arg', rmg_.rmg_id)])
                  for (market, model, ref) in zip(newMarketIDs, newIDs, refs)]
    if len(valueDicts) > 0:
        modelDB.dbCursor.executemany("""INSERT INTO ca_new_issue
        (dt, modeldb_id, marketdb_id, ref, rmg_id)
        VALUES (:date_arg, :model_arg, :market_arg, :ref_arg, :rmg_arg)""", valueDicts)
    logging.info('Create %d new ModelDB issues' % (len(valueDicts)))

def generateMergers(date, mergers, marketDB, modelDB):
    """Create merger corporate actions.
    """
    valueDicts = [dict([('date_arg', str(date)),
                        ('seq_arg', m.ca_seq),
                        ('model_arg', m.modelID.getIDString()),
                        ('new_market_arg', m.newMarketID),
                        ('old_market_arg', m.oldMarketID),
                        ('ratio_arg', m.shareRatio),
                        ('cash_arg', m.cashAmount),
                        ('curr_arg', m.currency_id),
                        ('rmg_arg', rmg_.rmg_id)])
                  for m in mergers]
    if len(valueDicts) > 0:
        modelDB.dbCursor.setinputsizes(
            ratio_arg=cx_Oracle.NUMBER, seq_arg=cx_Oracle.NUMBER,
            cash_arg=cx_Oracle.NUMBER, curr_arg=cx_Oracle.NUMBER)
        modelDB.dbCursor.executemany("""INSERT INTO ca_merger_survivor
        (dt, ca_sequence, modeldb_id, new_marketdb_id, old_marketdb_id,
        share_ratio, cash_payment, currency_id, rmg_id)
        VALUES (:date_arg, :seq_arg, :model_arg, :new_market_arg,
        :old_market_arg, :ratio_arg, :cash_arg, :curr_arg, :rmg_arg)""", valueDicts)
    logging.info('Create %d merger survivor records'
                 % (len(valueDicts)))

def generateSpinOffs(date, spinoffs, marketDB, modelDB):
    """Create merger corporate actions.
    """
    mergedMarketModel = dict(mappedMarketModel)
    mergedMarketModel.update(newMarketModel)
    valueDicts = [dict([('date_arg', str(date)),
                        ('seq_arg', s.ca_seq),
                        ('parent_arg', s.parent.getIDString()),
                        ('child_arg', mergedMarketModel[s.child].getIDString()),
                        ('ratio_arg', s.shareRatio),
                        ('rmg_arg', rmg_.rmg_id)])
                  for s in spinoffs if s.child in mergedMarketModel]
    missing = [s for s in spinoffs if s.child not in mergedMarketModel]
    if len(missing) > 0:
        logging.error('Dropping spin-off records with unmapped child: %s'
                      % ','.join([str((s.ca_seq, s.child)) for s in missing]))
    if len(valueDicts) > 0:
        modelDB.dbCursor.setinputsizes(
            ratio_arg=cx_Oracle.NUMBER, seq_arg=cx_Oracle.NUMBER)
        modelDB.dbCursor.executemany("""INSERT INTO ca_spin_off
        (dt, ca_sequence, parent_id, child_id, share_ratio, rmg_id)
        VALUES (:date_arg, :seq_arg, :parent_arg, :child_arg, :ratio_arg, :rmg_arg)""",
                                     valueDicts)
    logging.info('Create %d spin-off records'
                 % (len(valueDicts)))

def getPaceID(child, marketDB, date, numDays):
    marketDB.dbCursor.execute("""SELECT id, from_dt, thru_dt
    FROM asset_dim_pace_id WHERE axioma_id = :market_arg
    AND trans_from_dt <= :trans_arg and trans_thru_dt > :trans_arg""",
                              market_arg=child,
                              trans_arg=marketDB.transDateTimeStr)
    r = [(i[0], marketDB.oracleToDate(i[1]), marketDB.oracleToDate(i[2]))
         for i in marketDB.dbCursor.fetchall()]
    if len(r) == 0:
        return None
    # Try exact match
    for (pace, from_dt, thru_dt) in r:
        if from_dt <= date and date < thru_dt:
            return pace
    # Try next few days day
    if numDays > 0:
        dateInc = datetime.timedelta(1)
    else:
        dateInc = datetime.timedelta(-1)
    for i in range(abs(numDays)):
        date = date + dateInc
        for (pace, from_dt, thru_dt) in r:
            if from_dt <= date and date < thru_dt:
                return pace
    return None

def pickMergerSurvivorPace(parents, modelParents, child, modelChild,
                           marketDB, modelDB, date):
    """Determine the survivor of the merger according to which PACE
    ID survived. This only works if the child and all parents have
    a PACE ID assigned.
    """
    childPace = getPaceID(child, marketDB, date, 5)
    childPace2 = getPaceID(child, marketDB, date + datetime.timedelta(7), 5)
    if childPace != childPace2:
        logging.warning('PACE ID for %s changes from %s to %s soon after merger'
                     ' on %s' % (child, childPace, childPace2, date))
        childPace = childPace2
    parentPace = [getPaceID(i, marketDB, date-datetime.timedelta(1), -5)
                  for i in parents]
    if childPace != None:
        for (market, model, pace) in zip(parents, modelParents, parentPace):
            if childPace == pace:
                return model

        if not numpy.alltrue([i != None for i in parentPace]):
            return None
        
        childPrevPace = getPaceID(child, marketDB,
                                  date-datetime.timedelta(1), -5)
        if childPrevPace == childPace and modelChild != None:
            return modelChild
        else:
            return 'New Company'
    return None
    
def pickMergerSurvivorMCap(parents, modelParents, child, modelChild,
                           rmg, marketDB, modelDB, date):
    """Determine the survivor of the merger according to which
    MarketDB ID has the largest common equity.
    If we don't have common equity for all parents, make no decision.
    """
    marketIssues = parents + [child]

    ceList = []
    for marketID in marketIssues:
        ce = sorted(marketDB.getFundamentalDataHistory(
            'AnnualCommonEquity', marketID, date - datetime.timedelta(365),
            date + datetime.timedelta(1), date + datetime.timedelta(1),
            rmg.currency_code, MarketDB.CurrencyProvider(marketDB, 10)))
        ceList.append([val for (dt, val, trans) in ce])
    marketCEMap = dict()
    for (a, parent, val) in zip(marketIssues, modelParents + [modelChild],
                                ceList):
        if len(val) == 0:
            if a != child:
                return None
            else:
                marketCEMap[a] = (0.0, parent)
        else:
            marketCEMap[a] = (val[0], parent)

    if not numpy.alltrue([i in marketCEMap for i in marketIssues]):
        # Not all common equity present -> no decision
        return None

    largestIdx = numpy.argmax([i[0] for i in marketCEMap.values()])
    largestMCap = list(marketCEMap.values())[largestIdx][0]
    largestParent = list(marketCEMap.keys())[largestIdx]
    largestModelParent = list(marketCEMap.values())[largestIdx][1]
    
    del marketCEMap[largestParent]
    if len(marketCEMap) == 0:
        return largestModelParent
    
    if largestMCap <= 0.0:
        return None
    
    secondIdx = numpy.argmax([i[0] for i in marketCEMap.values()])
    secondMCap = list(marketCEMap.values())[secondIdx][0]

    if (secondMCap / largestMCap) < 0.75:
        return largestModelParent
    return None

def pickMergerSurvivor(parents, modelParents, child, modelChild,
                       rmg, marketDB, modelDB, date):
    paceSurvivor = pickMergerSurvivorPace(parents, modelParents, child,
                                          modelChild, marketDB, modelDB, date)
    mcapSurvivor = pickMergerSurvivorMCap(parents, modelParents, child,
                                          modelChild, rmg, marketDB, modelDB,
                                          date)
    if paceSurvivor is not mcapSurvivor:
        logging.warning('PACE and Market Cap. pick different survivor'
                     ' for %s on %s: %s, %s' % (child, date, paceSurvivor,
                                               mcapSurvivor))
    if paceSurvivor is None:
        return mcapSurvivor
    return paceSurvivor

def processConvergences(marketConvergences, rmg, marketDB, modelDB, date):
    # flag convergences we can handle: only cash, or only stock
    # or stock + cash with only one target which we have a MarketID for
    logging.info('processing convergences')
    for conv in marketConvergences:
        handleMerger = True
        stockTerm = None
        cashTerm = None
        if conv.parent in badMarketIDs:
            handleMerger = False
        for option in conv.options:
            stockTerms = [t for t in option
                          if isinstance(t, MarketDB.ConvergenceStockTerm)]
            cashTerms = [t for t in option
                         if isinstance(t, MarketDB.ConvergenceCashTerm)]
            assert(len(cashTerms) + len(stockTerms) == len(option))
            if len(stockTerms) > 1 or len(cashTerms) > 1:
                handleMerger = False
                break
            
            if len(stockTerms) == 1:
                if stockTerms[0].child == None:
                    # Can't handle merger with unknown child
                    handleMerger = False
                    break
                if stockTerm != None \
                       and stockTerm.child != stockTerms[0].child:
                    handleMerger = False
                    break
                if stockTerms[0].child in badMarketIDs:
                    handleMerger = False
                    break
                stockTerm = stockTerms[0]
                    
        conv.canHandle = handleMerger

    # Group convergences by their children
    convergenceGroups = dict() # MarketID -> [Struct]
    for conv in marketConvergences:
        # gather all target Market IDs involved (including None)
        stockTargets = dict() # MarketID -> 1 map
        for option in conv.options:
            stockTerms = [t for t in option
                          if isinstance(t, MarketDB.ConvergenceStockTerm)]
            cashTerms = [t for t in option
                         if isinstance(t, MarketDB.ConvergenceCashTerm)]
            for t in stockTerms:
                stockTargets[t.child] = 1
            for t in cashTerms:
                pass # XXX Handle child here as well
        stockIDs = list(stockTargets.keys())
        # gather all existing convergence groups for the stocks
        myGroup = Utilities.Struct()
        myGroup.marketIDs = stockIDs
        myGroup.convergences = [conv]
        myGroup.canHandle = conv.canHandle
        
        groups = dict() # Group -> 1 map
        for s in stockIDs:
            if s in convergenceGroups:
                groups[convergenceGroups[s]] = 1
        # merge all groups into myGroup
        for g in groups.keys():
            mergedStocks = dict([(i,1) for i
                                 in myGroup.marketIDs + g.marketIDs])
            mergedConvergences = dict([(i, 1) for i in myGroup.convergences
                                       + g.convergences])
            myGroup.canHandle = (myGroup.canHandle and g.canHandle)
            myGroup.marketIDs = list(mergedStocks.keys())
            myGroup.convergences = list(mergedConvergences.keys())
        # set group for all involved stocks to this
        for s in myGroup.marketIDs:
            convergenceGroups[s] = myGroup
    convergenceGroups = list(dict([(g, 1) for g in convergenceGroups.values()]).keys())
    # process group convergence groups
    #  - pick a survivor
    # if no survivor can be picked, flag group as bad
    # else create ModelDB merger record for survivor
    # and declare everybody else dead
    unmappedModel_MarketMap = dict([(j,i) for (i,j) in unmappedModelIDs.items()])
    for group in [g for g in convergenceGroups if g.canHandle]:
        # Collect all companies involved, i.e. parents and the child
        assert(len(group.marketIDs) == 1)
        parents = [conv.parent for conv in group.convergences]
        # Map parents to their ModelIDs
        modelParents = []
        for p in parents:
            if p in mappedMarketModel:
                modelParents.append(mappedMarketModel[p])
            elif p in unmappedModel_MarketMap:
                modelParents.append(unmappedModel_MarketMap[p])
        if len(parents) != len(modelParents):
            # Can't find parents -> bad convergence
            logging.warning("Can't find ModelIDs for all parents in: %s"
                         % ','.join(parents))
            group.canHandle = False
            continue
        # Map child
        child = group.marketIDs[0]
        modelChild = None
        if child in mappedMarketModel:
            # Child already exists
            modelChild = mappedMarketModel[child]
        else:
            if child not in unmappedMarketIDs:
                logging.error('Merger child %s not an active ID' % child)
                group.canHandle = False
                continue
        # Check that child exists has not been declared dead
        assert(child not in unmappedModel_MarketMap)
        survivor = pickMergerSurvivor(parents, modelParents, child, modelChild,
                                      rmg, marketDB, modelDB, date)
        if survivor is None:
            group.canHandle = False
            continue
        if not isinstance(survivor, ModelID.ModelID):
            logging.info('%s is considered a new company, convergences %s'
                         % (child, ','.join([str(c.conv_id) for c
                                             in group.convergences])))
            # Declare parents dead
            for conv in group.convergences:
                parent = conv.parent
                if parent in mappedMarketModel:
                    unmappedModelIDs[mappedMarketModel[parent]] = parent
                    del mappedMarketModel[parent]
            if modelChild != None:
                # Declare current child ID dead
                unmappedModelIDs[modelChild] = child
                del mappedMarketModel[child]
            # Add child to list of new issues
            unmappedMarketIDs[child] = 2
        else:
            survivorMarket = 'None'
            if modelChild != None and survivor == modelChild:
                survivorMarket = child
            for (parent, modelParent) in zip(parents, modelParents):
                if modelParent == survivor:
                    survivorMarket = parent
            logging.info('%s is considered a continuation of %s, ModelID %s'\
                         ', convergences %s'
                         % (child, survivorMarket, survivor.getIDString(),
                            ','.join([str(c.conv_id) for c
                                      in group.convergences])))
            # Declare parents dead
            for conv in group.convergences:
                parent = conv.parent
                if parent in mappedMarketModel:
                    unmappedModelIDs[mappedMarketModel[parent]] = parent
                    del mappedMarketModel[parent]
            # If child has a ModelID but it doesn't survive, declare it
            # dead as well
            if modelChild != None and modelChild != survivor:
                unmappedModelIDs[modelChild] = child
                del mappedMarketModel[child]
            # New mapping for survivor.
            # If the survivor is already mapped, then it continues unchanged
            # so we don't need a merger record.
            if child in mappedMarketModel:
                continue
            mappedMarketModel[child] = survivor
            if survivor in unmappedModelIDs:
                # Can't find parents -> bad convergence
                logging.warning("Can't find ModelIDs for all parents in: %s"
                             % ','.join(parents))
                group.canHandle = False
                continue
            del unmappedModelIDs[survivor]
            # If child is a new MarketID, don't create a ModelID for it
            if child in unmappedMarketIDs:
                del unmappedMarketIDs[child]
            merger = Utilities.Struct()
            merger.ca_seq = 0
            merger.modelID = survivor
            merger.newMarketID = child
            merger.oldMarketID = survivorMarket
            merger.shareRatio = 1.0
            merger.cashAmount = 0.0
            merger.currency_id = None
            # Pick terms, if the survivor is not among the parents,
            # it continues unchanges, i.e., shareRatio of 1, no cash
            for conv in group.convergences:
                if survivorMarket == conv.parent:
                    done = False
                    merger.ca_seq = conv.sequenceNumber
                    #check if there's a stock only term
                    for option in conv.options:
                        stockTerms = [t for t in option if isinstance(
                            t, MarketDB.ConvergenceStockTerm)]
                        cashTerms = [t for t in option  if isinstance(
                            t, MarketDB.ConvergenceCashTerm)]
                        if len(stockTerms) == 1 and len(cashTerms) == 0:
                            merger.shareRatio = stockTerms[0].share_ratio
                            done = True
                            break
                    if done:
                        break
                    #check if there's a stock + cash term
                    for option in conv.options:
                        stockTerms = [t for t in option if isinstance(
                            t, MarketDB.ConvergenceStockTerm)]
                        cashTerms = [t for t in option  if isinstance(
                            t, MarketDB.ConvergenceCashTerm)]
                        if len(stockTerms) == 1 and len(cashTerms) == 1:
                            merger.shareRatio = stockTerms[0].share_ratio
                            merger.cashAmount = cashTerms[0].amount
                            merger.currency_id = cashTerms[0].currency_id
                            done = True
                            break
                    if done:
                        break
                    #check if there's a cash term
                    for option in conv.options:
                        stockTerms = [t for t in option if isinstance(
                            t, MarketDB.ConvergenceStockTerm)]
                        cashTerms = [t for t in option  if isinstance(
                            t, MarketDB.ConvergenceCashTerm)]
                        if len(stockTerms) == 0 and len(cashTerms) == 1:
                            merger.cashAmount = cashTerms[0].amount
                            merger.currency_id = cashTerms[0].currency_id
                            done = True
                            break
                    assert(done)
            mergers.append(merger)
            
    # process bad convergence groups:
    #  - write them to the log for later handling
    #  - declare the companies involved dead
    for group in [g for g in convergenceGroups if not g.canHandle]:
        logging.error("Can't handle convergences %s on %s"
                      % (','.join([str(c.conv_id) for c
                                   in group.convergences]),
                         date))
        for conv in group.convergences:
            parent = conv.parent
            if parent in mappedMarketModel:
                unmappedModelIDs[mappedMarketModel[parent]] = parent
                del mappedMarketModel[parent]

def pickSpinOffSurvivorMCap(parent, children, marketDB, modelDB, date):
    """Determine the survivor of the spin-off according to which
    MarketDB ID has the largest market capitalization.
    If we don't have market capitalization for all children, make no decision.
    """
    marketIssues = [c[0] for c in children]
    prices = marketDB.getPrices([date], marketIssues, 'USD')
    tso = marketDB.getSharesOutstanding([date], marketIssues)
    mcap = prices.data * tso.data

    marketCapMap = dict()
    for (a, val) in zip(marketIssues, mcap[:,0]):
        if val is not ma.masked:
            marketCapMap[a] = val

    if not numpy.alltrue([i in marketCapMap for i in marketIssues]):
        # Not all market cap present -> no decision
        return None
    largestIdx = numpy.argmax(list(marketCapMap.values()))
    largestMCap = list(marketCapMap.values())[largestIdx]
    largestChild = list(marketCapMap.keys())[largestIdx]
    
    del marketCapMap[largestChild]
    if len(marketCapMap) == 0:
        return largestChild
    
    if largestMCap <= 0.0:
        return None
    
    secondIdx = numpy.argmax(list(marketCapMap.values()))
    secondMCap = list(marketCapMap.values())[secondIdx]

    if (secondMCap / largestMCap) < 0.75:
        return largestChild
    return None

def pickSpinOffSurvivorPace(parent, children, marketDB, modelDB, date):
    parentPace = getPaceID(parent, marketDB, date-datetime.timedelta(1), -10)
    childrenPace = [getPaceID(c[0], marketDB, date, 10) for c in children]
    if parentPace != None:
        for (child, pace) in zip(children, childrenPace):
            if parentPace == pace:
                return child[0]
    if parentPace != None \
           and numpy.alltrue([i != None for i in childrenPace]):
        return 'New Companies'

    return None
            
def pickSpinOffSurvivor(parent, children, marketDB, modelDB, date):
    paceSurvivor = pickSpinOffSurvivorPace(parent, children, marketDB, modelDB,
                                           date)
    mcapSurvivor = pickSpinOffSurvivorMCap(parent, children, marketDB, modelDB,
                                           date)
    if paceSurvivor is not mcapSurvivor:
        logging.warning('PACE and Market Cap. pick different survivor'
                     ' for %s on %s: %s, %s' % (parent, date, paceSurvivor,
                                                mcapSurvivor))
    if paceSurvivor is None:
        return mcapSurvivor
    return paceSurvivor

def processDivergences(marketDivergences, marketDB, modelDB, date):
    unmappedModel_MarketMap = dict([(j,i) for (i,j)
                                    in unmappedModelIDs.items()])
    for div in marketDivergences:
        modelParent = mappedMarketModel.get(div.parent, None)
        if modelParent is None:
            modelParent = unmappedModel_MarketMap.get(div.parent, None)
        if modelParent is None:
            logging.warning('Spin-off parent %s has no ModelID on %s, ignoring it'
                         % (div.parent, date))
            continue
        if div.parentSurvives:
            survivor = pickSpinOffSurvivor(
                div.parent, div.children + [(div.parent,None)], marketDB,
                modelDB, date)
        else:
            survivor = pickSpinOffSurvivor(
                div.parent, div.children, marketDB, modelDB, date)
        if survivor == div.parent or survivor == 'New Companies' \
               or survivor == None:
            # Parent is survivor, enter spin-off records as-is
            for (c, ratio) in div.children:
                spinoff = Utilities.Struct()
                spinoff.ca_seq = div.sequenceNumber
                spinoff.parent = modelParent
                spinoff.child = c
                spinoff.shareRatio = ratio
                spinoffs.append(spinoff)
                if c in unmappedMarketIDs:
                    unmappedMarketIDs[c] = 2
            # Check that the parent behaves as expected
            if div.parentSurvives != (div.parent in mappedMarketModel):
                logging.error('Divergence and Market ID life-time differ'
                              ' for parent %s/%s'
                              % (div.parentSurvives,
                                 div.parent in mappedMarketModel))
        else:
            # One of the spun-off companies is declared the survivor.
            # Add a merger record to change the ModelID to map
            # to the survivor after the spin-off.
            # If the parent survives the spin-off, make sure a new
            # ModelID is issued for its MarketDB ID and add
            # a spin-off record for it with share ratio 1
            if survivor in mappedMarketModel:
                # survivor already exists, how should we handle this?
                logging.fatal('Survivor %s in divergence of %s already'
                              'has ModelID %s. Dropping it'
                              % (survivor, div.parent,
                                 mappedMarketModel[survivor]))
                continue
            if survivor in unmappedMarketIDs:
                # We don't need a new ModelID for the survivor
                del unmappedMarketIDs[survivor]

            if div.parentSurvives:
                if div.parent not in mappedMarketModel:
                    # Make sure parent ModelID stays alive
                    assert(div.parent in unmappedModel_MarketMap)
                    del unmappedModelIDs[modelParent]
                    mappedMarketModel[div.parent] = modelParent
            else:
                # Make sure parent MarketID is not mapped
                if div.parent in unmappedMarketIDs:
                    del unmappedMarketIDs[div.parent]
                if div.parent in mappedMarketModel:
                    del mappedMarketModel[div.parent]
                # Make sure parent ModelID stays alive
                if modelParent in unmappedModelIDs:
                    del unmappedModelIDs[modelParent]
                mappedMarketModel[survivor] = modelParent
            if div.parentSurvives:
                spinoff = Utilities.Struct()
                spinoff.ca_seq = div.sequenceNumber
                spinoff.parent = modelParent
                spinoff.child = div.parent
                spinoff.shareRatio = 1.0
                spinoffs.append(spinoff)
                unmappedMarketIDs[div.parent] = 2
            for (c, ratio) in div.children:
                if c != survivor:
                    spinoff = Utilities.Struct()
                    spinoff.ca_seq = div.sequenceNumber
                    spinoff.parent = modelParent
                    spinoff.child = c
                    spinoff.shareRatio = ratio
                    spinoffs.append(spinoff)
                    if c in unmappedMarketIDs:
                        unmappedMarketIDs[c] = 2
                else:
                    merger = Utilities.Struct()
                    merger.ca_seq = div.sequenceNumber + 1
                    merger.modelID = modelParent
                    merger.newMarketID = survivor
                    merger.oldMarketID = div.parent
                    merger.shareRatio = ratio
                    merger.cashAmount = 0.0
                    merger.currency_id = None
                    mergers.append(merger)

def createBadAssetList(marketDivergences, marketConvergences):
    """Build list of Market IDs that need to be handled manually:
     - involved in both divergence and convergence
     - involved as both parent and child in convergence
     - involved in more than one convergence
     """
    global badMarketIDs
    convParents = set()
    convChildren = set()
    badMarketIDs.clear()
    for conv in marketConvergences:
        if conv.parent != None:
            if conv.parent in convParents:
                badMarketIDs.add(conv.parent)
            convParents.add(conv.parent)
        for option in conv.options:
            stockTerms = [t for t in option if isinstance(
                t, MarketDB.ConvergenceStockTerm)]
            cashTerms = [t for t in option  if isinstance(
                t, MarketDB.ConvergenceCashTerm)]
            for c in [t.child for t in stockTerms if t.child != None]:
                convChildren.add(c)
            # XXX Handle cash child ID as well
    divParents = set()
    divChildren = set()
    for div in marketDivergences:
        if div.parent != None:
            divParents.add(div.parent)
        for c in [c for c in div.children if c != None]:
            divChildren.add(c)
    allConv = convParents | convChildren
    allDiv = divParents | divChildren
    badMarketIDs |= (allConv & allDiv)
    badMarketIDs |= (convParents & convChildren)
    if len(badMarketIDs) > 0:
        logging.warning('%d MarketDB IDs flagged as bad: %s'
                     % (len(badMarketIDs), ','.join(badMarketIDs)))

def getActiveAssets(marketDB, date):
    return dict([(i,0) for i in marketDB.getAssets(date)])

def processCorporateActions(marketDB, modelDB, date):
    # get currently active model<->market map
    global unmappedMarketIDs
    global mergers
    global spinoffs
    mappedMarketModel.clear()
    unmappedModelIDs.clear()
    mergers = list()
    spinoffs = list()
    badMarketIDs.clear()
    newMarketModel.clear()

    issuePairs = modelDB.getIssueMapPairs(date)

    marketDivergences = marketDB.getAssetDivergences(date)
    marketConvergences = marketDB.getAssetConvergences(date)
    marketAssets = getActiveAssets(marketDB, date)

    for (model, market) in issuePairs:
        if market in marketAssets:
            marketAssets[market] = 1
            mappedMarketModel[market] = model
        else:
            unmappedModelIDs[model] = market
    unmappedMarketIDs = dict([(i,1) for (i,j) in marketAssets.items()
                              if j == 0])

    logging.info('%d potentially new model assets' % len(unmappedMarketIDs))
    print(unmappedMarketIDs)
    logging.info('%d potentially dead model assets' % len(unmappedModelIDs))
    logging.info('%d raw divergences' % len(marketDivergences))
    logging.info('%d raw convergences' % len(marketConvergences))

    createBadAssetList(marketDivergences, marketConvergences)
    processConvergences(marketConvergences, rmg_, marketDB, modelDB, date)
    processDivergences(marketDivergences, marketDB, modelDB, date)
    generateNewIssues(date, unmappedMarketIDs, rmg_, marketDB, modelDB)
    generateMergers(date, mergers, marketDB, modelDB)
    generateSpinOffs(date, spinoffs, marketDB, modelDB)
    generateDeadIssues(date, unmappedModelIDs, marketDB, modelDB)

if __name__ == '__main__':
    usage = "usage: %prog [options] <rmg-id> <YYYY-MM-DD>"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                         default=False, dest="testOnly",
                         help="don't change the database")
    Utilities.addDefaultCommandLine(cmdlineParser)
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    Utilities.processDefaultCommandLine(options, cmdlineParser)

    modelDB_ = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser,
                              passwd=options.modelDBPasswd)
    marketDB_ = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser, passwd=options.marketDBPasswd)

    date = Utilities.parseISODate(args[1])
    rmg_id_ = int(args[0])
    rmg_ = modelDB_.getRiskModelGroup(rmg_id_)
    rmg_.setRMGInfoForDate(date)
    if rmg_ == None:
        print('No record for risk model group %d on %s' % (rmg_id_, date))
        sys.exit(1)

    logging.info('Creating ModelDB corporate actions on %s' % date)
    status = 0
    try:
        processCorporateActions(marketDB_, modelDB_, date)
    except Exception:
        logging.fatal('Exception caught during processing. Reverting changes',
                      exc_info=True)
        status = 1
        modelDB_.revertChanges()
    if options.testOnly:
        logging.info('Reverting changes')
        modelDB_.revertChanges()
    else:
        modelDB_.commitChanges()
    modelDB_.finalize()
    marketDB_.finalize()
    sys.exit(status)

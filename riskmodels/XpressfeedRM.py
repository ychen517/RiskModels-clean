''' TODO: Add Documentation
'''
import logging

from marketdb import VendorUtils
from marketdb import XPS


class XpressfeedData(object):
    
    """Class to retrieve the values for a fundamental data item from
       MarketDB. The item names are resolved to codes via the meta_codes table.
    """
    def __init__(self, connections):
        self.connections_ = connections
        
        # 1-A Load AxiomaDB/MarketDB-ID to ModelDB-ID Mapping 
        self.axID2ModelDbIDHistoryMap_ = None
        self.modelID2AxIDHistoryMap_ = None
        
        # 1-B Sub-Issue Mapping
        self.subIssueID2ModelDbIDHistoryMap_ = None
        self.modelID2SubIssueIDHistoryMap_ = None
    
        # 2. Create XpressFeed (in MarketDB) bridge to Xpressfeed database
        self.xpsSource_ = XPS.FundamentalData(connections)
         
#     def getFundamentalDataMxMap(self, subIssueIdList, axItemIdList): 
#         # 1. Convert subIssueIds to Axioma-IDs (MarketDB-IDs)
#         axIdList = self._getAxiomaIDs(subIssueIdList)
#          
#         # 2. Get Xpressfeed Fundamental Data
#         xpsFundDataMxMap = self.xpsSource_.getFundamentalDataMxMap(axIdList, axItemIdList)
#         
#         # 3. Convert from AxiomaID to SubIssueID via ModelDB
#         return xpsFundDataMxMap    
   
    def getFundamentalDataMatrix(self, subIssueIdList, axItemCodeID): 
        # 1. Convert subIssueIds to Axioma-IDs (MarketDB-IDs)
        axIdList = self._getAxiomaIDs(subIssueIdList)
        return self.xpsSource_.getFundamentalDataMatrix(axIdList, axItemCodeID)  
 
    #--------------------------------#
    #    Private Utility Methods     #
    #--------------------------------#
    # 1. Load AxiomaDB/MarketDB-ID to ModelDB-ID Mapping 
    def _getAxiomaToModelDbIDHistoryMap(self):
        if (self.axID2ModelDbIDHistoryMap_ == None):
            modelDbIDDP = VendorUtils.ModelDbIDHistoryMap(self.connections_.modelDB)
            self.axID2ModelDbIDHistoryMap_ = modelDbIDDP.getAxiomaIDToModelDbIDHistoryMap()
        return self.axID2ModelDbIDHistoryMap_
    
    def _getModelDbIDToAxiomaIDHistoryMap(self):
        if (self.modelID2AxIDHistoryMap_ == None):
            modelDbIDDP = VendorUtils.ModelDbIDHistoryMap(self.connections_.modelDB)
            self.modelID2AxIDHistoryMap_ = modelDbIDDP.getInverseAssetIDHistoryMap()
        return self.modelID2AxIDHistoryMap_
       
    def _getSubIssueIDToModelDbIDHistoryMap(self):
        if (self.subIssueID2ModelDbIDHistoryMap_ is None):
            subIssueIDDP = VendorUtils.SubIssueIDHistoryMap(self.connections_.modelDB)
            self.subIssueID2ModelDbIDHistoryMap_ =  subIssueIDDP.getModelDbIDToSubIssueIDHistoryMap()
        return self.subIssueID2ModelDbIDHistoryMap_ 
    
    def _getModelDbIDToSubIssueIDHistoryMap(self):
        if (self.modelID2SubIssueIDHistoryMap_ == None):
            subIssueIDDP = VendorUtils.SubIssueIDHistoryMap(self.connections_.modelDB)
            #self.subIssueID2ModelIDHistoryMap_ = subIssueIDDP.getModelDbIDToSubIssueIDHistoryMap()
            self.modelID2SubIssueIDHistoryMap_ = subIssueIDDP.getInverseAssetIDHistoryMap()
        return self.modelID2SubIssueIDHistoryMap_
    
    def _getSubIssueIDsToAxiomaIDMap(self, subIssueList, dateIn):
        subIssueIDMap = {}
        for subIssue in subIssueList:
            subIssueID = subIssue.getSubIDString()
            logging.debug("Processing SubIssue: %s for Date:%s with ModelID:%s"%(subIssueID, dateIn.date(), subIssue.getModelID()))
            axiomaID = self._getCorrespondingAxiomaID(subIssueID, dateIn)
            if axiomaID is not None:
                subIssueIDMap.update({ axiomaID : subIssueID})
        return subIssueIDMap    
    
    def _getCorrespondingAxiomaID(self, subIssueID, dateIn):
        modelDbID = subIssueID[:10]
        modelDBID2AxIDHistory = self.modelID2AxIDHistoryMap_.get(modelDbID)
        return modelDBID2AxIDHistory.getVendorIDForDate(dateIn) if modelDBID2AxIDHistory  is not None else None
    
    def _getAxiomaIDs(self, subIssueIdList):
        axIDList = set()
        subIssueID2ModelDbIDHistMap = self._getSubIssueIDToModelDbIDHistoryMap()
        for subIssueID in subIssueIdList:
            subIssueIdHistoryMap = subIssueID2ModelDbIDHistMap.get(subIssueID)
            
            if subIssueIdHistoryMap is not None: # Ex. D36M7CJ4H011 
                aidOverTimeMap = subIssueIdHistoryMap.getVendorIDOverTimeMap()
            
                for modelDbID, timeLine in aidOverTimeMap.items():
                    dateRangeList = timeLine.getDateRangeList() # vs. getEffDateRangeList()
                    
                    for dateRange in dateRangeList:
                        modelDb2AxIDHistoryMap = self._getModelDbIDToAxiomaIDHistoryMap()
                        axIDHistory = modelDb2AxIDHistoryMap.get(modelDbID)
                        #print subIssueID, ":", modelDbID, "!=", dateRange, '!=', axIDHistory
                        if axIDHistory is not None:  # Ex. D11C2RB5L611 : D11C2RB5L6 != From: 2010-06-01 Thru: 2013-05-09
                            axIDsOverTimeMap = axIDHistory.getVendorIDOverTimeMap()   # This is a OneToManyMap instance
                            for axID, dateRangeList in axIDsOverTimeMap.items():
                                axIDList.add(axID)
        return list(axIDList)
    
    def _getSubIssueID(self, axiomaIDList):
        subIssueIDList = []
        modelID2AxiomaIDHistMap = self._getModelDbIDToSubIssueIDHistoryMap()
        
        for axiomaID in axiomaIDList:
            modelIDHistoryMap = modelID2AxiomaIDHistMap.get(axiomaID)
              
            if modelIDHistoryMap is not None: # Ex. D36M7CJ4H011 
                aidOverTimeMap = modelIDHistoryMap.getVendorIDOverTimeMap()
                for modelID, timeLine in aidOverTimeMap.items():
                    subIssueIDList.append('%s11'%modelID)
                
#                 for modelDbID, timeLine in aidOverTimeMap.iteritems():
#                     dateRangeList = timeLine.getDateRangeList() # vs. getEffDateRangeList()
#                     
#                     for dateRange in dateRangeList:
#                         modelDb2AxIDHistoryMap = self._getModelDbIDToAxiomaIDHistoryMap()
#                         axIDHistory = modelDb2AxIDHistoryMap.get(modelDbID)
#                         #print subIssueID, ":", modelDbID, "!=", dateRange, '!=', axIDHistory
#                         if axIDHistory is not None:  # Ex. D11C2RB5L611 : D11C2RB5L6 != From: 2010-06-01 Thru: 2013-05-09
#                             axIDsOverTimeMap = axIDHistory.getVendorIDOverTimeMap()   # This is a OneToManyMap instance
#                             for axID, dateRangeList in axIDsOverTimeMap.iteritems():
#                                 axIDList.add(axID)
        subIssueIDs = set(subIssueIDList)
        return list(subIssueIDs) 

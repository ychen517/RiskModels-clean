import datetime
import numpy.ma as ma
import numpy
import pandas
import logging
from riskmodels import Matrices
from riskmodels import Utilities

class Classification:
    """Provides a classification based on a classification
    name and date as present in ModelDB.
    """
    def __init__(self, name, date):
        self.date = date # the date of the classification revision to use
        self.name = name # name of the classification member
        self.member = None # the ModelDB member of this classification
        self.revision = None # the ModelDB revision of this classification
        self.nodes = None
        self.log_ = None
        
    def getLogger(self):
        if self.log_ == None:
            self.log_ = logging.getLogger('Classification')
        return self.log_
    
    def getLeafNodes(self, modelDB):
        """Returns the leaves of the classification which are used
        as industries in the risk model.
        """
        if self.nodes == None:
            member = self.getClassificationMember(modelDB)
            industryList = modelDB.getMdlClassificationMemberLeaves(
                member, self.date)
            # (industry id, industry struct) map
            if industryList:
                self.nodes = dict([(ind.id, ind) for ind in industryList])
            else:
                self.nodes = {}
        return self.nodes
    
    def getDescriptions(self, modelDB):
        descs = sorted(ind.description for ind
                 in list(self.getLeafNodes(modelDB).values()))
        return descs
    
    def getNames(self, modelDB):
        names = sorted(ind.name for ind
                 in list(self.getLeafNodes(modelDB).values()))
        return names
    
    def getClassificationChildren(self, parentClass, modelDB):
        """Returns a list of the children of the given node in the
        classification."""
        children = modelDB.getMdlClassificationChildren(parentClass)
        return children
   
    def getClassificationParents(self, rootName, modelDB):
        root = [r for r in self.getClassificationRoots(
            modelDB) if r.name == rootName]
        return self.getClassificationChildren(
            root[0], modelDB)

    def getAllParents(self, childClass, modelDB):
        """Returns all parents of the given node in the
        classification."""
        parents = modelDB.getMdlClassificationAllParents(childClass)
        return parents
    
    def getClassificationMember(self, modelDB):
        """Returns the classification member object of this industry
        classification.
        """
        if self.member is None:
            family = modelDB.getMdlClassificationFamily('INDUSTRIES')
            memberDict = dict([(i.name,i) for i in
                               modelDB.getMdlClassificationFamilyMembers(family)])
            self.member = memberDict[self.name]
        return self.member
    
    def getClassificationRevision(self, modelDB):
        """Returns the classification revision object of this industry
        classification.
        """
        if self.revision is None:
            member = self.getClassificationMember(modelDB)
            self.revision = modelDB.getMdlClassificationMemberRevision(
                member, self.date)
        return self.revision
    
    def getAssetConstituents(self, modelDB, assetList, date, level=None):
        """Returns the classification_constituent information for a list of assets
        from this classification on the supplied date.
        Return is a dict mapping assets to the constituent information.
        """
        revision = self.getClassificationRevision(modelDB)
        return modelDB.getMdlAssetClassifications(revision, assetList, date, level)
    
    def getNumLevels(self, modelDB):
        """Returns the depth of this industry
        classification, which is the length of the
        list of root classifications.
        """
        member = self.getClassificationMember(modelDB)
        return len(modelDB.getMdlClassificationMemberRoots(member, self.date))

    def getNodesAtLevel(self, modelDB, levelname):
        """Returns list of industries that live under
        the root name whose name is levelname.  If levelname
        is None, then use the first root available
        """
        roots = self.getClassificationRoots(modelDB)
        if levelname is not None:
           root = [r for r in roots if r.name==levelname]
        else:
           root = roots
        
        return self.getClassificationChildren(root[0], modelDB)

    def getClassificationRoots(self, modelDB):
        """Returns the root classification objects of this industry
        classification.
        """
        member = self.getClassificationMember(modelDB)
        return modelDB.getMdlClassificationMemberRoots(member, self.date)
    
    def getAllClassifications(self, modelDB):
        """Returns all non-root classification objects of this industry
        classification.
        """
        member = self.getClassificationMember(modelDB)
        return modelDB.getMdlClassificationMembers(member, self.date)
    
    def copyClassification(self, parentID, children, marketDB, modelDB,
                           mktToMdlMap, isLeaf, revID, additionalParentID):
        """Internal method to copy the children of a MarketDB classification
        into ModelDB with their hierarchy settings.
        """
        valueDicts = []
        hierDicts = []
        mapDicts = []
        for s in children:
            assert(not s.isRoot)
            modelDB.dbCursor.execute(
                """SELECT classification_seq.nextval FROM DUAL""")
            nextID = modelDB.dbCursor.fetchall()[0][0]
            mktToMdlMap[s.id] = nextID
            if isLeaf == 'Y':
                name = s.description.encode('utf-8')
            else:
                name = s.name.encode('utf-8')
            values = dict([('name_arg', name),
                           ('id_arg', nextID),
                           ('rev_arg', revID),
                           ('leaf_arg', isLeaf),
                           ('desc_arg', s.description.encode('utf-8'))])
            valueDicts.append(values)
            mapDicts.append({'mdl_ref': nextID, 'mkt_ref': s.id})
            hier = dict([('parent_arg', parentID),
                         ('child_arg', nextID),
                         ('weight_arg', s.weight)])
            hierDicts.append(hier)
            if not additionalParentID is None:
                hier = dict([('parent_arg', additionalParentID),
                             ('child_arg', nextID),
                             ('weight_arg', s.weight)])
                hierDicts.append(hier)
        
        modelDB.dbCursor.executemany("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'N', :leaf_arg,
        :rev_arg)""", valueDicts)
        modelDB.dbCursor.executemany("""INSERT INTO classification_dim_hier
        (parent_classification_id, child_classification_id, weight)
        VALUES(:parent_arg, :child_arg, :weight_arg)""", hierDicts)
    
    def getExposures(self, date, subIssues, factors, modelDB, level=None, returnDF=False):
        """Returns a matrix of exposures of the assets to the classification
        scheme on the given date.
        The assets are given as a list of sub-issue objects.
        The factors are given as a list of factor names.
        The return value is an (industry, sub-issue) array containing
        the exposures. Assets and industries are in the same order as in
        the provided lists.
        """
        expMatrix = Matrices.allMasked((len(factors), len(subIssues)))
        issues = [s.getModelID() for s in subIssues]
        assetIdx = dict(zip(issues, range(len(subIssues))))
        revision = self.getClassificationRevision(modelDB)
        sidCls = modelDB.getMdlAssetClassifications(revision, issues, date, level)
        factorIdxMap = dict([(j,i) for (i,j) in enumerate(factors)])
        for (issue, clsStruct) in sidCls.items():
            fIdx = factorIdxMap[clsStruct.classification.description]
            aIdx = assetIdx[issue]
            exposure = clsStruct.weight
            if exposure != 1.0:
                self.getLogger().critical('Incorrect exposure for %s: %g', 
                    issue.getIDString(), exposure)
                exposure = 1.0
            expMatrix[fIdx, aIdx] = exposure
        missing = numpy.flatnonzero(ma.sum(expMatrix, axis=0).filled(0.0)
                                    < 1.0)
        if len(missing) > 0:
            self.getLogger().info(
                '%d assets are missing an industry classification', len(missing))
            self.getLogger().debug("missing Axioma IDs: "
                                  + ', '.join([subIssues[i].getSubIDString()
                                               for i in missing]))
        tooMuch = numpy.flatnonzero(ma.sum(expMatrix, axis=0).filled(0.0) > 1.0)
        if len(tooMuch) > 0:
            self.getLogger().critical(
                '%d assets have too much exposure', len(tooMuch))
            self.getLogger().info("problematic Axioma IDs: "
                                  + ', '.join([subIssues[i].getSubIDString()
                                               for i in tooMuch]))
        if returnDF:
            expMatrix = pandas.DataFrame(expMatrix, index=factors, columns=subIssues).T
        return expMatrix 

class GICSIndustries(Classification):
    """Provides an industry classification based on GICS Industries.
    GICS sub-industries are ignored and assets which don't have a GICS
    code, are assigned one based on their SIC code.
    """
    def __init__(self, gicsDate):
        Classification.__init__(self, 'GICSIndustries', gicsDate)
        
    def createMdlClassification(self, mdlRevision, modelDB, marketDB):
        """Create the sector down to industry hierarchy of the GICS
        on the classification date in the ModelDB.
        The member name is GICS (which is assumed to already exist
        and a new revision is added with the from/thru
        of the corresponding MarketDB classification/revision.
        The revision ID for the ModelDB revision is specified as the
        mdlRevision parameter.
        """
        mktToMdlMap = dict()
        mdlMember = self.getClassificationMember(modelDB)
        # Get MarketDB root classification for GICS
        industryFamily = marketDB.getClassificationFamily('INDUSTRIES')
        assert(industryFamily != None)
        industryMembers = marketDB.getClassificationFamilyMembers(
            industryFamily)
        gicsMember = dict([(i.name,i) for i in industryMembers])['GICS']
        # Get revision for MarketDB classification
        mktRevision = marketDB.getClassificationMemberRevision(
            gicsMember, self.date)
        # Get current root for this classification
        mktRoot = marketDB.getClassificationMemberRoot(gicsMember, self.date)
        # Insert ModelDB revision based on MarketDB revision
        modelDB.dbCursor.execute("""INSERT INTO classification_revision
        (id, member_id, from_dt, thru_dt)
        VALUES(:id_arg, :member_arg, :from_arg, :thru_arg)""",
                                 id_arg=mdlRevision,
                                 member_arg=mdlMember.id,
                                 from_arg=mktRevision.from_dt,
                                 thru_arg=mktRevision.thru_dt)
        # create sector parent                        
        modelDB.dbCursor.execute("""SELECT classification_seq.nextval
        FROM DUAL""")
        nextID = modelDB.dbCursor.fetchall()[0][0]
        modelDB.dbCursor.execute("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                 id_arg=nextID, rev_arg=mdlRevision,
                                 name_arg="Sectors".encode('utf-8'), 
                                 desc_arg="Sectors".encode('utf-8'))
        # modelDB.dbCursor.execute("""INSERT INTO classification_market_map
        # (model_ref_id, market_ref_id) VALUES(:mdl_ref, :mkt_ref)""",
        #                          mdl_ref=nextID, mkt_ref=mktRoot.id)
        mdlSECParent = modelDB.getMdlClassificationByID(nextID)
        mktToMdlMap[mktRoot.id] = mdlSECParent.id
        
        # create industry group parent
        modelDB.dbCursor.execute("""SELECT classification_seq.nextval
        FROM DUAL""")
        nextID = modelDB.dbCursor.fetchall()[0][0]
        modelDB.dbCursor.execute("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                 id_arg=nextID, rev_arg=mdlRevision,
                                 name_arg="Industry Groups".encode('utf-8'),
                                 desc_arg="Industry Groups".encode('utf-8'))
        mdlIGRParent = modelDB.getMdlClassificationByID(nextID)
        mktToMdlMap[mktRoot.id] = mdlIGRParent.id
        
        # create industry parent
        modelDB.dbCursor.execute("""SELECT classification_seq.nextval
        FROM DUAL""")
        nextID = modelDB.dbCursor.fetchall()[0][0]
        modelDB.dbCursor.execute("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                 id_arg=nextID, rev_arg=mdlRevision,
                                 name_arg="Industries".encode('utf-8'),
                                 desc_arg="Industries".encode('utf-8'))
        mdlINDParent = modelDB.getMdlClassificationByID(nextID)
        mktToMdlMap[mktRoot.id] = mdlINDParent.id
        
        assert(not mktRoot.isLeaf)
        sectors = marketDB.getClassificationChildren(mktRoot)
        self.copyClassification(mdlSECParent.id, sectors, marketDB, modelDB,
                                mktToMdlMap, 'N', mdlRevision, None)
        industryGroups = []
        for sec in sectors:
            assert(not sec.isLeaf)
            children = marketDB.getClassificationChildren(sec)
            self.copyClassification(mktToMdlMap[sec.id], children,
                                    marketDB, modelDB, mktToMdlMap, 'N',
                                    mdlRevision, mdlIGRParent.id)
            industryGroups.extend(children)
        industries = []
        for igr in industryGroups:
            assert(not igr.isLeaf)
            children = marketDB.getClassificationChildren(igr)
            self.copyClassification(mktToMdlMap[igr.id], children,
                                    marketDB, modelDB, mktToMdlMap, 'Y',
                                    mdlRevision, mdlINDParent.id)
            industries.extend(children)

class GICSSectors(Classification):
    """Provides an industry classification based on GICS Sectors.
    GICS industry groups, industries, and sub-industries are ignored.
    """
    def __init__(self, gicsDate):
        Classification.__init__(self, 'GICSSectors', gicsDate)
        self.baseClassification = GICSIndustryGroups(gicsDate)
        self.parents = None
    
    def getParentNodes(self, modelDB):
        if self.parents is None:
            roots = [r for r in self.baseClassification.\
                        getClassificationRoots(modelDB) if r.name=='Sectors']
            assert(len(roots)==1)
            self.parents = self.baseClassification.\
                            getClassificationChildren(roots[0], modelDB)
        return self.parents
    
    def getLeafNodes(self, modelDB):
        if self.baseClassification.nodes == None:
            industryList = self.getParentNodes(modelDB)
            self.baseClassification.nodes = dict([(ind.id, ind) for ind in industryList])
        return self.baseClassification.nodes
    
    def getAssetConstituents(self, modelDB, assetList, date):
        revision = self.baseClassification.getClassificationRevision(modelDB)
        return modelDB.getMdlAssetClassifications(
                        revision, assetList, date, level=-1)
    
    def getExposures(self, date, subIssues, factors, modelDB, level=None, returnDF=False):
        #parentNames = [n.description for n in self.getParentNodes(modelDB)]
        return self.baseClassification.getExposures(
                        date, subIssues, factors, modelDB, level=-1, returnDF=returnDF)

class SimpleClassification(GICSSectors):
    """Provides a coarse industry classification that only categorizes 
    stocks into Financials, Industrials, and Technology.
    Assignments are based on consolidating GICS Sectors.
    """
    sectorMap = {'Industrials': ('Energy', 'Materials', 'Industrials', 
                                 'Consumer Discretionary', 'Consumer Staples', 
                                 'Health Care', 'Telecommunication Services',
                                 'Utilities'),
                 'Financials': ('Financials',),
                 'Information Technology': ('Information Technology',)}

    def getClassificationChildren(self, parentClass, modelDB):
        return self.getSupersectorNodes(modelDB)

    def getClassificationRoots(self, modelDB):
        member = self.baseClassification.getClassificationMember(modelDB)
        return [modelDB.getMdlClassificationMemberRoots(
                            member, self.baseClassification.date)[0]]

    def getSupersectorNodes(self, modelDB):
        if self.nodes is None:
            self.nodes = list()
            for supersector in sorted(self.sectorMap.keys()):
                n = Utilities.Struct()
                n.description = supersector
                n.name = supersector
                n.id = supersector
                self.nodes.append(n)
        return self.nodes

    def getLeafNodes(self, modelDB):
        return dict([(ind.id, ind) for ind in self.getSupersectorNodes(modelDB)])

    def getAssetConstituents(self, modelDB, assetList, date):
        pass

    def getExposures(self, date, subIssues, factors, modelDB, level=None, returnDF=False):
        sectorNames = [n.description for n in self.getParentNodes(modelDB)]
        sectorIdxMap = dict([(j,i) for (i,j) in enumerate(sectorNames)])
        sectorExposures = self.baseClassification.getExposures(
                        date, subIssues, sectorNames, modelDB, level=-1, returnDF=False)
        expMatrix = Matrices.allMasked((len(list(self.sectorMap.keys())), len(subIssues)))
        for (i, supersector) in enumerate(factors):
            sectorList = self.sectorMap.get(supersector)
            a = ma.take(sectorExposures, 
                            [sectorIdxMap[s] for s in sectorList], axis=0)
            expMatrix[i,:] = ma.sum(ma.take(sectorExposures, 
                            [sectorIdxMap[s] for s in sectorList], axis=0), axis=0)
        if returnDF:
            expMatrix = pandas.DataFrame(expMatrix, index=factors, columns=subIssues).T
        return expMatrix

class GICSIndustryGroups(Classification):
    """Provides an industry classification based on GICS Industry Groups.
    GICS industries and sub-industries are ignored.
    """
    def __init__(self, gicsDate):
        Classification.__init__(self, 'GICSIndustryGroups', gicsDate)
        
    def createMdlClassification(self, mdlRevision, modelDB, marketDB):
        """Create the sector down to industry hierarchy of the GICS
        on the classification date in the ModelDB.
        The member name is GICS (which is assumed to already exist
        and a new revision is added with the from/thru
        of the corresponding MarketDB classification/revision.
        The revision ID for the ModelDB revision is specified as the
        mdlRevision parameter.
        """
        mktToMdlMap = dict()
        mdlMember = self.getClassificationMember(modelDB)
        # Get MarketDB root classification for GICS
        industryFamily = marketDB.getClassificationFamily('INDUSTRIES')
        assert(industryFamily != None)
        industryMembers = marketDB.getClassificationFamilyMembers(
            industryFamily)
        gicsMember = dict([(i.name,i) for i in industryMembers])['GICS']
        # Get revision for MarketDB classification
        mktRevision = marketDB.getClassificationMemberRevision(
            gicsMember, self.date)
        # Get current root for this classification
        mktRoot = marketDB.getClassificationMemberRoot(gicsMember, self.date)
        # Insert ModelDB revision based on MarketDB revision
        modelDB.dbCursor.execute("""INSERT INTO classification_revision
        (id, member_id, from_dt, thru_dt)
        VALUES(:id_arg, :member_arg, :from_arg, :thru_arg)""",
                                 id_arg=mdlRevision,
                                 member_arg=mdlMember.id,
                                 from_arg=mktRevision.from_dt,
                                 thru_arg=mktRevision.thru_dt)
        # create sector parent                        
        modelDB.dbCursor.execute("""SELECT classification_seq.nextval
        FROM DUAL""")
        nextID = modelDB.dbCursor.fetchall()[0][0]
        modelDB.dbCursor.execute("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                 id_arg=nextID, rev_arg=mdlRevision,
                                 name_arg="Sectors".encode('utf-8'), 
                                 desc_arg="Sectors".encode('utf-8'))
        # modelDB.dbCursor.execute("""INSERT INTO classification_market_map
        # (model_ref_id, market_ref_id) VALUES(:mdl_ref, :mkt_ref)""",
        #                          mdl_ref=nextID, mkt_ref=mktRoot.id)
        mdlSECParent = modelDB.getMdlClassificationByID(nextID)
        mktToMdlMap[mktRoot.id] = mdlSECParent.id
        
        # create industry group parent
        modelDB.dbCursor.execute("""SELECT classification_seq.nextval
        FROM DUAL""")
        nextID = modelDB.dbCursor.fetchall()[0][0]
        modelDB.dbCursor.execute("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                 id_arg=nextID, rev_arg=mdlRevision,
                                 name_arg="Industry Groups".encode('utf-8'),
                                 desc_arg="Industry Groups".encode('utf-8'))
        mdlIGRParent = modelDB.getMdlClassificationByID(nextID)
        mktToMdlMap[mktRoot.id] = mdlIGRParent.id
        
        assert(not mktRoot.isLeaf)
        sectors = marketDB.getClassificationChildren(mktRoot)
        self.copyClassification(mdlSECParent.id, sectors, marketDB, modelDB,
                                mktToMdlMap, 'N', mdlRevision, None)
        industryGroups = []
        for sec in sectors:
            assert(not sec.isLeaf)
            children = marketDB.getClassificationChildren(sec)
            self.copyClassification(mktToMdlMap[sec.id], children,
                                    marketDB, modelDB, mktToMdlMap, 'Y',
                                    mdlRevision, mdlIGRParent.id)
            industryGroups.extend(children)

class GICSBasedClassification(Classification):
    """Provides an industry classification based on merging
    GICS Industries
    """
    def __init__(self, name, date, codeToIndMap, lowestLevel):
        Classification.__init__(self, name, date)
        self.codeToIndMap = codeToIndMap
        assert(lowestLevel in ('Industry', 'IndustryGroup', 'Sector'))
        self.lowestLevel = lowestLevel
    
    def createMdlClassification(self, mdlRevision, modelDB, marketDB):
        """Create classification by merging GICS Sub-Industries.
        """
        mktToMdlMap = dict()
        mdlMember = self.getClassificationMember(modelDB)
        # Get MarketDB root classification for GICS
        industryFamily = marketDB.getClassificationFamily('INDUSTRIES')
        assert(industryFamily != None)
        industryMembers = marketDB.getClassificationFamilyMembers(
            industryFamily)
        gicsMember = dict([(i.name,i) for i in industryMembers])['GICS']
        # Get revision for MarketDB classification
        mktRevision = marketDB.getClassificationMemberRevision(
            gicsMember, self.date)
        # Get current root for this classification
        mktRoot = marketDB.getClassificationMemberRoot(gicsMember, self.date)
        # Insert ModelDB revision based on MarketDB revision
        modelDB.dbCursor.execute("""INSERT INTO classification_revision
        (id, member_id, from_dt, thru_dt)
        VALUES(:id_arg, :member_arg, :from_arg, :thru_arg)""",
                                 id_arg=mdlRevision,
                                 member_arg=mdlMember.id,
                                 from_arg=mktRevision.from_dt,
                                 thru_arg=mktRevision.thru_dt)
        # create sector parent                        
        modelDB.dbCursor.execute("""SELECT classification_seq.nextval
        FROM DUAL""")
        nextID = modelDB.dbCursor.fetchall()[0][0]
        modelDB.dbCursor.execute("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                 id_arg=nextID, rev_arg=mdlRevision,
                                 name_arg="Sectors".encode('utf-8'), 
                                 desc_arg="Sectors".encode('utf-8'))
        mdlSECParent = modelDB.getMdlClassificationByID(nextID)
        mktToMdlMap[mktRoot.id] = mdlSECParent.id
        
        if self.lowestLevel in ('Industry', 'IndustryGroup'):
            # create industry group parent
            modelDB.dbCursor.execute("""SELECT classification_seq.nextval
            FROM DUAL""")
            nextID = modelDB.dbCursor.fetchall()[0][0]
            modelDB.dbCursor.execute(
                """INSERT INTO classification_ref
                   (id, name, description, is_root, is_leaf, revision_id)
                   VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N',
                           :rev_arg)""",
                id_arg=nextID, rev_arg=mdlRevision,
                name_arg="Industry Groups".encode('utf-8'),
                desc_arg="Industry Groups".encode('utf-8'))
            mdlIGRParent = modelDB.getMdlClassificationByID(nextID)
            mktToMdlMap[mktRoot.id] = mdlIGRParent.id
        
        if self.lowestLevel == 'Industry':
            # create industry parent
            modelDB.dbCursor.execute("""SELECT classification_seq.nextval
            FROM DUAL""")
            nextID = modelDB.dbCursor.fetchall()[0][0]
            modelDB.dbCursor.execute("""INSERT INTO classification_ref
            (id, name, description, is_root, is_leaf, revision_id)
            VALUES(:id_arg, :name_arg, :desc_arg, 'Y', 'N', :rev_arg)""",
                                     id_arg=nextID, rev_arg=mdlRevision,
                                     name_arg="Industries".encode('utf-8'),
                                     desc_arg="Industries".encode('utf-8'))
            mdlINDParent = modelDB.getMdlClassificationByID(nextID)
            mktToMdlMap[mktRoot.id] = mdlINDParent.id
        
        assert(not mktRoot.isLeaf)
        if self.lowestLevel in ('IndustryGroup', 'Industry'):
            sectors = marketDB.getClassificationChildren(mktRoot)
            self.copyClassification(mdlSECParent.id, sectors, marketDB,
                                    modelDB,
                                    mktToMdlMap, 'N', mdlRevision, None)
        if self.lowestLevel == 'Industry':
            industryGroups = []
            for sec in sectors:
                assert(not sec.isLeaf)
                children = marketDB.getClassificationChildren(sec)
                self.copyClassification(mktToMdlMap[sec.id], children,
                                        marketDB, modelDB, mktToMdlMap, 'N',
                                        mdlRevision, mdlIGRParent.id)
                industryGroups.extend(children)
        if self.lowestLevel == 'Sector':
            nextToLowest = [mktRoot]
            codeLength = 2
            lowestParentId = None
        elif self.lowestLevel == 'IndustryGroup':
            nextToLowest = sectors
            lowestParentId = mdlIGRParent.id
            codeLength = 4
        elif self.lowestLevel == 'Industry':
            nextToLowest = industryGroups
            lowestParentId = mdlINDParent.id
            codeLength = 6
        
        # Build list of leaf codes that remain unchanged
        changedCodes = set()
        for codes in self.codeToIndMap.keys():
            changedCodes.update([i[:codeLength] for i in codes])
        for ntl in nextToLowest:
            assert(not ntl.isLeaf)
            children = marketDB.getClassificationChildren(ntl)
            unchangedChildren = [c for c in children
                                 if not c.code in changedCodes]
            if len(unchangedChildren) > 0:
                self.copyClassification(mktToMdlMap[ntl.id], unchangedChildren,
                                        marketDB, modelDB, mktToMdlMap, 'Y',
                                        mdlRevision, lowestParentId)
            changedChildren = [c for c in children if c.code in changedCodes]
            if len(changedChildren) > 0:
                self.copyChangedClassification(
                    mktToMdlMap[ntl.id], changedChildren,
                    marketDB, modelDB, mktToMdlMap, 'Y',
                    mdlRevision, lowestParentId)
    
    def copyChangedClassification(self, parentID, children, marketDB, modelDB,
                           mktToMdlMap, isLeaf, revID, additionalParentID):
        """Internal method to copy the merged children of a MarketDB
        classification into ModelDB with their hierarchy settings.
        """
        valueDicts = []
        hierDicts = []
        childrenCodes = set([c.code for c in children])
        codeLength = len(children[0].code)
        changedIndustries = set([
                industry for (codes, industry) in self.codeToIndMap.items()
                if len(childrenCodes
                       & set([c[:codeLength] for c in codes])) > 0])
        for ind in changedIndustries:
            weight = 1.0
            isLeaf = 'Y'
            modelDB.dbCursor.execute(
                """SELECT classification_seq.nextval FROM DUAL""")
            nextID = modelDB.dbCursor.fetchall()[0][0]
            values = dict([('name_arg', ind),
                           ('id_arg', nextID),
                           ('rev_arg', revID),
                           ('leaf_arg', isLeaf),
                           ('desc_arg', ind)])
            valueDicts.append(values)
            
            hier = dict([('parent_arg', parentID),
                         ('child_arg', nextID),
                         ('weight_arg', weight)])
            hierDicts.append(hier)
            if not additionalParentID is None:
                hier = dict([('parent_arg', additionalParentID),
                             ('child_arg', nextID),
                             ('weight_arg', weight)])
                hierDicts.append(hier)
        
        modelDB.dbCursor.executemany("""INSERT INTO classification_ref
        (id, name, description, is_root, is_leaf, revision_id)
        VALUES(:id_arg, :name_arg, :desc_arg, 'N', :leaf_arg,
        :rev_arg)""", valueDicts)
        modelDB.dbCursor.executemany("""INSERT INTO classification_dim_hier
        (parent_classification_id, child_classification_id, weight)
        VALUES(:parent_arg, :child_arg, :weight_arg)""", hierDicts)

class GICSCustomCN(GICSBasedClassification):
    """Industry classification scheme for China, based on GICS
    IndustryGroups.  Select industry groups have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('1010',): 'Energy ex Coal',
            ('10102050',): 'Coal & Consumable Fuels',
            ('151010','151030'): 'Chemicals',
            ('151020',): 'Construction Materials',
            ('151050',): 'Paper & Forest Products',
            ('151040',): 'Metals & Mining ex Steel',
            ('15104050',): 'Steel',
            ('201030',): 'Construction & Engineering',
            ('201010','201060'): 'Machinery',
            ('201020','201040'): 'Electrical Equipment',
            ('201050','201070'): 'Trading Companies, Distributors & Conglomerates',
            ('203010','203020','203030'): 'Transportation Non-Infrastructure',
            ('203040','203050'): 'Transportation Infrastructure',
            ('251010',): 'Auto Components',
            ('251020',): 'Automobiles',
            ('252010',): 'Household Durables',
            ('252020','252030'): 'Textiles, Apparel & Luxury Goods',
            ('253010','253020'): 'Consumer Services',
            ('301010','302010','302030','303010','303020'): 'Beverages & Tobacco',
            ('302020',): 'Food Products',
            ('3510','3520'): 'Health Care',
            ('4010','4020','4030'): 'Financials',
            ('452030','452040','4530'): 'Semiconductors & Electronics',
            ('452010',): 'Communications Equipment',
            ('452020',): 'Computers & Peripherals',
            ('5510',): 'Utilities',
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-CN', date, codeToIndMap,
                            'IndustryGroup')

class GICSCustomCA(GICSBasedClassification):
    """Industry classification scheme for Canada, based on GICS
    IndustryGroups.  Select industry groups have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010', '151020', '151030'):
                'Materials ex Metals, Mining & Forestry',
            ('151040',): 'Metals & Mining ex Gold',
            ('15104030',): 'Gold',
            ('151050',): 'Paper & Forest Products',
            ('2510', '2520', '2530', '2550'):
                'Consumer Discretionary ex Media',
            ('3020', '3030'): 'Food & Staples Products',
            ('3510', '3520'): 'Health Care',
            ('4520', '4530'): 'Technology Hardware'
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-CA', date, codeToIndMap,
                            'IndustryGroup')

class GICSCustomCA2(GICSBasedClassification):
    """Industry classification scheme for Canada, based on GICS
    2018 IndustryGroups.   This is comparable to GICSCustomCA,
    except that Media is no longer explicitly removed from 
    Consumer Discretionary since it is no longer part of 
    that sector in GICS 2018
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010', '151020', '151030'):
                'Materials ex Metals, Mining & Forestry',
            ('151040',): 'Metals & Mining ex Gold',
            ('15104030',): 'Gold',
            ('151050',): 'Paper & Forest Products',
            ('2510', '2520', '2530', '2550'):
                'Consumer Discretionary',
            ('3020', '3030'): 'Food & Staples Products',
            ('3510', '3520'): 'Health Care',
            ('4520', '4530'): 'Technology Hardware'
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-CA2', date, codeToIndMap,
                            'IndustryGroup')

class GICSCustomCA3(GICSBasedClassification):
    """Industry classification scheme for Canada, based on GICS
    IndustryGroups.  Select industry groups have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('2010','2020', '2030'): 'Industrials',
            ('2510', '2520', '2530', '2550'): 'Consumer Discretionary',
            ('3010', '3020', '3030'): 'Consumer Staples',
            ('3510', '3520'): 'Health Care',
            ('4010', '4020', '4030'): 'Financials',
            ('4510', '4520', '4530'): 'Information Technology',
            ('5010', '5020'): 'Communication Services',
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-CA3', date, codeToIndMap,
                            'IndustryGroup')


class GICSCustomCA4(GICSBasedClassification):
    """Industry classification scheme for Canada, based on GICS
    2018 IndustryGroups.   This is comparable to GICSCustomCA,
    except that Media is no longer explicitly removed from 
    Consumer Discretionary since it is no longer part of 
    that sector in GICS 2018
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010', '151020', '151030'):
                'Materials ex Metals, Mining & Forestry',
            ('151040',): 'Metals & Mining ex Gold',
            ('15104030',): 'Gold',
            ('151050',): 'Paper & Forest Products',
            ('2510', '2520', '2530', '2550'):
                'Consumer Discretionary',
            ('3020', '3030'): 'Food & Staples Products',
            ('3510', '3520'): 'Health Care',
            ('4520', '4530'): 'Technology Hardware',
            ('101010',): 'Energy Equipment & Services',
            ('101020',): 'Oil, Gas & Consumable Fuels',
            ('601010',): 'Equity Real Estate Investment Trusts (REITs)',
            ('601020',): 'Real Estate Management & Development',
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-CA4', date, codeToIndMap,
                            'IndustryGroup')

class GICSCustomAU(GICSBasedClassification):
    """Industry classification scheme for Australia, based on GICS
    IndustryGroups.  Select industry groups have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010', '151020', '151030', '151050'):
                'Materials ex Metals & Mining',
            ('151040',): 'Metals & Mining ex Gold & Steel',
            ('15104030',): 'Gold',
            ('15104050',): 'Steel',
            ('2510', '2520', '2530', '2550'):
                'Consumer Discretionary ex Media',
            ('3020', '3030'): 'Food & Staples Products',
            ('3510', '3520'): 'Health Care',
            ('4510', '4520', '4530'): 'Information Technology',
            ('404020',): 'Real Estate Investment Trusts (REITs)',
            ('4020', '404030'): 'Diversified Financials'
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-AU', date, codeToIndMap,
                            'IndustryGroup')

class GICSCustomJP(GICSBasedClassification):
    """Industry classification scheme for Japan, based on GICS
    Industries.  Select industries have been merged to better 
    reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS Industry Groups.
    """
    def __init__(self, date):
        if date < datetime.date(2008,8,30):
            codeToIndMap = {
                ('101010', '101020'): 'Energy',
                ('201010', '201060'): 'Machinery',
                ('251010', '251020'): 'Automobiles & Components',
                ('302010', '302030'): 'Beverages & Tobacco',
                ('303010', '303020'): 'Household & Personal Products',
                ('351010', '351030'): 'Health Care Equipment & Technology',
                ('352010', '352030'): 'Biotechnology & Life Sciences',
                ('401010', '401020'): 'Banks',
                ('501010', '501020'): 'Telecommunication Services',
                ('551010', '551020',
                 '551030', '551040', '551050'): 'Utilities'
                }
        elif date < datetime.date(2016,9,1):
            codeToIndMap = {
                ('101010', '101020'): 'Energy',
                ('201010', '201060'): 'Machinery',
                ('302010', '302030'): 'Beverages & Tobacco',
                ('303010', '303020'): 'Household & Personal Products',
                ('351010', '351030'): 'Health Care Equipment & Technology',
                ('352010', '352030'): 'Biotechnology & Life Sciences',
                ('401010', '401020'): 'Banks',
                ('501010', '501020'): 'Telecommunication Services',
                ('551010', '551030', '551040', '551050'): 'Electric Utilities',
                }
        else:
            codeToIndMap = {
                ('101010', '101020'): 'Energy',
                ('201010', '201060'): 'Machinery',
                ('203010', '203020'): 'Air Freight & Airlines',
                ('302010', '302020', '302030'): 'Food Beverage & Tobacco',
                ('303010', '303020'): 'Household & Personal Products',
                ('351010', '351030'): 'Health Care Equipment & Technology',
                ('401010', '401020'): 'Banks',
                ('402010', '402040'): 'Diversified Financial Services & Mortgage REITs',
                ('501010', '501020'): 'Telecommunication Services',
                ('551010', '551020', '551030', '551040', '551050'): 'Utilities',
                }

        GICSBasedClassification.__init__(
                            self, 'GICSCustom-JP', date, codeToIndMap,
                            'Industry')

class GICSCustomJP2(GICSBasedClassification):
    """Industry classification scheme for Japan, based on GICS
    Industries.  Select industries have been merged to better 
    reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS Industry Groups.
    """
    def __init__(self, date):
        if date < datetime.date(2008,8,30):
            codeToIndMap = {
                ('101010', '101020'): 'Energy',
                ('201010', '201060'): 'Machinery',
                ('251010', '251020'): 'Automobiles & Components',
                ('302010', '302030'): 'Beverages & Tobacco',
                ('303010', '303020'): 'Household & Personal Products',
                ('351010', '351030'): 'Health Care Equipment & Technology',
                ('352010', '352030'): 'Biotechnology & Life Sciences',
                ('401010', '401020'): 'Banks',
                ('501010', '501020'): 'Telecommunication Services',
                ('551010', '551020',
                 '551030', '551040', '551050'): 'Utilities'
                }
        elif date < datetime.date(2016,9,1):
            codeToIndMap = {
                ('101010', '101020'): 'Energy',
                ('201010', '201060'): 'Machinery',
                ('302010', '302030'): 'Beverages & Tobacco',
                ('303010', '303020'): 'Household & Personal Products',
                ('351010', '351030'): 'Health Care Equipment & Technology',
                ('352010', '352030'): 'Biotechnology & Life Sciences',
                ('401010', '401020'): 'Banks',
                ('501010', '501020'): 'Telecommunication Services',
                ('551010', '551030', '551040', '551050'): 'Electric Utilities',
                }
        else:
            codeToIndMap = {
                ('101010', '101020'): 'Energy',
                ('201010', '201060'): 'Machinery',
                ('203010', '203020'): 'Air Freight & Airlines',
                ('302010', '302020', '302030'): 'Food Beverage & Tobacco',
                ('303010', '303020'): 'Household & Personal Products',
                ('351010', '351030'): 'Health Care Equipment & Technology',
                ('401010', '401020'): 'Banks',
                ('501010', '501020'): 'Telecommunication Services',
                ('402010', '402040'): 'Diversified Financial Services & Mortgage REITs',
                ('551010', '551020', '551030', '551040', '551050'): 'Utilities',
                }

        GICSBasedClassification.__init__(
                            self, 'GICSCustom-JP2', date, codeToIndMap,
                            'Industry')

class GICSCustomGB(GICSBasedClassification):
    """Industry classification scheme for the GB, based on GICS
    Industries.  Select industries have been merged to better 
    reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS Industry Groups.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('101010', '101020'): 'Energy',
            ('151020', '151040'): 'Metals & Mining',
            ('201020', '201030'): 'Construction & Engineering',
            ('151030', '151050'): 'Forestry, Containers & Packaging',
            ('201040', '201060'): 'Electrical Equipment & Machinery',
            ('203010', '203030', '203040', '203050'): 'Transportation ex Airlines',
            ('251010', '251020'): 'Automobiles & Components',
            ('252020', '252030'): 'Textiles, Apparel & Luxury Goods',
            ('303010', '303020'): 'Household & Personal Products',
            ('351010', '351020', '351030'): 'Health Care Equipment & Services',
            ('352010', '352030'): 'Biotechnology & Life Sciences',
            ('255010', '255020', '255030'): 'General Retail',
            ('253010', '253020'): 'Consumer Services',
            ('501010', '501020'): 'Telecommunication Services',
            ('551010', '551020', '551030', '551040', '551050'): 'Utilities',
            ('401010', '401020'): 'Banks',
            ('402010', '402020'): 'Diversified Financial Services',
            ('451010', '451020'): 'Internet & IT Services',
            ('452010', '452020', '452040'): 'Technology Hardware'
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-GB', date, codeToIndMap,
                            'Industry')

class GICSCustomGB4(GICSBasedClassification):
    """Industry classification scheme for the UK4 models, based on GICS Industry Groups.
    Select groups have been split to better reflect the distribution of assets across
    sectors in the local market.  No merged entity spans multiple GICS Sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010',): 'Chemicals',
            ('151020',): 'Construction Materials',
            ('151040',): 'Metals & Mining',
            ('151030', '151050'): 'Other Materials and Packaging',
            ('201010',): 'Aerospace & Defense',
            ('201020',): 'Building Products',
            ('201030',): 'Construction & Engineering',
            ('201040',): 'Electrical Equipment',
            ('201050', '201060'): 'Other Industrials',
            ('201070',): 'Trading Companies & Distributors',
            ('202010',): 'Commercial Services & Supplies',
            ('202020',): 'Professional Services',
            ('255010', '255030', '255040'): 'Non-Internet Retail',
            ('255020',): 'Internet & Direct Marketing Retail',
            ('302010', '302030'): 'Beverages & Tobacco',
            ('302020',): 'Food Products',
            ('352010', '352030'): 'Biotechnology & Life Sciences',
            ('352020',): 'Pharmaceuticals',
            ('402010', '402040'): 'Diversified Financial Services',
            ('402020',): 'Consumer Finance',
            ('402030',): 'Capital Markets',
            ('451020',): 'IT Services',
            ('451030',): 'Software',
            ('4520', '4530'): 'Technology Hardware',
            ('551010', '551020', '551030', '551040'): 'Utilities',
            ('551050',): 'Independent Power and Renewable Electricity Producers',
            ('601010',): 'Equity Real Estate Investment Trusts (REITs)',
            ('601020',): 'Real Estate Management & Development',
            }
        GICSBasedClassification.__init__(self, 'GICSCustom-GB4', date, codeToIndMap, 'IndustryGroup')

class GICSCustomNA(GICSBasedClassification):
    """Industry classification scheme for North America, based on GICS
    Industries.  Select industries have been merged to better
    reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS Industry Groups.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010', '151020',
             '151030', '151050'): 'Materials ex Metals & Mining',
            ('151040',): 'Metals & Mining ex Gold',
            ('15104030',): 'Gold',
            ('201020', '201030',
             '201050', '201060', '201070'): 'Industrial & Machinery',
            ('202010', '202020'): 'Commercial & Professional Services',
            ('203010', '203020', '203030', '203040', '203050'): 'Transportation',
            ('251010', '251020'): 'Automobiles & Components',
            ('252020', '252030'): 'Leisure & Apparel',
            ('253010', '253020'): 'Consumer Services',
            ('255010', '255020', '255030', '255040'): 'Retailing',
            ('302010', '302020', '302030'): 'Food, Beverage & Tobacco',
            ('303010', '303020'): 'Household & Personal Products',
            ('351010', '351030'): 'Health Care Equipment & Technology',
            ('352010', '352030'): 'Biotechnology & Life Sciences',
            ('402010', '402020', '402030'): 'Diversified Financials',
            ('404020', '404030'): 'Real Estate',
            ('452030', '452040'): 'Electronic Equipment, Instruments & Components',
            ('501010', '501020'): 'Telecommunication Services',
            ('551010', '551020',
             '551030', '551040', '551050'): 'Utilities'
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-NA', date, codeToIndMap,
                            'Industry')


class GICSCustomNA4(GICSBasedClassification):
    """Industry classification scheme for North America, based on GICS
    2018 IndustryGroups. This is comparable to GICS 2018,
    except that Gold is explicitly removed from 
    Metals & Mining since it by itself is a strong factor
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151040',): 'Metals & Mining ex Gold',
            ('15104030',): 'Gold',
            }
        GICSBasedClassification.__init__(
                            self, 'GICSCustom-NA4', date, codeToIndMap,
                            'Industry')

class GICSCustomUS(GICSBasedClassification):
    def __init__(self, date):
        codeToIndMap = {
                ('151010','151020'): 'Chemicals & Construction Materials',
                ('201050','201060'): 'Industrial Conglomerates & Machinery',
                ('203010','203020','203050'): 'Airline, Air Freight & Transportation Infrastructure',
                ('251010','251020'): 'Automobiles & Components',
                ('255010','255030'): 'Distributors & Multiline Retail',
                ('302010','302030'): 'Beverages & Tobacco',
                ('303010','303020'): 'Household & Personal Products',
                ('452020','452040'): 'Computers & Peripherals',
                ('501010','501020'): 'Telecommunication Services'
                }
        GICSBasedClassification.__init__(
                self, 'GICSCustom-US', date, codeToIndMap, 'Industry')

class GICSCustomNoOE(GICSBasedClassification):
    def __init__(self, date):
        codeToIndMap = {
                ('452020','452040'): 'Computers & Peripherals',
                }
        GICSBasedClassification.__init__(
                self, 'GICSCustom-NoOE', date, codeToIndMap, 'Industry')

class GICSCustomNoMortgageREITs(GICSBasedClassification):
    def __init__(self, date):
        codeToIndMap = {
                ('402010', '402040'): 'Diversified Financial Services & Mortgage REITs',
                }
        GICSBasedClassification.__init__(
                self, 'GICSCustom-NoMortgageREITs', date, codeToIndMap, 'Industry')

class GICSCustomNoMortgageREITs2018(GICSBasedClassification):
    def __init__(self, date):
        codeToIndMap = {
                ('402010', '402040'): 'Diversified Financial Services & Mortgage REITs',
                }
        GICSBasedClassification.__init__(
                self, 'GICSCustom-NoMortgageREITs2018', date, codeToIndMap, 'Industry')

class GICSIndustriesGold(GICSBasedClassification):
    """Industry classification scheme based on GICS
    Industries, with Gold industry added.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151040',): 'Metals & Mining ex Gold',
            ('15104030',): 'Gold',
            }
        GICSBasedClassification.__init__(
                            self, 'GICSIndustries-Gold', date, codeToIndMap,
                            'Industry')

class GICSCustomSubIndustries(GICSBasedClassification):
    def __init__(self, date):
        codeToIndMap = {
            ('10101010',): 'Oil & Gas Drilling',
            ('10101020',): 'Oil & Gas Equipment & Services',
            ('10102010',): 'Integrated Oil & Gas',
            ('10102020',): 'Oil & Gas Exploration & Production',
            ('10102030',): 'Oil & Gas Refining & Marketing',
            ('10102040',): 'Oil & Gas Storage & Transportation',
            ('10102050',): 'Coal & Consumable Fuels',
            ('15101010',): 'Commodity Chemicals',
            ('15101020',): 'Diversified Chemicals',
            ('15101030',): 'Fertilizers & Agricultural Chemicals',
            ('15101040',): 'Industrial Gases',
            ('15101050',): 'Specialty Chemicals',
            ('15102010',): 'Construction Materials',
            ('15103010',): 'Metal & Glass Containers',
            ('15103020',): 'Paper Packaging',
            ('15104010',): 'Aluminum',
            ('15104020',): 'Diversified Metals & Mining',
            ('15104030',): 'Gold',
            ('15104040',): 'Precious Metals & Minerals',
            ('15104050',): 'Steel',
            ('15105010',): 'Forest Products',
            ('15105020',): 'Paper Products',
            ('20101010',): 'Aerospace & Defense',
            ('20102010',): 'Building Products',
            ('20103010',): 'Construction & Engineering',
            ('20104010',): 'Electrical Components & Equipment',
            ('20104020',): 'Heavy Electrical Equipment',
            ('20105010',): 'Industrial Conglomerates',
            ('20106010',): 'Construction & Farm Machinery & Heavy Trucks',
            ('20106020',): 'Industrial Machinery',
            ('20107010',): 'Trading Companies & Distributors',
            ('20201010',): 'Commercial Printing',
            ('20201050',): 'Environmental & Facilities Services',
            ('20201060',): 'Office Services & Supplies',
            ('20201070',): 'Diversified Support Services',
            ('20201080',): 'Security & Alarm Services',
            ('20202010',): 'Human Resource & Employment Services',
            ('20202020',): 'Research & Consulting Services',
            ('20301010',): 'Air Freight & Logistics',
            ('20302010',): 'Airlines',
            ('20303010',): 'Marine',
            ('20304010',): 'Railroads',
            ('20304020',): 'Trucking',
            ('20305010',): 'Airport Services',
            ('20305020',): 'Highways & Railtracks',
            ('20305030',): 'Marine Ports & Services',
            ('25101010',): 'Auto Parts & Equipment',
            ('25101020',): 'Tires & Rubber',
            ('25102010',): 'Automobile Manufacturers',
            ('25102020',): 'Motorcycle Manufacturers',
            ('25201010',): 'Consumer Electronics',
            ('25201020',): 'Home Furnishings',
            ('25201030',): 'Homebuilding',
            ('25201040',): 'Household Appliances',
            ('25201050',): 'Housewares & Specialties',
            ('25202010',): 'Leisure Products',
            ('25202020',): 'Photographic Products',
            ('25203010',): 'Apparel Accessories & Luxury Goods',
            ('25203020',): 'Footwear',
            ('25203030',): 'Textiles',
            ('25301010',): 'Casinos & Gaming',
            ('25301020',): 'Hotels Resorts & Cruise Lines',
            ('25301030',): 'Leisure Facilities',
            ('25301040',): 'Restaurants',
            ('25302010',): 'Education Services',
            ('25302020',): 'Specialized Consumer Services',
            ('25401010',): 'Advertising',
            ('25401020',): 'Broadcasting',
            ('25401025',): 'Cable & Satellite',
            ('25401030',): 'Movies & Entertainment',
            ('25401040',): 'Publishing',
            ('25501010',): 'Distributors',
            ('25502010',): 'Catalog Retail',
            ('25502020',): 'Internet Retail',
            ('25503010',): 'Department Stores',
            ('25503020',): 'General Merchandise Stores',
            ('25504010',): 'Apparel Retail',
            ('25504020',): 'Computer & Electronics Retail',
            ('25504030',): 'Home Improvement Retail',
            ('25504040',): 'Specialty Stores',
            ('25504050',): 'Automotive Retail',
            ('25504060',): 'Homefurnishing Retail',
            ('30101010',): 'Drug Retail',
            ('30101020',): 'Food Distributors',
            ('30101030',): 'Food Retail',
            ('30101040',): 'Hypermarkets & Super Centers',
            ('30201010',): 'Brewers',
            ('30201020',): 'Distillers & Vintners',
            ('30201030',): 'Soft Drinks',
            ('30202010',): 'Agricultural Products',
            ('30202030',): 'Packaged Foods & Meats',
            ('30203010',): 'Tobacco',
            ('30301010',): 'Household Products',
            ('30302010',): 'Personal Products',
            ('35101010',): 'Health Care Equipment',
            ('35101020',): 'Health Care Supplies',
            ('35102010',): 'Health Care Distributors',
            ('35102015',): 'Health Care Services',
            ('35102020',): 'Health Care Facilities',
            ('35102030',): 'Managed Health Care',
            ('35103010',): 'Health Care Technology',
            ('35201010',): 'Biotechnology',
            ('35202010',): 'Pharmaceuticals',
            ('35203010',): 'Life Sciences Tools & Services',
            ('40101010',): 'Diversified Banks',
            ('40101015',): 'Regional Banks',
            ('40102010',): 'Thrifts & Mortgage Finance',
            ('40201020',): 'Other Diversified Financial Services',
            ('40201030',): 'Multi-Sector Holdings',
            ('40201040',): 'Specialized Finance',
            ('40202010',): 'Consumer Finance',
            ('40203010',): 'Asset Management & Custody Banks',
            ('40203020',): 'Investment Banking & Brokerage',
            ('40203030',): 'Diversified Capital Markets',
            ('40301010',): 'Insurance Brokers',
            ('40301020',): 'Life & Health Insurance',
            ('40301030',): 'Multi-line Insurance',
            ('40301040',): 'Property & Casualty Insurance',
            ('40301050',): 'Reinsurance',
            ('40402010',): 'Diversified REITs',
            ('40402020',): 'Industrial REITs',
            ('40402030',): 'Mortgage REITs',
            ('40402040',): 'Office REITs',
            ('40402050',): 'Residential REITs',
            ('40402060',): 'Retail REITs',
            ('40402070',): 'Specialized REITs',
            ('40403010',): 'Diversified Real Estate Activities',
            ('40403020',): 'Real Estate Operating Companies',
            ('40403030',): 'Real Estate Development',
            ('40403040',): 'Real Estate Services',
            ('45101010',): 'Internet Software & Services',
            ('45102010',): 'IT Consulting & Other Services',
            ('45102020',): 'Data Processing & Outsourced Services',
            ('45103010',): 'Application Software',
            ('45103020',): 'Systems Software',
            ('45103030',): 'Home Entertainment Software',
            ('45201020',): 'Communications Equipment',
            ('45202010',): 'Computer Hardware',
            ('45202020',): 'Computer Storage & Peripherals',
            ('45203010',): 'Electronic Equipment & Instruments',
            ('45203015',): 'Electronic Components',
            ('45203020',): 'Electronic Manufacturing Services',
            ('45203030',): 'Technology Distributors',
            ('45204010',): 'Office Electronics',
            ('45301010',): 'Semiconductor Equipment',
            ('45301020',): 'Semiconductors',
            ('50101010',): 'Alternative Carriers',
            ('50101020',): 'Integrated Telecommunication Services',
            ('50102010',): 'Wireless Telecommunication Services',
            ('55101010',): 'Electric Utilities',
            ('55102010',): 'Gas Utilities',
            ('55103010',): 'Multi-Utilities',
            ('55104010',): 'Water Utilities',
            ('55105010',): 'Independent Power Producers & Energy Traders'
            }

        GICSBasedClassification.__init__(
                self, 'GICSCustom-SubInd', date, codeToIndMap, 'Industry')

class GICSCustomTW(GICSBasedClassification):
    """Industry classification scheme for Taiwan, based on GICS
    IndustryGroups.  Select industry groups have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010',):'Chemicals',
            ('151020', '151030', '151040', '151050',):'Materials ex Chemicals',
            ('201010','201060','201070','201020','201030','201050',):\
                'Capital Goods ex Electrical Equipment',
            ('201040',):'Electrical Equipment',
            ('2510','2540','2550',):'Consumer Discretionary ex Durables & Services',
            ('3010','3020','3030',):'Consumer Staples',
            ('3510','3520',):'Health Care',
            ('452010',):'Communications Equipment',
            ('45202010',):'Computer Hardware',
            ('45202020',):'Computer Storage & Peripherals',
            ('452030','452040',):'Electronics',
            ('45301010',):'Semiconductor Equipment',
            ('45301020',):'Semiconductors',
            }

        GICSBasedClassification.__init__(
                            self, 'GICSCustom-TW', date, codeToIndMap,
                            'IndustryGroup')

class CommoditiesEmpty(GICSBasedClassification):
    def __init__(self, date):
        GICSBasedClassification.__init__(
            self, 'CommoditiesEmpty', date, {},
            'IndustryGroup')

class GICSCustomAU2(GICSBasedClassification):
    """Industry classification scheme for Australia, based on GICS
    IndustryGroups.  Select industry groups have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
            ('151010', '151020', '151030', '151050'):'Materials ex Metals & Mining',
            ('151040',): 'Metals & Mining ex Gold & Steel',
            ('15104030',): 'Gold',
            ('15104050',): 'Steel',
            ('2510', '2520', '2530', '2550'):'Consumer Discretionary ex Media',
            ('3010','3020', '3030'): 'Consumer Staples',
            ('3510', '3520'): 'Health Care',
            ('4510', '4520', '4530'): 'Information Technology',
            ('402010','402020','402030','402040'): 'Diversified Financials',
            ('601020',): 'Real Estate Management & Development',
            ('601010',): 'Equity Real Estate Investment Trusts (REITs)'
            }
        GICSBasedClassification.__init__(self, 'GICSCustom-AU2', date, codeToIndMap,'IndustryGroup')

class GICSCustomEU(GICSBasedClassification):
    """Industry classification scheme for EU4, based on GICS Industries
    """
    def __init__(self, date):
        codeToIndMap = {
            ('202010', '202020'): 'Commercial & Professional Services',
            ('253010', '253020'): 'Consumer Services',
            ('255030', '255040'): 'Multiline & Specialty Retail',
            ('302010', '302030'): 'Beverages & Tobacco',
            ('303010', '303020'): 'Household & Personal Products',
            ('401010', '401020'): 'Banks',
            ('402010', '402020', '402040'): 'Diversified Financial Services',
            ('551020', '551030', '551040'): 'Gas, Water & Multi-Utilities'
           }
        GICSBasedClassification.__init__(self, 'GICSCustom-EU', date, codeToIndMap, 'Industry')

class GICSCustomEM(GICSBasedClassification):
    """Industry classification scheme for EM Model, based on GICS
    Industries.  Select industries have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
                ('202010','202020'): 'Commercial & Professional Services',
                ('351010','351030'): 'Health Care Equipment, Supplies & Technology',
                ('352010','352030'): 'Biotechnology, Life Sciences Tools & Services',
                ('401010','401020'): 'Banks, Thrifts & Mortgage Finance',
                ('402010','402040'): 'Diversified Financial Services & Mortgage REITs',
                ('551030','551050'): 'Multi-Utilities'
            }
        GICSBasedClassification.__init__(self, 'GICSCustom-EM', date, codeToIndMap,'Industry')

class GICSCustomEM2(GICSBasedClassification):
    """Industry classification scheme for EM Model, based on GICS
    Industries.  Select industries have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
                ('202010','202020'): 'Commercial & Professional Services',
                ('351010','351030'): 'Health Care Equipment, Supplies & Technology',
                ('352010','352030'): 'Biotechnology, Life Sciences Tools & Services',
                ('401010','401020'): 'Banks, Thrifts & Mortgage Finance',
                ('402010','402040'): 'Diversified Financial Services & Mortgage REITs',
                ('551030','551050'): 'Multi-Utilities & Independent Power Producers & Traders'
            }
        GICSBasedClassification.__init__(self, 'GICSCustom-EM2', date, codeToIndMap,'Industry')

class GICSCustomCN2(GICSBasedClassification):
    """Industry classification scheme for CN4 Model, based on GICS
    Industries.  Select industries have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
                        ('151020',):'Construction Materials',
                        ('151050',):'Paper & Forest Products',
                        ('201010',):'Aerospace & Defense',
                        ('201030',):'Construction & Engineering',
                        ('201060',):'Machinery',
                        ('251010',):'Auto Components',
                        ('251020',): 'Automobiles',
                        ('252010',):'Household Durables',
                        ('302020',):'Food Products',
                        ('451010',):'Internet Software & Services',
                        ('451020',):'IT Services',
                        ('451030',):'Software',
                        ('452010',):'Communications Equipment',
                        ('452020',):'Technology Hardware, Storage & Peripherals',
                        ('452030',):'Electronic Equipment, Instruments & Components',
                        ('453010',):'Semiconductors & Semiconductor Equipment',
                        ('551050',):'Independent Power and Renewable Electricity Producers',
                        ('10102050',):  'Coal & Consumable Fuels',
                        ('101010','101020'):      'Energy ex Coal',
                        ('15104050',):  'Steel',
                        ('151040',):    'Metals & Mining ex Steel',
                        ('151010','151030'):    'Chemicals',
                        ('201020','201040'):    'Electrical Equipment',
                        ('201050','201070'):    'Trading Companies, Distributors & Conglomerates',
                        ('202010','202020'):      'Commercial & Professional Services',
                        ('203010','203020','203030'): 'Transportation Non-Infrastructure',
                        ('203040','203050'): 'Transportation Infrastructure',
                        ('252020','252030'): 'Textiles, Apparel & Luxury Goods',
                        ('253010','253020'): 'Consumer Services',
                        ('255010','255020','255030','255040'):  'Retailing',
                        ('302010','302030'): 'Beverages & Tobacco',
                        ('303010','303020'): 'Household & Personal Products',
                        ('351010','351020','351030'):  'Health Care Equipment & Services',
                        ('352010','352020','352030'):  'Pharmaceuticals Biotechnology & Life Sciences',
                        ('401010','401020'):    'Banks',
                        ('402010','402020','402030','402040','403010'):    'Financials ex Banks',
                        ('501010','501020'):  'Telecommunication Services',
                        ('551010','551020','551030','551040'):'Utilities ex Renewable',
                        ('601010','601020'):  'Real Estate'
                        }
        GICSBasedClassification.__init__(self, 'GICSCustom-CN2', date, codeToIndMap,'IndustryGroup')

class GICSCustomCN2b(GICSBasedClassification):
    """Industry classification scheme for CN4 Model, based on GICS
    Industries.  Select industries have been split or merged
    to better reflect the distribution of assets across sectors in the
    local market.  No merged entity spans multiple GICS sectors.
    """
    def __init__(self, date):
        codeToIndMap = {
                        ('151020',):'Construction Materials',
                        ('151050',):'Paper & Forest Products',
                        ('201010',):'Aerospace & Defense',
                        ('201030',):'Construction & Engineering',
                        ('201060',):'Machinery',
                        ('251010',):'Auto Components',
                        ('251020',): 'Automobiles',
                        ('252010',):'Household Durables',
                        # ('254010','502010','502020','502030'):'Media & Entertainment',
                        ('254010','502010','502030'):'Media',
                        ('502020',):'Entertainment',
                        ('302020',):'Food Products',
                        # ('451010',):'Internet Software & Services', this factor should be removed in GICS2018
                        ('451020',):'IT Services',
                        ('451030',):'Software',
                        ('452010',):'Communications Equipment',
                        ('452020',):'Technology Hardware, Storage & Peripherals',
                        ('452030',):'Electronic Equipment, Instruments & Components',
                        ('453010',):'Semiconductors & Semiconductor Equipment',
                        ('551050',):'Independent Power and Renewable Electricity Producers',
                        ('10102050',):  'Coal & Consumable Fuels',
                        ('101010','101020'):      'Energy ex Coal',
                        ('15104050',):  'Steel',
                        ('151040',):    'Metals & Mining ex Steel',
                        ('151010','151030'):    'Chemicals',
                        ('201020','201040'):    'Electrical Equipment',
                        ('201050','201070'):    'Trading Companies, Distributors & Conglomerates',
                        ('202010','202020'):      'Commercial & Professional Services',
                        ('203010','203020','203030'): 'Transportation Non-Infrastructure',
                        ('203040','203050'): 'Transportation Infrastructure',
                        ('252020','252030'): 'Textiles, Apparel & Luxury Goods',
                        ('253010','253020'): 'Consumer Services',
                        # ('255010','255020','255030','255040','451010'):  'Retailing',
                        ('255010','255020','255030','255040'):  'Retailing',
                        ('302010','302030'): 'Beverages & Tobacco',
                        ('303010','303020'): 'Household & Personal Products',
                        ('351010','351020','351030'):  'Health Care Equipment & Services',
                        ('352010','352020','352030'):  'Pharmaceuticals Biotechnology & Life Sciences',
                        ('401010','401020'):    'Banks',
                        ('402010','402020','402030','402040','403010'):    'Financials ex Banks',
                        ('501010','501020'):  'Telecommunication Services',
                        ('551010','551020','551030','551040'):'Utilities ex Renewable',
                        ('601010','601020'):  'Real Estate'
                        }
        GICSBasedClassification.__init__(self, 'GICSCustom-CN2b', date, codeToIndMap,'IndustryGroup')

import itertools
import pymssql
import pandas
import pandas as pd

daymap = {'Y': 365, 'M': 31, 'W': 7, 'D': 1}

curve_categories = ['lc', 'll', 'sl', 'sc', 'scb']

def nodeNameKey(node):
    if isinstance(node, str):
        nodename = node
    else:
        nodename = node.name
    tenoridx = nodename.rfind('.')
    if tenoridx == -1:
        tenor = nodename
        curvename = ''
    else:
        tenor = nodename[tenoridx+1:]
        curvename = nodename[:tenoridx]
    unit = tenor[-1].upper()
    if unit not in daymap:
        raise Exception('Could not find unit %s in daymap' % unit)
    days = daymap[unit] * int(tenor[:-1])
    return (curvename, days)

class CurveLoader(object):
    def __init__(self, macDB):
        port = macDB['port']
        host = macDB['host']
        database= macDB['database']
        dbUser= macDB['dbUser']
        dbPass= macDB['dbPass']
        self.cn_ = pymssql.connect(database=database, host=host, user=dbUser, password=dbPass, port = port)
        self.cn_.autocommit(True)
        self.cursor_ = self.cn_.cursor()

    def _buildResult(self, cols):
        result = []
        for row in self.cursor_:
            result.append(Struct(dict(zip(cols, row))))
        return result

    def getAllCurves(self):
        query = """
        SELECT
        CurveShortName, CurveLongName, CurveTypeEnum, CountryEnum, CurrencyEnum 
        from marketdata.dbo.Curve C
        """

        self.cursor_.execute(query)
        cols = ['name', 'description', 'type', 'country', 'currency']
        return self._buildResult(cols)

    def getAllCurveNodes(self, name):
        query = """
        SELECT
        DISTINCT cn.NodeShortName, cn.tenorenum
        from marketdata.dbo.Curve C 
        JOIN marketdata.dbo.CurveNodes cn ON C.CurveId=cn.CurveId
        where C.CurveShortName=%s
        """
        
        self.cursor_.execute(query, (name,))
        cols = ['name', 'tenor']
        return sorted(self._buildResult(cols), key=nodeNameKey)

    def getAllFactorNodes(self):
        query = """
        SELECT
        DISTINCT C.CurveShortName, cn.tenorenum
        FROM marketdata.dbo.DerivCurveUniversalModel dc
        JOIN marketdata.dbo.Curve C ON dc.CurveShortName = C.CurveShortName
        JOIN marketdata.dbo.CurveNodes cn ON C.CurveId=cn.CurveId
        WHERE cn.TenorEnum IN ('6M', '1Y', '2Y', '5Y', '10Y', '30Y') and dc.SpecificRisk is null
        """

        self.cursor_.execute(query)
        cols = ['name', 'tenor']
        return self._buildResult(cols)

    def getNodesHistory(self, names, start=None, end=None):

        orignames = names
        if not isinstance(names[0], str):
            names = [n.name for n in names]
        if start is None and end is None:
            query = """SELECT
                cn.NodeShortName, cq.TradeDate, cq.Quote
                FROM marketdata.dbo.CurveNodeQuoteFinal cq join marketdata.dbo.CurveNodes cn on cq.CurveNodeId=cn.CurveNodeId
                WHERE cn.NodeShortName in %s 
            """
            self.cursor_.execute(query, (tuple(names), ))
        else:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            query = """SELECT
                cn.NodeShortName, cq.TradeDate, cq.Quote
                FROM marketdata.dbo.CurveNodeQuoteFinal cq join marketdata.dbo.CurveNodes cn on cq.CurveNodeId=cn.CurveNodeId
                WHERE cn.NodeShortName in %s and cq.TradeDate >= %s and cq.TradeDate <= %s
            """
            self.cursor_.execute(query, (tuple(names), start_str, end_str))

        df = pandas.DataFrame(self.cursor_.fetchall(), columns=['name', 'dt', 'value']).pivot('dt', 'name', 'value').reindex(columns=names)
        df.columns = orignames
        return df

class CreditCurveLoader(object):
    def __init__(self, macDB):
        port = macDB['port']
        host = macDB['host']
        database= macDB['database']
        dbUser= macDB['dbUser']
        dbPass= macDB['dbPass']
        self.cn_ = pymssql.connect(database=database, host=host, user=dbUser, password=dbPass, port = port)
        self.cn_.autocommit(True)
        self.cursor_ = self.cn_.cursor()

    def getClusterCurveNames(self, pattern):
        query = """
        select * 
        from AxCurve.dbo.xcClusterCurveUniverse where CurveShortName like %(pattern)s
        """
        df = pd.read_sql_query(query, self.cn_, params={'pattern': pattern})
        return df

    def getIssuerCurveNames(self, pattern):
        query = """
        select * 
        from AxCurve.dbo.xcIssuerCurveUniverse where CurveShortName like %(pattern)s
        """
        df = pd.read_sql_query(query, self.cn_, params={'pattern': pattern})
        return df

    def getAxiomaDataId(self, curveName):
        query = "select AxiomaDataId from AxCurve.dbo.xcClusterCurveUniverse where CurveShortName=%s"
        self.cursor_.execute(query, (curveName,))
        results = self.cursor_.fetchall()
        if len(results) != 0:
            return results[0][0]
        query = "select AxiomaDataId from AxCurve.dbo.xcIssuerCurveUniverse where CurveShortName=%s" + \
                "and isPrimary = 'Y' and ToDate = '9999-12-31'"
        self.cursor_.execute(query, (curveName,))
        results = self.cursor_.fetchall()
        if len(results) != 0:
            return results[0][0]
        return None

    def getCurveNodesHistory(self, curveName, tenors, category="sl", version=5, start=None, end= None):
        if category.lower() not in curve_categories:
            raise Exception('Unknown category: ' + category)
        axid = curveName
        if not isinstance(axid, int):
            axid = self.getAxiomaDataId(curveName)
        if axid is None:
            raise Exception('Unknown curveName: ' + curveName)
        if start is None and end is None:
            query = "SELECT TradeDate, " + ', '.join('[' + x + ']' for x in tenors) + " from AxCurve.dbo.xcCurve where AxiomaDataId=%s" + \
                " and Version = %s" + \
                " and category = %s"
            self.cursor_.execute(query, (axid, version, category.lower()))
        else:
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            query = "SELECT TradeDate, " + ', '.join('[' + x + ']' for x in tenors) + " from AxCurve.dbo.xcCurve where AxiomaDataId=%s" + \
                    " and Version = %s" + \
                    " and category = %s" + \
                    " and TradeDate >= %s" + \
                    " and TradeDate <= %s"

            self.cursor_.execute(query, (axid, version, category.lower(), start_str, end_str))
        df = pandas.DataFrame(self.cursor_.fetchall(), columns=['date'] + tenors).set_index('date')
        df.index = pandas.to_datetime(df.index) 
        return df

class ResearchCurveLoader(object):
    def __init__(self):
        port = '1433'
        host='saturn'
        database='MarketData'
        dbUser='MarketDataLoader'
        dbPass='mdl1234'
        self.cn_ = pymssql.connect(database=database, host=host, user=dbUser, password=dbPass, port = port)
        self.cursor_ = self.cn_.cursor()

    def getHistory(self):
        query = """SELECT TimeInYears, TradeDate, Level FROM MarketData.dbo.ResearchCurves
            where CurveName = 'US.USD.GVT.ZC'
            and Category = 'AxiomaBootStrapper'
            order by TradeDate, TimeInYears
        """
        self.cursor_.execute(query)
        df = pandas.DataFrame(self.cursor_.fetchall(), columns=['tenor', 'dt', 'value'])
        return df

# utility class
class Struct:
    def __init__(self, copy=None):
        if copy is not None:
            if isinstance(copy, dict):
                self.__dict__ = copy.copy()
            else:
                self.__dict__ = dict(copy.__dict__)

            for name in self.__dict__.keys():
                child = self.__dict__[name]
                if isinstance(child, dict):
                    self.__dict__[name] = Struct(copy=child)
                elif isinstance(child, list):
                    self.__dict__[name] = [Struct(x) if isinstance(x, dict) else x for x in child]

    def getFields(self): return list(self.__dict__.values())
    def getFieldNames(self): return list(self.__dict__.keys())
    def setField(self, name, val): self.__dict__[name] = val
    def getField(self, name): return self.__dict__[name]
    def __str__(self):
        return '(%s)' % ', '.join(['%s: %s' % (i, j) for (i,j)
                                  in self.__dict__.items()])
    def __repr__(self):
        return self.__str__()

    def __getitem__(self, name):
        return self.__dict__[name]

    def __setitem__(self, name, val):
        self.__dict__[name] = val

    def __contains__(self, name):
        return (name in self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_dict(self):
        return self.__dict__.copy()

    def update(self,other):
        assert isinstance(other,dict) or isinstance(other,Struct)
        if isinstance(other,dict):
            for k,v in other.items():
                self.__dict__[k]=v
        if isinstance(other,Struct):
            for k in other.getFieldNames():
                self.__dict__[k]=other.getField(k)

    def to_pickle(self,fname):
        import pickle as pkl
        with open(fname, 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
    
    @classmethod
    def from_pickle(cls,fname):
        import pickle as pkl
        with open(fname, 'rb') as f:
            return pkl.load(f)
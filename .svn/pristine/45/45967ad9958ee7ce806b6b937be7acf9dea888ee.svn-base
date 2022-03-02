
import pandas as pd
import os
import sys
import csv
import time
import datetime

class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = '|'
    quotechar = '"'
    quoting = csv.QUOTE_NONE
    escapechar  = '\\'
    
class UnivJoiner:   
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def getCreationTimeLine(self):
        gmtime = time.gmtime(time.mktime(datetime.datetime.now().timetuple()))
        utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
        l = '#CreationTimestamp: %sZ' % utctime.strftime('%Y-%m-%d %H:%M:%S')
        return l
        
    def renameSomeColumns(self, df):
        columnMap = {}
        columnMap['Industry'] = 'EQ-Industry'
        columnMap['Country'] = 'EQ-Country'
        columnMap['Local'] = 'EQ-Local'
        columnMap['Market'] = 'EQ-Market'
        columnMap['Style'] = 'EQ-Style'
        df.rename(columns = columnMap, inplace = True)
    
    def removeExtraColumns(self, df):
        i = 0
        columnsToRemove = []
        for columnType in df.columns.get_level_values(1).values:
#             print columnType
            if 'EQ-' in columnType:
                columnName = df.columns.get_level_values(0).values[i]
                columnsToRemove.append(columnName)
#                 DFB.drop('column_name', axis=1, inplace=True)
            i += 1
        for columnName in columnsToRemove:
            df.drop(columnName, axis=1, inplace=True)
            
    def reorderColumns(self, df):
        print(df.columns.get_level_values(0).tolist())
        print(df.columns.get_level_values(1).tolist())
        res = df.sort_index(axis = 1)
        print(res.columns.get_level_values(0).tolist())
        print(res.columns.get_level_values(1).tolist())
        return res
    
    def join(self, dfA, dfB, resFile):
        self.removeExtraColumns(dfA)
        res = dfA.combine_first(dfB)
        self.renameSomeColumns(res)
        return res
                                
    def doIt(self, date, fileA, fileB, resFile):
        self.date = date
        inA = os.path.join(self.inDir, fileA)
        inB = os.path.join(self.inDir, fileB)

        dfA = self.getDataFrame(inA)
        dfB = self.getDataFrame(inB)

        print(len(dfA), len(dfB))
        print(len(dfA) + len(dfB))
        # print dfA.head()
        dfJoined = self.join(dfA, dfB, resFile)
        print(len(dfJoined))
        print('done joining')
        
        # diff = dfB.index.difference(dfJoined.index)
#         print len(dfJoinedOuter.index)
#         print dfJoinedOuter

#         dfJoined.fillna(value = '', inplace = True) 
#         dfJoined = self.reorderColumns(dfJoined)
#         for each in dfJoinedOuter.columns.get_level_values(0).values:
#             print each        
        self.writeJoined(dfJoined, resFile)

    # def processExclusions(self, date, fileName):
    #     ids = self.getExclusionRows(date)
    #     print len(ids)
    #     inFile = os.path.join(self.inDir, fileName)
    #     df = self.getDataFrame(inFile)
    #     res =  df[df.index.map(lambda x: x not in ids)]
    #     resFile = os.path.join(self.outDir, fileName)
    #     self.writeJoined(res, resFile)

    def getColumns(self, fileName):
        columns1 = []
        columns2 = []
        
        with open(fileName) as f:
            for l in f:
                l = l.strip()
                if '#Columns: ' in l:
                    columns1 = l.split('|')[1:]
                elif '#Type: ' in l:
                    columns2 = l.split('|')[1:]
        return [columns1, columns2]

    def getDataFrame(self, filename):
        table = pd.read_table(filename, sep= '|', skiprows = 4, skip_blank_lines = True, header=None)
        table.set_index(table[0].values, inplace=True)
        table.drop(0, axis=1, inplace=True)
        table.columns = self.getColumns(filename)
        return table

    def writeJoined(self, dfJoined, fileName):  
        outName = os.path.join(self.outDir, fileName)   
        dfJoined.fillna(value = '', inplace = True) 
        dfJoined = self.reorderColumns(dfJoined)     
#         dfJoined.fillna(value = '', inplace = True) 
#         for each in dfJoined.columns.get_level_values(0).values:
#             print each
        with open(outName, 'w') as f:
            writer = csv.writer(f, dialect=my_dialect)
            l = ['#DataDate: %s' % self.date]
            writer.writerow(l)
            l = [self.getCreationTimeLine()]
            writer.writerow(l)            
            l = ['#Columns: AxiomaID']
            l.extend(dfJoined.columns.get_level_values(0).values)
#             for each in dfJoined.columns.get_level_values(0).values:
#                 print each
            writer.writerow(l)
            l = ['#Type: AxiomaID']
            l.extend(dfJoined.columns.get_level_values(1).values)
            writer.writerow(l)
            for index, row in dfJoined.iterrows():
                l = [index]
                l.extend(row.values)
                writer.writerow(l)
                
class UnivJoinerComm(UnivJoiner): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def renameSomeColumns(self, df):
        columnMap = {}
        # columnMap['Style'] = 'CO-Style'
        columnMap['Style'] = 'Commodity'
        df.rename(columns = columnMap, inplace = True)
    
    def removeExtraColumns(self, df):
        columnsToRemove = []
        i = 0
        for columnType in df.columns.get_level_values(1).values:
#             print columnType
            if 'Commodity' in columnType:
                columnName = df.columns.get_level_values(0).values[i]
                columnsToRemove.append(columnName)
            i += 1
        for columnName in columnsToRemove:
            df.drop(columnName, axis=1, inplace=True)

class UnivJoinerIdm(UnivJoiner): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir
    
    def removeExtraColumns(self, df):
        df.drop('CompanyID', axis=1, inplace=True)
        df.drop('Exchange', axis=1, inplace=True)
    
    def renameSomeColumns(self, df):
        columnMap = {}
        columnMap['ISIN'] = 'Client ISIN'
        columnMap['CUSIP'] = 'Client CUSIP'
        columnMap['SEDOL(7)'] = 'Client SEDOL'
        columnMap['Ticker'] = 'Client Ticker'
        df.rename(columns = columnMap, inplace = True)
        
    def reorderColumns(self, df):
        cols = df.columns.tolist()
        print(cols)
        #cols = ['Client ISIN', 'Client CUSIP', 'Client SEDOL', 'Client Ticker', \
        #'Issuer', 'Description', 'Country', 'Currency', 'AssetType']
        cols = ['Client ISIN', 'Client CUSIP', 'Client SEDOL', 'Client Ticker', \
        'Issuer', 'Description', 'Country', 'Currency', 'AssetType', 'Proxy ID']
        return df[cols]

    def getDataFrame(self, filename):
        table = pd.read_table(filename, sep= '|', skiprows = 3, skip_blank_lines = True, header=None)
        table.set_index(table[0].values, inplace=True)
        table.drop(0, axis=1, inplace=True)
        table.columns = self.getColumns(filename)
        return table

    def getColumns(self, fileName):
        columns = []
        
        with open(fileName) as f:
            for l in f:
                l = l.strip()
                if '#Columns: ' in l:
                    columns = l.split('|')[1:]
        return columns  

    def join(self, dfA, dfB, resFile):
        self.renameSomeColumns(dfB)
        self.removeExtraColumns(dfB)
        res = dfA.combine_first(dfB)
        return res
        
    def writeJoined(self, dfJoined, fileName):  
        outName = os.path.join(self.outDir, fileName)  
        dfJoined.fillna(value = '', inplace = True) 
        dfJoined = self.reorderColumns(dfJoined)  
        # dfJoined.drop_duplicates(subset='AxiomaID', inplace=True)

        # print dfJoined.columns.get_level_values(0).values
        with open(outName, 'w') as f:
            writer = csv.writer(f, dialect=my_dialect)
            l = ['#DataDate: %s' % self.date]
            writer.writerow(l)
            l = [self.getCreationTimeLine()]
            writer.writerow(l)
            l = ['#Columns: AxiomaID']
            l.extend(dfJoined.columns.values)
            writer.writerow(l)
            for index, row in dfJoined.iterrows():
                l = [index]
                l.extend(row.values)
#                 print l
                writer.writerow(l)

class UnivJoinerIdmComm(UnivJoinerIdm): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def renameSomeColumns(self, df):
        columnMap = {}
        columnMap['Ticker'] = 'Client Ticker'
        df.rename(columns = columnMap, inplace = True)
        df['AssetType'] = 'Commodity'

    def removeExtraColumns(self, df):
        pass

    def join(self, dfA, dfB, resFile):
        self.renameSomeColumns(dfB)
        self.removeExtraColumns(dfB)
        res = dfA.combine_first(dfB)
        return res

class UnivJoinerAtt(UnivJoinerIdm): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def getDataFrame(self, filename):
        table = pd.read_table(filename, sep= '|', skiprows = 5, skip_blank_lines = True, header=None)
        table.set_index(table[0].values, inplace=True)
        table.drop(0, axis=1, inplace=True)
        table.columns = self.getColumns(filename)
        return table

    def removeExtraColumns(self, df):
        df.drop('1-Day Return', axis=1, inplace=True)
        df.drop('20-Day ADV', axis=1, inplace=True)
        df.drop('Cumulative Return', axis=1, inplace=True)
        df.drop('Historical Beta', axis=1, inplace=True)
        df.drop('Market Cap', axis=1, inplace=True)
        df.drop('RolloverFlag', axis=1, inplace=True)

    def renameSomeColumns(self, df):
        pass

    def join(self, dfA, dfB, resFile):
        self.renameSomeColumns(dfB)
        self.removeExtraColumns(dfB)
        res = dfA.combine_first(dfB)
        return res
        # return pd.concat([dfA, dfB], levels = 0)
      
    def writeJoined(self, dfJoined, fileName):  
        outName = os.path.join(self.outDir, fileName)    
        dfJoined.fillna(value = '', inplace = True) 
        # print dfJoined.columns.get_level_values(0).values
        with open(outName, 'w') as f:
            writer = csv.writer(f, dialect=my_dialect)
            l = ['#DataDate: %s' % self.date]
            writer.writerow(l)
            l = [self.getCreationTimeLine()]
            writer.writerow(l)
            l = ['#Columns: AxiomaID']
            l.extend(dfJoined.columns.values)
            writer.writerow(l)
            l = ['#Type: ID', 'NA', 'Attribute']
            writer.writerow(l)
            l = ['#Unit: ID', 'Text', 'CurrencyPerShare']
            writer.writerow(l)
            for index, row in dfJoined.iterrows():
                l = [index]
                l.extend(row.values)
#                 print l
                writer.writerow(l)

class UnivJoinerAttComm(UnivJoinerAtt): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def removeExtraColumns(self, df):
        df.drop('RolloverFlag', axis=1, inplace=True)
        df.drop('Open Interest', axis=1, inplace=True)
        df.drop('20-Day ADV', axis=1, inplace=True)
        df.drop('1-Day Return', axis=1, inplace=True)
        df.drop('Historical Beta', axis=1, inplace=True)
        df.drop('Cumulative Return', axis=1, inplace=True)
        df.drop('Axioma Series ID', axis=1, inplace=True)
        df.drop('Contract Year', axis=1, inplace=True)
        df.drop('Contract Month', axis=1, inplace=True)
        df.drop('Last Trade Date', axis=1, inplace=True)

    def renameSomeColumns(self, df):
        columnMap = {}
        columnMap['Future Price'] = 'Price'
        df.rename(columns = columnMap, inplace = True)
            
class UnivJoinerRsk(UnivJoinerIdm): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def getDataFrame(self, filename):
        table = pd.read_table(filename, sep= '|', skiprows = 5, skip_blank_lines = True, header=None)
        table.set_index(table[0].values, inplace=True)
        table.drop(0, axis=1, inplace=True)
        table.columns = self.getColumns(filename)
        return table
           
    def removeExtraColumns(self, df):
        df.drop('Estimation Universe', axis=1, inplace=True)
        df.drop('Estimation Universe Weight', axis=1, inplace=True)
        df.drop('Industry Source', axis=1, inplace=True)
        df.drop('Predicted Beta', axis=1, inplace=True)
        df.drop('Specific Return', axis=1, inplace=True)
        df.drop('Total Risk', axis=1, inplace=True)

    def writeJoined(self, dfJoined, fileName):  
        outName = os.path.join(self.outDir, fileName)    
        dfJoined.fillna(value = '', inplace = True) 
        # print dfJoined.columns.get_level_values(0).values
        with open(outName, 'w') as f:
            writer = csv.writer(f, dialect=my_dialect)
            l = ['#DataDate: %s' % self.date]
            writer.writerow(l)
            l = [self.getCreationTimeLine()]
            writer.writerow(l)
            l = ['#Columns: AxiomaID']
            l.extend(dfJoined.columns.values)
            writer.writerow(l)
            l = ['#Type: ID', 'Attribute']
            writer.writerow(l)
            l = ['#Unit: ID', 'Percent']
            writer.writerow(l)
            for index, row in dfJoined.iterrows():
                l = [index]
                l.extend(row.values)
#                 print l
                writer.writerow(l)

class UnivJoinerRskComm(UnivJoinerRsk): 
    def __init__(self, inDir, outDir):
        self.inDir = inDir
        self.outDir = outDir

    def removeExtraColumns(self, df):
        df.drop('Estimation Universe', axis=1, inplace=True)
        df.drop('Estimation Universe Weight', axis=1, inplace=True)
        df.drop('Predicted Beta', axis=1, inplace=True)
        df.drop('Specific Return', axis=1, inplace=True)
        df.drop('Total Risk', axis=1, inplace=True)

def fiPlusEq():
    inDir = '%s/parts' % fileDate
    outDir = '%s/joined' % fileDate
    j = UnivJoiner(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.exp' % (client, fileDate), \
            'AXWW21-MH.%s.exp' % fileDate, \
            'AXUN-MH.%s.%s.exp' % (client, fileDate))         
    j = UnivJoinerIdm(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.idm' % (client, fileDate), \
            'AXWW21-MH.%s.idm' % fileDate, \
            'AXUN-MH.%s.%s.idm' % (client, fileDate)) 
    j = UnivJoinerAtt(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.att' % (client, fileDate), \
            'AXWW21-MH.%s.att' % fileDate, \
            'AXUN-MH.%s.%s.att' % (client, fileDate)) 
    j = UnivJoinerRsk(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.rsk' % (client, fileDate), \
            'AXWW21-MH.%s.rsk' % fileDate, \
            'AXUN-MH.%s.%s.rsk' % (client, fileDate)) 


def fiPlusEqPlusComm():

    inDir = '%s/joined' % fileDate
    outDir = '%s/joinedComm' % fileDate
    j = UnivJoinerComm(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.exp' % (client, fileDate), \
            'AXCOM-MH.%s.exp' % fileDate, \
            'AXUN-MH.%s.%s.exp' % (client, fileDate)) 
        
    j = UnivJoinerIdmComm(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.idm' % (client, fileDate), \
            'AXCOM-MH.%s.idm' % fileDate, \
            'AXUN-MH.%s.%s.idm' % (client, fileDate)) 
        
    j = UnivJoinerAttComm(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.att' % (client, fileDate), \
            'AXCOM-MH.%s.att' % fileDate, \
            'AXUN-MH.%s.%s.att' % (client, fileDate)) 
        
    j = UnivJoinerRskComm(inDir, outDir)
    j.doIt(date, \
            'AXUN-MH.%s.%s.rsk' % (client, fileDate), \
            'AXCOM-MH.%s.rsk' % fileDate, \
            'AXUN-MH.%s.%s.rsk' % (client, fileDate)) 


if __name__ == '__main__':
    date = sys.argv[1] 
    print(date)
    client = sys.argv[2]
    print(client)
    fileDate = ''.join(date.split('-'))

    print('joining eq')
    fiPlusEq()
    time.sleep(3)
    print('joining comm')
    fiPlusEqPlusComm()


import pandas as pd
import os
import sys
import csv
import time
import datetime
import glob

class UnivFilter:
    def __init__(self):
        pass

    def getExclusionRows(self, date, fileName):
        rows = []
        with open(fileName) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if date == row['Date']:
                    rows.append(row['AxiomaID'])
                    #    if not row['AssetType'] == 'Stock':
                    #        rows.append(row['AxiomaID'])
                    # print(row['Date'], row['AxiomaID'], row['Description'], row['ISIN'], row['AssetType'], row['Total Risk'], row['Specific Risk'])
        return rows

    def process(self):
        date = sys.argv[1]
        print(date)
        fileDate = ''.join(date.split('-'))
        inDir = '%s/joinedComm' % fileDate
        outDir = '%s/excluded' % inDir
        excludeFileName = '%s/exclude.csv' % inDir
        ids = self.getExclusionRows(date, excludeFileName)

        for fileType in ['exp', 'idm', 'att', 'rsk', 'isc', 'cov']:
            for fullName in  glob.glob('%s/joinedComm/*.%s' % (fileDate, fileType)):
                shortName = fullName.split('/')[2]
                outName = os.path.join(outDir, shortName)
                lines = self.extractFilterredLines(fullName, ids)
                outF = open(outName,'w')
                for line in lines:
                    outF.write(line + os.linesep)

    def extractFilterredLines(self, fullName, ids):
        lines = []
        with open(fullName) as f:
            for l in f:
                l = l.strip()
                if l.split('|')[0] not in ids:
                    lines.append(l)
        return lines

if __name__ == '__main__':
    f = UnivFilter()
    f.process()


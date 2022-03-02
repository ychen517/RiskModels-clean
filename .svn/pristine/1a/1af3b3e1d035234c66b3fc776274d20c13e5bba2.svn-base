
import csv
import datetime
import tempfile
import os
import logging
import shutil
import time

class FactorLibrary:
    def __init__(self, vendorDB, marketDB, modelDB, factorName, options):
        self.vendorDB=vendorDB
        self.marketDB=marketDB
        self.modelDB=modelDB
        self.options = options
        self.factorName = factorName

    def getHeaders(self, dt):
        raise NotImplementedError()

    def getConstituents(self, dt):
        raise NotImplementedError()

    def getDistFileName(self, dt):
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        return 'AX%s.%s.zip' % (self.factorName, shortdate)

    def getOutFileName(self, dt, create=False):
        """
          Get file name so that multiple methods can access this
        """
        subDirs=self.options.appendDateDirs
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        # create the subdirectory if it does not exist
        if subDirs:
            targetDir=self.options.flatDir.rstrip('/') + '/%4d/%02d' % (dt.year, dt.month)
        else: 
            targetDir=self.options.flatDir.rstrip('/')
        try:
            if create:
                os.makedirs(targetDir)
        except OSError as e:
            if e.errno != 17:
                raise
            else:
                pass
        outFileName='%s/%s.%s.csv' % (targetDir, self.factorName,shortdate)
        return outFileName

    def writeUFFoutput(self, dt):
        """
          Can use a single method for output regardless of the factor library type
          expect to see the following defined and setup already
          - the header information in self.header as a list indexed by information about the header
          - constituents in self.constituents as a list of lists with the columns obeying the header information
          - factor library type is present in self.factorName
        """
        subDirs=self.options.appendDateDirs
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        # headers and data is already pre-populated for this factor name earlier
        headers = self.headers
        data = self.data
        if not data:
            logging.warning('No data for %s so bailing', dt)
            return None
        
        outFileName=self.getOutFileName(dt, create=True)
        targetDir=os.path.dirname(outFileName)
        tmpfile = tempfile.mkstemp(suffix=shortdate, prefix=self.factorName, dir=targetDir)
        os.close(tmpfile[0])
        tmpfilename=tmpfile[1]
        outFile=open(tmpfilename,'w')
        with outFile as f:
            writer = csv.writer(f)
            writer.writerows(headers + [[]] + data)

        logging.info('Move UFF file %s to %s', tmpfilename, outFileName)
        shutil.move(tmpfilename, outFileName)
        os.chmod(outFileName, 0o644)
        return outFileName


    def writeDateHeader(self, dt, outFile):
        outFile.write('#DataDate: %s\n' % dt)
        # write createDate in UTC
        now=datetime.datetime.now()
        gmtime = time.gmtime(time.mktime(now.timetuple()))
        utctime = datetime.datetime(year=gmtime.tm_year,
                                    month=gmtime.tm_mon,
                                    day=gmtime.tm_mday,
                                    hour=gmtime.tm_hour,
                                    minute=gmtime.tm_min,
                                    second=gmtime.tm_sec)
        outFile.write('#CreationTimestamp: %sZ\n' %
                      utctime.strftime('%Y-%m-%d %H:%M:%S'))
        outFile.write("#FlatFileVersion: 3.3\n")

    def writeEncryptedFileOutput(self, dt):
        """
           use a simple encrypted file for output
        """
        from riskmodels.wombat import scrambleString
        import base64

        subDirs=self.options.appendDateDirs
        shortdate='%04d%02d%02d' % (dt.year, dt.month, dt.day)
        # headers and data is already pre-populated for this factor name earlier
        headers = self.headers
        data = self.data
        if not data:
            logging.warning('No data for %s so bailing', dt)
            return None
        
        outFileName=self.getOutFileName(dt, create=True)
        outFileNameClear=outFileName+'.clear.txt'
        targetDir=os.path.dirname(outFileName)
        tmpfile = tempfile.mkstemp(suffix=shortdate, prefix=self.factorName, dir=targetDir)
        os.close(tmpfile[0])
        tmpfilename=tmpfile[1]
        outFile=open(tmpfilename,'w')
        outFile1=open(outFileNameClear,'w')
        self.writeDateHeader(dt, outFile)
        self.writeDateHeader(dt, outFile1)
        # for now write out the various headers based on what we have for HOLT

        outFile.write('#Columns:'+'|'.join( ['AxiomaID'] + headers[1][1:]))
        outFile.write('\n')
        outFile.write('#Type:'+'|'.join( ['AxiomaID'] + [h.replace('|',',') for h in headers[0][1:]]))
        outFile.write('\n')
        outFile.write('#Unit:'+'|'.join( ['ID'] + [h.replace('|',',') for h in headers[4][1:]]))
        outFile.write('\n')
        outFile.write('#Desc:'+'|'.join( ['AxiomaID'] + [h.replace('|',',') for h in headers[2][1:]]))
        outFile.write('\n')

        outFile1.write('#Columns:'+'|'.join( ['AxiomaID'] + headers[1][1:]))
        outFile1.write('\n')
        outFile1.write('#Type:'+'|'.join( ['AxiomaID'] + [h.replace('|',',') for h in headers[0][1:]]))
        outFile1.write('\n')
        outFile1.write('#Unit:'+'|'.join( ['ID'] + [h.replace('|',',') for h in headers[4][1:]]))
        outFile1.write('\n')
        outFile1.write('#Desc:'+'|'.join( ['AxiomaID'] + [h.replace('|',',') for h in headers[2][1:]]))
        outFile1.write('\n')
        for dd in data:
            assetid=dd[0]
            dataString='|'.join([str(h) for h in dd[1:]])
            #outFile.write('%s|%s|%d\n' % (axid, wombat3(dataString),wombat(wombat3(dataString))))
            encVal = scrambleString(dataString, 25, assetid, dt, '')
            base64val = base64.b64encode(encVal)
            outFile.write('%s|%s\n' % (assetid, base64val.decode('utf8')))
            outFile1.write('|'.join([str(h) for h in dd]))
            outFile1.write('\n')
        #.............
        outFile.close()
        outFile1.close()
        logging.info('Move encrypted file %s to %s', tmpfilename, outFileName)
        shutil.move(tmpfilename, outFileName)
        os.chmod(outFileName, 0o644)
        return outFileName

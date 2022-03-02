

import glob
import gzip
import logging


def getETFs(dir, dt, etfs):
    files=sorted(glob.glob('%s/%s*.gz' % (dir, str(dt).replace('-',''))))
    if len(files)==0:
        return {}
    # only look at the last file of the day.  No need to process all of them
    fileName=files[-1]
    if etfs:
        etfList=etfs.split(',')
    else:
        etfList=None
    etfList=[i.upper() for i in etfList]

    ETF_DICT={}
    logging.info('Processing file %s', fileName)
    f=gzip.open(fileName)
    for line in f:
        line=line.strip().split('|')
        if len(line)<10:
            continue
        recType,inst,name,junk,junk1,cusip, sedol,isin,ric, exchange=line[0:10]
        if recType=='HI':
            if not etfList or (etfList and len([i for i in etfList if inst.find(i) >=0]) > 0):
                if inst not in ETF_DICT:
                    ETF_DICT[inst]=[cusip, sedol, isin, ric, name, exchange]

    f.close()
    logging.info('%d ETFs',  len(list(ETF_DICT.keys())))
    logging.debug('%s', list(ETF_DICT.keys()))
    return ETF_DICT
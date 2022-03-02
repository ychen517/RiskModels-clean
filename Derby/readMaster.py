
import datetime
import io
import struct
import sys
import zipfile

BASEDATE = datetime.date(1970,1,1)

def decodeDate(d):
    days = struct.unpack("!H", d)[0]
    #print days
    delta = datetime.timedelta(days)
    return BASEDATE+delta

def decodeDouble(d):
    r = struct.unpack(">d", d)
    return r[0]

if __name__=='__main__':
    if zipfile.is_zipfile(sys.argv[1]):
        jarfile = zipfile.ZipFile(sys.argv[1])
        inFile = io.BytesIO(jarfile.read('Master'))
        inFile2 = io.BytesIO(jarfile.read('Currency'))
        jarfile.close()
    else:
        inFile = open(sys.argv[1], 'rb')
        inFile2 = None
    filedate = inFile.read(2)
    #print len(filedate),filedate
    print('File Date: ',decodeDate(filedate))
    nums = inFile.read(2)
    num_ids = nums[0]
    num_reserved = nums[1]
    allIds = dict()
    for i in range(num_ids):
        idnum = ord(inFile.read(1))
        idnamlen = ord(inFile.read(1))
        idname = inFile.read(idnamlen).decode('utf-8')
        idlen = ord(inFile.read(1))
        allIds[idnum] = (idname, idlen)
    #print allIds
    axid = inFile.read(9).decode('utf-8')
    while axid:
        fromDt = decodeDate(inFile.read(2))
        thruDt = decodeDate(inFile.read(2))
        print(axid,'From:',fromDt,'Thru:',thruDt)
        assert inFile.read(num_reserved) == b' '*num_reserved
        numMaps = ord(inFile.read(1))
        for i in range(numMaps):
            mapNum = ord(inFile.read(1))
            (name, mylen) = allIds[mapNum]
            if mylen == 0:
                idlen = ord(inFile.read(1))
                value = inFile.read(idlen).decode('utf-8')
            else:
                value = inFile.read(mylen).decode('utf-8')
            idfromDt = decodeDate(inFile.read(2))
            idthruDt = decodeDate(inFile.read(2))
            print('%s: %s From: %s Thru: %s' % (name,value,idfromDt,idthruDt))
        print()
        axid = inFile.read(9).decode('utf-8')
    
    inFile.close()
    if inFile2:
        curCode = inFile2.read(5).decode('utf-8')
        while curCode:
            idlen = ord(inFile2.read(1))
            desc = inFile2.read(idlen).decode('utf-8')
            rate = decodeDouble(inFile2.read(8))
            rfr = decodeDouble(inFile2.read(8))
            cumret = decodeDouble(inFile2.read(8))
            print('Currency %s: %s: rate: %g, rfr: %g, cumret: %g' % \
                (curCode, desc, rate, rfr, cumret))
            curCode = inFile2.read(5).decode('utf-8')
        inFile2.close()

        

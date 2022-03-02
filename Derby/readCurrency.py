
import io
import struct
import sys
import zipfile

if __name__=='__main__':
    if zipfile.is_zipfile(sys.argv[1]):
        jarfile = zipfile.ZipFile(sys.argv[1])
        inFile = io.StringIO(jarfile.read('Currency'))
        jarfile.close()
    else:
        inFile = open(sys.argv[1], 'rb')
    isoCode = inFile.read(5)
    while isoCode:
        isoCode = isoCode[:isoCode.find('\0')]
        length = ord(inFile.read(1))
        desc = inFile.read(length)
        fxrate = struct.unpack('>d', inFile.read(8))[0]
        rfr = struct.unpack('>d', inFile.read(8))[0]
        cumRfr = struct.unpack('>d', inFile.read(8))[0]
        print(isoCode, desc, fxrate, rfr, cumRfr)
        isoCode = inFile.read(5)
    inFile.close()

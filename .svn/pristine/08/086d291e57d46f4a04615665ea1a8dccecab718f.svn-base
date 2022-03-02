import datetime
import hashlib
import struct
import sys

MINC=33
MAXC=123
LEN = MAXC - MINC + 1
QUOTIENT = 0x04c11db7

# Scramble the license string.
#
def wombat3(input):
  result=''
  for i in range(len(input)):
    if i == 0:
      prev = 0
    else:
      prev = ord(result[i-1])
    c = input[i]
    if ord(c)>=MINC and ord(c)<=MAXC:
      c = chr(((ord(c) + (LEN // 2) - MINC + i + prev) % LEN) + MINC)
    result += c
  return result

# Unscramble the obfuscated license string.
#
def wombat2(input):
    result = ''
    for i in range(len(input)):
        if i == 0:
            prev = 0
        else:
            prev = ord(input[i-1])
        
        c = input[i]
        if ord(c) >= MINC and ord(c) <= MAXC:
            c = chr(((ord(c) + LEN - (LEN // 2) - MINC + LEN - (i % LEN) + LEN - (prev % LEN) ) % LEN) + MINC)
        result += c
    return result

# An implementation of the CRC-32 checksum taken from
# http://www.cl.cam.ac.uk/Research/SRG/bluebook/21/crc/node6.html
# We prepend "A#Io" to the data string.
def wombat(input):
  result = ord('A') << 24
  result |= ord('#') << 16
  result |= ord('I') << 8
  result |= ord('o')
  result = ~result & 0xffffffff
  
  for c in input:
    o = ord(c)
    for j in range(8):
      if result & 0x80000000:
        result = ((result << 1) & 0xFFFFFFFF) ^ QUOTIENT ^ (o >> 7)
      else:
        result = (result << 1) ^ (o >> 7)
      o = (o << 1) & 0xFF
    
  return ~result & 0xffffffff

def getDigestBytes(assetid, dateobj, attributeName):
    """
    Returns bytes to use when scrambling/unscrambling data for this combination

    :param assetid: str the primary id (AxiomaID)
    :param dateobj: datetime.date or datetime.datetime object
    :param attributeName: str value of the column_display_name for this attribute
    :return: bytes of digest information
    """
    assetBytes = assetid.encode('utf-8') + attributeName.encode('utf-8')
    x = ((dateobj.year - 1900) * 100 + dateobj.month - 1) * 100 + dateobj.isoweekday()
    b = assetBytes + struct.pack('>q', x)
    m = hashlib.sha1()
    m.update(b)
    return m.digest()

def scrambleString(inputvalue, minLength, assetid, dateobj, attributeName):
    '''
    Produces obfuscated strings

    :param inputvalue: str string to be scrambled
    :param assetid: str the primary id (AxiomaID)
    :param dateobj: datetime.date or datetime.datetime object
    :param attributeName: str value of the column_display_name for this attribute
    :return: bytes with scrambled value
    '''
    d = getDigestBytes(assetid, dateobj, attributeName)
    if inputvalue is None:
        inputvalue = "\u0001"
    while len(inputvalue) < minLength:
        inputvalue += "\u0000"
    inputvalue = inputvalue.encode('utf-8')
    charlist = [None] * len(inputvalue)
    for i in range(len(inputvalue)):
        charlist[i] = inputvalue[i] ^ d[i % len(d)]
    return bytes(charlist)

def unscrambleString(scrambled, assetid, dateobj, attributeName):
    '''
    Reverses scrambleString()

    :param scrambled: bytes to be unscrambled
    :param assetid: str the primary id (AxiomaID)
    :param dateobj: datetime.date or datetime.datetime object
    :param attributeName: str value of the column_display_name for this attribute
    :return: str with unscrambled value
    '''
    d = getDigestBytes(assetid, dateobj, attributeName)
    if scrambled is None:
        return None
    charlist = [None] * len(scrambled)
    for i in range(len(scrambled)):
        charlist[i] = chr(scrambled[i] ^ d[i % len(d)])
    return ''.join(charlist)

if __name__ == '__main__':
    if sys.argv[1] == 'code':
        scrambled = wombat3(sys.argv[2])
        print('Scrambled license: ' + scrambled)
        print('Signature: %d' % wombat(scrambled))
    elif sys.argv[1] == 'decode':
        scrambled = sys.argv[2]
        print('Unscrambled license: ' + wombat2(scrambled))
        print('Signature: %d' % wombat(scrambled))
    else:
        print('Usage: ' + sys.argv[0] + ' code|decode <license string>')

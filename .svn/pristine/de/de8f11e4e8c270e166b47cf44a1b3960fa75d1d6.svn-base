
class ModelID:
    __prime = 9576890767
    # this big number is the multiplicative inverse of __prime mod __maxSeq
    __inverse = 999619050863
    __chars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P',
                'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' ]
    __numChars = len(__chars)
    __numDigits = 8
    __charMap = dict(zip(__chars, list(range(__numChars))))
    __maxSeq = pow(len(__chars), __numDigits)
    __prefix = 'D'
    
    def __init__(self, index=None, string=None):
        assert(index == None or string == None)
        if index != None:
            self.string = self._indexToString(index)
        else:
            self.string = string.strip()

    def __eq__(self, other):
        return self.string == other.string
    def __ne__(self, other):
        return self.string != other.string
    def __lt__(self, other):
        return self.string < other.string
    def __le__(self, other):
        return self.string <= other.string
    def __gt__(self, other):
        return self.string > other.string
    def __ge__(self, other):
        return self.string >= other.string
    def __hash__(self):
        return self.string.__hash__()
    def __repr__(self):
        return 'ModelID(%s)' % self.string
    def getCashCurrency(self):
        assert(self.isCashAsset())
        return self.string[5:8]
    def getIDString(self):
        return self.string
    def getPublicID(self):
        return self.string[1:]
    def isCashAsset(self):
        return self.string[1:5] == 'CSH_'
    def _indexToString(self, index):
        if index < 0 or index >= self.__maxSeq:
            raise IndexError
        modVal = (index * self.__prime) % self.__maxSeq
        string = []
        for i in range(self.__numDigits):
            ci = modVal % self.__numChars
            modVal //= self.__numChars
            string.append(self.__chars[ci])
        string.reverse()
        baseID = ''.join(string)
        return self.__prefix + baseID + self.checkDigit(baseID)

    def getIndex(self):
        string = self.getPublicID()[:-1]
        sum=0
        for c in string:
            sum = sum*self.__numChars + self.__chars.index(c)
        return (sum*self.__inverse)%self.__maxSeq

    def checkDigit(self, baseID):
        """Implements CUSIP checkdigit calculation based on specification from
        <a href="http://www.cusip.com">www.cusip.com</a>.
        """
        OFFSET_ALPHA = ord('A')
        OFFSET_NUM = ord('0')
        checkDigit = -1
        if len(baseID) == 8:
            sumD = 0
            for i in range(8):
                factor = (i % 2) + 1
                val = ord(baseID[i]) - OFFSET_ALPHA
                if val < 0:
                    val = ord(baseID[i]) - OFFSET_NUM
                else:
                    val += 10  # add 10 for alpha characters
                mult = factor * val
                if mult > 9:
                    sumD += (mult // 10)
                sumD += ( mult % 10 )
            mod = sumD % 10
            if mod == 0:
                checkDigit = 0
            else:
                checkDigit = 10 - mod
        return chr(checkDigit + OFFSET_NUM)


if __name__ == '__main__':
    while True:
        mid = input("Enter a model ID: ").upper()
        if mid == '':
            break
        m = ModelID(string=mid)
        print('%s corresponds with AxiomaDataId %d' % (m.getIDString(), m.getIndex()))

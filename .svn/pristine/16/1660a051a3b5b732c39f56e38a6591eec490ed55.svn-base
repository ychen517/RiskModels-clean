# Script to move dated files (pattern *yyyyMMdd.???) from one directory
# to directories by date. Default is yyyy/MM sub-directories but that
# can be changed with the --subdir option.

import logging
import optparse
import os
import os.path
import shutil

def createDateDict(dateStr):
    dateDict = {}
    dateDict['yyyy'] = dateStr[:4]
    dateDict['MM'] = dateStr[4:6]
    dateDict['dd'] = dateStr[6:8]
    return dateDict

def main():
    usage = "usage: %prog [options] src-dir top-target-dir"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("-n", action="store_true",
                             default=False, dest="testOnly",
                             help="dry-run, only print the move commands")
    cmdlineParser.add_option("--subdir", action="store",
                             default='%(yyyy)s/%(MM)s', dest="subDir",
                             help="date-specific path under top-target-dir")
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    srcDir = args[0]
    tgtDir = args[1]
    datePath = options.subDir
    assert(os.path.isdir(srcDir))
    assert(os.path.isdir(tgtDir))
    for entry in sorted(os.listdir(srcDir)):
        srcPath = os.path.join(srcDir, entry)
        if not os.path.isfile(srcPath):
            continue
        # Assume that the file ends in yyyymmdd.???
        datePart = os.path.splitext(entry)[0][-8:]
        dateDict = createDateDict(datePart)
        targetPath = os.path.join(tgtDir, datePath % dateDict, entry)
        if options.testOnly:
            print('shutil.move(%s, %s)' % (srcPath, targetPath))
        else:
            shutil.move(srcPath, targetPath)

if __name__ == '__main__':
    main()

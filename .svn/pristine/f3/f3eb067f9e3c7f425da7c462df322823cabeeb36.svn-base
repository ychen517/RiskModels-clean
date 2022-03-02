import datetime
import logging
import argparse
import os.path
import shutil

def buildSourcePath(sourceDir, modelFamily, date):
    return os.path.join(sourceDir, '%d' % date.year, '%02d' % date.month,
                        'CorpActions-%s-%d%02d%02d.xml' % (modelFamily, date.year, date.month, date.day))

def buildTargetPath(options, modelFamily, date):
    basePath = options.targetDir
    if options.appendDateDirs:
        basePath = os.path.join(basePath, '%d' % date.year, '%02d' % date.month)
    return os.path.join(
        basePath, 'CorpActions-%s-%d%02d%02d.xml' % (modelFamily, date.year, date.month, date.day))

def createTargetDirectory(target):
    try:
        os.makedirs(target)
    except OSError as e:
        if e.errno != 17:
            raise
        else:
            pass

def main():
    usage = "usage: %prog [options] publicDir archiveDir correctionFiles"
    cmdlineParser = argparse.ArgumentParser()
    cmdlineParser.add_argument('publicDir', help='path to the published Derby files')
    cmdlineParser.add_argument('archiveDir', help='directory to hold the archived corporate action files')
    cmdlineParser.add_argument('correctionFile', nargs='+', help='corrected corporate action files')
    args = cmdlineParser.parse_args()

    createTargetDirectory(args.archiveDir)
    for correctionsFile in args.correctionFile:
        if not os.path.exists(correctionsFile):
            print("Corrected corporate action file {} does not exist".format(correctionsFile))
        else:
            year = correctionsFile[-12:-8]
            month = correctionsFile[-8:-6]
            publicFile = os.path.join(args.publicDir, year, month, correctionsFile)
            archiveFile = os.path.join(args.archiveDir, correctionsFile)
            if not os.path.exists(publicFile):
                print("Corporate action file {} is not present in public directory. Skipping".format(correctionsFile))
                continue
            # Move copy published file from public directory to archive and then replace with corretion
            print("copy {} to {}".format(publicFile, archiveFile))
            shutil.copy2(publicFile, archiveFile)
            print("copy {} to {}".format(correctionsFile, publicFile))
            shutil.copy2(correctionsFile, publicFile)

if __name__ == '__main__':
    main()

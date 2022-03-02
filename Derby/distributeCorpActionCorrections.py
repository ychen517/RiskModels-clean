
import datetime
import logging
import optparse
import xml.dom.minidom as minidom
import os.path

from marketdb import MarketDB
from riskmodels import ModelDB
from riskmodels import ModelID
from riskmodels import Utilities
from riskmodels import writeFlatFiles
import writeCorporateActionXML

def removeTextElements(node):
    for el in list(node.childNodes):
        if el.nodeType == minidom.Node.TEXT_NODE:
            node.removeChild(el)
        else:
            removeTextElements(el)

def buildSourcePath(sourceDir, modelFamily, date):
    return os.path.join(sourceDir, '%d' % date.year, '%02d' % date.month,
                        'CorpActions-%s-%d%02d%02d.xml' % (modelFamily, date.year, date.month, date.day))
def buildTargetPath(options, modelFamily, date):
    basePath = options.targetDir
    if options.appendDateDirs:
        basePath = os.path.join(basePath, '%d' % date.year, '%02d' % date.month)
    return os.path.join(
        basePath, 'CorpActions-%s-%d%02d%02d.xml' % (modelFamily, date.year, date.month, date.day))

def findMatchingElementByAttribute(source, element, attName):
    """Find the element in source that has the same tag as 'element' and also matches its value
    for attName.
    Returns the element in source or None if no match can be found.
    """
    targetVal = element.getAttribute(attName)
    for srcEl in source.getElementsByTagName(element.tagName):
        if srcEl.getAttribute(attName) == targetVal:
            return srcEl
    return None

def mergeCorporateActions(updateElement, sourceFile, targetFile, activeAssets):
    sourceXml = minidom.parse(sourceFile)
    root = sourceXml.documentElement
    removeTextElements(root)
    # Update create time element
    root.setAttribute('createtime', str(datetime.datetime.now()))
    
    dateElement = findMatchingElementByAttribute(root, updateElement, 'val')
    updateDateStr = updateElement.getAttribute('val')
    updatesApplied = False
    if dateElement is None:
        logging.info('Added new ex_dt element for %s', updateDateStr)
        dateElement = sourceXml.createElement('ex_dt')
        dateElement.setAttribute('val', updateDateStr)
        root.appendChild(dateElement)
    
    for assetElement in updateElement.getElementsByTagName('asset'):
        assetId = assetElement.getAttribute('ax_id')
        mdlAsset = ModelID.ModelID(string='D' + assetId)
        if mdlAsset not in activeAssets and not assetId.startswith('CSH_'):
            # Skip corrections for asset that are not part of this model family
            logging.info('asset %s is not part of models. Skipping', assetId)
            continue
        oldAssetEl = findMatchingElementByAttribute(dateElement, assetElement, 'ax_id')
        if oldAssetEl is not None:
            dateElement.removeChild(oldAssetEl)
        dateElement.appendChild(assetElement.cloneNode(True))
        updatesApplied = True
    
    if updatesApplied:
        # write XML file 
        targetDateStr = targetFile[-12:-4]
        with writeFlatFiles.TempFile(targetFile, targetDateStr) as outFile:
            sourceXml.writexml(outFile.getFile(), '  ', '  ', newl='\n', encoding='UTF-8')
    else:
        logging.info('No file written since no updates applied to this model')

def createTargetDirectory(options, d):
    if options.appendDateDirs:
        target = os.path.join(options.targetDir, '%04d' % d.year, '%02d' % d.month)
        try:
            os.makedirs(target)
        except OSError as e:
            if e.errno != 17:
                raise
            else:
                pass

def main():
    usage = "usage: %prog [options] correctionFile sourceDir"
    cmdlineParser = optparse.OptionParser(usage=usage)
    cmdlineParser.add_option("--model-families", action="store",
                             default='all', dest="modelFamilies",
                             help="comma-separated list of model families to update. Default=all")
    cmdlineParser.add_option("-d", "--directory", action="store",
                             default='.', dest="targetDir",
                             help="directory for modified files")
    cmdlineParser.add_option("--target-sub-dirs", action="store_true",
                             default=False, dest="appendDateDirs",
                             help="Append yyyy/mm to end of output directory path")
    
    Utilities.addDefaultCommandLine(cmdlineParser)
    
    (options, args) = cmdlineParser.parse_args()
    if len(args) != 2:
        cmdlineParser.error("Incorrect number of arguments")
    
    Utilities.processDefaultCommandLine(options, cmdlineParser)
    modelDB = ModelDB.ModelDB(sid=options.modelDBSID, user=options.modelDBUser, passwd=options.modelDBPasswd)
    marketDB = MarketDB.MarketDB(sid=options.marketDBSID, user=options.marketDBUser,
                                 passwd=options.marketDBPasswd)
    modelDict=modelDB.getModelFamilies()
    allModelFamilies = set(modelDict.values())
    if options.modelFamilies == 'all':
       modelFamilies = allModelFamilies
    else:
        modelFamilies = set(options.modelFamilies.split(','))
        if len(modelFamilies - allModelFamilies) > 0:
            logging.error('Unknown model familie(s): %s', modelFamilies - allModelFamilies)
            return
    familyModelMap = writeCorporateActionXML.getRiskModelFamilyToModelMap(modelFamilies, modelDB, marketDB)
    
    correctionsFileName = args[0]
    sourceDir = args[1]
    correctionsDoc = minidom.parse(correctionsFileName)
    removeTextElements(correctionsDoc.documentElement)
    dateElements = correctionsDoc.getElementsByTagName('ex_dt')
    modelOptions = Utilities.Struct()
    modelOptions.preliminary = False
    modelOptions.allowPartialModels = True
    for dateElement in dateElements:
        date = Utilities.parseISODate(dateElement.getAttribute('val'))
        for modelFamily in modelFamilies:
            # Determine source file. First find the first model family file on or after the specified date
            # Then check the target directory if we already processed that file for a different date.
            # If so, use the current target file. Otherwise use the source file
            sourceDate = None
            for d in [date + datetime.timedelta(days=i) for i in range(30)]:
                sourceFile = buildSourcePath(sourceDir, modelFamily, d)
                if os.path.exists(sourceFile):
                    sourceDate = d
                    break
            if sourceDate is None:
                logging.fatal("Couldn't find corporate action file to patch for %s on %s. Skipping",
                              modelFamily, date)
                continue
            else:
                targetFile = buildTargetPath(options, modelFamily, sourceDate)
                if os.path.exists(targetFile):
                    logging.info('Using already patched corporate action file on %s for %s on %s',
                                 sourceDate, modelFamily, date)
                    sourceFile = targetFile
                else:
                    logging.info('Using corporate action file on %s to patch %s on %s',
                                 sourceDate, modelFamily, date)
            createTargetDirectory(options, sourceDate)
            activeModels = writeCorporateActionXML.getActiveModels(familyModelMap[modelFamily],
                                                                   d, modelDB, False)
            modelFamilyAssets = writeCorporateActionXML.processModelFamilyDay(
                activeModels, sourceDate, modelDB, marketDB, modelOptions, True)[0]
            mergeCorporateActions(dateElement, sourceFile, targetFile, modelFamilyAssets)
    modelDB.finalize()

if __name__ == '__main__':
    main()

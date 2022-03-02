# Reads a corporate action file and replaces all clear-text corporate actions with their
# encrypted counterparts or vice versa.
import argparse
import base64
import codecs
import datetime
import xml.dom.minidom as minidom
import os
import logging
import logging.config

from riskmodels import ModelID
from riskmodels import Utilities
from riskmodels import wombat
from riskmodels import writeFlatFiles
import writeCorporateActionXML

def processFile(inFileName, outFileName, tagToReplace, transform):
    document = minidom.parse(inFileName)
    writeCorporateActionXML.removeTextElements(document.documentElement)
    dateElements = document.getElementsByTagName('ex_dt')
    for dateElement in dateElements:
        dateStr = dateElement.getAttribute('val')
        date = Utilities.parseISODate(dateStr)
        for assetElement in dateElement.getElementsByTagName('asset'):
            axid = assetElement.getAttribute('ax_id')
            mdlID = ModelID.ModelID(string='D' + axid)
            for actionElement in assetElement.getElementsByTagName(tagToReplace):
                encAction = transform(mdlID, date, actionElement, document)
                assetElement.replaceChild(encAction, actionElement)
    with writeFlatFiles.TempFile(outFileName, "conv") as outFile:
        with outFile.getFile() as g:
            document.writexml(g, '  ', '  ', newl='\n', encoding='UTF-8')

def encryptFile(inFileName, outFileName):
    processFile(inFileName, outFileName, 'action', writeCorporateActionXML.encryptAction)

def decryptFile(inFileName, outFileName):
    processFile(inFileName, outFileName, 'encryptedaction', decryptAction)

def decryptAction(mdlID, date, encNode, document):
    encryptedAction = encNode.getAttribute("val")
    val = base64.b64decode(encryptedAction)
    actionList = wombat.unscrambleString(val, mdlID.getPublicID(), date, '').split(':')
    elem = document.createElement('action')
    for action in actionList:
        elem.setAttribute(*action.split('='))
    return elem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputFile", help="file to be encoded.")
    parser.add_argument("outputFile", help="output encoded file generated from input file.")
    parser.add_argument("--decrypt", help="decrypt instead of encrypt", action='store_true')
    args = parser.parse_args()
    if args.decrypt:
        decryptFile(args.inputFile, args.outputFile)    
    else:
        encryptFile(args.inputFile, args.outputFile)    

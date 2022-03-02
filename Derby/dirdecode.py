'''
Documentation missing here!
'''

import argparse
import encryptCorpActions as filedecode
import os
from time import time
import logging
import logging.config

'''
  Invocation Example: E:\Axioma\Data\RiskModels\2.1\Derby   
                      E:\Axioma\CorporateActionsFiles\Decoded  -m E:\Axioma\Data\RiskModels\2.1\Derby -p CorpActions-WW21 -o
                      
  Where: Input Directory = E:\Axioma\Data\RiskModels\2.1\Derby  
         Output Directory = E:\Axioma\CorporateActionsFiles\Decoded
         Mount Point = E:\Axioma\Data\RiskModels\2.1\Derby (all subdirectories contained within the mount point will be mirrored into
                                                            the output directory (i.e. the same directory structure will be preserved;
                                                            otherwise, all decoded files will be dumped into a single user-specified 
                                                            output directory.)
         Pattern to Match = CorpActions-WW21  (Only input files that match this pattern will be decoded)
         -o flag indicates that the output files will be overwritten if they exist (default value is false)
'''
def dirdecode(inputDir, outputDir, fileSelectionPattern = None, mountPoint = None, outputFilePrefix = None, overwrite = False):
    """List all files contained in the user-specified directory."""
    #Traverse directory with a call back
    visitor = AxiomaVisitor(outputDir, fileSelectionPattern, mountPoint, overwrite)
    #os.path.walk(inputDir, visitor.visit, '(User data)')
    start = time()
    os.path.walk(inputDir, visitor.visit, '')
    print('\nTotal Elapsed Time:', time() - start)
    
class ReadableDir(argparse.Action):
    def __call__(self,parser, namespace, values, option_string = None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("ReadableDir:{0} is not a valid path.".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("ReadableDir:{0} is not a readable directory".format(prospective_dir))

class WriteableDir(argparse.Action):
    def __call__(self,parser, namespace, values, option_string = None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("WriteableDir:{0} is not a valid path.".format(prospective_dir))
        if os.access(prospective_dir, os.W_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("WriteableDir:{0} is not a writeable directory.".format(prospective_dir))

class AxiomaVisitor():
    def __init__(self, outputDir=None, selectionPattern=None, mountPoint=None, overwrite=False):
        self.selectionPattern = selectionPattern
        self.outputDir = outputDir
        self.mountPoint = mountPoint
        self.overwrite = overwrite
        logging.config.fileConfig("log.config")
        
    def visit(self, arg, dirname, names):
        for name in names:
            sourceFullPath = os.path.join(dirname, name)
    
            if not os.path.isdir(sourceFullPath):
                #Construct appropriate output file
                outputFileName = name
                newOutputDir = self.outputDir
                
                if self.mountPoint is not None and self.mountPoint in sourceFullPath:
                    fileNameIndex = sourceFullPath.find(name)
                    initSubDirIndex = sourceFullPath.find(self.mountPoint)
                    subDirectory = sourceFullPath[initSubDirIndex + len(self.mountPoint):fileNameIndex]
                    #newOutputDir = os.path.join(self.outputDir, subDirectory)
                    newOutputDir = self.outputDir + subDirectory
                    
                if self.selectionPattern == None or self.selectionPattern in name:
                     # Verify first if file exist and if it's okay to overwrite it
                     filePath = newOutputDir + outputFileName
                     if (not self.overwrite and os.path.isfile(filePath)):
                         logging.info('Skipping existing file:%s'%filePath)
                     else:
                         transformFile(sourceFullPath, newOutputDir, outputFileName)
                         #print 'Completed Corporate Files Conversion in: ' + dirname

def transformFile(inputFile, outputDir, outputFile):
    #create output directory (recursively) if it does not exist
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
       
    filedecode.decryptFile(inputFile, outputDir + outputFile)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    try:
        parser.add_argument("inputDir", help="Directory containing corporate action files to be decoded.", action=ReadableDir)
        parser.add_argument("outputDir", help="Directory to save decoded corporate action files.", action=WriteableDir)
        parser.add_argument('-p', '--pattern', help="User provided pattern to select files containing encrypted corporate actions.\
                                                     If no pattern is provided, all files found will be decoded (default:None).",
                            nargs='?')
        parser.add_argument('-t', '--tag', help="User provided string to label output files (default:None).", nargs='?')
        parser.add_argument('-m', '--mount_point', help="User provided mount point from input directory to mirror directory structure into output directory (default:None).", nargs='?')
        parser.add_argument('-o', '--overwrite', help="Overwrite the output file if it exists (default:False).",  action='store_true')
         
        args = parser.parse_args()
        dirdecode(args.inputDir, args.outputDir, args.pattern, args.mount_point, args.tag, args.overwrite)
    except argparse.ArgumentTypeError as ex:
        print(ex)
        
    

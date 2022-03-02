Version: 3.2 					Date: 2011-07-28

Windows version:
Files required: RskConvert.bat
                RskConvert.js
Usage: RskConvert.bat source-directory [backup-directory]

Perl version:
Files required: RskConverter
Usage: RskConverter source-directory [backup-directory]

The following applies to both versions of the script:

source-directory is the directory which contains the .rsk and .idm
files to be converted

optional backup-directory is a directory where the original files will
be copied.  Backup copies have the string ".orig" appended to the
filename.  If backup-directory is omitted, the source directory is
used.

For each file which ends in .rsk or .idm in the source-directory, the
script will remove the new fields from each line.  The original file
will be copied to the backup-directory and renamed as above.  If the
backup file already exists, no action will be taken on the file.  For
each file processed, the script prints the name of the file as it is
being processed.  If a file is skipped, nothing is printed.


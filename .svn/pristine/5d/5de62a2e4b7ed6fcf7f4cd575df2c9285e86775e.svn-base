#!/bin/sh
#$Id: runMutualFundModelTransfer.sh 210868 2016-07-12 14:30:09Z vsmani $:
# simple shell script to compute N days back and pass it to our regular model transfer
daysBack=5
testOnly=""
USAGE="[-d days] configFile date subids"
while getopts 'nd:' OPTION
do
 case "$OPTION" in
   d) daysBack="$OPTARG";;
   n) testOnly="-n";;
   [?]) echo "Usage: $0 $USAGE"
        exit 1;;
 esac
done
shift $(($OPTIND-1))

configFile=$1
sections="SubIssueData,SubIssueReturn"
dt=$2
subids=$3
startdt=`date "+%Y-%m-%d" --date="$daysBack days ago"`
cmd="python3 transfer.py $testOnly $configFile sections=$sections dates=$startdt:$dt sub-issue-ids=$subids asset-type=mutualfunds"

echo $cmd
$cmd

#!/bin/bash
db=$1
jobid=$2
assetList=$3
user=$4
#outputdir=$4

# used for jobcontrol runs
# we need .sh because we want to 'source /home/ops-rm/global/scripts/setup'

cmd="source /home/ops-rm/global3/scripts/setup"
echo $cmd
$cmd

cd /home/ops-rm/global3/scripts/MarketDB/QA
if [ "$1" == "PROD_DB" ]; then
    echo "PROD"
    cmd="python3 BulkQAReporter.py $PROD_DB --JobID=$jobid --AxidList=$assetList --ForceDB=GLPROD"
    if [ "$user" != "" ]; then
        cmd="$cmd --email $user@axioma.com"
    fi
    echo $cmd
    $cmd
else
    echo "SDG"
    cmd="python3 BulkQAReporter.py $DB --JobID=$jobid --AxidList=$assetList --ForceDB=GLSDG"
    if [ "$user" != "" ]; then
        cmd="$cmd --email $user@axioma.com"
    fi
    echo $cmd
    $cmd
fi

status=$? 
echo exit status $status
exit $status

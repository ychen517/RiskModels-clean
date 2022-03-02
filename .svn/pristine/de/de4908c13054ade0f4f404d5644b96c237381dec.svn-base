#!/bin/bash

#
# modelIDCleanup -- runs on Fridays to clean up assets in ModelDB
#                   which have been deleted from MarketDB
#
# usage: modelIDCleanup.sh yyyy-mm-dd [program options]
#

if [[ $# -lt 1 ]] ; then
    echo "Usage: modelIDCleanup.sh yyyy-mm-dd [program options]"
    exit 2
fi

dt=$1
shift
myDt=$(date +%A -d $dt)
if [[ "$myDt" != "Friday" ]]; then
    echo "Today is not Friday, exiting"
    exit 0
fi

python3 $(dirname $0)/modelIDCleanup.py $* $dt
s=$?
exit $s


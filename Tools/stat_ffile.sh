#!/bin/sh
alias stat='python3 generateStatisticalModel.py $STND_DB --model'
export dt=$1
export em=$2
export mdlist="AUAxioma2009MH_S USAxioma2009MH_S CAAxioma2009MH_S GBAxioma2009MH_S EUAxioma2011MH_S WWAxioma2011MH_S EMAxioma2011MH_S JPAxioma2009MH_S CNAxioma2010MH_S APAxioma2011MH_S APxJPAxioma2011MH_S NAAxioma2011MH_S WWxUSAxioma2011MH_S"
export emailFile="email.${date}"

for md in $mdlist
do
    rm -f $emailFile
    touch $emailFile
    rm -f tmp/*facretHist*csv
    stat -n -m$md --force --verbose $dt
    cat tmp/*facretHist*csv >> $emailFile
    cat $emailFile | mail -s "${md} factor returns ${dt}" "${USER}@axiomainc.com,${em}@axiomainc.com"
done

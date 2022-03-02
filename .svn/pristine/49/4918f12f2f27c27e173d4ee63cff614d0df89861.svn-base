#!/bin/bash

# Set up market env variables
export EU0=GB,AT,BE,CZ,DK,FI,FR,DE,GR,HU
export EU0_b=IE,IT,LU,NL,NO,PL,PT,RU,ES,SE,CH,TR
export EU1=BG,HR,CY,EE,LV,LT,RO,SK,SI,IS
export EU2=MT,RS,UA,KZ
export EU3=MK,BA,ME
export AP0=TW,JP,AU,CN,HK
export AP0_b=IN,ID,KR,MY,NZ,PK,PH,SG,TH,LK,XC
export AP2=VN,BD
export AM0=AR,BR,CA,CL,CO,MX,PE
export AM1=VE
export AM2=EC,JM,TT
export AF0=EG,IL,MA,ZA
export AF1=BH,BW,JO,KW,MU,OM,QA,AE
export AF2=GH,KE,LB,NA,NG,SA,TN,ZM
export AF3=CI,MW,PS,TZ,ZW,UG
export WW0=$EU0,$EU0_b,$AP0,$AP0_b,$AM0,$AF0,US
export WW1=$EU1,$AM1,$AF1
export WW2=$EU2,$AP2,$AM2,$AF2
export WW3=$EU3,$AF3

export dir=$1
export endDate=$2
export step=$3
export cfile=$4
export numer=$5
export cmd="python3 Tools/runMkt.py $dir -f --cf=$cfile --step=$step --numeraire=$numer"

# Zeroth batch
$cmd --rmgs=US --start-date=1980-01-02 --end-date=1994-01-04
# First batch
for rm in `echo $WW0 | sed 's/,/ /g'`
do
    $cmd --rmgs=$rm --start-date=1994-01-04 --end-date=${endDate}
done
# Second batch
for rm in `echo $WW1 | sed 's/,/ /g'`
do
    $cmd --rmgs=$rm --start-date=1997-01-01 --end-date=${endDate}
done
# Third batch
for rm in `echo $WW2 | sed 's/,/ /g'`
do
    $cmd --rmgs=$rm --start-date=2004-01-04 --end-date=${endDate}
done
# Fourth batch
for rm in `echo $WW3 | sed 's/,/ /g'`
do
    $cmd --rmgs=$rm --not-in-models --start-date=2012-01-01 --end-date=${endDate}
done

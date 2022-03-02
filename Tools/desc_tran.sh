#!/bin/bash

# Set up market env variables
# there are 95 single countries.
export AP0=TW,IN,ID,KR
export AP0_b=MY,NZ,PK,PH,SG,TH,LK,XC
export CN_HK=CN,HK
export EU0=GB,AT,BE,CZ,DK,FI,FR,DE,GR,HU
export EU0_b=IE,IT,LU,NL,NO,PL,PT,RU,ES,SE,CH,TR
export JP_AU=JP,AU
export US=US
export WW0=AR,BR,CA,CL,CO,MX,PE,EG,IL,MA,ZA
export WW1=BG,HR,CY,EE,LV,LT,RO,SK,SI,IS,VE,BH,BW,JO,KW,MU,OM,QA,AE
export WW2=MT,RS,UA,KZ,VN,BD,EC,JM,TT,GH,KE,LB,NA,NG,SA,TN,ZM
export WW3=MK,BA,ME,CI,MW,PS,TZ,ZW,UG

export endDate=$1
export step=$2
export numer=$3
export cmd="python3 runMkt.py"
export cfile=test.config

# the order is important - based on running time of the jobs. the longer, the high priority.
export year=1999
$cmd --rmgs=$WW1 --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
export year=2005
$cmd --rmgs=$WW2 --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
export year=1995
$cmd --rmgs=$EU0 --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$WW0 --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$EU0_b --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$US --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$AP0 --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$CN_HK --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$AP0_b --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
$cmd --rmgs=$JP_AU --cf=$cfile --step=$step -f --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate
export year=2012
$cmd --rmgs=$WW3 --cf=$cfile --step=$step -f --not-in-models --numeraire=$numer --start-date=${year}-01-01 --end-date=$endDate

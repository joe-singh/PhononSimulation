#!/bin/bash

s="0" 
d="1000"
while [ $s -lt 6 ]
do
  var=$(awk -v s=$s 'BEGIN {print s/10 }')
  python3 Main.py $var "CorrectedTimes100Longitudinal.txt"
  s=$[s+1]
done

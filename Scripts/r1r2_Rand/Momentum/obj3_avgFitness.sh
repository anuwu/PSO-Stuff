#! /bin/bash

i=0
cnt=0

rm -f avgFitness.txt

while [ $i -lt $2 ]
do
	i=`expr $i + 1`
	printf "$i "
	python obj3.py $1 >> avgFitness.txt
	cnt=`expr $cnt + $?`
done

printf "\n"
printf "Avg fitness value when correct gbest $(grep "1 [0-9]*" avgFitness.txt | awk '{SUM+=$2;NUM+=1}END{print SUM/NUM}') on average\n"

if [ $cnt -lt $2 ]
then
	printf "Avg fitness value when wrong gbest $(grep "0 [0-9]*" avgFitness.txt | awk '{SUM+=$2;NUM+=1}END{print SUM/NUM}') on average\n"
fi	

rm -f avgFitness.txt
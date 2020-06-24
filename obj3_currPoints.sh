#! /bin/bash

i=0
cnt=0
rm -f curr.txt

while [ $i -lt $1 ]
do
	i=`expr $i + 1`
	printf "$i "
	python obj3.py >> curr.txt
	cnt=`expr $cnt + $?`
done

printf "\n"
printf "No of points near correct gbest $(grep "1 [0-9]*" curr.txt | awk '{SUM+=$2;NUM+=1}END{print SUM/NUM}') times on average\n"

if [ $cnt -lt $1 ]
then
	printf "No of points near wrong gbest $(grep "0 [0-9]*" curr.txt | awk '{SUM+=$2;NUM+=1}END{print SUM/NUM}') times on average\n"
fi	

rm -f curr.txt

clear
sl
printf "Good morning Saaku!\n"
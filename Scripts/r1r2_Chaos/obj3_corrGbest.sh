#! /bin/bash

i=0
cnt=0

while [ $i -lt $2 ]
do
	i=`expr $i + 1`
	printf "$i "
	python obj3.py $1
	cnt=`expr $cnt + $?`
done

printf "\nPSO finds correct gbest, $cnt out of $2 times\n"
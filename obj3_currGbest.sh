#! /bin/bash

i=0
cnt=0

while [ $i -lt $1 ]
do
	i=`expr $i + 1`
	printf "$i "
	python obj3.py
	cnt=`expr $cnt + $?`
done

printf "\nChaos finds correct gbest, $cnt out of $1 times\n"
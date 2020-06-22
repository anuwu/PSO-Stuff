#! /bin/bash

i=0
same=0

while [ $i -lt $1 ]
do
	i=`expr $i + 1`
	printf "$i "
	python obj3.py
	same=`expr $same + $?`
done

diff=`expr $1 - $same`
printf "\nChaotic performs better $diff out of $1 times\n"
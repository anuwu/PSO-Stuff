#! /bin/bash

inrt="05"
while [ $inrt -lt 100 ]
do
	# echo $inrt

	i=0
	printf "0.$inrt "
	rm -f curr.txt
	while [ $i -lt 25 ]
	do
		i=`expr $i + 1`
		python convergence.py $1 0.$inrt >> curr.txt
	done

	printf "$(grep "1 [0-9]*" curr.txt | awk '{SUM+=$2;NUM+=1}END{print SUM/NUM}')\n"
	#rm -f curr.txt

	inrt=$[$inrt+5]
done

i=0
printf "1.00 "
rm -f curr.txt
while [ $i -lt 25 ]
do
	i=`expr $i + 1`
	python convergence.py $1 1.0 >> curr.txt
done

printf "$(grep "1 [0-9]*" curr.txt | awk '{SUM+=$2;NUM+=1}END{print SUM/NUM}')\n"
rm -f curr.txt
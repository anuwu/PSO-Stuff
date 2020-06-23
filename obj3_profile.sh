#! /bin/bash

i=0
same=0

while [ $i -lt $1 ]
do
	i=`expr $i + 1`
	printf "$i\n"
	python obj3.py
	same=`expr $same + $?`
done

# Comparison on final average fitness values
# printf "\nChaotic performs atleast as good as vanilla, $same out of $1 times\n"

# Anomaly detection
printf "\ncntChaos < cntVanilla but better average fitness, $same out of $1 times\n"
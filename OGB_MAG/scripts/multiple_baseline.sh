for b in 0.1 0.2 0.3 0.4 1
do
	for seed in 0 1 2
	do
		sbatch ./baseline.sh $b $seed
	done
done
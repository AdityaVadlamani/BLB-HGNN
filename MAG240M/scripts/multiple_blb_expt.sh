for s in 2 3 4
do
	for b in 0.1 0.2 0.3 0.4 1
	do
		for seed in 0 1 2
		do
			sbatch ./blb_expt.sh $b $s $seed
		done
	done
done

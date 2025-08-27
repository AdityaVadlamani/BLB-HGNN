for b in 0.1 0.2 0.3 0.4 1
do
	for s in 2
	do
		for ensemble_method in "avg_params" "voting" "deep_ensemble"
		do
			for seed in 0 1 2
			do
				sbatch ./model_merging.sh $b $s $seed $ensemble_method
			done
		done
	done
done
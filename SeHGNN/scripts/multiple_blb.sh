for b in 0.1 0.2 0.3 0.4 0.5 0.6 1.0
do
    for s in 1 2 
    do
        echo $b and $s
        if [[ $s -eq 1 ]]; then
            sbatch ./train_blb.sh $b SeHGNN 
        else
            sbatch ./train_blb.sh $b SeHGNN $s
        fi
    done
done
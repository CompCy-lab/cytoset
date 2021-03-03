#!/bin/bash


#SBATCH --job-name=cellcnn_ICS_256
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH --output=./ICS_256.out
#SBATCH --error=./ICS_256.err
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=haidyi@cs.unc.edu

data_path=/proj/yunligrp/users/lehuang/scripts/cyto/ICS
data_name=ICS
max_epochs=100
ncell=256
nsubset=20000
seed_list=("1" "2" "3" "4" "12345")

for seed in ${seed_list[@]}
do
  echo "ncell $ncell, seed: $seed"
  python cellCnn_eval.py \
    --data_root ${data_path} \
    --data_name ${data_name} \
    --max_epochs ${max_epochs} \
    --nsubset ${nsubset} \
    --ncell ${ncell} \
    --seed ${seed}
done

exit

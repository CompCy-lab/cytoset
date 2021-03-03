#!/bin/bash

data_path=/Users/haidyi/Documents/proj/CytoSet/Data/NK_cell
data_name=NK
max_epochs=20
nsubset=1000
ncell_list=("256" "512" "1024")
seed_list=("1" "2" "3" "4" "5")

for ncell in ${ncell_list[@]}
do
  echo "ncell: $ncell"
  for seed in ${seed_list[@]}
  do
    python cellCnn_eval.py \
      --data_root ${data_path} \
      --data_name ${data_name} \
      --max_epochs ${max_epochs} \
      --nsubset ${nsubset} \
      --ncell ${ncell} \
      --seed ${seed}
  done
done

exit
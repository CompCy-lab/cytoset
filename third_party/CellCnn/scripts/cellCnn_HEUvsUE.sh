#!/bin/bash

data_path=/proj/yunligrp/users/lehuang/scripts/cyto/HEUvsUE
data_name=HEUvsUE
max_epochs=100
nsubset=20000
ncell=4096
seed=("1" "2" "3" "4" "12345")

for seed in ${seed[@]}
  do
    python cellCnn.py \
      --data_root ${data_path} \
      --data_name ${data_name} \
      --max_epochs ${max_epochs} \
      --nsubset ${nsubset} \
      --ncell ${ncell} \
      --seed ${seed}
done

exit
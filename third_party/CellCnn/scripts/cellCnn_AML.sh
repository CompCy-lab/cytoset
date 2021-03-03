#!/bin/bash

data_path=/proj/yunligrp/users/lehuang/scripts/cyto/AML
data_name=AML
max_epochs=20
nsubset=10000
ncell=256
seed=("1" "12" "123" "1234" "12345")

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
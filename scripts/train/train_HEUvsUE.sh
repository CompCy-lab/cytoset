#!/bin/bash

h_dim=256
in_dim=10
pool=max
out_pool=max
nblock=2
ncell=1024
nsubset=20000
co_factor=5

train_fcs_info=/playpen-ssd/haidyi/proj/cytoset/data/HEUvsUE/train/train_labels.csv
valid_fcs_info=/playpen-ssd/haidyi/proj/cytoset/data/HEUvsUE/valid/valid_labels.csv
test_fcs_info=/playpen-ssd/haidyi/proj/cytoset/data/HEUvsUE/test/test_labels.csv
markerfile=/playpen-ssd/haidyi/proj/cytoset/data/HEUvsUE/marker.csv

seed_list=("2" "22" "222" "2222" "22222")
lr=0.0001
batch_size=200
n_epochs=50
log_dir=/playpen-ssd/haidyi/proj/cytoset/data/exp/HEUvsUE_${ncell}
log_interval=1
save_interval=1
patience=5

bin_file=../../train.py
gpu=$1
#seed=$2

for seed in ${seed_list[@]}
do	
  CUDA_VISIBLE_DEVICES=${gpu} python ${bin_file} \
    --in_dim ${in_dim} \
    --h_dim ${h_dim} \
    --nblock ${nblock} \
    --pool ${pool} \
    --out_pool ${out_pool} \
    --ncell ${ncell} \
    --nsubset ${nsubset} \
    --co_factor ${co_factor} \
    --train_fcs_info ${train_fcs_info} \
    --valid_fcs_info ${valid_fcs_info} \
    --test_fcs_info ${test_fcs_info} \
    --markerfile ${markerfile} \
    --lr ${lr} \
    --generate_valid \
    --test_rsampling \
    --shuffle \
    --batch_size ${batch_size} \
    --n_epochs ${n_epochs} \
    --log_dir ${log_dir}_${nblock}_${seed} \
    --log_interval ${log_interval} \
    --save_interval ${save_interval} \
    --patience ${patience} \
    --seed ${seed}
done

exit

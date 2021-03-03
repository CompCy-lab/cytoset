#!/bin/bash

h_dim=128
in_dim=33
pool=max
out_pool=max
nblock=1
ncell_list=("1024" "512" "256")
nsubset=20000
co_factor=5

train_fcs_info=/home/haidyi/Desktop/Data/CytoSet/AML/train/train_labels.csv
test_fcs_info=/home/haidyi/Desktop/Data/CytoSet/AML/test/test_labels.csv
markerfile=/home/haidyi/Desktop/Data/CytoSet/AML/marker.csv

seed_list=("1" "2" "3" "4" "5")
lr=0.0001
wts_decay=0.001
batch_size=200
n_epochs=20
log_dir=/home/haidyi/Desktop/Data/CytoSet/exp/AML
log_interval=1
save_interval=1
patience=5

bin_file=../../train.py
gpu=$1


for ncell in ${ncell_list[@]}
do
  echo "ncell: $ncell"
  for seed in ${seed_list[@]}
  do
    CUDA_VISIBLE_DEVICES=${gpu} python ${bin_file} \
      --in_dim ${in_dim} \
      --h_dim ${h_dim} \
      --pool ${pool} \
      --out_pool ${out_pool} \
      --nblock ${nblock} \
      --ncell ${ncell} \
      --nsubset ${nsubset} \
      --co_factor ${co_factor} \
      --train_fcs_info ${train_fcs_info} \
      --test_fcs_info ${test_fcs_info} \
      --markerfile ${markerfile} \
      --lr ${lr} \
      --wts_decay ${wts_decay} \
      --generate_valid \
      --test_rsampling \
      --shuffle \
      --batch_size ${batch_size} \
      --n_epochs ${n_epochs} \
      --log_dir ${log_dir}_${ncell}_${nblock}_${seed} \
      --log_interval ${log_interval} \
      --save_interval ${save_interval} \
      --patience ${patience} \
      --seed ${seed}
  done
done

exit

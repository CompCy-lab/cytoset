#!/bin/bash

fcs_data_dir=../../Data/raw/HIV
fcs_info_file=${fcs_data_dir}/attachments/clinical_data_flow_repository.csv
markers=FSC-A,FSC-H,SSC-A,KI67,CD3,CD28,CD45RO,CD8,CD4,CD57,CD14s,CCR5,CD19,CD27,CCR7,CD127
train_markers=FSC-A,FSC-H,SSC-A,KI67,CD3,CD28,CD45RO,CD8,CD4,CD57,CD14s,CCR5,CD19,CD27,CCR7,CD127
id_name=ID
label_name=Death
pos_label=1
neg_label=0
train_prop=0.8
out_dir=./HIV
seed=12345

python pre_process.py \
  --fcs_data_dir ${fcs_data_dir} \
  --fcs_info_file ${fcs_info_file} \
  --marker ${markers} \
  --train_marker ${train_markers} \
  --id_name ${id_name} \
  --label_name ${label_name} \
  --pos ${pos_label} \
  --neg ${neg_label} \
  --train_prop ${train_prop} \
  --out_dir ${out_dir} \
  --seed ${seed}

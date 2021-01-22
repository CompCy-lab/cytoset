#!/bin/bash

fcs_data_dir=../Data/HEUvsUE
fcs_info_file=${fcs_data_dir}/attachments/HEUvsUE.csv
markers=FSC-A,SSC-A,IFN-a,CD123,MHCII,CD14,CD11c,IL-6,IL-12,TNF-a,Time
id_name='FCS file'
label_name=Condition
pos_label=UE
neg_label=HEU
train_prop=0.8
out_dir=./HEUvsUE
seed=12345

python pre_process.py \
  --fcs_data_dir ${fcs_data_dir} \
  --fcs_info_file ${fcs_info_file} \
  --marker ${markers} \
  --id_name "${id_name}" \
  --label_name ${label_name} \
  --pos ${pos_label} \
  --neg ${neg_label} \
  --train_prop ${train_prop} \
  --out_dir ${out_dir} \
  --seed ${seed}

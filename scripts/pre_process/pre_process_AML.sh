#!/bin/bash

fcs_data_dir=./AML_processed
fcs_info_file=${fcs_data_dir}/sample_with_labels.csv
markers=IgG1-FITC,IgG1-PE,CD45-ECD-1,IgG1-PC5,IgG1-PC7,Kappa-FITC,Lambda-PE,CD45-ECD-2,CD19-PC5,CD20-PC7,CD7-FITC,CD4-PE,CD45-ECD-3,CD8-PC5,CD2-PC7,CD15-FITC,CD13-PE,CD45-ECD-4,CD16-PC5,CD56-PC7,CD14-FITC,CD11c-PE,CD45-ECD-5,CD64-PC5,CD33-PC7,HLA-DR-FITC,CD117-PE,CD45-ECD-6,CD34-PC5,CD38-PC7,CD5-FITC,CD19-PE,CD45-ECD-7,CD3-PC5,CD10-PC7,FL1-Log,FL2-Log,7AAD,FL4-Log,FL5-Log
train_markers=IgG1-FITC,IgG1-PE,CD45-ECD-1,IgG1-PC5,IgG1-PC7,CD45-ECD-2,CD19-PC5,CD20-PC7,CD7-FITC,CD4-PE,CD45-ECD-3,CD8-PC5,CD2-PC7,CD15-FITC,CD13-PE,CD45-ECD-4,CD16-PC5,CD56-PC7,CD14-FITC,CD11c-PE,CD45-ECD-5,CD64-PC5,CD33-PC7,HLA-DR-FITC,CD117-PE,CD45-ECD-6,CD34-PC5,CD38-PC7,CD5-FITC,CD19-PE,CD45-ECD-7,CD3-PC5,CD10-PC7
id_name=FCS_file
label_name=Condition
pos_label=aml
neg_label=normal
train_prop=0.8
out_dir=./AML
seed=12345

python pre_process.py \
  --fcs_data_dir ${fcs_data_dir} \
  --fcs_info_file ${fcs_info_file} \
  --marker ${markers} \
  --train_marker ${train_markers} \
  --id_name "${id_name}" \
  --label_name ${label_name} \
  --pos ${pos_label} \
  --neg ${neg_label} \
  --train_prop ${train_prop} \
  --out_dir ${out_dir} \
  --seed ${seed}

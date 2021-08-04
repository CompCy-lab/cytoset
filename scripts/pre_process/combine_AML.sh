#!/bin/bash

fcs_data_dir=/home/haidyi/Desktop/Data/CytoSet/raw/AML
fcs_info_file=${fcs_data_dir}/attachments/AML.csv
id_name='FCS file'
label_name=Condition
individual_name=Individual
sub_individual_name='Tube number'
marker=FL1-1-Log,FL2-1-Log,FL3-1-Log,FL4-1-Log,FL5-1-Log,FL1-2-Log,FL2-2-Log,FL3-2-Log,FL4-2-Log,FL5-2-Log,FL1-3-Log,FL2-3-Log,FL3-3-Log,FL4-3-Log,FL5-3-Log,FL1-4-Log,FL2-4-Log,FL3-4-Log,FL4-4-Log,FL5-4-Log,FL1-5-Log,FL2-5-Log,FL3-5-Log,FL4-5-Log,FL5-5-Log,FL1-6-Log,FL2-6-Log,FL3-6-Log,FL4-6-Log,FL5-6-Log,FL1-7-Log,FL2-7-Log,FL3-7-Log,FL4-7-Log,FL5-7-Log,FL1-8-Log,FL2-8-Log,FL3-8-Log,FL4-8-Log,FL5-8-Log
opt_marker=IgG1-FITC,IgG1-PE,CD45-ECD-1,IgG1-PC5,IgG1-PC7,Kappa-FITC,Lambda-PE,CD45-ECD-2,CD19-PC5,CD20-PC7,CD7-FITC,CD4-PE,CD45-ECD-3,CD8-PC5,CD2-PC7,CD15-FITC,CD13-PE,CD45-ECD-4,CD16-PC5,CD56-PC7,CD14-FITC,CD11c-PE,CD45-ECD-5,CD64-PC5,CD33-PC7,HLA-DR-FITC,CD117-PE,CD45-ECD-6,CD34-PC5,CD38-PC7,CD5-FITC,CD19-PE,CD45-ECD-7,CD3-PC5,CD10-PC7,FL1-Log,FL2-Log,7AAD,FL4-Log,FL5-Log
marker_idx=2,3,4,5,6
combine_axis=column
nproc=6
out_dir=./AML_processed

python multiproc_combine_fcs.py \
  --fcs_data_dir ${fcs_data_dir} \
  --fcs_info_file ${fcs_info_file} \
  --id_name "${id_name}" \
  --label_name ${label_name} \
  --individual_name ${individual_name} \
  --sub_individual_name "${sub_individual_name}" \
  --marker ${marker} \
  --opt_marker ${opt_marker} \
  --marker_idx ${marker_idx} \
  --combine_axis ${combine_axis} \
  --nprocs ${nproc} \
  --out_dir ${out_dir}

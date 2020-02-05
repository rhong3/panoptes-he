#!/bin/bash
location_of_py_file="Main.py" #full path to Main.py
mode="train" #train/validate/test
output_dir="UI_test" #output file directory
batchsize=24 #default 24 (required)
architecture="P2" #P1/P2/P3/P4/PC1/PC2/PC3/PC4
feature="histology" #feature to predict; see README
epoch=100000 #default 100000 (required)
modeltoload="../path/to/trained/model" #only for validate and test
imagefile="../path/to/image/to/test" #only for test
resolution="NA" #default NA (required)
BMI="" #required for test
age="" #required for test
label_file="../gui_test.csv" #required for train and validate
split_file="NA" #required for train and validate (default None)


python3 ${location_of_py_file} \
--mode ${mode} \
--out_dir ${output_dir} \
--batchsize ${batchsize} \
--architecture ${architecture} \
--feature ${feature} \
--epoch ${epoch} \
--modeltoload ${modeltoload} \
--imagefile ${imagefile} \
--resolution ${resolution} \
--label_file ${label_file} \
--split_file ${split_file}
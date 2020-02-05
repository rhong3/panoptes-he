# **Panoptes for cancer H&E image prediction and visualization (PyPI version)**
panoptes is a InceptionResnet-based multi-resolution CNN architecture for cancer H&E histopathological image features 
prediction. It is initially created to predict visualize features of endometrial carcinoma (UCEC), hoping to automate
and assist gynecologic pathologists making quicker and accurate decisions and diagnosis without sequencing analyses.
Details can be found in the paper: 
It can also be applied to other cancer types. 
### Features included 
Currently, it includes training/validating/testing of following features of endometrial cancer:
 - 18 mutations (ARID1A, ATM, BRCA2, CTCF, CTNNB1, FAT1, FBXW7, FGFR2, JAK1, KRAS, MTOR, 
 PIK3CA, PIK3R1, PPP2R1A, PTEN, RPL22, TP53, ZFHX3)
 - 4 molecular subtypes (CNV.H, CNV.L, MSI, POLE); please type "subtype" in bash script. if you want to predict only 1
 subtype (eg. CNV.H), then type "CNV.H". 
 - Histological subtypes (Endometrioid, Serous); please type "histology" in bash script
### Modes
 - train: training a new model from scratch. 
 - validate: load a trained model and validate it on a set of prepared samples.
 - test: load a trained model and apply to a intact H&E slide.
### Variants
PC are architectures with the branch integrating BMI and age; P are original Panoptes
 - Panoptes1 (InceptionResnetV1-based; P1/PC1) 
 - Panoptes2 (InceptionResnetV2-based; P2/PC2) 
 - Panoptes3 (InceptionResnetV1-based; P3/PC3) 
 - Panoptes2 (InceptionResnetV2-based; P4/PC4)
### Usage
 - Create folders `Results`, `images`, `tiles` under the main folder `Panoptes` if they are not there. 
 - Install the package version through pip `pip install panoptes-he `
 - Requirements are listed in `requirements.txt`
 - Scanned H&E slide files should be put in `images` folder under `Panoptes` folder
 - For train and validate mode, label file must also be provided. Example can be found in `sample_lable.csv`
 - For train and validate mode, random split is default for data separation. If you prefer a customized split, please
 provide a split file (example can be found in `sample_sep_file.csv`)
 - Output will be in a folder with name of your choice under `Results` folder.
 - In validate and test, to load pre-trained model, please enter the full path to it. 
 - In test, please enter the full path to the image to test. 
 - For bash input, please refer to example at `Scripts/sample_bash_script.sh`
 - If downloaded from this repository, double click the Panoptes.app to run (for Mac with GUI only). Or you can run 
 `python Main.py` in the main folder. 

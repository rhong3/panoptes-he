# **Panoptes for cancer H&E image prediction and visualization (PyPI version)**
panoptes is a InceptionResnet-based multi-resolution CNN architecture for cancer H&E histopathological image features 
prediction. It is initially created to predict visualize features of endometrial carcinoma (UCEC), hoping to automate
and assist gynecologic pathologists making quicker and accurate decisions and diagnosis without sequencing analyses.
Details can be found in the paper: https://doi.org/10.1016/j.xcrm.2021.100400
It can also be applied to other cancer types. 
### Features included 
Currently, it includes training/validating/testing of following features of endometrial cancer:
 - 18 mutations (ARID1A, ATM, BRCA2, CTCF, CTNNB1, FAT1, FBXW7, FGFR2, JAK1, KRAS, MTOR, 
 PIK3CA, PIK3R1, PPP2R1A, PTEN, RPL22, TP53, ZFHX3)
 - 4 molecular subtypes (CNV.H, CNV.L, MSI, POLE). if you want to predict only 1 subtype (eg. CNV.H), 
 then type "CNV.H". 
 - Histological subtypes (Endometrioid, Serous)
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
 - Install the package version through pip `pip install panoptes-he `
 - Requirements are listed in `requirements.txt`
 - `from panoptes.execute import panoptes, get_default_data`
 - Get default label_file and split_file: `label_file, split_file = get_default_data()`
### Parameters 
`panoptes(mode, outdir, feature, architecture, log_dir, image_dir, tile_dir=None, modeltoload=None,
             imagefile=None, batchsize=24, epoch=100000, resolution=None, BMI=np.nan, age=np.nan, label_file=None,
             split_file=None, cancer='UCEC')`
 - mode(required): select a mode to use (train, validate, test)
 - outdir(required): name of the output directory
 - feature(required): select a feature to predict (histology, subtype, POLE, MSI, CNV.L, CNV.H, ARID1A, ATM, BRCA2, 
 CTCF, CTNNB1, FAT1, FBXW7, FGFR2, JAK1, KRAS, MTOR, PIK3CA, PIK3R1, PPP2R1A, PTEN, RPL22, TP53, ZFHX3)
 - architecture(required): select a architecture to use (P1, P2, P3, P4, PC1, PC2, PC3, PC4)
 - log_dir(required): directory for log file, which will also contain the outdir.
 - image_dir(required): directory contains the svs/scn scanned H&E slides
 - tile_dir(required for train and validate): directory contains tiles of svs/scn scanned H&E slides
 - modeltoload(required for validate and test): full path to trained model to load (without .meta)
 - imagefile(required for test): the svs/scn scanned H&E slide to be tested in image_dir
 - batchsize(optional, default=24): batch size
 - epoch(optional, default=100): max epoch; early stop is enabled.
 - resolution(optional): resolution of scanned H&E slides. If known, enter 20 or 40. Otherwise, leave blank. 
 - BMI(optional): patient BMI for test with PC models.
 - age(optional): patient age for test with PC models.
 - label_file(required for train and validate): label dictionary. Example can be used `sample_lable.csv`. 
 - split_file(optional): For train and validate mode, random split is default for data separation. 
 - cancer(currently optional): Currently only UCEC is supported.
 If you prefer a customized split, please provide a split file (example can be used `sample_sep_file.csv`)
 
Main repository and commandline, bash, GUI version can be found in: https://github.com/rhong3/Panoptes
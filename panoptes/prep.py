"""
Input preparation functions; GUI input; tfrecords preparation; weight calculation.

Created on 01/21/2020

@author: RH
"""
import os
import sys
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import argparse
import easygui
import staintools
import panoptes.data_input as data_input
import panoptes.Slicer as Slicer
import panoptes.sample_prep as sample_prep


# get inputs; if GUI available, use GUI; otherwise, switch to interactive command line automatically;
# non-interactive input script submission also available
def input_handler():
    try:
        # Box1 consent
        msg = "Hello! Hola! Bonjour! Ciao! I'm Panoptes GUI." \
              "By clicking continue, you agree with my terms and conditions. " \
              "Do you want to continue?"
        title = "Please Confirm"
        if easygui.ccbox(msg, title):  # show a Continue/Cancel dialog
            pass  # user chose Continue
        else:  # user chose Cancel
            sys.exit(0)
        # Box2 train/validation/test
        msg = "How may I help you? (Train a new model/Validate a model/Test to predict a new sample)"
        choices = ['train', 'validate', 'test']
        mode = easygui.buttonbox(msg, choices=choices)
        # Box3 output directory name
        out_dir = None
        while out_dir is None:
            msg = "Okay! Where would you like your results to go? (Name a folder under 'Results' directory)"
            out_dir = easygui.enterbox(msg)
            if out_dir == '': out_dir = None
        # Box4 feature to predict
        msg = "What would you like to predict today?"
        title = "Select a feature to predict"
        choices = ["histology", "subtype", "POLE", "MSI", "CNV.L", "CNV.H", "ARID1A",
                   "ATM", "BRCA2", "CTCF", "CTNNB1", "FAT1", "FBXW7", "FGFR2", "JAK1", "KRAS", "MTOR", "PIK3CA",
                   "PIK3R1", "PPP2R1A", "PTEN", "RPL22", "TP53", "ZFHX3"]
        feature = easygui.choicebox(msg, title, choices)
        # Box5 architecture to use
        msg = "Which architecture do you want to use?"
        title = "Select an architecture"
        choices = ["P1", "P2", "P3", "P4", "PC1", "PC2", "PC3", "PC4"]
        architecture = easygui.choicebox(msg, title, choices)
        # Box6&7
        if mode == "validate":
            # Box6 full path to pretrained model
            modeltoload = None
            msg = "I need some more information to proceed. " \
                  "Please enter the full path to trained model to be loaded (without .meta)." \
                  "Please make sure that the model to be loaded is of the same architecture you chose."
            while modeltoload is None:
                modeltoload  = easygui.enterbox(msg)
                if not os.path.isfile('.'.join([modeltoload, 'meta'])):
                    msg = "Invalid Input! Try again" \
                          "Please enter the full path to trained model to be loaded (without .meta)." \
                          "Please make sure that the model to be loaded is of the same architecture you chose."
                    modeltoload = None
            imagefile = None
            BMI =  None
            age = None
            # Box7.5.3 label file
            msg = "Please input full path to label file (ENTER to skip):"
            label_file = easygui.enterbox(msg)
            # Box7.5.4 customized sample split
            msg = "Please input full path to sample split file (ENTER to skip):"
            split_file = easygui.enterbox(msg)
            if not os.path.isfile(label_file): label_file = None
            if not os.path.isfile(split_file): split_file = None
        elif mode == "test":
            # Box6 full path to pretrained model
            modeltoload = None
            msg = "I need some more information to proceed. " \
                  "Please enter the full path to trained model to be loaded (without .meta)." \
                  "Please make sure that the model to be loaded is of the same architecture you chose."
            while modeltoload is None:
                modeltoload = easygui.enterbox(msg)
                if not os.path.isfile('.'.join([modeltoload, 'meta'])):
                    msg = "Invalid Input! Try again" \
                          "Please enter the full path to trained model to be loaded (without .meta)." \
                          "Please make sure that the model to be loaded is of the same architecture you chose."
                    modeltoload = None
            # Box7 image to predict
            msg = "Select the slide you want to predict?"
            title = "Select a slide in the 'images' folder."
            choices = [f for f in os.listdir("../images")]
            imagefile = easygui.choicebox(msg, title, choices)
            # Box7.5.1 patient BMI if known
            msg = "Enter the patient's BMI if known (ENTER to skip)"
            BMI = easygui.enterbox(msg)
            # Box7.5.2 patient age if known
            msg = "Enter the patient's age if known (ENTER to skip)"
            age = easygui.enterbox(msg)
            if BMI == '': BMI = np.nan
            if age == '': age = np.nan
            label_file = None
            split_file = None
        else:
            modeltoload = None
            imagefile = None
            BMI = None
            age = None
            # Box7.5.3 label file
            msg = "Please input full path to label file (ENTER to skip):"
            label_file = easygui.enterbox(msg)
            # Box7.5.4 customized sample split
            msg = "Please input full path to sample split file (ENTER to skip):"
            split_file = easygui.enterbox(msg)
            if not os.path.isfile(label_file): label_file = None
            if not os.path.isfile(split_file): split_file = None
        # Box8 default hyperparameters (batch size, max epoch, max image resolution)
        msg = "Almost there! Do you agree with our default batch size and max epoch number?" \
              "Batch_size = 24; max epoch number = infinity; Max resolution of original slides = None"
        title = "Please Confirm"
        if easygui.ccbox(msg, title):  # show a Continue/Cancel dialog
            batchsize = 24
            epoch = 100000
            resolution = None
            pass  # user chose Continue
        else:  # user chose Cancel; ask for their choices of hyperparameters
            # Box9
            msg = "Please enter your choice (integer only)"
            title = "Enter your choice"
            fieldNames = ["batch size", "max epoch", "max resolution"]
            fieldValues = []  # we start with blanks for the values
            fieldValues = easygui.multenterbox(msg, title, fieldNames)
            # make sure that none of the fields was left blank
            while 1:
                if fieldValues is None: break
                errmsg = ""
                for i in range(len(fieldNames)):
                    if fieldValues[i].strip() == "":
                        errmsg = errmsg + ('"%s" is a required field.\n\n' % fieldNames[i])
                if errmsg == "": break  # no problems found
                fieldValues = easygui.multenterbox(errmsg, title, fieldNames, fieldValues)
            batchsize = fieldValues[0]
            epoch = fieldValues[1]
            resolution = fieldValues[2]

    except Exception as e:  # NON-GUI INPUT
        # non-interactive submission scripts
        parser = argparse.ArgumentParser(description="Parse some arguments")
        parser.add_argument('--mode', type=str, choices=['train', 'validate', 'test'])
        parser.add_argument('--out_dir', type=str)
        parser.add_argument('--batchsize', type=int)
        parser.add_argument('--architecture', type=str)
        parser.add_argument('--feature', type=str)
        parser.add_argument('--epoch', type=int)
        parser.add_argument('--modeltoload', type=str, default="NA")
        parser.add_argument('--imagefile', type=str)
        parser.add_argument('--resolution', type=str)
        parser.add_argument('--BMI', type=float)
        parser.add_argument('--age', type=float)
        parser.add_argument('--label_file', type=str)
        parser.add_argument('--split_file', type=str)

        args = parser.parse_args()

        mode = args.mode
        out_dir = args.out_dir
        batchsize = args.batchsize
        architecture = args.architecture
        feature = args.feature
        epoch = args.epoch
        modeltoload = args.modeltoload
        imagefile = args.imagefile
        resolution = args.resolution
        BMI = args.BMI
        age = args.age
        label_file = args.label_file
        split_file = args.split_file

        # check for invalid non-interactive input
        if mode not in ['train', 'validate', 'test']:
            mode = None
        if feature not in ["histology", "subtype", "POLE", "MSI", "CNV.L", "CNV.H",
                           "ARID1A", "ATM", "BRCA2", "CTCF", "CTNNB1", "FAT1", "FBXW7", "FGFR2", "JAK1", "KRAS",
                           "MTOR", "PIK3CA", "PIK3R1", "PPP2R1A", "PTEN", "RPL22", "TP53", "ZFHX3"]:
            feature = None
        if architecture not in ["P1", "P2", "P3", "P4", "PC1", "PC2", "PC3", "PC4"]:
            architecture = None
        if mode != "train":
            if not os.path.isfile(modeltoload):
                modeltoload = None
        if mode == "test":
            if imagefile not in [f for f in os.listdir("../images")]:
                imagefile = None
        # enter mode (train/validation/test)
        while mode is None:
            mode = input("Please input a mode (train/validation/test): ")
            if mode not in ['train', 'validate', 'test']:
                print("Invalid input! Try again!")
                mode = None
        # enter output directory
        while out_dir is None:
            out_dir = input("Please input a directory name for outputs (under 'Results' directory): ")
            if out_dir == '': out_dir = None
        # enter feature to predict
        while feature is None:
            feature = input("Please input a feature to predict: ")
            if feature not in ["histology", "subtype", "POLE", "MSI", "CNV.L", "CNV.H",
                               "ARID1A", "ATM", "BRCA2", "CTCF", "CTNNB1", "FAT1", "FBXW7", "FGFR2", "JAK1", "KRAS",
                               "MTOR", "PIK3CA", "PIK3R1", "PPP2R1A", "PTEN", "RPL22", "TP53", "ZFHX3"]:
                print("Invalid input! Try again!")
                feature = None
        # enter architecture to use
        while architecture is None:
            architecture = input("Please input an architecture to use: ")
            if architecture not in ["P1", "P2", "P3", "P4", "PC1", "PC2", "PC3", "PC4"]:
                print("Invalid input! Try again!")
                architecture = None
        # enter pretrained model
        while modeltoload is None and mode != "train":
            modeltoload = str(input("Please input full path to trained model to load (without .meta): ")) or None
            if modeltoload is not None:
                if not os.path.isfile('.'.join([modeltoload, 'meta'])):
                    print("Invalid path! Try again!")
                    modeltoload = None
        # enter image file to predict
        while imagefile is None and mode == "test":
            imagefile = str(input("Please input a slide to predict (in 'images' directory): ")) or None
            if imagefile not in [f for f in os.listdir("../images")]:
                print("Invalid Image! Try again!")
                imagefile = None
        # enter hyperparameters
        if batchsize is None: batchsize = input("Please input batch size (DEFAULT=24; ENTER to skip): ") or 24
        if epoch is None: epoch = input("Please input epoch size (DEFAULT=infinity; ENTER to skip): ") or 100000
        if resolution is None:
            resolution = input("Please input the max resolution of slides (ENTER to skip): ") or None
        # enter BMI and age if known
        if mode == "test":
            if not isinstance(BMI, float): BMI = input("Please input patient BMI (ENTER to skip): ") or np.nan
            if not isinstance(age, float): age = input("Please input patient age (ENTER to skip): ") or np.nan
        # enter label file
        if label_file is None and mode != "test":
            label_file = str(input("Please input full path to label file (ENTER to skip): ")) or None
            if label_file is not None:
                if not os.path.isfile(label_file):
                    print("Invalid label file! Default will be used.")
                    label_file = "NA"
            else:
                label_file = "NA"
        # enter customized sample split file
        if split_file is None and mode != "test":
            split_file = str(input("Please input full path to sample split file (ENTER to skip): ")) or None
            if split_file is not None:
                if not os.path.isfile(split_file):
                    print("Invalid split file! Random split will be used.")
                    split_file = "NA"
            else:
                split_file = "NA"

    return mode, out_dir, feature, architecture, modeltoload, imagefile, batchsize, epoch, resolution, \
           BMI, age, label_file, split_file


# count numbers of training and testing tiles
def counters(totlist_dir, cls):
    trlist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    telist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    valist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    trcc = len(trlist['label'])
    tecc = len(telist['label'])
    vacc = len(valist['label'])
    weigh = []
    for i in range(cls):
        ccct = len(trlist.loc[trlist['label'] == i])+len(valist.loc[valist['label'] == i])\
               + len(telist.loc[telist['label'] == i])
        wt = ((trcc+tecc+vacc)/cls)/ccct
        weigh.append(wt)
    weigh = tf.constant(weigh)
    return trcc, tecc, vacc, weigh


# read images
def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img


# used for tfrecord float generation
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


# used for tfrecord labels generation
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# used for tfrecord images generation
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# generate tfrecords for real test image
def testloader(data_dir, imgg, resolution, BMI, age):
    slist = sample_prep.testpaired_tile_ids_in(imgg, data_dir, resolution=resolution)
    slist.insert(loc=0, column='Num', value=slist.index)
    slist.insert(loc=4, column='BMI', value=BMI)
    slist.insert(loc=4, column='age', value=age)
    slist.to_csv(data_dir + '/te_sample.csv', header=True, index=False)
    imlista = slist['L0path'].values.tolist()
    imlistb = slist['L1path'].values.tolist()
    imlistc = slist['L2path'].values.tolist()
    wtlist = slist['BMI'].values.tolist()
    aglist = slist['age'].values.tolist()
    filename = data_dir + '/test.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(imlista)):
        try:
            # Load the image
            imga = load_image(imlista[i])
            imgb = load_image(imlistb[i])
            imgc = load_image(imlistc[i])
            wt = wtlist[i]
            ag = aglist[i]
            # Create a feature
            feature = {'test/BMI': _float_feature(wt),
                       'test/age': _float_feature(ag),
                       'test/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       'test/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       'test/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image: ' + imlista[i] + '~' + imlistb[i] + '~' + imlistc[i])
            pass

    writer.close()


# loading images for dictionaries and generate tfrecords
def loader(totlist_dir, ds):
    if ds == 'train':
        slist = pd.read_csv(totlist_dir + '/tr_sample.csv', header=0)
    elif ds == 'validation':
        slist = pd.read_csv(totlist_dir + '/va_sample.csv', header=0)
    elif ds == 'test':
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    else:
        slist = pd.read_csv(totlist_dir + '/te_sample.csv', header=0)
    imlista = slist['L0path'].values.tolist()
    imlistb = slist['L1path'].values.tolist()
    imlistc = slist['L2path'].values.tolist()
    lblist = slist['label'].values.tolist()
    wtlist = slist['BMI'].values.tolist()
    aglist = slist['age'].values.tolist()
    filename = totlist_dir + '/' + ds + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(len(lblist)):
        try:
            # Load the image
            imga = load_image(imlista[i])
            imgb = load_image(imlistb[i])
            imgc = load_image(imlistc[i])
            label = lblist[i]
            wt = wtlist[i]
            ag = aglist[i]
            # Create a feature
            feature = {ds + '/label': _int64_feature(label),
                       ds + '/BMI': _float_feature(wt),
                       ds + '/age': _float_feature(ag),
                       ds + '/imageL0': _bytes_feature(tf.compat.as_bytes(imga.tostring())),
                       ds + '/imageL1': _bytes_feature(tf.compat.as_bytes(imgb.tostring())),
                       ds + '/imageL2': _bytes_feature(tf.compat.as_bytes(imgc.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
        except AttributeError:
            print('Error image: ' + imlista[i] + '~' + imlistb[i] + '~' + imlistc[i])
            pass
    writer.close()


# load tfrecords and prepare datasets
def tfreloader(mode, ep, bs, cls, ctr, cte, cva, data_dir):
    filename = data_dir + '/' + mode + '.tfrecords'
    if mode == 'train':
        ct = ctr
    elif mode == 'test':
        ct = cte
    else:
        ct = cva

    datasets = data_input.DataSet(bs, ct, ep=ep, cls=cls, mode=mode, filename=filename)

    return datasets


# check images to be cut (not in tiles)
def check_new_image(ref_file, tiledir="../tiles"):
    todolist=[]
    existed = os.listdir(tiledir)
    for idx, row in ref_file.iterrows():
        if row['patient'] not in existed:
            todolist.append([str(row['patient']), str(row['sld']), str(row['sld_num'])])
    return todolist


# cutting image into tiles
def cutter(img, outdirr, dp=None, resolution=None):
    try:
        os.mkdir(outdirr)
    except(FileExistsError):
        pass
    # load standard image for normalization
    std = staintools.read_image("../colorstandard.png")
    std = staintools.LuminosityStandardizer.standardize(std)
    if resolution == 20:
        for m in range(1, 4):
            level = int(m / 2)
            tff = int(m % 2 + 1)
            otdir = "{}/level{}".format(outdirr, str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                numx, numy, raw, tct = Slicer.tile(image_file=img, outdir=otdir,
                                                   level=level, std_img=std, ft=tff, dp=dp)
            except Exception as e:
                print('Error!')
                pass
    elif resolution == 40:
        for m in range(1, 4):
            level = int(m / 3 + 1)
            tff = int(m / level)
            otdir = "{}/level{}".format(outdirr, str(m))
            try:
                os.mkdir(otdir)
            except(FileExistsError):
                pass
            try:
                numx, numy, raw, tct = Slicer.tile(image_file=img, outdir=otdir,
                                                   level=level, std_img=std, ft=tff, dp=dp)
            except Exception as e:
                print('Error!')
                pass
    else:
        if "TCGA" in img:
            for m in range(1, 4):
                level = int(m / 3 + 1)
                tff = int(m / level)
                otdir = "{}/level{}".format(outdirr, str(m))
                try:
                    os.mkdir(otdir)
                except(FileExistsError):
                    pass
                try:
                    numx, numy, raw, tct = Slicer.tile(image_file=img, outdir=otdir,
                                                                             level=level, std_img=std, ft=tff, dp=dp)
                except Exception as e:
                    print('Error!')
                    pass
        else:
            for m in range(1, 4):
                level = int(m / 2)
                tff = int(m % 2 + 1)
                otdir = "{}/level{}".format(outdirr, str(m))
                try:
                    os.mkdir(otdir)
                except(FileExistsError):
                    pass
                try:
                    numx, numy, raw, tct = Slicer.tile(image_file=img, outdir=otdir,
                                                                             level=level, std_img=std, ft=tff, dp=dp)
                except Exception as e:
                    print('Error!')
                    pass

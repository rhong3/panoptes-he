"""
Input preparation functions; GUI input; tfrecords preparation; weight calculation.

Created on 01/21/2020

@author: RH
"""
import os
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
import staintools
import panoptes.data_input as data_input
import panoptes.Slicer as Slicer
import panoptes.sample_prep as sample_prep


def valid_input(mode, outdir, feature, architecture, log_dir, tile_dir, image_dir, modeltoload, imagefile,
                     batchsize, epoch, resolution, BMI, age, label_file, split_file):
    if mode not in ['train', 'validate', 'test']:
        print("Invalid mode!")
        exit(0)
    if not isinstance(outdir, str) or not isinstance(log_dir, str):
        print("Invalid output or log directory!")
        exit(0)
    if feature not in ["histology", "subtype", "POLE", "MSI", "CNV.L", "CNV.H", "ARID1A", "ATM", "BRCA2", "CTCF",
                       "CTNNB1", "FAT1", "FBXW7", "FGFR2", "JAK1", "KRAS", "MTOR", "PIK3CA", "PIK3R1", "PPP2R1A",
                       "PTEN", "RPL22", "TP53", "ZFHX3"]:
        print("Invalid feature to predict!")
        exit(0)
    if architecture not in ["P1", "P2", "P3", "P4", "PC1", "PC2", "PC3", "PC4"]:
        print("Invalid architecture!")
        exit(0)
    if not isinstance(image_dir, str):
        print("Invalid image directory!")
        exit(0)
    if not isinstance(batchsize, int):
        print("Invalid batch size!")
        exit(0)
    if not isinstance(epoch, int):
        print("Invalid max epoch!")
        exit(0)
    if mode == "test":
        if imagefile is None:
            print("Missing image file!")
            exit(0)
        elif not isinstance(imagefile, str):
            print("Invalid image file!")
            exit(0)
    else:
        if tile_dir is None:
            print("Missing tile directory!")
            exit(0)
        elif not isinstance(tile_dir, str):
            print("Invalid tile directory!")
            exit(0)
        if label_file is None:
            print("Missing label file!")
            exit(0)
        elif not os.path.isfile(label_file):
            print("Invalid label file!")
            exit(0)
        if split_file is not None:
            if not os.path.isfile(split_file):
                print("Invalid split file!")
                exit(0)
    if mode != "train":
        if modeltoload is None:
            print("Missing model to load!")
            exit(0)
        elif not os.path.isfile(modeltoload+".meta"):
            print("Invalid model to load!")
            exit(0)
    if resolution is not None and resolution not in [20, 40]:
        print("Invalid resolution!")
        exit(0)
    if not isinstance(BMI, float):
        print("Invalid BMI!")
        exit(0)
    if not isinstance(age, float):
        print("Invalid age!")
        exit(0)


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
def check_new_image(ref_file, tiledir):
    todolist=[]
    existed = os.listdir(tiledir)
    for idx, row in ref_file.iterrows():
        if row['patient'] not in existed:
            todolist.append([str(row['patient']), str(row['sld']), str(row['sld_num'])])
    return todolist


# cutting image into tiles
def cutter(img, outdirr, imgdir, dp=None, resolution=None):
    try:
        os.mkdir(outdirr)
    except(FileExistsError):
        pass
    import panoptes
    # load standard image for normalization
    std = staintools.read_image("{}/colorstandard.png".format(panoptes.__path__[0]))
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
                                                   level=level, std_img=std, ft=tff, dp=dp, path_to_slide=imgdir)
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
                                                   level=level, std_img=std, ft=tff, dp=dp, path_to_slide=imgdir)
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
                                                       level=level, std_img=std, ft=tff, dp=dp, path_to_slide=imgdir)
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
                                                       level=level, std_img=std, ft=tff, dp=dp, path_to_slide=imgdir)
                except Exception as e:
                    print('Error!')
                    pass

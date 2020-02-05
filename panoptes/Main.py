"""
Main method

Created on 01/21/2020

@author: RH
"""
import os
import sys
import tensorflow as tf
import pandas as pd
from openslide import OpenSlide
import numpy as np
import cv2
import time
import matplotlib
import prep
import cnn
import sample_prep
matplotlib.use('Agg')


if __name__ == "__main__":
    tf.reset_default_graph()
    # getting input variables
    mode, outdir, feature, architecture, modeltoload, imagefile, batchsize, epoch, resolution, \
    BMI, age, label_file, split_file = prep.input_handler()
    if BMI is not None:
        BMI = float(BMI)
    else:
        BMI =np.nan
    if age is not None:
        age = float(age)
    else:
        age =np.nan
    batchsize = int(batchsize)
    epoch = int(epoch)
    if resolution is not None:
        if resolution != "NA":
            resolution = int(resolution)
        else:
            resolution = None
    if label_file is None or label_file == "NA":
        label_file = '../sample_label.csv'
    if split_file == "NA":
        split_file = None

    print("All set! Your inputs are: ")
    print(["mode: {}".format(mode), "output: {}".format(outdir), "feature: {}".format(feature),
           "architecture: {}".format(architecture), "pretrained model: {}".format(modeltoload),
           "slide to predirct: {}".format(imagefile), "batch size: {}".format(batchsize),
           "maximum epoch: {}".format(epoch), "slide max resolution: {}".format(resolution),
           "patient BMI: {}".format(BMI), "patient age: {}".format(age), "label file: {}".format(label_file),
           "sample split file: {}".format(split_file)], flush=True)

    if architecture in ["PC1", "PC2", "PC3", "PC4"]:
        sup = True
    else:
        sup = False

    if feature == 'subtype':
        classes = 4
    else:
        classes = 2

    # input image dimension
    INPUT_DIM = [batchsize, 299, 299, 3]
    # hyper parameters
    HYPERPARAMS = {
        "batch_size": batchsize,
        "dropout": 0.3,
        "learning_rate": 1E-4,
        "classes": classes,
        "sup": sup
    }

    img_dir = '../tiles/'
    LOG_DIR = "../Results/{}".format(outdir)
    out_dir = "../Results/{}/out".format(outdir)

    if mode == "test":
        start_time = time.time()
        modelname = modeltoload.split(sep='/')[-1]
        modelpath = '/'.join(modeltoload.split(sep='/')[:-1])
        data_dir = "../Results/{}".format(outdir)
        METAGRAPH_DIR = modelpath
        # make directories if not exist
        for DIR in (img_dir, LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
            try:
                os.mkdir(DIR)
            except FileExistsError:
                pass

        if feature == 'histology':
            pos_score = "Serous_score"
            neg_score = "Endometrioid_score"
        else:
            pos_score = "POS_score"
            neg_score = "NEG_score"

        if resolution == 40:
            ft = 1
            level = 1
        elif resolution == 20:
            level = 0
            ft = 2
        else:
            if "TCGA" in imagefile:
                ft = 1
                level = 1
            else:
                level = 0
                ft = 2
        slide = OpenSlide("../images/" + imagefile)

        # Get dimension of slide
        bounds_width = slide.level_dimensions[level][0]
        bounds_height = slide.level_dimensions[level][1]
        x = 0
        y = 0
        half_width_region = 49 * ft
        full_width_region = 299 * ft
        stepsize = (full_width_region - half_width_region)

        # number of tiles can be cut
        n_x = int((bounds_width - 1) / stepsize)
        n_y = int((bounds_height - 1) / stepsize)

        lowres = slide.read_region((x, y), level + 1, (int(n_x * stepsize / 4), int(n_y * stepsize / 4)))
        raw_img = np.array(lowres)[:, :, :3]
        fct = ft

        # cut tiles
        if not os.path.isfile(data_dir + '/level1/dict.csv'):
            prep.cutter(imagefile, LOG_DIR, resolution=resolution)
        # make tfrecords
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            prep.testloader(data_dir, imagefile, resolution, BMI, age)
        # reload pretrained model
        m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=modelname, log_dir=LOG_DIR, meta_dir=METAGRAPH_DIR,
                          model=architecture)
        print("Loaded! Ready for test!")
        # decode tfrecords
        HE = prep.tfreloader(mode, 1, batchsize, classes, None, None, None, data_dir)
        # prediction
        m.inference(HE, outdir, realtest=True, bs=batchsize, pmd=feature)
        # load tiles dictionary
        slist = pd.read_csv(data_dir + '/te_sample.csv', header=0)
        # load dictionary of predictions on tiles
        teresult = pd.read_csv(out_dir + '/Test.csv', header=0)
        # join 2 dictionaries
        joined = pd.merge(slist, teresult, how='inner', on=['Num'])
        joined = joined.drop(columns=['Num'])
        tile_dict = pd.read_csv(data_dir + '/level1/dict.csv', header=0)
        tile_dict = tile_dict.rename(index=str, columns={"Loc": "L0path"})
        joined_dict = pd.merge(joined, tile_dict, how='inner', on=['L0path'])
        # slide level prediction
        if joined_dict[pos_score].mean() > 0.5:
            print("Positive! Prediction score = " + str(joined_dict[pos_score].mean().round(5)))
        else:
            print("Negative! Prediction score = " + str(joined_dict[pos_score].mean().round(5)))
        # save joined dictionary
        joined_dict.to_csv(out_dir + '/finaldict.csv', index=False)

        # output heat map of pos and neg.
        # initialize a graph and for each RGB channel
        opt = np.full((n_x, n_y), 0)
        hm_R = np.full((n_x, n_y), 0)
        hm_G = np.full((n_x, n_y), 0)
        hm_B = np.full((n_x, n_y), 0)

        # Positive is labeled red in output heat map
        for index, row in joined_dict.iterrows():
            opt[int(row["X_pos"]), int(row["Y_pos"])] = 255
            if row[pos_score] >= 0.5:
                hm_R[int(row["X_pos"]), int(row["Y_pos"])] = 255
                hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row[pos_score] - 0.5) * 2) * 255)
                hm_B[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row[pos_score] - 0.5) * 2) * 255)
            else:
                hm_B[int(row["X_pos"]), int(row["Y_pos"])] = 255
                hm_G[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row[neg_score] - 0.5) * 2) * 255)
                hm_R[int(row["X_pos"]), int(row["Y_pos"])] = int((1 - (row[neg_score] - 0.5) * 2) * 255)

        # expand 5 times
        opt = opt.repeat(50, axis=0).repeat(50, axis=1)

        # small-scaled original image
        ori_img = cv2.resize(raw_img, (np.shape(opt)[0], np.shape(opt)[1]))
        ori_img = ori_img[:np.shape(opt)[1], :np.shape(opt)[0], :3]
        tq = ori_img[:, :, 0]
        ori_img[:, :, 0] = ori_img[:, :, 2]
        ori_img[:, :, 2] = tq
        cv2.imwrite(out_dir + '/Original_scaled.png', ori_img)

        # binary output image
        topt = np.transpose(opt)
        opt = np.full((np.shape(topt)[0], np.shape(topt)[1], 3), 0)
        opt[:, :, 0] = topt
        opt[:, :, 1] = topt
        opt[:, :, 2] = topt
        cv2.imwrite(out_dir + '/Mask.png', opt * 255)

        # output heatmap
        hm_R = np.transpose(hm_R)
        hm_G = np.transpose(hm_G)
        hm_B = np.transpose(hm_B)
        hm_R = hm_R.repeat(50, axis=0).repeat(50, axis=1)
        hm_G = hm_G.repeat(50, axis=0).repeat(50, axis=1)
        hm_B = hm_B.repeat(50, axis=0).repeat(50, axis=1)
        hm = np.dstack([hm_B, hm_G, hm_R])
        cv2.imwrite(out_dir + '/HM.png', hm)

        # superimpose heatmap on scaled original image
        overlay = ori_img * 0.5 + hm * 0.5
        cv2.imwrite(out_dir + '/Overlay.png', overlay)

        # # Time measure tool
        print("--- %s seconds ---" % (time.time() - start_time))

    elif mode == "validate":
        modelname = modeltoload.split(sep='/')[-1]
        data_dir = "../Results/{}/data".format(outdir)
        METAGRAPH_DIR = "../Results/{}".format(outdir)
        # make directories if not exist
        for DIR in (img_dir, LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
            try:
                os.mkdir(DIR)
            except FileExistsError:
                pass
        # check images to be cut
        reff = pd.read_csv(label_file, header=0)
        tocut = prep.check_new_image(reff, img_dir)
        # cut into tiles
        for im in tocut:
            prep.cutter(im[1], img_dir + '/' + im[0], dp=im[2], resolution=resolution)

        # get counts of testing, validation, and training datasets;
        # if not exist, prepare testing and training datasets from sampling; package into tfrecords
        if os.path.isfile(data_dir + '/tr_sample.csv') and os.path.isfile(data_dir + '/te_sample.csv') \
                and os.path.isfile(data_dir + '/va_sample.csv'):
            trc, tec, vac, weights = prep.counters(data_dir, classes)
            trs = pd.read_csv(data_dir + '/tr_sample.csv', header=0)
            tes = pd.read_csv(data_dir + '/te_sample.csv', header=0)
            vas = pd.read_csv(data_dir + '/va_sample.csv', header=0)
        else:
            alll = sample_prep.big_image_sum(pmd=feature, path=img_dir, ref_file=label_file)
            trs, tes, vas = sample_prep.set_sep(alll, path=data_dir, cls=classes, cut=0.2,
                                                resolution=resolution, sep_file=split_file, batchsize=batchsize)
            trc, tec, vac, weights = prep.counters(data_dir, classes)
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            prep.loader(data_dir, 'test')
        if not os.path.isfile(data_dir + '/train.tfrecords'):
            prep.loader(data_dir, 'train')
        if not os.path.isfile(data_dir + '/validation.tfrecords'):
            prep.loader(data_dir, 'validation')
        # reload pretrained model
        m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, meta_graph=modelname, log_dir=LOG_DIR,
                          meta_dir=METAGRAPH_DIR, model=architecture, weights=weights)
        print("Loaded! Ready for test!")
        # validating
        if tec >= batchsize:
            THE = prep.tfreloader('test', 1, batchsize, classes, trc, tec, vac, data_dir)
            m.inference(THE, outdir, testset=tes, pmd=feature)
        else:
            print("Not enough testing images!")

    else:
        data_dir = "../Results/{}/data".format(outdir)
        METAGRAPH_DIR = "../Results/{}".format(outdir)
        # make directories if not exist
        for DIR in (img_dir, LOG_DIR, METAGRAPH_DIR, data_dir, out_dir):
            try:
                os.mkdir(DIR)
            except FileExistsError:
                pass
        # determine images to be cut
        reff = pd.read_csv(label_file, header=0)
        tocut = prep.check_new_image(reff, img_dir)
        # cut images into tiles
        for im in tocut:
            prep.cutter(im[1], img_dir + '/' + im[0], dp=im[2], resolution=resolution)

        # get counts of testing, validation, and training datasets;
        # if not exist, prepare testing and training datasets from sampling; package into tfrecords
        if os.path.isfile(data_dir + '/tr_sample.csv') and os.path.isfile(data_dir + '/te_sample.csv') \
                and os.path.isfile(data_dir + '/va_sample.csv'):
            trc, tec, vac, weights = prep.counters(data_dir, classes)
            trs = pd.read_csv(data_dir + '/tr_sample.csv', header=0)
            tes = pd.read_csv(data_dir + '/te_sample.csv', header=0)
            vas = pd.read_csv(data_dir + '/va_sample.csv', header=0)
        else:
            alll = sample_prep.big_image_sum(pmd=feature, path=img_dir, ref_file=label_file)
            trs, tes, vas = sample_prep.set_sep(alll, path=data_dir, cls=classes, cut=0.2,
                                                resolution=resolution, sep_file=split_file, batchsize=batchsize)
            trc, tec, vac, weights = prep.counters(data_dir, classes)
        if not os.path.isfile(data_dir + '/test.tfrecords'):
            prep.loader(data_dir, 'test')
        if not os.path.isfile(data_dir + '/train.tfrecords'):
            prep.loader(data_dir, 'train')
        if not os.path.isfile(data_dir + '/validation.tfrecords'):
            prep.loader(data_dir, 'validation')
        if sup:
            print("Integrating clinical variables!")
        # prepare to train from scratch
        m = cnn.INCEPTION(INPUT_DIM, HYPERPARAMS, log_dir=LOG_DIR, model=architecture, weights=weights)
        print("Start a new training!")
        # decode training and validation sets
        HE = prep.tfreloader('train', epoch, batchsize, classes, trc, tec, vac, data_dir)
        VHE = prep.tfreloader('validation', epoch*100, batchsize, classes, trc, tec, vac, data_dir)
        itt = int(trc * epoch / batchsize) + 1
        if trc <= 2 * batchsize or vac <= batchsize:
            print("Not enough training/validation images!")
        else:
            # training
            m.train(HE, VHE, trc, batchsize, pmd=feature, dirr=outdir, max_iter=itt, save=True, outdir=METAGRAPH_DIR)
        if tec >= batchsize:
            # internal testing
            THE = prep.tfreloader('test', 1, batchsize, classes, trc, tec, vac, data_dir)
            m.inference(THE, outdir, testset=tes, pmd=feature)
        else:
            print("Not enough testing images!")

    sys.exit(0)


"""
Prepare training and testing datasets as CSV dictionaries

01/21/2020

@author: RH
"""
import os
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re


# pair tiles of 10x, 5x, 2.5x of the same area into tile sets for real test image
def testpaired_tile_ids_in(imgdirr, root_dir, resolution=None):
    if resolution is None:
        if "TCGA" in imgdirr:
            fac = 2000
        else:
            fac = 1000
    else:
        fac = int(resolution * 50)
    ids = []
    for level in range(1, 4):
        dirrr = root_dir + '/level{}'.format(str(level))
        for id in os.listdir(dirrr):
            if '.png' in id:
                x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                y = int(float(re.split('.p', id.split('y-', 1)[1])[0]) / fac)
                ids.append([level, dirrr + '/' + id, x, y])
            else:
                print('Skipping ID:', id)
    ids = pd.DataFrame(ids, columns=['level', 'path', 'x', 'y'])
    idsa = ids.loc[ids['level'] == 1]
    idsa = idsa.drop(columns=['level'])
    idsa = idsa.rename(index=str, columns={"path": "L0path"})
    idsb = ids.loc[ids['level'] == 2]
    idsb = idsb.drop(columns=['level'])
    idsb = idsb.rename(index=str, columns={"path": "L1path"})
    idsc = ids.loc[ids['level'] == 3]
    idsc = idsc.drop(columns=['level'])
    idsc = idsc.rename(index=str, columns={"path": "L2path"})
    idsa = pd.merge(idsa, idsb, on=['x', 'y'], how='left', validate="many_to_many")
    idsa['x'] = idsa['x'] - (idsa['x'] % 2)
    idsa['y'] = idsa['y'] - (idsa['y'] % 2)
    idsa = pd.merge(idsa, idsc, on=['x', 'y'], how='left', validate="many_to_many")
    idsa = idsa.drop(columns=['x', 'y'])
    idsa = idsa.dropna()
    idsa = idsa.reset_index(drop=True)

    return idsa


# pair tiles of 10x, 5x, 2.5x of the same area into tile sets
def paired_tile_ids_in(slide, root_dir, label=None, age=None, BMI=None, resolution=None):
    dira = os.path.isdir(root_dir + 'level1')
    dirb = os.path.isdir(root_dir + 'level2')
    dirc = os.path.isdir(root_dir + 'level3')
    if dira and dirb and dirc:
        if resolution is None:
            if "TCGA" in root_dir:
                fac = 2000
            else:
                fac = 1000
        else:
            fac = int(resolution*50)
        ids = []
        for level in range(1, 4):
            dirr = root_dir + 'level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('_', id.split('y-', 1)[1])[0]) / fac)
                    try:
                        dup = re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0]
                    except IndexError:
                        dup = np.nan
                    ids.append([slide, label, level, dirr + '/' + id, x, y, dup])
                else:
                    print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['slide', 'label', 'level', 'path', 'x', 'y', 'dup'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L0path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['slide', 'label', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L1path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['slide', 'label', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L2path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y', 'dup'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y', 'dup'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
        idsa['age'] = age
        idsa['BMI'] = BMI
    else:
        idsa = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])

    return idsa


# Prepare label at per patient level according to label file
def big_image_sum(pmd, path, ref_file):
    ref = pd.read_csv(ref_file, header=0)
    big_images = []
    if pmd == 'subtype':
        ref = ref.loc[ref['subtype_0NA'] == 0]
        for idx, row in ref.iterrows():
            if row['subtype_POLE'] == 1:
                big_images.append([row['patient'], 0, path + "/{}/".format(str(row['patient'])),
                                   row['age'], row['BMI']])
            elif row['subtype_MSI'] == 1:
                big_images.append([row['patient'], 1, path + "/{}/".format(str(row['patient'])),
                                   row['age'], row['BMI']])
            elif row['subtype_CNV.L'] == 1:
                big_images.append([row['patient'], 2, path + "/{}/".format(str(row['patient'])),
                                   row['age'], row['BMI']])
            elif row['subtype_CNV.H'] == 1:
                big_images.append([row['patient'], 3, path + "/{}/".format(str(row['patient'])),
                                   row['age'], row['BMI']])
    elif pmd == 'histology':
        ref = ref.loc[ref['histology_Mixed'] == 0]
        for idx, row in ref.iterrows():
            if row['histology_Endometrioid'] == 1:
                big_images.append([row['patient'], 0, path + "/{}/".format(str(row['patient'])),
                                   row['age'], row['BMI']])
            if row['histology_Serous'] == 1:
                big_images.append([row['patient'], 1, path + "/{}/".format(str(row['patient'])),
                                   row['age'], row['BMI']])
    elif pmd in ['CNV.L', 'MSI', 'CNV.H', 'POLE']:
        ref = ref.loc[ref['subtype_0NA'] == 0]
        for idx, row in ref.iterrows():
            big_images.append([row['patient'], int(row['subtype_{}'.format(pmd)]),
                               path + "/{}/".format(str(row['patient'])), row['age'], row['BMI']])
    else:
        ref = ref.dropna(subset=[pmd])
        for idx, row in ref.iterrows():
            big_images.append([row['patient'], int(row[pmd]),
                               path + "/{}/".format(str(row['patient'])), row['age'], row['BMI']])

    datapd = pd.DataFrame(big_images, columns=['slide', 'label', 'path', 'age', 'BMI'])
    datapd = datapd.dropna()

    return datapd


# seperate into training and testing (can separate ramdomly or according to prepared dictionary);
# each type is the same separation ratio on big images;
# test and train csv files contain tiles' path.
def set_sep(alll, path, cls, resolution=None, sep_file=None, cut=0.2, batchsize=24):
    trlist = []
    telist = []
    valist = []
    # if no customized split file, ramdomly sampling according to the ratio
    if sep_file is None:
        for i in range(cls):
            subset = alll.loc[alll['label'] == i]
            unq = list(subset.slide.unique())
            np.random.shuffle(unq)
            validation = unq[:int(len(unq) * cut / 2)]
            valist.append(subset[subset['slide'].isin(validation)])
            test = unq[int(len(unq) * cut / 2):int(len(unq) * cut)]
            telist.append(subset[subset['slide'].isin(test)])
            train = unq[int(len(unq) * cut):]
            trlist.append(subset[subset['slide'].isin(train)])
    else:
        # split samples according to split file
        split = pd.read_csv(sep_file, header=0)
        train = split.loc[split['set'] == 'train']['slide'].tolist()
        validation = split.loc[split['set'] == 'validation']['slide'].tolist()
        test = split.loc[split['set'] == 'test']['slide'].tolist()

        valist.append(alll[alll['slide'].isin(validation)])
        telist.append(alll[alll['slide'].isin(test)])
        trlist.append(alll[alll['slide'].isin(train)])

    test = pd.concat(telist)
    train = pd.concat(trlist)
    validation = pd.concat(valist)

    test_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    train_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    validation_tiles = pd.DataFrame(columns=['slide', 'label', 'L0path', 'L1path', 'L2path', 'age', 'BMI'])
    # generate paired tile sets
    for idx, row in test.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'],  row['path'], row['label'],
                                      row['age'], row['BMI'], resolution=resolution)
        test_tiles = pd.concat([test_tiles, tile_ids])
    for idx, row in train.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['path'], row['label'],
                                      row['age'], row['BMI'], resolution=resolution)
        train_tiles = pd.concat([train_tiles, tile_ids])
    for idx, row in validation.iterrows():
        tile_ids = paired_tile_ids_in(row['slide'], row['path'], row['label'],
                                      row['age'], row['BMI'], resolution=resolution)
        validation_tiles = pd.concat([validation_tiles, tile_ids])

    # No shuffle on test set
    train_tiles = sku.shuffle(train_tiles)
    validation_tiles = sku.shuffle(validation_tiles)
    # restrict total number of tile sets in each sets
    if train_tiles.shape[0] > int(batchsize * 80000 / 3):
        train_tiles = train_tiles.sample(int(batchsize * 80000 / 3), replace=False)
        print('Truncate training set!')
    if validation_tiles.shape[0] > int(batchsize * 80000 / 30):
        validation_tiles = validation_tiles.sample(int(batchsize * 80000 / 30), replace=False)
        print('Truncate validation set!')
    if test_tiles.shape[0] > int(batchsize * 80000 / 3):
        test_tiles = test_tiles.sample(int(batchsize * 80000 / 3), replace=False)
        print('Truncate test set!')

    test_tiles.to_csv(path + '/te_sample.csv', header=True, index=False)
    train_tiles.to_csv(path + '/tr_sample.csv', header=True, index=False)
    validation_tiles.to_csv(path + '/va_sample.csv', header=True, index=False)

    return train_tiles, test_tiles, validation_tiles

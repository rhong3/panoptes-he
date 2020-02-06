"""
Calculation of metrics including accuracy, AUROC, and PRC, outputing CAM of tiles, and output
last layer activation for tSNE 2.0

Created on 01/21/2020

@author: RH
"""
import numpy as np
import sklearn.metrics
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import matplotlib
matplotlib.use('Agg')


# Plot ROC and PRC plots
def ROC_PRC(outtl, pdx, path, name, fdict, dm, accur, pmd):
    if pmd == 'subtype':
        rdd = 4
    else:
        rdd = 2
    if rdd > 2:
        # Compute ROC and PRC curve and ROC and PRC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        # PRC
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        microy = []
        microscore = []
        for i in range(rdd):
            fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(
                np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')), np.asarray(pdx[:, i]).ravel())
            try:
                roc_auc[i] = sklearn.metrics.roc_auc_score(
                    np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')), np.asarray(pdx[:, i]).ravel())
            except ValueError:
                roc_auc[i] = np.nan

            microy.extend(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')))
            microscore.extend(np.asarray(pdx[:, i]).ravel())

            precision[i], recall[i], _ = \
                sklearn.metrics.precision_recall_curve(np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')),
                                                   np.asarray(pdx[:, i]).ravel())
            try:
                average_precision[i] = \
                    sklearn.metrics.average_precision_score(
                        np.asarray((outtl.iloc[:, 0].values == int(i)).astype('uint8')), np.asarray(pdx[:, i]).ravel())
            except ValueError:
                average_precision[i] = np.nan

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(np.asarray(microy).ravel(),
                                                              np.asarray(microscore).ravel())
        roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = sklearn.metrics.precision_recall_curve(np.asarray(microy).ravel(),
                                                                                    np.asarray(microscore).ravel())
        average_precision["micro"] = sklearn.metrics.average_precision_score(np.asarray(microy).ravel(),
                                                                         np.asarray(microscore).ravel(),
                                                                         average="micro")

        # Compute macro-average ROC curve and ROC area

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(rdd)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(rdd):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= rdd

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = sklearn.metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.5f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.5f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
        for i, color in zip(range(rdd), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of {0} (area = {1:0.5f})'.format(fdict[i], roc_auc[i]))
            print('{0} AUC of {1} = {2:0.5f}'.format(dm, fdict[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(name))
        plt.legend(loc="lower right")
        plt.savefig("{}/{}_{}_ROC.png".format(path, name, dm))

        print('Average precision score, micro-averaged over all classes: {0:0.5f}'.format(average_precision["micro"]))
        # Plot all PRC curves
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red'])
        plt.figure(figsize=(7, 9))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        lines.append(l)
        labels.append('iso-f1 curves')

        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('micro-average Precision-recall (area = {0:0.5f})'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(rdd), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('Precision-recall for {0} (area = {1:0.5f})'.format(fdict[i], average_precision[i]))
            print('{0} Average Precision of {1} = {2:0.5f}'.format(dm, fdict[i], average_precision[i]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('{} Precision-Recall curve: Average Accu={}'.format(name, accur))
        plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=12))
        plt.savefig("{}/{}_{}_PRC.png".format(path, name, dm))

    else:
        tl = outtl.values[:, 0].ravel()
        y_score = np.asarray(pdx[:, 1]).ravel()
        auc = sklearn.metrics.roc_auc_score(tl, y_score)
        auc = round(auc, 5)
        print('{0} AUC = {1:0.5f}'.format(dm, auc))
        fpr, tpr, _ = sklearn.metrics.roc_curve(tl, y_score)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.5f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{} ROC of {}'.format(name, pmd))
        plt.legend(loc="lower right")
        plt.savefig("{}/{}_{}_ROC.png".format(path, name, dm))

        average_precision = sklearn.metrics.average_precision_score(tl, y_score)
        print('Average precision-recall score: {0:0.5f}'.format(average_precision))
        plt.figure()
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
        precision, recall, _ = sklearn.metrics.precision_recall_curve(tl, y_score)
        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('{} {} PRC: AP={:0.5f}; Accu={}'.format(pmd, name, average_precision, accur))
        plt.savefig("{}/{}_{}_PRC.png".format(path, name, dm))


# patient (slide) level; need prediction scores, true labels, output path, and name of the files for metrics;
# accuracy, AUROC; AUPRC.
def slide_metrics(inter_pd, path, name, fordict, pmd):
    inter_pd = inter_pd.drop(['L0path', 'L1path', 'L2path', 'label', 'Prediction'], axis=1)
    inter_pd = inter_pd.groupby(['slide']).mean()
    inter_pd = inter_pd.round({'True_label': 0})
    if pmd == 'subtype':
        inter_pd['Prediction'] = inter_pd[
            ['POLE_score', 'MSI_score', 'CNV-L_score', 'CNV-H_score']].idxmax(axis=1)
        redict = {'MSI_score': int(1), 'CNV-L_score': int(2), 'CNV-H_score': int(3), 'POLE_score': int(0)}
    elif pmd == 'histology':
        inter_pd['Prediction'] = inter_pd[
            ['Endometrioid_score', 'Serous_score']].idxmax(axis=1)
        redict = {'Endometrioid_score': int(0), 'Serous_score': int(1)}
    else:
        inter_pd['Prediction'] = inter_pd[['NEG_score', 'POS_score']].idxmax(axis=1)
        redict = {'NEG_score': int(0), 'POS_score': int(1)}
    inter_pd['Prediction'] = inter_pd['Prediction'].replace(redict)

    # accuracy calculations
    tott = inter_pd.shape[0]
    accout = inter_pd.loc[inter_pd['Prediction'] == inter_pd['True_label']]
    accu = accout.shape[0]
    accurr = round(accu/tott, 5)
    print('Slide Total Accuracy: '+str(accurr))
    if pmd == 'subtype':
        for i in range(4):
            accua = accout[accout.True_label == i].shape[0]
            tota = inter_pd[inter_pd.True_label == i].shape[0]
            try:
                accuar = round(accua / tota, 5)
                print('Slide {} Accuracy: '.format(fordict[i])+str(accuar))
            except ZeroDivisionError:
                print("No data for {}.".format(fordict[i]))
    try:
        outtl_slide = inter_pd['True_label'].to_frame(name='True_lable')
        if pmd == 'subtype':
            pdx_slide = inter_pd[['POLE_score', 'MSI_score', 'CNV-L_score', 'CNV-H_score']].values
        elif pmd == 'histology':
            pdx_slide = inter_pd[['Endometrioid_score', 'Serous_score']].values
        else:
            pdx_slide = inter_pd[['NEG_score', 'POS_score']].values
        ROC_PRC(outtl_slide, pdx_slide, path, name, fordict, 'slide', accurr, pmd)
    except ValueError:
        print('Not able to generate plots based on this set!')
    inter_pd['Prediction'] = inter_pd['Prediction'].replace(fordict)
    inter_pd['True_label'] = inter_pd['True_label'].replace(fordict)
    inter_pd.to_csv("{}/{}_slide.csv".format(path, name), index=True)


# for real image prediction, just output the prediction scores as csv
def realout(pdx, path, name, pmd):
    if pmd == 'subtype':
        lbdict = {1: 'MSI', 2: 'CNV-L', 3: 'CNV-H', 0: 'POLE'}
    elif pmd == 'histology':
        lbdict = {0: 'Endometrioid', 1: 'Serous'}
    else:
        lbdict = {0: 'negative', 1: pmd}
    pdx = np.asmatrix(pdx)
    prl = pdx.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns=['Prediction'])
    prl = prl.replace(lbdict)
    if pmd == 'subtype':
        out = pd.DataFrame(pdx, columns=['POLE_score', 'MSI_score', 'CNV-L_score', 'CNV-H_score'])
    elif pmd == 'histology':
        out = pd.DataFrame(pdx, columns=['Endometrioid_score', 'Serous_score'])
    else:
        out = pd.DataFrame(pdx, columns=['NEG_score', 'POS_score'])
    out.reset_index(drop=True, inplace=True)
    prl.reset_index(drop=True, inplace=True)
    out = pd.concat([out, prl], axis=1)
    out.insert(loc=0, column='Num', value=out.index)
    out.to_csv("{}/{}.csv".format(path, name), index=False)


# tile level; need prediction scores, true labels, output path, and name of the files for metrics; accuracy, AUROC; PRC.
def metrics(pdx, tl, path, name, pmd, ori_test=None):
    # format clean up
    tl = np.asmatrix(tl)
    tl = tl.argmax(axis=1).astype('uint8')
    pdxt = np.asmatrix(pdx)
    prl = pdxt.argmax(axis=1).astype('uint8')
    prl = pd.DataFrame(prl, columns=['Prediction'])
    if pmd == 'subtype':
        lbdict = {1: 'MSI', 2: 'CNV-L', 3: 'CNV-H', 0: 'POLE'}
        outt = pd.DataFrame(pdxt, columns=['POLE_score', 'MSI_score', 'CNV-L_score', 'CNV-H_score'])
    elif pmd == 'histology':
        lbdict = {0: 'Endometrioid', 1: 'Serous'}
        outt = pd.DataFrame(pdxt, columns=['Endometrioid_score', 'Serous_score'])
    else:
        lbdict = {0: 'negative', 1: pmd}
        outt = pd.DataFrame(pdxt, columns=['NEG_score', 'POS_score'])
    outtlt = pd.DataFrame(tl, columns=['True_label'])
    if name == 'Validation' or name == 'Training':
        outtlt = outtlt.round(0)
    outt.reset_index(drop=True, inplace=True)
    prl.reset_index(drop=True, inplace=True)
    outtlt.reset_index(drop=True, inplace=True)
    out = pd.concat([outt, prl, outtlt], axis=1)
    if ori_test is not None:
        ori_test.reset_index(drop=True, inplace=True)
        out.reset_index(drop=True, inplace=True)
        out = pd.concat([ori_test, out], axis=1)
        slide_metrics(out, path, name, lbdict, pmd)

    stprl = prl.replace(lbdict)
    stouttl = outtlt.replace(lbdict)
    outt.reset_index(drop=True, inplace=True)
    stprl.reset_index(drop=True, inplace=True)
    stouttl.reset_index(drop=True, inplace=True)
    stout = pd.concat([outt, stprl, stouttl], axis=1)
    if ori_test is not None:
        ori_test.reset_index(drop=True, inplace=True)
        stout.reset_index(drop=True, inplace=True)
        stout = pd.concat([ori_test, stout], axis=1)
    stout.to_csv("{}/{}_tile.csv".format(path, name), index=False)

    # accuracy calculations
    tott = out.shape[0]
    accout = out.loc[out['Prediction'] == out['True_label']]
    accu = accout.shape[0]
    accurw = round(accu/tott, 5)
    print('Tile Total Accuracy: '+str(accurw))
    if pmd == 'subtype':
        for i in range(4):
            accua = accout[accout.True_label == i].shape[0]
            tota = out[out.True_label == i].shape[0]
            try:
                accuar = round(accua / tota, 5)
                print('Tile {} Accuracy: '.format(lbdict[i])+str(accuar))
            except ZeroDivisionError:
                print("No data for {}.".format(lbdict[i]))
    try:
        ROC_PRC(outtlt, pdxt, path, name, lbdict, 'tile', accurw, pmd)
    except ValueError:
        print('Not able to generate plots based on this set!')

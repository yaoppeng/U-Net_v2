from batchgenerators.utilities.file_and_folder_operations import *
from joblib import delayed, Parallel
import os
import argparse
import tqdm
import sys

import numpy as np

from PIL import Image
from tabulate import tabulate

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib.eval_functions import *
# from nnunetv2.evaluation.polyp.eval_functions import *
# from nnunetv2.evaluation.polyp.utils import *


def compute_score(pred_root, gt_root, sample, Thresholds):
    pred, gt = sample

    assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0]

    pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))  # (0, 255)
    gt_mask = np.array(Image.open(os.path.join(gt_root, gt)))  # (False, True)

    if len(pred_mask.shape) != 2:
        pred_mask = pred_mask[:, :, 0]
    if len(gt_mask.shape) != 2:
        gt_mask = gt_mask[:, :, 0]

    assert pred_mask.shape == gt_mask.shape

    if gt_mask.max() == 255:
        gt_mask = gt_mask.astype(np.float64) / 255
    gt_mask = (gt_mask > 0.5).astype(np.float64)

    if pred_mask.max() == 255:
        pred_mask = pred_mask.astype(np.float64) / 255

    s_measure = StructureMeasure(pred_mask, gt_mask)
    # Smeasure[i] = StructureMeasure(pred_mask, gt_mask)

    wf_measure = original_WFb(pred_mask, gt_mask)
    # wFmeasure[i] = original_WFb(pred_mask, gt_mask)

    mae = np.mean(np.abs(gt_mask - pred_mask))
    # MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

    threshold_E = np.zeros(len(Thresholds))
    threshold_F = np.zeros(len(Thresholds))
    threshold_Pr = np.zeros(len(Thresholds))
    threshold_Rec = np.zeros(len(Thresholds))
    threshold_Iou = np.zeros(len(Thresholds))
    threshold_Spe = np.zeros(len(Thresholds))
    threshold_Dic = np.zeros(len(Thresholds))

    for j, threshold in enumerate(Thresholds):
        threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] \
            = Fmeasure_calu(pred_mask, gt_mask, threshold)

        Bi_pred = np.zeros_like(pred_mask)
        Bi_pred[pred_mask >= threshold] = 1
        threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)

    return s_measure, wf_measure, mae, threshold_E, threshold_F, threshold_Pr, threshold_Rec, \
        threshold_Iou, threshold_Spe, threshold_Dic


def evaluate(result_path, pred_root_ori, gt_root_ori, verbose=False, debug_dataset_len=5):
    """

    param result_path: path to save the score csv file
    param pred_root: where the predicted png files are stored
    param gt_root: where the gt are stored
    :return:
    """
    if os.path.isdir(result_path) is False:
        os.makedirs(result_path)

    # method = os.path.split(opt.Eval.pred_root)[-1]  # "UACANet-L"
    Thresholds = [0.5]  # np.linspace(1, 0, 256)
    headers = ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm',
               'maxDic', 'maxIoU', 'meanSen', 'maxSen', 'meanSpe', 'maxSpe']

    results = []

    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    print('#' * 20, 'Start Evaluation', '#' * 20)
    if verbose is True:
        datasets = tqdm.tqdm(datasets[:debug_dataset_len], desc='Expr - ', total=len(datasets),
                             position=0, bar_format=
                             '{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}', leave=False)
    else:
        datasets = datasets[:debug_dataset_len]

    for dataset in datasets:
        pred_root = os.path.join(pred_root_ori, dataset)
        gt_root = os.path.join(gt_root_ori, dataset, 'masks')

        preds = subfiles(pred_root, suffix=".png", join=False)
        gts = subfiles(gt_root, suffix=".png", join=False)

        # preds = os.listdir(pred_root)
        # gts = os.listdir(gt_root)

        preds.sort()
        gts.sort()

        threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
        threshold_IoU = np.zeros((len(preds), len(Thresholds)))
        # threshold_Precision = np.zeros((len(preds), len(Thresholds)))
        # threshold_Recall = np.zeros((len(preds), len(Thresholds)))
        threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
        threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
        threshold_Dice = np.zeros((len(preds), len(Thresholds)))

        Smeasure = np.zeros(len(preds))
        wFmeasure = np.zeros(len(preds))
        MAE = np.zeros(len(preds))

        if verbose is True:
            samples = tqdm.auto.tqdm(enumerate(zip(preds, gts)), desc=dataset + ' - Evaluation',
                                     total=len(
                                         preds), ncols=90, position=0, leave=True,
                                     bar_format='{desc:<20}{percentage:3.0f}%|{bar:10}{r_bar}')
        else:
            samples = enumerate(zip(preds, gts))
        # loky, threading
        ans = Parallel(-1, backend="loky")(
            delayed(compute_score)(pred_root, gt_root, sample, Thresholds) for i, sample in samples)
        for i, an in enumerate(ans):
            s_measure, wf_measure, mae, threshold_E, threshold_F, threshold_Pr, threshold_Rec, \
                threshold_Iou, threshold_Spe, threshold_Dic = an

            Smeasure[i] = s_measure
            wFmeasure[i] = wf_measure
            MAE[i] = mae

            threshold_Emeasure[i, :] = threshold_E
            threshold_Fmeasure[i, :] = threshold_F
            threshold_Sensitivity[i, :] = threshold_Rec
            threshold_Specificity[i, :] = threshold_Spe
            threshold_Dice[i, :] = threshold_Dic
            threshold_IoU[i, :] = threshold_Iou

        result = []

        mae = np.mean(MAE)
        Sm = np.mean(Smeasure)
        wFm = np.mean(wFmeasure)

        column_E = np.mean(threshold_Emeasure, axis=0)
        meanEm = np.mean(column_E)
        maxEm = np.max(column_E)

        column_Sen = np.mean(threshold_Sensitivity, axis=0)
        meanSen = np.mean(column_Sen)
        maxSen = np.max(column_Sen)

        column_Spe = np.mean(threshold_Specificity, axis=0)
        meanSpe = np.mean(column_Spe)
        maxSpe = np.max(column_Spe)

        column_Dic = np.mean(threshold_Dice, axis=0)
        meanDic = np.mean(column_Dic)
        maxDic = np.max(column_Dic)

        column_IoU = np.mean(threshold_IoU, axis=0)
        meanIoU = np.mean(column_IoU)
        maxIoU = np.max(column_IoU)

        # result.extend([meanDic, meanIoU, wFm, Sm, meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe])
        # results_1.append([dataset, *result])

        out = []
        for metric in headers:
            out.append(eval(metric))  # get the variable value according to their corresponding name

        result.extend(out)
        results.append([dataset, *result])

        csv = os.path.join(result_path, 'result_' + dataset + '.csv')
        if os.path.isfile(csv) is True:
            csv = open(csv, 'a')
        else:
            csv = open(csv, 'w')
            csv.write(', '.join([' ', *headers]) + '\n')

        out_str = ' ,'
        for metric in result:
            out_str += '{:.4f}'.format(metric) + ','
        out_str += '\n'

        csv.write(out_str)
        csv.close()
    tab = tabulate(results, headers=['dataset', *headers], floatfmt=".3f")

    print(tab)
    print("#" * 20, "End Evaluation", "#" * 20)

    return tab, results


if __name__ == "__main__":
    result_path = "/afs/crc.nd.edu/user/y/ypeng4/data/trained_models/Dataset123_Polyp/PolypTrainer__nnUNetPlans__2d/unknown_FusedMBConv_8/fold_0/validation"
    pred_root = "/afs/crc.nd.edu/user/y/ypeng4/data/trained_models/Dataset123_Polyp/PolypTrainer__nnUNetPlans__2d/unknown_FusedMBConv_8/fold_0/validation"
    gt_root = "/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/polyp/TestDataset"
    evaluate(result_path, pred_root, gt_root, verbose=True)

    # args = parse_args()
    # opt = load_config(args.config)
    # evaluate(opt, args)


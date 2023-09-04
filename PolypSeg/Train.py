import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.pvt import PolypPVT
from tabulate import tabulate
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import wandb

import matplotlib.pyplot as plt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def calcuate_score(gt, pred):
    mae = np.mean(np.abs(gt-pred))

    intersection_arr = (pred == 1) & (gt == 1)
    intersection = np.sum(intersection_arr == 1)
    num_gt = np.sum(gt)
    num_pred = np.sum(pred)

    smooth = 1e-5
    iou = (intersection + smooth) / (num_gt+num_pred-intersection + smooth)
    dice = (2 * intersection + smooth) / (num_gt + num_pred + smooth)

    return dice, iou, mae


def test_1(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        # eval Dice
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice

    return DSC / num1


def test(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    # DSC = 0.0
    maes = []
    dscs = []
    ious = []
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1  = model(image)
        # eval Dice
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        # print(np.unique(input_flat))
        input_flat = input_flat >= 0.5

        dice, iou, mae = calcuate_score(target_flat, input_flat)
        maes.append(mae)
        dscs.append(dice)
        ious.append(iou)
        # mae.append(np.mean(np.abs(target_flat - input_flat)))
        # # print(np.unique(input_flat))
        # intersection = (input_flat * target_flat)
        # dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        # dice = '{:.4f}'.format(dice)
        # dice = float(dice)
        # DSC = DSC + dice

    # return DSC / num1
    return np.mean(dscs), np.mean(ious), np.mean(maes)


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25] 
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2= model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2 
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model 
    save_path = opt.train_save

    latest_path = os.path.join(save_path, "latest")
    best_path = os.path.join(save_path, "best")
    if not os.path.exists(latest_path):
        os.makedirs(latest_path)
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    torch.save(model.state_dict(), os.path.join(latest_path,
                                                f"epoch_{epoch}.pth"))
    # choose the best model

    global dict_plot
   
    # test1path = './dataset/TestDataset/'
    test1path = '/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/polyp/TestDataset'

    dices = []
    ious = []
    maes = []
    results = []
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
            dsc, iou, mae = test(model, test1path, dataset)

            wandb.log({f"dsc/{dataset}": dsc}, step=epoch)
            wandb.log({f"iou/{dataset}": iou}, step=epoch)
            wandb.log({f"mae/{dataset}": mae}, step=epoch)

            dices.append(dsc)
            ious.append(iou)
            maes.append(mae)

            results.append([dataset, dsc, iou, mae])

        mean_dice = np.mean(dices)

        results.append(['mean', np.mean(dices), np.mean(ious), np.mean(maes)])
        wandb.log({f"mean_dsc": mean_dice}, step=epoch)

        tab = tabulate(results, headers=['dataset', 'dsc', 'iou', 'mae'], floatfmt=".3f")

        print(tab)
        if mean_dice > best:
            best = mean_dice
            # torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
            torch.save(model.state_dict(), os.path.join(best_path, f"best_epoch_{epoch}.pth"))
            print(f'got best dice {best} at epoch {epoch}'.center(70, '='))


def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
    
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    # model_name = 'PolypPVT'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/polyp/TrainDataset',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='/afs/crc.nd.edu/user/y/ypeng4/data/raw_data/polyp/TestDataset/Kvasir',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/model_pth/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # model = PolypPVT().cuda()
    # torch.cuda.set_device(0)  # set your gpu device
    from lib.my_pvt import GCNFuse
    # model = GCNFuse().cuda()

    from lib.pvt_3 import PVTNetwork
    model = PVTNetwork().cuda()

    # from lib.pvt_4 import PVTNetwork_1
    # model = PVTNetwork_1().cuda()

    # from lib.pvt_5 import PVTNetwork_2
    # model = PVTNetwork_2().cuda()

    # model = PolypPVT().cuda()

    # from lib.res2unetv2 import Res2Network
    # model = Res2Network(n_classes=1, channel=64).cuda()

    model_name = model.__class__.__name__
    print(model.__class__.__name__.center(50, "="))

    opt.train_save = os.path.join(opt.train_save, model_name)

    wandb.login(key="66b58ac7004a123a43487d7a6cf34ebb4571a7ea")
    wandb.init(project="Polyp_ori_latest",
               dir="/afs/crc.nd.edu/user/y/ypeng4/Polyp-PVT_2/wandb",
               name=model.__class__.__name__,
               resume="allow",  # must resume, otherwise crash
               # id=id,
               config={"class_name": str(model.__class__.__name__)})

    best = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)

# -*- coding: utf-8 -*-
import os
import shutil
import time
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import torchvision.utils as vutil
from torch.utils.data import DataLoader
import torchsummary

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 1
LR = 0.001
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 32
BASE_CHANNEL = 32
INPUT_CHANNEL = 1
INPUT_SIZE = 28
MODEL_FOLDER = './save_model'
IMAGE_FOLDER = './save_image'
INSTANCE_FOLDER = None


class Model(nn.Module):
    def __init__(self, input_ch, num_classes, base_ch):
        super(Model, self).__init__()

        self.num_classes = num_classes
        self.base_ch = base_ch
        self.feature_length = base_ch * 4

        self.net = nn.Sequential(
            nn.Conv2d(input_ch, base_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(base_ch * 2, self.feature_length, kernel_size=3,
                      padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.fc = nn.Linear(in_features=self.feature_length,
                            out_features=num_classes)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, self.feature_length)
        output = self.fc(output)
        return output


def load_dataset():
    train_dataset = datasets.MNIST(root='./data',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_dataset = datasets.MNIST(root='./data',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    return train_dataset, test_dataset


def hook_func(module, input, output):
    """
    Hook function of register_forward_hook

    Parameters:
    -----------
    module: module of neural network
    input: input of module
    output: output of module
    """
    image_name = get_image_name_for_hook(module)
    data = output.clone().detach()
    data = data.permute(1, 0, 2, 3)
    vutil.save_image(data, image_name, pad_value=0.5)


def get_image_name_for_hook(module):
    """
    Generate image filename for hook function

    Parameters:
    -----------
    module: module of neural network
    """
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    base_name = str(module).split('(')[0]
    index = 0
    image_name = '.'  # '.' is surely exist, to make first loop condition True
    while os.path.exists(image_name):
        index += 1
        image_name = os.path.join(
            INSTANCE_FOLDER, '%s_%d.png' % (base_name, index))
    return image_name


if __name__ == '__main__':
    time_beg = time.time()

    train_dataset, test_dataset = load_dataset()
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=TRAIN_BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=TEST_BATCH_SIZE,
                             shuffle=False)

    model = Model(input_ch=1, num_classes=10, base_ch=BASE_CHANNEL).cuda()
    torchsummary.summary(
        model, input_size=(INPUT_CHANNEL, INPUT_SIZE, INPUT_SIZE))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loss = []
    for ep in range(EPOCH):
        # ----------------- train -----------------
        model.train()
        time_beg_epoch = time.time()
        loss_recorder = []
        for data, classes in train_loader:
            data, classes = data.cuda(), classes.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, classes)
            loss.backward()
            optimizer.step()

            loss_recorder.append(loss.item())
            time_cost = time.time() - time_beg_epoch
            print('\rEpoch: %d, Loss: %0.4f, Time cost (s): %0.2f' % (
                ep, loss_recorder[-1], time_cost), end='')

        # print train info after one epoch
        train_loss.append(loss_recorder)
        mean_loss_epoch = torch.mean(torch.Tensor(loss_recorder))
        time_cost_epoch = time.time() - time_beg_epoch
        print('\rEpoch: %d, Mean loss: %0.4f, Epoch time cost (s): %0.2f' % (
            ep, mean_loss_epoch.item(), time_cost_epoch), end='')

        # save model
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        model_filename = os.path.join(MODEL_FOLDER, 'epoch_%d.pth' % ep)
        torch.save(model.state_dict(), model_filename)

        # ----------------- test -----------------
        model.eval()
        correct = 0
        total = 0
        for data, classes in test_loader:
            data, classes = data.cuda(), classes.cuda()
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += classes.size(0)
            correct += (predicted == classes).sum().item()
        print(', Test accuracy: %0.4f' % (correct / total))

    print('Total time cost: ', time.time() - time_beg)

    # ----------------- visualization -----------------
    # clear output folder
    if os.path.exists(IMAGE_FOLDER):
        shutil.rmtree(IMAGE_FOLDER)

    model.eval()
    modules_for_plot = (torch.nn.ReLU, torch.nn.Conv2d,
                        torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)
    for name, module in model.named_modules():
        if isinstance(module, modules_for_plot):
            module.register_forward_hook(hook_func)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False)
    index = 1
    for data, classes in test_loader:
        INSTANCE_FOLDER = os.path.join(
            IMAGE_FOLDER, '%d-%d' % (index, classes.item()))
        data, classes = data.cuda(), classes.cuda()
        outputs = model(data)

        index += 1
        if index > 20:
            break

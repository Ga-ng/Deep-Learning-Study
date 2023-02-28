## 라이브러리 추가하기
import argparse

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from model import UNet
from dataset import *
from util import *

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## 여러가지 parmeter 들을 터미널이나 동적으로 넘어겨줄 수 있도록 parser를 사용해보자

# parser object 생성
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# parser에 argument들을 추가해보자.
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./results", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

args = parser.parse_args()

## training 에 필요한 하이퍼파라미터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

# 데이터가 저장되어 있는
data_dir = args.data_dir

# train 될 네트워크가 저장될
ckpt_dir = args.ckpt_dir

# 텐서보드 로그파일이 저장될
log_dir = args.log_dir

result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# train이 GPU, CPU 어디서 동작할지? 결정하는 device flag
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## 이번에는 실제로 트레이닝을 진행하고 vaildation 을 할 수 있는 Frame work을 구현해 보자.

## 네트워크 학습하기

if mode == 'train':
    # training data set을 불러올 때 적용할 다양한 Transform 를 나열하자.
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

    # 데이터가 저장된 폴더로부터 필요한 데이터를 불러오는 DataLoader를 구현해보자.
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    # validation 에 필요한 데이터셋을 불러오자.
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그 밖에 부수적인 변수들 선언하기
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    # training data set을 불러올 때 적용할 다양한 Transform 를 나열하자.
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    # 데이터가 저장된 폴더로부터 필요한 데이터를 불러오는 DataLoader를 구현해보자.
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 그 밖에 부수적인 변수들 선언하기
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성
# 네트워크를 불러오고 네트워크가 학습되는 도메인이 CPU인지 GPU인지를 명시해 주기 위해 to(device) 명시
net = UNet().to(device)

# LOSS function 정의
fn_loss = nn.BCEWithLogitsLoss().to(device)

# optimizier 설정 - Adam 사용
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 아웃풋을 저장하기 위한 필요한 몇가지 함수들을 선언
# tensor -> numpy
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

# normalization 된 데이터를 반대로 denorm
fn_denorm = lambda x, mean, std: (x * std) + mean

# 네트워크의 아웃풋 이미지를 binary class로 분류해주는 함수 정의
fn_class = lambda x: 1.0 * (x > 0.5)

# tensorboard 사용에 필요한 SummaryWrite 선언
# writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
# writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))



## 실제로 training 이 수행되는 for 구현
st_epoch = 0


if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train() # network 에게 train mode 임을 알려주는 메서드를 활성화
        loss_arr = []

        for batch, data in enumerate(loader_train, 1):
            # nerowrk에 입풋을 받아 output을 출력하는 forward path
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # backpropagation 하는 backward path 구현
            optim.zero_grad()

            loss = fn_loss(output, label)
            loss.backward()

            optim.step()

            # loss function 계산
            loss_arr += [loss.item()]

            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # tensorborad 에 input, output, label을 저장
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

        #     writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        #     writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        #     writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
        #
        # ## loss 를 텐서보드에 저장
        # writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # 이까지가 network 를 train하는 부분


        ## 아래는 network를 vaildation 하는 부분
        # vaildation 에는 backpropagation 하는 부분이 없기 때문에 torch.no_grad()해서 사전에 막기

        with torch.no_grad():
            # network에게 현재 vaildation 모드임을 명시
            net.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val, 1):
                # training 과 마찬가지로 forward path 작성
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # vailation 은 backpropagation 안하니까 backward path 는 구현하지 않음

                # loss function 계산하기
                loss = fn_loss(input, label)

                loss_arr += [loss.item()]

                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # tensorboard에 label, input, output을 저장하는 부분을 작성
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                # writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                # writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

            # loss fun 저장하는 부분 작성
            # writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

            # 10번, 50 번 마다 저장하고 싶으면
            # if epoch // 5 == 0: # 해 주면 됨

            # network 한번씩 돌 때마다 저장
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)



        ## 학습이 완료되면 두개의 wrtier를 close 해주기
        # writer_train.close()
        # writer_val.close()
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    ## 아래는 network를 vaildation 하는 부분
    # vaildation 에는 backpropagation 하는 부분이 없기 때문에 torch.no_grad()해서 사전에 막기

    with torch.no_grad():
        # network에게 현재 vaildation 모드임을 명시
        net.eval()
        loss_arr = []

        for batch, data in enumerate(loader_test, 1):
            # training 과 마찬가지로 forward path 작성
            label = data['label'].to(device)
            input = data['input'].to(device)

            output = net(input)

            # vailation 은 backpropagation 안하니까 backward path 는 구현하지 않음

            # loss function 계산하기
            loss = fn_loss(output, label)

            loss_arr += [loss.item()]

            print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                  (batch, num_batch_test, np.mean(loss_arr)))

            # tensorboard에 label, input, output을 저장하는 부분을 작성
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            for j in range(label.shape[0]):
                id = num_batch_test * (batch - 1) + j

                # png 파일로 저장하는 방법
                plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
                plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

                # numpy type 으로 저장
                np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
                np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

    ## 최종 test set에 최종 loss function 값을 출
    print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_test, np.mean(loss_arr)))
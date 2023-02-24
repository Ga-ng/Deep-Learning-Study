## 라이브러리 추가하기
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

## training 에 필요한 하이퍼파라미터 설정하기
lr = 1e-3
batch_size = 4
num_epoch = 100

# 데이터가 저장되어 있는
data_dir = './datasets'

# train 될 네트워크가 저장될
chkpt_dir = './checkpoint'

# 텐서보드 로그파일이 저장될
log_dir = './log'

# train이 GPU, CPU 어디서 동작할지? 결정하는 device flag
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## UNet Network 구현
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 반복적으로 나오는 레이어를 함수로 설정
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []

            # 우선 Convolution layer에 대한 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]

            # 다음은 batch normalization layer에 대한 정의
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            # RELU 정의
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr

        ## Contracting path
        # Naming -> encoder, stage: 1, CBR: 몇번째인지
        # 첫번째 stage, 각 함수는 파란색 화살표
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

        # 빨간색 화살표
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # stage: 2
        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # 맨 아래 마지막 레이어 추가로 encoder 파트 종료
        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        ## Expansive path - 오른쪽
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2*512, out_channels=512)
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2*256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2*128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2*64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        ## segmentation 에 필요한 n개의 클래스에 대한 output을 만들기 위해
        ## 녹색화살표와 같이 1x1 conv 레이어 정의

        self.fc = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

        # 이까지 하면 UNet에서 필요한 레이어들을 모두 초기화 한 것

    ## 이번엔 각 레이어들을 연결
    # x는 인풋 이미지
    def forward(self, x):
        # 첫번째 스테이지의 첫번째 레이어를 연결
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        # 이까지 하면 엔코더 스테이지 다 연결

        # 다음은 디코더를 연결
        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        ## 채널 방향으로 연결하는 함수를 concatination, cat 이라고 함
        ## dim: 1 -> 채널방향, 0 -> batch 방향, 2 -> y 방향, 3 -> x 방향
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)

        return x


## 데이터를 로드하고, 로드함에 있어서 여러가지 트랜스폼 함수를 구현해보자.

## 데이터 로더 구현
class Dataset(torch.utils.data.Dataset):
    # 선언을 할 때, 데이터 셋의 경로와 트랜스폼들을 인자로 받음
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 데이터 리스트를 불러오자. 데이터는 어떤 형식인가?
        # input_%%%.npy 형식임.
        lst_data = os.listdir(self.data_dir)

        # label, input이라는 prefix로 된 리스트를 정리함.
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    # 함수의 길이를 확인하는 메서드
    def __len__(self):
        return len(self.lst_label)

    # 실제로 데이터를 가져오는 메서드
    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 이미지가 0,255 사이 이므로 0,1 사이로 Normalized
        label = label/255.0
        input = input/255.0

        # NN 에 들어가는 기본적으로 3차원이어야함.
        # 따라서 newaxis 를 이용해서 마지막 axis를 임의로 생성해주기
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        # 라벨과 인풋을 dict 형태로
        data = {'input': input, 'label': label}

        # 트랜스폼 함수를 넘어주면 ,, 정의가 되어 있으면 이 트랜스폼 함수를 통과한 데이터셋을 받아오기
        if self.transform:
            data = self.transform(data)

        return data



## 잘 구현 되었는지 확인
# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))
# data = dataset_train.__getitem__(0)
#
# input = data['input']
# label = data['label']
#
# ## 출력
# plt.subplot(121)
# plt.imshow(input.squeeze())
#
# plt.subplot(122)
# plt.imshow(label.squeeze())
#
# plt.show()

## 트랜스폼 형태로 데이터 전처리를 하자.
#
#
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # numpty 는 (y, x, ch), pytorch 는 (ch, y, x) 로 순서가 다름
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = label.transpose((2, 0, 1)).astype(np.float32)

        # from_numpy를 적용하기
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

## 다음으로 많이 쓰이는 Normalization Transform
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std
        # 라벨은 0 또는 1 인 클래스로 정의되어 있기에 input에만 해주면 됨

        data = {'label': label, 'input': input}

        return data

## 랜덤하게 좌우 상하 filp을 하는 트랜스폼
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

## 이런식으로 필요한걸 트랜스폼 함수로 구현해서 사용하자.

## 잘 구현했는지 확인

# 여러개의 트랜스폼을 묶어서 사용할 수 있는 compose 함수
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

# 아래처럼 Dataset에 트랜스폼으로 인자로 넘겨주면 성공
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
data = dataset_train.__getitem__(0)

input = data['input']
label = data['label']

## 출력
plt.subplot(121)
plt.imshow(input.squeeze())

plt.subplot(122)
plt.imshow(label.squeeze())

plt.show()
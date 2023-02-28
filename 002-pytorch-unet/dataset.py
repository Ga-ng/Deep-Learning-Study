## DataLoader file
import os
import numpy as np
import torch
import torch.nn as nn


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
        input = input.transpose((2, 0, 1)).astype(np.float32)

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

# ## 잘 구현했는지 확인
#
# # 여러개의 트랜스폼을 묶어서 사용할 수 있는 compose 함수
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
#
# # 아래처럼 Dataset에 트랜스폼으로 인자로 넘겨주면 성공
# dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
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

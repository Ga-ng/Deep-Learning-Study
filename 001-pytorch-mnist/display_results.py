import os
import numpy as np
import matplotlib.pyplot as plt

##
result_dir = './results/numpy'

## 저장이 된 데이터 리스트 추출
lst_data = os.listdir(result_dir)

# input, label, output list 로 분리
lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

## 다음은 데이터를 불로오는 부분
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))


## matplot 으로 출력
plt.subplot(131)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(132)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')
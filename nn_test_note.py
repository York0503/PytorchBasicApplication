import torch

# 創建Tensor
## 傳入陣列(向量) 一維陣列
data = torch.tensor([1,2,3,4,5])
print(data)

## 傳入陣列(向量) 二維陣列
data = torch.tensor([[1,2,3], [4,5,6]])
print(data)

## 創建隨機矩陣(EX:2行5列)
### 一個 0 ~ 1 的隨機數字
data = torch.rand(2,5)
print(data)

## 創建全是1的矩陣(EX:2行5列)
data = torch.ones(2,5)
print(data)

## 創建全是0的矩陣(EX:2行5列)
data = torch.zeros(2,5)
print(data)

########################################################################

# 很多時候我們不知道資料的狀態
# 可以此函數去了解資料，並取得所需資料
## 取得資料矩陣的形狀(幾行幾列)
data = torch.zeros(2,5)
print(data.shape)

## 取得資料矩陣的有幾列
print(data.shape[0])
## 取得資料矩陣的有幾行
print(data.shape[1])

########################################################################

# 取的當前Tensor的資料類行或轉換資料類型
## 取得Tenseor內儲存數據的資料類型(ex:float32、int)
print(data.dtype)

## 轉換資料類型
data = data.to(torch.int)
print(data.dtype)

########################################################################

# Tensor基本操作
## 注意:tensor與tensor之間的計算或操作必須形狀相同(行列數相同)
## 取得tensor裡面的5(還是tensor型態)
data = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]])
print(data[0][4])

## 取得tensor裡面真正的數字5
print(data[0][4].item())

########################################################################

# Tensor四則運算
## 加法
### 1 1 1  0 0 0   1 1 1
### 1 1 1  0 0 0　 = 1 1 1
### 1 1 1  0 0 0   1 1 1
data_a = torch.ones([3,3])
data_b = torch.zeros([3,3])
print(data_a + data_b)

### 1 1 1  1 2 3   2 3 4
### 1 1 1  4 5 6　 = 5 6 7
### 1 1 1  7 8 9   8 9 10
data_a = torch.ones([3,3])
data_b = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(data_a + data_b)

## 減法
### 1 1 1  1 2 3   0 -1 -2
### 1 1 1  4 5 6　 = -3 -4 -5
### 1 1 1  7 8 9   -6 -7 -8
data_a = torch.ones([3,3])
data_b = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(data_a - data_b)

## 乘法
### 1 1 1  1 2 3   1 2 3
### 1 1 1  4 5 6　 = 4 5 6
### 1 1 1  7 8 9   7 8 9
data_a = torch.ones([3,3])
data_b = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(data_a * data_b)

### 所有欄位對應相乘
### 1 2 3       3 6 9
### 4 5 6　* 3  = 12 15 18
### 7 8 9       21 24 27
print(data_b * 3)

## 除法
### 1 1 1  1 2 3   1 0.5 0.3333
### 1 1 1  4 5 6　 = 0.25 0.2 0.1667
### 1 1 1  7 8 9   0.1429 0.125 0.1111
data_a = torch.ones([3,3])
data_b = torch.tensor([[1,2,3], [4,5,6], [7,8,9]])
print(data_a / data_b)

### 所有欄位對應相除
### 1 2 3       0.3333 0.6667 1
### 4 5 6　/ 3  = 1.3333 1.6667 2
### 7 8 9       2.3333 2.6667 3
print(data_b / 3)

########################################################################

## tensor總和
### 取得tensor內所有值總和
print(torch.sum(data_b))

### 計算每一列的總和
#### 加入參數"軸"的概念 // axis
### 1 2 3
### 4 5 6
### 7 8 9
#### ans = [12, 15, 18]
print(torch.sum(data_b, axis = 0))
#### ans = [6, 15, 24]
print(torch.sum(data_b, axis = 1))

########################################################################

## tensor平均值
### 須注意資料型態要float or complex dtype才可使用mean計算
### 1 2 3
### 4 5 6
### 7 8 9
#### ans = [2., 5., 8.]
print(torch.mean(data_b.to(torch.float), axis = 1))

########################################################################

# tensor準確率
predict = torch.tensor([1,1,1,0,0,0,2,2,2])
label = torch.tensor([3,1,1,3,0,0,3,2,2])

## 是否符合條件
## ans = [ True, False, False,  True, False, False,  True, False, False]
print(label > predict)

## ans = [False,  True,  True, False,  True,  True, False,  True,  True]
print(label == predict)

## 計算相等的數量有幾個
## ans = 6
print(torch.sum(predict == label).item())

## 計算準確率
### 須注意資料型態要float or complex dtype才可使用mean計算
### ans = 0.6666666865348816
print(torch.mean((predict == label).to(torch.float)).item())

########################################################################

# 取得資料進行sort並取得該資料欄位sort後的索引
## sort默認由小到大排序
data = torch.tensor([1,2,3,11,12,13,-1,-2,-3])

### ans = [8, 7, 6, 0, 1, 2, 3, 4, 5]
print(torch.argsort(data))

## 取得最小資料的索引
### ans = 8
print(torch.argmin(data))

## 取得最大資料的索引
### ans = 5
print(torch.argmax(data))

data = torch.tensor([[1,2,3,11,12,13,-1,-2,-3],[4,5,6,-4,-5,-6,7,8,9]])
### 1, 2, 3, 11, 12, 13, -1, -2, -3
### 4, 5, 6, -4, -5, -6, 7, 8, 9

## 取得每一行最小值的索引
### ans = [0, 0, 0, 1, 1, 1, 0, 0, 0]
print(torch.argmin(data, axis = 0))

## 取得每一行最大值的索引
### ans = [1, 1, 1, 0, 0, 0, 1, 1, 1]
print(torch.argmax(data, axis = 0))

## 取得每一列最小值的索引
### ans = [8, 5]
print(torch.argmin(data, axis = 1))

## 取得每一列最大值的索引
### ans = [5, 8]
print(torch.argmax(data, axis = 1))

########################################################################

# 將資料變形
## 行列互換(已知行列數)
data = data.view(9, 2)
print(data)

## 行列互換(已知行數)
data = data.view(-1, 2)
print(data)

## 行列互換(已知列數)
data = data.view(9, -1)
print(data)

## 變形為一維矩陣
data = data.view(-1)
print(data)

########################################################################

# 神經網絡介紹
## 類似黑盒子，也可以視為一個函數
## 函數會有指定需要的"輸入input"和一個經過函數內部計算後的"輸出output"

## 兩層神經網絡(多層以此類推)
### 有一個輸入為784個像素點形成的灰階圖 -> 進入函數 -> 輸出10個機率(為0,1,2,3,4,5,6,7,8,9的機率)
### [1, 784]*[784, 444] = [1, 444]*[444, 555] = [1, 555]*[555, 512] = [1, 512]*[512, 10] = [1, 10]
#### 輸入層  in_channel 784 out_channel 444
#### 隱藏層1  in_channel 444 out_channel 555
#### 隱藏層2  in_channel 555 out_channel 512
#### 輸出層  in_channel 512 out_channel 10

########################################################################
## 使用torch中的nn模組(torch.nn 是一個構建神經網路的模組（module）)

import torch.nn as nn

## 序列化Sequential
### 開始建立神經網絡
model = nn.Sequential(
    ### Linear 建立要寫入的層
    ### 第一層輸入層、線性層(全連接層)
    ### Linear("in_features", "out_features")

    ### nn.ReLU()
    ### ReLU 全名為 Rectified Linear Unit，中譯為「修正線性單元」
    ### 又稱「激活函數」、「激勵函式」
    ### 用意是增加類神經網路裡頭的非線性特徵

    ### nn.Softmax()
    ### 將正無窮至負無窮的數值轉為機率(0到1的值)

    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 555),
    nn.ReLU(),
    nn.Linear(555, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()

)

print(model)

########################################################################
## random一個隨機的784像素點進行測試
data = torch.rand(1, 784)
model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 555),
    nn.ReLU(),
    nn.Linear(555, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
)
predict = model(data)
print(predict)

########################################################################

import pandas as pd  ## 讀取資料檔

# 讀取csv檔
raw_df = pd.read_csv('train.csv')
print(raw_df)

########################################################################

# 特徵與標籤
## 標籤
### 此csv檔案標籤名稱叫label
label = raw_df['label']
print(label)

### 將原本取得的data frame資料類型
### 轉成numpy的ndarray
label = raw_df['label'].values
print(label)

## 特徵
raw_df = raw_df.drop(['label'], axis = 1)
feature = raw_df.values
print(feature)

## 驗證lable跟feature的數量是否相同
raw_df = pd.read_csv('train.csv')
lable = raw_df['label'].values
raw_df = raw_df.drop(['label'], axis = 1)
feature = raw_df.values

### 驗證
print(len(label), len(feature))

########################################################################
## 將資料分為兩個數據集，"訓練資料" 跟 "測試資料"
### 訓練特徵 train_feature
### 訓練標籤 train_label
### 前80%的資料為訓練資料，且算出來要為整數(int)
train_feature = feature[:int(len(feature)*0.8)]
train_label = feature[:int(len(label)*0.8)]

### 測試特徵 test_feature
### 測試標籤 test_label
### 後20%的資料為測試資料，且算出來要為整數(int)
test_feature = feature[int(len(feature)*0.8):]
test_label = feature[int(len(label)*0.8):]

### 檢查是否有切對
print(len(train_feature), len(train_label), len(test_feature), len(test_label))

########################################################################
# 數據與模型結構完整code
import torch
import torch.nn as nn
import pandas as pd

### 建立神經網絡
model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 555),
    nn.ReLU(),
    nn.Linear(555, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
)

## 讀取csv檔
raw_df = pd.read_csv('train.csv')

## 取得標籤資料
lable = raw_df['label'].values

## 取得特徵資料
raw_df = raw_df.drop(['label'], axis = 1)
feature = raw_df.values

### 資料拆分"訓練"與"測試"
### 訓練特徵 train_feature // 訓練標籤 train_labe
### 前80%為訓練資料
train_feature = feature[:int(len(feature)*0.8)]
train_label = lable[:int(len(label)*0.8)]

### 測試特徵 test_feature // 測試標籤 test_label
### 後20%為測試資料
test_feature = feature[int(len(feature)*0.8):]
test_label = lable[int(len(label)*0.8):]

### 將資料轉tensor
train_feature = torch.tensor(train_feature).to(torch.float)
train_label = torch.tensor(train_label)
test_feature = torch.tensor(test_feature).to(torch.float)
test_label = torch.tensor(test_label)

########################################################################
# 訓練
## 梯度下降
### 損失函數
### 瞎子下山
loss_function = nn.CrossEntropyLoss()

### 優化器
### 優化怎麼下山設定或調整參數
### params = 參數，填入要優化那些參數
### lr(learning rate)(學習率) = 每次迭代時權重的更新步伐
### 大學習率（High lr）：收斂速度快，但可能會跳過最優解，導致模型不穩定
### 小學習率（Low lr）：收斂速度慢，但有可能更精確地接近最優解，然而過低的值可能會陷入局部最小值或收斂過慢。
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0001)


## 訓練次數
for i in range(100):
  # 清空優化器梯度(梯度值歸零//偏導清空)
  optimizer.zero_grad()
  # 預測(前向傳播) // 將預測資料放入model進行預測
  predict = model(train_feature)
  # 損失函數 // 將預測值跟標準答案放進去計算損失函數
  loss = loss_function(predict, train_label)
  # 反向傳播 // 計算損失對所有參數的梯度
  loss.backward()
  # 梯度下降
  optimizer.step()

  # 檢視損失函數在訓練的時候會越來越小
  print(loss.item())

########################################################################
# 查看準確率
for i in range(100):
  optimizer.zero_grad()
  predict = model(train_feature)
  # predict資料
  ## 0.1 0.1 0.8 0.0 0.0 0.0 0.0 0.0 0.0 0.0  // 2
  ## 0.3 0.2 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0  // 3
  ## 0.9 0.1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  // 0
  resule = torch.argmax(predict, axis = 1)
  train_acc = torch.mean((resule == train_label).to(torch.float))
  # 輸出準確率
  print(train_acc.item())

  loss = loss_function(predict, train_label)
  loss.backward()
  optimizer.step()
  print(loss.item())

########################################################################
# 數據與模型結構完整code
import torch
import torch.nn as nn

### 建立神經網絡
model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 555),
    nn.ReLU(),
    nn.Linear(555, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
)

## 讀取csv檔
raw_df = pd.read_csv('train.csv')

## 取得標籤資料
lable = raw_df['label'].values

## 取得特徵資料
raw_df = raw_df.drop(['label'], axis = 1)
feature = raw_df.values

### 資料拆分"訓練"與"測試"
### 前80%為訓練資料
train_feature = feature[:int(len(feature)*0.8)]
train_label = lable[:int(len(label)*0.8)]

### 後20%為測試資料
test_feature = feature[int(len(feature)*0.8):]
test_label = lable[int(len(label)*0.8):]

### 將資料轉tensor
train_feature = torch.tensor(train_feature).to(torch.float)
train_label = torch.tensor(train_label)
test_feature = torch.tensor(test_feature).to(torch.float)
test_label = torch.tensor(test_label)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0001)

## 開始訓練
for i in range(100):
  optimizer.zero_grad()
  predict = model(train_feature)
  loss = loss_function(predict, train_label)
  loss.backward()
  optimizer.step()
  print(loss.item())

  # 保存模型
torch.save(model.state_dict(), './mymodel.pt')

########################################################################
# 引用已訓練的模型
params = torch.load('./mymodel.pt')
# 參數放入模型
model.load_state_dict(params)
new_test_data = test_feature[100:120]
new_test_lable = test_label[100:120]
predict = model(new_test_data)
result = torch.argmax(predict, axis=1)
print(new_test_lable)
print(result)

########################################################################

# 使用GPU加速
# 將資料與模型放到GPU裡進行加速運算
# ///如果不是GPU環境或支援cuda會報錯///
train_feature = torch.tensor(train_feature).to(torch.float).cuda()
train_label = torch.tensor(train_label).cuda()
test_feature = torch.tensor(test_feature).to(torch.float).cuda()
test_label = torch.tensor(test_label).cuda()

model = nn.Sequential(
    nn.Linear(784, 444),
    nn.ReLU(),
    nn.Linear(444, 555),
    nn.ReLU(),
    nn.Linear(555, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
    nn.Softmax()
).cuda()
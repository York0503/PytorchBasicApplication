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

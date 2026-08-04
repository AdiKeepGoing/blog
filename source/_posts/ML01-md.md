---
title: ML01.md
categories:
  - 学习笔记
tags:
  - 机器学习
  - null
  - null
description: 📌开始机器学习方面理论知识的了解
date: 2026-08-04 15:39:56
katex: true
---

# ML01

**ML algorithms**

- Supervised learning(j监督学习)
- Unsupervised learning（无监督学习）

## Supervised learning

**定义**

- input ---------> output，Learns from  data **labeled** with the "right answers".
- 监督学习算法类型：Regression algorithms(回归算法)、classification algorithms(分类算法)
- 其中，回归算法是预测多种可能的输出，分类算法是预测可能的小部分输出

## Unsupervised learning

**定义**

- Find something interesting in **unlabeled** data.
- 无监督学习算法类型：clustering algorithms(聚类算法)、Anomaly detection（异常检测）、Dimensionality reduction(降维)
- 聚类算法：应用于新闻分类、用户分类等。异常检测：用于寻找不同寻常的数据点，应用于金融系统的欺诈检测等。降维：将大数据集压缩为小得多的数据集

## 线性回归模型

**术语**

- 训练集
- （输入变量[输入标签]，输出变量[目标变量]）
- y-hat指的是，y的预测值或者预估值
- f(x)=wx+b，这里的w和b是为常数，也被称为系数或者权重
- 示例

~~~
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. 创建模拟数据
# x: 房子的占地面积 (平方米)
# y: 房子的价格 (万元)
np.random.seed(0)
x = 20 + 80 * np.random.rand(100, 1)  # 占地面积在20到100平方米之间
y = 50 + 2 * x + 30 * np.random.randn(100, 1) # 价格 = 50 + 2*面积 + 随机噪音

# 确保价格不为负数
y = np.maximum(y, 10) 

# 2. 创建线性回归模型
model = LinearRegression()

# 3. 训练模型
model.fit(x, y)

# 4. 获取模型参数
# 截距 (Intercept)
intercept = model.intercept_[0]
# 系数 (Coefficient)
coefficient = model.coef_[0][0]

print(f"线性回归模型参数：")
print(f"截距 (y轴截距): {intercept:.2f}")
print(f"系数 (面积对价格的影响): {coefficient:.2f}")

# 5. 进行预测
# 假设我们想预测一个面积为 70 平方米的房子的价格
area_to_predict = np.array([[70]])
predicted_price = model.predict(area_to_predict)[0][0]

print(f"\n预测一个面积为 {area_to_predict[0][0]} 平方米的房子的价格： {predicted_price:.2f} 万元")

# 6. 可视化结果
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='实际房屋数据')
plt.plot(x, model.predict(x), color='red', label='回归线')
plt.scatter(area_to_predict, predicted_price, color='green', s=100, zorder=5, label='预测点')

plt.title('线性回归模型：房屋面积与价格预测')
plt.xlabel('占地面积 (平方米)')
plt.ylabel('价格 (万元)')
plt.legend()
plt.grid(True)
plt.show()
~~~

结果输出：

~~~
线性回归模型参数：
截距 (y轴截距): 57.14
系数 (面积对价格的影响): 1.98
~~~

![01](E:\blog\my-blog\pictures\01.png)

**代价函数**

- 通过预测y-hat和目标值y的差值，即（y-hat - y），以此衡量预测值与目标值之间的偏差。
  $$
  J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (h_w(x^{(i)}) - y^{(i)})^2
  $$
  平方误差代价函数。目标是，选择w和b，使J（w,b）的值最小。

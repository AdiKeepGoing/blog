---
title: paper02
categories:
  - 文章阅读
tags:
  - 类增量学习
  - Adapter-prototype
description: 📌Expandable Subspace Ensemble for Pre-Trained Model-Based Class-Incremental Learning
mathjax: true
date: 2026-10-31 15:22:44
---

# 背景

~~~
现实世界的数据会不断出现新的类别，但模型在学习新类别时容易遗忘旧类别；
而预训练模型虽然能力很强，现有增量学习方法仍然难以同时兼顾“学习新知识”和“保留旧知识”。

现有技术及其痛点：
1.类增量学习（Class-Incremental Learning, CIL）：学习新类别时，模型容易覆盖之前已经学到的知识，产生灾难性遗忘
2.基于预训练模型的CIL：现有方法通常会冻结预训练模型主体，只增加少量可学习参数，例如 Prompt，这样可以保留预训练模型本身的通用能力。可是，当新任务到来时，继续优化这些 Prompt 仍然可能修改旧任务已经使用过的 Prompt，因此新旧任务之间会发生冲突，依旧可能导致遗忘。
3.可扩展网络（Expandable Networks）：传统增量学习中，有一种思路是：每来一个新任务，就新增一套 backbone，让不同任务使用不同的特征空间。这样，新任务的训练不会破坏旧任务的网络，因此可以减少任务之间的冲突。
但这种方法又有两个明显问题。
第一，代价太大。如果每个任务都保存一整套预训练 backbone，而预训练模型本身参数量很大，随着任务不断增加，存储和计算成本会迅速增长。
第二，传统可扩展网络通常需要保存旧类别样本，也就是 exemplars，用于重新训练或校准统一分类器。
~~~

# 贡献

~~~
由上述三个背景提出了，两个主要矛盾（EASE要解决的两个核心的主要问题）：
1.如何低成本地构造任务特定子空间，而不是每次扩展整个 backbone？
2.在不保存旧样本的情况下，如何让不断扩展的特征空间仍然能够对所有旧类和新类进行统一分类？
~~~

# 方法

图解：

~~~
First Incremental Stage:
D1(第一次任务)：这里class_1是cat,class_2是dog
预训练的Backbone冻结，不更新。
只增加一个轻量级Adapter，训练A1适应当前的任务-->服务于任务1的特征子空间
图片经过A1后，提取出类别原型P_1,1:某个类别在某个特征空间中的代表性特征

Second Incremental Stage:
D2(第二次任务)：这里class_3是bird,class_4是lion
冻结backbone和A1
继续增加一个轻量级Adapter，训练A2适应当前的任务-->服务于任务2的特征子空间，这里的Adapter可以理解为Expandable Subspace（可拓展子空间）

这里class_3和class_4，也经过了A1,得到了P_2,1【任务2类别，在Adapter A_1子空间里的Prototype】
但是EASE不保存旧数据（Exemplar-Free），所以到了任务2：猫、狗的训练图片已经没有了无法直接计算，所以作者想：虽然没有猫的图片了，但我能不能“估计”猫在新空间里的 Prototype？答案就是：Semantic-Guided Prototype Complement/中文可以叫：语义引导的原型补全。

目前已经掌握的信息：class3,class4,A1，A2
需要推理的信息：class1在A2中的位置
在A1中通过class1_prototype推理得到class1与class3,class4的位置关系，通过该位置关系，在A2中反向推理得到class1的位置。
~~~

公式：

~~~
问题定义与传统方法：1、2、3
可拓展子空间构建：4、5、6、7
语义引导的原型补全：8、9
子空间集成与预测：10、11、12
~~~



(1)类增量学习的优化目标：$$f^*=\arg\min_{f\in\mathcal H} \mathbb E_{(x,y)\sim \mathcal D_t^1\cup\cdots\cup\mathcal D_t^b} \mathbb I(y\neq f(x)) \tag{1}$$

* 该公式为理论性问题的定义，表达的是理想的优化目标（CIL的目标）；
* argmin表示从H中选择一个最理想的模型（错误率最低的模型）
* E:对所有的任务合并求期望
* Ⅱ:指示函数，预测错误时取 1，否则取 0

* 该公式表示：在所有已学习任务的数据分布上，寻找一个使分类错误率最小的模型，即不断学习新类别的同时，仍能正确识别旧类别。

---

(2)基于 Prompt 的预训练模型增量学习：$$\min_{\mathrm{Pool}\cup W} \sum_{(x,y)\in\mathcal D^b} \ell\left(W^\top\bar{\phi}(x;\mathrm{Pool}),y\right) +\mathcal L_{\mathrm{Pool}} \tag{2}$$

* 针对Prompt-based CIL的思想的公式：

* $$\bar{\phi}(x;\mathrm{Pool})$$:预训练模型的特征提取器（输入图像，prompt）,这里的prompt是指：一组可以通过训练更新的向量

* $$W^\top\bar{\phi}(x;\mathrm{Pool}) $$:W：分类器权重，判断图像属于哪个类别

* $$交叉熵损失 \ell$$:通过损失函数衡量预测结果与真实标签之间的差距，差距越大损失越大

* $$Prompt 损失 \mathcal L_{\mathrm{Pool}}$$:约束 Prompt 的选择或学习过程

---

(3) 基于可扩展 Backbone 的增量学习：

$$
\min_{\phi_{\mathrm{new}}\cup W_E}
\sum_{(x,y)\in\mathcal D^b\cup\mathcal E}
\ell\left(
W_E^\top
[\bar{\phi}_{\mathrm{old}}(x),\phi_{\mathrm{new}}(x)],
y
\right)
\tag{3}
$$

* 冻结旧 Backbone，再新增一个 Backbone 学习当前任务；将新旧 Backbone 的特征拼接起来，同时使用当前任务数据和保存的旧样本来训练一个更大的统一分类器。

* **特征拼接**：\([\bar{\phi}_{\mathrm{old}}(x),\phi_{\mathrm{new}}(x)]\)，由旧 Backbone 与新 Backbone 的特征组成，旧 Backbone 的参数保持冻结。

* **分类器**：\(W_E\in\mathbb R^{2d\times |\mathcal Y^b|}\)。

* 该方法有两个缺点：1.参数量太大2.依赖旧样本

* 由此引出，EASE：不再增加参数量较大的Backbone，而是增加轻量的Adapter

---

(4)$$Adapter 的特征调整x_o=\sigma(x_iW_{\mathrm{down}})W_{\mathrm{up}}+\mathrm{MLP}(x_i) \tag{4} $$

* **MLP**：ViT 的每个 Transformer Block 里，本来就有一个 MLP 模块。

* 主干网络保持不动，在 MLP 旁边增加一条很小的支路，这条支路就是 Adapter

~~~
                 ┌──────── MLP ───────────┐
输入 x_i ────────┤                        ├── 相加 → x_o
                 └→ Adapter 小支路 ──────┘
作为 MLP 的一个 side branch 加进去
~~~

* $$ x_iW_{\mathrm{down}} $$:降维（不希望增加太多参数），把高维特征压缩到一个很小的瓶颈空间，因此Adapter 经常被称为：

  **bottleneck module，瓶颈模块。**

* $$ \sigma(x_iW_{\mathrm{down}}) $$:非线性激活，让 Adapter 不只是做简单的线性变换，而能够学习更复杂的任务特征。

* 原来的 ViT 给一个通用特征，Adapter 再在这个通用特征上增加一点当前任务需要的修正。

  所以 Adapter 不是把原来的特征完全推翻，而是增加一个**残差项**。

---

(5)$$任务特定 Adapter 的训练目标min_{\mathcal A\cup W} \sum_{(x,y)\in\mathcal D^b} \ell\left( W^\top\bar{\phi}(x;\mathcal A),y \right) \tag{5}$$

* $$\bar{\phi}(x;\mathcal A)$$:预训练模型的特征提取

* $$W^\top\bar{\phi}(x;\mathcal A)$$:线性分类
* $$\ell(\cdot,y)$$:分类损失，比较预测结果和真实标签y之间的差距

---

(6)$$ \Phi(x)= [\phi(x;\mathcal A_1),\cdots,\phi(x;\mathcal A_b)] \in\mathbb R^{bd} \tag{6} $$

*  每一个 Adapter 都提取一次特征、然后把它们拼接
* 这里容易知道，可以低成本的拓展新的特征子空间，旧类别在新空间中的prototype怎么办

---

(7)$$类别原型的计算p_{i,b}= \frac{1}{N} \sum_{j=1}^{|\mathcal D^b|} \mathbb I(y_j=i)\phi(x_j;\mathcal A_b) \tag{7} $$

* 主要作用：把同一个类别的所有样本特征取平均，得到这个类别的 Prototype（类别原型）
* $$ p_{i,b} $$:第 \(i\) 个类别，在 Adapter \(A_b\) 对应子空间中的类别原型
* $\phi(x_j;A_b)$:图片 \(x_j\) 经过预训练 Backbone + Adapter \(A_b\) 后得到的特征
* $ \mathbb I(y_j=i)$:(指示函数)只把属于类别 \(i\) 的样本挑出来
* 最终得到：某个类别在特征空间中的“中心位置”

---

(8) 新旧类别的语义相似度计算：

$$
\mathrm{Sim}_{i,j}
=
\frac{P_{o,o}[i]}{\lVert P_{o,o}[i]\rVert_2}
\frac{P_{n,o}[j]^\top}{\lVert P_{n,o}[j]\rVert_2}
\tag{8}
$$

* 计算旧类别 \(i\) 和新类别 \(j\) 在旧子空间里的余弦相似度

* 旧子空间：co-occurrence space，在这个空间里，旧类别和新类别的 Prototype 都能够得到

* $P_{o,o}$：旧类别在旧子空间里的 Prototype

  $P_{n,o}$：新类别在旧子空间里的 Prototype

  $P_{n,n}$：新类别在新子空间里的 Prototype

  $\hat P_{o,n}$：我们最终想估计的旧类别在新子空间里的 Prototype

* 把相似度做 Softmax：$ \mathrm{Sim}_{i,j} = \frac{ \exp(\mathrm{Sim}_{i,j}) }{ \sum_j\exp(\mathrm{Sim}_{i,j}) } $把相似度转换成一组可以用于加权的权重

---

(9) 语义引导的旧类别原型补全：

$$
\hat P_{o,n}[i]
=
\sum_j \mathrm{Sim}_{i,j}\times P_{n,n}[j]
\tag{9}
$$

* 用新类别在新子空间中的 Prototype，按照公式（8）得到的类别相似度进行加权组合，从而估计旧类别在新子空间中的 Prototype。
* 用公式8得到的关系，在新的Adapter重建特征
* 7：提取 Prototype-->8：计算类别关系-->9:补全 Prototype



---

(10)$$完整类别原型矩阵 P= \begin{bmatrix} P_{1,1} & \hat P_{1,2} & \cdots & \hat P_{1,B}\\ P_{2,1} & P_{2,2} & \cdots & \hat P_{2,B}\\ \vdots & \vdots & \ddots & \vdots\\ P_{B,1} & P_{B,2} & \cdots & P_{B,B} \end{bmatrix} \tag{10} $$

* 公式（6）已经把测试图像的所有子空间特征拼接起来了$ \Phi(x) = [ \phi(x;A_1),\dots,\phi(x;A_B) ] $,现在 Prototype 也要对应地在所有子空间中都有表示,于是对于某个类别，就可以得到：$ P_y= [ p_{y,1},p_{y,2},\dots,p_{y,B} ] $,然后拿它和：$ \Phi(x) $做相似度比较。得到下方的公式11

---

(11)$$多子空间集成分类[P_{b,1},P_{b,2},\cdots,P_{b,B}]^\top \Phi(x) = \sum_i P_{b,i}^\top\phi(x;\mathcal A_i) \tag{11} $$

* 同一个类别在不同子空间里分别和测试图像做匹配，然后把这些匹配分数加起来

---

(12)$$子空间重加权分类P_{b,b}^\top\phi(x;\mathcal A_b) + \alpha \sum_{i\neq b} P_{b,i}^\top\phi(x;\mathcal A_i) \tag{12}$$

* 某个类别应该最适合在它自己任务对应的 Adapter 中识别
* $ P_{b,b}^\top\phi(x;A_b) $:类别 \(b\) 所属任务自己的专属子空间得分
* $ \alpha \sum_{i\neq b} P_{b,i}^\top\phi(x;A_i) $:其他子空间的得分仍然保留，但统一乘一个较小的系数 α

# 实验

> EASE 在 7 个 CIL 基准数据集上整体取得了最优或具有竞争力的性能；相比 Prompt-based 方法，它在相近的参数开销下性能更高；相比传统 exemplar-based CIL 方法，它无需保存旧类别样本仍能取得较好结果。消融实验进一步表明，任务特定 Adapter、语义引导原型补全和子空间重加权三个模块均能有效提升性能；t-SNE 可视化和超参数实验也验证了任务特定子空间的有效性以及方法的稳定性。

论文自己也指出一个限制：虽然 Adapter 很轻量，只占 Backbone 参数量的一小部分，但随着任务越来越多，仍然需要不断保存新的 Adapter，因此模型大小还是会逐渐增长，未来可以考虑对 Adapter 进行压缩。

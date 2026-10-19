---
title: paper01
categories:
  - 文章阅读
tags:
  - AIGC
  - 静态泛化
  - 动态适应
description: 📌Fleet: Few Shots Lead Effective AI-generated Image Detection(ICML 2026,中科院计算所)
date: 2026-09-21 17:27:32
mathjax: true
---

~~~
对论文的解读分为以下四个方面：
背景
贡献
方法
实验
~~~

### 相关概念

#### GAN

GAN(Generative Adversarial Network，生成对抗网络)是一种**生成模型框架**。

GAN 最核心的是两个网络互相对抗：
$$
\boxed{\text{生成器 Generator} + \text{判别器 Discriminator}}
$$

$$
\min_G \max_D
V(D,G)
=
\mathbb E_{x\sim p_{\text{data}}}
[\log D(x)]
+
\mathbb E_{z\sim p_z}
[\log(1-D(G(z)))]
$$



#### 扩散模型

扩散模型：是一类生成模型
核心思想：训练时学会“如何把噪声一点点去掉”，生成时从一张纯噪声开始，逐步去噪，最终得到一张有意义的图像。
stable diffusion（一个文字生成图片的模型）背后的核心机制就是扩散模型。

前向扩散：原图 → 加一点噪声 → 再加噪声 → …… → 完全变成随机噪声
反向传播：纯噪声 → 去一点噪声 → 再去一点 → …… → 猫的图片
神经网络真正学习的是逐步去噪的过程。

目前理解的神经网络训练的核心机制：
$$
w
\rightarrow
\hat\epsilon
\rightarrow
L
$$

知道当前的神经网络参数w，通过该参数计算出epsilon-hat（也就是对噪声的预测值），通过训练算法，计算出损失Loss=（ε-ε-hat）^2,也就是前向传播。
$$
L
\rightarrow
\hat\epsilon
\rightarrow
w
$$

已经经过前向传播得到了模型预测值，并根据预测值和真实值计算出 Loss。接下来通过反向传播计算 Loss 对各个模型参数的梯度，判断“参数稍微改变时，Loss 会如何变化”。然后优化器利用这些梯度，通过梯度下降等方法不断调整参数 w，使 Loss 尽可能减小，最终得到一组能够让模型预测结果接近真实答案的参数。

#### DDPM

Denoising Diffusion Probabilistic Model:去噪扩散概率模型
核心思想：前向不断加噪，反向学习逐步去噪
$$
DDPM向前扩散：q(x_t \mid x_{t-1})
=
\mathcal{N}
\left(
x_t;
\sqrt{1-\beta_t}\,x_{t-1},
\beta_t I
\right)
$$

给上一时刻的图片 \(x_{t-1}\) 加一点高斯噪声，得到 \(x_t)，β__t可以理解为第t步要加多少噪声。

ε=随机高斯噪声。
$$
向前加噪的采样形式：x_t
=
\sqrt{1-\beta_t}\,x_{t-1}
+
\sqrt{\beta_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$



$$
\alpha_t = 1-\beta_t
$$

$$
\bar{\alpha}_t
=
\prod_{s=1}^{t}\alpha_s
$$



$$
从x_0直接得到x_t：x_t
=
\sqrt{\bar{\alpha}_t}\,x_0
+
\sqrt{1-\bar{\alpha}_t}\,\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

$$
L
=
\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon
-
\epsilon_\theta(x_t,t)
\right\|^2
\right]
$$

$$
DDPM反向过程：p_\theta(x_{t-1} \mid x_t)
=
\mathcal{N}
\left(
x_{t-1};
\mu_\theta(x_t,t),
\Sigma_\theta(x_t,t)
\right)
$$

$$
反向采样：x_{t-1}
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{1-\alpha_t}
{\sqrt{1-\bar{\alpha}_t}}
\epsilon_\theta(x_t,t)
\right)
+
\sigma_t z,
\qquad
z \sim \mathcal{N}(0,I)
$$


~~~
待学习：
转置卷积
幅度谱(magnitude spectrum)
方位角平均(azimuthal averaging)
~~~

#### 滤波器

按照某种规则，对输入信号进行处理，只保留或者强化我们需要的部分，削弱不想要的部分。--信号处理

图像处理滤波器：

图像本质可以看作一种二维信号：
$$
\text{图像}
\rightarrow
\text{滤波器}
\rightarrow
\text{处理后的图像}
$$
例如均值滤波器：
$$
K= \frac{1}{9} \begin{bmatrix} 1&1&1\\ 1&1&1\\ 1&1&1 \end{bmatrix} 
$$
它会把周围像素平均，所以图片变得更平滑，本质上属于一种**低通滤波**。

而边缘检测滤波器：
$$
K= \begin{bmatrix} -1&-1&-1\\ -1&8&-1\\ -1&-1&-1 \end{bmatrix}
$$
会突出像素变化大的位置，因此相当于强调高频信息。



#### 特征

将图像转换成一串数字：
$$
z=[z_1,z_2,\ldots,z_D]
$$
这串数字就叫做特征向量。

例如，一张图片经过 DINOv3（DINOv3 负责提取图片中丰富的视觉信息） 后得到：
$$
z\in\mathbb{R}^{1024}
$$
说明图片被表示成一个 1024 维的向量。每个维度不一定有明确的人类含义，但整体能够表达物体、纹理、结构、风格等信息。

#### Backbone

Backbone 可以理解成模型的“基础视觉特征提取器”。

Fleet 使用预训练的 **DINOv3** **作为视觉主干**。DINOv3 已经在大量图像上学过丰富的视觉知识，因此 Fleet 不需要从零开始学习。

论文将 DINOv3 的原始参数记为：
$$
\theta_{\mathrm{dino}}
$$
为了**避免少量样本破坏 DINOv3 原来的知识，论文冻结其主体参数，只通过 LoRA 对它进行轻量调整**。

#### LoRA

如果直接微调整个大型模型，需要更新大量参数，而且少量样本很容易造成**过拟合**和**灾难性遗忘**。

LoRA 不直接修改原权重，而是额外学习一个低秩更新量：
$$
\theta_{\mathrm{dino}}+\Delta\theta_{\mathrm{lora}}
$$
其中：

- $$
  \theta_{\mathrm{dino}}:原来冻结的 DINOv3 参数
  $$

  

- $$
  \Delta\theta_{\mathrm{lora}}:需要学习的小规模增量参数。
  $$

  

可以类比为：不重写整本教材，只在旁边增加少量补充笔记。

Fleet 使用的 LoRA rank 为 8，缩放系数为 16。

#### 子空间

完整的高维特征中包含多种信息，例如：

- 图像内容；
- 纹理；
- 色彩；
- 真实相机的成像规律；
- AI 生成模型的上采样或去噪痕迹。

Fleet 不希望所有信息都混在一个空间中，而是把特征映射到 \(C\) 个不同的子空间：
$$
\mathcal{S}_1,\mathcal{S}_2,\ldots,\mathcal{S}_C
$$
经过训练后，部分子空间会更偏向 AI 图像，部分子空间会更偏向 Non-AI 图像。

实验表明 \(C=8\) 时效果最好，因此可以把它理解成模型内部有 8 个不同的“特征专家”。

#### 路由

路由就是：

> 对于当前图片，决定应该重点使用哪些子空间。

模型为每个子空间生成一个权重：
$$
\alpha=[\alpha_1,\alpha_2,\ldots,\alpha_C]
$$
并且经过 Softmax 后满足：
$$
\sum_{i=1}^{C}\alpha_i=1
$$
例如：
$$
\alpha=[0.05,0.10,0.60,0.15,0.05,0.02,0.02,0.01]
$$
表示第 3 个子空间对当前图片最重要。

### 早期检测方法

#### 利用显性的低级伪影

##### 频域中的周期性上采样异常

title:periodic upsampling anomalies in the frequency domain
CNN类生成器（主要是GAN），在生成图像时的“上采样”操作中，在频域留下周期性的异常痕迹，检测器可以利用这个痕迹分辨真假图。

为什么GAN类模型会留下周期性的痕迹：
该类模型不是一步画出1024*1024的图片，而是从低分辨率开始，逐层“上采样”放大，每一层把分辨率翻倍，直到目标尺寸。上采样常用的操作是“转置卷积”（transposed convolution，也叫deconvolution），会导致经典问题“棋盘格伪影”，也就是像素值按固定的空间周期反复波动。

 Sources:
  CNNDetection 项目主页 (https://peterwang512.github.io/CNNDetection/)
  CNNDetection GitHub 仓库 (https://github.com/PeterWang512/CNNDetection)
  论文介绍(知乎) (https://zhuanlan.zhihu.com/p/112522632)

##### 根源于邻域像素关系的结构不一致

structural inconsistencies rooted in neighborhood pixel relationships
上采样会破坏相邻像素的自然关系，是一种比频域尖峰更通用更持久的伪造痕迹。

使用方法：NPR(neighboring pixel relationships)

#### 针对扩散模型的兴起

##### 重建差异（DIRE）

让扩散模型临摹一张图片。如果原图本来就是类似扩散模型画出来的，它会临摹得更像；真实照片则更难完全复原。
具体流程：
1. 输入图片 x；
2. 利用 DDIM inversion，把图片反推成噪声 x_T；
3. 再用扩散模型从该噪声去噪，得到重建图片 x'；
4. 计算原图与重建图的差异：

$$
\mathrm{DIRE}=|x-x'|
$$

5.把这张“误差图”送入一个二分类器，判断是真实图还是生成图。

source:
https://openaccess.thecvf.com/content/ICCV2023/html/Wang_DIRE_for_Diffusion-Generated_Image_Detection_ICCV_2023_paper.html

##### 伪造子空间（Nguyen 2025/Effort）

可以认为一张图可以粗略写成：
$$
\text{图像}=\text{场景内容}+\text{微观取证痕迹} 
$$
方法大致是：

1. 只使用真实图片，训练多个局部预测滤波器；
2. 每个滤波器根据周围像素预测中心像素；
3. 用真实像素减去预测值，得到残差；
4. 残差中主要保留相机成像、压缩、上采样和生成网络留下的细微规律；
5. 在多个尺度上对这些残差建立小型线性模型；
6. 这个模型的参数就构成该图片的“取证自描述”；
7. 用真实图片的自描述拟合一个 GMM 分布。新图片如果与真实分布相差很远，就判为 AI 图片。



##### 像素级轨迹（PLM）

它的出发点是：检测器很容易走捷径，例如看到某些物体、颜色或画风，就认为图片是 AI 生成的。这样的语义线索换一个数据集或生成器就可能失效。

PLM 会在图片送入分类器之前，对每个像素值进行一个非单调映射。例如，对 \(0~255\) 的像素值重新排列或非线性变换：
$$
\phi(v)=v-\operatorname{round}(v/256,2)\times256
$$
这样会产生两个效果：

- 原来相近的像素值经过映射后可能相差很大，正常的颜色和轮廓语义被破坏；
- 上采样、插值等生成过程留下的细微像素相关性和高频异常被放大。

于是分类器很难继续根据“这是一张人脸”“这是某种画风”来判断，只能更多依赖棋盘状噪声、局部像素关系和高频生成痕迹。

### Fleet

Few Shots Lead Effective AI-generated Image Detection:少量样本即可带来高效的AI生成图像检测
论文核心论点：不必追求用历史数据集训练出“一劳永逸”的静态特征检测器，而是只需用少量的新模型样本（新模型生成的图片），就能让检测器快速适配新出现的生成器，实现有效检测。

Fleet是实现该论点的核心方法。

#### 背景

 AI 图像检测方法不断演进（GAN-->DDPM）， 传统方法依赖“静态伪造特征假设”，在新模型上出现严重性能下降，某些方法甚至接近完全失效。

论文认为，试图找到一种永远有效的“万能伪造特征”过于理想化。

更现实的场景是：

1. 一个新生成器出现；
2. 原检测器无法识别；
3. 收集少量该生成器产生的图片；
4. 让检测器快速适应该生成器；
5. 同时保留对旧生成器和真实图片的识别能力。

因此，作者提出从：
$$
\text{静态零样本泛化}
$$
转向：
$$
\text{动态少样本适应}
$$
也就是不要求检测器永远不变，而是允许它随着生成技术的发展持续更新。

**现有少样本方法仍有两个问题：**

虽然已有方法也开始采用少样本学习，但仍存在不足。

第一，适应能力有限。一些方法只是在固定特征空间中进行原型匹配，没有真正更新特征提取能力，因此很难捕获新生成器独有的痕迹。

第二，容易灾难性遗忘。直接使用新样本微调模型，可能提高新生成器的检测率，却破坏原有特征空间，导致：

- 旧生成器检测性能下降；
- 真实图片被误判为 AI 图片；
- 多次连续适配后模型越来越不稳定。

#### 贡献

~~~
(1)New Paradigm: We expose the failure of the static artifact hypothesis and propose a dynamic paradigm of continuous few-shot evolution. （揭示了“静态伪造痕迹假设”的局限性，并提出一种“持续少样本演化”的动态检测范式，使检测器能够随着新型生成模型的出现而不断适应。）

(2)New Method: We introduce Fleet, a subspace routing framework that leverages avoidance routing to adapt to novel forgeries efficiently while maintaining stable Non-AI representations with limited drift.
（提出了 Fleet—一种子空间路由框架。该框架利用“规避路由”机制，高效适应来自新型生成模型的伪造图像，同时限制真实/非 AI 图像特征表示的漂移，从而保持检测器对 Non-AI 图像的稳定识别能力。）

(3)New Benchmark: We construct Treasure, a benchmark integrating closed-source engines and multidimensional annotations to rigorously stress-test adaptation in open-world scenarios.
（我们构建了Treasure,一个整合了闭源 引擎和多维度注释的基准,用于严格测试开放世界 场景中的适应能力。）
~~~

#### 方法

1.数据集不够全面：（闭源商业生成模型覆盖不足；最新扩散模型、自回归模型和统一多模态模型覆盖不足；缺少对不同语义、艺术风格和生成架构的系统分析。）

2.图像检测方法：（零样本检测方法：静态伪造特征假设。少样本检测方法：部分方法只进行静态原型匹配、忽视灾难性遗忘。）

构建treasure数据集

所以 Fleet 要解决的核心矛盾是：

> 如何用极少的新样本学会识别新生成器，同时尽量不破坏原有的 Non-AI 表示和旧知识？

现有基准往往包含较早、较少或以开源模型为主的生成器，无法体现现实中闭源商业模型的快速发展。

因此作者还构建了 Treasure 数据集：

- 64 类生成模型；
- 约 36 万张图片；
- 包含 20 个闭源商业生成器；
- 覆盖写实、2D 插画、3D 渲染和传统艺术等风格。

它的作用是暴露传统静态方法在新生成器上的性能崩溃，并验证 Fleet 是否能够通过少样本快速适应。



Fleet:**是一种互斥子空间路由方法**

高频分支负责观察图片是“怎么形成的”；

多个子空间相当于不同的特征通道；

路由权重决定一张图片主要走哪些通道；

预训练负责让 AI 和 Non-AI 走不同通道；

少样本适配负责纠正新 AI 图片走错通道的问题。

##### Figure 3：Fleet 整体框架图解

###### 1. 先认识图中的颜色和符号

| 图中元素 | 含义 |
| --- | --- |
| 红色子空间/柱子 | 训练后更偏向处理 AI 图像的子空间 |
| 绿色子空间/柱子 | 训练后更偏向处理 Non-AI 图像的子空间 |
| $\mathrm{SubSp}_1,\ldots,\mathrm{SubSp}_N$ | 多个可学习的特征子空间 |
| $\mathrm{Addr}_1,\ldots,\mathrm{Addr}_N$ | 各子空间的“地址”，用于和路由信号进行匹配 |
| $\alpha_1,\ldots,\alpha_N$ | 当前图片进入各子空间的路由权重 |
| 雪花标志 | DINOv3 主干参数被冻结，不进行普通的全参数更新 |
| LoRA | 加在 DINOv3 上的少量可训练参数 |
| $P$ | 投影网络，负责融合各子空间的输出 |
| $h_{\mathrm{final}}$ | 融合后的最终图像特征，用于真假分类 |

###### 2. 上方视觉分支：提取图片的主要特征

输入图片 $x$ 首先进入视觉编码器 $E_{\mathrm{vis}}$，即 DINOv3：

$$
z=E_{\mathrm{vis}}\left(x;\theta_{\mathrm{dino}}+\Delta\theta_{\mathrm{lora}}\right).
$$

其中，DINOv3 主体被冻结，训练时主要更新 LoRA，因此不需要用少量样本重训整个大模型。

提取到的视觉特征 $z$ 通过两组不同的仿射变换，分别产生：

$$
\text{子空间地址 }a_i
\qquad\text{和}\qquad
\text{子空间特征 }s_i.
$$

- $a_i$ 相当于第 $i$ 个子空间的“门牌号”，用于判断当前图片与这个子空间是否匹配；
- $s_i$ 相当于送入第 $i$ 个子空间的实际特征内容。

所以图中的 `Addr1 ... AddrN` 和 `SubSp1 ... SubSpN` 虽然都来自视觉特征 $z$，但作用不同：前者负责**匹配路由**，后者负责**保存和处理信息**。

###### 3. 下方高频分支：决定图片应该走哪条路

同一张输入图片还会进入下方的高频分支：

$$
x
\xrightarrow{\mathrm{FFT}}
\text{频域图像}
\xrightarrow{\text{高通掩码}}
\text{保留高频成分}
\xrightarrow{\mathrm{IFFT}}
X_{\mathrm{high}}.
$$

这一过程会削弱颜色、物体类别和画面语义等低频信息，尽量保留纹理边缘、局部噪声和生成过程留下的细微痕迹。

$X_{\mathrm{high}}$ 随后进入高频编码器 $E_{\mathrm{freq}}$，得到路由信号：

$$
r_{\mathrm{freq}}=E_{\mathrm{freq}}(X_{\mathrm{high}}).
$$

图中的圆圈乘号表示：使用 $r_{\mathrm{freq}}$ 与每个子空间地址 $a_i$ 进行匹配。经过点积和 Softmax 后，得到路由权重：

$$
\boldsymbol{\alpha}
=[\alpha_1,\alpha_2,\ldots,\alpha_N],
\qquad
\sum_{i=1}^{N}\alpha_i=1.
$$

如果 $\alpha_2$ 较大，就表示这张图片更需要使用第 2 个子空间；如果 $\alpha_4$ 很小，第 4 个子空间对这张图片的贡献就较小。

> 需要注意：$\alpha_i$ 不是“这张图是 AI 的概率”，而是“这张图对第 $i$ 个子空间的使用权重”。真正的 AI/Non-AI 分类发生在得到 $h_{\mathrm{final}}$ 之后。

###### 4. 中间子空间：按照路由权重分配信息

得到 $\alpha_i$ 后，Fleet 用它控制对应子空间特征 $s_i$ 的贡献：

$$
s_i\longrightarrow \alpha_i s_i.
$$

图中每个 $\alpha_i$ 向上连接到对应的 `SubSp`，表示路由与子空间是一一对应的：

$$
\alpha_1\leftrightarrow \mathrm{SubSp}_1,
\quad
\alpha_2\leftrightarrow \mathrm{SubSp}_2,
\quad\ldots\quad,
\alpha_N\leftrightarrow \mathrm{SubSp}_N.
$$

权重大的子空间保留更多信息，权重小的子空间贡献更少。所有加权后的子空间特征被拼接起来，再送入投影网络 $P$：

$$
h_{\mathrm{final}}
=P\left(\operatorname{Concat}left(
\alpha_1s_1,\ldots,\alpha_Ns_N
\right).
$$

这里的 $h_{\mathrm{final}}$ 才是 Fleet 最后用于分类的综合特征。

###### 5. 右侧分类模块：判断 AI 或 Non-AI

图的右侧用一个球形特征空间表示分类过程。红色和绿色箭头可理解为 AI 与 Non-AI 的类别方向或类别原型。

分类损失 $L_{\mathrm{cls}}$ 会让：

- AI 图像的 $h_{\mathrm{final}}$ 靠近 AI 类别原型；
- Non-AI 图像的 $h_{\mathrm{final}}$ 靠近 Non-AI 类别原型；
- 两类最终特征彼此分开。

因此，$L_{\mathrm{cls}}$ 解决的是“最终能否分对类别”，而路由损失解决的是“图片在模型内部应该走哪些子空间”。

###### 6. 下方 Pretrain：预训练阶段如何形成红绿子空间

图中下方左侧的 `Pretrain` 小框展示了 AI 和 Non-AI 的路由权重分布。

预训练阶段主要使用两个路由约束：

1. **正交损失 $L_{\mathrm{orth}}$**：让 AI 与 Non-AI 的平均路由分布尽量少重叠，使两类图片倾向于使用不同的子空间；
2. **覆盖损失 $L_{\mathrm{cov}}$**：防止所有图片都只使用少数几个子空间，避免其余子空间长期闲置。

二者配合后，模型逐渐形成图中所示的结构：

$$
\text{AI 图像主要激活红色子空间},
\qquad
\text{Non-AI 图像主要激活绿色子空间}.
$$

图中的 $L_{\alpha}$ 可以理解为对**路由权重 $\boldsymbol{\alpha}$ 施加的损失统称**，不是论文另外提出的一个独立损失：

- 预训练时，它主要对应 $L_{\mathrm{orth}}$ 和 $L_{\mathrm{cov}}$；
- 少样本适配时，它主要对应 $L_{\mathrm{avoid}}$。

###### 7. 下方 Few Shot：新生成图像走错路时如何纠正

新生成器产生的图像可能与真实图片非常接近，因此它的高频信号可能错误地激活绿色的 Non-AI 子空间。图中 `New Fake` 上方的柱状图就是在表示这种路由错误。

少样本适配时，$L_{\mathrm{avoid}}$ 不要求新 AI 图像完全复制旧 AI 图像的路由，而是先让它避开 Non-AI 路由锚点：

$$
\text{压低绿色 Non-AI 路径}
\quad\Longrightarrow\quad
\text{将权重重新分配到更合适的红色 AI 路径}.
$$

图中的训练前/训练后柱子用来对比规避路由前后的权重变化：错误的 Non-AI 路径受到抑制，相关 AI 路径的权重上升。

之所以可以实现“重新分流”，是因为 Softmax 保证：

$$
\sum_{i=1}^{N}\alpha_i=1.
$$

当一部分错误路径的权重下降时，减少的权重必须转移到其他路径。这就是论文所说的 **Shunt-and-Isolate（分流并隔离）**。

###### 8. 把整张图连成一次完整的前向过程

对于一张待检测图片，Fleet 的计算过程可以概括为：

$$
\boxed{
\begin{aligned}
\text{输入图片 }x
&\longrightarrow
\begin{cases}
\text{DINOv3+LoRA：产生地址 }a_i\text{ 和特征 }s_i,\\
\text{高频分支：产生路由信号 }r_{\mathrm{freq}}
\end{cases}\\
&\longrightarrow \text{计算路由权重 }\alpha_i\\
&\longrightarrow \text{对子空间特征加权 }\alpha_i s_i\\
&\longrightarrow \text{拼接并经过投影网络 }P\\
&\longrightarrow h_{\mathrm{final}}\\
&\longrightarrow \text{判断 AI 或 Non-AI}.
\end{aligned}}
$$

训练时，模型还会在这条前向流程的基础上计算不同损失；推理时不再计算这些损失，只使用已经训练好的路由结构和分类特征进行判断。

###### 9. 容易混淆的三个问题

| 容易产生的误解 | 正确理解 |
| --- | --- |
| 高频分支直接完成真假分类 | 高频分支主要生成路由信号，决定各子空间的权重 |
| $\alpha_i$ 是 AI 概率 | $\alpha_i$ 是第 $i$ 个子空间的使用权重 |
| 红色、绿色子空间由人工预先指定 | 它们是通过 $L_{\mathrm{orth}}$、$L_{\mathrm{cov}}$ 等损失逐渐学习出的功能偏向 |

> **汇报时可以这样讲：**Figure 3 展示了 Fleet 的双分支子空间路由框架。DINOv3 分支负责提取可用于分类的视觉特征，高频分支负责从生成痕迹中得到路由信号。路由信号与各子空间地址匹配后产生权重，再对各子空间特征进行加权融合。预训练利用正交与覆盖约束形成相对分离的 AI/Non-AI 路由；出现新生成器后，再通过规避损失压低错误的 Non-AI 路径，并把特征重新分流到更合适的 AI 子空间。

##### 4.1 子空间路由框架

###### 1. 双分支结构

**视觉分支：提取“图片是什么”**

输入图像 \(x\) 经过 DINOv3 和 LoRA：
$$
z=E_{\mathrm{vis}} \left( x;\theta_{\mathrm{dino}}+\Delta\theta_{\mathrm{lora}} \right) \in\mathbb{R}^{D}\hspace{4em}(1)
$$
其中：

- $$
  E_{\mathrm{vis}}：DINOv3 视觉编码器；
  $$

- \(x\)：输入图片；

- \(z\)：提取出的视觉特征；

- \(D=1024\)：特征维度。

这个分支会保留比较丰富的视觉信息，例如场景、物体、纹理和结构。

**高频分支：判断“图片是怎么形成的”**

AI 图像生成过程可能在高频区域留下人眼不容易看到的痕迹，例如：

- 上采样的周期性异常；
- 不自然的局部像素关系；
- 生成模型特有的纹理和噪声；
- 去噪过程留下的统计规律。

因此，Fleet 先对图像做快速傅里叶变换 FFT：
$$
\mathcal{F}(x)
$$
FFT 的作用是把图片从空间域转换到频率域：

- 低频：大面积颜色、轮廓、场景等平缓变化；
- 高频：边缘、纹理、噪声和细微像素变化。

$$
然后使用高通掩码 M_{\mathrm{high}} 去掉低频部分：
$$

$$
X_{\mathrm{high}} = \mathcal{F}^{-1} \left( \mathcal{F}(x)\odot M_{\mathrm{high}} \right)\hspace{4em}(3)
$$

其中：

- $$
  \odot：逐元素相乘；
  $$

- $$
  M_{\mathrm{high}}：高通滤波掩码；
  $$

- $$
  \mathcal{F}^{-1}：逆傅里叶变换；
  $$

- $$
  X_{\mathrm{high}}：只保留高频信息的图像。
  $$

论文实现中保留频率最高的约 50%。
$$
之后把X_{\mathrm{high}}输入 Xception 网络：
$$

$$
r_{\mathrm{freq}} = E_{\mathrm{freq}}(X_{\mathrm{high}}) \in\mathbb{R}^{d}
$$

$$
得到高频路由信号r_{\mathrm{freq}}，其中d=128。
$$

这个信号主要不是直接负责真假分类，而是负责告诉系统：

> 当前图片的底层生成痕迹更适合进入哪些子空间？



###### 2.每个子空间都有“地址”和“内容”

对于第 \(i\) 个子空间，Fleet 从视觉特征 \(z\) 中产生两个向量：
$$
\begin{aligned}
a_i &= W_i^{\mathrm{addr}} z + b_i^{\mathrm{addr}} \\
s_i &= W_i^{\mathrm{feat}} z + b_i^{\mathrm{feat}} \hspace{4em} (2)
\end{aligned}
$$
其中：

- \(a_i\)：第 \(i\) 个子空间的地址；
- \(s_i\)：第 \(i\) 个子空间的特征内容；
- \(W,b\)：可学习的线性层参数；

“线性层”或者“仿射变换”本质上就是：

\[ y=Wx+b \]

它负责把一个向量转换到另一个特征空间。

###### 3.如何计算路由权重

Fleet 计算高频路由信号与各子空间地址的点积：
$$
r_{\mathrm{freq}}^\top a_i
$$
点积越大，说明当前图片的高频痕迹与第 \(i\) 个子空间越匹配。

然后通过 Softmax 得到路由权重：
$$
\alpha_i= \frac{ \exp \left( r_{\mathrm{freq}}^\top a_i/\sqrt d \right) }{ \sum_{j=1}^{C} \exp \left( r_{\mathrm{freq}}^\top a_j/\sqrt d \right) }\hspace{4em}(5)
$$
这里除以 \(\sqrt d\) 是为了避免向量维度较高时点积过大，使 Softmax 过早饱和。这也是注意力机制中的常见做法。

Softmax 的关键作用是形成竞争关系：

- 某个子空间的权重增加；
- 其他子空间的相对权重就会下降；
- 所有权重之和始终为 1。

这对后面的 avoidance routing 非常重要：只要压低错误路径的权重，流量就会自动转移到其他路径。

###### 4. 如何得到最终特征？

每个子空间的特征乘以对应权重：
$$
\alpha_i s_i
$$
然后将它们拼接：
$$
\mathrm{Concat} \left( \alpha_1s_1,\ldots,\alpha_Cs_C \right)
$$
再经过一个投影网络 \(P(dot)\)：
$$
h_{\mathrm{final}} = P\left( \mathrm{Concat} ( \alpha_1s_1,\ldots,\alpha_Cs_C ) \right)
$$

$$
得到最终特征 h_{\mathrm{final}}
$$

这里使用拼接而不是直接相加，可以保留每个子空间自己的信息，随后再由投影网络完成融合。

论文说明，最终特征会用于构建 AI 和 Non-AI 的类别原型并完成分类。类别原型可以理解为某一类别在特征空间中的代表中心：
$$
\begin{aligned}
p_{\mathrm{AI}}\approx \operatorname{Mean}(h_{\mathrm{AI}})  \\ 
p_{\mathrm{Non-AI}}\approx \operatorname{Mean}(h_{\mathrm{Non-AI}})
\end{aligned}
$$
待检测图片离哪个原型更近，就更倾向于被判断为哪一类。论文第 4 节没有继续展开这一分类步骤的具体公式，因此这里是对“原型分类”概念的直观理解。





##### 4.2 互斥路由预训练

这一阶段不仅要让模型学会区分 AI 与 Non-AI，还要提前把内部的“路”整理好：

- AI 图像和 Non-AI 图像尽量走不同的子空间；
- 不能只使用一两个子空间，其他子空间也要发挥作用。

预训练的总损失为：

$$
L_{\mathrm{pre}}
=L_{\mathrm{cls}}
+\lambda_{\mathrm{orth}}L_{\mathrm{orth}}
+\lambda_{\mathrm{cov}}L_{\mathrm{cov}}. \tag{10}
$$

| 损失 | 直观作用 |
| --- | --- |
| $L_{\mathrm{cls}}$ | 让最终特征能够区分 AI 和 Non-AI |
| $L_{\mathrm{orth}}$ | 让两类图像尽量使用不同的路由 |
| $L_{\mathrm{cov}}$ | 防止所有样本都挤进少数子空间 |

###### 1. 对比分类损失 $L_{\mathrm{cls}}$

论文使用 InfoNCE 对比学习损失：同类样本靠近，异类样本远离。

$$
L_{\mathrm{cls}}
=-\frac{1}{B}\sum_{i=1}^{B}
\log
\frac{\exp\left(\operatorname{sim}(h_i,h_{\mathrm{pos}})/\tau\right)}
{\sum_{j\ne i}\exp\left(\operatorname{sim}(h_i,h_j)/\tau\right)}. \tag{7}
$$

| 符号 | 含义 |
| --- | --- |
| $B$ | 一个 batch 中的样本数 |
| $h_i$ | 当前锚点样本 $i$ 的最终特征 |
| $h_{\mathrm{pos}}$ | 与锚点同类别的正样本特征 |
| $h_j$ | batch 中参与比较的其他样本特征 |
| $\operatorname{sim}(\cdot,\cdot)$ | 特征相似度，通常采用余弦相似度 |
| $\tau$ | 温度系数，论文设为 $0.07$ |

余弦相似度的常见写法为：

$$
\operatorname{sim}(a,b)
=\frac{a^\top b}{\lVert a\rVert_2\lVert b\rVert_2}.
$$

- **分子**表示锚点与同类样本有多相似；
- **分母**表示锚点与所有对比样本的相似程度；
- 最小化该损失，就是让同类样本的相似度在所有对比对象中占比更大。

> 一句话理解：$L_{\mathrm{cls}}$ 负责“分得开”——AI 特征聚在一起，Non-AI 特征聚在一起，两类彼此分开。

###### 2. 路由正交约束 $L_{\mathrm{orth}}$

一张图片的路由向量记为：

$$
\boldsymbol{\alpha}=[\alpha_1,\alpha_2,\ldots,\alpha_C],
\qquad \sum_{c=1}^{C}\alpha_c=1.
$$

其中，$\alpha_c$ 表示这张图片对第 $c$ 个子空间的使用程度。Fleet 先分别计算一个 batch 中 AI 与 Non-AI 样本的平均路由：

$$
\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}
=\frac{1}{N_{\mathrm{AI}}}
\sum_{n=1}^{N_{\mathrm{AI}}}\boldsymbol{\alpha}^{(n)}_{\mathrm{AI}},
\qquad
\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}
=\frac{1}{N_{\mathrm{Non-AI}}}
\sum_{n=1}^{N_{\mathrm{Non-AI}}}\boldsymbol{\alpha}^{(n)}_{\mathrm{Non-AI}}.
$$

然后计算两个平均路由向量的点积：

$$
L_{\mathrm{orth}}
=\sum_{c=1}^{C}
\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}[c]
\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}[c]
=\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}^\top
\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}. \tag{8}
$$

| 符号 | 含义 |
| --- | --- |
| $C$ | 子空间数量 |
| $c$ | 子空间编号 |
| $N_{\mathrm{AI}}$、$N_{\mathrm{Non-AI}}$ | 当前 batch 中两类样本的数量 |
| $\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}[c]$ | AI 样本对第 $c$ 个子空间的平均权重 |
| $\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}[c]$ | Non-AI 样本对第 $c$ 个子空间的平均权重 |

路由权重由 Softmax 得到，都是非负数。因此，两个向量的点积越小，说明它们共同激活的子空间越少。

例如：

$$
\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}=[0.45,0.45,0.05,0.05],
\qquad
\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}=[0.05,0.05,0.45,0.45].
$$

此时：

$$
L_{\mathrm{orth}}
=0.45\times0.05+0.45\times0.05
+0.05\times0.45+0.05\times0.45
=0.09.
$$

前两个子空间偏向 AI，后两个子空间偏向 Non-AI，所以两类路由重叠较少。

> 注意：这里的“正交”是指 **AI 与 Non-AI 的平均路由分布尽量不重叠**，并不是要求所有子空间的投影矩阵在严格的线性代数意义上两两正交。

###### 3. 子空间覆盖约束 $L_{\mathrm{cov}}$

只使用正交约束时，模型可能让 AI 全部走子空间 1、Non-AI 全部走子空间 2，而剩余子空间完全闲置。这种现象称为**路由塌缩**。为此，论文加入覆盖损失：

$$
L_{\mathrm{cov}}
=-\frac{1}{C}\sum_{c=1}^{C}
\log\left(
\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}[c]
+\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}[c]
+\epsilon
\right). \tag{9}
$$

| 符号 | 含义 |
| --- | --- |
| $\bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}[c]+\bar{\boldsymbol{\alpha}}_{\mathrm{AI}}[c]$ | 第 $c$ 个子空间被两类样本使用的总体程度 |
| $\epsilon$ | 很小的正数，用于避免 $\log(0)$ |

如果某个子空间几乎没有被使用，括号内的数会接近 $0$，而 $-\log(x)$ 在 $x$ 接近 $0$ 时会变得很大，模型就会受到较大的惩罚。

需要注意，覆盖损失不是要求 AI 和 Non-AI 都平均使用所有子空间，而是要求：**每个子空间至少能被其中一类样本使用，不要长期闲置。**

###### 4. 预训练阶段小结

| 损失 | 解决的问题 | 如果没有它 |
| --- | --- | --- |
| $L_{\mathrm{cls}}$ | 建立真假分类能力 | 最终特征无法区分 AI 与 Non-AI |
| $L_{\mathrm{orth}}$ | 分开两类样本的路由 | AI 与 Non-AI 可能混用相同子空间 |
| $L_{\mathrm{cov}}$ | 保证多个子空间得到使用 | 路由可能集中在少数子空间并发生塌缩 |

附录中的预训练权重为：

$$
\lambda_{\mathrm{cls}}=1,
\qquad \lambda_{\mathrm{orth}}=0.1,
\qquad \lambda_{\mathrm{cov}}=10^{-4}.
$$

预训练使用 AIGIBench 训练集，共约 14.4 万张图像，其中 AI 图像来自 ProGAN 和 Stable Diffusion v1.4。完成后，模型已经形成初步的 AI/Non-AI 路由结构。

##### 4.3 基于规避路由的少样本适配

新生成器的痕迹可能与旧生成器不同。例如，一张新 AI 图像可能错误地激活 Non-AI 子空间：

$$
\boldsymbol{\alpha}_{\mathrm{new\ AI}}
\approx \bar{\boldsymbol{\alpha}}_{\mathrm{Non-AI}}.
$$

Fleet 不直接大幅改写整个特征空间，而是先纠正这张图片的路由：让它避开明显错误的 Non-AI 路径。这就是 **avoidance routing（规避路由）**。

整个适配过程可以概括为：

$$
\text{新样本走错路}
\xrightarrow{L_{\mathrm{avoid}}}
\text{压低错误路径}
\xrightarrow{\mathrm{Softmax}}
\text{权重重新分配到其他子空间}.
$$

###### 1. 回放集与路由锚点

作者从预训练数据中保留 1,000 张图像作为回放集：500 张 AI 图像和 500 张 Non-AI 图像。

$$
M=M_{\mathrm{AI}}\cup M_{\mathrm{Non-AI}}.
$$

回放集有两个用途：

1. 计算 AI 与 Non-AI 的典型路由；
2. 适配新生成器时帮助模型复习旧知识，减轻灾难性遗忘。

Fleet 对两类回放样本的路由向量分别求平均，得到两个固定的路由锚点：

$$
\boldsymbol{\mu}_{\mathrm{Non-AI}}
=\frac{1}{|M_{\mathrm{Non-AI}}|}
\sum_{x\in M_{\mathrm{Non-AI}}}\boldsymbol{\alpha}(x),
\qquad
\boldsymbol{\mu}_{\mathrm{AI}}
=\frac{1}{|M_{\mathrm{AI}}|}
\sum_{x\in M_{\mathrm{AI}}}\boldsymbol{\alpha}(x). \tag{11}
$$

| 符号 | 含义 |
| --- | --- |
| $M_{\mathrm{AI}}$、$M_{\mathrm{Non-AI}}$ | 回放集中的 AI、Non-AI 样本集合 |
| $\lvert M_{\mathrm{AI}}\rvert$、$\lvert M_{\mathrm{Non-AI}}\rvert$ | 两类回放样本数，均为 500 |
| $\boldsymbol{\alpha}(x)$ | 图像 $x$ 的完整路由向量 |
| $\boldsymbol{\mu}_{\mathrm{AI}}$ | 旧模型中典型的 AI 路由模式 |
| $\boldsymbol{\mu}_{\mathrm{Non-AI}}$ | 旧模型中典型的 Non-AI 路由模式 |

可以把 $\boldsymbol{\mu}_{\mathrm{AI}}$ 和 $\boldsymbol{\mu}_{\mathrm{Non-AI}}$ 理解成两类路由的“中心模板”。

###### 2. 规避损失 $L_{\mathrm{avoid}}$

规避损失要求新样本避开错误类别的路由锚点：

$$
L_{\mathrm{avoid}}
=\frac{1}{|S_{\mathrm{AI}}|}
\sum_{i\in S_{\mathrm{AI}}}
\boldsymbol{\alpha}_i^\top\boldsymbol{\mu}_{\mathrm{Non-AI}}
+\frac{1}{|S_{\mathrm{Non-AI}}|}
\sum_{j\in S_{\mathrm{Non-AI}}}
\boldsymbol{\alpha}_j^\top\boldsymbol{\mu}_{\mathrm{AI}}. \tag{12}
$$

| 符号 | 含义 |
| --- | --- |
| $S$ | 当前少样本支持集 |
| $S_{\mathrm{AI}}$、$S_{\mathrm{Non-AI}}$ | 支持集中的 AI、Non-AI 样本 |
| $\lvert S_{\mathrm{AI}}\rvert$、$\lvert S_{\mathrm{Non-AI}}\rvert$ | 两类支持样本的数量 |
| $\boldsymbol{\alpha}_i$ | 第 $i$ 个样本的完整 $C$ 维路由向量 |
| $\boldsymbol{\alpha}_i^\top\boldsymbol{\mu}$ | 样本路由与某个路由锚点的重合程度 |

以一张新 AI 图像为例，公式最小化：

$$
\boldsymbol{\alpha}_{\mathrm{new\ AI}}^\top
\boldsymbol{\mu}_{\mathrm{Non-AI}}.
$$

- 如果新 AI 图像的路由与 Non-AI 锚点很相似，点积较大，损失也较大；
- 训练会压低这些 Non-AI 相关路径的权重；
- 因为 Softmax 保证所有路由权重之和为 $1$，被压低的权重会重新分配给其他子空间；
- 预训练已经建立了两类路由结构，所以这些权重通常会转移到更合适的 AI 子空间。

这就是论文所说的 **Shunt-and-Isolate（分流并隔离）**。

这里有两个容易混淆的地方：

1. $\boldsymbol{\alpha}_i$ 是第 $i$ 个样本的完整路由向量，不是“第 $i$ 个子空间的一个权重”；
2. $L_{\mathrm{avoid}}$ 直接要求新 AI 远离 Non-AI 锚点，但没有强制它完全靠近旧 AI 锚点。这样可以为新生成器的未知伪造痕迹保留新的路由空间。

**数值示例：**假设模型有 4 个子空间：

$$
\boldsymbol{\mu}_{\mathrm{AI}}=[0.45,0.45,0.05,0.05],
\qquad
\boldsymbol{\mu}_{\mathrm{Non-AI}}=[0.05,0.05,0.45,0.45].
$$

一张新 AI 图像适配前可能被错误路由为：

$$
\boldsymbol{\alpha}_{\mathrm{before}}=[0.10,0.10,0.40,0.40].
$$

经过规避路由后可能变为：

$$
\boldsymbol{\alpha}_{\mathrm{after}}=[0.40,0.45,0.10,0.05].
$$

这表示新 AI 图像从 Non-AI 相关子空间被重新分流到了 AI 相关子空间。以上数值仅用于帮助理解，并非论文中的真实样本结果。

###### 3. 防遗忘蒸馏损失 $L_{\mathrm{distill}}$

如果模型只学习少量新样本，可能学会识别新生成器，却忘记原来学过的生成器和 Non-AI 图像，这叫**灾难性遗忘**。

Fleet 保存适配前的旧模型，并要求新旧模型对同一张回放图片产生相近的最终特征：

$$
L_{\mathrm{distill}}
=\frac{1}{|M|}\sum_{k\in M}
\left[
1-\operatorname{sim}\left(
h_{\mathrm{final}}^{\mathrm{curr}}(x_k),
h_{\mathrm{final}}^{\mathrm{old}}(x_k)
\right)
\right]. \tag{13}
$$

| 符号 | 含义 |
| --- | --- |
| $x_k$ | 第 $k$ 个回放样本 |
| $h_{\mathrm{final}}^{\mathrm{old}}(x_k)$ | 适配前的旧模型提取的特征 |
| $h_{\mathrm{final}}^{\mathrm{curr}}(x_k)$ | 当前模型提取的特征 |
| $\operatorname{sim}(\cdot,\cdot)$ | 余弦相似度 |

如果新旧特征的余弦相似度为 $0.98$，对应损失就是 $1-0.98=0.02$，说明旧知识保持得较好；如果相似度很低，损失就会变大，迫使当前模型不要偏离旧模型太多。

> 这里的蒸馏不是“小模型模仿大模型”，而是“更新后的模型在旧样本上模仿更新前的自己”。旧模型相当于教师，当前模型相当于学生。

###### 4. 少样本适配总损失

论文正文给出的适配总损失为：

$$
L_{\mathrm{adapt}}
=L_{\mathrm{cls}}
+\lambda_{\mathrm{avoid}}L_{\mathrm{avoid}}
+\lambda_{\mathrm{distill}}L_{\mathrm{distill}}. \tag{14}
$$

三个损失分别负责：

| 损失 | 作用 | 如果没有它 |
| --- | --- | --- |
| $L_{\mathrm{cls}}$ | 让新样本能够被正确分类 | 路由合理，但最终分类仍可能不准 |
| $L_{\mathrm{avoid}}$ | 让新样本避开错误路由 | 新 AI 图像可能继续进入 Non-AI 子空间 |
| $L_{\mathrm{distill}}$ | 保持旧样本的特征表示 | 学习新生成器时容易发生灾难性遗忘 |

正文公式省略了 $L_{\mathrm{cls}}$ 前面的系数；附录中的实际实现可写为：

$$
L_{\mathrm{adapt}}
=\lambda_{\mathrm{cls}}L_{\mathrm{cls}}
+\lambda_{\mathrm{avoid}}L_{\mathrm{avoid}}
+\lambda_{\mathrm{distill}}L_{\mathrm{distill}},
$$

其中：

$$
\lambda_{\mathrm{cls}}=5,
\qquad \lambda_{\mathrm{avoid}}=20,
\qquad \lambda_{\mathrm{distill}}=10.
$$

适配阶段共训练 20 个 epoch。前 5 个 epoch 令 $\lambda_{\mathrm{cls}}=0$，暂时不优化分类损失，先纠正路由并稳定旧特征；第 6 个 epoch 起再加入分类损失。这可以概括成：

$$
\text{前 5 轮：先把路走对}
\quad\Longrightarrow\quad
\text{后 15 轮：再把类别分准}.
$$

附录还说明，每个适配 batch 由“当前支持集的全部样本 + 从回放集中交替抽取的 32 张图片”组成，学习率为 $3\times10^{-5}$。

##### 4.4 两个训练阶段的关系

| 阶段 | 使用的数据 | 主要损失 | 主要目标 |
| --- | --- | --- | --- |
| 预训练 | 大量历史 AI 与 Non-AI 图像 | $L_{\mathrm{cls}}+L_{\mathrm{orth}}+L_{\mathrm{cov}}$ | 建立分类能力，并形成互斥且不塌缩的路由结构 |
| 少样本适配 | 新生成器的少量样本 + 回放集 | $L_{\mathrm{cls}}+L_{\mathrm{avoid}}+L_{\mathrm{distill}}$ | 纠正新样本的路由，同时保护旧知识 |

> Fleet 在预训练阶段先为 AI 和 Non-AI 建立相对分离的子空间；当新生成器出现时，它不依赖大幅修改全部特征，而是用规避损失压低错误路径，再利用 Softmax 的竞争关系把新样本分流到更合适的子空间，同时通过回放和蒸馏减轻灾难性遗忘。



#### 方法总结

##### 图解：

* FFT：（快速傅里叶变换）FFT 的作用是把图片从**空间域**转换到**频率域**：$\mathcal{F}(x)$

* mask:$M_{\mathrm{high}}$ 是高通滤波掩码，操作为$\mathcal{F}(x)\odot M_{\mathrm{high}}$，保留高频带

* IFTT:（快速傅里叶逆变换）：将过滤后的频谱，转换为**空间域**：$ X_{\mathrm{high}}=\mathcal F^{-1}(F_{\mathrm{high}}) $，主要包含边缘，细微纹理，局部噪声，高频生成痕迹等

* 专门处理高频图像的**卷积特征提取器**：$E_{\mathrm{freq}}$（论文实现中采用的是 **Xception**）。由卷积特征提取器得到**路由信号**：$r_{\mathrm{freq}} = E_{\mathrm{freq}}(X_{\mathrm{high}}) \in\mathbb R^d$.

* 并不是直接转换成路由信号，而是经过了一个**投影层**：Proj：将Xception 输出的特征转换成固定的 $d$ 维路由信号：$text{Xception 特征} \xrightarrow{\mathrm{Proj}} r_{\mathrm{freq}}\in\mathbb R^d$.

* 视觉编码器 $E_{\mathrm{vis}}$：即是DINOv3：$z=E_{\mathrm{vis}}\left(x;\theta_{\mathrm{dino}}+\Delta\theta_{\mathrm{lora}}\right)$.为了**避免少量样本破坏 DINOv3 原来的知识，论文冻结其主体参数，只通过 LoRA 对它进行轻量调整**。如果直接微调整个大型模型，需要更新大量参数，而且少量样本很容易造成**过拟合**和**灾难性遗忘**。

  LoRA 不直接修改原权重，而是额外学习一个低秩更新量

* Affine：通过两组不同的仿射变换，分别产生$\text{子空间地址 }a_i{和}\text{子空间特征 }s_i$.前者负责**匹配路由**，后者负责**保存和处理信息**

* 图中的圆圈乘号：使用 $r_{\mathrm{freq}}$ 与每个子空间地址 $a_i$ 进行匹配。经过点积和 Softmax 后，得到**路由权重**：

  $$
  \boldsymbol{\alpha}
  =[\alpha_1,\alpha_2,\ldots,\alpha_N],
  \qquad
  \sum_{i=1}^{N}\alpha_i=1.
  $$

  如果 $\alpha_2$ 较大，就表示这张图片更需要使用第 2 个子空间；如果 $\alpha_4$ 很小，第 4 个子空间对这张图片的贡献就较小。

* 路由权重作用到子空间：

* 会先并行计算所有子空间特征：$ s_1,s_2,\ldots,s_C. $然后用路由权重缩放它们:$\tilde s_i=\alpha_i s_i. $

* 拼接：$ u = \operatorname{Concat} (\alpha_1s_1,\alpha_2s_2,\alpha_3s_3,\alpha_4s_4)$.

* 投影网络 \(P)：$h_{\mathrm{final}}=P(u)$.，这里P如何知道应该怎样融合:(P\) 一开始是随机初始化的，并不知道哪些特征重要。
  训练时，根据 $h_{\mathrm{final}}$计算分类损失：$h_{\mathrm{final}}\longrightarrow
  L_{\mathrm{cls}}.$然后反向传播：$L_{\mathrm{cls}}
  \longrightarrow
  h_{\mathrm{final}}
  \longrightarrow
  P
  \longrightarrow
  \alpha_i s_i.
  $
  因此，$L_{\mathrm{cls}}$ 会同时训练：

  - 投影网络 \(P\)；
  - 子空间特征 \(s_i\)；
  - 路由权重 \(\alpha_i\) 的产生过程。如果某些子空间组合有利于区分 AI 与 Non-AI，\(P\) 就会逐渐学会加强这些组合。

##### 公式：

$$
z = E_{\mathrm{vis}} \left( x; \theta_{\mathrm{dino}} + \Delta\theta_{\mathrm{lora}} \right) \in\mathbb{R}^{D} \tag{1}
$$

```
Evis是视觉编码器，也就是DINOv3,冻结DINOv3参数，只用LoRA进行小规模学习，防止过拟合和灾难性遗忘。z即为输出的图像特征。
```

------

$$
\begin{aligned}
a_i &= W_i^{\mathrm{addr}} z + b_i^{\mathrm{addr}} \\
s_i &= W_i^{\mathrm{feat}} z + b_i^{\mathrm{feat}} \hspace{4em} (2)
\end{aligned}
$$

```
其中W和b是作者定义的特定的线性投影函数，通过（1）生成的图像特征z，得到地址和特征。
```

------

$$
X_{\mathrm{high}}
=
\mathcal{F}^{-1}
\left(
\mathcal{F}(x)\odot M_{\mathrm{high}}
\right)
\tag{3}
$$

~~~
为了消除低频成分影响，保留对生成机制敏感的高频信号：先做快速傅里叶变换，再利用M_high（高通滤波掩码）将低频成分置零，然后在做逆变换
~~~

------

$$
r_{\mathrm{freq}}
=
E_{\mathrm{freq}}(X_{\mathrm{high}})
\in\mathbb{R}^{d}
\tag{4}
$$

~~~
由（3）得到高频信号X_high后，再输入到卷积编码器E_freg中，得到路由信号
~~~

------

$$
\alpha_i
=
\frac{
\exp
\left(
r_{\mathrm{freq}}^\top a_i/\sqrt d
\right)
}{
\sum_{j=1}^{C}
\exp
\left(
r_{\mathrm{freq}}^\top a_j/\sqrt d
\right)
}
\tag{5}
$$

~~~
本质是使用Softmax规避路由。
exp即为e^x，将所有的数变成正数，并放大分数之间的差异；
除以根号d，因为向量维度较高时，点积可能非常大，这样可以让数值更稳定；
分母：将所有子空间的指数分数相加，用来归一化。
α_i：表示第i个子空间的路由权重。
~~~

------

$$
h_{\mathrm{final}}
=
P
\left(
\mathrm{Concat}
(
\alpha_1s_1,\ldots,\alpha_Cs_C
)
\right)
\tag{6}
$$

~~~
cantact：不同子空间的特征首尾相接
P：投影网络，用来融合、压缩和重新映射拼接后的特征
h_final:最终图像特征，同时包含DINOv3的视觉表示，高频信号决定的路由选择，不同子空间的特征信息。
~~~

------

$$
L_{\mathrm{cls}}
=
-\frac1B
\sum_{i=1}^{B}
\log
\frac{
\exp
\left(
\mathrm{sim}(h_i,h_{\mathrm{pos}})/\tau
\right)
}{
\sum_{j\ne i}
\exp
\left(
\mathrm{sim}(h_i,h_j)/\tau
\right)
}
\tag{7}
$$

~~~
对比学习损失：L_cls
该公式的目标：让同类别样本靠近，让不同类别样本远离。
为确保路由机制的有效性，在预训练阶段在子空间之间执行功能区分。

h_i 表示锚点样本,hpos 表示与锚点属于同一类别的正样本,而h_j表示来自批次中不同类别的负样本。 B 和 τ 分别表示批次大小和温度。

这里希望分子趋向于1.分母趋向于0；损失L_cls就会越小。
~~~

------

$$
L_{\mathrm{orth}}
=
\sum_{c=1}^{C}
\bar{\alpha}_{\mathrm{Non-AI}}[c]
\bar{\alpha}_{\mathrm{AI}}[c]
\tag{8}
$$

~~~
路由正交损失：
AI和Non-AI点积，其中横线表示平均路由权重，c表示向量的第c个分量。
如果AI和Non-AI的路由权重类似，那么损失会更大。如果路由权重分布不重合，那么损失会很小。这里就是最小化该损失。
~~~

---

$$
L_{\mathrm{cov}}
=
-\frac1C
\sum_{c=1}^{C}
\log
\left(
\bar{\alpha}_{\mathrm{Non-AI}}[c]
+
\bar{\alpha}_{\mathrm{AI}}[c]
+\epsilon
\right)
\tag{9}
$$

~~~
子空间覆盖损失：
防止某些子空间永远不被使用；
括号中的内容表示，第c个子空间被两类样本使用的总程度，如果这个数字接近0，说明这个子空间基本闲置。子空间使用的越少，损失越大。
ε是一个极小的正数，防止出现log(0)
~~~

---

$$
L_{\mathrm{pre}}
=
L_{\mathrm{cls}}
+
\lambda_{\mathrm{orth}}L_{\mathrm{orth}}
+
\lambda_{\mathrm{cov}}L_{\mathrm{cov}}
\tag{10}
$$

~~~
预训练总损失：
其中λ表示损失权重，决定每项损失的重要程度。
论文中，设置λ_orth=0.1，λ_cov=0.0001
~~~

---

$$
\mu_{\mathrm{Non-AI}}
=
\frac1{|M_{\mathrm{Non-AI}}|}
\sum_{x\in M_{\mathrm{Non-AI}}}
\alpha(x)
$$

$$
\mu_{\mathrm{AI}}
=
\frac1{|M_{\mathrm{AI}}|}
\sum_{x\in M_{\mathrm{AI}}}
\alpha(x)
\tag{11}
$$

~~~
通过规避路径进行少量样本适应：
M：Replay set，回放集。论文保留 1,000 张旧图片：
- 500 张 AI；- 500 张 Non-AI。

M_AI:回放集中的 AI 图片子集。

|M_AI|:AI 回放样本的数量：|M_AI|=500

4.x_in M_AI:表示遍历 AI 回放集中的每一张图片 \(x\)。
5.alpha(x):图片x对应的路由权重
6.μ_AI
将所有旧 AI 图片的路由向量求平均，得到典型 AI 路由模式。
它可以理解为：
μ_AI=AI类路由中心
μ_Non-AI=Non-AI类路由中心
论文将它们称为 anchor，也就是锚点。

在适应阶段,为了解决新模型生成的图像意外触发非 AI子空间激活的问题,我们提出了一种规避路由策略。 最初,我们从训练集中随机选择1,000张非AI和AI图 像来构建重放集 M,遵循持续学习中的既定做法( Rebuffi 等,2017)。  重放锚点的计算。我们利用重放集 M 来计算在预训 练阶段建立的固定锚点分布(质心):

~~~

---

$$
L_{\mathrm{avoid}}
=
\frac1{|S_{\mathrm{AI}}|}
\sum_{i\in S_{\mathrm{AI}}}
\left(
\alpha_i\cdot\mu_{\mathrm{Non-AI}}
\right)
+
\frac1{|S_{\mathrm{Non-AI}}|}
\sum_{j\in S_{\mathrm{Non-AI}}}
\left(
\alpha_j\cdot\mu_{\mathrm{AI}}
\right)
\tag{12}
$$

~~~
规避路由损失：
S:包含少量新生成器样本和相应的 Non-AI 样本，用于适配模型
S_AI:支持集中的新 AI 图片。
α_i:第i张支持图片的路由分布

公式第一项括号：计算新 AI 图片路由与旧 Non-AI 路由中心的相似程度，最小化它，就是要求：新AI路由远离Non-AI路由中心

公式第二项括号：要求支持集中的 Non-AI 图片避开 AI 路由中心。
~~~

---

$$
L_{\mathrm{distill}}
=
\frac1{|M|}
\sum_{k\in M}
\left(
1-
\mathrm{sim}
\left(
h_{\mathrm{final}}^{\mathrm{curr}}(x_k),
h_{\mathrm{final}}^{\mathrm{old}}(x_k)
\right)
\right)
\tag{13}
$$

~~~
防遗忘蒸馏损失：
sim(适配前的旧模型对x_k提取的最终特征,正在适配的新模型对同一张图片提取的最终特征),相似度越大，新旧特征的差异越小，损失越小。
使得更新后的模型尽量保持旧样本的特征方向。

知识蒸馏：这里的“教师模型”是适配前的旧模型，“学生模型”是正在适配的新模型。旧模型告诉新模型：
你可以学习新生成器，但对于这些旧图片，提取出的特征不要变化太大。
~~~

---

$$
L_{\mathrm{adapt}}
=
L_{\mathrm{cls}}
+
\lambda_{\mathrm{avoid}}L_{\mathrm{avoid}}
+
\lambda_{\mathrm{distill}}L_{\mathrm{distill}}
\tag{14}
$$

~~~
少样本适配总损失：
论文设置：λ_cls=5；λ_avoid=20;λ_distill=10；
前5个epoch 中：λ_cls=0;此时只优化路由和知识保持。后续再恢复分类损失，相当于：
1. 先让新样本走对路；
2. 再让最终分类更加准确。
~~~

---

~~~
输入图片 x

1. 使用 DINOv3 + LoRA 提取视觉特征 z             公式(1)
2. 从 z 生成每个子空间的地址 a_i 和特征 s_i       公式(2)
3. 对 x 做 FFT 和高通滤波，得到高频图像           公式(3)
4. 从高频图像提取路由信号 r_freq                  公式(4)
5. 比较 r_freq 与每个地址，得到路由权重 alpha      公式(5)
6. 用 alpha 加权各子空间特征，得到最终特征          公式(6)

预训练：
7. 拉近同类样本、推远异类样本                    公式(7)
8. 让 AI 和 Non-AI 使用不同路由                   公式(8)
9. 防止某些子空间完全闲置                         公式(9)
10. 将三个损失加起来进行预训练                    公式(10)

少样本适配：
11. 计算旧 AI 和 Non-AI 的平均路由锚点            公式(11)
12. 让新 AI 避开 Non-AI 路由，让 Non-AI 避开 AI   公式(12)
13. 保持新旧模型对回放样本的特征一致              公式(13)
14. 将分类、规避和蒸馏损失相加                    公式(14)

较为重要的：5、8、12、13
~~~

#### 实验：

历史数据预训练-->选择一个新生成器-->抽取少量支持样本-->少样本适配-->在多个测试集上评价-->与其他方法比较

5.1数据集和评估协议

~~~
支持集符合学习新知识，回放集负责复习旧知识，查询集负责考试

协议 I 在 Treasure 数据集的每个类别 上进行少样本评估,该数据集包含 64 个不同的生成模型。 协议 II 在公共 AIGIBench 测试集的每个类别上进行少 样本评估,该测试集包含 13 个测试子集。

~~~

5.2基线与评估指标

~~~
用Fleet与下面两个方法进行比较：
零样本方法：训练完成后不使用新生成器样本，直接检测新生成器图像。
该比较主要作用：使用少量新样本适配的 Fleet，是否比完全不适配的静态检测器更有效？

少样本方法（FSD）：
该比较的主要作用：在同样只提供少量新样本的情况下，Fleet 的路由适配是否优于已有少样本学习方法？


~~~

Table1:

~~~
实验结果集：Table1
用于比较不同方法在两个基准上的表现，并同时回答三个问题：
1.能不能识别新生成器产生的AI图像
2.会不会把真实图像误判成AI
3.少样本适配后，会不会忘记预训练阶段的旧知识

分别测试ZeroShot和FewShot两个方法，在treasure数据集和AIGIBench-13数据集上的表现

其中Method指的是，在两个大类下的不同方法。Venue指的是，该方法出自何处。

主要的对比指标示例说明如下：
~~~

| 指标         | 数值  | 含义                           |
| ------------ | ----- | ------------------------------ |
| AI           | 84.23 | 84.23% 的新 AI 图像被正确识别  |
| Non-AI       | 97.50 | 97.50% 的真实图像被正确判断    |
| mAcc         | 90.87 | AI 与 Non-AI 准确率的平均值    |
| mAP          | 95.26 | 模型整体置信度排序能力较强     |
| Pretrain Val | 99.97 | 少样本适配后几乎没有忘记旧知识 |

~~~
表1的核心结论可以概括为：
Fleet 在只使用少量新生成器样本的情况下，能够显著提高新 AI 图像检测性能；随着 shot 数量增加，性能继续提升。同时，它保持了较高的 Non-AI 准确率和接近 100% 的预训练验证集准确率，说明 Fleet 在快速适应新生成器的同时，有效减轻了灾难性遗忘。
~~~

table2(消融实验):

| 方法                       | Query Set | Non-AI | Pretrain Val | mAcc      |
| -------------------------- | --------- | ------ | ------------ | --------- |
| Ours                       | 91.01     | 95.51  | 99.95        | **93.26** |
| w/o \(L_{\mathrm{avoid}}\) | 81.60     | 96.89  | 99.97        | 89.25     |
| w/o \(L_{\mathrm{cov}}\)   | 90.66     | 94.14  | 99.94        | 92.40     |
| w/o \(L_{\alpha}\)         | 82.69     | 95.74  | 99.95        | 89.22     |
| baseline                   | 86.08     | 96.61  | 99.97        | 91.35     |

~~~
ours:表示完整的Fleet
w/0是without的缩写，w/o_L_avoid表示Fleet删掉规避损失阶段，其中需要特别注意的是，w/o(L_alpha)，论文规定L_alpha包含，L_orth,L_cov,L_avoid，也就是去掉所有路由相关的专门损失。

为什么没有做其他损失的相关实验，以L_orth为例，我理解的是，该损失是规避路由损失的前提，没办法抛弃正交路由损失后，再将规避路由损失加进去，所以作者更倾向于将三者相互作用的损失L_α(有关路由的损失)进行实验。

baseline:这里指的是参数高效微调方案，使用 DINOv3；
使用 LoRA；
使用相同的回放集；
使用对比分类损失；
使用蒸馏损失；
但是不使用 Fleet 的子空间路由机制。
这个数据说明子空间路由结构本身带来了额外的提升。

Table 2 的核心结论
1. L_avoid对新生成器检测最关键；
2. L_cov能改善子空间使用和整体平衡；
3. 去掉全部路由损失后，性能明显下降；
4. Fleet 的子空间路由结构优于普通 DINOv3+LoRA 微调；
5. 路由损失主要提高新任务适配能力，而旧知识保留仍然稳定。
~~~

Table3:

| LoRA Rank | AI    | Non-AI | Pretrain Val | Acc       |
| --------- | ----- | ------ | ------------ | --------- |
| Rank = 2  | 85.85 | 96.65  | 99.94        | 91.25     |
| Rank = 4  | 91.03 | 95.21  | 99.94        | 93.12     |
| Rank = 8  | 91.01 | 95.51  | 99.95        | **93.26** |
| Rank = 16 | 90.64 | 93.89  | 99.98        | 92.26     |
| Rank = 32 | 92.00 | 90.91  | 99.90        | 91.46     |

~~~
Table 3 的核心结论
1. rank 太小会限制模型学习新生成器；
2. rank 太大可能损害 Non-AI 分类稳定性；
3. Rank = 8 的 AI/Non-AI 平衡最好，整体 Acc 最高；
4. Rank = 4 与 Rank = 8 非常接近，说明 Fleet 对合理范围内的 rank 并不过度敏感；
5. 更大的 LoRA 并不必然带来更好的综合性能。
~~~

table4(少样本连续适配):在多步连续适应场景之外具有良好的潜力

table5:

~~~
Mixture of Experts，专家混合模型。

默认 Fleet：多个子空间从随机状态开始：随机参数-->通过训练逐渐形成分工
MoE_init：先让子空间从一个已经具有基本投影能力的起点开始:预训练低维投影-->初始化多个子空间-->继续训练形成分工

Table 5 分别研究了子空间初始化和地址设计。MoE_init 只是使用预训练低维投影为子空间提供结构化初始值，后续仍然使用动态地址；它相比随机初始化只带来 0.27 个百分点的提升，说明 Fleet 不依赖特殊初始化。Static \(A_i\) 则把每个子空间的输入依赖地址替换成对所有图片相同的固定原型，使综合准确率从 93.26% 降到 92.21%，且主要损失发生在 AI 类。这表明动态地址能够同时结合视觉特征和高频信号，更适合匹配不断变化的新生成器痕迹。
~~~

##### 结论与局限

~~~
论文在结论中指出，作者通过Treasure数据集和Fleet框架，将AI生成图像检测从传统的静态泛化问题，转化为面向新生成器的少样本动态适配问题。Fleet利用频率信息计算路由权重，通过互斥子空间缓解新型AI图像特征被Non-AI特征淹没的问题；在少样本适配阶段，又通过规避路由和知识蒸馏，在学习新生成器的同时减少对旧知识的遗忘。实验结果表明，该方法在多个基准、新型生成器适配和顺序适配方面具有较好的效果。不过，它目前主要面向整图生成检测，尚未充分验证局部编辑和换脸任务；持续少样本学习仍处于初步探索阶段；同时，方法依赖回放集，会产生存储、隐私和长期扩展方面的成本。因此，Fleet更适合被理解为一种有潜力的动态检测范式，而不是已经完全解决了开放世界AI图像检测问题。
~~~






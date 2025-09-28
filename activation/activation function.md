## 激活函数公式总结

### sigmoid

$$
\sigma (x) = 1 / (1 + e^{-x})
$$

特征：

- 单调递增
- 结果属于(0, 1)，在x=0附近线性，在较远处饱和
- 关于(0, 0.5)中心对称

缺点：

- 需要指数计算
- 两端输入值部分，梯度趋近于0
- 输出不是零中心，降低收敛速度

### softmax

$$
softmax(x) = e^{x_i} / \Sigma^n_{j=1}e^{x_j}
$$

特征：  
- 值域(0, 1)
- 所有输出值为1


有指数爆炸风险，解决方案：  

$$
softmax(x) = e^{x_i - max(x)} / \Sigma_{j=1}^ne^{x_j - max(x)}
$$

### Tanh

$$
tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
$$

特征：
- 单调递增，值域(-1, 1)
- 关于原点中心对称  
- 类似sigmoid，在原点附近类似线性，两端饱和

缺点：
- 类似sigmoid，两端梯度较小，需要指数计算

### ReLU

$$
ReLU(x) = max(0, x)
$$

特征：
- 输入为正，梯度不会消失
- 复杂度低

缺点：
- 不是零中心
- 如果输入为负，梯度永远为0（死亡ReLU问题）

### Leaky ReLU

$$
LeakyReLU(x) = \begin{cases}
x, & x>0,\\
\alpha x, & x \le 0
\end{cases}
$$

其中$\alpha$为小正数（通常为0.01）

特征：
- 避免了死亡ReLU问题

缺点：
- 不是零中心

### ELU

$$
ELU(x) = \begin{cases}
x, & x>0,\\
\alpha (e^x - 1), & x \le 0
\end{cases}
$$

$\alpha$为正数，一般为1.0

特征：  
- 输出均值更接近零  

缺点：  
- 有指数运算

### Swish

$$
Swish(x) = x \cdot \sigma(\beta x)
$$

其中$\beta$为可学习参数。当$\beta$为1时，为SiLU。这是一个非单调平滑函数，处处可微，可以通过调整参数模拟不同形状。$\beta$趋于0时，Swish接近$x/2$；$\beta$趋于$\infty$时，接近ReLU函数。  

特征：  
- 平滑、处处可微，可模拟不同形状
- 实验证明，深层网络中表现由于ReLU

缺点： 
- 复杂度高，性能提升可能不能弥补增加的成本

### GeLU

$$
GeLU(x) = x \cdot \Phi(x)
$$

由于正态分布函数计算较为昂贵，会采用近似公式  

$$
GeLU(x) = 0.5x(1 + tanh(\sqrt{2 / \pi}(x + 0.044715x^3)))
$$

特征：
- 平滑，处处可微
- x为负会抑制，避免神经元死亡问题
- 表现优异，GPT、BERT均使用GeLU

### SwiGLU

$$
SwiGLU(x) = Swish_{\beta = 1}(Wx + b) \otimes V_x
$$

常和FFN组合使用。

特征：  

- 结合了GLU和Swish，更好地控制信息
- 表现优异

缺点：
- 复杂度太高
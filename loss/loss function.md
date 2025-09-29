## 损失函数公式总结

### MAE Loss（L1 Loss）

$$
L = (\Sigma_{i=1}^n|y_i - \hat{y_i}|) / n
$$

特征： 

- 对异常不敏感
- 零点处不可导，优化困难
- 不关注大误差样本

### MSE Loss（L2 Loss）

$$
L = (\Sigma_{i=1}^n (y_i - \hat{y_i})^2) / n
$$

特征：  

- 曲线光滑，易于求导，计算简单
- 重视大误差样本
- 对异常值敏感

### CE Loss（交叉熵）

$$
H(P, Q) = - \Sigma_{i=1}^N P(x_i)logQ(x_i)
$$

其中真实分布为P，预测分布为Q。

分类任务中采用独热编码时，可简化为  

$$
L = - (\Sigma_{i=1}^N \Sigma_{j=1}^C y_c log(\hat{y_c})) / N
$$

其中C为类别总数， $\hat{y_c}$是softmax后的概率。此时对某个样本，假设真实类别为k，预测概率为p，则有  

$$
L = -log(p_k)
$$

设原始分数为x，则有  

$$
L = - \Sigma_{i=1}^C y_i(x_i - log(\Sigma exp(x_j)))
$$

特征： 

- 有效处理多分类问题
- 对错误预测有较大惩罚  
- 对类别不平衡敏感
- 可能对噪声标签敏感

### BCE Loss

*有说法认为BCE是CE的特殊形式。从我的角度看，CE（某种意义上）只关注真实样本的对应情况，和BCE不是简单的包含关系*

$$
L = -(\Sigma_{i=1}^N[y_i log(\hat{y_i}) + (1 - y_i) log(1 - \hat{y_i})]) / N
$$

二分类问题的归一化方式从softmax替换为sigmoid，即  

$$
y_i = \sigma(x_i)
$$

特点：  
- 不要求类别互斥
- 适合二分类/多标签分类（对于二分类，softmax和sigmoid是数学上等价的）  
- 对类别不平衡敏感

### KL Divergence

真实分布P，预测分布Q

$$
D_{KL}(P || Q) = \Sigma_{i=1}^N P(x_i)[logP(x_i) - logQ(x_i)]
$$

特征：  

- 非对称
- 非负，P=Q时为0
- Q(x)接近0时数值不稳定

此外  

$$
KL(P||Q) = CE(P,Q)-H(P)
$$

### Focal Loss

Focal Loss从交叉熵的基础上引入调制因子。

$$
L = -(\Sigma_{i=1}^N \alpha_t(1 - p_t)^ \gamma log(p_t)) / N
$$

且
$$
p_t = \hat{p} \cdot y + (1 - \hat{p}) \cdot (1 - y)
$$

$$
\alpha_t = \alpha \cdot y + (1 - \alpha) \cdot (1 - y)
$$

对于易分类样本，乘以小于1的因子，损失贡献被压低；对于难分类样本，乘以接近1的因子，损失被保留。$\alpha$是类别平衡参数，增大时少数类损失增加，减小时多数类损失增加。

特征： 
- 用于二分类问题，解决类别不平衡问题
- 引入额外参数
- 训练初期可能不稳定
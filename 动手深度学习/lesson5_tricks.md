# 1 欠拟合和过拟合


```python
import math
import numpy as np
import torch
from torch import nn
```

    /opt/miniconda3/envs/pt12/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## 1.1 生成数据集
使用以下三阶多项式生成训练集和测试集
$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$


```python
max_degree = 20 # 多项式最大阶数
n_train, n_test = 100, 100 # 训练和测试集数量
true_w = np.zeros(max_degree) # 生成真实标签
true_w[0:4] = np.array = ([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!
    
labels = np.dot()
```

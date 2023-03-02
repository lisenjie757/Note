```python
import numpy as np 
import torch
import time
```

# 测试速度


```python
## numpy
a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("Numpy Version: "+str(1000*(toc-tic))+"ms")

## torch cpu
a_t = torch.from_numpy(a)
b_t = torch.from_numpy(b)

tic = time.time()
c = a_t @ b_t
toc = time.time()

print(c)
print("Torch CPU Version: "+str(1000*(toc-tic))+"ms")

## torch gpu
a_t = torch.from_numpy(a).cuda()
b_t = torch.from_numpy(b).cuda()

tic = time.time()
c = a_t @ b_t
toc = time.time()

print(c)
print("Torch GPU Version: "+str(1000*(toc-tic))+"ms")

## for loop
c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()

print(c)
print("For Loop : "+str(1000*(toc-tic))+"ms")
```

    249913.9129535616
    Numpy Version: 1.6908645629882812ms
    tensor(249913.9130, dtype=torch.float64)
    Torch CPU Version: 1.4238357543945312ms
    tensor(249913.9130, device='cuda:0', dtype=torch.float64)
    Torch GPU Version: 1274.2106914520264ms
    249913.91295356725
    For Loop : 286.07654571533203ms


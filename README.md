# 介绍

- 一些使用triton实现的高效算法。
- xhs上发过的所有的示例在**triton_kernels**文件夹下，包含10+高效算子。
- 基础triton示例见**Triton-Easy-Tutorials**文件夹.
- **mdy_triton**下是一些可以加速HF模型训练的工具，一行import即可加速。详细见**Triton-Easy-Tutorials**下的README。使用方法如下：
```python
git clone https://github.com/mdy666/mdy_triton.git
cd mdy_triton
pip install .
# go to your code dir
from mdy_triton.replace_kernel import *
```




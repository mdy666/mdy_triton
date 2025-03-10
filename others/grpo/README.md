# 使用方法
看好自己的trl版本，建议更新到最新。将replace_grpo_trainer.py移动到open-r1/src/open_r1/下，在grpo.py的最开头加入：
```python
# 第一行
from src.open_r1.replace_grpo_trainer import trigger


import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
```

# BUG
之前的grpo_loss可能有个bug，训练时dlosses不连续，算出的梯度可能是错误的，sorry，目前已解决
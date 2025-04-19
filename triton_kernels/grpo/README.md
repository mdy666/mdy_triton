
# Install 

```bash
pip install .
```
# Quick start

The TRL version must be 0.16. Add it to the first line of `grpo.py` in `open-r1`.

```python
from triton_grpo_loss import trigger
```

# Why save lots of memory

- the diffenence between torch code and titon code

 ![alt text](./imgs/triton_code.jpg)

# The Grad of Clamp OP

 ![alt text](./imgs/clamp.jpg)



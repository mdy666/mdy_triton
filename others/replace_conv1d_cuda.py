import importlib
import causal_conv1d_triton

moudle = importlib.import_module('mamba_ssm.ops.triton.ssd_combined')
moudle.causal_conv1d_cuda = causal_conv1d_triton
moudle.causal_conv1d_fn = causal_conv1d_triton.causal_conv1d_fn

trigger = None
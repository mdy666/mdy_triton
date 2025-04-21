import torch
import trl
assert trl.__version__.startswith("0.16"), "please pip install trl==0.16"
from trl.extras.profiling import profiling_decorator
from .core import fused_selective_log_softmax, triton_grpo_loss
from transformers.trainer import OptimizerNames, DistributedType

# trade-off between memory and speed
STEP = 1

@profiling_decorator
def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
    # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
    logp_list = []
    bs = input_ids[0]
    for idx in range(0, bs, STEP):
        logits = model(input_ids=input_ids[idx:idx+STEP], 
                       attention_mask=attention_mask[idx:idx+STEP] if attention_mask is not None else None, 
                       logits_to_keep=logits_to_keep + 1).logits
        logp = fused_selective_log_softmax(logits, 
                                           input_ids[idx:idx+STEP], 
                                           self.temperature, 
                                           mask=attention_mask[idx:idx+STEP] if attention_mask is not None else None)
        logp_list.append(logp)
    logps = torch.cat(logp_list, axis=0)
    return logps

@profiling_decorator
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    # Compute the per-token log probabilities for the model

    kwargs = {}
    # For LOMO optimizers you need to explicitly use the learnign rate
    if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        kwargs["learning_rate"] = self._get_learning_rate()

    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
        kwargs["scale_wrt_gas"] = False

    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
    

    ref_per_token_logps = inputs["ref_per_token_logps"]
    advantages = inputs["advantages"]
    old_per_token_logps = inputs["old_per_token_logps"]
    
    bs = input_ids.size(0)
    loss_list = []
    per_token_kl_list = []
    is_clipped_list = []
    total_tokens_this_mbs = completion_mask.sum()
    for idx in range(0, bs, STEP):
        logits = model(input_ids=input_ids[idx:idx+STEP], 
                       attention_mask=attention_mask[idx:idx+STEP], 
                       logits_to_keep=logits_to_keep + 1).logits
        per_token_loss, per_token_kl, is_clipped = triton_grpo_loss(logits, 
                                                                    old_per_token_logps[idx:idx+STEP] if old_per_token_logps is not None else None,
                                                                    ref_per_token_logps[idx:idx+STEP] if ref_per_token_logps is not None else None,
                                                                    completion_ids[idx:idx+STEP],
                                                                    advantages[idx:idx+STEP],
                                                                    completion_mask[idx:idx+STEP],
                                                                    self.temperature,
                                                                    self.beta,
                                                                    self.epsilon_low,
                                                                    self.epsilon_high,)
        loss = (per_token_loss * completion_mask[idx:idx+STEP]).sum() / total_tokens_this_mbs
        if torch.is_grad_enabled():
            self.accelerator.backward(loss, **kwargs)

        loss_list.append(loss.detach())
        per_token_kl_list.append(per_token_kl)
        is_clipped_list.append(is_clipped)

    loss = torch.stack(loss_list).sum()
    per_token_kl = torch.cat(per_token_kl_list, axis=0)
    is_clipped = torch.cat(is_clipped_list, axis=0)

    # Log the metrics
    mode = "eval" if self.control.should_evaluate else "train"

    if self.beta != 0.0:
        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
    self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
    return loss


def training_step(
    self, model, num_items_in_batch=None
) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (`nn.Module`):
            The model to train.
        inputs (`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument `labels`. Check your model's documentation for all accepted arguments.

    Return:
        `torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()

    inputs = self._prepare_inputs(inputs)

    with self.compute_loss_context_manager():
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

    del inputs
    torch.cuda.empty_cache()

    return loss.detach()

trl.GRPOTrainer._get_per_token_logps = _get_per_token_logps
trl.GRPOTrainer.compute_loss = compute_loss
trl.GRPOTrainer.training_step = training_step
trigger = None
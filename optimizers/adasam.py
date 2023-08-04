import torch
from torch import Tensor

from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable, _get_value, _default_to_fused_or_foreach, _differentiable_doc, _foreach_doc, _maximize_doc
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype
from typing import List, Optional

__all__ = ["Adasam"]



class Adasam(Optimizer):
    def __init__(self,params, lr=1e-2, rho=0.9, lr_decay=0, weight_decay=0, adaptive=False, initial_accumulator_value=0,
                 eps=1e-10, foreach=None, differentiable=False,):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "rho": rho,
            "adaptive": adaptive,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "foreach": foreach,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = torch.tensor(0.0)
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        for p in group["params"]:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []

            self._init_group(group, params_with_grad, grads, state_sums, state_steps)

            lr=group["lr"]
            weight_decay=group["weight_decay"]
            eps=group["eps"]
            rho=group["rho"]

            # for p in group["params"]:
            for (p, grad, state_sum, step_t) in zip(params_with_grad, grads, state_sums, state_steps):
                if p.grad is None:
                    continue
                
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)
                state_sum.addcmul_(grad, grad, value=1)
                std = state_sum.sqrt().add_(eps)
                # 正規化項なし
                # obj = grad.addcmul(std, grad, value=rho)
                # 正規化項あり
                obj = grad.addcmul(std, grad.div(grad_norm), value=rho)
                p.add_(obj, alpha=-lr)

        return loss
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
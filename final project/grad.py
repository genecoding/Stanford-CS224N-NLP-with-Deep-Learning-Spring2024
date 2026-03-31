# Implementation of PCGrad and CAGrad.
# Inspired by:
#     https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py
#     https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py
#     https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/PCGrad.py
#     https://github.com/Cranial-XIX/CAGrad/blob/main/mtrl/mtrl_files/cagrad.py
#     https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/CAGrad.py


import torch
import random
import numpy as np
from scipy.optimize import minimize


class PCGrad:
    """PCGrad + AMP + Sampling Strategy + LoRA/DoRA Support"""
    def __init__(self, optimizer, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler

    def step(self, losses, task_weights=None):
        """
        losses: list of losses [loss_sst, loss_para, loss_sts]
        task_weights: array of weights [weight_sst, weight_para, weight_sts],
                      calculated from sampling strategy
        """
        self.num_tasks = num_tasks = len(losses)
        if task_weights is None:
            task_weights = [1.0] * self.num_tasks

        # 1. Collect task gradients
        task_grads = self._get_task_grads(losses)
        # multiply weights after backward to avoid underflow for float16
        task_grads = [task_grads[i] * task_weights[i] for i in range(num_tasks)]
            
        # 2. Perform Gradient Surgery
        shuffled_indices = list(range(num_tasks))
        pc_grads = [g.clone() for g in task_grads]

        for i in range(num_tasks):
            random.shuffle(shuffled_indices)
            for j in shuffled_indices:
                if i == j: continue  # skip self projection
                g_i = pc_grads[i]
                g_j = task_grads[j]
                # check for conflict: dot product < 0
                dot_product = torch.dot(g_i, g_j)
                if dot_product < 0:
                    # project g_i onto the normal plane of g_j
                    pc_grads[i] -= (dot_product / (torch.norm(g_j)**2).clamp(min=1e-8)) * g_j

        # 3. Sum projected gradients and apply to model
        new_grad = torch.stack(pc_grads).sum(dim=0)
        self._set_and_step(new_grad)
        del task_grads, pc_grads, new_grad  # free memory
    
    def _get_grad(self):
        """Collect a gradient for every parameter, even if it's None (set to 0)."""
        grad = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:  # avoid getting grad from frozen layers
                    if p.grad is not None:
                        grad.append(p.grad.detach().clone().flatten())
                    else:
                        # if grad is None, for multi-head scenario
                        grad.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        return grad

    def _set_grad(self, new_grad):
        """Put the modified flat gradient back into parameter .grad fields."""
        start = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    numel = p.numel()
                    p.grad = new_grad[start:start + numel].view(p.shape)
                    start += numel

    def _get_task_grads(self, losses):
        task_grads = []
        for i, loss in enumerate(losses):
            self.optimizer.zero_grad()
            # is_last = i == self.num_tasks - 1
            if self.scaler:
                # scale and backward; use retain_graph for shared backbones, release at final task
                # self.scaler.scale(loss).backward(retain_graph=not is_last)
                # since we have different inputs for each task, retain_graph is not necessary
                self.scaler.scale(loss).backward()
                inv_scale = 1.0 / self.scaler.get_scale()
                # flatten all gradients into a single vector for this task
                grad = self._get_grad()
                # inv_scale: manually unscale list of grads because they're still scaled;
                #            we don't call scaler.unscale_(optimizer) here
                task_grads.append(torch.cat(grad) * inv_scale)
            else:
                # loss.backward(retain_graph=not is_last)
                loss.backward()
                grad = self._get_grad()
                task_grads.append(torch.cat(grad))
        return task_grads

    def _set_and_step(self, new_grad):
        if self.scaler:
            # since scaler.unscale_() wasn't called, scaler.step() will call it automatically;
            # to avoid scaler unscaling new_grad, we have to call unscale_ manually before _set_grad
            self.scaler.unscale_(self.optimizer)
            self._set_grad(new_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self._set_grad(new_grad)
            self.optimizer.step()


class CAGrad(PCGrad):
    """CAGrad + AMP + Sampling Strategy + LoRA/DoRA Support"""
    def __init__(self, optimizer, scaler=None, c=0.5):
        super().__init__(optimizer, scaler)
        self.cagrad_c = c

    def step(self, losses, task_weights=None):
        self.num_tasks = num_tasks = len(losses)
        if task_weights is None:
            task_weights = [1.0 / num_tasks] * num_tasks
            
        # 1. Collect task gradients
        task_grads = self._get_task_grads(losses)
        grads = torch.stack(task_grads)  # (num_tasks, num_params)
                
        # 2. Solve the Dual Objective
        GG = (grads @ grads.T).cpu()  # (num_tasks, num_tasks)
        scale = (torch.diag(GG) + 1e-4).sqrt().mean()
        GG = (GG / scale.pow(2)).numpy()
        Gg = GG.mean(axis=1)  # gi dot g0 (scaled), (num_tasks,)
        g0_norm = np.sqrt(Gg.mean() + 1e-4)  # norm of the average gradient
        
        # √Φ, radius of the ball constraint (0 = GD, higher = more conservative)
        c = g0_norm * self.cagrad_c
        
        # solve dual optimization for weights w
        def obj(w):
            return w.T @ Gg + c * np.sqrt(w.T @ GG @ w + 1e-4)  # gw⊤g0 + √Φ‖gw‖
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # sum == 1
        bnds = [(0, None) for _ in range(num_tasks)]
        res = minimize(obj, np.ones(num_tasks) / num_tasks, 
                       method='SLSQP', bounds=bnds, constraints=cons)
        w = res.x
        
        gw_norm = np.sqrt(w.T @ GG @ w + 1e-4)
        lmbda = c / (gw_norm + 1e-4)
        w = torch.from_numpy(w).to(grads.device).to(grads.dtype)  # align type with grads for AMP
        task_weights = torch.from_numpy(task_weights).to(grads.device).to(grads.dtype)
        # d* = g0 + λgw = (1/n + λwi) gi, we can see 1/n as task_weights 
        # final_weight = (1 / num_tasks + w * lmbda).view(-1, 1)
        final_weight = (task_weights + w * lmbda).view(-1, 1)
        new_grad = (final_weight * grads).sum(dim=0) / (1 + self.cagrad_c**2)
        
        # 3. Apply to model
        self._set_and_step(new_grad)
        del task_grads, grads, new_grad  # free memory


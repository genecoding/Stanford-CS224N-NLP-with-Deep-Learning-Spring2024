# Implementation of SMART
# Inspired by
#     https://github.com/archinetai/smart-pytorch/blob/main/smart_pytorch/smart_pytorch.py
#     https://github.com/namisan/mt-dnn/blob/master/mt_dnn/perturbation.py


import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def inf_norm(x):
    return torch.norm(x, p=float('inf'), dim=-1, keepdim=True)


def kl_loss(input, target):
    return F.kl_div(F.log_softmax(input, dim=-1),
                    F.softmax(target, dim=-1), 
                    reduction='batchmean')


def symmetrized_kl_loss(logits_a, logits_b):
    """
    Computes D_kl(P||Q) + D_kl(Q||P)
    Note: p, q are targets (probs), log_p, log_q are inputs (log_probs).
    """
    p = F.softmax(logits_a, dim=-1)
    log_p = F.log_softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)

    return F.kl_div(log_p, q, reduction='batchmean') \
           + F.kl_div(log_q, p, reduction='batchmean')                    


class SMARTLoss(nn.Module):
    """
    Smoothness-Inducing Adversarial Regularization
    Note: we detach original_output here.
    """
    def __init__(self, model, epsilon=1e-6, noise_var=1e-5, step_size=1e-3, iter_steps=1, lambda_s=1.0):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.noise_var = noise_var
        self.step_size = step_size
        self.iter_steps = iter_steps
        self.lambda_s = lambda_s

    def forward(self, input_ids, attention_mask, task, original_output):
        """
        task: 'classification' (Sentiment/Para) or 'regression' (STS)
        original_output: the output already computed in the standard forward pass
        """
        # 1. Get initial embeddings
        embeds = self.model.bert.embed(input_ids)
        
        # 2. Initialize noise
        noise = torch.randn_like(embeds, requires_grad=True) * self.noise_var
        
        # 3. Inner loop to find the worst-case perturbation (gradient ascent)
        for _ in range(self.iter_steps):
            perturbed_output = self.model.forward_with_embeds(embeds+noise, attention_mask, task)
            
            if task in {'sst', 'para'}:
                # classification, maximize KL Divergence
                dist = symmetrized_kl_loss(perturbed_output, original_output.detach())
            elif task == 'sts':
                # regression, maximize Squared Error
                dist = F.mse_loss(perturbed_output, original_output.detach())
            
            # get gradient of distance w.r.t noise
            grad, = torch.autograd.grad(dist, noise)
            noise = noise + self.step_size * grad
            noise_norm = inf_norm(noise)
            noise = noise / (noise_norm + self.epsilon)
            noise = noise.detach().requires_grad_()

        # 4. Final perturbed pass to get the smoothness loss
        final_output = self.model.forward_with_embeds(embeds+noise, attention_mask, task)
        
        if task in {'sst', 'para'}:
            loss = symmetrized_kl_loss(final_output, original_output.detach())
        elif task == 'sts':
            loss = F.mse_loss(final_output, original_output.detach())
            
        return self.lambda_s * loss


class BPPOptimization(nn.Module):
    """
    Bregman Proximal Point Optimization
    Note: we detach target_output here.
    """
    def __init__(self, model, mu=1.0, momentum=0.999):
        super().__init__()
        self.model = model
        self.mu = mu
        self.momentum = momentum
        
        # create target model to track the stable state
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, task, current_output):
        """Calculate Bregman divergence between current and target model."""
        if task in {'sst', 'para'}:
            # KL Divergence for classification (Sentiment/Para)
            if task == 'sst':
                target_output = self.target_model.predict_sentiment(input_ids, attention_mask)
            elif task == 'para':
                target_output = self.target_model.predict_paraphrase(input_ids, attention_mask)            
            prox_loss = symmetrized_kl_loss(current_output, target_output.detach())
        elif task == 'sts':
            # MSE for regression (STS)
            target_output = self.target_model(input_ids, attention_mask)
            prox_loss = F.mse_loss(current_output, target_output.detach())
            
        return self.mu * prox_loss
    
    @torch.no_grad()
    def update_target_model(self):
        """Update the target model using Exponential Moving Average."""
        for t_param, m_param in zip(self.target_model.parameters(), self.model.parameters()):
            if m_param.requires_grad:  # only update parameters that are actually trained
                t_param.data.mul_(self.momentum).add_(m_param.data, alpha=1.0 - self.momentum)


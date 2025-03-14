import torch
import numpy as np
from torch import nn

def acf_diff(x): return torch.sqrt(torch.pow(x, 2).sum(0))
def cc_diff(x): return torch.abs(x).sum(0)
def cov_diff(x): return torch.abs(x).mean()

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x, seed=None):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo
        self.seed = seed

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)        
        return {'abs_loss': self.reg * self.loss_componentwise['abs_loss'].mean(), 'rel_loss': self.reg * self.loss_componentwise['rel_loss'].mean()}


    def compute(self, x_fake):
        raise NotImplementedError()
    
    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class MeanLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(MeanLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.mean = x_real.mean(1)

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.mean(1) - self.mean)


class StdLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(StdLoss, self).__init__(norm_foo=torch.abs, **kwargs)
        self.std_real = x_real.std(1)

    def compute(self, x_fake, **kwargs):
        return self.norm_foo(x_fake.std(1) - self.std_real)


class VARLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(VARLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='var')

    def compute(self, x_fake):
        abs_loss_list = []
        rel_loss_list = []
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='var')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                eps = 1e-8  # 분모 0 방지용
                rel_metric = abs_metric / (torch.abs(self.var[i][t].to(x_fake.device)) + eps)

                abs_loss_list.append(abs_metric)
                rel_loss_list.append(rel_metric)

        abs_loss_componentwise = torch.stack(abs_loss_list)
        rel_loss_componentwise = torch.stack(rel_loss_list)

        return {
            'abs_loss': abs_loss_componentwise,
            'rel_loss': rel_loss_componentwise
        }

class ESLoss(Loss):
    def __init__(self, x_real, alpha=0.05, **kwargs):
        name = kwargs.pop('name')
        super(ESLoss, self).__init__(name=name)
        self.alpha = alpha
        self.var = tail_metric(x=x_real, alpha=self.alpha, statistic='es')

    def compute(self, x_fake):
        abs_loss_list = []
        rel_loss_list = []
        
        var_fake = tail_metric(x=x_fake, alpha=self.alpha, statistic='es')
        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                abs_metric = torch.abs(var_fake[i][t] - self.var[i][t].to(x_fake.device))
                eps = 1e-8  # 분모 0 방지용
                rel_metric = abs_metric / (torch.abs(self.var[i][t].to(x_fake.device)) + eps)
                
                abs_loss_list.append(abs_metric)
                rel_loss_list.append(rel_metric)
        abs_loss_componentwise = torch.stack(abs_loss_list)
        rel_loss_componentwise = torch.stack(rel_loss_list)

        return {
            'abs_loss': abs_loss_componentwise,
            'rel_loss': rel_loss_componentwise
        }

def tail_metric(x, alpha, statistic):
    res = list()
    for i in range(x.shape[2]):
        tmp_res = list()
        # Exclude the initial point
        for t in range(x.shape[1]):
            x_ti = x[:, t, i]
            sorted_arr, _ = torch.sort(x_ti)
            var_alpha_index = int(alpha * len(sorted_arr))
            var_alpha = sorted_arr[var_alpha_index]
            if statistic == "es":
                es_values = sorted_arr[:var_alpha_index + 1]
                es_alpha = es_values.mean()
                tmp_res.append(es_alpha)
            else:
                tmp_res.append(var_alpha)
        res.append(tmp_res)
    return res

class MaxDrawbackLoss(Loss):
    def __init__(self, x_real, **kwargs):
        name = kwargs.pop('name')
        super(MaxDrawbackLoss, self).__init__(name=name)
        self.max_drawback = compute_max_drawdown(pnls=x_real)

    def compute(self, x_fake):
        loss = list()
        max_drawback_fake = compute_max_drawdown(pnls=x_fake)
        loss = torch.abs(self.max_drawback - max_drawback_fake)
        return loss


def compute_max_drawdown(pnls: torch.Tensor):
    """
    Compute the maximum drawdown for a batch of PnL trajectories.

    :param pnls: Tensor of shape [N, T], where N is the number of batches and T is the number of time steps.
                 This tensor represents the cumulative PnL for each batch.
    :return: Tensor of shape [N] representing the maximum drawdown for each batch.
    """
    # Compute the running maximum PnL at each time step for each batch
    running_max = torch.cummax(pnls, dim=1)[0]  # Shape [N, T], the maximum PnL at each time step

    # Compute the drawdown at each time step: (Peak PnL - Current PnL) / Peak PnL
    drawdowns = (running_max - pnls)  # Shape [N, T]

    # Compute the maximum drawdown for each batch
    max_drawdown = torch.max(drawdowns, dim=1)[0]  # Shape [N], maximum drawdown for each batch

    return max_drawdown

class CumulativePnLLoss(Loss):
    def __init__(self, x_real, **kwargs):
        name = kwargs.pop('name')
        super(CumulativePnLLoss, self).__init__(name=name)
        self.cum_pnl = x_real[:,-1]

    def compute(self, x_fake):
        cum_pnl_fake = x_fake[:,-1]
        loss = torch.abs(self.cum_pnl - cum_pnl_fake)
        return loss
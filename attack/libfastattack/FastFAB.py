from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math

import torch
from torch.cuda.amp import GradScaler, autocast

from collections import abc as container_abcs

from torchattacks.attack import Attack

class FastFAB(Attack):

    def __init__(self, model, eps, alpha_max=0.1, eta=1.05, beta=0.9, seed=0):
        super().__init__("FastFAB", model)
        self.eps = eps
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.seed = seed
        self._supported_mode = ['default']
        self.scaler = GradScaler()

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)

        return adv_images

    def _get_predicted_label(self, x):
        with torch.no_grad():
            with autocast(): outputs = self.model(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            with autocast(): y = self.model(im)

            g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
            grad_mask = torch.zeros_like(y)
            for counter in range(y.shape[-1]):
                zero_gradients(im)
                grad_mask[:, counter] = 1.0
                y.backward(grad_mask, retain_graph=True)
                grad_mask[:, counter] = 0.0
                g2[counter] = im.grad.data

            g2 = torch.transpose(g2, 0, 1).detach()
            y2 = y.detach()
            df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
            dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
    
            df = df.float()
            dg = dg.float()

            df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def attack_single_run(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        # self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])

        if use_rand_start:
            t = 2 * torch.rand(x1.shape).to(self.device) - 1
            x1 = im2 + (torch.min(res2,
                                  self.eps * torch.ones(res2.shape)
                                  .to(self.device)
                                  ).reshape([-1, *[1]*self.ndims])
                        ) * t / (t.reshape([t.shape[0], -1]).abs()
                                 .max(dim=1, keepdim=True)[0]
                                 .reshape([-1, *[1]*self.ndims])) * .5

            x1 = x1.clamp(0.0, 1.0)

        counter_iter = 0
        while counter_iter < 10:
            with torch.no_grad():
                df, dg = self.get_diff_logits_grads_batch(x1, la2)
                dist1 = df.abs() / (1e-12 +
                                    dg.abs()
                                    .view(dg.shape[0], dg.shape[1], -1)
                                    .sum(dim=-1))

                ind = dist1.min(dim=1)[1]
                dg2 = dg[u1, ind]
                b = (- df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1)
                                     .sum(dim=-1))
                w = dg2.reshape([bs, -1])

                d3 = projection_linf(
                    torch.cat((x1.reshape([bs, -1]), x0), 0),
                    torch.cat((w, w), 0),
                    torch.cat((b, b), 0))
                d1 = torch.reshape(d3[:bs], x1.shape)
                d2 = torch.reshape(d3[-bs:], x1.shape)
                a0 = d3.abs().max(dim=1, keepdim=True)[0].view(-1, *[1]*self.ndims)
                a0 = torch.max(a0, 1e-8 * torch.ones(
                    a0.shape).to(self.device))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                alpha = torch.min(torch.max(a1 / (a1 + a2),
                                            torch.zeros(a1.shape)
                                            .to(self.device)),
                                  self.alpha_max * torch.ones(a1.shape)
                                  .to(self.device))
                x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                      (im2 + d2 * self.eta) * alpha).clamp(0.0, 1.0)

                is_adv = self._get_predicted_label(x1) != la2

                if is_adv.sum() > 0:
                    ind_adv = is_adv.nonzero().squeeze()
                    ind_adv = self.check_shape(ind_adv)
                    t = (x1[ind_adv] - im2[ind_adv]).reshape(
                        [ind_adv.shape[0], -1]).abs().max(dim=1)[0]

                    adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv]).\
                        float().reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                        * (t >= res2[ind_adv]).float().reshape(
                        [-1, *[1]*self.ndims])
                    res2[ind_adv] = t * (t < res2[ind_adv]).float()\
                        + res2[ind_adv] * (t >= res2[ind_adv]).float()
                    x1[ind_adv] = im2[ind_adv] + (
                        x1[ind_adv] - im2[ind_adv]) * self.beta

                counter_iter += 1

        ind_succ = res2 < 1e10

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c


    def perturb(self, x, y):
        adv = x.clone()

        with torch.no_grad():
            with autocast(): acc = self.model(x).max(1)[1] == y

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=True)

                with autocast(): acc_curr = self.model(adv_curr).max(1)[1] == y_to_fool
                res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).max(1)[0]
                acc_curr = torch.max(acc_curr, res > self.eps)

                ind_curr = (acc_curr == 0).nonzero().squeeze()
                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone().half()
        return adv


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane.clone()

    sign = 2 * ((w * t).sum(1) - b >= 0) - 1
    w.mul_(sign.unsqueeze(1))
    b.mul_(sign)

    a = (w < 0).float()
    d = (a - t) * (w != 0).float()

    p = a - t * (2 * a - 1)
    indp = torch.argsort(p, dim=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)

    indp2 = indp.flip((1,))
    ws = w.gather(1, indp2)
    bs2 = - ws * d.gather(1, indp2)

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    b2 = sb[:, -1] - s[:, -1] * p.gather(1, indp[:, 0:1]).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)

        counter2 = counter4.long().unsqueeze(1)
        indcurr = indp_.gather(1, indp_.size(1) - 1 - counter2)
        b2 = (sb_.gather(1, counter2) - s_.gather(1, counter2) * p_.gather(1, indcurr)).squeeze(1)
        c = b_ - b2 > 0

        lb = torch.where(c, counter4, lb)
        ub = torch.where(c, ub, counter4)

    lb = lb.long()

    if c_l.any():
        lmbd_opt = torch.clamp_min((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), min=0).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = torch.clamp_min((b[c2] - sb[c2, lb]) / (-s[c2, lb]), min=0).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * a[c2] + torch.max(-lmbd_opt, d[c2]) * (1 - a[c2])

    return d * (w != 0).float()


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)
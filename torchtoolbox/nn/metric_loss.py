__all__ = ['L2Softmax', 'ArcLoss', 'AMSoftmax', 'CircleLossFC']

from torch import nn
from torch.nn import functional as F
import torch
import math


class NormLinear(nn.Module):
    def __init__(self, in_features, classes, weight_norm=False, feature_norm=False):
        super(NormLinear, self).__init__()
        self.weight_norm = weight_norm
        self.feature_norm = feature_norm

        self.classes = classes
        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(classes, in_features))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x):
        weight = F.normalize(self.weight, 2, dim=-1) if self.weight_norm else self.weight
        if self.feature_norm:
            x = F.normalize(x, 2, dim=-1)

        return F.linear(x, weight)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.classes)


class L2Softmax(nn.Module):
    r"""L2Softmax from
    `"L2-constrained Softmax Loss for Discriminative Face Verification"
    <https://arxiv.org/abs/1703.09507>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    alpha: float.
        The scaling parameter, a hypersphere with small alpha
        will limit surface area for embedding features.
    p: float, default is 0.9.
        The expected average softmax probability for correctly
        classifying a feature.
    from_normx: bool, default is False.
         Whether input has already been normalized.

    Outputs:
        - **loss**: loss tensor with shape (1,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, embedding_size, classes, alpha=64, p=0.9):
        super(L2Softmax, self).__init__()
        alpha_low = math.log(p * (classes - 2) / (1 - p))
        assert alpha > alpha_low, "For given probability of p={}, alpha should higher than {}.".format(p, alpha_low)
        self.alpha = alpha
        self.linear = NormLinear(embedding_size, classes, True, True)

    def forward(self, x, target):
        x = self.linear(x)
        x = x * self.alpha
        return x


class ArcLoss(nn.Module):
    r"""ArcLoss from
    `"ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
    <https://arxiv.org/abs/1801.07698>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float.
        Margin parameter for loss.
    s: int.
        Scale parameter for loss.

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, embedding_size, classes, m=0.5, s=64, easy_margin=True):
        super(ArcLoss, self).__init__()
        assert s > 0.
        assert 0 <= m <= (math.pi / 2)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = math.sin(math.pi - m) * m
        self.threshold = math.cos(math.pi - m)
        self.classes = classes
        self.easy_margin = easy_margin
        self.linear = NormLinear(embedding_size, classes, True, True)

    @torch.no_grad()
    def _get_body(self, x, target):
        cos_t = torch.gather(x, 1, target.unsqueeze(1))  # cos(theta_yi)
        if self.easy_margin:
            # cond = torch.relu(cos_t)
            cond = torch.clamp_min(cos_t, min=0.)
        else:
            cond_v = cos_t - self.threshold
            # cond = torch.relu(cond_v)
            cond = torch.clamp_min(cond_v, min=0.)
        cond = cond.bool()
        new_zy = torch.cos(torch.acos(cos_t) + self.m).type(cos_t.dtype)  # cos(theta_yi + m), use `.type()` to fix FP16
        if self.easy_margin:
            zy_keep = cos_t
        else:
            zy_keep = cos_t - self.mm  # (cos(theta_yi) - sin(pi - m)*m)
        new_zy = torch.where(cond, new_zy, zy_keep)
        diff = new_zy - cos_t  # cos(theta_yi + m) - cos(theta_yi)
        gt_one_hot = F.one_hot(target, num_classes=self.classes)
        body = gt_one_hot * diff
        return body

    def forward(self, x, target):
        x = self.linear(x)
        body = self._get_body(x, target)
        x = x + body
        x = x * self.s
        return x


class AMSoftmax(nn.Module):
    r"""CosLoss from
       `"CosFace: Large Margin Cosine Loss for Deep Face Recognition"
       <https://arxiv.org/abs/1801.09414>`_ paper.

       It is also AM-Softmax from
       `"Additive Margin Softmax for Face Verification"
       <https://arxiv.org/abs/1801.05599>`_ paper.

    Parameters
    ----------
    classes: int.
        Number of classes.
    m: float, default 0.4
        Margin parameter for loss.
    s: int, default 64
        Scale parameter for loss.


    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """
    def __init__(self, embedding_size, classes, m=0.35, s=64):
        super(AMSoftmax, self).__init__()
        assert m > 0 and s > 0
        self.classes = classes
        self.scale = s
        self.margin = m
        self.linear = NormLinear(embedding_size, classes, True, True)

    def forward(self, x, target):
        x = self.linear(x)
        sparse_target = F.one_hot(target, num_classes=self.classes)
        x = x - sparse_target * self.margin
        x = x * self.scale
        return x


class CircleLossFC(nn.Module):
    def __init__(self, embedding_size, classes, m=0.25, gamma=256):
        super(CircleLossFC, self).__init__()
        self.m = m
        self.gamma = gamma
        self.dp = 1 - m
        self.dn = m
        self.classes = classes
        self.linear = NormLinear(embedding_size, classes, True, True)

    @torch.no_grad()
    def get_param(self, x):
        ap = torch.relu(1 + self.m - x.detach())
        an = torch.relu(x.detach() + self.m)
        return ap, an

    def forward(self, x, target):
        x = self.linear(x)
        ap, an = self.get_param(x)
        gt_one_hot = F.one_hot(target, num_classes=self.classes)
        x = gt_one_hot * ap * (x - self.dp) + (1 - gt_one_hot) * an * (x - self.dn)
        x = x * self.gamma
        return x

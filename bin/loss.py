# The loss function is taken from PyCox


import torch
from torch import Tensor


def events_from_censored(is_censored):
    """`is_censored` flag (1 = alive/censored, 0 = death) to the
    event indicator (1 = death) expected by the Cox losses and the C-index."""
    return 1 - is_censored

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    if events.sum() == 0:
        return log_h.sum() * 0.0
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.

    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.

    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor, device="cuda") -> Tensor:
        return cox_ph_loss(log_h, durations, events)


class NegativeLogLikelihood(torch.nn.Module):
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
        # self.L2_reg = config['l2_reg']
        # self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, device="cuda"):

        """
        y = np.array([[1,6,2,2,4,5,1]])
        mask = np.ones((7, 7))
        mask[(y.T - y) > 0] = 0
        """
        
        # print("risk_pred", risk_pred[:5], risk_pred[:5].shape)
        y = torch.reshape(y, risk_pred.shape)
        e = torch.reshape(e, risk_pred.shape)
        if e.sum() == 0:
            return risk_pred.sum() * 0.0
        # print("y", y[:5], y[:5].shape)
        # print("e", e[:5])
        # print("risk_pred", risk_pred)
        mask = torch.ones(y.shape[0], y.shape[0]).to(device)
        # print("y", y)
        # print("yT", y.T)
        # print(torch.exp(risk_pred))
        mask[(y.T - y) > 0] = 0
        # print(mask)
        log_loss = torch.exp(risk_pred) * mask
        # print("log_loss", log_loss)
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
        # l2_loss = self.reg(model)
        # return neg_log_loss + l2_loss
        return neg_log_loss#  + l2_loss


def pairwise_rank_loss(log_h, durations, events):
    """Logistic loss over comparable pairs (i died before j's time): pushes risk_i above risk_j."""
    log_h, t, e = log_h.view(-1), durations.view(-1), events.view(-1)
    comparable = (t.unsqueeze(1) < t.unsqueeze(0)) & (e.unsqueeze(1) > 0)
    if not comparable.any():
        return log_h.sum() * 0.0
    return torch.nn.functional.softplus(-(log_h.unsqueeze(1) - log_h.unsqueeze(0)))[comparable].mean()


class CoxRankLoss(torch.nn.Module):
    def __init__(self, rank_weight: float = 0.0):
        super().__init__()
        self.cox, self.rank_weight = CoxPHLoss(), rank_weight

    def forward(self, log_h, durations, events, device="cuda"):
        loss = self.cox(log_h, durations, events)
        if self.rank_weight:
            loss = loss + self.rank_weight * pairwise_rank_loss(log_h, durations, events)
        return loss

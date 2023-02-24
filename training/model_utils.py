from __future__ import print_function
import torch
from collections import defaultdict


ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
ALPHABET_DICT = defaultdict(lambda: 0)
REVERSE_ALPHABET_DICT = {}
for i, letter in enumerate(ALPHABET):
    ALPHABET_DICT[i] = letter
    REVERSE_ALPHABET_DICT[letter] = i
ALPHABET_DICT[0] = "X"
REVERSE_ALPHABET_DICT['X'] = 0

class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()


def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model,
        2,
        4000,
        torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9),
        step,
    )


def pna_aggregate(features, mask):
    """PNA aggregation of features shaped `(B, L, N, K)` from neighbors to nodes"""

    d = mask.unsqueeze(-1).sum(-2)
    # denom = ...
    # S_pos = torch.log(d + 1) / denom
    # S_neg = 1 / S_pos
    # print(f'{features.shape=}, {d.shape=}')
    # f_mean = features.sum(2) / d.unsqueeze(-1)
    # print(d)
    f_mean = features.sum(2) / (d + 1e-6)
    feat = features.clone()
    d_mask = (d.squeeze(-1) == 0)
    feat[d_mask] = - float('inf')
    f_max = feat.max(2)[0]
    feat[d_mask] = float('inf')
    f_min = features.min(2)[0]
    f = torch.cat([f_mean, f_max, f_min], -1)
    f[d_mask] = 0
    return f



def loss_nll(S, log_probs, mask, X, X_pred, ignore_unknown):
    """Negative log probabilities"""
    if log_probs is None:
        true_false = 0
        pp = 0
    else:
        max_prob, S_argmaxed = torch.max(torch.softmax(log_probs, -1), -1)  # [B, L]
        pp = torch.exp(- torch.log(max_prob).mean(-1)).sum()
        if ignore_unknown:
            S_argmaxed += 1
        true_false = (S == S_argmaxed).float()
    if X_pred is None:
        rmsd = 0
    else:
        sqd = (X[:, :, 2, :] - X_pred[:, :, 2, :]) ** 2
        mean_sqd = (sqd * mask.unsqueeze(-1)).sum(-1).sum(-1) / mask.sum(-1)
        rmsd = torch.sqrt(mean_sqd).sum()
    return true_false, rmsd, pp


def loss_smoothed(S, logits, mask, no_smoothing, ignore_unknown, weight=0.1):
    """Negative log probabilities"""
    if logits is None:
        return torch.tensor(0)
    S_onehot = torch.nn.functional.one_hot(S, 21).float()
    if not no_smoothing:
        S_onehot = S_onehot + weight / float(S_onehot.size(-1))
        S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    if ignore_unknown:
        mask_S = mask * (S != 0)
        S_onehot = S_onehot[:, :, 1:]
    else:
        mask_S = mask

    loss = torch.nn.CrossEntropyLoss(reduction="none")(
        logits.transpose(-1, -2), S_onehot.transpose(-1, -2)
    )
    loss = loss[(mask_S).bool()].sum() / mask_S.sum()
    return loss


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


def context_to_rbf(context, min_rbf, max_rbf, nb_rbf):
    device = context.device
    mu = torch.linspace(min_rbf, max_rbf, nb_rbf, device=device)
    mu = mu.view([1, -1])
    sigma = (max_rbf - min_rbf) / nb_rbf
    context_expand = torch.unsqueeze(context, -1)
    RBF = torch.exp(-(((context_expand - mu) / sigma) ** 2))
    return RBF


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


def get_std_opt(parameters, d_model, step, lr=None):
    if lr is None:
        return NoamOpt(
            d_model,
            2,
            4000,
            torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9),
            step,
        )
    else:
        return torch.optim.Adam(parameters, lr=lr)
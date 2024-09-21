import torch
import torch.nn as nn


class InnerProductWithWeightsAffinity(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InnerProductWithWeightsAffinity, self).__init__()
        self.d = output_dim
        self.A = torch.nn.Linear(input_dim, output_dim)

    def _forward(self, X, Y, weights, use_global):
        assert X.shape[1] == Y.shape[1] == self.d, (X.shape[1], Y.shape[1], self.d)
        if use_global:
            coefficients = torch.tanh(self.A(weights))
            res = torch.matmul(X * coefficients, Y.transpose(0, 1))
        else:
            res = torch.matmul(X, Y.transpose(0, 1))
        res = torch.nn.functional.softplus(res) - 0.5
        return res

    def forward(self, Xs, Ys, Ws, use_global=True):
        return [self._forward(X, Y, W, use_global) for X, Y, W in zip(Xs, Ys, Ws)]
    
# * Gaussian kernel affinity for synthetic dataset edge affinity
class GaussianWithWeightAffinity(nn.Module):
    """
    Affinity Layer to compute the affinity matrix via gaussian kernel from feature space.
    Me = exp(- L2(X, Y) / sigma)
    Parameter: scale of weight d, gaussian kernel sigma
    Input: edgewise (pairwise) feature Xs, Ys with shape (b x 1 x num_edges)
    Output: edgewise affinity matrix Me
    """

    def __init__(self, input_dim, d, sigma):
        super(GaussianWithWeightAffinity, self).__init__()
        self.d = d
        self.sigma = sigma
        self.A = torch.nn.Linear(input_dim, self.d)

    def _forward(self, X, Y, weights, use_global, ae=1):
        assert X.shape[0] == Y.shape[0] == self.d
        
        if use_global:
            coefficients = torch.tanh(self.A(weights))
            X = X * coefficients

        X = X.unsqueeze(-1).expand(*X.shape, Y.shape[1])
        Y = Y.unsqueeze(0).expand(*Y.shape[:1], X.shape[1], Y.shape[1])
        dist = torch.sum(torch.pow(X - Y, 2), dim=0)
        dist[torch.isnan(dist)] = float("Inf")
        Me = torch.exp(- dist / self.sigma) * ae
        return Me
    
    def forward(self, Xs, Ys, Ws, use_global=True):
        return [self._forward(X.T, Y.T, W, use_global) for X, Y, W in zip(Xs, Ys, Ws)]
import torch
import math

class Laplace(torch.nn.Module):
    def __init__(self, s=None):
        super().__init__()
        self.s = s

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y):
        if self.s is None:
            XX = X.matmul(X.t())
            XY = X.matmul(Y.t())
            YY = Y.matmul(Y.t())

            dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)
            s = self.median(dnorm2.detach()) / torch.tensor(math.log(X.size(0) + 1))
        else:
            s = self.s

        gamma = 1.0 / s
        K_XY = (-gamma * torch.cdist(X, Y)).exp()

        return K_XY

class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        return X.matmul(Y.t())

class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def median(self, tensor):
        tensor = tensor.flatten().sort()[0]
        length = tensor.shape[0]

        if length % 2 == 0:
            szh = length // 2
            kth = [szh - 1, szh]
        else:
            kth = [(length - 1) // 2]
        return tensor[kth].mean()

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            sigma = self.median(dnorm2.detach()) / (2 * torch.tensor(math.log(X.size(0) + 1)))
        else:
            sigma = self.sigma ** 2

        gamma = 1.0 / (2 * sigma)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY

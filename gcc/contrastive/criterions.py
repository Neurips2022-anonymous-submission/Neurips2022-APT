import torch
from torch import nn
import numpy as np


class NCESoftmaxLoss(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class NCESoftmaxLoss_var(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_var, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, graph_idx, device):
        unique_graph = list(set(graph_idx))
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)

        flag = True
        for index in unique_graph:
            sum = 0
            count = 0
            for i in range(len(graph_idx)):
                if index == graph_idx[i]:
                    sum = sum + loss[index]
                    count = count + 1
            if flag is True:
                loss_sum = sum / count
                loss_sum = loss_sum.reshape(-1)
            else:
                tmp = sum / count
                loss_sum = torch.cat((loss_sum, tmp.reshape(-1)), dim=0)
            flag = False

        flag = True if len(unique_graph) == 1 else False
        # print("sum")
        # print(loss_sum)

        return torch.var(loss_sum), flag


class NCESoftmaxLoss_reduce(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_reduce, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        return loss


class NCESoftmaxLoss_sam(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_sam, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, pp):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        p = np.quantile(loss.cpu().detach().numpy(), pp, interpolation='lower')
        count = 0
        loss_t = 0
        # print("sam-uncer")
        # print(loss)
        for i in range(loss.cpu().detach().numpy().shape[0]):
            if loss.cpu().detach().numpy()[i] > p:
                count = count + 1
                if loss_t == 0:
                    loss_t = loss[i]
                else:
                    loss_t = loss_t + loss[i]

        return loss_t / count


class NCESoftmaxLoss_sam_threshold(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_sam_threshold, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, threshold):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        # p = np.quantile(loss.cpu().detach().numpy(), 0.25, interpolation='lower')
        count = 0
        loss_t = 0
        # print("sam-uncer")
        # print(loss)
        for i in range(loss.cpu().detach().numpy().shape[0]):
            if loss.cpu().detach().numpy()[i] > threshold:
                count = count + 1
                if loss_t == 0:
                    loss_t = loss[i]
                else:
                    loss_t = loss_t + loss[i]
        if count == 0:
            count = 1
            loss_t = loss[0]
        return loss_t / count


class NCESoftmaxLoss_sam_threshold_new(nn.Module):
    def __init__(self):
        super(NCESoftmaxLoss_sam_threshold_new, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x, threshold):
        bsz = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([bsz]).cuda().long()
        loss = self.criterion(x, label)
        count = 0
        loss_t = 0
        # print("sam-uncer")
        # print(loss)
        for i in range(loss.cpu().detach().numpy().shape[0]):
            if loss.cpu().detach().numpy()[i] > threshold:
                count = count + 1
                if loss_t == 0:
                    loss_t = loss[i]
                else:
                    loss_t = loss_t + loss[i]
        if count == 0:
            count = 1
            loss_t = loss[0]
        return loss_t / count, count


class NCESoftmaxLossNS(nn.Module):
    """Softmax cross-entropy loss (a.k.a., info-NCE loss in CPC paper)"""

    def __init__(self):
        super(NCESoftmaxLossNS, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        bsz = x.shape[0]
        x = x.squeeze()
        # positives on the diagonal
        label = torch.arange(bsz).cuda().long()
        loss = self.criterion(x, label)
        return loss

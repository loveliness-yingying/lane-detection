import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.Sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, predict, ground):
        predict = predict[:,0,:,:]
        ground = ground[:,0,:,:]
        label1 = ground[ground == 1.0]
        label2 = ground[ground == 0.0]
        infer1 = predict[ground==1.0]
        infer1 = self.Sigmoid(infer1)
        infer2 = predict[ground==0.0]
        infer2 = self.Sigmoid(infer2)
        return self.loss(infer1,label1) + 0.5*self.loss(infer2, label2)




class SoftmaxLoss(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLoss, self).__init__()
        self.nll = nn.NLLLoss(ignore_index=255)

    def forward(self,predict,ground):
        predict = predict[2]
        predict = predict[:,0:5,:,:]
        ground = ground[:, 1, :, :]
        ground = ground.long()
        log_score = F.log_softmax(predict, dim=1)
        loss = self.nll(log_score,ground)
        return loss


class L2Loss(torch.nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.loss= torch.nn.MSELoss()

    def forward(self,predict,ground):
        predict = predict[2]
        predict = predict[:,5,:,:]
        background = ground[:, 0, :, :]
        ground = ground[:,2,:,:]
        label = ground[background==1.0]
        infer = predict[background==1.0]
        loss = self.loss(infer, label)
        return loss


class sadLoss(torch.nn.Module):
    def __init__(self):
        super(sadLoss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction='mean')

    def forward(self,cls_out):
        x1 = cls_out[0]
        x2 = cls_out[1]
        x1 = x1.pow(2).sum(dim=1)
        x1_feature = x1.view(x1.shape[0], -1, x1.shape[1] * x1.shape[2])
        x1_softmax = F.softmax(x1_feature, dim=-1)
        x2 = x2.pow(2).sum(dim=1)
        x2_feature = x2.view(x2.shape[0], -1, x2.shape[1] * x2.shape[2])
        x2_softmax = F.softmax(x2_feature, dim=-1)
        loss = self.loss(x1_softmax,x2_softmax)
        return loss
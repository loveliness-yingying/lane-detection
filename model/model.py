import torch
from model.backbone import resnet
import torch.nn.functional as F


class non_bottleneck_1d(torch.nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = torch.nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = torch.nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = torch.nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = torch.nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = torch.nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = torch.nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = torch.nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class UpsamplerBlock2(torch.nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(ninput, noutput, 3, stride=(2, 1), padding=1, output_padding=(1,0), bias=True)
        self.bn = torch.nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class UpsamplerBlock1(torch.nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = torch.nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)



class parsingNet(torch.nn.Module):
    def __init__(self,  pretrained=True, backbone='50', cls_dim=(37, 10, 4), use_aux=False):
        super(parsingNet, self).__init__()

        self.model = resnet(backbone, pretrained=pretrained)
        self.upsample1 = UpsamplerBlock1(512,64)
        self.upsample2 = UpsamplerBlock2(64,16)
        self.nonbottle1 = non_bottleneck_1d(64,0,1)
        self.nonbottle2 = non_bottleneck_1d(16,0,1)

        self.output_conv = torch.nn.Conv2d(16, 6, 1, stride=1, padding=0) #记得修改


    def decoder1(self,x4):
        feature = self.upsample1.forward(x4)
        feature = self.nonbottle1.forward(feature)
        feature = self.nonbottle1.forward(feature)
        return feature


    def decoder2(self, x5):
        feature = self.upsample2.forward(x5)
        feature = self.nonbottle2.forward(feature)
        feature = self.nonbottle2.forward(feature)
        return feature



    def forward(self, x):
        x2, x3, x4 = self.model(x)
        x5 = self.decoder1(x4)
        x6 = self.decoder2(x5)
        output = self.output_conv(x6)
        return x3, x5, output


import torch
from models.layers.activation_factory import ActivationFactory


class Conv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ks=(3,3,3), stride=1, padding=1,dilation=1, groups=1, bh=False, act=False, dropout_prob=0.0):
        super(Conv3d, self).__init__()

        self.bh = bh
        self.act = act
        self.drop = False

        bias = True
        if self.bh:
            self.bh = torch.nn.BatchNorm3d(out_channels)
            bias = False

        self.conv3d = torch.nn.Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=ks, stride=stride,
                                       padding=padding, dilation=dilation,
                                       groups=groups, bias=bias)
        if dropout_prob > 0.0:
            self.drop = True
            self.dropout = torch.nn.Dropout(p=dropout_prob)

        if act:
            self.act = ActivationFactory.factory(act_name=self.act)

    def forward(self, x):
        model = self.conv3d(x)
        if self.bh:
            model = self.bh(model)
        if self.act:
            model = self.act(model)
        if self.drop:
            model = self.dropout(model)

        return model

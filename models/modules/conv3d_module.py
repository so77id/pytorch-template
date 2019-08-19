import torch
from models.layers.conv_layers import Conv3d


class Conv3dModule(torch.nn.Module):
    def __init__(self, in_channels, dropout_prob, out_channels_list=[], ks=(3,3,3), ks_pool=(1,2,2), bh=False, act=False):
        super(Conv3dModule, self).__init__()

        self.conv_1 = Conv3d(in_channels=in_channels, out_channels=out_channels_list[0], ks=ks,
                             stride=1, padding=1, dilation=1, groups=1, bh=bh, act=act, dropout_prob=dropout_prob)

        self.pool_1 = torch.nn.MaxPool3d(kernel_size=ks_pool)


        self.conv_2 = Conv3d(in_channels=out_channels_list[0], out_channels=out_channels_list[1], ks=ks,
                             stride=1, padding=1, dilation=1, groups=1, bh=bh, act=act, dropout_prob=dropout_prob)

        self.pool_2 = torch.nn.MaxPool3d(kernel_size=ks_pool)


        self.conv_3 = Conv3d(in_channels=out_channels_list[1], out_channels=out_channels_list[2], ks=ks,
                             stride=1, padding=1, dilation=1, groups=1, bh=bh, act=act, dropout_prob=dropout_prob)

        self.pool_3 = torch.nn.MaxPool3d(kernel_size=ks_pool)



    def forward(self, x, flatten=True):

        model = self.conv_1(x)
        model = self.pool_1(model)

        model = self.conv_2(model)
        model = self.pool_2(model)

        model = self.conv_3(model)
        model = self.pool_3(model)

        return model

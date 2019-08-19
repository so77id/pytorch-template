import torch
from models.layers.linear_layers import Linear


class ClassifierModule(torch.nn.Module):
    def __init__(self, in_size, n_labels, dropout_prob, bh=False):
        super(ClassifierModule, self).__init__()

        self.linear1 = Linear(in_features=in_size, out_features=512, bh=bh, act='relu', dropout_prob=dropout_prob)
        self.linear2 = Linear(in_features=512, out_features=256, bh=bh, act='relu', dropout_prob=dropout_prob)
        self.linear3 = Linear(in_features=256, out_features=n_labels, bh=bh, act='softmax')


    def forward(self, x):
        model = self.linear1(x)
        model = self.linear2(model)
        model = self.linear3(model)

        return model

class ClassifierModule_1l(torch.nn.Module):
    def __init__(self, in_size, n_labels):
        super(ClassifierModule_1l, self).__init__()
        self.linear1 = Linear(in_features=in_size, out_features=n_labels, bh=False)


    def forward(self, x):
        model = self.linear1(x)

        return model

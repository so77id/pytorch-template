import torch
from models.layers.activation_factory import ActivationFactory

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bh=False, act=False, dropout_prob=0.0):
        super(Linear, self).__init__()

        self.bh = bh
        self.act = act
        self.drop = False

        bias = True
        if self.bh:
            self.bh = torch.nn.BatchNorm1d(out_features)
            bias = False

        self.lineal = torch.nn.Linear(in_features=in_features,
                                      out_features=out_features,
                                      bias=bias)

        if dropout_prob > 0.0:
            self.drop = True
            self.dropout = torch.nn.Dropout(p=dropout_prob)

        if act:
            self.act = ActivationFactory.factory(act_name=self.act)

    def forward(self, x):
        model = self.lineal(x)
        if self.bh:
            model = self.bh(model)
        if self.act:
            model = self.act(model)
        if self.drop:
            model = self.dropout(model)

        return model

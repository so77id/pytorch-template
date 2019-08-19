import torch

class BaseModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def loss(self, y_preds, y):
        raise NotImplementedError

    def load_weights(self, path):
        print("Loading weigths from: {}".format(path))
        pretrained_dict = torch.load(path)

        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        print("weigths loaded:")
        for k,v in pretrained_dict.items():
            print("\t" + k)
        # 3. load the new state dict
        self.load_state_dict(pretrained_dict)

    def save_weights(self, path):
        raise NotImplementedError

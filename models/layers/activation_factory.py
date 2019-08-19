import torch


class ActivationFactory(object):
    """Activation factory return instance of model specified in type."""
    @staticmethod
    def factory(*args, **kwargs):
        if kwargs["act_name"] == "relu":
            return torch.nn.ReLU()
        elif kwargs["act_name"] == "softmax":
            return torch.nn.Softmax()
        elif kwargs["act_name"] == "sigmoid":
            return torch.nn.Sigmoid()

        else:
            assert 0, "Bad act_name in activation creation: " + kwargs["act_name"]

        return None

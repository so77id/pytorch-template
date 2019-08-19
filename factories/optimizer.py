from __future__ import absolute_import, division, print_function

import torch


class OptimizerFactory(object):
    """Optimizer factory return instance of model specified in type."""
    @staticmethod
    def factory(model_params, optimizer_name, *args, **kwargs):
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model_params, **kwargs)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model_params, **kwargs)
        elif optimizer_name == "ASGD":
            optimizer = torch.optim.ASGD(model_params, **kwargs)
        elif optimizer_name == "Adamax":
            optimizer = torch.optim.Adamax(model_params, **kwargs)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model_params, **kwargs)
        elif optimizer_name == "Adadelta":
            optimizer = torch.optim.Adadelta(model_params, **kwargs)
        else:
            assert 0, "Bad optimizer name to creation: " + kwargs["optimizer"]

        return optimizer


def get_optimizer_parameters(opt_name, lr=1e-3, wd=0.005):

    if opt_name == "adadelta":
        # Adadelta
        optimizer_kwargs = { "optimizer_name":"Adadelta",
                             "rho": 0.9,
                             "eps": 1e-6,
                             "lr": lr,
                             "weight_decay": wd}
    elif opt_name == "adam":
        # Adam
        # lr = 1e-5
        optimizer_kwargs = { "optimizer_name":"Adam",
                          "betas": (0.9, 0.999),
                          "eps": 1e-8,
                          "lr": lr,
                          "weight_decay": wd,
                          "amsgrad": False}

    elif opt_name == "adamax":
        # Adamax
        # lr = 1e-5
        optimizer_kwargs = { "optimizer_name":"Adamax",
                           "betas": (0.9, 0.999),
                           "eps": 1e-8,
                           "lr": lr,
                           "weight_decay": wd}
    elif opt_name == "asgd":
        # ASGD
        optimizer_kwargs = { "optimizer_name":"ASGD",
                            "lambd": 1e-4,
                            "alpha": 0.75,
                            "t0": 1e6,
                            "lr": lr,
                            "weight_decay": wd}

    elif opt_name == "rmsprop":
        # RMSprop
        optimizer_kwargs = { "optimizer_name":"RMSprop",
                             "momentum": 0,
                             "alpha": 0.99,
                             "eps": 1e-8,
                             "centered": False,
                             "lr": lr,
                             "weight_decay": wd}

    elif opt_name == "sgd":
        # SGD
        optimizer_kwargs = { "optimizer_name":"SGD",
                            "momentum": 0,
                            "dampening": 0,
                            "nesterov": False,
                            "lr": lr,
                            "weight_decay": wd}

    return optimizer_kwargs

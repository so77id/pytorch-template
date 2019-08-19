from __future__ import absolute_import, division, print_function

import torch

from models.c3d import C3dNet
from models.c3d_lstm import C3dLstmNet
from models.c3d_block_lstm import C3dBlockLstmNet
from models.cnn3d import CNN3dNet
from models.cnn3d_lstm import CNN3dLSTMNet
from models.cnn3d_block_lstm import Cnn3dBlockLstmNet

from models.crnn import CRNNNet
from models.crnn_nf import CRNNNFNet
from models.crnn_simple_sum import CRNNSimpleSum
from models.crnn_skip import CRNNSkip
from models.crnn_attention import CRNNAttention
from models.crnn_attention_nn import CRNNAttentionNN

from models.inceptionv3_lstm import InceptionV3LstmNet
from models.vgg16_lstm import VGG16LstmNet

from models.resnet101_lstm import ResNet101LstmNet
from models.resnet18_lstm import ResNet18LstmNet
from models.resnet3d import resnet101, resnet18
from models.i3d import InceptionI3d

class ModelFactory(object):
    """Model factory return instance of model specified in type."""
    @staticmethod
    def factory(*args, **kwargs):

        if kwargs["model_name"] == "c3d":
            model = C3dNet(*args, **kwargs)

        elif kwargs["model_name"] == "c3d_top":
            model = C3dTOPNet(*args, **kwargs)

        elif kwargs["model_name"] == "c3d_lstm":
            model = C3dLstmNet(*args, **kwargs)

        elif kwargs["model_name"] == "c3d_block_lstm":
            model = C3dBlockLstmNet(*args, **kwargs)

        elif kwargs["model_name"] == "cnn3d":
            model = CNN3dNet(*args, **kwargs)

        elif kwargs["model_name"] == "cnn3d_lstm":
            model = CNN3dLSTMNet(*args, **kwargs)

        elif kwargs["model_name"] == "cnn3d_block_lstm":
            model = Cnn3dBlockLstmNet(*args, **kwargs)

        # resnets
        elif kwargs["model_name"] == "resnet3d_101":
            model = resnet101(*args, **kwargs)

        elif kwargs["model_name"] == "resnet3d_18":
            model = resnet18(*args, **kwargs)

        # i3d
        elif kwargs["model_name"] == "i3d":
            model = InceptionI3d(*args, **kwargs)

        elif kwargs["model_name"] == "crnn":
            model = CRNNNet(*args, **kwargs)

        elif kwargs["model_name"] == "crnn_nf":
            model = CRNNNFNet(*args, **kwargs)

        elif kwargs["model_name"] == "crnn_simple_sum":
            model = CRNNSimpleSum(*args, **kwargs)

        elif kwargs["model_name"] == "crnn_skip":
            model = CRNNSkip(*args, **kwargs)

        elif kwargs["model_name"] == "crnn_attention":
            model = CRNNAttention(*args, **kwargs)

        elif kwargs["model_name"] == "crnn_attention_nn":
            model = CRNNAttentionNN(*args, **kwargs)

        elif kwargs["model_name"] == "inceptionv3_lstm":
            model = InceptionV3LstmNet(*args, **kwargs)

        elif kwargs["model_name"] == "vgg16_lstm":
            model = VGG16LstmNet(*args, **kwargs)

        elif kwargs["model_name"] == "resnet101_lstm":
            model = ResNet101LstmNet(*args, **kwargs)

        elif kwargs["model_name"] == "resnet18_lstm":
            model = ResNet18LstmNet(*args, **kwargs)

        else:
            assert 0, "Bad model_name of model creation: " + kwargs["model_name"]

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

            model = torch.nn.DataParallel(model)

        return model

import torch

def get_dataset_by_model(model_name, dataset_name, *args, **kwargs):
    if dataset_name in ["UCF-101"]:
        return 'UCF-101'
    if model_name in ["c3d", "c3d_top", "c3d_lstm", "cnn3d", "cnn3d_lstm", "inceptionv3_lstm", "vgg16_lstm", "resnet3d_101","resnet3d_18", "i3d", "resnet101_lstm", "resnet18_lstm"]:
        return 'video'
    elif model_name in ["c3d_block_lstm", "cnn3d_block_lstm", "crnn", "crnn_nf", "crnn_simple_sum", "crnn_skip", "crnn_attention", "crnn_attention_nn"]:
        return "blocks"


def get_model_params(model_name, dataset_name, n_labels, pretrained, bh=False, dropout_prob=0.5):
    n_gpus = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1

    batch_size = 10
    img_size = 100
    n_frames = 25
    n_blocks = 5
    frames_per_block = 5
    model_kwargs = {}

    if model_name == "inceptionv3_lstm":
        batch_size = 3
        img_size = 300
        model_kwargs = {}

    elif  model_name == "vgg16_lstm":
        batch_size = 3
        img_size = 224
        model_kwargs = {}

    elif model_name == "resnet101_lstm":
        batch_size = 3
        img_size = 224
        model_kwargs = {}

    elif model_name == "resnet18_lstm":
        img_size = 224
        model_kwargs = {}

    elif model_name == "c3d":
        n_frames = 16
        n_blocks = 4
        frames_per_block = 4
        model_kwargs = {}

    elif model_name == "resnet3d_101":
        model_kwargs = {}

    elif model_name == "resnet3d_18":
        batch_size = 20
        model_kwargs = {}

    elif model_name == "i3d":
        batch_size = 5
        img_size = 226
        n_frames = 64
        n_blocks = 8
        frames_per_block = 8
        model_kwargs = {
            "dataset_name": dataset_name
        }

    elif model_name == "c3d_block_lstm":
        model_kwargs = {}

    elif model_name == "c3d_top":
        batch_size = 10
        model_kwargs = {
            "bh": bh
        }


    model_kwargs = {
        "model_name": model_name,
        "img_size": img_size,
        "n_labels": n_labels,
        "dropout_prob": dropout_prob,
        "pretrained": pretrained,
        "n_frames": n_frames,
        "n_blocks": n_blocks,
        "frames_per_block": frames_per_block,
        "model_kwargs": model_kwargs
    }
    batch_size =  batch_size * n_gpus

    return model_kwargs, batch_size


# Model vars


# model_name = 'crnn'
# batch_size = 10 * torch.cuda.device_count()
# img_size = 100
# n_frames = 25
# n_blocks = 5
# frames_per_block = 5
# pretrained = False
# model_kwargs = {
#     "bh": True
# }

# model_name = 'crnn_nf'
# batch_size = 8 * torch.cuda.device_count()
# img_size = 100
# n_frames = 25
# n_blocks = 5
# frames_per_block = 5
# pretrained = False
# model_kwargs = {}

# model_name = 'crnn_simple_sum'
# batch_size = 10 * max(1, torch.cuda.device_count())
# img_size = 100
# n_frames = 25
# n_blocks = 5
# frames_per_block = 5
# pretrained = False
# model_kwargs = {
#     "bh": True,
#     "norm_sum": True
# }

# model_name = 'crnn_skip'
# batch_size = 6 * max(1, torch.cuda.device_count())
# img_size = 100
# n_frames = 25
# n_blocks = 5
# frames_per_block = 5
# pretrained = False
# model_kwargs = {
#     "bh": True,
#     "norm_sum": True,
#     "nc": 2,
#     "sc": 1
# }

# model_name = 'crnn_attention'
# batch_size = 10 * torch.cuda.device_count()
# img_size = 100
# n_frames = 25
# n_blocks = 5
# frames_per_block = 5
# pretrained = False
# model_kwargs = {
#     "norm_sum": False,
#     "t_max": n_blocks
# }

# model_name = 'crnn_attention_nn'
# batch_size = 10 * torch.cuda.device_count()
# img_size = 100
# n_frames = 25
# n_blocks = 5
# frames_per_block = 5
# pretrained = False
# model_kwargs = {
#     "bh": True,
#     "rcnn_bh": False,
#     "dropout": dropout_prob,
#     "rcnn_dropout": dropout_prob,
#     "norm_sum": False,
#     "seq_size": n_blocks
# }

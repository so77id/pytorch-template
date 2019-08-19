import torchvision

from utils.dataset_transforms import VideoRescale, VideoRandomCrop, VideoToTensor, TensorClipToTensorBlocks

from videotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, RandomHorizontalFlip
from videotransforms.volume_transforms import ClipToTensor

def get_transformations(dataset_type, train=True, data_augmentation=False, rotation_angle=30, img_size=100, crop_ratio=0.8, n_blocks=4, *args, **kwargs):
    if data_augmentation and train:
        transform_list = [
            RandomHorizontalFlip(),
            RandomRotation(rotation_angle),
            Resize((img_size, img_size)),
            # RandomCrop((int(img_size * crop_ratio), int(img_size * crop_ratio))),
            # Resize((img_size, img_size)),
            ClipToTensor()
        ]
        # transform_list = [
        #     Resize((img_size, img_size)),
        #     ClipToTensor()
        # ]
    else:
        transform_list = [
            Resize((img_size, img_size)),
            ClipToTensor()
        ]

    # transform_list = [
    #     VideoRescale(img_size),
    #     VideoToTensor()
    # ]
    #
    # transform =  torchvision.transforms.Compose(transform_list)


    if dataset_type == "blocks" or dataset_type == "UCF-101":
        transform_list.append(TensorClipToTensorBlocks(n_blocks))

    transform = Compose(transform_list)

    return transform




#
# class TranformFactory(object):
#     """Dataset factory return instance of transform specified in type."""
#     @staticmethod
#     def factory(dataset_type, transform_type, *args, **kwargs):
#         if dataset_type == "video":
#             if transform_type == "rescale":
#                 return VideoRescale(*args, **kwargs)
#             if transform_type == "random_crop":
#                 return VideoRandomCrop(*args, **kwargs)
#             if transform_type == "to_tensor":
#                 return VideoToTensor()
#
#         if dataset_type == "landmark":
#             if transform_type == "rescale":
#                 return LandmarkRescale(*args, **kwargs)
#             if transform_type == "random_crop":
#                 return LandmarkRandomCrop(*args, **kwargs)
#             if transform_type == "to_tensor":
#                 return LandmarkToTensor()
#
#         if dataset_type == "blocks":
#             if transform_type == "rescale":
#                 return BlockRescale(*args, **kwargs)
#             if transform_type == "random_crop":
#                 return BlockRandomCrop(*args, **kwargs)
#             if transform_type == "to_tensor":
#                 return BlockToTensor()
#
#
#         assert 0, "Bad transform_type of transform creation: " + transform_type

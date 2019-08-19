import torch
import numpy as np
from torchvision import transforms

from skimage import transform

# Video transformations
class VideoRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, clip):
        new_clip = []
        for image in clip:
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))

            new_clip.append(img)

        return new_clip


class VideoRandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        h, w = clip[0].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_clip = []
        for image in clip:
            img = image[top: top + new_h,
                        left: left + new_w]
            new_clip.append(img)

        return new_clip


class VideoToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, clip):
        clip = np.array(clip)

        # swap color axis because
        # numpy image: T X H x W x C
        # torch image: C X T X H X W
        clip = clip.transpose((3, 0, 1, 2))
        return torch.from_numpy(clip)


class TensorClipToTensorBlocks(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, n_blocks):
        self.n_blocks = n_blocks

    def __call__(self, clip):
        # B x C X T X H X W
        C, T, H, W = clip.shape
        clip = clip.view(C, self.n_blocks, -1, H, W).transpose(0,1)

        return clip


# Video blocks transformation
class BlockRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, clip):
        new_clip = []
        for block in clip:
            new_block = []
            for image in block:
                h, w = image.shape[:2]
                if isinstance(self.output_size, int):
                    if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                    else:
                        new_h, new_w = self.output_size, self.output_size * w / h
                else:
                    new_h, new_w = self.output_size

                new_h, new_w = int(new_h), int(new_w)

                img = transform.resize(image, (new_h, new_w))

                new_block.append(img)
            new_clip.append(new_block)

        return new_clip


class BlockRandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip):
        h, w = clip[0].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_clip = []
        for block in clip:
            new_block = []
            for image in clip:
                img = image[top: top + new_h,
                            left: left + new_w]
                new_block.append(img)
            new_clip.append(new_block)

        return new_clip


class BlockToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, clip):
        clip = np.array(clip)

        # swap color axis because
        # numpy image: B X T X H x W x C
        # torch image: B X C X T X H X W
        clip = clip.transpose((0, 4, 1, 2, 3))
        return torch.from_numpy(clip)


# Landmark Transformations
class LandmarkRescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, clip_data):
        clip = clip_data["clip"]
        clip_landmarks = clip_data["landmarks"]

        new_clip = []
        new_landmarks = []
        for image, landmarks in zip(clip, clip_landmarks):
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h, new_w))
            landmarks = np.round(np.multiply(landmarks, np.array([new_w / w, new_h / h])[np.newaxis, :])/self.output_size).astype(np.int)
            new_clip.append(img)
            new_landmarks.append(landmarks)

        return {"clip": np.array(new_clip), "landmarks": np.array(new_landmarks)}


class LandmarkRandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, clip_data):
        # TODO FIX
        clip = clip_data["clip"]
        clip_landmarks = clip_data["landmarks"]


        h, w = clip[0].shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_clip = []
        new_landmarks = []
        for image, landmarks in zip(clip, clip_landmarks):
            img = image[top: top + new_h,
                        left: left + new_w]

            landmarks = landmarks - np.array([left, top])[np.newaxis, :]

            new_clip.append(img)
            new_landmarks.append(landmarks)

        return {"clip": np.array(new_clip), "landmarks": np.array(new_landmarks)}


class LandmarkToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, clip_data):
        clip = clip_data["clip"]
        clip_landmarks = clip_data["landmarks"]

        # swap color axis because
        # numpy image: T X H x W x C
        # torch image: C X T X H X W
        clip = clip.transpose((3, 0, 1, 2))

        return {"clip": torch.from_numpy(clip), "landmarks": torch.from_numpy(clip_landmarks.flatten())}

from __future__ import absolute_import, division, print_function

import imutils
import numpy as np
from tqdm import tqdm



def classic_data_augmentation(x, y, names, flip=True, rotate=True, min_angle=-10, max_angle=10, step_angle=2):
    """
    data -> { "x": np.array(n, f, w, h, c), "y": np.array(n, l) }
    n -> number of videos
    f -> number of frames
    w -> width
    h -> height
    c -> channels
    l -> number of labels
    """
    print("Data augmentation for each video:")


    video_multiplication = 1
    if flip:
        video_multiplication += 1
    if rotate:
        # negative angles
        angles = list(range(min_angle, 0, step_angle))
        angles += list(range(step_angle, max_angle + step_angle, step_angle))
        video_multiplication += len(angles)

    videos = []
    labels = []
    new_names = []

    ix = 0
    loop = tqdm(zip(x, y, names), total=len(x))
    for video, label, name in loop:
        videos.append(video)
        labels.append(label)
        new_names.append(name)
        ix += 1

        if flip:
            aug_video = np.zeros(video.shape)
            for i, frame in enumerate(video):
                aug_video[i] = np.fliplr(frame)

            videos.append(aug_video)
            labels.append(label)
            new_names.append(name+"-flip")
            ix += 1

        if rotate:
            for angle in angles:
                aug_video = np.zeros(video.shape)
                for i, frame in enumerate(video):
                    aug_video[i] = imutils.rotate(frame, angle)

                videos.append(aug_video)
                labels.append(label)
                new_names.append(name+"-rotate-{}".format(angle))
                ix += 1
    #
    # for video in videos:
    #     for frame in video:
    #         cv2.imshow("0", np.uint8(frame))
    #         cv2.waitKey(0)

    return videos, labels, new_names

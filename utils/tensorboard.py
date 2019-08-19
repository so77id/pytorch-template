import torch
import matplotlib
from textwrap import wrap
import re
import itertools

import numpy as np
from tensorboardX import SummaryWriter


def init_writer(log_dir):
    return SummaryWriter(log_dir)

def save_graph(model, device, dataloader, tb_writer):
    with torch.no_grad():
        for batch in dataloader:
            tb_writer.add_graph(model, input_to_model=batch['x'].float().to(device))
            break

def save_weigths(model, tb_writer, epoch):
    for name, value in model.named_parameters():
        name = "-".join(name.split("."))

        tb_writer.add_histogram("Weigths/{}".format(name), value, global_step=epoch)

def save_grads(grad_dict, tb_writer, epoch):
    for name, value in grad_dict.items():
        name = "-".join(name.split("."))
        tb_writer.add_histogram("Grads/{}".format(name), value, global_step=epoch)

def save_video_inputs(video, tb_writer, name, epoch):
    tb_writer.add_video(name, video.unsqueeze(0), epoch)



def save_confusion_matrix(correct_labels, predict_labels, labels,  tb_writer, epoch, normalize=False, title='Confusion matrix'):

    cm, type = create_confusion_matrix(correct_labels, predict_labels, labels, normalize)

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    # if normalize:
    #     type = '.2f'
    # else:
    #     type = 'd'

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=100, facecolor='w', edgecolor='k')
    fig.title = title
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=8, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=8, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], type), horizontalalignment="center", fontsize=8, verticalalignment='center', color= "black")
    fig.set_tight_layout(True)

    tb_writer.add_figure(title, fig, global_step=epoch)


from sklearn.metrics import confusion_matrix

def create_confusion_matrix(correct_labels, predict_labels, labels, normalize=False):
    """This function create a confusion matrix normalized depending that normalize variable.

    Parameters
    ----------
    correct_labels : numpy array
        list of correct labels.
    predict_labels : numpy array
        list of predicted labels.
    labels : numpy array
        list of name of labels.
    normalize : bool
        describe if return confusion matrix normalized or normal.

    Returns
    -------
    type
        return confusion matrix.

    """
    cm = confusion_matrix(correct_labels, predict_labels)

    type = 'd'
    if normalize:
        cm = cm.astype('float')*100 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        type = '.2f'

    return cm, type

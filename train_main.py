import os
import torch
import sys
import numpy as np

from factories.optimizer import get_optimizer_parameters

from train.train import train

from factories.optimizer import OptimizerFactory
from factories.model import ModelFactory
from factories.dataset import DatasetFactory

from utils.model_utils import get_dataset_by_model, get_model_params
from utils.dataset_utils import get_dataset_params
from utils.args import get_args
from utils.experiment import init_experiment, ini_checkpoint_experiment
from utils.slack import send_slack

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def main():
    args = get_args()


    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using:", device)


    # Logs vars
    metadata_path = '../metadata'


    # Train vars
    ini_epoch = 0
    n_epochs = args["n_epochs"]
    lr = args["lr"]
    wd = args["wd"]
    opt_name = args["opt_name"]

    # Network vars
    dropout_prob = args["dropout_prob"]


    # Get optimizer vars
    optimizer_kwargs = get_optimizer_parameters(opt_name, lr=lr, wd=wd)

    # Dataset vars
    dataset_name = args["dataset_name"]

    dataset_params = get_dataset_params(dataset_name)

    model_name =  args["model_name"]
    pretrained = args["pretrained"]

    model_params, batch_size = get_model_params(model_name,
                                                dataset_name,
                                                dataset_params["n_labels"],
                                                pretrained=pretrained,
                                                bh=False,
                                                dropout_prob=dropout_prob)


    data_augmentation_kwargs = {
        "data_augmentation": args["data_augmentation"],
        "rotation_angle":30,
        "img_size": model_params["img_size"],
        "crop_ratio":0.8
    }

    dataset_type = get_dataset_by_model(model_name, dataset_name)

    # Dataset loaders
    dataset_path_pattern = '../datasets/{}/kfold_lists/without_crop/groups/{}-600-{}.list'

    # Creating model
    model = ModelFactory.factory(**model_params)
    model.to(device)

    print(model.parameters)

    # Train set
    # files = [dataset_path_pattern.format(dataset_name, mode, 0) for mode in ["train", "validation"]]
    files = dataset_path_pattern.format(dataset_name, "train", 0)

    train_set = DatasetFactory.factory(dataset_type=dataset_type,
                                       csv_file=files,
                                       n_frames=model_params["n_frames"],
                                       n_blocks=model_params["n_blocks"],
                                       frames_per_block=model_params["frames_per_block"],
                                       train=True,
                                       **dataset_params,
                                       **data_augmentation_kwargs)

    trainloader = torch.utils.data.DataLoader(train_set,
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=20)

    # Test set
    files = dataset_path_pattern.format(dataset_name, "validation", 0)
    test_set = DatasetFactory.factory(dataset_type=dataset_type,
                                      csv_file=files,
                                      n_frames=model_params["n_frames"],
                                      n_blocks=model_params["n_blocks"],
                                      frames_per_block=model_params["frames_per_block"],
                                      train=False,
                                      **dataset_params,
                                      **data_augmentation_kwargs)

    testloader = torch.utils.data.DataLoader(test_set,
                                             batch_size=batch_size,shuffle=True,
                                             num_workers=20)



    # Create optimizer
    optimizer = OptimizerFactory.factory(model.parameters(), **optimizer_kwargs)


    filename = args["checkpoint_path"]
    if filename != "":
        if os.path.isfile(filename):
            ini_checkpoint_experiment(filename, model_name, dataset_name)

            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            ini_epoch = checkpoint['epoch']

            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(filename))
            exit()
    else:
        # Init experiment
        model_dir, log_dir, experiment_id = init_experiment(metadata_path, dataset_name, model_name)


    train_metrics = train(model=model,
                          optimizer=optimizer,
                          ini_epoch=ini_epoch,
                          n_epochs=n_epochs,
                          device=device,
                          trainloader=trainloader,
                          testloader=testloader,
                          model_dir=model_dir,
                          log_dir=log_dir)

    print("test acc:", train_metrics["test"]["acc"])
    print("Kfold finished log path:", log_dir)

    msg = """
    ```
    Exp Name: `{}``
    Host Machine: `{}`
    acc list: `{}`
    log path: `{}`
    ```
    """.format(experiment_id, os.environ["HOSTNAME"],  train_metrics["test"]["acc"], log_dir)
    send_slack(msg)



if __name__ == "__main__":
    sys.exit(main())

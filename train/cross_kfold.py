import torch
import os
import numpy as np


from factories.optimizer import OptimizerFactory
from factories.model import ModelFactory
from factories.dataset import DatasetFactory


from train.train import train_with_many_test
from utils.slack import send_slack
from utils.gdrive import GDriveCrossKfold

from utils.experiment import init_experiment, init_kfold_subexperiment, ini_kfold_checkpoint_experiment

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def merge_datasets(dataset_list):
    db = dataset_list[0]
    for dataset in dataset_list[1:]:
        db.clips.extend(dataset.clips)
        db.labels.extend(dataset.labels)
        db.clip_name.extend(dataset.clip_name)

    return db


def save_kfold_checkpoint(i, models_dir, logs_dir, experiment_id, test_acc_list):
    # Saving kfold checkpoint
    print("Saving kfold checkpoint in:", "{}/kfold-checkpoint.pth".format(models_dir))
    state = {'k': i,
             'test_acc_list': test_acc_list,
             #"models_dir": models_dir,
             #"logs_dir": logs_dir,
             #"experiment_id": experiment_id
             }

    torch.save(state, "{}/kfold-checkpoint.pth".format(models_dir))


def kfold(k, device, n_epochs, optimizer_kwargs, batch_size, dataset_type,
          model_params, dataset_params, dataset_path_pattern, metadata_path,
          data_augmentation_kwargs, checkpoint_file):

    # dataset_name = dataset_params["dataset_name"]
    dataset_name = "_".join([dataset_name for dataset_name, _  in dataset_params.items()])
    model_name = model_params["model_name"]


    init_k = 0
    if checkpoint_file != "":
        if os.path.isfile(checkpoint_file):

            print("=> loading kfold checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            # init_k = checkpoint['k']
            test_acc_list = checkpoint["test_acc_list"]
            #../metadata/CK-experiments/vgg16_lstm/2019-06-19/2/checkpoints/kfold-checkpoint.pth
            c_split = checkpoint_file.split("/")

            models_dir = "{}/checkpoints".format("/".join(c_split[:6]))
            logs_dir = "{}/logs".format("/".join(c_split[:6]))
            experiment_id = "/".join(c_split[2:6])

            ini_kfold_checkpoint_experiment(checkpoint_file, model_name, dataset_name, models_dir, logs_dir, experiment_id)

            print("=> loaded kfold checkpoint_file '{}' (k {})".format(checkpoint_file, init_k))

        else:
            print("=> no checkpoint_file found at '{}'".format(checkpoint_file))
            exit()
    else:
        # Init experiment
        test_acc_list = {}
        models_dir, logs_dir, experiment_id = init_experiment(metadata_path, dataset_name, model_name)

    save_kfold_checkpoint(0, models_dir, logs_dir, experiment_id, test_acc_list)
    print(model_params["model_name"])
    print(list(dataset_params.keys()))
    gdrive = GDriveCrossKfold(model_name=model_params["model_name"],
                              dataset_list=list(dataset_params.keys()),
                              transfer=model_params["pretrained"],
                              data_augmentation=data_augmentation_kwargs["type"])
    gdrive.init_exp(os.environ["HOSTNAME"], logs_dir)


    for i in range(init_k, k):
        model_dir, log_dir, kfold_experiment_id = init_kfold_subexperiment(models_dir, logs_dir, i, experiment_id)

        # see if exists checkpoint for this experiment
        filename = "{}/kfold-{}/checkpoint.pth".format(models_dir, i)
        ini_epoch = 0
        checkpoint = None
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            ini_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            if ini_epoch >= n_epochs:
                print("==> this checkpoint has already been trained the number of epochs indicated")
                continue
        else:
            print("No checkpoint found ({}), creating new".format(filename))


        # Load model
        model = ModelFactory.factory(**model_params)
        model.to(device)

        print(model.parameters)

        # Train set
        if data_augmentation_kwargs["type"] == "s":
            modes = ["train", "validation", "syn"]
        else:
            modes = ["train", "validation"]

        train_sets = []
        test_sets = {}

        for dataset_name, dataset_param in dataset_params.items():
            print("Loading:", dataset_name)
            files = [dataset_path_pattern.format(dataset_name, mode, i) for mode in modes]

            train_set = DatasetFactory.factory(dataset_type=dataset_type,
                                               csv_file=files,
                                               n_frames=model_params["n_frames"],
                                               n_blocks=model_params["n_blocks"],
                                               frames_per_block=model_params["frames_per_block"],
                                               train=True,
                                               **dataset_param,
                                               **data_augmentation_kwargs)
            train_sets.append(train_set)

            # Test set
            files = dataset_path_pattern.format(dataset_name, "test", i)
            test_sets[dataset_name] = DatasetFactory.factory(dataset_type=dataset_type,
                                                             csv_file=files,
                                                             n_frames=model_params["n_frames"],
                                                             n_blocks=model_params["n_blocks"],
                                                             frames_per_block=model_params["frames_per_block"],
                                                             train=False,
                                                             **dataset_param,
                                                             **data_augmentation_kwargs)

            # test_sets.append(test_set)

        train_set = merge_datasets(train_sets)
        # test_set = merge_datasets(test_sets)

        trainloader = torch.utils.data.DataLoader(train_set,
                                                  batch_size=batch_size,
                                                  shuffle=True, num_workers=10)

        testloaders = {}
        for db_name, test_set in test_sets.items():
            testloaders[db_name] = torch.utils.data.DataLoader(test_set,
                                                               batch_size=batch_size,shuffle=True,
                                                               num_workers=10)



        # Create optimizer
        optimizer = OptimizerFactory.factory(model.parameters(), **optimizer_kwargs)


        if checkpoint is not None:
            if isinstance(model, torch.nn.DataParallel):
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])


        train_metrics = train_with_many_test(model=model,
                                             optimizer=optimizer,
                                             ini_epoch=ini_epoch,
                                             n_epochs=n_epochs,
                                             device=device,
                                             trainloader=trainloader,
                                             testloaders=testloaders,
                                             model_dir=model_dir,
                                             log_dir=log_dir)

        # if "test" in train_metrics and "acc" in train_metrics["test"]:
        if "test" in train_metrics:
            for db_name, test_metric in train_metrics["test"].items():
                if db_name not in test_acc_list:
                    test_acc_list[db_name] = {}

                test_acc_list[db_name][i] = test_metric["acc"]

        # Saving kfold checkpoint
        save_kfold_checkpoint(i+1, models_dir, logs_dir, experiment_id, test_acc_list)

    acc_maps = {}
    for db_name, acc_map in test_acc_list.items():
        test_list = np.zeros(k,dtype=np.float)
        for i,v in acc_map.items():
            test_list[i] = v
        acc_maps[db_name] = list(test_list)



    results_msg = """
    ----
    Dataset: {}
    Test Acc mean: {}
    Test Acc list: {}
    """

    results_str = ""

    print("Test acc by dataset")
    for db_name, acc_list in acc_maps.items():
        print("dataset:", db_name)
        print("kfold test acc:", np.mean(acc_list))
        results_str += results_msg.format(db_name, np.mean(acc_list), acc_list)

    print("Kfold finished log path:", logs_dir)


    gdrive.end_exp(acc_maps)

    msg = """
    ```
    Cross dataset kfold

    Exp Name: `{}``
    Host Machine: `{}`
    Test results:
    {}
    Kfold log path: `{}`
    ```
    """.format(experiment_id, os.environ["HOSTNAME"], results_str, logs_dir)
    send_slack(msg)

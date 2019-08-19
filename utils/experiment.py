import datetime
import os
import shutil


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0

    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)
# from utils.dirs import create_dirs

def init_experiment(path, dataset_name, model_name, config_files=None, exp_path=""):
    # get the experiments dirs
    models_dir, log_dir, metadata_dir, configs_dir, experiment_id = get_experiment_dirs(
        path, dataset_name, model_name, exp_path)
    # create the experiments dirs
    create_dirs([models_dir, log_dir, configs_dir])
    # Log metadata
    log_metadata_experiment(
        models_dir, log_dir, experiment_id
    )

    return models_dir, log_dir, experiment_id


def get_experiment_id(path, exp_name):
    current_date = str(datetime.datetime.now().date())
    exp_dir = "{}/{}/{}".format(path, exp_name, current_date)
    exp_id = "{}/{}".format(exp_name, current_date)

    if os.path.exists(exp_dir):
        filenames = os.listdir(exp_dir)
        filenames = sorted(filenames, key=float)
        if len(filenames) > 0:
            last_name = int(filenames[-1])
            exp_dir = "{}/{}".format(exp_dir, (last_name + 1))
            exp_id = "{}/{}".format(exp_id, (last_name + 1))
        else:
            exp_dir = "{}/{}".format(exp_dir, "1")
            exp_id = "{}/{}".format(exp_id, "1")
    else:
        exp_dir = "{}/{}".format(exp_dir, "1")
        exp_id = "{}/{}".format(exp_id, "1")

    return exp_dir, exp_id

def get_experiment_dirs(path, dataset_name, model_name, exp_path=""):
    # Creating metadata experiment folder
    exp_path = path if exp_path == "" else exp_path

    exp_name = "{}-experiments/{}".format(dataset_name, model_name)

    exp_dir, exp_id = get_experiment_id(exp_path, exp_name)

    models_dir = "{}/checkpoints".format(exp_dir)
    log_dir = "{}/logs".format(exp_dir)

    configs_dir = "{}/configs".format(exp_dir)

    return models_dir, log_dir, exp_dir, configs_dir, exp_id


def log_metadata_experiment(
    models_dir, log_dir, exp_id
):
    # Saving config file for future experiment reproduction
    # config_path = "{}/config.json".format(    print("\n---------------------------------------")
    # print("Copy config file into metadata folder:")
    # print("---------------------------------------")
#     print("Config files paths:")
#     for key, value in configs_files.items():
#         print(key, ":", value)
#         shutil.copy(value, "{}/{}.json".format(configs_dir, key))
    print("\n----------------------")
    print("Exp: {}, saved in:".format(exp_id))
    print("----------------------")
    print("Models:", models_dir)
    print("Tensorboard logs:", log_dir)
    print("\n")


def init_kfold_subexperiment(models_dir, logs_dir, k, experiment_id):
    model_dir = "{}/kfold-{}".format(models_dir, k)
    log_dir = "{}/kfold-{}".format(logs_dir, k)
    kfold_experiment_id = "{}/kfold-{}".format(experiment_id, k)

    create_dirs([model_dir, log_dir])

    log_kfold_metadata_subexperiment(model_dir, log_dir, k)

    return model_dir, log_dir, kfold_experiment_id


def log_kfold_metadata_subexperiment(model_dir, log_dir, k):
    print("\nKfold {} Experiments\nsaved in:".format(k))
    print("----------------------")
    print("Models:", model_dir)
    print("Tensorboard logs:", log_dir)
    print("\n")

def ini_checkpoint_experiment(checkpoint_file, model_name, dataset_name, cross=False):
    # ../metadata/CK-experiments/c3d_top/2019-06-11/4/checkpoints/checkpoint.pth
    # ../metadata/cross_1vall_train_CK-experiments/c3d/2019-07-18/4/checkpoints/checkpoint.pth
    metadata_path = "/".join(checkpoint_file.split("/")[0:2])
    checkpoint_model_name = checkpoint_file.split("/")[3]

    checkpoint_dataset_name = checkpoint_file.split("/")[2].split("-")[0]

    checkpoint_data = "/".join(checkpoint_file.split("/")[4:6])

    if cross:
        dataset_name = "cross_1vall_train_{}".format(dataset_name)

    if checkpoint_model_name != model_name:
        raise ValueError("train model name is not same of checkpoint model name")
    if checkpoint_dataset_name != dataset_name:
        raise ValueError("dataset name is not same of checkpoint dataset name")

    experiment_id = "{}-experiments/{}/{}".format(dataset_name, model_name, checkpoint_data)
    model_dir = "{}/{}/checkpoints".format(metadata_path, experiment_id)
    log_dir = "{}/{}/logs".format(metadata_path, experiment_id)

    # Log metadata
    log_metadata_experiment(
        model_dir, log_dir, experiment_id
    )

    return model_dir, log_dir, experiment_id

def ini_kfold_checkpoint_experiment(checkpoint_file, model_name, dataset_name, models_dir, logs_dir, experiment_id):
    # ../metadata/CK-experiments/c3d_top/2019-06-11/4/checkpoints/checkpoint.pth
    # ../metadata/CK-experiments/c3d/2019-06-16/2/checkpoints/kfold_checkpoint.pth
    checkpoint_model_name = checkpoint_file.split("/")[3]
    checkpoint_dataset_name = checkpoint_file.split("/")[2].split("-")[0]
    checkpoint_data = "/".join(checkpoint_file.split("/")[4:6])

    if checkpoint_model_name != model_name:
        raise ValueError("train model name is not same of checkpoint model name")
    if checkpoint_dataset_name != dataset_name:
        raise ValueError("dataset name is not same of checkpoint dataset name")

    # experiment_id = "{}-experiments/{}/{}".format(dataset_name, model_name, checkpoint_data)
    # model_dir = "{}/{}/checkpoints".format(metadata_path, experiment_id)
    # log_dir = "{}/{}/logs".format(metadata_path, experiment_id)

    # Log metadata
    log_metadata_experiment(
        models_dir, logs_dir, experiment_id
    )

    # return model_dir, log_dir, experiment_id

import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-m', '--model_name', required=True, help='Model name')
    argparser.add_argument('-d', '--dataset_name', required=True, help='Dataset name')
    argparser.add_argument('-e', '--n_epochs', required=True, help='Number of epochs', type=int)
    argparser.add_argument('-o', '--opt_name', required=True, help='Optimizer name')
    argparser.add_argument('-lr', '--lr', required=True, help='Learning rate', type=float)
    argparser.add_argument('-wd', '--wd', required=True, help='Weigth decay', type=float)

    argparser.add_argument('-drop', '--dropout_prob', required=False, help='dropout_prob', default=0.0, type=float)
    argparser.add_argument('-p', '--pretrained', type=str2bool, nargs='?', const=True, default=False, help="Pretrained")
    argparser.add_argument('-c', '--checkpoint_path', required=False, default="", help="Checkpoint path")
    argparser.add_argument('-da', '--data_augmentation', required=False, help='Data augmentation (c --> Classic, p --> pytorch, s --> synthetic) separated by comma ,', default="")


    args = argparser.parse_args()
    config_files = {
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "n_epochs": args.n_epochs,
        "opt_name": args.opt_name,
        "lr": args.lr,
        "wd": args.wd,
        "dropout_prob": args.dropout_prob,
        "pretrained": args.pretrained,
        "checkpoint_path": args.checkpoint_path,
        "data_augmentation": args.data_augmentation.split(","),
    }

    print(config_files)
    return config_files



def get_1vall_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-m', '--model_name', required=True, help='Model name')
    argparser.add_argument('-td', '--train_dataset_name', required=True, help='Train dataset name')
    argparser.add_argument('-ld', '--test_dataset_list', required=True, help='Test dataset list by ,')
    argparser.add_argument('-e', '--n_epochs', required=True, help='Number of epochs', type=int)
    argparser.add_argument('-o', '--opt_name', required=True, help='Optimizer name')
    argparser.add_argument('-lr', '--lr', required=True, help='Learning rate', type=float)
    argparser.add_argument('-wd', '--wd', required=True, help='Weigth decay', type=float)

    argparser.add_argument('-drop', '--dropout_prob', required=False, help='dropout_prob', default=0.0, type=float)
    argparser.add_argument('-p', '--pretrained', type=str2bool, nargs='?', const=True, default=False, help="Pretrained")
    argparser.add_argument('-c', '--checkpoint_path', required=False, default="", help="Checkpoint path")
    argparser.add_argument('-da', '--data_augmentation', required=False, help='Data augmentation (c --> Classic, p --> pytorch, s --> synthetic) separated by comma ,', default="")


    args = argparser.parse_args()
    config_files = {
        "model_name": args.model_name,
        "train_dataset": args.train_dataset_name,
        "test_datasets": args.test_dataset_list.split(','),
        "n_epochs": args.n_epochs,
        "opt_name": args.opt_name,
        "lr": args.lr,
        "wd": args.wd,
        "dropout_prob": args.dropout_prob,
        "pretrained": args.pretrained,
        "checkpoint_path": args.checkpoint_path,
        "data_augmentation": args.data_augmentation.split(","),
    }

    print(config_files)
    return config_files


def get_crosskfold_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-m', '--model_name', required=True, help='Model name')
    argparser.add_argument('-ld', '--dataset_list', required=True, help='Train dataset list by ,')
    argparser.add_argument('-e', '--n_epochs', required=True, help='Number of epochs', type=int)
    argparser.add_argument('-o', '--opt_name', required=True, help='Optimizer name')
    argparser.add_argument('-lr', '--lr', required=True, help='Learning rate', type=float)
    argparser.add_argument('-wd', '--wd', required=True, help='Weigth decay', type=float)

    argparser.add_argument('-drop', '--dropout_prob', required=False, help='dropout_prob', default=0.0, type=float)
    argparser.add_argument('-p', '--pretrained', type=str2bool, nargs='?', const=True, default=False, help="Pretrained")
    argparser.add_argument('-c', '--checkpoint_path', required=False, default="", help="Checkpoint path")
    argparser.add_argument('-da', '--data_augmentation', required=False, help='Data augmentation (c --> Classic, p --> pytorch, s --> synthetic) separated by comma ,', default="")


    args = argparser.parse_args()
    config_files = {
        "model_name": args.model_name,
        "dataset_list": args.dataset_list.split(','),
        "n_epochs": args.n_epochs,
        "opt_name": args.opt_name,
        "lr": args.lr,
        "wd": args.wd,
        "dropout_prob": args.dropout_prob,
        "pretrained": args.pretrained,
        "checkpoint_path": args.checkpoint_path,
        "data_augmentation": args.data_augmentation.split(","),
    }

    print(config_files)
    return config_files

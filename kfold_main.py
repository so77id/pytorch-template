import torch
import sys
import numpy as np

from factories.optimizer import get_optimizer_parameters

from train.kfold import kfold

from utils.model_utils import get_dataset_by_model, get_model_params
from utils.dataset_utils import get_dataset_params
from utils.args import get_args

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
        "type": [],
        "data_augmentation": False,
        "img_size": model_params["img_size"]
    }

    if "c" in args["data_augmentation"]:
        data_augmentation_kwargs["type"].append("c")
    if "p" in args["data_augmentation"]:
        data_augmentation_kwargs["type"].append("p")
        data_augmentation_kwargs["data_augmentation"] = True
        data_augmentation_kwargs["rotation_angle"] = 30
        data_augmentation_kwargs["crop_ratio"] = 0.8
    if  "s" in args["data_augmentation"]:
        data_augmentation_kwargs["type"].append("s")

    dataset_type = get_dataset_by_model(model_name, dataset_name)

    # Dataset loaders
    dataset_path_pattern = '../datasets/{}/kfold_lists/with_crop/groups/{}-{}.list'

    kfold(k=dataset_params["k"],
          device=device,
          n_epochs=n_epochs,
          optimizer_kwargs=optimizer_kwargs,
          batch_size=batch_size,
          dataset_type=dataset_type,
          model_params=model_params,
          dataset_params=dataset_params,
          dataset_path_pattern=dataset_path_pattern,
          metadata_path=metadata_path,
          data_augmentation_kwargs=data_augmentation_kwargs,
          checkpoint_file=args["checkpoint_path"])




if __name__ == "__main__":
    sys.exit(main())

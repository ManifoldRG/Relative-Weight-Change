import torch.nn as nn
import torch.optim as optim

from train import training
from data import load_cifar10
from delta import setup_delta_tracking
from utils import load_model

def run_experiment(epochs, model_name, training_type):
    """
    Runs a basic experiment.
    """

    # load data
    loaders = load_cifar10()

    # load model
    model = load_model(model_name, training_type)
    
    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), 0.05,
                          momentum=0.9,
                          weight_decay=5e-4)

    # get tracking dictionaries
    prev_list, mse_delta_dict, mae_delta_dict, rmae_delta_dict, layer_names = setup_delta_tracking(
        model, model_name, training_type)

    # train model
    mse_delta_dict, mae_delta_dict, rmae_delta_dict = training(epochs, loaders, model, model_name,
                                                               optimizer, criterion, prev_list,
                                                               mse_delta_dict, mae_delta_dict,
                                                               rmae_delta_dict, layer_names, training_type)

    return mse_delta_dict, mae_delta_dict, rmae_delta_dict

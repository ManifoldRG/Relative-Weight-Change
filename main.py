import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import torch
from train import training
from data import load_cifar10
from delta import setup_delta_tracking
from utils import load_model
from comet_ml import Experiment


def run_experiment(epochs, model_name, training_type, configs, exp_name):
    """ Runs the basic experiment"""

    print(epochs, "CONFIGS: ", configs)

    experiment = Experiment(api_key="ZgD8zJEiZErhwIzPMfZpitMjq",
                            project_name="relative-weight-change", workspace="ayushm-agrawal")

    experiment.set_name(exp_name)

    experiment.log_parameters(configs)

    # set seed for reproducibility.
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # gpu training specific seed settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    loaders = load_cifar10(configs)

    # load model
    model = load_model(model_name, training_type)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), configs.lr,
                          momentum=configs.momentum,
                          weight_decay=configs.weight_decay)

    # get tracking dictionaries
    prev_list, mse_delta_dict, mae_delta_dict, rmae_delta_dict, layer_names = setup_delta_tracking(
        model, model_name, training_type)

    # train model
    mse_delta_dict, mae_delta_dict, rmae_delta_dict = training(epochs, loaders, model, model_name,
                                                               optimizer, criterion, prev_list,
                                                               mse_delta_dict, mae_delta_dict,
                                                               rmae_delta_dict, layer_names, training_type, configs, experiment)

    experiment.end()

    return mse_delta_dict, mae_delta_dict, rmae_delta_dict

import random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from comet_ml import Experiment

from train import training
from utils import load_model
from data import load_dataset
from delta import setup_delta_tracking




def run_experiment(epochs, model_name, training_type, configs):
    """ Runs the basic experiment"""

    print(epochs, "CONFIGS: ", configs)

    experiment = Experiment(api_key=configs.api_key,
                            project_name=configs.project_name, workspace="ayushm-agrawal")

    experiment.set_name(configs.exp_name)

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
    loaders = load_dataset(configs)

    # load model
    model = load_model(model_name, training_type, configs)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), configs.lr,
                          momentum=configs.momentum,
                          weight_decay=configs.weight_decay)

    # get tracking dictionaries
    prev_list, rmae_delta_dict = setup_delta_tracking(model)

    # train model
    rmae_delta_dict, train_acc_arr, test_acc_arr = training(epochs, loaders, model, optimizer, criterion, prev_list, rmae_delta_dict, configs, experiment)

    experiment.end()

    return rmae_delta_dict, train_acc_arr, test_acc_arr

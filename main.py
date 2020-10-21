import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.vgg import vgg19
import random
import numpy as np
import torch
from model import ResNet18
from vgg import vgg19_bn
from train import training
from data import load_cifar10
from delta import setup_delta_tracking


def run_experiment(model_name, training_type, configs):

    print("CONFIGS: ", configs)

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

    # pick model
    if model_name == "Resnet18":
        # load weights
        if training_type == "pretrained":
            model = torchvision.models.resnet18(pretrained=True)
            model.fc.Linear = nn.Linear(model.fc.in_features, 10)
        elif training_type == "no_pretrain":
            model = ResNet18()
    elif model_name == "VGG19":
        # load weights
        if training_type == "pretrained":
            model = torchvision.models.vgg19(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, 10),
            )
        elif training_type == "no_pretrain":
            model = vgg19_bn()

    else:
        print("Please provide a model")

    # push model to cuda
    model = model.cuda()
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
                                                               rmae_delta_dict, layer_names, training_type, configs)

    return mse_delta_dict, mae_delta_dict, rmae_delta_dict

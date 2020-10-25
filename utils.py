import itertools

import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision

from model import ResNet18
from vanilla import VanillaCNN2
import vgg


def plot_deltas(delta_dict, model_name, save_name):
    """Basic plotting function.
     Don't need this anymore.
    """

    marker = itertools.cycle(('.', '$...$', "v", "1", "8", "s", "p",
                              "P", "*", "+", "x", "d", "_", 4, 5, "D", "H", "3", "<", ">", "^"))
    plt.figure(figsize=(30, 20), dpi=200)
    # style
    plt.style.use('seaborn-darkgrid')

    for layer in delta_dict:

        if model_name == "Resnet18":
            if layer != 'conv1.weight' and layer != 'linear.weight' and layer != 'layer3.0.shortcut.0.weight' and layer != 'layer2.0.shortcut.0.weight':
                plt.plot(delta_dict[layer], label=layer, marker=next(
                    marker), animated=True, linewidth=0.5)

    plt.legend(loc=1, prop={'size': 8})
    plt.savefig(save_name)


def load_model(model_name, training_type):
    """
    Loads model.
    """
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
            model = vgg.__dict__['vgg19_bn'](num_classes=10)
    elif model_name == 'Vanilla':
        print("Loading Vanillaa")
        model = VanillaCNN2()
    else:
        print("Please provide a model")

    # push model to cuda
    model = model.cuda()

    return model

import torch.nn as nn
import torchvision
import torch

from resnet import ResNet18
from xception import Xception, xception
from vanilla import VanillaCNN2
from vgg import vgg19_bn
from alexnet import AlexNet


def load_model(model_name, training_type, configs):
    """
    Loads model.
    """
    # pick model
    if model_name == "Resnet18":
        # load weights
        if training_type == "pretrained":
            print("Loading pretrained Resnet18")
            model = torchvision.models.resnet18(pretrained=True)
            model.fc.Linear = nn.Linear(model.fc.in_features, 10)

        elif training_type == "untrained":
            print("Loading untrained Resnet18")
            if configs.dataset == "MNIST" or configs.dataset == "FashionMNIST":
                model = ResNet18(num_classes=10, input_channels=1)
            elif configs.dataset == "CIFAR-100":
                model = ResNet18(num_classes=100, input_channels=3)
            else:
                model = ResNet18(num_classes=10, input_channels=3)

    elif model_name == "Xception":
        # load model with pretrained weights
        if training_type == "pretrained":

            print("Loading pretrained Xception")

            model = xception()
            model.fc.Linear = nn.Linear(model.fc.in_features, 10)

        # load model without pretrained weights
        elif training_type == "untrained":

            print("Loading untrained Xception")

            if configs.dataset == "MNIST" or configs.dataset == "FashionMNIST":
                model = Xception(num_classes=10, input_channels=1)

            elif configs.dataset == "CIFAR-100":
                model = Xception(num_classes=100, input_channels=3)

            else:
                model = Xception(num_classes=10, input_channels=3)

    elif model_name == "VGG19":
        # load weights
        if training_type == "pretrained":

            print("Loading pretrained VGG19")

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

        elif training_type == "untrained":

            print("Loading untrained VGG19")

            if configs.dataset == "MNIST" or configs.dataset == "FashionMNIST":

                model = vgg19_bn(in_channels=1, num_classes=10)

            elif configs.dataset == "CIFAR-100":
                model = vgg19_bn(in_channels=3, num_classes=100)
            else:
                model = vgg19_bn(in_channels=3, num_classes=10)

    elif model_name == "Vanilla":

        print("Loading Vanilla")

        if configs.dataset == "MNIST" or configs.dataset == "FashionMNIST":
            model = VanillaCNN2(num_classes=10, input_channels=1)

        elif configs.dataset == "CIFAR-100":
            model = VanillaCNN2(num_classes=100, input_channels=3)

        else:
            model = VanillaCNN2(num_classes=10, input_channels=3)

    elif model_name == "Alexnet":

        print("Loading Alexnet")

        if configs.dataset == "MNIST" or configs.dataset == "FashionMNIST":
            model = AlexNet(num_classes=10, input_channels=1)

        elif configs.dataset == "CIFAR-100":
            model = AlexNet(num_classes=100, input_channels=3)

        else:
            model = AlexNet(num_classes=10, input_channels=3)

    else:

        print("Please provide a model")

    # push model to cuda
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print("\nModel moved to Data Parallel")
    model.cuda()

    return model


def freeze_resnet_1(model):

    model.conv1.weight.requires_grad = False
    model.layer1[0].conv1.weight.requires_grad = False
    model.layer1[0].conv2.weight.requires_grad = False
    model.layer1[1].conv1.weight.requires_grad = False
    model.layer1[1].conv2.weight.requires_grad = False

    model.layer2[0].conv1.weight.requires_grad = False
    model.layer2[0].conv2.weight.requires_grad = False
    model.layer2[0].shortcut[0].weight.requires_grad = False
    model.layer2[1].conv1.weight.requires_grad = False
    model.layer2[1].conv2.weight.requires_grad = False

    print("Freeezing Type-1")


def freeze_resnet_2(model, epoch):

    if epoch == 20 or epoch == 3:
        model.conv1.weight.requires_grad = False
        model.layer1[0].conv1.weight.requires_grad = False
        model.layer1[0].conv2.weight.requires_grad = False
        model.layer1[1].conv1.weight.requires_grad = False
        model.layer1[1].conv2.weight.requires_grad = False

        print("Freeezing Type-2 at 20 Epochs")

    if epoch == 40 or epoch == 5:
        model.layer2[0].conv1.weight.requires_grad = False
        model.layer2[0].conv2.weight.requires_grad = False
        model.layer2[0].shortcut[0].weight.requires_grad = False
        model.layer2[1].conv1.weight.requires_grad = False
        model.layer2[1].conv2.weight.requires_grad = False

        print("Freeezing Type-2 at 40 Epochs")

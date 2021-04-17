import torch
import torch.nn as nn
import torchvision
from models.alexnet import AlexNet
from models.resnet import ResNet18
from models.vgg import vgg19_bn


def load_model(model_name, training_type, configs):
    """
    Loads model.
    """

    dataset = configs.dataset.lower()
    # set the input channels and num_classes
    if dataset == "mnist" or dataset == "fashionmnist":
        configs.num_classes = 10
        configs.input_channels = 1
    elif dataset == "cifar-100":
        configs.num_classes = 100
        configs.input_channels = 3
    elif dataset == "imagenet":
        configs.num_classes = 1000
        configs.input_channels = 3
    else:
        configs.num_classes = 10
        configs.input_channels = 3

    # pick model
    if model_name == "Resnet18":
        # load weights
        if training_type == "pretrained":
            print("Loading pretrained Resnet18")
            model = torchvision.models.resnet18(pretrained=True)
            model.fc.Linear = nn.Linear(model.fc.in_features, 10)

        elif training_type == "untrained":
            print("Loading untrained Resnet18")
            model = ResNet18(num_classes=configs.num_classes,
                             input_channels=configs.input_channels)
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

            model = vgg19_bn(in_channels=configs.input_channels,
                             num_classes=configs.num_classes)
    elif model_name = "resnet101":
        if training_type == 'pretrained':
            print("Loading pretrained ResNet-101")

            model = torchvision.models.resnet101(pretrained=True)
        elif training_type ==  "untrained":
            print("Loading untrained ResNet101")

            model = torchvision.models.resnet101()
    else:
        print("Please provide a model")

    # push model to cuda
    if torch.cuda.device_count() > 1:
        print(f"Number of GPUs available are {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
        print("\nModel moved to Data Parallel")
    model.cuda()

    return model

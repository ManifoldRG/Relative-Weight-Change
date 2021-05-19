import torch
import torch.nn as nn
import torchvision
from models.resnet import ResNet18, ResNet50
from models.vgg import vgg19_bn
from models.efficientnet import load_efficientnet


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            accuracy = correct_k.mul_(100.0 / batch_size)
            res.append(accuracy)
        return res

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
            model.fc.Linear = nn.Linear(model.fc.in_features, configs.num_classes)

        elif training_type == "untrained":
            print("Loading untrained Resnet18")
            model = ResNet18(num_classes=configs.num_classes,
                             input_channels=configs.input_channels)
    elif  model_name == "Resnet50":
        # load weights
        if training_type == "pretrained":
            print(f"Loading pretrained {model_name}")
            model = torchvision.models.resnet50(pretrained=True)
            model.fc.Linear = nn.Linear(model.fc.in_features, configs.num_classes)

        elif training_type == "untrained":
            print(f"Loading untrained {model_name}")
            model = ResNet50(num_classes=configs.num_classes,
                             input_channels=configs.input_channels)

    
    elif model_name == "Resnet101":
        if training_type == 'pretrained':
            print(f"Loading pretrained {model_name}")

            model = torchvision.models.resnet101(pretrained=True)
        elif training_type ==  "untrained":
            print(f"Loading untrained {model_name}")

            model = torchvision.models.resnet101()

    elif model_name == "VGG19":
        # load weights
        if training_type == "pretrained":

            print(f"Loading pretrained {model_name}")

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

            print(f"Loading untrained {model_name}")

            model = vgg19_bn(in_channels=configs.input_channels,
                             num_classes=configs.num_classes)

    elif model_name.startswith("EfficientNet"):

        if training_type == 'pretrained':
            print(f"Loading pretrained {model_name}")

            model = load_efficientnet(model_name, configs.num_classes, configs.input_channels, True)

        elif training_type ==  "untrained":
            print(f"Loading untrained {model_name}")

            model = load_efficientnet(model_name, configs.num_classes, configs.input_channels, False)
    else:
        print("Please provide a model")

    # push model to cuda
    if torch.cuda.device_count() > 1:
        print(f"Number of GPUs available are {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
        print("\nModel moved to Data Parallel")
    model.cuda()

    return model

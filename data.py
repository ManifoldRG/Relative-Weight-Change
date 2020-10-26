import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torch.utils.data import DataLoader


def load_dataset(configs):
    if configs.dataset == "MNIST":
        return load_mnist(configs)
    elif configs.dataset == "FashionMNIST":
        return load_fashionmnist(configs)
    elif configs.dataset == "CIFAR-10":
        return load_cifar10(configs)
    elif configs.dataset == "CIFAR-100":
        return load_cifar100(configs)
    

def load_cifar10(configs):
    # transform for the training data
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # load datasets, downloading if needed
    train_set = CIFAR10('./data/cifar10', train=True, download=True,
                        transform=train_transforms)
    test_set = CIFAR10('./data/cifar10', train=False, download=True,
                       transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=configs.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=configs.batch_size, num_workers=0)

    print('Number of iterations required to get through training data of length {}: {}'.format(
        len(train_set), len(train_loader)))

    print(train_set.data.shape)
    print(test_set.data.shape)

    return {'train': train_loader, 'test': test_loader}

def load_cifar100(configs):
    # transform for the training data
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # load datasets, downloading if needed
    train_set = CIFAR100('./data/cifar100', train=True, download=True,
                        transform=train_transforms)
    test_set = CIFAR100('./data/cifar100', train=False, download=True,
                       transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=configs.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=configs.batch_size, num_workers=0)

    print('Number of iterations required to get through training data of length {}: {}'.format(
        len(train_set), len(train_loader)))

    print(train_set.data.shape)
    print(test_set.data.shape)

    return {'train': train_loader, 'test': test_loader}

def load_fashionmnist(configs):
    # transform for the training data
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load datasets, downloading if needed
    train_set = FashionMNIST('./data/fashionmnist', train=True, download=True,
                        transform=train_transforms)
    test_set = FashionMNIST('./data/fashionmnist', train=False, download=True,
                       transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=configs.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=configs.batch_size, num_workers=0)

    print('Number of iterations required to get through training data of length {}: {}'.format(
        len(train_set), len(train_loader)))

    print(train_set.data.shape)
    print(test_set.data.shape)

    return {'train': train_loader, 'test': test_loader}
    return {'train': train_loader, 'test': test_loader}

def load_mnist(configs):
    # transform for the training data
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # load datasets, downloading if needed
    train_set = MNIST('./data/mnist', train=True, download=True,
                        transform=train_transforms)
    test_set = MNIST('./data/mnist', train=False, download=True,
                       transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=configs.batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=configs.batch_size, num_workers=0)

    print('Number of iterations required to get through training data of length {}: {}'.format(
        len(train_set), len(train_loader)))

    print(train_set.data.shape)
    print(test_set.data.shape)

    return {'train': train_loader, 'test': test_loader}
    return {'train': train_loader, 'test': test_loader}
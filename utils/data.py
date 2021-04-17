import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST


def load_dataset(configs):
    dataset = configs.dataset.lower()
    if dataset == "mnist":
        return load_mnist(configs)
    elif dataset == "fashionmnist":
        return load_fashionmnist(configs)
    elif dataset == "cifar-10":
        return load_cifar10(configs)
    elif dataset == "cifar-100":
        return load_cifar100(configs)
    elif dataset == "imagenet":
        return load_imagenet(configs)
    

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

    print(f'Number of iterations required to get through training data of length {len(train_set)}: {len(train_loader)}')
 
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
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
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

def load_mnist(configs):
    # transform for the training data
    train_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((32, 32)),
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

def load_imagenet(configs):
    # transform for the training data
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_data_path = configs.data_path + "/train/"
    val_data_path = configs.data_path + "/val/"
    train_set = datasets.ImageFolder(train_data_path, transforms=train_transforms)
    val_set = datasets.ImageFolder(val_data_path, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=configs.batch_size, 
                                        num_workers=configs.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=configs.batch_size, 
                                        num_workers=configs.num_workers, shuffle=True)

    print('Number of iterations required to get through training data of length {}: {}'.format(
        len(train_set), len(train_loader)))

    print(train_set.data.shape)
    print(val_set.data.shape)

    return {'train': train_loader, 'test': val_loader}

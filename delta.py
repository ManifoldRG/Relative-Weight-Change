import numpy as np


def mse(prev, curr):

    """
    Computes L2 norm.
    """

    # subtract
    sub = np.subtract(prev, curr)
    # square
    squared = np.square(sub)

    # mean squared error
    return squared.mean()


def mae(prev, curr):
    """
    Computes L1 norm.
    """
    # subtract
    sub = np.subtract(prev, curr)
    # absolute
    absolute = np.abs(sub)

    # mean absolute error
    return absolute.mean()


def rmae(prev, curr):

    """
    Computes Reltaive L1 norm.
    """

    # subtract
    sub = np.subtract(prev, curr)
    # absolute
    absolute = np.abs(sub)
    # normalization term
    normalizer = np.abs(prev)
    # relative error
    relative = absolute/normalizer

    # mean absolute error
    return relative.mean()


def setup_delta_tracking(model, model_name, training_type):
    """
    This sets up tracking for deltas.
    """

    # load model layers.
    if model_name == "Resnet18":
        if training_type == "pretrained":
            prev_list = load_layers_resnet(model, pretrained=True)
        elif training_type == "no_pretrain":
            prev_list = load_layers_resnet(model, pretrained=False)
    elif model_name == "VGG19":
        prev_list = load_layers_vgg(model)

    # setup tracking dictionaries
    mse_delta_dict, mae_delta_dict, rmae_delta_dict = {}, {}, {}
    layer_names = []

    for name, param in model.named_parameters():
        if len(param.size()) > 1:
            layer_names.append(name)

    for i, layer in zip(range(len(prev_list)), layer_names):
        mse_delta_dict[layer] = []
        mae_delta_dict[layer] = []
        rmae_delta_dict[layer] = []

    return prev_list, mse_delta_dict, mae_delta_dict, rmae_delta_dict, layer_names


def compute_delta(model, model_name, layer_names, prev_list, mse_delta_dict, mae_delta_dict, rmae_delta_dict, training_type):

    """
    This computes the deltas for the experiment.
    """

    # load model layers.
    if model_name == "Resnet18":
        if training_type == "pretrained":
            curr_list = load_layers_resnet(model, pretrained=True)
        elif training_type == "no_pretrain":
            curr_list = load_layers_resnet(model, pretrained=False)
    elif model_name == "VGG19":
        curr_list = load_layers_vgg(model)

    # update dictionaries with deltas.
    for i, layer in zip(range(len(prev_list)), layer_names):

        # compute L2.
        layer_mse_delta = mse(prev_list[i], curr_list[i])
        # compute L1.
        layer_mae_delta = mae(prev_list[i], curr_list[i])
        # compute RL1.
        layer_rmae_delta = rmae(prev_list[i], curr_list[i])

        # update dictionaries.
        mse_delta_dict[layer].append(layer_mse_delta)
        mae_delta_dict[layer].append(layer_mae_delta)
        rmae_delta_dict[layer].append(layer_rmae_delta)

    # update previous.
    prev_list = curr_list.copy()

    return mse_delta_dict, mae_delta_dict, rmae_delta_dict, prev_list


def load_layers_resnet(model, pretrained=False):

    """
    This loads the layer weights for resnet.
    """
    layer_list = []

    layer_list.append(model.conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer1[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer1[0].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer1[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer1[1].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer2[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer2[0].conv2.weight.data.cpu().numpy())

    if pretrained:
        layer_list.append(
            model.layer2[0].downsample[0].weight.data.cpu().numpy())
    else:
        layer_list.append(
            model.layer2[0].shortcut[0].weight.data.cpu().numpy())

    layer_list.append(model.layer2[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer2[1].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer3[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer3[0].conv2.weight.data.cpu().numpy())

    if pretrained:
        layer_list.append(
            model.layer3[0].downsample[0].weight.data.cpu().numpy())
    else:
        layer_list.append(
            model.layer3[0].shortcut[0].weight.data.cpu().numpy())

    layer_list.append(model.layer3[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer3[1].conv2.weight.data.cpu().numpy())
    layer_list.append(model.layer4[0].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer4[0].conv2.weight.data.cpu().numpy())

    if pretrained:
        layer_list.append(
            model.layer4[0].downsample[0].weight.data.cpu().numpy())
    else:
        layer_list.append(
            model.layer4[0].shortcut[0].weight.data.cpu().numpy())

    layer_list.append(model.layer4[1].conv1.weight.data.cpu().numpy())
    layer_list.append(model.layer4[1].conv2.weight.data.cpu().numpy())
    if pretrained:
        layer_list.append(model.fc.weight.data.cpu().numpy())
    else:
        layer_list.append(model.linear.weight.data.cpu().numpy())

    return layer_list


def load_layers_vgg(model):

    """
    This loads the layer weights for vgg.
    """


    layer_list = []

    layer_list.append(model.features[0].weight.data.cpu().numpy())
    layer_list.append(model.features[3].weight.data.cpu().numpy())
    layer_list.append(model.features[7].weight.data.cpu().numpy())
    layer_list.append(model.features[10].weight.data.cpu().numpy())
    layer_list.append(model.features[14].weight.data.cpu().numpy())
    layer_list.append(model.features[17].weight.data.cpu().numpy())
    layer_list.append(model.features[20].weight.data.cpu().numpy())
    layer_list.append(model.features[23].weight.data.cpu().numpy())
    layer_list.append(model.features[27].weight.data.cpu().numpy())
    layer_list.append(model.features[30].weight.data.cpu().numpy())
    layer_list.append(model.features[33].weight.data.cpu().numpy())
    layer_list.append(model.features[36].weight.data.cpu().numpy())
    layer_list.append(model.features[40].weight.data.cpu().numpy())
    layer_list.append(model.features[43].weight.data.cpu().numpy())
    layer_list.append(model.features[46].weight.data.cpu().numpy())
    layer_list.append(model.features[49].weight.data.cpu().numpy())
    layer_list.append(model.classifier[0].weight.data.cpu().numpy())
    layer_list.append(model.classifier[3].weight.data.cpu().numpy())
    layer_list.append(model.classifier[6].weight.data.cpu().numpy())

    return layer_list
